"""
Cut each ground-truth object out of the floor point cloud, starting from its 3D position
(instances_3d.csv from pano_clicks_to_3d.py). Result: a per-point label for every object.

For each object:
  1. take the points within a class-specific radius of its position (about the object's size);
  2. remove building surfaces (walls, ceilings, floors, soffits) with RANSAC. Only the parts
     of a plane that reach the edge of the crop are removed, so the flat face of a sign or
     cabinet is kept, and corners / wall strips / ceiling steps are removed too;
  3. cluster what is left with DBSCAN and keep the cluster touching the object's position.
     If it is too big for its class (connected to a door frame, bracket, pipe), retry with a
     smaller DBSCAN distance; if still too big, clip to the class size (status CHECK);
  4. fallback for flush-mounted objects that vanished with the wall (flat signs, indicators,
     call points): grow a region from the position by colour, within 2.5 cm of the wall;
  5. sanity checks: too big for its class / very few points / shared with a neighbour.
Objects whose clusters touch are split between them by nearest position.

Inputs
  --cloud       the floor point cloud the ground truth is defined on (e.g. floor2_5mm.ply)
  --instances   instances_3d.csv from pano_clicks_to_3d.py (other columns are carried over)
  --floor       optional: only objects of this floor (if the CSV holds several floors)

Outputs (in --out-dir)
  gt_labelled.ply         the whole cloud with two extra fields: class_id (-1 = background)
                          and instance_id (0 = background). Open in CloudCompare and colour
                          by either scalar field.
  gt_objects_only.ply     only the object points (fast to load for a check)
  instances_gt.csv        instances_3d.csv + mask results: method, n_points, mask centre,
                          box size, status. Fill attribute / ifc_space / bim_guid here.
  review/<id>_<class>.png front and top view of each object: grey = surroundings, red = mask
  fix/<id>_<class>_crop.ply  surroundings of every object that is not "ok", for hand fixing

Status meanings
  ok            cut out by plane removal + clustering
  colour        found by the colour fallback - check the image
  CHECK: ...    something unusual (too big, clipped, very few points, split with a neighbour)
  FAILED        nothing found
  manual        taken from your hand-cut file in --fixes-dir

Fixing objects by hand (everything that is not "ok")
  For each such object the script writes fix/<id>_<class>_crop.ply: only the points around
  it, with a field current_mask (1 = what the script picked). In CloudCompare:
    1. open the crop (small, loads instantly); colour by current_mask to see the current cut
    2. select it, Segment tool (scissors), draw a polygon around the object, "Segment In",
       confirm; repeat from another view if needed until only the object is left
    3. save that part as fix/<id>.ply  (e.g. 7.ply) - binary PLY
  Then re-run with --fixes-dir <out-dir>\\fix ; those objects get status "manual".
  If an automatic result is fine after all, simply do not make a file for it.

Usage
  python cutout_instances.py --cloud D:\\hotel\\floor2_5mm.ply --instances D:\\hotel\\gt\\instances_3d.csv
         --out-dir D:\\hotel\\gt\\cutout --floor 2

Requires: pip install numpy open3d matplotlib
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

CLASSES = ["Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
           "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED", "Fire Call Point",
           "Fire Blanket", "Emergency Door Release", "Evacuation Chair"]
# crop radius around the clicked position = roughly the largest size of the object (m).
# The click lands on the object's surface, so the object can extend this far from it.
CROP_RADIUS = {0: 0.70, 1: 0.25, 2: 1.00, 3: 0.50, 4: 0.20, 5: 0.25, 6: 0.60, 7: 0.70,
               8: 0.20, 9: 0.45, 10: 0.20, 11: 1.40}
# largest plausible object size (m); bigger masks are flagged
MAX_SIZE = {0: 0.90, 1: 0.30, 2: 1.20, 3: 0.70, 4: 0.25, 5: 0.25, 6: 0.70, 7: 0.80,
            8: 0.25, 9: 0.50, 10: 0.25, 11: 1.60}


def safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


# ---------------------------------------------------------------- geometry helpers
def plane_basis(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def remove_big_planes(pts, seed, crop_r, dist, max_planes=4, edge=0.85, band_eps=0.03):
    """Remove building surfaces (walls, ceilings, floors, soffits) with RANSAC.

    For each RANSAC plane, the points within `dist` of it are split into connected pieces.
    Only pieces that reach the edge of the crop sphere are removed: they are bigger than
    the object, so they are building surfaces. Pieces that end inside the crop (the object's
    own face, or an object face that happens to lie in the same plane as a wall strip) are
    kept. A plane with no such piece is excluded from the next RANSAC search, so smaller
    wall strips behind it can still be found.
    Returns (keep_mask, list of (unit normal, d) of removed planes)."""
    import open3d as o3d
    keep = np.ones(len(pts), bool)
    search = np.ones(len(pts), bool)
    r_seed = np.linalg.norm(pts - seed, axis=1)
    planes = []
    for _ in range(max_planes + 4):
        if len(planes) >= max_planes:
            break
        idx = np.flatnonzero(keep & search)
        if len(idx) < 200:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[idx]))
        model, inl = pc.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=400)
        inl = idx[np.asarray(inl)]
        if len(inl) < 100:
            break
        nrm = np.linalg.norm(model[:3])
        n, d = np.asarray(model[:3]) / nrm, model[3] / nrm
        band = np.flatnonzero(keep & (np.abs(pts @ n + d) < dist))
        lab = dbscan(pts[band], band_eps, 5)
        removed = 0
        for c in set(lab[lab >= 0]):
            piece = band[lab == c]
            if np.sum(r_seed[piece] > edge * crop_r) >= 20:
                keep[piece] = False
                removed += len(piece)
        if removed == 0:
            search[inl] = False          # object face: keep it, look for other planes
            continue
        search[inl] = False
        planes.append((n, d))
    return keep, planes


def dbscan(pts, eps, min_pts):
    import open3d as o3d
    if len(pts) == 0:
        return np.zeros(0, int)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))


def colour_region(pts, cols, seed, planes, eps, max_r, near_plane=0.025, dcol=45.0):
    """Flush objects: points close to a removed plane whose colour matches the colour at
    the seed, connected to the seed."""
    if cols is None or not planes:
        return None
    d_seed = np.linalg.norm(pts - seed, axis=1)
    near = d_seed < max_r
    on_plane = np.zeros(len(pts), bool)
    for n, d in planes:
        on_plane |= np.abs(pts @ n + d) < near_plane
    cand = np.flatnonzero(near & on_plane)
    if len(cand) < 10:
        return None
    core = cand[d_seed[cand] < 0.015]
    if len(core) < 3:
        return None
    c0 = np.median(cols[core], axis=0)
    # the seed colour must differ from the wall colour, otherwise colour cannot separate them
    wall = np.median(cols[cand], axis=0)
    if np.linalg.norm(c0 - wall) < dcol:
        return None
    sim = cand[np.linalg.norm(cols[cand] - c0, axis=1) < dcol]
    if len(sim) < 10:
        return None
    lab = dbscan(pts[sim], eps, 5)
    if (lab < 0).all():
        return None
    best = None
    for k in set(lab[lab >= 0]):
        dmin = np.min(np.linalg.norm(pts[sim[lab == k]] - seed, axis=1))
        if dmin < 0.03 and (best is None or dmin < best[0]):
            best = (dmin, k)
    return None if best is None else sim[lab == best[1]]


def seed_cluster(pts, rest, seed, eps, min_pts, tol):
    """DBSCAN on the remaining points; return the cluster that comes closest to the seed."""
    lab = dbscan(pts[rest], eps, min_pts)
    best = None
    for c in set(lab[lab >= 0]):
        m = rest[lab == c]
        dmin = np.min(np.linalg.norm(pts[m] - seed, axis=1))
        if dmin < tol and (best is None or dmin < best[0]):
            best = (dmin, m)
    return None if best is None else best[1]


def add_rim(pts, obj, planes, dist, lift=0.004, grow=0.012):
    """Plane removal also strips the object's lowest few mm next to the wall/ceiling.
    Add back points in that band that lie on the object side of the plane (> lift from it)
    and directly under the object's outline (within `grow`)."""
    import open3d as o3d
    obj = np.asarray(obj)
    extra = []
    for n, d in planes:
        sd_obj = pts[obj] @ n + d
        side = np.sign(np.median(sd_obj))
        if side == 0:
            continue
        sd = (pts @ n + d) * side
        band = np.flatnonzero((sd > lift) & (sd < dist + 0.002))
        if len(band) == 0:
            continue
        u, v = plane_basis(n)
        o2 = np.c_[pts[obj] @ u, pts[obj] @ v, np.zeros(len(obj))]
        b2 = np.c_[pts[band] @ u, pts[band] @ v, np.zeros(len(band))]
        t2 = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(o2)))
        t3 = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[obj])))
        for j, q in zip(band, b2):
            # under the object's outline AND touching the object in 3D
            if t2.search_radius_vector_3d(q, grow)[0] > 0 and \
                    t3.search_radius_vector_3d(pts[j], grow + dist)[0] > 0:
                extra.append(j)
    return np.union1d(obj, np.asarray(extra, dtype=int)) if extra else obj


def cut_one(tree, P, C, seed, cid, a):
    """Returns (global indices of the object's points, method, notes)."""
    r = CROP_RADIUS.get(cid, 0.5) * a.radius_scale
    k, idx, _ = tree.search_radius_vector_3d(seed, r)
    idx = np.asarray(idx)
    if k < 20:
        return np.zeros(0, int), "FAILED", f"only {k} points within {r:.2f} m"
    pts = P[idx]
    cols = C[idx] if C is not None else None
    keep, planes = remove_big_planes(pts, seed, r, a.plane_dist)
    rest = np.flatnonzero(keep)
    if len(rest) >= a.min_points:
        max_size = MAX_SIZE.get(cid, 1.0)
        best, note = None, f"{len(planes)} plane(s) removed"
        # try the normal eps first; if the object comes out too big (connected to a door
        # frame, bracket, pipe...), retry with smaller eps to break thin connections
        for f in (1.0, 0.6, 0.4):
            cand = seed_cluster(pts, rest, seed, a.eps * f, a.min_points, a.seed_tol)
            if cand is None:
                break
            best = cand
            if np.ptp(pts[best], axis=0).max() <= max_size:
                if f < 1.0:
                    note += f", connection cut (eps x{f})"
                break
        if best is not None:
            method = "ok"
            if np.ptp(pts[best], axis=0).max() > max_size:
                # last resort: keep only the part within the object's size around the position
                best = best[np.linalg.norm(pts[best] - seed, axis=1) < 0.55 * max_size]
                method, note = "clipped", note + ", clipped to class size"
            if len(best):
                obj = add_rim(pts, best, planes, a.plane_dist)
                return idx[obj], method, note
    reg = colour_region(pts, cols, seed, planes, a.eps, r)
    if reg is not None:
        return idx[reg], "colour", "flush object, found by colour"
    return np.zeros(0, int), "FAILED", f"{len(planes)} plane(s) removed, no cluster at the position"


def load_fixes(fix_dir, tree, P, tol=0.002):
    """Hand-cut objects: <instance_id>.ply (anything after the number is ignored, files
    ending in _crop.ply are skipped). Each point is matched to the nearest cloud point."""
    import glob
    import open3d as o3d
    out = {}
    for f in sorted(glob.glob(os.path.join(fix_dir, "*.ply"))):
        name = os.path.basename(f)
        m = re.match(r"^(\d+)", name)
        if not m or name.endswith("_crop.ply"):
            continue
        Q = np.asarray(o3d.io.read_point_cloud(f).points)
        idx = []
        for q in Q:
            k, ii, d2 = tree.search_knn_vector_3d(q, 1)
            if k and d2[0] <= tol * tol:
                idx.append(ii[0])
        idx = np.unique(np.asarray(idx, dtype=int))
        out[int(m.group(1))] = (idx, f"hand-cut ({name}, {len(idx)}/{len(Q)} points matched)")
        print(f"  fix {name}: {len(idx)} of {len(Q)} points matched to the cloud")
    return out


# ---------------------------------------------------------------- review image
def review_png(path, tree, P, seed, mask_idx, r, title, max_bg=40000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg = np.asarray(tree.search_radius_vector_3d(seed, r * 1.3)[1], dtype=int)
    if len(bg) > max_bg:
        bg = np.random.default_rng(0).choice(bg, max_bg, replace=False)
    fg = P[mask_idx] if len(mask_idx) else np.zeros((0, 3))
    # "front" = looking along the direction of least spread of the surroundings (at the wall)
    B = P[bg] - seed
    if len(B) > 10:
        w, V = np.linalg.eigh(np.cov(B.T))
        normal = V[:, 0]
    else:
        normal = np.array([0, 0, 1.0])
    if abs(normal[2]) > 0.7:          # ceiling/floor: look from below/above, keep x/y
        e1, e2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    else:                              # wall: horizontal axis along the wall, vertical = z
        e1 = np.cross([0, 0, 1.0], normal)
        e1 /= np.linalg.norm(e1)
        e2 = np.array([0, 0, 1.0])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    for k, (a1, a2, name) in enumerate([(e1, e2, "front"), (e1, normal, "top / section")]):
        ax[k].scatter(B @ a1, B @ a2, s=0.3, c="0.7", linewidths=0)
        if len(fg):
            F = fg - seed
            ax[k].scatter(F @ a1, F @ a2, s=0.8, c="red", linewidths=0)
        ax[k].plot(0, 0, "b+", ms=12, mew=2)
        ax[k].set_aspect("equal")
        ax[k].set_title(name, fontsize=9)
        ax[k].tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- output
def write_ply(path, P, C, extra):
    n = len(P)
    dt = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    head = ["property double x", "property double y", "property double z"]
    if C is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    for name, arr in extra.items():
        dt.append((name, "<i4"))
        head.append(f"property int {name}")
    a = np.empty(n, dtype=dt)
    a["x"], a["y"], a["z"] = P[:, 0], P[:, 1], P[:, 2]
    if C is not None:
        c8 = np.clip(C, 0, 255).astype(np.uint8)
        a["red"], a["green"], a["blue"] = c8[:, 0], c8[:, 1], c8[:, 2]
    for name, arr in extra.items():
        a[name] = arr
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n" + "\n".join(head) + "\nend_header\n").encode("ascii"))
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--floor", help="only objects of this floor")
    ap.add_argument("--plane-dist", type=float, default=0.012,
                    help="points within this distance (m) of a wall/ceiling plane are removed (default 0.012)")
    ap.add_argument("--eps", type=float, default=0.025,
                    help="DBSCAN neighbour distance (m), ~5x the point spacing (default 0.025)")
    ap.add_argument("--min-points", type=int, default=10, help="DBSCAN min points (default 10)")
    ap.add_argument("--seed-tol", type=float, default=0.10,
                    help="the object cluster must come within this distance (m) of the position")
    ap.add_argument("--radius-scale", type=float, default=1.0,
                    help="multiply all crop radii (e.g. 1.3 if objects are cut too small)")
    ap.add_argument("--no-images", action="store_true", help="skip the review images")
    ap.add_argument("--fixes-dir",
                    help="folder with hand-cut objects from CloudCompare: <instance_id>.ply "
                         "(e.g. 7.ply). These replace the automatic result for that object.")
    a = ap.parse_args()

    import open3d as o3d
    os.makedirs(os.path.join(a.out_dir, "review"), exist_ok=True)

    rows = list(csv.DictReader(open(a.instances, newline="", encoding="utf-8")))
    if a.floor:
        rows = [r for r in rows if str(r.get("floor", "")) == str(a.floor)]
    rows = [r for r in rows if int(r["class_id"]) >= 0]
    print(f"{len(rows)} objects to cut out")

    print(f"loading {a.cloud} ...")
    pcd = o3d.io.read_point_cloud(a.cloud)
    P = np.asarray(pcd.points, dtype=np.float64)
    C = (np.asarray(pcd.colors) * 255.0) if pcd.has_colors() else None
    print(f"  {len(P):,} points, colours: {'yes' if C is not None else 'no'}")
    tree = o3d.geometry.KDTreeFlann(pcd)

    fixes = load_fixes(a.fixes_dir, tree, P) if a.fixes_dir else {}

    owner = {}                         # point index -> list of row numbers that claim it
    results = []
    for i, r in enumerate(rows):
        cid = int(r["class_id"])
        seed = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        if int(r["instance_id"]) in fixes:
            m, note = fixes[int(r["instance_id"])]
            method = "manual"
        else:
            m, method, note = cut_one(tree, P, C, seed, cid, a)
        results.append({"row": r, "cid": cid, "seed": seed, "idx": m, "method": method, "note": note})
        for p in m:
            owner.setdefault(int(p), []).append(i)
        print(f"  #{r['instance_id']:>4} {CLASSES[cid]:<22} {method:<6} {len(m):>6} pts  {note}")

    # points claimed by several objects -> nearest position wins
    shared = {p: o for p, o in owner.items() if len(o) > 1}
    split_with = {i: set() for i in range(len(rows))}
    if shared:
        for p, o in shared.items():
            d = [np.linalg.norm(P[p] - results[i]["seed"]) for i in o]
            win = o[int(np.argmin(d))]
            for i in o:
                if i != win:
                    split_with[i].add(rows[win]["instance_id"])
                    split_with[win].add(rows[i]["instance_id"])
            owner[p] = [win]
        for i, res in enumerate(results):
            res["idx"] = np.array([p for p in res["idx"] if owner[int(p)] == [i]], dtype=int)

    cls_f = np.full(len(P), -1, np.int32)
    ins_f = np.zeros(len(P), np.int32)
    out_rows = []
    for i, res in enumerate(results):
        r, cid, m = res["row"], res["cid"], res["idx"]
        status = res["method"] if res["method"] != "FAILED" else "FAILED"
        checks = []
        if len(m):
            cls_f[m] = cid
            ins_f[m] = int(r["instance_id"])
            lo, hi = np.percentile(P[m], 0.5, axis=0), np.percentile(P[m], 99.5, axis=0)
            ext = hi - lo
            if ext.max() > MAX_SIZE.get(cid, 1.0):
                checks.append(f"larger than expected ({ext.max():.2f} m)")
            if len(m) < 30:
                checks.append("very few points")
            if res["method"] == "clipped":
                checks.append("connected to something larger, clipped to class size")
            if split_with[i]:
                checks.append("touches #" + ",#".join(sorted(split_with[i])) + " - split by nearest position")
            centre = (lo + hi) / 2
        else:
            ext, centre = np.full(3, np.nan), np.full(3, np.nan)
        if checks and status not in ("FAILED", "manual"):
            status = "CHECK: " + "; ".join(checks)
        out_rows.append({**r, "method": res["method"], "n_points": len(m),
                         "box_cx": f"{centre[0]:.3f}", "box_cy": f"{centre[1]:.3f}", "box_cz": f"{centre[2]:.3f}",
                         "size_x": f"{ext[0]:.3f}", "size_y": f"{ext[1]:.3f}", "size_z": f"{ext[2]:.3f}",
                         "status": status})
        if not a.no_images:
            title = f"#{r['instance_id']} {CLASSES[cid]} - {status} ({len(m)} pts)"
            review_png(os.path.join(a.out_dir, "review", f"{int(r['instance_id']):04d}_{safe(CLASSES[cid])}.png"),
                       tree, P, res["seed"], m, CROP_RADIUS.get(cid, 0.5) * a.radius_scale, title)

    # ---- crops of the objects that need a look, for fixing by hand in CloudCompare
    fix_dir = os.path.join(a.out_dir, "fix")
    n_crops = 0
    for res, o in zip(results, out_rows):
        if o["status"] in ("ok", "manual"):
            continue
        os.makedirs(fix_dir, exist_ok=True)
        rr = CROP_RADIUS.get(res["cid"], 0.5) * a.radius_scale * 1.2
        ci = np.asarray(tree.search_radius_vector_3d(res["seed"], rr)[1], dtype=int)
        cur = np.isin(ci, res["idx"]).astype(np.int32)
        write_ply(os.path.join(fix_dir, f"{int(o['instance_id'])}_{safe(CLASSES[res['cid']])}_crop.ply"),
                  P[ci], C[ci] if C is not None else None, {"current_mask": cur})
        n_crops += 1

    # ---- write outputs
    cols = list(rows[0].keys()) + ["method", "n_points", "box_cx", "box_cy", "box_cz",
                                   "size_x", "size_y", "size_z", "status"] if rows else []
    with open(os.path.join(a.out_dir, "instances_gt.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    write_ply(os.path.join(a.out_dir, "gt_labelled.ply"), P, C,
              {"class_id": cls_f, "instance_id": ins_f})
    obj = np.flatnonzero(ins_f > 0)
    write_ply(os.path.join(a.out_dir, "gt_objects_only.ply"), P[obj], C[obj] if C is not None else None,
              {"class_id": cls_f[obj], "instance_id": ins_f[obj]})

    n_ok = sum(o["status"] in ("ok", "manual") for o in out_rows)
    n_col = sum(o["status"] == "colour" for o in out_rows)
    n_chk = sum(o["status"].startswith("CHECK") for o in out_rows)
    n_fail = sum(o["status"] == "FAILED" for o in out_rows)
    print(f"\n{len(out_rows)} objects: {n_ok} ok, {n_col} by colour (check), {n_chk} CHECK, {n_fail} FAILED")
    print(f"{len(obj):,} object points labelled. Outputs in {a.out_dir}")
    if n_crops:
        print(f"{n_crops} crops for manual fixing written to {fix_dir} (see the top of this script)")


if __name__ == "__main__":
    sys.exit(main())
