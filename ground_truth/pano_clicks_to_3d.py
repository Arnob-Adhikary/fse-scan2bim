"""
Turn point annotations made on scan-station panoramas (X-AnyLabeling) into 3D
ground-truth object positions in the point cloud.

For each click:
  1. the click pixel is turned into a viewing ray from that station
     (same equirectangular convention as project_mask_to_3d.py, incl. the mirrored azimuth);
  2. the station's own (subsampled) point cloud is searched for points within a
     small angle of that ray; the NEAREST surface along the ray is taken
     (so a click on a smoke detector hits the detector, not the ceiling behind it);
  3. the 3D point is the median of that nearest-surface patch.
Clicks of the same class from different panoramas are merged into one object when they
lie within a class-specific radius (about the object's size, because each station sees a
different side of it). Two clicks in the SAME panorama are never merged.

Inputs
  --json-dir    X-AnyLabeling .json files, one per panorama. File names must start with
                S001_pano... (after renaming with station_map.py) or pano_000... (original).
  --pose-dir    pano pose files from extract_e57.py (S001_pano_pose.json or pano_000_pose.json)
  --cloud       ONE point cloud (e.g. the merged, subsampled floor PLY), or
  --cloud-dir   one PLY per station: S001.ply / 000_<name>.ply (subsample_e57_stations.py)
                or S001_scan.ply / scan_000.ply (extract_e57.py)
                With --cloud, the nearest surface is found by stepping along the click ray
                through a KD-tree of the whole cloud; use one floor at a time to save RAM.
  --stations    stations.csv from station_map.py (links station / scan index / image index / floor)

Labels: the 12 class names (any case, spaces or underscores), the class number "0".."11",
or both as "5 = Smoke Detector".
Subclass (optional): add it after a "|" in the label, e.g. "0 = Fire Extinguisher | CO2" or
"3 = Fire Exit Sign | left", or type it in the shape's description field. Leave it out or
write "unknown" when you cannot see it. Per object, the subclass seen in most views is kept;
disagreeing views are flagged.
Use the POINT tool; rectangles/polygons are accepted too (their centre is used).

Outputs (in --out-dir)
  clicks_3d.csv                   every click with its 3D point, range and status
  instances_3d.csv                merged objects: id, class, x/y/z, floor, how many views, flags
  instances_3d.ply                one coloured point per object - load on top of the floor cloud
                                  in CloudCompare (raise point size) to check positions
  floor<N>_c<NN>_<class>.txt      x y z per object, per floor and class (input for the cut-out step)

Usage
  python pano_clicks_to_3d.py --json-dir D:\\hotel\\pano_labels --pose-dir D:\\hotel\\extract
         --cloud D:\\hotel\\floor1_5mm.ply --stations D:\\hotel\\stations\\stations.csv --out-dir D:\\hotel\\gt
  (or --cloud-dir D:\\hotel\\5mm for one PLY per station)

Requires: pip install numpy open3d
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

CLASSES = ["Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
           "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED", "Fire Call Point",
           "Fire Blanket", "Emergency Door Release", "Evacuation Chair"]
COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
           (0, 128, 128), (170, 110, 40)]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


NAME_TO_ID = {norm(c): i for i, c in enumerate(CLASSES)}
NAME_TO_ID.update({str(i): i for i in range(len(CLASSES))})


def class_id(label):
    """Accepts 'Smoke Detector', 'smoke_detector', '5', and '5 = Smoke Detector'."""
    lab = str(label).strip()
    if norm(lab) in NAME_TO_ID:
        return NAME_TO_ID[norm(lab)]
    m = re.match(r"^\s*(\d+)\s*[=:\-_.)\s]", lab + " ")
    if m and int(m.group(1)) < len(CLASSES):
        return int(m.group(1))
    tail = re.split(r"[=:]", lab)[-1]
    return NAME_TO_ID.get(norm(tail), -1)


# ---------------------------------------------------------------- inputs
def read_stations(path):
    rows = list(csv.DictReader(open(path, newline="")))
    by_station = {int(r["station"]): r for r in rows}
    by_image = {int(r["image_index"]): r for r in rows if r.get("image_index") not in ("", None)}
    return by_station, by_image


def station_of_json(fn, by_image):
    stem = os.path.basename(fn)
    m = re.match(r"^S(\d{3})", stem)
    if m:
        return int(m.group(1))
    m = re.match(r"^pano_(\d{3})", stem)
    if m and int(m.group(1)) in by_image:
        return int(by_image[int(m.group(1))]["station"])
    return None


def first_existing(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_pose(path):
    p = json.load(open(path))
    rot, tr = p.get("rotation"), p.get("translation")
    if isinstance(rot, dict):
        R = quat_to_R(rot["w"], rot["x"], rot["y"], rot["z"])
    elif "rotation_matrix" in p:
        R = np.asarray(p["rotation_matrix"], dtype=float)
    else:
        raise ValueError(f"no rotation in {path}")
    t = np.array([tr["x"], tr["y"], tr["z"]] if isinstance(tr, dict) else tr, dtype=float)
    return R, t


def load_cloud(path):
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError(f"no points read from {path}")
    return pts


def clicks_from_json(path):
    d = json.load(open(path, encoding="utf-8"))
    W, H = d.get("imageWidth"), d.get("imageHeight")
    out = []
    for s in d.get("shapes", []):
        pts = np.asarray(s.get("points", []), dtype=float)
        if pts.size == 0:
            continue
        stype = s.get("shape_type", "point")
        if stype == "rectangle" and len(pts) == 2:
            u, v = pts.mean(axis=0)
        else:
            u, v = pts.mean(axis=0) if len(pts) > 1 else pts[0]
        label = s.get("label", "")
        sub = ""
        if "|" in label:
            label, sub = [x.strip() for x in label.split("|", 1)]
        if not sub and str(s.get("description") or "").strip():
            sub = str(s["description"]).strip()
        if not sub and isinstance(s.get("attributes"), dict) and s["attributes"]:
            sub = " ".join(str(v) for v in s["attributes"].values() if str(v).strip())
        out.append({"label": label, "subclass": sub, "u": float(u), "v": float(v),
                    "shape_type": stype, "group_id": s.get("group_id")})
    return W, H, out


# ---------------------------------------------------------------- geometry
def pixel_to_world_dir(u, v, W, H, R):
    """Inverse of build_range_image() in project_mask_to_3d.py (mirrored azimuth)."""
    az = u / W * 2 * np.pi - np.pi
    el = np.pi / 2 - v / H * np.pi
    local = np.array([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    return R @ local


def hit_surface(dirs, rng, pts, ray, radii_deg, depth_band):
    """Nearest surface patch along the ray. Returns (xyz, range, n_points, radius_used)."""
    cosang = dirs @ ray
    for rad in radii_deg:
        cand = np.flatnonzero(cosang > np.cos(np.radians(rad)))
        if len(cand) >= 3:
            r = rng[cand]
            r0 = np.percentile(r, 2) if len(r) >= 50 else r.min()
            front = cand[r <= r0 + depth_band]
            return np.median(pts[front], axis=0), float(np.median(rng[front])), len(front), rad
    return None, None, 0, None


def march_hit(tree, pts, origin, ray, radii_deg, depth_band, s_min=0.3, s_max=40.0):
    """Single merged cloud: step along the ray with a growing search sphere (a cone of
    half-angle `rad`) until it first touches >= 3 points = the nearest surface."""
    for rad in radii_deg:
        tan = np.tan(np.radians(rad))
        s = s_min
        while s < s_max:
            r = max(0.01, s * tan)
            k, idx, _ = tree.search_radius_vector_3d(origin + s * ray, r)
            if k >= 3:
                # gather the surface patch just behind the first hit
                cand = set(idx)
                s2 = s + r
                while s2 <= s + r + depth_band:
                    _, idx2, _ = tree.search_radius_vector_3d(origin + s2 * ray, max(0.01, s2 * tan))
                    cand.update(idx2)
                    s2 += max(0.01, s2 * tan)
                cand = np.fromiter(cand, dtype=np.int64)
                rel = pts[cand] - origin
                along = rel @ ray
                perp = np.linalg.norm(rel - np.outer(along, ray), axis=1)
                keep = (along > 0) & (perp <= np.maximum(0.01, along * tan))
                if keep.sum() >= 3:
                    cand, along = cand[keep], along[keep]
                    front = cand[along <= along.min() + depth_band]
                    rr = np.linalg.norm(pts[front] - origin, axis=1)
                    return np.median(pts[front], axis=0), float(np.median(rr)), len(front), rad
            s += r
    return None, None, 0, None


# ---------------------------------------------------------------- subclass voting
def known(sub):
    return norm(sub) not in ("", "unknown", "unk", "na", "none", "unclear")


def vote_subclass(members):
    """Most frequent known subclass over the views; 'unknown' if none was given."""
    votes = defaultdict(list)
    for m in members:
        if known(m["subclass"]):
            votes[norm(m["subclass"])].append(m["subclass"])
    if not votes:
        return ("unknown" if any(m["subclass"] for m in members) else ""), ""
    ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
    if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
        winner = "TIE"
    else:
        winner = ranked[0][1][0]
    summary = " ".join(f"{v[0]}:{len(v)}" for _, v in ranked)
    return winner, summary


# ---------------------------------------------------------------- merging
# A click hits the object's SURFACE, and different stations see different sides of it,
# so clicks on one object can be up to about half its size apart. Default merge radius
# per class, roughly the object's size (m):
MERGE_RADIUS = {0: 0.25, 1: 0.15, 2: 0.6, 3: 0.4, 4: 0.15, 5: 0.15, 6: 0.4, 7: 0.4,
                8: 0.15, 9: 0.3, 10: 0.15, 11: 0.8}


def merge(clicks, radius_override=None):
    """Complete-linkage merge of same-class clicks, closest pairs first.

    Two clusters merge only if EVERY pair of clicks in them is closer than the class
    radius, and never if they contain two clicks from the same panorama (you clicked
    two different objects there). This keeps side-by-side objects apart.
    """
    objs = []
    by_cls = defaultdict(list)
    for c in clicks:
        if c["xyz"] is not None:
            by_cls[c["class_id"]].append(c)
    for cid, cs in by_cls.items():
        radius = radius_override or MERGE_RADIUS.get(cid, 0.2)
        P = np.array([c["xyz"] for c in cs])
        files = [c["file"] for c in cs]
        n = len(cs)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        same = np.array([[fa == fb for fb in files] for fa in files])
        D[same] = np.inf                       # includes the diagonal
        members = {i: [i] for i in range(n)}
        active = np.ones(n, bool)
        while True:
            Dm = np.where(active[:, None] & active[None, :], D, np.inf)
            k = int(np.argmin(Dm))
            a, b = divmod(k, n)
            if not np.isfinite(Dm[a, b]) or Dm[a, b] >= radius:
                break
            members[a] += members.pop(b)
            active[b] = False
            D[a, :] = np.maximum(D[a, :], D[b, :])     # complete linkage (inf = cannot-link)
            D[:, a] = D[a, :]
            D[a, a] = np.inf
        for a, g in members.items():
            xyz = np.median(P[g], axis=0)
            objs.append({"class_id": cid, "xyz": xyz, "members": [cs[i] for i in g],
                         "radius": radius,
                         "spread": float(np.max(np.linalg.norm(P[g] - xyz, axis=1)))})
    return objs


def flag_neighbours(objs, check_radius):
    """Same-class objects close together: two real objects, or one object clicked badly?"""
    for o in objs:
        o.setdefault("flags", [])
    for a in range(len(objs)):
        for b in range(a + 1, len(objs)):
            oa, ob = objs[a], objs[b]
            if oa["class_id"] != ob["class_id"]:
                continue
            d = float(np.linalg.norm(oa["xyz"] - ob["xyz"]))
            if d < check_radius:
                oa["flags"].append(f"same class as #{ob['id']} {d:.2f} m away")
                ob["flags"].append(f"same class as #{oa['id']} {d:.2f} m away")


# ---------------------------------------------------------------- outputs
def write_ply(path, xyz, rgb, cls, n_views):
    n = len(xyz)
    dt = [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"),
          ("blue", "u1"), ("class_id", "<i4"), ("n_views", "<i4")]
    a = np.empty(n, dtype=dt)
    if n:
        a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        a["class_id"], a["n_views"] = cls, n_views
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property double x\nproperty double y\nproperty double z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "property int class_id\nproperty int n_views\nend_header\n").encode("ascii"))
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--pose-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cloud", help="one point cloud file for everything (e.g. floor1_5mm.ply)")
    g.add_argument("--cloud-dir", help="folder with one PLY per station")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--out-dir", "--out", dest="out", required=True,
                    help="folder for the results (created if missing)")
    ap.add_argument("--merge-radius", type=float, default=None,
                    help="same-class clicks closer than this (m) are one object "
                         "(default: per class, see MERGE_RADIUS in the script)")
    ap.add_argument("--check-radius", type=float, default=0.6,
                    help="flag same-class objects closer than this (m) for a manual look (default 0.6)")
    ap.add_argument("--ray-radius", type=float, nargs="+", default=[0.15, 0.3, 0.6, 1.0],
                    help="search cone half-angles in degrees, tried in order (default 0.15 0.3 0.6 1.0)")
    ap.add_argument("--depth-band", type=float, default=0.03,
                    help="points within this depth (m) behind the nearest hit form the surface patch")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    by_station, by_image = read_stations(a.stations)
    jsons = sorted(glob.glob(os.path.join(a.json_dir, "*.json")))
    jsons = [j for j in jsons if not j.endswith("_pose.json")]
    print(f"{len(jsons)} annotation files")

    tree = all_pts = None
    if a.cloud:
        import open3d as o3d
        print(f"loading {a.cloud} and building KD-tree (can take a few minutes for a big cloud)...")
        pcd = o3d.io.read_point_cloud(a.cloud)
        all_pts = np.asarray(pcd.points, dtype=np.float64)
        if len(all_pts) == 0:
            sys.exit(f"no points read from {a.cloud}")
        tree = o3d.geometry.KDTreeFlann(pcd)
        print(f"  {len(all_pts):,} points")

    clicks, warnings = [], []
    for jp in jsons:
        st = station_of_json(jp, by_image)
        W, H, cl = clicks_from_json(jp)
        name = os.path.basename(jp)
        if not cl:
            continue
        if st is None or st not in by_station:
            warnings.append(f"{name}: cannot tell which station this is - skipped")
            continue
        info = by_station[st]
        si, ii = int(info["scan_index"]), int(info["image_index"])
        pose = first_existing([os.path.join(a.pose_dir, f"S{st:03d}_pano_pose.json"),
                               os.path.join(a.pose_dir, f"pano_{ii:03d}_pose.json")])
        cloud = a.cloud or first_existing([os.path.join(a.cloud_dir, f"S{st:03d}.ply"),
                                           os.path.join(a.cloud_dir, f"S{st:03d}_scan.ply"),
                                           os.path.join(a.cloud_dir, f"{si:03d}_*.ply"),
                                           os.path.join(a.cloud_dir, f"scan_{si:03d}.ply")])
        if not pose or not cloud:
            warnings.append(f"{name}: station {st}: pose {'found' if pose else 'MISSING'}, "
                            f"cloud {'found' if cloud else 'MISSING'} - skipped")
            continue
        if W is None or H is None:
            warnings.append(f"{name}: no imageWidth/imageHeight in JSON - skipped")
            continue
        R, t = load_pose(pose)
        if not a.cloud:
            pts = load_cloud(cloud)
            rel = pts - t
            rng = np.linalg.norm(rel, axis=1)
            ok = rng > 0.2                       # drop points at the scanner itself
            pts, rel, rng = pts[ok], rel[ok], rng[ok]
            dirs = rel / rng[:, None]

        n_ok = 0
        for c in cl:
            cid = class_id(c["label"])
            ray = pixel_to_world_dir(c["u"], c["v"], W, H, R)
            if a.cloud:
                xyz, r, n, rad = march_hit(tree, all_pts, t, ray, a.ray_radius, a.depth_band)
            else:
                xyz, r, n, rad = hit_surface(dirs, rng, pts, ray, a.ray_radius, a.depth_band)
            status = "ok" if xyz is not None else "no points near the click ray"
            if cid < 0:
                status = f"unknown label '{c['label']}'" + ("" if xyz is None else " (kept, class -1)")
            if xyz is not None and rad > a.ray_radius[0]:
                status += f" (wider search {rad} deg)"
            n_ok += xyz is not None
            clicks.append({"file": name, "station": st, "floor": info.get("floor", ""),
                           "label": c["label"], "subclass": c["subclass"], "class_id": cid,
                           "u": c["u"], "v": c["v"],
                           "xyz": xyz, "range": r, "n_pts": n, "status": status})
        print(f"  {name}: station S{st:03d}, {len(cl)} clicks, {n_ok} hit a surface")

    # ---- write per-click table
    with open(os.path.join(a.out, "clicks_3d.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "station", "floor", "label", "subclass", "class_id", "u", "v",
                    "x", "y", "z", "range_m", "n_surface_pts", "status"])
        for c in clicks:
            xyz = c["xyz"] if c["xyz"] is not None else [np.nan] * 3
            w.writerow([c["file"], c["station"], c["floor"], c["label"], c["subclass"], c["class_id"],
                        f"{c['u']:.1f}", f"{c['v']:.1f}", *(f"{v:.3f}" for v in xyz),
                        "" if c["range"] is None else f"{c['range']:.2f}", c["n_pts"], c["status"]])

    # ---- merge into objects
    objs = merge(clicks, a.merge_radius)
    floor_of = lambda o: max({str(m["floor"]) for m in o["members"]},
                             key=lambda f: sum(str(m["floor"]) == f for m in o["members"]))
    objs.sort(key=lambda o: (floor_of(o), o["class_id"], o["xyz"][0], o["xyz"][1]))
    for i, o in enumerate(objs, start=1):
        o["id"] = i
    flag_neighbours(objs, a.check_radius)
    for o in objs:
        o["floor"] = floor_of(o)
        if o["spread"] > o["radius"]:
            o["flags"].append(f"views spread {o['spread']:.2f} m - check clicks")
        if len({m['floor'] for m in o["members"]}) > 1:
            o["flags"].append("seen from stations on different floors")
        o["subclass"], o["sub_votes"] = vote_subclass(o["members"])
        if len({norm(m["subclass"]) for m in o["members"] if known(m["subclass"])}) > 1:
            o["flags"].append(f"subclass differs between views ({o['sub_votes']})")

    with open(os.path.join(a.out, "instances_3d.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "class_id", "class", "floor", "x", "y", "z", "n_views",
                    "stations", "spread_m", "flags", "attribute", "attribute_votes",
                    "ifc_space", "bim_guid", "schedule_row"])
        for o in objs:
            cname = CLASSES[o["class_id"]] if 0 <= o["class_id"] < len(CLASSES) else "UNKNOWN"
            sts = " ".join(sorted({f"S{m['station']:03d}" for m in o["members"]}))
            w.writerow([o["id"], o["class_id"], cname, o["floor"], *(f"{v:.3f}" for v in o["xyz"]),
                        len(o["members"]), sts, f"{o['spread']:.3f}", "; ".join(o["flags"]),
                        o["subclass"], o["sub_votes"], "", "", ""])

    xyz = np.array([o["xyz"] for o in objs]) if objs else np.zeros((0, 3))
    cls = np.array([o["class_id"] for o in objs], dtype=np.int32)
    rgb = np.array([COLOURS[c] if 0 <= c < len(COLOURS) else (255, 255, 255) for c in cls],
                   dtype=np.uint8).reshape(-1, 3)
    write_ply(os.path.join(a.out, "instances_3d.ply"), xyz, rgb, cls,
              np.array([len(o["members"]) for o in objs], dtype=np.int32))

    groups = defaultdict(list)
    for o in objs:
        groups[(o["floor"], o["class_id"])].append(o)
    for (fl, cid), os_ in groups.items():
        cname = norm(CLASSES[cid]) if 0 <= cid < len(CLASSES) else "unknown"
        p = os.path.join(a.out, f"floor{fl}_c{max(cid, 0):02d}_{cname}.txt" if cid >= 0
                         else f"floor{fl}_unknown.txt")
        with open(p, "w") as f:
            for o in os_:
                f.write(f"{o['xyz'][0]:.4f} {o['xyz'][1]:.4f} {o['xyz'][2]:.4f}\n")

    # ---- summary
    print(f"\n{len(clicks)} clicks -> {len(objs)} objects")
    per = defaultdict(int)
    for o in objs:
        per[(o["floor"], o["class_id"])] += 1
    for (fl, cid), n in sorted(per.items()):
        cname = CLASSES[cid] if 0 <= cid < len(CLASSES) else "UNKNOWN"
        print(f"  floor {fl}  {cid:>2} {cname:<24} {n}")
    bad = [c for c in clicks if c["status"] != "ok"]
    flagged = [o for o in objs if o["flags"]]
    if bad:
        print(f"\n{len(bad)} clicks need a look (see status in clicks_3d.csv), e.g.:")
        for c in bad[:10]:
            print(f"  {c['file']} '{c['label']}' at ({c['u']:.0f},{c['v']:.0f}): {c['status']}")
    if flagged:
        print(f"\n{len(flagged)} objects flagged (see flags in instances_3d.csv), e.g.:")
        for o in flagged[:10]:
            print(f"  #{o['id']} {CLASSES[o['class_id']] if o['class_id'] >= 0 else '?'}: {'; '.join(o['flags'])}")
    for wmsg in warnings:
        print("WARNING:", wmsg)
    print(f"\nOutputs in {a.out}")


if __name__ == "__main__":
    sys.exit(main())
