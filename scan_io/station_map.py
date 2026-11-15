"""
Map E57 file order to real scan-station numbers, find floors and neighbouring
stations, and (optionally) rename already-extracted files to station numbers.

Why: an E57 stores scans sorted by name as text ("1", "10", "100", "101", "11", ...),
so index 0 is station 1, index 1 is station 10, etc. The panoramas (images2D) have
their OWN order, which is not guaranteed to match the scan order - this script
pairs each panorama with its scan through the GUID the E57 stores for that link.

Outputs (in out_dir):
  stations.csv          one row per station: station no., E57 scan index, E57 image
                        index, name, x/y/z position, floor, 4 nearest stations
  stations_floor_N.png  top view of each floor with station numbers

Usage:
  python station_map.py hotel.e57 hotel_stations
  # then preview renaming of files in folders made by the earlier scripts:
  python station_map.py hotel.e57 hotel_stations --rename-dir D:\\scans\\hotel_5mm D:\\scans\\hotel_extract
  # if the preview looks right, actually rename:
  python station_map.py hotel.e57 hotel_stations --rename-dir D:\\scans\\hotel_5mm D:\\scans\\hotel_extract --apply

Renaming rules (index = the 3-digit number the earlier scripts wrote):
  000_<name>.ply        (subsample_e57_stations.py)  -> S001.ply
  scan_000.ply / scan_000_pose.json (extract_e57.py) -> S001_scan.ply / S001_scan_pose.json
  pano_000.jpg / pano_000_pose.json (extract_e57.py) -> S001_pano.jpg / S001_pano_pose.json
  pano_000<anything>    (e.g. later cube-map faces)  -> S001_pano<anything>
Each renamed folder gets a rename_log.csv; --undo-dir <folder> reverses it.

Requires: pip install pye57 numpy matplotlib
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import pye57


def sval(node, key):
    try:
        return node[key].value() if node.isDefined(key) else None
    except Exception:
        return None


def station_number(name, fallback):
    nums = re.findall(r"\d+", name or "")
    return int(nums[-1]) if nums else fallback


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def read_stations(e57_path):
    e57 = pye57.E57(e57_path)
    root = e57.image_file.root()
    data3d = root["data3D"]
    rows, guid_to_scan = [], {}
    for i in range(len(data3d)):
        node = data3d[i]
        name = sval(node, "name") or ""
        guid = sval(node, "guid")
        try:
            t = np.asarray(e57.get_header(i).translation, dtype=float)
        except Exception:
            t = np.array([np.nan] * 3)
        rows.append({"scan_index": i, "name": name, "guid": guid,
                     "station": station_number(name, i + 1),
                     "x": t[0], "y": t[1], "z": t[2], "image_index": None})
        if guid:
            guid_to_scan[guid] = i

    # pair panoramas with scans: GUID link first, then same name, then same index
    images2d = root["images2D"] if root.isDefined("images2D") else []
    name_to_scan = {r["name"]: r["scan_index"] for r in rows}
    unmatched = []
    for j in range(len(images2d)):
        node = images2d[j]
        assoc, iname = sval(node, "associatedData3DGuid"), sval(node, "name")
        if assoc in guid_to_scan:
            k, how = guid_to_scan[assoc], "guid"
        elif iname in name_to_scan:
            k, how = name_to_scan[iname], "name"
        elif j < len(rows):
            k, how = j, "index (UNVERIFIED)"
        else:
            unmatched.append(j)
            continue
        if rows[k]["image_index"] is not None:
            print(f"WARNING: scan {k} already has image {rows[k]['image_index']}; image {j} also points to it")
        rows[k]["image_index"], rows[k]["image_match"] = j, how
    if unmatched:
        print(f"WARNING: images not matched to any scan: {unmatched}")
    return rows


def assign_floors(rows, gap=1.0, min_size=3, breaks=None):
    """Group stations into floors by scanner height.

    breaks given (e.g. [2.8]): floor = 1 + number of breaks below the station's z.
    Otherwise automatic: split the sorted heights at every jump > gap metres; groups with
    fewer than min_size stations (stair landings, ramps) are merged into the nearest floor
    and flagged 'between floors' in the note column.
    """
    for r in rows:
        r["note"] = ""
    if breaks:
        b = sorted(breaks)
        for r in rows:
            r["floor"] = 1 + sum(r["z"] > x for x in b)
    else:
        order = sorted(range(len(rows)), key=lambda k: rows[k]["z"])
        groups, cur = [], [order[0]]
        for a, k in zip(order, order[1:]):
            if rows[k]["z"] - rows[a]["z"] > gap:
                groups.append(cur)
                cur = []
            cur.append(k)
        groups.append(cur)
        zmed = lambda g: float(np.median([rows[k]["z"] for k in g]))
        big = [g for g in groups if len(g) >= min_size] or groups
        # a small group far (> 2 m) from every big floor is a real floor with few stations
        big += [g for g in groups if not any(g is h for h in big)
                and min(abs(zmed(g) - zmed(h)) for h in big) > 2.0]
        big.sort(key=zmed)
        med = [zmed(g) for g in big]
        for fl, g in enumerate(big, start=1):
            for k in g:
                rows[k]["floor"] = fl
        for g in groups:
            if any(g is h for h in big):
                continue
            for k in g:
                fl = int(np.argmin([abs(rows[k]["z"] - m) for m in med])) + 1
                rows[k]["floor"] = fl
    # flag stations far from their floor's typical height (stairs, landings, ramps)
    for f in {r["floor"] for r in rows}:
        m = np.median([r["z"] for r in rows if r["floor"] == f])
        for r in rows:
            if r["floor"] == f and abs(r["z"] - m) > 0.6:
                r["note"] = f"between floors? z {r['z'] - m:+.2f} m from floor median"


def add_neighbours(rows, k_nn=4):
    for f in sorted({r["floor"] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r["floor"] == f]
        xy = np.array([[rows[k]["x"], rows[k]["y"]] for k in idx])
        for a, k in enumerate(idx):
            d = np.hypot(*(xy - xy[a]).T)
            near = [b for b in np.argsort(d) if b != a][:k_nn]
            rows[k]["neighbours"] = " ".join(f"S{rows[idx[b]]['station']:03d}({d[b]:.1f}m)" for b in near)


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["station", "scan_index", "image_index", "image_match", "name", "floor",
            "x", "y", "z", "neighbours", "note"]
    path = os.path.join(out_dir, "stations.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["station"]):
            w.writerow({**r, "x": f"{r['x']:.3f}", "y": f"{r['y']:.3f}", "z": f"{r['z']:.3f}"})
    print(f"wrote {path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping floor plots")
        return
    for f in sorted({r["floor"] for r in rows}):
        fr = [r for r in rows if r["floor"] == f]
        xs, ys = [r["x"] for r in fr], [r["y"] for r in fr]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        fig, ax = plt.subplots(figsize=(14, 14 * (max(ys) - min(ys) + 2) / (max(xs) - min(xs) + 2)))
        ax.scatter(xs, ys, s=12, color="#1f77b4")
        for r in fr:
            ax.annotate(str(r["station"]), (r["x"], r["y"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
        zs = [r["z"] for r in fr]
        ax.set_title(f"Floor {f}: {len(fr)} stations, scanner height {min(zs):.2f}-{max(zs):.2f} m")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"stations_floor_{f}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


def plan_renames(rows, folder):
    by_scan = {r["scan_index"]: r for r in rows}
    by_image = {r["image_index"]: r for r in rows if r["image_index"] is not None}
    plans = []
    for fn in sorted(os.listdir(folder)):
        if fn == "rename_log.csv" or re.match(r"^S\d{3}", fn):
            continue
        m = re.match(r"^(scan_|pano_)?(\d{3})(.*)$", fn)
        if not m:
            continue
        kind, idx, rest = m.group(1) or "", int(m.group(2)), m.group(3)
        r = (by_image if kind == "pano_" else by_scan).get(idx)
        if r is None:
            print(f"  ? no station for {fn}")
            continue
        if not kind and rest.startswith("_" + safe(r["name"]) + "."):
            rest = rest[len(safe(r["name"])) + 1:]          # 000_<name>.ply -> S001.ply
        new = f"S{r['station']:03d}" + (f"_{kind[:-1]}" if kind else "") + rest
        plans.append((fn, new))
    news = [n for _, n in plans]
    clash = {n for n in news if news.count(n) > 1 or (os.path.exists(os.path.join(folder, n)))}
    if clash:
        print(f"  STOP: target names clash or already exist: {sorted(clash)[:5]} ...")
        return []
    return plans


def rename(rows, folders, apply):
    for folder in folders:
        plans = plan_renames(rows, folder)
        print(f"\n{folder}: {len(plans)} files to rename" + ("" if apply else " (preview, add --apply)"))
        for old, new in plans[:8]:
            print(f"  {old}  ->  {new}")
        if len(plans) > 8:
            print(f"  ... and {len(plans) - 8} more")
        if apply and plans:
            log = os.path.join(folder, "rename_log.csv")
            with open(log, "a", newline="") as f:
                w = csv.writer(f)
                for old, new in plans:
                    os.rename(os.path.join(folder, old), os.path.join(folder, new))
                    w.writerow([old, new])
            print(f"  renamed; log in {log}")


def undo(folder):
    log = os.path.join(folder, "rename_log.csv")
    with open(log, newline="") as f:
        pairs = list(csv.reader(f))
    for old, new in reversed(pairs):
        if os.path.exists(os.path.join(folder, new)):
            os.rename(os.path.join(folder, new), os.path.join(folder, old))
    os.rename(log, log + ".undone")
    print(f"{folder}: restored {len(pairs)} names")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e57", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--floor-gap", type=float, default=1.0,
                    help="height jump (m) that starts a new floor (default 1.0)")
    ap.add_argument("--floor-breaks", type=float, nargs="*",
                    help="manual floor boundaries in scanner z (m), e.g. --floor-breaks 2.8")
    ap.add_argument("--rename-dir", nargs="*", default=[], help="folders whose files to rename")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: preview only)")
    ap.add_argument("--undo-dir", nargs="*", default=[], help="reverse an earlier rename")
    a = ap.parse_args()

    if a.undo_dir:
        for d in a.undo_dir:
            undo(d)
        return
    if not (a.e57 and a.out_dir):
        ap.error("e57 and out_dir are required")

    rows = read_stations(a.e57)
    print(f"{len(rows)} scans, {sum(r['image_index'] is not None for r in rows)} with a panorama")
    hows = {}
    for r in rows:
        if r["image_index"] is not None:
            hows[r["image_match"]] = hows.get(r["image_match"], 0) + 1
    if hows:
        print("panorama-to-scan pairing:", hows)
    if len({r["station"] for r in rows}) != len(rows):
        print("WARNING: station numbers parsed from names are not unique - check the 'name' column")
    assign_floors(rows, a.floor_gap, breaks=a.floor_breaks)
    add_neighbours(rows)
    for f in sorted({r["floor"] for r in rows}):
        zs = [r["z"] for r in rows if r["floor"] == f]
        print(f"floor {f}: {len(zs)} stations, z {min(zs):.2f} to {max(zs):.2f} m")
    for r in sorted(rows, key=lambda r: r["station"]):
        if r["note"]:
            print(f"  S{r['station']:03d} (floor {r['floor']}): {r['note']}")
    write_outputs(rows, a.out_dir)
    if a.rename_dir:
        rename(rows, a.rename_dir, a.apply)


if __name__ == "__main__":
    sys.exit(main())
