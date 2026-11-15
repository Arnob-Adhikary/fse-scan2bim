"""
Small corrections to the ground truth written by cutout_instances.py, without re-running
anything: change an object's class, merge a duplicate into another object, or delete one.

Edits, in --gt-dir:
  instances_gt.csv, gt_labelled.ply, gt_objects_only.ply
The originals are kept once as *.bak (the first time you run this).

Examples
  python edit_gt.py --gt-dir D:\\hotel\\gt\\cutout_f1 --relabel 5 4
        object 5 becomes class 4 (Fire Indicator)
  python edit_gt.py --gt-dir D:\\hotel\\gt\\cutout_f1 --merge 5 34
        object 5 is a duplicate of 34: its points go to 34, its row is removed
  python edit_gt.py --gt-dir D:\\hotel\\gt\\cutout_f1 --delete 12
        object 12 is not a real object: its points become background, its row is removed
Several edits can be given in one call; they are applied in the order relabel, merge, delete.

Also correct the label in the X-AnyLabeling JSON, so a future re-run gives the same result.
NOTE: re-running pano_clicks_to_3d.py renumbers the objects, so hand-cut fix files
(<id>.ply) would then have to be renamed.

Requires: pip install numpy
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np

CLASSES = ["Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
           "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED", "Fire Call Point",
           "Fire Blanket", "Emergency Door Release", "Evacuation Chair"]
PLY_TYPES = {"double": "<f8", "float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2", "char": "i1"}


def read_ply(path):
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline().decode("ascii").strip()
            head.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in head:
            sys.exit(f"{path}: expected a binary little-endian PLY (as written by cutout_instances.py)")
        dt = [(l.split()[2], PLY_TYPES[l.split()[1]]) for l in head if l.startswith("property")]
        data = np.fromfile(f, dtype=dt)
    return head, data.copy()


def write_ply(path, head, data):
    n_line = [i for i, l in enumerate(head) if l.startswith("element vertex")][0]
    head = list(head)
    head[n_line] = f"element vertex {len(data)}"
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(data.tobytes())


def backup(path):
    if os.path.exists(path) and not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, help="the --out-dir of cutout_instances.py")
    ap.add_argument("--relabel", nargs=2, type=int, action="append", default=[],
                    metavar=("ID", "CLASS_ID"))
    ap.add_argument("--merge", nargs=2, type=int, action="append", default=[],
                    metavar=("FROM_ID", "INTO_ID"))
    ap.add_argument("--delete", type=int, action="append", default=[], metavar="ID")
    a = ap.parse_args()

    csv_path = os.path.join(a.gt_dir, "instances_gt.csv")
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    cols = list(rows[0].keys())
    by_id = {int(r["instance_id"]): r for r in rows}
    for i in [x for x, _ in a.relabel] + [x for p in a.merge for x in p] + a.delete:
        if i not in by_id:
            sys.exit(f"object {i} is not in {csv_path}")

    plys = [p for p in ("gt_labelled.ply", "gt_objects_only.ply")
            if os.path.exists(os.path.join(a.gt_dir, p))]
    loaded = {p: read_ply(os.path.join(a.gt_dir, p)) for p in plys}

    for i, c in a.relabel:
        r = by_id[i]
        print(f"#{i}: {r['class']} -> {CLASSES[c]}")
        r["class_id"], r["class"] = str(c), CLASSES[c]
        for k in ("attribute", "attribute_votes"):   # the old subclass belonged to the old class
            if k in r:
                r[k] = ""
        for head, d in loaded.values():
            d["class_id"][d["instance_id"] == i] = c

    for src, dst in a.merge:
        rs, rd = by_id[src], by_id[dst]
        print(f"#{src} ({rs['class']}) merged into #{dst} ({rd['class']})")
        if "stations" in rd:
            sts = sorted(set(rs.get("stations", "").split()) | set(rd["stations"].split()))
            rd["stations"] = " ".join(sts)
        if "n_views" in rd:
            rd["n_views"] = str(int(rd["n_views"] or 0) + int(rs.get("n_views", 0) or 0))
        for head, d in loaded.values():
            m = d["instance_id"] == src
            d["instance_id"][m] = dst
            d["class_id"][m] = int(rd["class_id"])
        if plys:
            d = loaded[plys[0]][1]
            m = d["instance_id"] == dst
            rd["n_points"] = str(int(m.sum()))
            if m.sum() and "box_cx" in rd:
                X = np.c_[d["x"][m], d["y"][m], d["z"][m]]
                lo, hi = np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)
                for k, v in zip(("box_cx", "box_cy", "box_cz"), (lo + hi) / 2):
                    rd[k] = f"{v:.3f}"
                for k, v in zip(("size_x", "size_y", "size_z"), hi - lo):
                    rd[k] = f"{v:.3f}"
        rd["status"] = (rd.get("status", "") + f"; merged #{src}").lstrip("; ")
        del by_id[src]

    for i in a.delete:
        print(f"#{i} ({by_id[i]['class']}) deleted")
        for head, d in loaded.values():
            m = d["instance_id"] == i
            d["instance_id"][m] = 0
            d["class_id"][m] = -1
        del by_id[i]

    # objects-only file: drop points that became background
    if "gt_objects_only.ply" in loaded:
        head, d = loaded["gt_objects_only.ply"]
        loaded["gt_objects_only.ply"] = (head, d[d["instance_id"] > 0])

    backup(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows([by_id[i] for i in sorted(by_id)])
    for p, (head, d) in loaded.items():
        backup(os.path.join(a.gt_dir, p))
        write_ply(os.path.join(a.gt_dir, p), head, d)
    print(f"done: {len(by_id)} objects in {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
