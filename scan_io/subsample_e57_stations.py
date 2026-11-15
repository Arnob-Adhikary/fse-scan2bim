"""
Subsample every scan station in one E57 file to a fixed spacing and save
each station as its own PLY file (global/registered coordinates).

- Reads one station at a time, so RAM use stays at one station's size.
- Keeps ORIGINAL points (one real point per voxel, no averaging), so
  coordinates and colours are true scan values - suitable for ground truth.
- Writes XYZ as float64, so large Dutch RD coordinates keep full precision
  (no CloudCompare "global shift" needed when exporting).
- Keeps RGB and intensity if the E57 has them.

Usage (Windows, in the folder with the script):
    python subsample_e57_stations.py "D:\\scans\\hotel.e57" "D:\\scans\\hotel_5mm"
    python subsample_e57_stations.py hotel.e57 out_dir --voxel 0.005
    python subsample_e57_stations.py hotel.e57 out_dir --only 0 1 2   # test on 3 stations

Requires: pip install pye57 numpy
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pye57


def station_name(e57, i):
    try:
        name = e57.get_header(i).node["name"].value()
    except Exception:
        name = ""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return f"{i:03d}_{name}" if name else f"{i:03d}"


def voxel_keep_first(xyz, voxel):
    """Index of one original point per voxel (the first one encountered)."""
    ijk = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, idx = np.unique(key, return_index=True)
    return np.sort(idx)


def write_ply(path, xyz, rgb=None, intensity=None):
    n = len(xyz)
    fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    header = ["property double x", "property double y", "property double z"]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    if intensity is not None:
        fields += [("intensity", "<f4")]
        header += ["property float intensity"]
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if intensity is not None:
        arr["intensity"] = intensity
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n" + "\n".join(header) +
                 "\nend_header\n").encode("ascii"))
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e57")
    ap.add_argument("out_dir")
    ap.add_argument("--voxel", type=float, default=0.005, help="spacing in metres (default 0.005)")
    ap.add_argument("--only", type=int, nargs="*", help="station indices to process (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="redo stations whose PLY already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    e57 = pye57.E57(args.e57)
    n_scans = e57.scan_count
    todo = args.only if args.only else range(n_scans)
    print(f"{args.e57}: {n_scans} stations, voxel {args.voxel * 1000:.1f} mm")

    for i in todo:
        name = station_name(e57, i)
        out = os.path.join(args.out_dir, f"{name}.ply")
        if os.path.exists(out) and not args.overwrite:
            print(f"[{i + 1}/{n_scans}] {name}: exists, skipped")
            continue
        t0 = time.time()
        d = e57.read_scan(i, colors=True, intensity=True, ignore_missing_fields=True)
        xyz = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]).astype(np.float64)

        valid = np.isfinite(xyz).all(axis=1)
        if "cartesianInvalidState" in d:
            valid &= np.asarray(d["cartesianInvalidState"]) == 0
        n_raw = len(xyz)

        rgb = None
        if all(k in d for k in ("colorRed", "colorGreen", "colorBlue")):
            rgb = np.column_stack([d["colorRed"], d["colorGreen"], d["colorBlue"]])
        inten = np.asarray(d["intensity"], dtype=np.float32) if "intensity" in d else None

        xyz = xyz[valid]
        rgb = rgb[valid] if rgb is not None else None
        inten = inten[valid] if inten is not None else None

        keep = voxel_keep_first(xyz, args.voxel)
        if rgb is not None:
            rgb = rgb[keep]
            if rgb.max() > 255:  # some writers store 16-bit colour
                rgb = rgb / rgb.max() * 255
            rgb = rgb.astype(np.uint8)
        write_ply(out, xyz[keep], rgb, inten[keep] if inten is not None else None)
        print(f"[{i + 1}/{n_scans}] {name}: {n_raw:,} -> {len(keep):,} points "
              f"({100 * len(keep) / max(n_raw, 1):.0f}%), {time.time() - t0:.0f} s")
        del d, xyz, rgb, inten, keep

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
