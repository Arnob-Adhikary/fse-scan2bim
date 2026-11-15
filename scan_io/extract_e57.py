"""
STEP 2 — Extract the point cloud and the embedded panoramas + their pose.

Run inspect_e57.py FIRST so you know whether your images use
'sphericalRepresentation' (full 360 pano — the normal case for a Leica
integrated-camera scan) or something else. This script handles the spherical
case; adjust the representation name if yours differs.

Usage:
    python extract_e57.py path/to/your_scan.e57 output_folder/
"""
import sys
import os
import json
import numpy as np
import open3d as o3d
import pye57


def extract_point_clouds(e57, out_dir):
    data3d = e57.image_file.root()["data3D"]
    for i in range(len(data3d)):
        d = e57.read_scan(i, colors=True, ignore_missing_fields=True)
        xyz = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if "colorRed" in d:
            rgb = np.column_stack([d["colorRed"], d["colorGreen"], d["colorBlue"]]) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        header = e57.get_header(i)
        out_ply = os.path.join(out_dir, f"scan_{i:03d}.ply")
        o3d.io.write_point_cloud(out_ply, pcd)

        # save this scan's pose too -- some scanners store per-scan pose here
        pose = {
            "rotation_matrix": header.rotation_matrix.tolist(),
            "translation": header.translation.tolist(),
        }
        with open(os.path.join(out_dir, f"scan_{i:03d}_pose.json"), "w") as f:
            json.dump(pose, f, indent=2)

        print(f"  scan {i}: {len(xyz)} points -> {out_ply}")


def extract_images(e57, out_dir):
    images2d = e57.image_file.root()["images2D"]
    print(f"  found {len(images2d)} embedded image(s)")

    for i in range(len(images2d)):
        node = images2d[i]

        # --- pose: separate from the scanner's own pose, since the camera
        # centre is offset from the scanner centre (Leica documents this) ---
        pose = {"rotation": None, "translation": None}
        if node.isDefined("pose"):
            p = node["pose"]
            if p.isDefined("rotation"):
                r = p["rotation"]
                pose["rotation"] = {k: r[k].value() for k in ("w", "x", "y", "z")}
            if p.isDefined("translation"):
                t = p["translation"]
                pose["translation"] = {k: t[k].value() for k in ("x", "y", "z")}

        # --- the actual image bytes ---
        # try the common representations in order; print what's found so you
        # can tell which branch your data actually uses
        blob = None
        rep_used = None
        for rep_name, blob_field in [
            ("sphericalRepresentation", "jpegImage"),
            ("pinholeRepresentation", "jpegImage"),
            ("visualReferenceRepresentation", "jpegImage"),
        ]:
            if node.isDefined(rep_name):
                rep = node[rep_name]
                if rep.isDefined(blob_field):
                    blob = rep[blob_field]
                    rep_used = rep_name
                    break

        if blob is None:
            print(f"  image {i}: no JPEG blob found under known representations — "
                  f"re-run inspect_e57.py and check the field names printed there.")
            continue

        img_bytes = blob.read_buffer().tobytes()
        img_path = os.path.join(out_dir, f"pano_{i:03d}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        with open(os.path.join(out_dir, f"pano_{i:03d}_pose.json"), "w") as f:
            json.dump({"representation": rep_used, **pose}, f, indent=2)

        print(f"  image {i} ({rep_used}) -> {img_path}")


def main(e57_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    e57 = pye57.E57(e57_path)

    print("Extracting point cloud(s)...")
    extract_point_clouds(e57, out_dir)

    print("\nExtracting embedded panoramas + pose...")
    extract_images(e57, out_dir)

    print(f"\nDone. Everything is in {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_e57.py path/to/your_scan.e57 output_folder/")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
