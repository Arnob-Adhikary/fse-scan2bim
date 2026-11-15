"""
DEBUG STEP 2 — Try several plausible axis conventions at once.

We've confirmed the pose data itself is clean (proper rotation, matches
between scan and pano). So if the full-panorama colorization still looks
wrong, the remaining suspects are: which way the rotation is applied (R vs
its transpose), and which way azimuth increases (some panorama exporters
sweep the opposite direction, which mirrors everything left-right).

This writes FOUR versions of the colorized point cloud, one per candidate
convention. Open all four in CloudCompare and see which one actually matches
the real room -- then tell me which (by suffix) and we lock that convention
into every script.

Usage:
    python diagnose_azimuth_convention.py scan_002.ply pano_002_pose.json pano_002.jpg
"""
import sys
import json
import numpy as np
import open3d as o3d
import cv2
from pyquaternion import Quaternion


def load_pose(pose_json_path):
    with open(pose_json_path) as f:
        pose = json.load(f)
    rot = pose["rotation"]
    trans = pose["translation"]
    q = Quaternion(rot["w"], rot["x"], rot["y"], rot["z"])
    return q.rotation_matrix, np.array([trans["x"], trans["y"], trans["z"]])


def colorize(points, img, R, t, mirror_azimuth):
    height, width = img.shape[:2]
    rel_world = points - t
    rel_local = (R @ rel_world.T).T  # R already chosen as-is or transposed by caller

    r = np.linalg.norm(rel_local, axis=1)
    r_safe = np.where(r == 0, 1e-9, r)
    sign = -1.0 if mirror_azimuth else 1.0
    az = np.arctan2(sign * rel_local[:, 1], rel_local[:, 0])
    el = np.arcsin(np.clip(rel_local[:, 2] / r_safe, -1, 1))

    col = ((az + np.pi) / (2 * np.pi) * width).astype(int) % width
    row = np.clip(((np.pi / 2 - el) / np.pi * height).astype(int), 0, height - 1)

    bgr = img[row, col]
    return bgr[:, ::-1].astype(np.float64) / 255.0


def main(ply_path, pose_json_path, image_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    height, width = img.shape[:2]
    print(f"Image dimensions: {width} x {height}  (ratio {width/height:.2f}; "
          f"true equirectangular should be ~2.00)")

    R, t = load_pose(pose_json_path)

    variants = [
        ("v1_Rt_normal",      R.T, False),
        ("v2_R_notranspose",  R,   False),
        ("v3_Rt_mirrored",    R.T, True),
        ("v4_R_mirrored",     R,   True),
    ]

    for suffix, R_used, mirror in variants:
        rgb = colorize(points, img, R_used, t, mirror)
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(points)
        out.colors = o3d.utility.Vector3dVector(rgb)
        out_path = ply_path.replace(".ply", f"_{suffix}.ply")
        o3d.io.write_point_cloud(out_path, out)
        print(f"  wrote {out_path}")

    print("\nOpen all four in CloudCompare and tell me which suffix looks "
          "correct (or closest) -- that pins down the fix for every script.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python diagnose_azimuth_convention.py scan.ply pose.json pano.jpg")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
