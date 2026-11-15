"""
DEBUG STEP — Colorize the WHOLE point cloud from the panorama, not just one mask.

Why: a single object's mask is a noisy thing to debug against -- if it looks
wrong, you can't easily tell whether the mask, the pose, or the pixel<->3D
formula is at fault. Coloring every point in the cloud from the panorama's own
pixels gives you a comprehensive, unambiguous check: since you know the room,
you can immediately tell whether the projection is correct, mirrored,
rotated, or flipped -- just by comparing the result to reality.

Usage:
    python colorize_cloud_from_pano.py scan_000.ply pano_000_pose.json pano_000.jpg

Then open the output .ply in CloudCompare and compare against what you know
of the real room:
  - Looks basically right (a door in the photo lines up with a door-shaped
    gap in the geometry, etc.) -> the projection math is fine, the problem
    is elsewhere (check you're using the matching scan/pose/mask files).
  - Content is left-right mirrored (something that should be on your left
    appears on your right) -> flip the azimuth sign: change
    `az = np.arctan2(rel_local[:, 1], rel_local[:, 0])`
    to
    `az = np.arctan2(-rel_local[:, 1], rel_local[:, 0])`
    in BOTH this script and project_mask_to_3d.py.
  - Content is rotated by some fixed amount around the room (e.g. what's in
    front of you in real life shows up on the side in the render) -> the
    azimuth phase reference doesn't match Leica's column layout. Try adding
    a constant offset to `az` (e.g. `az = az + np.pi/2`) and re-run --
    adjust the offset until it lines up.
  - Content is upside down -> flip the row formula: change
    `row = ... ((np.pi/2 - el) / np.pi * height) ...`
    to
    `row = ... ((el + np.pi/2) / np.pi * height) ...`
  - Content is scattered/nonsensical everywhere, not a clean flip or
    rotation -> more likely a mismatched scan/pose pair than a convention
    issue. Double check scan_000.ply is paired with pano_000_pose.json,
    not pano_001 or a scan_000_pose.json (the SCAN's own pose, not the
    PANORAMA's pose -- they're close but not identical).
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


def main(ply_path, pose_json_path, image_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    img = cv2.imread(image_path)  # BGR
    height, width = img.shape[:2]

    R, t = load_pose(pose_json_path)
    rel_world = points - t
    rel_local = (R.T @ rel_world.T).T

    r = np.linalg.norm(rel_local, axis=1)
    r_safe = np.where(r == 0, 1e-9, r)
    az = np.arctan2(rel_local[:, 1], rel_local[:, 0])
    el = np.arcsin(np.clip(rel_local[:, 2] / r_safe, -1, 1))

    col = ((az + np.pi) / (2 * np.pi) * width).astype(int) % width
    row = np.clip(((np.pi / 2 - el) / np.pi * height).astype(int), 0, height - 1)

    bgr = img[row, col]  # sample a color for every point, in one shot
    rgb = bgr[:, ::-1].astype(np.float64) / 255.0

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(points)
    out.colors = o3d.utility.Vector3dVector(rgb)

    out_path = ply_path.replace(".ply", "_colorized_by_pano.ply")
    o3d.io.write_point_cloud(out_path, out)
    print(f"Written: {out_path}")
    print("Open this in CloudCompare and compare against the real room -- "
          "see the comments at the top of this file for what each kind of "
          "mismatch (mirrored / rotated / upside down / scattered) tells you.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python colorize_cloud_from_pano.py scan.ply pose.json pano.jpg")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
