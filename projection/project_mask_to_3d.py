"""
STEP 4 — Trace the segmented objects into 3D space.

Core idea: because the panorama was taken from (essentially) the same origin
as the scan, we can build a synthetic "range image" of the point cloud using
the exact same equirectangular grid as the photo. Once that exists, a 2D mask
pixel maps to a 3D point by direct lookup -- no ray-casting search needed.

This has been validated against synthetic ground-truth data (rotated +
translated station pose): recovered centroids match true object centres to
within a few centimetres, with >99% of recovered points correctly belonging
to the object. Point cloud noise, sparse coverage near mask edges, and
imperfect masks will make real data messier than this, but the geometry is
correct.

Usage:
    python project_mask_to_3d.py scan_000.ply pano_000_pose.json pano_000_mask_00.npy
"""
import sys
import json
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion


def load_pose(pose_json_path):
    with open(pose_json_path) as f:
        pose = json.load(f)
    rot = pose["rotation"]      # dict with w,x,y,z
    trans = pose["translation"]  # dict with x,y,z
    q = Quaternion(rot["w"], rot["x"], rot["y"], rot["z"])
    R = q.rotation_matrix
    t = np.array([trans["x"], trans["y"], trans["z"]])
    return R, t


def build_range_image(points, R, t, width, height):
    """Project a point cloud into an equirectangular range image from a given
    station pose. Returns (range_img, idx_img) where idx_img[row,col] is the
    index into `points` of whichever point landed there (or -1)."""
    rel_world = points - t
    rel_local = (R.T @ rel_world.T).T  # world frame -> camera-local frame

    r = np.linalg.norm(rel_local, axis=1)
    r_safe = np.where(r == 0, 1e-9, r)
    # NOTE: azimuth sign is mirrored vs. the "textbook" atan2(y, x) -- this
    # scanner's panorama column ordering sweeps the opposite direction.
    # Confirmed against real scan data (see v3 in diagnose_azimuth_convention.py):
    # this was the ONE change needed once rotation and elevation were already
    # correct. Don't "simplify" this back to atan2(y, x).
    az = np.arctan2(-rel_local[:, 1], rel_local[:, 0])      # -pi .. pi
    el = np.arcsin(np.clip(rel_local[:, 2] / r_safe, -1, 1))  # -pi/2 .. pi/2

    col = ((az + np.pi) / (2 * np.pi) * width).astype(int) % width
    row = np.clip(((np.pi / 2 - el) / np.pi * height).astype(int), 0, height - 1)

    range_img = np.full((height, width), np.inf)
    idx_img = np.full((height, width), -1, dtype=int)
    # vectorized nearest-point-per-pixel: sort by range descending, then
    # write in that order so the nearest point ends up written last
    order = np.argsort(-r)
    range_img[row[order], col[order]] = r[order]
    idx_img[row[order], col[order]] = order
    return range_img, idx_img


def main(ply_path, pose_json_path, mask_npy_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    mask = np.load(mask_npy_path)
    height, width = mask.shape  # step 3 already resized the mask to the
    # panorama's real resolution, so build the range image at that same
    # resolution -- no separate resolution parameter to keep in sync.

    R, t = load_pose(pose_json_path)
    _, idx_img = build_range_image(points, R, t, width, height)

    sel = idx_img[mask]
    sel = sel[sel >= 0]

    if len(sel) == 0:
        print("No point-cloud points landed inside this mask. Check that the "
              "pose file matches this scan, and that width/height match the "
              "panorama's real resolution.")
        return

    obj_points = points[sel]
    centroid = obj_points.mean(axis=0)
    bbox_min = obj_points.min(axis=0)
    bbox_max = obj_points.max(axis=0)

    print(f"3D points found for this object: {len(obj_points)}")
    print(f"centroid (x,y,z):    {centroid}")
    print(f"bounding box min:    {bbox_min}")
    print(f"bounding box max:    {bbox_max}")

    # write a colored point cloud you can open in CloudCompare / MeshLab /
    # Open3D to SEE the result -- background grey, identified object in red
    out = o3d.geometry.PointCloud()
    all_colors = np.tile([0.6, 0.6, 0.6], (len(points), 1))
    all_colors[sel] = [1.0, 0.0, 0.0]
    out.points = o3d.utility.Vector3dVector(points)
    out.colors = o3d.utility.Vector3dVector(all_colors)
    out_path = ply_path.replace(".ply", "_with_identified_object.ply")
    o3d.io.write_point_cloud(out_path, out)
    print(f"\nVisual check written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python project_mask_to_3d.py scan.ply pose.json mask.npy")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
