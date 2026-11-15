"""
Equirectangular -> 18 rectilinear face images, FLAT output for building a
training dataset.

NAMING / RE-RUN BEHAVIOR: your frames/ folder grows over time as you extract
more videos (see extract_and_filter_frames.py), and you'll point this script
at that SAME growing folder repeatedly rather than running it once on a
final, complete set. Re-converting frames you already cube-mapped in a
previous run would just waste time re-doing identical work at your scale
(potentially tens of thousands of faces), so this checks, per source frame,
whether its full set of 18 expected output files already exists in
output_dir -- and if so, skips it. If only SOME of a frame's 18 faces exist
(an interrupted previous run), it's treated as incomplete and reprocessed,
not skipped, so a crash partway through never leaves silently-missing faces.

Output filenames are still prefixed with the source frame's own stem (e.g.
schoolA_video1_frame_000042_horizontal_00.jpg), which is also what makes the
skip check possible -- and, combined with the video-stem prefix now added by
extract_and_filter_frames.py, stays unique across every video and every
school.

Usage:
    python cubemap_convert_for_training.py frames_folder/ training_faces/ \\
        --nb-splits 6 --resolution 1024
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def rotation_matrix_yaw(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_pitch(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def get_rotation_matrix(face_type, index, nb_splits):
    yaw = (index / nb_splits) * 2 * np.pi - np.pi
    if face_type == "top":
        pitch = np.pi / 6
    elif face_type == "bottom":
        pitch = -np.pi / 6
    else:
        pitch = 0.0
    return rotation_matrix_yaw(yaw) @ rotation_matrix_pitch(pitch)


def face_configs(nb_splits):
    configs = []
    for i in range(nb_splits):
        configs.append(("horizontal", i))
        configs.append(("top", i))
        configs.append(("bottom", i))
    return configs


def compute_face_map(face_type, index, nb_splits, split_resolution, src_h, src_w):
    R = get_rotation_matrix(face_type, index, nb_splits)
    f = split_resolution / 2.0

    xs, ys = np.meshgrid(np.arange(split_resolution), np.arange(split_resolution))
    x_norm = (xs - split_resolution / 2.0) / f
    y_norm = (ys - split_resolution / 2.0) / f
    z_norm = np.ones_like(x_norm)

    V = np.stack([x_norm, y_norm, z_norm], axis=-1)
    V_rot = V @ R.T
    V_rot = V_rot / np.linalg.norm(V_rot, axis=-1, keepdims=True)

    lon = np.arctan2(V_rot[..., 0], V_rot[..., 2])
    lat = np.arcsin(np.clip(V_rot[..., 1], -1, 1))

    map_x = ((lon + np.pi) / (2 * np.pi) * src_w).astype(np.float32)
    map_y = ((lat + np.pi / 2) / np.pi * src_h).astype(np.float32)
    return map_x, map_y


def expected_face_names(stem, nb_splits):
    return [f"{stem}_{face_type}_{index:02d}.jpg"
            for face_type, index in face_configs(nb_splits)]


def already_converted(stem, output_dir, nb_splits):
    """True only if EVERY expected face for this frame already exists --
    partial sets (from an interrupted previous run) are not treated as done."""
    return all((output_dir / name).exists()
               for name in expected_face_names(stem, nb_splits))


def equirect_to_faces_flat(pano_path, output_dir, nb_splits=6, split_resolution=640):
    pano_path = Path(pano_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(pano_path))
    src_h, src_w = img.shape[:2]
    stem = pano_path.stem

    count = 0
    for face_type, index in face_configs(nb_splits):
        map_x, map_y = compute_face_map(face_type, index, nb_splits,
                                         split_resolution, src_h, src_w)
        face_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
        name = f"{stem}_{face_type}_{index:02d}.jpg"
        cv2.imwrite(str(output_dir / name), face_img)
        count += 1

    return count


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def batch_equirect_to_faces_flat(input_dir, output_dir, nb_splits=6, split_resolution=640):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f"No images found in {input_dir} (looked for {sorted(IMAGE_EXTS)})")
        return

    to_process = [p for p in images if not already_converted(p.stem, output_dir, nb_splits)]
    skipped = len(images) - len(to_process)

    print(f"Found {len(images)} source image(s): {skipped} already fully "
          f"converted (skipping), {len(to_process)} to process -> "
          f"{len(to_process) * nb_splits * 3} new face images.")

    ok, failed, total_faces = 0, [], 0
    for i, img_path in enumerate(to_process, 1):
        try:
            n = equirect_to_faces_flat(img_path, output_dir, nb_splits, split_resolution)
            total_faces += n
            ok += 1
        except Exception as e:
            failed.append((img_path.name, str(e)))
        print(f"  [{i}/{len(to_process)}] {img_path.name}", end="\r")

    print()
    print(f"Done: {ok}/{len(to_process)} source images converted, "
          f"{total_faces} face images written, {skipped} skipped (already done).")
    if failed:
        print(f"{len(failed)} failed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pano", help="A single panorama file, OR a folder of panoramas.")
    ap.add_argument("output_dir", help="Flat output folder -- all faces from "
                                        "every source image land here together.")
    ap.add_argument("--nb-splits", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()

    src = Path(args.pano)
    if src.is_dir():
        batch_equirect_to_faces_flat(src, args.output_dir, args.nb_splits, args.resolution)
    else:
        if already_converted(src.stem, Path(args.output_dir), args.nb_splits):
            print(f"{src.name} already fully converted in {args.output_dir} -- skipping.")
        else:
            n = equirect_to_faces_flat(src, args.output_dir, args.nb_splits, args.resolution)
            print(f"Wrote {n} faces to {args.output_dir}")
