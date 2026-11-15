"""
Equirectangular -> 18 rectilinear face images (modified cube-map conversion).

Implements the algorithm from Wen et al., "Semantic BIM enrichment for
firefighting assets" (Fire-ART paper), Fig. 9: 6 horizontal faces (30-degree
overlap) + 6 faces tilted 30 deg up + 6 tilted 30 deg down = 18 faces total,
instead of the conventional 6-face cube map. This avoids assets being split
across face boundaries and keeps ceiling/floor-mounted assets in natural-
looking (not fisheye-distorted) views.

This is purely an IMAGE-SPACE operation -- it does not touch world/BIM
coordinates at all, so it has no interaction with (and needs no consistency
with) the azimuth/elevation convention used elsewhere in the point-cloud
back-projection code. It converts an equirectangular image to rectilinear
faces and, later, converts rectilinear predictions back to an equirectangular
mask -- what happens in between (running a detector on each face) is the only
thing that needs both ends to agree with each other, which this module
guarantees by construction.

Usage:
    python cubemap_convert_v2.py pano.jpg output_dir/ --nb-splits 6 --resolution 640
"""
import argparse
import json
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
    """Returns (map_x, map_y): for each pixel in the output rectilinear
    face, which source equirectangular pixel to sample from (for cv2.remap)."""
    R = get_rotation_matrix(face_type, index, nb_splits)
    f = split_resolution / 2.0

    xs, ys = np.meshgrid(np.arange(split_resolution), np.arange(split_resolution))
    x_norm = (xs - split_resolution / 2.0) / f
    y_norm = (ys - split_resolution / 2.0) / f
    z_norm = np.ones_like(x_norm)

    V = np.stack([x_norm, y_norm, z_norm], axis=-1)  # (res, res, 3)
    V_rot = V @ R.T
    V_rot = V_rot / np.linalg.norm(V_rot, axis=-1, keepdims=True)

    lon = np.arctan2(V_rot[..., 0], V_rot[..., 2])
    lat = np.arcsin(np.clip(V_rot[..., 1], -1, 1))

    map_x = ((lon + np.pi) / (2 * np.pi) * src_w).astype(np.float32)
    map_y = ((lat + np.pi / 2) / np.pi * src_h).astype(np.float32)
    return map_x, map_y


def equirect_to_faces(pano_path, output_dir, nb_splits=6, split_resolution=640):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(pano_path))
    src_h, src_w = img.shape[:2]

    manifest = {"source": str(pano_path), "src_h": src_h, "src_w": src_w,
                "nb_splits": nb_splits, "split_resolution": split_resolution,
                "faces": []}

    for face_type, index in face_configs(nb_splits):
        map_x, map_y = compute_face_map(face_type, index, nb_splits,
                                         split_resolution, src_h, src_w)
        face_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
        name = f"{face_type}_{index:02d}.jpg"
        cv2.imwrite(str(output_dir / name), face_img)
        manifest["faces"].append({"file": name, "face_type": face_type, "index": index})

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def batch_equirect_to_faces(input_dir, output_dir, nb_splits=6, split_resolution=640):
    """Run equirect_to_faces over every image in input_dir. Each source image
    gets its own subfolder under output_dir (named after the image's
    filename stem), since each one has its own 18 faces + manifest.json that
    would otherwise collide in one flat folder."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f"No images found in {input_dir} (looked for {sorted(IMAGE_EXTS)})")
        return

    print(f"Found {len(images)} image(s) to convert.")
    ok, failed = 0, []
    for i, img_path in enumerate(images, 1):
        dest = output_dir / img_path.stem
        try:
            equirect_to_faces(img_path, dest, nb_splits, split_resolution)
            ok += 1
        except Exception as e:
            failed.append((img_path.name, str(e)))
        print(f"  [{i}/{len(images)}] {img_path.name} -> {dest}/", end="\r")

    print()  # clear the progress line
    print(f"Done: {ok}/{len(images)} converted successfully.")
    if failed:
        print(f"{len(failed)} failed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pano", help="A single panorama file, OR a folder of panoramas.")
    ap.add_argument("output_dir")
    ap.add_argument("--nb-splits", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=640)
    args = ap.parse_args()

    src = Path(args.pano)
    if src.is_dir():
        batch_equirect_to_faces(src, args.output_dir, args.nb_splits, args.resolution)
    else:
        m = equirect_to_faces(src, args.output_dir, args.nb_splits, args.resolution)
        print(f"Wrote {len(m['faces'])} faces to {args.output_dir}")
