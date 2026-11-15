"""
Cut out one crop per labelled instance, for Stage B (attribute) sorting.

Reads a YOLO-format dataset (images/ + labels/; segmentation polygons OR plain
boxes both work), and for every instance of the classes you ask for, saves a
padded crop of its bounding box.

Crops go into:
    output_dir/<class_name>/unsorted/<image_stem>__<n>.jpg

You then sort them by hand: create sibling folders next to "unsorted" (for
example co2, foam, powder, unknown) and drag the crops into them. In Windows
Explorer, set the view to "Large icons" or "Extra large icons" -- sorting from
a thumbnail grid is far faster than opening images one by one.

A manifest.csv is written too. It records, per crop, the source image, the
source frame, the video it came from, the box, and the crop size. The video
column is what lets you split train/val BY VIDEO later (a plain per-crop
random split would leak near-duplicate views of the same object into both).

Class names come from --classes-file: either a YOLO data.yaml (its `names:`
entry) or a plain text file with one class name per line, in class-id order.

Usage:
    python crop_instances.py dataset_root/ crops/ ^
        --classes-file data.yaml ^
        --keep "fire extinguisher" "exit sign"

    (dataset_root/ must contain images/ and labels/; use ^ for line breaks in
    Windows cmd, or put it all on one line.)
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FACE_SUFFIX = re.compile(r"_(horizontal|top|bottom)_\d{2}$")


def norm(name):
    return re.sub(r"[\s_\-]+", " ", name.strip().lower())


def load_class_names(path):
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        with open(path, encoding="utf-8") as f:
            names = yaml.safe_load(f)["names"]
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(n) for i, n in enumerate(names)}
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return {i: n for i, n in enumerate(lines)}


def parse_label_file(label_path):
    """Yield (class_id, xmin, ymin, xmax, ymax) in NORMALISED coordinates."""
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:  # plain box: cx cy w h
                cx, cy, w, h = vals
                yield cid, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            else:  # polygon: x1 y1 x2 y2 ...
                if len(vals) % 2:  # stray trailing value (e.g. a confidence)
                    vals = vals[:-1]
                xs, ys = vals[0::2], vals[1::2]
                yield cid, min(xs), min(ys), max(xs), max(ys)


def source_frame_and_video(stem):
    frame = FACE_SUFFIX.sub("", stem)
    video = frame.split("_frame_")[0] if "_frame_" in frame else frame
    return frame, video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help="Folder containing images/ and labels/")
    ap.add_argument("output_dir")
    ap.add_argument("--classes-file", required=True,
                    help="data.yaml, or a text file with one class name per line")
    ap.add_argument("--keep", nargs="+", required=True,
                    help="Class names to crop, e.g. \"fire extinguisher\" \"exit sign\"")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="Padding around the box, as a fraction of box size (default 0.15)")
    ap.add_argument("--min-size", type=int, default=24,
                    help="Skip crops whose shorter side is below this many pixels (default 24)")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to both exist.")

    names = load_class_names(args.classes_file)
    by_norm = {norm(n): cid for cid, n in names.items()}
    keep_ids = {}
    for k in args.keep:
        if norm(k) not in by_norm:
            raise SystemExit(f"Class '{k}' not found. Available: {sorted(names.values())}")
        keep_ids[by_norm[norm(k)]] = names[by_norm[norm(k)]]

    out = Path(args.output_dir)
    rows = []
    counts = {n: 0 for n in keep_ids.values()}
    skipped_small = 0

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"Scanning {len(images)} images for: {', '.join(keep_ids.values())}")

    for i, img_path in enumerate(images, 1):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists() or label_path.stat().st_size == 0:
            continue
        wanted = [t for t in parse_label_file(label_path) if t[0] in keep_ids]
        if not wanted:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  could not read {img_path.name}, skipping")
            continue
        H, W = img.shape[:2]
        frame, video = source_frame_and_video(img_path.stem)

        for n, (cid, x0, y0, x1, y1) in enumerate(wanted):
            bx0, by0, bx1, by1 = x0 * W, y0 * H, x1 * W, y1 * H
            pw, ph = (bx1 - bx0) * args.pad, (by1 - by0) * args.pad
            cx0 = int(max(0, np.floor(bx0 - pw)))
            cy0 = int(max(0, np.floor(by0 - ph)))
            cx1 = int(min(W, np.ceil(bx1 + pw)))
            cy1 = int(min(H, np.ceil(by1 + ph)))

            if min(cx1 - cx0, cy1 - cy0) < args.min_size:
                skipped_small += 1
                continue

            cname = keep_ids[cid]
            dest_dir = out / cname.replace(" ", "_") / "unsorted"
            dest_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{img_path.stem}__{n}.jpg"
            cv2.imwrite(str(dest_dir / fname), img[cy0:cy1, cx0:cx1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            counts[cname] += 1
            rows.append({
                "crop": f"{cname.replace(' ', '_')}/unsorted/{fname}",
                "class": cname,
                "source_image": img_path.name,
                "source_frame": frame,
                "video": video,
                "box_x0": cx0, "box_y0": cy0, "box_x1": cx1, "box_y1": cy1,
                "crop_w": cx1 - cx0, "crop_h": cy1 - cy0,
            })
        if i % 500 == 0:
            print(f"  {i}/{len(images)} images", end="\r")

    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    fields = ["crop", "class", "source_image", "source_frame", "video",
              "box_x0", "box_y0", "box_x1", "box_y1", "crop_w", "crop_h"]
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\nCrops written:")
    for n, c in counts.items():
        print(f"  {n}: {c}")
    print(f"Skipped as too small (< {args.min_size}px on the short side): {skipped_small}")
    print(f"Manifest: {manifest}")
    print("\nNext: open each <class>/unsorted folder, create sibling folders "
          "(e.g. co2, foam, powder, unknown) and drag crops into them.")


if __name__ == "__main__":
    main()
