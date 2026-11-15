"""
Stage 2 inference: give every detected object a sub-type with a saved probe.

    YOLO26-seg detection  ->  crop each box  ->  frozen DINOv2 features
                          ->  saved logistic-regression probe  ->  sub-type

The probe comes from probe_separability_v4.py --save-probe, which stores the
backbone, input size and feature type with the classifier, so the features
here are computed exactly as in training (same letterbox, same CLS + mean-patch
feature). This file imports that code, so keep both scripts in one folder.

Two ways to run it:

  1. On images (e.g. cube-map faces of the point-cloud panoramas): YOLO finds
     the objects, each box is cropped and classified.
        python classify_subtypes.py --probe probes/extinguisher_dinov2s_448.joblib ^
            --yolo runs/EXP004/weights/best.pt --images faces/ --out results/

  2. On crops you already have (same crop style as the training crops):
        python classify_subtypes.py --probe probes/extinguisher_dinov2s_448.joblib ^
            --crops some_crops/ --out results/

Outputs in --out:
  subtypes.csv   one row per object: image, box, detection confidence,
                 predicted sub-type, probability of each sub-type, status
  crops/<sub-type>/ and crops/uncertain/   (with --save-crops) the crops
                 sorted by prediction, named <image>__<n>.jpg like the
                 training crops, so checked ones can be added to training.

A prediction whose top probability is below --min-prob is marked "uncertain"
instead of being trusted; review those by eye.

Crop padding (--pad, --pad-px) must match how the training crops were cut.
If the training crops are the plain box, keep the defaults (no padding).

Requires: pip install ultralytics torch transformers scikit-learn joblib pillow numpy
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from probe_separability_v4 import IMAGE_EXTS, load_extractor
except ImportError:
    sys.exit("classify_subtypes.py needs probe_separability_v4.py in the same "
             "folder.")


def load_probe(path):
    import joblib
    import sklearn
    bundle = joblib.load(path)
    if bundle.get("feature") != "CLS+mean-patch" or \
            bundle.get("preprocess") != "letterbox":
        sys.exit("This probe was saved with different feature settings than "
                 "this script uses.")
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(f"Note: probe saved with scikit-learn "
              f"{bundle.get('sklearn_version')}, running "
              f"{sklearn.__version__}. If predictions look wrong, re-save the "
              f"probe with this version.")
    print(f"Probe: {path}\n  {bundle['model']} at {bundle['img_size']} px, "
          f"classes {bundle['classes']}, trained on {bundle['n_crops']} crops "
          f"{bundle['crops_per_class']}")
    return bundle


def list_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        sys.exit(f"{p} not found.")
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {p}.")
    return files


def crop_box(img, box, pad, pad_px):
    """img: PIL RGB; box: x1, y1, x2, y2 in pixels."""
    x1, y1, x2, y2 = box
    mx = pad * (x2 - x1) + pad_px
    my = pad * (y2 - y1) + pad_px
    w, h = img.size
    l, t = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
    r, b = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    return img.crop((l, t, r, b)), (l, t, r, b)


def detections_from_yolo(args):
    """Yields (image_path, PIL image, [(box, det_conf, det_name), ...])."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Detection needs ultralytics: pip install ultralytics")
    model = YOLO(args.yolo)
    names = model.names
    if args.yolo_class not in names:
        sys.exit(f"Class index {args.yolo_class} is not in this model: {names}")
    print(f"YOLO: {args.yolo}, class {args.yolo_class} "
          f"({names[args.yolo_class]}), conf >= {args.det_conf}, "
          f"imgsz {args.imgsz}")
    files = list_images(args.images)
    for i, f in enumerate(files, 1):
        res = model.predict(source=str(f), imgsz=args.imgsz,
                            conf=args.det_conf, classes=[args.yolo_class],
                            device=args.device, verbose=False)[0]
        # orig_img is BGR; convert once for cropping
        img = Image.fromarray(res.orig_img[:, :, ::-1].copy())
        dets = []
        if res.boxes is not None and len(res.boxes):
            for box, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                 res.boxes.conf.cpu().numpy()):
                dets.append((box.tolist(), float(conf), names[args.yolo_class]))
        print(f"  images {i}/{len(files)}", end="\r")
        yield f, img, dets
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Classify detected objects into sub-types with a saved "
                    "probe from probe_separability_v4.py --save-probe.")
    ap.add_argument("--probe", required=True, help="Saved .joblib probe")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="Image file or folder: run YOLO, then "
                                      "classify each detection")
    src.add_argument("--crops", help="Folder of ready-made crops to classify")
    ap.add_argument("--yolo", help="YOLO weights (best.pt), needed with --images")
    ap.add_argument("--yolo-class", type=int, default=0,
                    help="Detector class to classify (default 0 = Fire "
                         "Extinguisher)")
    ap.add_argument("--det-conf", type=float, default=0.25,
                    help="Minimum YOLO confidence for a detection")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="YOLO input size (1024, as in training, D-17)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="Crop margin as a fraction of box size; match the "
                         "training crops")
    ap.add_argument("--pad-px", type=float, default=0.0,
                    help="Extra crop margin in pixels; match the training crops")
    ap.add_argument("--min-prob", type=float, default=0.8,
                    help="Below this top probability the result is 'uncertain'")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--save-crops", action="store_true",
                    help="Also save the crops sorted into folders by prediction")
    args = ap.parse_args()
    if args.images and not args.yolo:
        sys.exit("--images needs --yolo (the detector weights).")

    bundle = load_probe(args.probe)
    clf, classes = bundle["classifier"], bundle["classes"]
    extract = load_extractor(bundle["model"], bundle["img_size"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.crops:
        source = ((f, None, None) for f in list_images(args.crops))
    else:
        source = detections_from_yolo(args)

    rows = []
    for f, img, dets in source:
        if img is None:                       # ready-made crop
            crops = [Image.open(f).convert("RGB")]
            meta = [dict(det="", x1="", y1="", x2="", y2="", det_conf="",
                         det_class="")]
            names = [f.name]
        else:
            if not dets:
                continue
            crops, meta, names = [], [], []
            for n, (box, conf, det_name) in enumerate(dets):
                c, (l, t, r, b) = crop_box(img, box, args.pad, args.pad_px)
                if c.size[0] < 2 or c.size[1] < 2:
                    continue
                crops.append(c)
                meta.append(dict(det=n, x1=l, y1=t, x2=r, y2=b,
                                 det_conf=round(conf, 3), det_class=det_name))
                names.append(f"{f.stem}__{n}.jpg")
        if not crops:
            continue
        proba = clf.predict_proba(extract(crops))
        for c, m, name, pr in zip(crops, meta, names, proba):
            k = int(pr.argmax())
            status = "ok" if pr[k] >= args.min_prob else "uncertain"
            row = {"image": str(f), **m, "subtype": classes[k],
                   "p_subtype": round(float(pr[k]), 3), "status": status}
            row.update({f"p_{cl}": round(float(p), 3)
                        for cl, p in zip(classes, pr)})
            rows.append(row)
            if args.save_crops:
                folder = out / "crops" / (classes[k] if status == "ok"
                                          else "uncertain")
                folder.mkdir(parents=True, exist_ok=True)
                c.save(folder / name, quality=95)

    if not rows:
        print("No objects to classify (no detections, or no crops).")
        return
    out_csv = out / "subtypes.csv"
    fields = ["image", "det", "det_class", "det_conf", "x1", "y1", "x2", "y2",
              "subtype", "p_subtype", "status"] + [f"p_{c}" for c in classes]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} object(s) classified -> {out_csv}")
    for c in classes:
        n = sum(r["subtype"] == c and r["status"] == "ok" for r in rows)
        print(f"  {c:<20}{n:>5}")
    n_unc = sum(r["status"] == "uncertain" for r in rows)
    print(f"  {'uncertain':<20}{n_unc:>5}   (top probability < {args.min_prob}; "
          f"check by eye)")
    if args.save_crops:
        print(f"Crops sorted into {out / 'crops'}")


if __name__ == "__main__":
    main()
