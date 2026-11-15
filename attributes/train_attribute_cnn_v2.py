"""
Fine-tune a small pretrained CNN on sorted crops (Stage B attribute classifier)
and measure it with cross-validation split BY VIDEO.

Meant for the exit-sign direction task, where the frozen-feature probe was
weak, but it works for any class folder laid out like:
    crops/exit_sign/
        direction left/  direction right/  direction up/  direction down/
        unknown/  unsorted/            <- skipped by default

Why this differs from the probe:
  * the whole network is fine-tuned, so it can learn arrow ORIENTATION;
  * NO random horizontal flip is used (that would turn left into right).
    Optionally --swap-flip flips the crop AND swaps the two named labels, which
    is a valid augmentation for left/right arrows -- test it, don't assume it;
  * input is rectangular (default 128 x 256, height x width) and crops are
    padded, not squashed, so a wide sign keeps its aspect ratio. For tall
    objects such as extinguishers use --img-h 256 --img-w 128;
  * the folds are built with the grouping code of probe_separability_v4.py
    (imported, so keep both scripts in one folder): same crops, same seed, same
    groups, so the numbers are directly comparable with the probe.

v2 changes (24 Oct 2026):
  * grouping now comes from probe_separability_v4.py. v1 made every 360-photo
    crop and every early-frame crop its own group, so crops cut from one photo
    or cube face, and clips name(1)/name(2) of one recording, could land on
    both sides of a split. v1 results are therefore somewhat optimistic.
  * --group-by location (via location_map.csv) and --folds 0 for
    leave-one-group-out, as in the probe script.
  * prints one settings line for the logbook, a per-location and per-source
    breakdown, and the location of each error in cnn_errors*.csv.

v2.1 changes (25 Oct 2026), all OFF by default so the v2 settings are
unchanged unless asked for:
  * viewpoint augmentation for small, steeply viewed signs (e.g. ceiling signs
    in 360 photos): --rot-deg, --perspective, --lowres-prob/--lowres-px and
    --crop-jitter, or --viewpoint-aug for the recommended preset
    (rotation +-25 deg, perspective 0.3, low-resolution 30-60 px with
    probability 0.3, crop jitter +-20%).
  * --tta-flip: average the prediction on the crop and on its mirror image
    (with the --swap-flip labels swapped back).
  * --repeats N: train every fold with N seeds (seed, seed+1, ...), report
    mean +- standard deviation over seeds and the result of the seed ensemble
    (averaged probabilities). --save-final then saves all N models.
  * every fold is re-seeded, so a fold's model depends only on its data and
    the seed, not on how many folds ran before it. GPU training can still
    differ slightly from run to run.
  * results are saved. Every run gets its own folder,
    <class_dir>/cnn_runs/<date-time>_<tag>/ (or --out-dir), holding:
        log.txt          everything printed on screen
        settings.json    all settings, data counts, groups, library versions
        metrics.json     overall, per seed, per class, confusion matrix,
                         per location and per source
        predictions.csv  every crop: true label, prediction, probabilities,
                         per-seed predictions, fold, location, source
        errors.csv       misclassified crops, most confidently wrong first
    and one line per run is appended to cnn_runs/summary.csv, so all runs can
    be compared in one table. --tag names the run, --note adds a remark.
  * --final-only --save-final PATH skips the cross-validation and only trains
    the final model(s) on all crops (one per seed), for the configuration that
    was already chosen. Takes minutes instead of an hour. The run folder then
    holds log.txt and settings.json.

The score is computed from the model's predictions on held-out videos after the
LAST epoch. Nothing is chosen using the held-out folds.

Requires: pip install torch torchvision scikit-learn pillow numpy
(the first run downloads the ImageNet weights)

Usage:
    python train_attribute_cnn_v2.py crops/exit_sign
    python train_attribute_cnn_v2.py crops/exit_sign --swap-flip "direction left" "direction right"
    python train_attribute_cnn_v2.py crops/exit_sign --arch resnet18
    python train_attribute_cnn_v2.py crops/exit_sign --arch resnet18 --img-h 192 --img-w 384 ^
        --swap-flip "direction left" "direction right" --group-by location --folds 0
    python train_attribute_cnn_v2.py crops/exit_sign --arch resnet18 --img-h 192 --img-w 384 ^
        --swap-flip "direction left" "direction right" --group-by location --folds 0 ^
        --viewpoint-aug --tta-flip --repeats 3
    python train_attribute_cnn_v2.py crops/exit_sign --save-final exit_sign_cnn.pt
"""
import argparse
import csv
import datetime
import json
import platform
import time
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
GREY = (124, 116, 104)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# --------------------------------------------------------------------------- #
# Data helpers (same conventions as probe_separability.py)
# --------------------------------------------------------------------------- #
try:
    from probe_separability_v4 import (find_location_map, group_of,
                                       load_location_map, location_groups,
                                       source_of)
except ImportError:
    sys.exit("train_attribute_cnn_v2.py needs probe_separability_v4.py in the "
             "same folder (it shares the grouping code).")

video_of = group_of


def collect(class_dir, exclude):
    items = []
    for sub in sorted(p for p in class_dir.iterdir() if p.is_dir()):
        if sub.name.lower() in exclude:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f, sub.name))
    return items


def letterbox_rect(img, width, height):
    """Scale to fit inside width x height keeping aspect ratio, pad with grey."""
    w, h = img.size
    s = min(width / w, height / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (width, height), GREY)
    canvas.paste(img, ((width - nw) // 2, (height - nh) // 2))
    return canvas


def to_array(img):
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((a - MEAN) / STD).astype(np.float32)


V1_AUG = dict(rot_deg=8.0, crop_jitter=0.08, perspective=0.0, lowres_prob=0.0)
VIEWPOINT_AUG = dict(rot_deg=25.0, crop_jitter=0.20, perspective=0.3,
                     lowres_prob=0.3)


def perspective_coeffs(out_pts, in_pts):
    """Coefficients for PIL's PERSPECTIVE transform that send each output
    point to the matching input point."""
    A, B = [], []
    for (x, y), (u, v) in zip(out_pts, in_pts):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def random_perspective(img, rng, scale):
    """Move each corner inward by up to `scale` x half the side (like
    torchvision's RandomPerspective). The whole sign stays in view and looks
    seen from a random oblique angle, e.g. from below."""
    w, h = img.size
    dx, dy = scale * w / 2, scale * h / 2
    corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    signs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    moved = [(x + sx * rng.uniform(0, dx), y + sy * rng.uniform(0, dy))
             for (x, y), (sx, sy) in zip(corners, signs)]
    return img.transform((w, h), Image.PERSPECTIVE,
                         perspective_coeffs(moved, corners),
                         resample=Image.BICUBIC, fillcolor=GREY)


def low_resolution(img, rng, lo, hi):
    """Shrink so the longer side is lo..hi px, then scale back up: a distant
    sign whose arrow is only a few pixels wide."""
    w, h = img.size
    s = rng.uniform(lo, hi) / max(w, h)
    if s >= 1:
        return img
    small = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


def augment(img, rng, aug):
    """Geometric + photometric jitter. No hue change (label colour can be a
    cue) and no flipping (handled separately, with label swapping). Rotation
    stays far below 45 deg, so an arrow never turns into another direction.
    With the v1 settings this draws exactly the same random numbers as v1."""
    if aug["perspective"] > 0:
        img = random_perspective(img, rng, aug["perspective"])
    rot = aug["rot_deg"]
    img = img.rotate(rng.uniform(-rot, rot), resample=Image.BICUBIC,
                     expand=rot > 8, fillcolor=GREY)
    # random crop / pad jitter per side: tighter or looser boxes
    w, h = img.size
    j = aug["crop_jitter"]
    pad = max(2, int((j + 0.04) * max(w, h)))
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), GREY)
    canvas.paste(img, (pad, pad))
    l, t, r, b = (int(rng.uniform(-j, j) * s) for s in (w, h, w, h))
    box = (pad + l, pad + t, pad + w + r, pad + h + b)
    if box[2] - box[0] > 4 and box[3] - box[1] > 4:
        img = canvas.crop(box)
    if aug["lowres_prob"] > 0 and rng.random() < aug["lowres_prob"]:
        img = low_resolution(img, rng, *aug["lowres_px"])
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    return img


class CropDataset:
    """Map-style dataset (no torch import needed here; DataLoader accepts any
    object with __len__ and __getitem__)."""

    def __init__(self, images, labels, width, height, train, swap_map, seed,
                 aug=None):
        self.images, self.labels = images, labels
        self.width, self.height = width, height
        self.train, self.swap_map = train, swap_map
        self.aug = aug
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, y = self.images[i], int(self.labels[i])
        if self.train:
            img = augment(img, self.rng, self.aug)
            if self.swap_map is not None and self.rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                y = self.swap_map.get(y, y)
        return to_array(letterbox_rect(img, self.width, self.height)), y


# --------------------------------------------------------------------------- #
# Model + training
# --------------------------------------------------------------------------- #
def build_model(arch, n_classes, pretrained):
    import torch.nn as nn
    from torchvision import models

    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        head = list(m.classifier.parameters())
    elif arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        head = list(m.fc.parameters())
    else:
        sys.exit(f"Unknown --arch {arch}")
    head_ids = {id(p) for p in head}
    backbone = [p for p in m.parameters() if id(p) not in head_ids]
    return m, backbone, head


def flip_permutation(n_classes, swap_map):
    """Column order that turns predictions on a mirrored crop back into
    predictions for the original crop (left and right swapped)."""
    return [swap_map.get(i, i) if swap_map else i for i in range(n_classes)]


def predict_proba(model, dataset, device, batch_size, flip_perm=None):
    """Softmax probabilities. With flip_perm (from flip_permutation), also
    predict the mirrored crop and average the two (flip TTA)."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)
            if flip_perm is not None:
                pf = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
                p = (p + pf[:, flip_perm]) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(train_imgs, train_y, n_classes, args, device, swap_map,
                eval_imgs=None, eval_y=None, tag="", seed=None, flip_perm=None):
    """Train from the pretrained weights. If eval data is given, the accuracy on
    it is printed every few epochs for diagnosis ONLY (never used to pick
    anything)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    model, backbone, head = build_model(args.arch, n_classes, not args.no_pretrained)
    model = model.to(device)

    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr / 5}, {"params": head, "lr": args.lr}],
        weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    seed = args.seed if seed is None else seed
    train_ds = CropDataset(train_imgs, train_y, args.img_w, args.img_h, True,
                           swap_map, seed, args.aug)
    bs = min(args.batch_size, len(train_ds))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                        num_workers=args.workers, drop_last=len(train_ds) > bs)
    eval_ds = None
    if eval_imgs is not None:
        eval_ds = CropDataset(eval_imgs, eval_y, args.img_w, args.img_h, False,
                              None, seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        if eval_ds is not None and (epoch % 5 == 0 or epoch == args.epochs):
            p = predict_proba(model, eval_ds, device, args.batch_size, flip_perm)
            acc = float((p.argmax(1) == np.asarray(eval_y)).mean())
            print(f"    {tag} epoch {epoch:>3}/{args.epochs}  "
                  f"train loss {total / len(train_ds):.3f}  held-out acc {acc:.3f}")
    return model


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(y_true, y_pred, classes, unit="video", split_desc=""):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix,
                                 f1_score)
    counts = np.bincount([classes.index(c) for c in y_true])
    print("\n" + "=" * 60)
    print(f"{len(y_true)} crops, {len(classes)} classes -- fine-tuned CNN, "
          f"out-of-fold predictions, {split_desc}")
    print("=" * 60)
    print(f"{'Accuracy (by ' + unit + ')':<25}: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Balanced accuracy        : {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro F1                 : {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Majority-class baseline  : {counts.max() / len(y_true):.3f}")
    print("\nPer-class results:")
    print(classification_report(y_true, y_pred, labels=classes, digits=3,
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    w = max(len(c) for c in classes) + 2
    print("Confusion matrix (rows = true label, columns = predicted):")
    print(" " * w + "".join(f"{c[-9:]:>10}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:<{w}}" + "".join(f"{v:>10}" for v in row))


# --------------------------------------------------------------------------- #
# Saving results
# --------------------------------------------------------------------------- #
class Tee:
    """Write to the screen and to a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def metric_dict(y_true, y_pred, classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix,
                                 f1_score)
    cr = classification_report(y_true, y_pred, labels=classes,
                               output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(
            balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": {c: {"precision": round(cr[c]["precision"], 4),
                          "recall": round(cr[c]["recall"], 4),
                          "f1": round(cr[c]["f1-score"], 4),
                          "support": int(cr[c]["support"])} for c in classes},
        "confusion_matrix": {"rows_true_cols_predicted": classes,
                             "matrix": confusion_matrix(
                                 y_true, y_pred, labels=classes).tolist()},
    }


def auto_tag(args):
    split = "lolo" if args.folds == 0 else f"{args.folds}fold"
    aug = "" if args.aug == {**V1_AUG, "lowres_px": args.aug["lowres_px"]} \
        else ("_vp" if args.viewpoint_aug else "_aug")
    return (f"{args.arch}_{args.img_h}x{args.img_w}_{split}-{args.group_by}"
            f"{aug}{'_tta' if args.tta_flip else ''}_x{args.repeats}")


SUMMARY_FIELDS = [
    "run", "date", "class_folder", "n_crops", "classes", "split", "n_groups",
    "arch", "input_hxw", "epochs", "augmentation", "tta_flip", "seeds",
    "accuracy", "balanced_accuracy", "macro_f1",
    "seed_accuracy_mean", "seed_accuracy_sd", "seed_balanced_mean",
    "seed_balanced_sd", "seed_f1_mean", "seed_f1_sd",
    "acc_own_video", "acc_own_360_photo", "worst_location",
    "worst_location_acc", "n_errors", "minutes", "note", "run_folder"]


def save_final_models(path, args, images, y, classes, device, swap_map,
                      seeds):
    """Train one model per seed on ALL crops and save them in one file,
    together with everything inference needs."""
    import torch
    state_dicts = []
    for r, sd in enumerate(seeds, 1):
        print(f"\nTraining final model on ALL crops (seed {sd}, "
              f"{r}/{len(seeds)})...")
        seed_everything(sd)
        model = train_model(images, y, len(classes), args, device,
                            swap_map, tag="final", seed=sd)
        state_dicts.append({k_: v.cpu() for k_, v in
                            model.state_dict().items()})
    meta = {"classes": classes, "arch": args.arch, "img_h": args.img_h,
            "img_w": args.img_w, "preprocess": "letterbox grey, ImageNet "
            "normalisation", "tta_flip": bool(args.tta_flip),
            "swap_flip": list(args.swap_flip) if args.swap_flip else None,
            "seeds": seeds, "aug": {k_: list(v) if isinstance(v, tuple)
                                    else v for k_, v in args.aug.items()},
            "n_crops": int(len(y)),
            "crops_per_class": {c: int((np.asarray(y) == i).sum())
                                for i, c in enumerate(classes)},
            "epochs": args.epochs, "note": args.note,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dicts[0], "state_dicts": state_dicts,
                **meta}, path)
    with open(str(path) + ".classes.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(state_dicts)} model(s) to {path}")
    return meta


def append_summary(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("class_dir", help="e.g. crops/exit_sign")
    ap.add_argument("--arch", default="efficientnet_b0",
                    choices=["efficientnet_b0", "resnet18"])
    ap.add_argument("--img-h", type=int, default=128, help="input height")
    ap.add_argument("--img-w", type=int, default=256, help="input width")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head learning rate; the pretrained body uses lr/5")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--swap-flip", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="flip horizontally 50%% of the time and swap these two "
                         "labels, e.g. \"direction left\" \"direction right\"")
    ap.add_argument("--exclude", nargs="*", default=["unknown", "unsorted"])
    ap.add_argument("--min-per-class", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5,
                    help="Number of grouped folds; 0 = leave-one-group-out")
    ap.add_argument("--group-by", choices=["video", "location"], default="video",
                    help="Unit kept on one side of each split (see "
                         "probe_separability_v4.py)")
    ap.add_argument("--location-map", default=None,
                    help="Path to location_map.csv (default: look next to the "
                         "crops and in parent folders)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (keep 0 on Windows unless you know "
                         "you need more)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init instead of ImageNet weights (for testing)")
    ap.add_argument("--save-final", metavar="PATH",
                    help="after evaluation, train on ALL crops and save the "
                         "model(s) here (plus PATH.classes.json); with "
                         "--repeats N, all N seed models are saved")
    g = ap.add_argument_group("viewpoint augmentation (v2.1; off by default)")
    g.add_argument("--viewpoint-aug", action="store_true",
                   help="preset for small, steeply viewed signs: --rot-deg 25 "
                        "--perspective 0.3 --lowres-prob 0.3 --crop-jitter "
                        "0.2. Individual flags given as well override it")
    g.add_argument("--rot-deg", type=float, default=None,
                   help="max rotation in degrees (v1: 8)")
    g.add_argument("--perspective", type=float, default=None,
                   help="perspective distortion 0..0.5, corner shift as a "
                        "fraction of half the side (v1: 0 = off)")
    g.add_argument("--lowres-prob", type=float, default=None,
                   help="probability of shrinking a crop to --lowres-px and "
                        "back (v1: 0 = off)")
    g.add_argument("--lowres-px", type=float, nargs=2, default=[30, 60],
                   metavar=("MIN", "MAX"),
                   help="longer side after shrinking, in px (default 30 60)")
    g.add_argument("--crop-jitter", type=float, default=None,
                   help="max box change per side as a fraction, tighter or "
                        "looser (v1: 0.08)")
    g2 = ap.add_argument_group("evaluation (v2.1)")
    g2.add_argument("--tta-flip", action="store_true",
                    help="also predict the mirrored crop and average. Needs "
                         "--swap-flip for left/right labels; without it, only "
                         "use this when a mirror image keeps its label")
    g2.add_argument("--repeats", type=int, default=1,
                    help="train every fold with this many seeds and report "
                         "mean +- std and the seed ensemble (default 1)")
    g3 = ap.add_argument_group("saving results")
    g3.add_argument("--out-dir", default=None,
                    help="where run folders and summary.csv go (default: "
                         "<class_dir>/cnn_runs)")
    g3.add_argument("--tag", default=None,
                    help="short run name, e.g. baseline or viewpoint (default: "
                         "built from the settings)")
    g3.add_argument("--note", default="",
                    help="free-text remark stored with the run")
    g3.add_argument("--final-only", action="store_true",
                    help="skip cross-validation; only train and save the final "
                         "model(s) on all crops (needs --save-final)")
    args = ap.parse_args()
    if args.final_only and not args.save_final:
        sys.exit("--final-only needs --save-final PATH")

    base = VIEWPOINT_AUG if args.viewpoint_aug else V1_AUG
    args.aug = {k: (getattr(args, k) if getattr(args, k) is not None else v)
                for k, v in base.items()}
    args.aug["lowres_px"] = tuple(args.lowres_px)
    if not 0 <= args.aug["rot_deg"] < 45:
        sys.exit("--rot-deg must be below 45, or arrows change direction.")
    if args.repeats < 1:
        sys.exit("--repeats must be at least 1.")

    class_dir = Path(args.class_dir)
    if not class_dir.is_dir():
        sys.exit(f"{class_dir} is not a folder.")
    exclude = {e.lower() for e in args.exclude}

    out_dir = Path(args.out_dir) if args.out_dir else class_dir / "cnn_runs"
    if out_dir.resolve().parent == class_dir.resolve():
        exclude.add(out_dir.name.lower())   # never read as a class folder
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", args.tag or auto_tag(args))
    started = datetime.datetime.now()
    run_dir = out_dir / f"{started:%Y%m%d-%H%M%S}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_fh = open(run_dir / "log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    print(f"Run folder: {run_dir}")
    print("Command: " + " ".join(f'"{a}"' if " " in a else a
                                 for a in sys.argv))
    print(f"Settings: arch={args.arch} (ImageNet "
          f"{'off' if args.no_pretrained else 'on'}), input {args.img_h}x"
          f"{args.img_w} HxW letterbox, epochs={args.epochs}, batch="
          f"{args.batch_size}, AdamW lr={args.lr} head / {args.lr / 5:g} body, "
          f"wd=0.01, cosine, label_smoothing={args.label_smoothing}, "
          f"class-weighted CE, aug=rot+-{args.aug['rot_deg']:g} "
          f"crop+-{args.aug['crop_jitter'] * 100:g}% "
          f"perspective={args.aug['perspective']:g} "
          f"lowres={args.aug['lowres_prob']:g}@{args.aug['lowres_px'][0]:g}-"
          f"{args.aug['lowres_px'][1]:g}px bright/contrast 0.7-1.3 "
          f"sat 0.8-1.2, swap_flip={args.swap_flip}, tta_flip={args.tta_flip}, "
          f"folds={args.folds if args.folds else 'leave-one-out'} grouped by "
          f"{args.group_by}, seeds={args.seed}"
          f"{'..' + str(args.seed + args.repeats - 1) if args.repeats > 1 else ''}"
          f" (re-seeded per fold), last-epoch predictions")
    items = collect(class_dir, exclude)
    counts = {}
    for _, lab in items:
        counts[lab] = counts.get(lab, 0) + 1
    print("Crops per attribute value:", counts)
    small = {k for k, v in counts.items() if v < args.min_per_class}
    if small:
        print(f"Dropping (fewer than {args.min_per_class} crops): {sorted(small)}")
        items = [(p, lab) for p, lab in items if lab not in small]
    classes = sorted({lab for _, lab in items})
    if len(classes) < 2:
        sys.exit("Need at least 2 attribute values with enough crops.")

    swap_map = None
    if args.swap_flip:
        a, b = args.swap_flip
        if a not in classes or b not in classes:
            sys.exit(f"--swap-flip labels must be among {classes}")
        swap_map = {classes.index(a): classes.index(b),
                    classes.index(b): classes.index(a)}
        print(f"Flip augmentation ON, swapping '{a}' <-> '{b}'")
    else:
        print("Flip augmentation OFF")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   Model: {args.arch}   "
          f"Input: {args.img_h}x{args.img_w} (HxW)   Epochs: {args.epochs}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)
    flip_perm = (flip_permutation(len(classes), swap_map) if args.tta_flip
                 else None)
    if args.tta_flip:
        print(f"Flip TTA ON{'' if swap_map else ' (no labels swapped)'}")

    images = []
    for p, _ in items:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
    y_str = np.array([lab for _, lab in items])
    y = np.array([classes.index(c) for c in y_str])
    locs = None
    if args.group_by == "location":
        mapping = load_location_map(find_location_map(class_dir,
                                                      args.location_map))
        groups, locs = location_groups(items, mapping)
    else:
        groups = np.array([video_of(p.name) for p, _ in items])

    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
    n_groups = len(set(groups))
    if n_groups < 2:
        sys.exit(f"Need crops from at least 2 different {args.group_by}s.")
    if args.folds == 0:
        k = n_groups
        cv = LeaveOneGroupOut()
        split_desc = f"leave-one-{args.group_by}-out ({k} folds)"
    else:
        k = min(args.folds, n_groups)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                  random_state=args.seed)
        split_desc = f"{k}-fold split by {args.group_by}"
    print(f"{n_groups} {args.group_by} groups, {split_desc}")

    if args.final_only:
        seeds = [args.seed + r for r in range(args.repeats)]
        print("\n--final-only: no cross-validation; training the final "
              "model(s) on all crops")
        meta = save_final_models(args.save_final, args, images, y, classes,
                                 device, swap_map, seeds)
        import sklearn
        import torchvision
        settings = {
            "run": run_dir.name, "mode": "final-only",
            "started": started.isoformat(timespec="seconds"),
            "command": sys.argv, "class_dir": str(class_dir.resolve()),
            "args": {k_: (list(v) if isinstance(v, tuple) else v)
                     for k_, v in vars(args).items() if k_ != "aug"},
            "model_file": str(Path(args.save_final).resolve()),
            "model_meta": meta,
            "versions": {"python": platform.python_version(),
                         "torch": torch.__version__,
                         "torchvision": torchvision.__version__,
                         "scikit_learn": sklearn.__version__,
                         "device": torch.cuda.get_device_name(0)
                         if device == "cuda" else "cpu"}}
        with open(run_dir / "settings.json", "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
        minutes = round((time.time() - t_start) / 60, 1)
        print(f"\nDone in {minutes} min. Settings in {run_dir}")
        sys.stdout.flush()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_fh.close()
        return

    # the same folds for every seed; only the training randomness changes
    splits = list(cv.split(np.zeros(len(y)), y_str, groups))
    fold_of = np.zeros(len(y), dtype=int)
    for fold, (_, te) in enumerate(splits, 1):
        fold_of[te] = fold
    seeds = [args.seed + r for r in range(args.repeats)]
    all_oof = []
    for r, sd in enumerate(seeds, 1):
        oof = np.zeros((len(items), len(classes)), dtype=np.float32)
        for fold, (tr, te) in enumerate(splits, 1):
            rep_txt = f"seed {sd} ({r}/{len(seeds)}), " if len(seeds) > 1 else ""
            print(f"\n{rep_txt}Fold {fold}/{k}: train {len(tr)} crops, "
                  f"test {len(te)} crops")
            seed_everything(sd)
            model = train_model([images[i] for i in tr], y[tr], len(classes),
                                args, device, swap_map,
                                [images[i] for i in te], y[te],
                                tag=f"fold {fold}", seed=sd,
                                flip_perm=flip_perm)
            test_ds = CropDataset([images[i] for i in te], y[te], args.img_w,
                                  args.img_h, False, None, sd)
            oof[te] = predict_proba(model, test_ds, device, args.batch_size,
                                    flip_perm)
        all_oof.append(oof)

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score)
    seed_preds = [np.array(classes)[o.argmax(axis=1)] for o in all_oof]
    seed_summary = None
    if len(seeds) > 1:
        print("\n" + "=" * 60)
        print(f"Single models, one per seed ({split_desc})")
        print("=" * 60)
        stats = []
        for sd, pr in zip(seeds, seed_preds):
            s3 = (accuracy_score(y_str, pr), balanced_accuracy_score(y_str, pr),
                  f1_score(y_str, pr, average="macro"))
            stats.append(s3)
            print(f"  seed {sd:<4} accuracy {s3[0]:.3f}  balanced {s3[1]:.3f}  "
                  f"macro F1 {s3[2]:.3f}")
        st = np.array(stats)
        mean, sd_ = st.mean(axis=0), st.std(axis=0, ddof=1)
        seed_summary = {"mean": {"accuracy": round(float(mean[0]), 4),
                                 "balanced_accuracy": round(float(mean[1]), 4),
                                 "macro_f1": round(float(mean[2]), 4)},
                        "sd": {"accuracy": round(float(sd_[0]), 4),
                               "balanced_accuracy": round(float(sd_[1]), 4),
                               "macro_f1": round(float(sd_[2]), 4)}}
        print(f"  mean +- sd  accuracy {mean[0]:.3f} +- {sd_[0]:.3f}  balanced "
              f"{mean[1]:.3f} +- {sd_[1]:.3f}  macro F1 {mean[2]:.3f} +- "
              f"{sd_[2]:.3f}")
        print("The report below is the ENSEMBLE of these seeds (probabilities "
              "averaged).")

    oof = np.mean(all_oof, axis=0)
    pred = np.array(classes)[oof.argmax(axis=1)]
    report(y_str, pred, classes, args.group_by,
           split_desc + (f", ensemble of {len(seeds)} seeds"
                         if len(seeds) > 1 else ""))

    def breakdown(title, keys, order=None):
        """Print the breakdown and return it as a dict for metrics.json."""
        print(f"\n{title}")
        out = {}
        for key in (order or sorted(set(keys))):
            m = keys == key
            if not m.any():
                continue
            entry = {"n_crops": int(m.sum()),
                     "accuracy": round(float((pred[m] == y_str[m]).mean()), 4),
                     "per_class_correct": {
                         c: [int(((pred[m] == c) & (y_str[m] == c)).sum()),
                             int((y_str[m] == c).sum())]
                         for c in classes if (y_str[m] == c).any()}}
            correct = int((pred[m] == y_str[m]).sum())
            per_class = ", ".join(
                f"{c}: {int(((pred[m] == c) & (y_str[m] == c)).sum())}/"
                f"{int((y_str[m] == c).sum())}"
                for c in classes if (y_str[m] == c).any())
            spread = ""
            if len(seeds) > 1:
                accs = [float((sp[m] == y_str[m]).mean()) for sp in seed_preds]
                spread = (f"  [single seeds {np.mean(accs):.3f} +- "
                          f"{np.std(accs, ddof=1):.3f}]")
                entry["seed_accuracy_mean"] = round(float(np.mean(accs)), 4)
                entry["seed_accuracy_sd"] = round(float(np.std(accs, ddof=1)), 4)
            print(f"  {key[:34]:<35}{int(m.sum()):>4} crops  accuracy "
                  f"{correct / m.sum():.3f}{spread}  ({per_class})")
            out[key] = entry
        return out

    by_location = by_source = {}
    if locs is not None:
        by_location = breakdown(
            "Breakdown by location (each one held out as a whole):", locs)
    srcs = np.array([source_of(p.name) for p, _ in items])
    if len(set(srcs)) > 1:
        by_source = breakdown("Breakdown by source (guessed from file names):",
                              srcs, ["own video", "own 360 photo",
                                     "own early frames", "other"])

    # ---- save results ----------------------------------------------------
    import sklearn
    import torchvision
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    settings = {
        "run": run_dir.name, "started": started.isoformat(timespec="seconds"),
        "command": sys.argv, "class_dir": str(class_dir.resolve()),
        "args": {k_: (list(v) if isinstance(v, tuple) else v)
                 for k_, v in vars(args).items() if k_ != "aug"},
        "augmentation": {k_: (list(v) if isinstance(v, tuple) else v)
                         for k_, v in args.aug.items()},
        "fixed": {"optimizer": "AdamW", "lr_head": args.lr,
                  "lr_body": args.lr / 5, "weight_decay": 0.01,
                  "schedule": "cosine", "loss": "class-weighted cross-entropy",
                  "photometric": "brightness/contrast 0.7-1.3, saturation "
                                 "0.8-1.2", "preprocess": "letterbox grey, "
                                 "ImageNet normalisation",
                  "prediction": "last epoch"},
        "data": {"n_crops": int(len(y)),
                 "crops_per_class": {c: int((y_str == c).sum())
                                     for c in classes},
                 "dropped_classes": sorted(small),
                 "split": split_desc, "n_groups": int(n_groups),
                 "groups": {str(gk): int((groups == gk).sum())
                            for gk in sorted(set(groups))}},
        "seeds": seeds,
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "torchvision": torchvision.__version__,
                     "scikit_learn": sklearn.__version__, "device": gpu},
        "note": args.note,
    }
    with open(run_dir / "settings.json", "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, ensure_ascii=False)

    ens_metrics = metric_dict(y_str, pred, classes)
    metrics = {
        "split": split_desc,
        "result": ens_metrics,
        "result_is": (f"ensemble of {len(seeds)} seeds" if len(seeds) > 1
                      else "single model"),
        "majority_baseline": round(float(
            max((y_str == c).mean() for c in classes)), 4),
        "per_seed": {str(sd): {k_: v for k_, v in
                               metric_dict(y_str, sp, classes).items()
                               if k_ in ("accuracy", "balanced_accuracy",
                                         "macro_f1")}
                     for sd, sp in zip(seeds, seed_preds)},
        "seed_mean_sd": seed_summary,
        "by_location": by_location,
        "by_source": by_source,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    with open(run_dir / "predictions.csv", "w", newline="",
              encoding="utf-8") as fh:
        fields = (["crop", "true", "predicted", "correct", "p_predicted"]
                  + [f"p_{c}" for c in classes]
                  + [f"pred_seed{sd}" for sd in seeds]
                  + ["fold", "group", "location", "source"])
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, (p, lab) in enumerate(items):
            row = {"crop": str(p.relative_to(class_dir)), "true": lab,
                   "predicted": pred[i], "correct": int(pred[i] == lab),
                   "p_predicted": round(float(oof[i].max()), 4),
                   "fold": int(fold_of[i]), "group": str(groups[i]),
                   "location": locs[i] if locs is not None else "",
                   "source": srcs[i]}
            row.update({f"p_{c}": round(float(oof[i, j]), 4)
                        for j, c in enumerate(classes)})
            row.update({f"pred_seed{sd}": sp[i]
                        for sd, sp in zip(seeds, seed_preds)})
            w.writerow(row)

    out_csv = run_dir / "errors.csv"
    rows = []
    for i, ((p, lab), pr, pd_) in enumerate(zip(items, oof, pred)):
        if pd_ != lab:
            rows.append({"crop": str(p.relative_to(class_dir)), "true": lab,
                         "predicted": pd_,
                         "p_true": round(float(pr[classes.index(lab)]), 3),
                         "p_predicted": round(float(pr[classes.index(pd_)]), 3),
                         "video": video_of(p.name),
                         "location": locs[i] if locs is not None else ""})
    rows.sort(key=lambda r: -r["p_predicted"])
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "true", "predicted", "p_true",
                                           "p_predicted", "video", "location"])
        w.writeheader()
        w.writerows(rows)
    # also keep a copy of the latest errors next to the crops, as before
    latest = class_dir / ("cnn_errors.csv" if args.group_by == "video"
                          else "cnn_errors_by_location.csv")
    latest.write_bytes(out_csv.read_bytes())
    print(f"\n{len(rows)} misclassified crops listed in {out_csv}")

    big_locs = {k_: v for k_, v in by_location.items()
                if v["n_crops"] >= 10 and not k_.startswith("(unmapped)")}
    worst = min(big_locs.items(), key=lambda kv: kv[1]["accuracy"]) \
        if big_locs else ("", {"accuracy": ""})
    summary_row = {
        "run": run_dir.name, "date": f"{started:%Y-%m-%d %H:%M}",
        "class_folder": class_dir.name, "n_crops": len(y),
        "classes": "; ".join(f"{c} {int((y_str == c).sum())}" for c in classes),
        "split": split_desc, "n_groups": n_groups, "arch": args.arch,
        "input_hxw": f"{args.img_h}x{args.img_w}", "epochs": args.epochs,
        "augmentation": (f"rot{args.aug['rot_deg']:g} "
                         f"crop{args.aug['crop_jitter']:g} "
                         f"persp{args.aug['perspective']:g} "
                         f"lowres{args.aug['lowres_prob']:g}"),
        "tta_flip": int(args.tta_flip),
        "seeds": " ".join(str(sd) for sd in seeds),
        "accuracy": ens_metrics["accuracy"],
        "balanced_accuracy": ens_metrics["balanced_accuracy"],
        "macro_f1": ens_metrics["macro_f1"],
        "acc_own_video": by_source.get("own video", {}).get("accuracy", ""),
        "acc_own_360_photo": by_source.get("own 360 photo", {}).get(
            "accuracy", ""),
        "worst_location": worst[0],
        "worst_location_acc": worst[1]["accuracy"],
        "n_errors": len(rows), "note": args.note, "run_folder": str(run_dir),
    }
    if seed_summary:
        summary_row.update({
            "seed_accuracy_mean": seed_summary["mean"]["accuracy"],
            "seed_accuracy_sd": seed_summary["sd"]["accuracy"],
            "seed_balanced_mean": seed_summary["mean"]["balanced_accuracy"],
            "seed_balanced_sd": seed_summary["sd"]["balanced_accuracy"],
            "seed_f1_mean": seed_summary["mean"]["macro_f1"],
            "seed_f1_sd": seed_summary["sd"]["macro_f1"]})

    if args.save_final:
        save_final_models(args.save_final, args, images, y, classes, device,
                          swap_map, seeds)

    minutes = round((time.time() - t_start) / 60, 1)
    summary_row["minutes"] = minutes
    append_summary(out_dir / "summary.csv", summary_row)
    print(f"\nResults saved in {run_dir} (took {minutes} min); one line "
          f"added to {out_dir / 'summary.csv'}")
    sys.stdout.flush()
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    log_fh.close()


if __name__ == "__main__":
    main()
