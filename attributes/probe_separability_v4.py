"""
How separable are my attribute classes? -- a linear probe on frozen features.

Takes ONE class folder of sorted crops, for example
    crops/fire_extinguisher/
        co2/  foam/  powder/  unknown/  unsorted/
(each subfolder name is an attribute value), turns every crop into a feature
vector with a frozen, pretrained DINOv2 or CLIP image model, then trains a
plain logistic regression on those vectors and measures how well it predicts
the attribute on videos it has NOT seen.

How to read the result:
  * High grouped accuracy, strong per-class recall   -> the classes are visually
    separable; a fine-tuned classifier should do at least as well.
  * Low accuracy, or one class always confused with another -> look at the
    confusion matrix and at probe_errors.csv, then at those crops. Often it is
    a labelling inconsistency, not a model problem.
  * Grouped accuracy much lower than the "random split" line -> that gap is the
    near-duplicate leakage you avoid by splitting by video.

Folders named "unknown" and "unsorted" are skipped by default.

Features are cached next to the crops (keyed by file name, so moving a crop
between folders does not trigger re-extraction); re-running after you sort
more crops only processes the new ones.

Requires: pip install torch transformers scikit-learn matplotlib pillow numpy

Usage:
    python probe_separability_v4.py crops/fire_extinguisher --model dinov2-small
    python probe_separability_v4.py crops/exit_sign --model dinov2-base --img-size 448
    python probe_separability_v4.py crops/fire_extinguisher --model clip-b16 --plot
    python probe_separability_v4.py crops/fire_extinguisher --img-size 448 --group-by location
    python probe_separability_v4.py crops/fire_extinguisher --img-size 448 --group-by location --folds 0

--group-by location keeps whole buildings out of training: dates in the crop
names (VID_20251219_..., IMG_20191223_..., 20240118_...) are looked up in
location_map.csv (columns date,location). Crops without a date, or whose date is
not in the map, stay in their video/photo group and are listed.

--save-probe PATH fits the probe on all crops after the evaluation and saves it
for inference with classify_subtypes.py.

--folds 0 holds out one group at a time (leave-one-location-out with
--group-by location): deterministic, independent of seed and file names.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

MODEL_ALIASES = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "clip-b32": "openai/clip-vit-base-patch32",
    "clip-b16": "openai/clip-vit-base-patch16",
}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_NORM = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
FACE_TAIL = re.compile(r"_(horizontal|top|bottom)_\d{2}$")


def source_of(filename):
    """Guess where a crop came from, from its file name:
         own video         <video>_frame_000012_<face>_NN__n.jpg
         own early frames  frame_000014_bottom_02__0.jpg   (no video prefix)
         own 360 photo     20240118_124305__0.jpg          (date_time stem)
                           IMG_20191223_093150_00_063_top_01__0.jpg
         other             anything else (e.g. Fire-ART)"""
    stem = Path(filename).stem
    if "_frame_" in stem:
        return "own video"
    if re.match(r"frame_\d+", stem):
        return "own early frames"
    if re.match(r"(IMG_)?\d{8}_\d{6}", stem):
        return "own 360 photo"
    return "other"


SPLIT_CLIP = re.compile(r"\(\d+\)$")


def group_of(filename):
    """The unit that must stay on one side of a train/test split, because its
    crops look alike: a whole video; all early frames together (their video is
    unknown); otherwise one source photo (all crops cut from it).

    Clips cut from one long recording and exported separately, e.g.
    VID_..._053(1) and VID_..._053(2), are the same video and form one group."""
    stem = Path(filename).stem
    src = source_of(filename)
    if src == "own video":
        return SPLIT_CLIP.sub("", stem.split("_frame_")[0])
    if src == "own early frames":
        return "early_frames"
    return FACE_TAIL.sub("", re.sub(r"__\d+$", "", stem))


video_of = group_of  # older name, kept so other scripts keep working


# --------------------------------------------------------------------------- #
# Location grouping (via location_map.csv with columns date,location)
# --------------------------------------------------------------------------- #
DATE_IN_NAME = re.compile(r"(?:^|_)((?:19|20)\d{6})_\d{6}")


def date_of(filename):
    """Recording date (YYYYMMDD) from VID_20251219_094741_..., IMG_20191223_...
    or 20240118_124305__0; None when the name has no date (early frames,
    Fire-ART)."""
    m = DATE_IN_NAME.search(Path(filename).stem)
    return m.group(1) if m else None


def normalise_date(text):
    """Accepts 20251219, 2025-12-19, 2025/12/19, 19-12-2025, 19.12.2025.
    Returns YYYYMMDD or None."""
    parts = [p for p in re.split(r"\D+", text.strip()) if p]
    if len(parts) == 1 and len(parts[0]) == 8:
        return parts[0]
    if len(parts) == 3:
        if len(parts[0]) == 4:          # year first
            y, m, d = parts
        elif len(parts[2]) == 4:        # day first (European order)
            d, m, y = parts
        else:
            return None
        return f"{int(y):04d}{int(m):02d}{int(d):02d}"
    return None


def find_location_map(class_dir, given):
    if given:
        p = Path(given)
        if not p.is_file():
            sys.exit(f"Location map not found: {p}")
        return p
    for folder in [class_dir, *class_dir.resolve().parents]:
        p = folder / "location_map.csv"
        if p.is_file():
            return p
    sys.exit("--group-by location needs location_map.csv. Put it next to the "
             "crops (or in a parent folder), or pass --location-map PATH.")


def load_location_map(path):
    """date -> location. Handles comma, semicolon or tab separators (Excel in
    Dutch locale saves with ';') and a UTF-8 byte-order mark."""
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""
    delim = max([",", ";", "\t"], key=first.count)
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
    if "date" not in cols or "location" not in cols:
        sys.exit(f"{path} needs columns 'date' and 'location' "
                 f"(found {reader.fieldnames}).")
    mapping, bad = {}, []
    for row in reader:
        raw = (row[cols["date"]] or "").strip()
        loc = (row[cols["location"]] or "").strip()
        if not raw and not loc:
            continue
        d = normalise_date(raw)
        if d is None or not loc:
            bad.append(raw)
            continue
        if d in mapping and mapping[d] != loc:
            print(f"Warning: date {d} is mapped to both '{mapping[d]}' and "
                  f"'{loc}' -- using '{mapping[d]}'.")
            continue
        mapping[d] = loc
    if bad:
        print(f"Warning: skipped {len(bad)} row(s) with an unreadable date: "
              f"{bad[:5]}")
    print(f"Location map: {path} ({len(mapping)} dates, "
          f"{len(set(mapping.values()))} locations)")
    return mapping


def location_groups(items, mapping):
    """Group = location for every crop whose date is in the map. Crops without
    a date, or with a date missing from the map, keep their video/photo group
    and are reported, so nothing is silently merged or dropped."""
    groups, locs, unmapped = [], [], {}
    for p, _ in items:
        d = date_of(p.name)
        loc = mapping.get(d) if d else None
        if loc:
            groups.append(f"loc:{loc}")
            locs.append(loc)
        else:
            key = d or source_of(p.name)
            unmapped[key] = unmapped.get(key, 0) + 1
            groups.append(group_of(p.name))
            locs.append("(unmapped) " + source_of(p.name))
    if unmapped:
        print("Not mapped to a location (kept in their video/photo group):")
        for k, n in sorted(unmapped.items()):
            what = "date not in map" if re.fullmatch(r"\d{8}", k) else "no date in name"
            print(f"  {k:<18}{n:>4} crop(s)  ({what})")
    return np.array(groups), np.array(locs)


def collect(class_dir, exclude):
    """Returns list of (path, attribute_label)."""
    items = []
    for sub in sorted(p for p in class_dir.iterdir() if p.is_dir()):
        if sub.name.lower() in exclude:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f, sub.name))
    return items


# --------------------------------------------------------------------------- #
# Feature extraction (frozen model)
# --------------------------------------------------------------------------- #
def letterbox(img, size):
    """Pad to a square (grey) instead of centre-cropping, so wide exit signs
    and tall extinguishers are never cut off."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def load_extractor(model_name, img_size):
    try:
        import torch
        from transformers import AutoModel, CLIPModel
    except ImportError:
        sys.exit("This needs PyTorch and transformers: "
                 "pip install torch transformers")

    hf_id = MODEL_ALIASES.get(model_name, model_name)
    is_clip = "clip" in hf_id.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_clip:
        model = CLIPModel.from_pretrained(hf_id)
        size, (mean, std) = 224, CLIP_NORM
        if img_size != 224:
            print("Note: CLIP is run at 224 px; --img-size is ignored.")
    else:
        model = AutoModel.from_pretrained(hf_id)
        size, (mean, std) = img_size, IMAGENET
        if size % 14:
            sys.exit("For DINOv2, --img-size must be a multiple of 14 "
                     "(e.g. 224, 336, 448).")
    model = model.eval().to(device)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    print(f"Model: {hf_id} on {device}, input {size}x{size}")

    @torch.no_grad()
    def extract(paths, batch_size=32):
        feats = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                src = p if isinstance(p, Image.Image) else Image.open(p)
                img = letterbox(src.convert("RGB"), size)
                batch.append(torch.from_numpy(np.asarray(img).copy()))
            x = torch.stack(batch).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean_t) / std_t
            if is_clip:
                f = model.get_image_features(pixel_values=x)
                if not torch.is_tensor(f):  # some library versions wrap it
                    f = getattr(f, "pooler_output", f[0])
            else:
                out = model(pixel_values=x)
                # CLS token + mean of patch tokens: the usual linear-probe feature
                f = torch.cat([out.last_hidden_state[:, 0],
                               out.last_hidden_state[:, 1:].mean(dim=1)], dim=1)
            feats.append(f.float().cpu().numpy())
            print(f"  features {min(i + batch_size, len(paths))}/{len(paths)}",
                  end="\r")
        print()
        return np.concatenate(feats, axis=0)

    return extract


def get_features(items, class_dir, model_name, img_size):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{model_name}_{img_size}")
    cache_path = class_dir / f"_features_{safe}.npz"
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(z["files"].tolist(), z["X"]))

    missing = [p for p, _ in items if p.name not in cache]
    if missing:
        print(f"Extracting features for {len(missing)} new crop(s) "
              f"({len(items) - len(missing)} cached)...")
        extract = load_extractor(model_name, img_size)
        X_new = extract(missing)
        for p, x in zip(missing, X_new):
            cache[p.name] = x
        np.savez(cache_path, files=np.array(list(cache.keys())),
                 X=np.stack(list(cache.values())))
    else:
        print(f"All {len(items)} crops already in feature cache.")
    return np.stack([cache[p.name] for p, _ in items])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def make_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"),
    )


def save_probe(path, X, y, args, n_groups):
    """Fit the probe on ALL crops and save it with everything inference needs
    to reproduce the features exactly (model, input size, feature type)."""
    import datetime
    import joblib
    import sklearn
    clf = make_clf().fit(X, y)
    bundle = {
        "classifier": clf,
        "classes": [str(c) for c in clf.classes_],
        "model": args.model,
        "hf_id": MODEL_ALIASES.get(args.model, args.model),
        "img_size": args.img_size,
        "feature": "CLS+mean-patch",
        "preprocess": "letterbox",
        "n_crops": int(len(y)),
        "crops_per_class": {str(c): int((y == c).sum()) for c in sorted(set(y))},
        "n_groups": int(n_groups),
        "trained_on": str(Path(args.class_dir).resolve()),
        "sklearn_version": sklearn.__version__,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"\nSaved probe trained on all {len(y)} crops to {path}")


def evaluate(X, y, groups, n_folds, seed, unit="video"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix,
                                 f1_score)
    from sklearn.model_selection import (LeaveOneGroupOut,
                                         StratifiedGroupKFold, StratifiedKFold,
                                         cross_val_predict)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_clf()
    classes = sorted(set(y))
    n_groups = len(set(groups))
    if n_groups < 2:
        sys.exit(f"Need crops from at least 2 different {unit}s to split by {unit}.")

    if n_folds == 0:
        # leave-one-group-out: every group is tested once by a model trained on
        # all the others. No randomness, so the result does not depend on the
        # seed or on how files are named or ordered.
        k = n_groups
        cv = LeaveOneGroupOut()
        split_desc = f"leave-one-{unit}-out ({k} folds)"
    else:
        k = min(n_folds, n_groups)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        split_desc = f"{k}-fold split BY {unit.upper()}"
    proba = cross_val_predict(clf, X, y, cv=cv, groups=groups,
                              method="predict_proba")
    pred = np.array(classes)[proba.argmax(axis=1)]

    # same thing with a plain random split, to show how much leakage inflates it
    class_counts = np.bincount([classes.index(c) for c in y])
    leaky_acc = None
    if class_counts.min() >= 2:
        skf = StratifiedKFold(n_splits=min(k, 5, class_counts.min()),
                              shuffle=True, random_state=seed)
        pred_leaky = cross_val_predict(clf, X, y, cv=skf)
        leaky_acc = accuracy_score(y, pred_leaky)

    maj = class_counts.max() / len(y)
    print("\n" + "=" * 60)
    print(f"{len(y)} crops, {len(classes)} classes, {n_groups} {unit} groups, "
          f"{split_desc}")
    print("=" * 60)
    print(f"{'Accuracy (by ' + unit + ')':<25}: {accuracy_score(y, pred):.3f}")
    print(f"Balanced accuracy        : {balanced_accuracy_score(y, pred):.3f}")
    print(f"Macro F1                 : {f1_score(y, pred, average='macro'):.3f}")
    print(f"Majority-class baseline  : {maj:.3f}   (what always guessing the "
          f"biggest class scores)")
    if leaky_acc is not None:
        print(f"Accuracy (random split)  : {leaky_acc:.3f}   (optimistic: "
              f"near-duplicates leak across the split)")
    print("\nPer-class results:")
    print(classification_report(y, pred, labels=classes, digits=3,
                                zero_division=0))
    cm = confusion_matrix(y, pred, labels=classes)
    w = max(len(c) for c in classes) + 2
    print("Confusion matrix (rows = true label, columns = predicted):")
    print(" " * w + "".join(f"{c[:9]:>10}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:<{w}}" + "".join(f"{v:>10}" for v in row))
    return classes, proba, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("class_dir", help="e.g. crops/fire_extinguisher "
                    "(contains one subfolder per attribute value)")
    ap.add_argument("--model", default="dinov2-small",
                    help=f"One of {sorted(MODEL_ALIASES)} or any Hugging Face "
                         f"DINOv2/CLIP model id")
    ap.add_argument("--img-size", type=int, default=224,
                    help="DINOv2 input size (multiple of 14). Try 336 or 448 "
                         "for small details such as exit-sign arrows.")
    ap.add_argument("--exclude", nargs="*", default=["unknown", "unsorted"],
                    help="Subfolders to skip (default: unknown unsorted)")
    ap.add_argument("--min-per-class", type=int, default=10,
                    help="Drop attribute values with fewer crops than this")
    ap.add_argument("--folds", type=int, default=5,
                    help="Number of grouped folds; 0 = leave-one-group-out "
                         "(with --group-by location: leave-one-location-out)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-by", choices=["video", "location"], default="video",
                    help="Unit kept on one side of each split: 'video' (each "
                         "video / source photo) or 'location' (whole building, "
                         "via location_map.csv with columns date,location)")
    ap.add_argument("--location-map", default=None,
                    help="Path to location_map.csv (default: look next to the "
                         "crops and in parent folders)")
    ap.add_argument("--save-probe", default=None, metavar="PATH",
                    help="After evaluating, fit the probe on ALL crops and save "
                         "it (e.g. probes/extinguisher_dinov2s_448.joblib) for "
                         "classify_subtypes.py")
    ap.add_argument("--plot", action="store_true",
                    help="Also save a 2-D t-SNE picture of the feature space")
    args = ap.parse_args()

    class_dir = Path(args.class_dir)
    if not class_dir.is_dir():
        sys.exit(f"{class_dir} is not a folder.")
    exclude = {e.lower() for e in args.exclude}
    # one line with every setting, to paste into the logbook
    print(f"Settings: model={args.model}, img_size={args.img_size} "
          f"(letterbox), feature=CLS+mean-patch, "
          f"classifier=StandardScaler+LogReg(C=0.5, balanced), "
          f"folds={args.folds if args.folds else 'leave-one-out'} grouped by "
          f"{'video/photo' if args.group_by == 'video' else 'location'}, "
          f"seed={args.seed}, "
          f"exclude={sorted(exclude)}, min_per_class={args.min_per_class}")
    items = collect(class_dir, exclude)
    if not items:
        sys.exit(f"No sorted crops found under {class_dir} (skipping "
                 f"{sorted(exclude)}). Did you create the attribute folders?")

    counts = {}
    for _, lab in items:
        counts[lab] = counts.get(lab, 0) + 1
    print("Crops per attribute value:", counts)
    small = {k for k, v in counts.items() if v < args.min_per_class}
    if small:
        print(f"Dropping (fewer than {args.min_per_class} crops): "
              f"{sorted(small)}")
        items = [(p, lab) for p, lab in items if lab not in small]
    if len({lab for _, lab in items}) < 2:
        sys.exit("Need at least 2 attribute values with enough crops.")

    X = get_features(items, class_dir, args.model, args.img_size)
    y = np.array([lab for _, lab in items])
    locs = None
    if args.group_by == "location":
        mapping = load_location_map(find_location_map(class_dir,
                                                      args.location_map))
        groups, locs = location_groups(items, mapping)
    else:
        groups = np.array([video_of(p.name) for p, _ in items])

    classes, proba, pred = evaluate(X, y, groups, args.folds, args.seed,
                                    unit=args.group_by)

    # Per-location results: each location was predicted by a model that never
    # saw that building
    if locs is not None:
        print("\nBreakdown by location (each one held out as a whole):")
        for loc in sorted(set(locs)):
            m = locs == loc
            correct = int((pred[m] == y[m]).sum())
            per_class = ", ".join(
                f"{c}: {int(((pred[m] == c) & (y[m] == c)).sum())}/{int((y[m] == c).sum())}"
                for c in classes if (y[m] == c).any())
            print(f"  {loc[:34]:<35}{int(m.sum()):>4} crops  accuracy "
                  f"{correct / m.sum():.3f}  ({per_class})")

    # Per-source results (source guessed from file names, see source_of)
    srcs = np.array([source_of(p.name) for p, _ in items])
    if len(set(srcs)) > 1:
        print("\nBreakdown by source (guessed from file names):")
        order = ["own video", "own 360 photo", "own early frames"]
        masks = [(n, srcs == n) for n in order if (srcs == n).any()]
        own = np.array([x.startswith("own") for x in srcs])
        if own.any():
            masks.append(("ALL OWN", own))
        if (srcs == "other").any():
            masks.append(("other", srcs == "other"))
        for label, m in masks:
            correct = int((pred[m] == y[m]).sum())
            per_class = ", ".join(
                f"{c}: {int(((pred[m] == c) & (y[m] == c)).sum())}/{int((y[m] == c).sum())}"
                for c in classes)
            print(f"  {label:<17}{int(m.sum()):>4} crops  accuracy "
                  f"{correct / m.sum():.3f}  ({per_class})")

    # misclassified crops, most confidently-wrong first: look at these by eye,
    # they are often mislabelled crops
    out_csv = class_dir / ("probe_errors.csv" if args.group_by == "video"
                           else "probe_errors_by_location.csv")
    rows = []
    for i, ((p, lab), pr, pd_) in enumerate(zip(items, proba, pred)):
        if pd_ != lab:
            rows.append({
                "crop": str(p.relative_to(class_dir)),
                "true": lab, "predicted": pd_,
                "p_true": round(float(pr[classes.index(lab)]), 3),
                "p_predicted": round(float(pr[classes.index(pd_)]), 3),
                "video": video_of(p.name),
                "location": locs[i] if locs is not None else "",
            })
    rows.sort(key=lambda r: -r["p_predicted"])
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "true", "predicted",
                                           "p_true", "p_predicted", "video",
                                           "location"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} misclassified crops listed in {out_csv} "
          f"(most confidently wrong first -- check these by eye).")

    if args.save_probe:
        save_probe(args.save_probe, X, y, args, len(set(groups)))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        Z = TSNE(n_components=2, init="pca", random_state=args.seed,
                 perplexity=max(5, min(30, len(y) // 4))
                 ).fit_transform(StandardScaler().fit_transform(X))
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in classes:
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7, label=f"{c} ({m.sum()})")
        ax.legend()
        ax.set_title(f"t-SNE of {args.model} features -- {class_dir.name}")
        ax.set_xticks([])
        ax.set_yticks([])
        out_png = class_dir / "probe_tsne.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
