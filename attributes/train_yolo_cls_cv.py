"""
YOLO classification (e.g. YOLO26-cls) on sorted crops, evaluated exactly like
train_attribute_cnn_v2.py: same crops, same grouping (location_map.csv), same
folds, same seeds, and results saved in the same format, so the two can be
compared crop by crop with compare_runs.py.

Keep this script in the same folder as train_attribute_cnn_v2.py and
probe_separability_v4.py (it imports their data and reporting code).

What differs from plain `yolo classify train`, and why:
  * Horizontal flip is OFF (Ultralytics default fliplr=0.5 would mirror a left
    arrow and still call it "left"). --swap-flip LABEL_A LABEL_B instead adds a
    mirrored copy of every training crop with those two labels swapped -- the
    offline equivalent of the label-swapping flip in the CNN script.
  * Every crop is letterboxed (padded with grey) to a square before YOLO sees
    it. Ultralytics' prediction transform resizes the short side and then
    centre-crops, which would cut off the ends of a wide exit sign -- often
    where the arrow is. --no-letterbox keeps the Ultralytics behaviour.
  * The held-out location is never given to Ultralytics: its "val" folder is a
    few training crops (Ultralytics needs one), training runs with val=False,
    and predictions come from last.pt -- nothing is chosen on held-out data.
Everything else is the standard Ultralytics classification recipe (RandAugment,
random erasing 0.4, RandomResizedCrop scale 0.5-1, AdamW "auto", warm-up);
change it with --auto-augment, --erasing, --scale, --epochs, --batch.

Outputs (as train_attribute_cnn_v2.py): <class_dir>/cnn_runs/<date-time>_<tag>/
    log.txt, settings.json, metrics.json, predictions.csv, errors.csv,
    yolo_logs/ (Ultralytics results.csv per fold)
and one line in cnn_runs/summary.csv.

Requires: pip install ultralytics (plus what train_attribute_cnn_v2.py needs)

Usage (exit-sign direction, leave-one-location-out, 3 seeds):
    python train_yolo_cls_cv.py "...\\crops\\3_=_Fire_Exit_Sign" --model yolo26n-cls.pt ^
        --imgsz 384 --swap-flip "direction left" "direction right" ^
        --exclude unknown unsorted results --group-by location --folds 0 ^
        --location-map "...\\stats\\location_map.csv" --repeats 3 --tag yolo26n-cls
Final models after a configuration is chosen (no cross-validation):
    ... --final-only --save-final "...\\models\\exit_sign_yolo26n_cls"
"""
import argparse
import datetime
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import train_attribute_cnn_v2 as cnn
    from probe_separability_v4 import (find_location_map, group_of, letterbox,
                                       load_location_map, location_groups,
                                       source_of)
except ImportError:
    sys.exit("train_yolo_cls_cv.py needs train_attribute_cnn_v2.py and "
             "probe_separability_v4.py in the same folder.")

GREY = (124, 116, 104)


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_fold_dataset(fold_dir, tr_idx, y_str, classes, lb_files, mir_files,
                       swap_names):
    """train/<class>/ = training crops (+ mirrored copies with labels swapped);
    val/<class>/ = up to 3 TRAINING crops per class, only because Ultralytics
    insists on a val folder. The held-out crops are not in here at all."""
    for split in ("train", "val"):
        for c in classes:
            (fold_dir / split / c).mkdir(parents=True, exist_ok=True)
    per_class_val = {c: 0 for c in classes}
    for i in tr_idx:
        c = y_str[i]
        link_or_copy(lb_files[i], fold_dir / "train" / c / lb_files[i].name)
        if swap_names is not None:
            cm = swap_names.get(c, c)
            link_or_copy(mir_files[i], fold_dir / "train" / cm / mir_files[i].name)
        if per_class_val[c] < 3:
            link_or_copy(lb_files[i], fold_dir / "val" / c / lb_files[i].name)
            per_class_val[c] += 1


def train_one(args, data_dir, project, name, seed, device):
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=str(data_dir), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, seed=seed, deterministic=True,
        fliplr=0.0, flipud=0.0, auto_augment=args.auto_augment or None,
        erasing=args.erasing, scale=args.scale, optimizer=args.optimizer,
        val=False, plots=False, project=str(project), name=name,
        exist_ok=True, workers=args.workers, device=device, verbose=False)
    save_dir = Path(model.trainer.save_dir)
    return save_dir, save_dir / "weights" / "last.pt"


def predict_probs(weights, files, classes, args, device, mir_files=None,
                  perm=None):
    """Class probabilities in `classes` order; with mir_files+perm, also the
    mirrored crops (flip TTA), averaged."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    idx = {v: k for k, v in m.names.items()}
    cols = [idx[c] for c in classes]

    def run(paths):
        out = []
        for i in range(0, len(paths), 64):
            res = m.predict(source=[str(p) for p in paths[i:i + 64]],
                            imgsz=args.imgsz, device=device, verbose=False)
            out += [r.probs.data.cpu().numpy()[cols] for r in res]
        return np.array(out, dtype=np.float32)

    p = run(files)
    if mir_files is not None:
        p = (p + run(mir_files)[:, perm]) / 2
    return p


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("class_dir")
    ap.add_argument("--model", default="yolo26n-cls.pt",
                    help="Ultralytics classification weights, e.g. "
                         "yolo26n-cls.pt, yolo26s-cls.pt, yolo11n-cls.pt")
    ap.add_argument("--imgsz", type=int, default=384,
                    help="square input size (384: a 2:1 sign is 384x192, as "
                         "in the ResNet runs)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--auto-augment", default="randaugment",
                    help="randaugment (Ultralytics default), autoaugment, "
                         "augmix, or '' for none")
    ap.add_argument("--erasing", type=float, default=0.4,
                    help="random erasing probability (Ultralytics default 0.4)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="RandomResizedCrop keeps 1-scale..1 of the area "
                         "(Ultralytics default 0.5)")
    ap.add_argument("--no-letterbox", action="store_true",
                    help="feed crops as they are (Ultralytics centre-crops at "
                         "prediction, which can cut off arrows)")
    ap.add_argument("--swap-flip", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="add a mirrored copy of every training crop, with "
                         "these two labels swapped")
    ap.add_argument("--tta-flip", action="store_true",
                    help="also predict the mirrored crop and average")
    ap.add_argument("--exclude", nargs="*", default=["unknown", "unsorted"])
    ap.add_argument("--min-per-class", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5,
                    help="grouped folds; 0 = leave-one-group-out")
    ap.add_argument("--group-by", choices=["video", "location"], default="video")
    ap.add_argument("--location-map", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--work-dir", default=None,
                    help="scratch folder for fold datasets and Ultralytics "
                         "runs (default: a short temp folder, deleted at the "
                         "end; short paths avoid Windows' 260-character limit)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--yolo-verbose", action="store_true",
                    help="show Ultralytics' own training output")
    ap.add_argument("--final-only", action="store_true",
                    help="skip cross-validation; train on all crops and save "
                         "(needs --save-final)")
    ap.add_argument("--save-final", default=None, metavar="PREFIX",
                    help="save final models as PREFIX_seed<N>.pt plus "
                         "PREFIX.json")
    args = ap.parse_args()
    if args.final_only and not args.save_final:
        sys.exit("--final-only needs --save-final PREFIX")
    if args.tta_flip and not args.swap_flip:
        print("Note: --tta-flip without --swap-flip assumes a mirror image "
              "keeps its label.")
    if not args.yolo_verbose:
        os.environ["YOLO_VERBOSE"] = "False"   # before importing ultralytics
    import torch
    try:
        import ultralytics
    except ImportError:
        sys.exit("pip install ultralytics")

    class_dir = Path(args.class_dir)
    if not class_dir.is_dir():
        sys.exit(f"{class_dir} is not a folder.")
    exclude = {e.lower() for e in args.exclude} | {"cnn_runs"}
    out_dir = Path(args.out_dir) if args.out_dir else class_dir / "cnn_runs"
    if out_dir.resolve().parent == class_dir.resolve():
        exclude.add(out_dir.name.lower())
    split_tag = "lolo" if args.folds == 0 else f"{args.folds}fold"
    auto = (f"{Path(args.model).stem}_{args.imgsz}_{split_tag}-{args.group_by}"
            f"{'_swap' if args.swap_flip else ''}"
            f"{'_tta' if args.tta_flip else ''}_x{args.repeats}")
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", args.tag or auto)
    started = datetime.datetime.now()
    run_dir = out_dir / f"{started:%Y%m%d-%H%M%S}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_fh = open(run_dir / "log.txt", "w", encoding="utf-8")
    sys.stdout = cnn.Tee(sys.__stdout__, log_fh)
    sys.stderr = cnn.Tee(sys.__stderr__, log_fh)
    print(f"Run folder: {run_dir}")
    print("Command: " + " ".join(f'"{a}"' if " " in a else a for a in sys.argv))
    print(f"Settings: model={args.model} (Ultralytics {ultralytics.__version__}),"
          f" imgsz={args.imgsz} {'letterbox' if not args.no_letterbox else 'no letterbox (centre crop)'},"
          f" epochs={args.epochs}, batch={args.batch}, optimizer={args.optimizer},"
          f" auto_augment={args.auto_augment or 'none'}, erasing={args.erasing},"
          f" scale={args.scale}, fliplr=0, swap_flip={args.swap_flip},"
          f" tta_flip={args.tta_flip}, folds="
          f"{args.folds if args.folds else 'leave-one-out'} grouped by "
          f"{args.group_by}, seeds={args.seed}"
          f"{'..' + str(args.seed + args.repeats - 1) if args.repeats > 1 else ''},"
          f" last-epoch weights")

    # ---- data (same order as train_attribute_cnn_v2.py -> same folds) ------
    items = cnn.collect(class_dir, exclude)
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
    y_str = np.array([lab for _, lab in items])
    swap_names = perm = None
    if args.swap_flip:
        a, b = args.swap_flip
        if a not in classes or b not in classes:
            sys.exit(f"--swap-flip labels must be among {classes}")
        swap_names = {a: b, b: a}
        perm = [classes.index(swap_names.get(c, c)) for c in classes]
    elif args.tta_flip:
        perm = list(range(len(classes)))

    locs = None
    if args.group_by == "location":
        mapping = load_location_map(find_location_map(class_dir,
                                                      args.location_map))
        groups, locs = location_groups(items, mapping)
    else:
        groups = np.array([group_of(p.name) for p, _ in items])
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
    n_groups = len(set(groups))
    if args.folds == 0:
        k, cv = n_groups, LeaveOneGroupOut()
        split_desc = f"leave-one-{args.group_by}-out ({k} folds)"
    else:
        k = min(args.folds, n_groups)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                  random_state=args.seed)
        split_desc = f"{k}-fold split by {args.group_by}"
    print(f"{n_groups} {args.group_by} groups, {split_desc}")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")

    # ---- letterboxed (and mirrored) copies, short names, short path ------
    work = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="ycv_"))
    work.mkdir(parents=True, exist_ok=True)
    img_dir = work / "img"
    img_dir.mkdir(exist_ok=True)
    lb_files, mir_files = [], []
    for i, (p, _) in enumerate(items):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if not args.no_letterbox:
                im = letterbox(im, args.imgsz)
            f, fm = img_dir / f"{i:05d}.png", img_dir / f"{i:05d}m.png"
            im.save(f)
            im.transpose(Image.FLIP_LEFT_RIGHT).save(fm)
        lb_files.append(f)
        mir_files.append(fm)
    seeds = [args.seed + r for r in range(args.repeats)]

    def finish(status):
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        minutes = round((time.time() - t_start) / 60, 1)
        print(f"\n{status} (took {minutes} min). Results in {run_dir}")
        sys.stdout.flush()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_fh.close()
        return minutes

    base_settings = {
        "run": run_dir.name, "started": started.isoformat(timespec="seconds"),
        "command": sys.argv, "class_dir": str(class_dir.resolve()),
        "args": vars(args), "classes": classes,
        "fixed": {"fliplr": 0.0, "flipud": 0.0, "val_during_training": False,
                  "weights_used": "last.pt", "preprocess":
                  "letterbox grey to square" if not args.no_letterbox
                  else "Ultralytics resize + centre crop"},
        "data": {"n_crops": len(items),
                 "crops_per_class": {c: int((y_str == c).sum()) for c in classes},
                 "dropped_classes": sorted(small), "split": split_desc,
                 "n_groups": int(n_groups)},
        "seeds": seeds,
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__,
                     "device": torch.cuda.get_device_name(0)
                     if device == 0 else "cpu"},
        "note": args.note}

    # ---- final models only ------------------------------------------------
    if args.final_only:
        prefix = Path(args.save_final)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        saved = []
        for sd in seeds:
            print(f"\nTraining final model on ALL crops (seed {sd})...")
            d = work / f"final_s{sd}"
            build_fold_dataset(d, range(len(items)), y_str, classes, lb_files,
                               mir_files, swap_names)
            _, last = train_one(args, d, work / "train", f"final_s{sd}", sd,
                                device)
            out = prefix.parent / f"{prefix.name}_seed{sd}.pt"
            shutil.copy2(last, out)
            saved.append(str(out))
            print(f"  saved {out}")
        meta = {"classes": classes, "model": args.model, "imgsz": args.imgsz,
                "letterbox": not args.no_letterbox, "swap_flip": args.swap_flip,
                "tta_flip": args.tta_flip, "seeds": seeds, "weights": saved,
                "n_crops": len(items), "note": args.note}
        with open(str(prefix) + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        base_settings.update({"mode": "final-only", "model_meta": meta})
        with open(run_dir / "settings.json", "w", encoding="utf-8") as fh:
            json.dump(base_settings, fh, indent=2, ensure_ascii=False)
        finish(f"Saved {len(saved)} model(s) and {prefix}.json")
        return

    # ---- cross-validation ---------------------------------------------------
    splits = list(cv.split(np.zeros(len(y_str)), y_str, groups))
    fold_of = np.zeros(len(y_str), dtype=int)
    for f_, (_, te) in enumerate(splits, 1):
        fold_of[te] = f_
    (run_dir / "yolo_logs").mkdir()
    all_oof = []
    for r, sd in enumerate(seeds, 1):
        oof = np.zeros((len(items), len(classes)), dtype=np.float32)
        for fold, (tr, te) in enumerate(splits, 1):
            t0 = time.time()
            name = f"s{sd}f{fold:02d}"
            d = work / name
            build_fold_dataset(d, tr, y_str, classes, lb_files, mir_files,
                               swap_names)
            save_dir, last = train_one(args, d, work / "train", name, sd, device)
            oof[te] = predict_probs(
                last, [lb_files[i] for i in te], classes, args, device,
                [mir_files[i] for i in te] if args.tta_flip else None, perm)
            acc = float((np.array(classes)[oof[te].argmax(1)] == y_str[te]).mean())
            print(f"seed {sd} ({r}/{len(seeds)}), fold {fold}/{k}: train "
                  f"{len(tr)}, test {len(te)}, held-out acc {acc:.3f} "
                  f"({time.time() - t0:.0f} s)")
            if (save_dir / "results.csv").exists():
                shutil.copy2(save_dir / "results.csv",
                             run_dir / "yolo_logs" / f"{name}_results.csv")
            if not args.keep_work:
                shutil.rmtree(d, ignore_errors=True)
                shutil.rmtree(save_dir, ignore_errors=True)
        all_oof.append(oof)

    # ---- report (same layout as train_attribute_cnn_v2.py) -----------------
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score)
    seed_preds = [np.array(classes)[o.argmax(1)] for o in all_oof]
    seed_summary = None
    if len(seeds) > 1:
        print("\n" + "=" * 60)
        print(f"Single models, one per seed ({split_desc})")
        print("=" * 60)
        st = []
        for sd, pr in zip(seeds, seed_preds):
            s3 = (accuracy_score(y_str, pr), balanced_accuracy_score(y_str, pr),
                  f1_score(y_str, pr, average="macro"))
            st.append(s3)
            print(f"  seed {sd:<4} accuracy {s3[0]:.3f}  balanced {s3[1]:.3f}  "
                  f"macro F1 {s3[2]:.3f}")
        st = np.array(st)
        mean, sd_ = st.mean(0), st.std(0, ddof=1)
        print(f"  mean +- sd  accuracy {mean[0]:.3f} +- {sd_[0]:.3f}  balanced "
              f"{mean[1]:.3f} +- {sd_[1]:.3f}  macro F1 {mean[2]:.3f} +- "
              f"{sd_[2]:.3f}")
        print("The report below is the ENSEMBLE of these seeds.")
        keys = ("accuracy", "balanced_accuracy", "macro_f1")
        seed_summary = {"mean": {k_: round(float(v), 4) for k_, v in zip(keys, mean)},
                        "sd": {k_: round(float(v), 4) for k_, v in zip(keys, sd_)}}
    oof = np.mean(all_oof, axis=0)
    pred = np.array(classes)[oof.argmax(1)]
    cnn.report(y_str, pred, classes, args.group_by,
               split_desc + (f", ensemble of {len(seeds)} seeds"
                             if len(seeds) > 1 else ""))

    srcs = np.array([source_of(p.name) for p, _ in items])

    def breakdown(title, keys_, order=None):
        print(f"\n{title}")
        out = {}
        for key in (order or sorted(set(keys_))):
            m = keys_ == key
            if not m.any():
                continue
            e = {"n_crops": int(m.sum()),
                 "accuracy": round(float((pred[m] == y_str[m]).mean()), 4),
                 "per_class_correct": {
                     c: [int(((pred[m] == c) & (y_str[m] == c)).sum()),
                         int((y_str[m] == c).sum())]
                     for c in classes if (y_str[m] == c).any()}}
            spread = ""
            if len(seeds) > 1:
                accs = [float((sp[m] == y_str[m]).mean()) for sp in seed_preds]
                e["seed_accuracy_mean"] = round(float(np.mean(accs)), 4)
                e["seed_accuracy_sd"] = round(float(np.std(accs, ddof=1)), 4)
                spread = (f"  [single seeds {np.mean(accs):.3f} +- "
                          f"{np.std(accs, ddof=1):.3f}]")
            pc = ", ".join(f"{c}: {v[0]}/{v[1]}" for c, v in
                           e["per_class_correct"].items())
            print(f"  {key[:34]:<35}{int(m.sum()):>4} crops  accuracy "
                  f"{e['accuracy']:.3f}{spread}  ({pc})")
            out[key] = e
        return out

    by_location = breakdown("Breakdown by location (each one held out as a "
                            "whole):", locs) if locs is not None else {}
    by_source = breakdown("Breakdown by source (guessed from file names):",
                          srcs, ["own video", "own 360 photo",
                                 "own early frames", "other"]) \
        if len(set(srcs)) > 1 else {}

    # ---- save ---------------------------------------------------------------
    import csv
    with open(run_dir / "settings.json", "w", encoding="utf-8") as fh:
        json.dump(base_settings, fh, indent=2, ensure_ascii=False)
    ens = cnn.metric_dict(y_str, pred, classes)
    metrics = {"split": split_desc, "result": ens,
               "result_is": (f"ensemble of {len(seeds)} seeds"
                             if len(seeds) > 1 else "single model"),
               "majority_baseline": round(float(max((y_str == c).mean()
                                                    for c in classes)), 4),
               "per_seed": {str(sd): {k_: v for k_, v in
                                      cnn.metric_dict(y_str, sp, classes).items()
                                      if k_ in ("accuracy", "balanced_accuracy",
                                                "macro_f1")}
                            for sd, sp in zip(seeds, seed_preds)},
               "seed_mean_sd": seed_summary, "by_location": by_location,
               "by_source": by_source}
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
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
            row.update({f"pred_seed{sd}": sp[i] for sd, sp in zip(seeds, seed_preds)})
            w.writerow(row)
    rows = []
    for i, (p, lab) in enumerate(items):
        if pred[i] != lab:
            rows.append({"crop": str(p.relative_to(class_dir)), "true": lab,
                         "predicted": pred[i],
                         "p_true": round(float(oof[i, classes.index(lab)]), 3),
                         "p_predicted": round(float(oof[i].max()), 3),
                         "video": group_of(p.name),
                         "location": locs[i] if locs is not None else ""})
    rows.sort(key=lambda r_: -r_["p_predicted"])
    with open(run_dir / "errors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "true", "predicted", "p_true",
                                           "p_predicted", "video", "location"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} misclassified crops listed in {run_dir / 'errors.csv'}")

    big = {k_: v for k_, v in by_location.items()
           if v["n_crops"] >= 10 and not k_.startswith("(unmapped)")}
    worst = min(big.items(), key=lambda kv: kv[1]["accuracy"]) if big \
        else ("", {"accuracy": ""})
    row = {"run": run_dir.name, "date": f"{started:%Y-%m-%d %H:%M}",
           "class_folder": class_dir.name, "n_crops": len(items),
           "classes": "; ".join(f"{c} {int((y_str == c).sum())}" for c in classes),
           "split": split_desc, "n_groups": n_groups,
           "arch": Path(args.model).stem,
           "input_hxw": f"{args.imgsz}x{args.imgsz}"
                        f"{' letterbox' if not args.no_letterbox else ''}",
           "epochs": args.epochs,
           "augmentation": (f"ultralytics {args.auto_augment or 'none'} "
                            f"erasing{args.erasing} scale{args.scale} fliplr0"
                            f"{' +mirror-swap' if args.swap_flip else ''}"),
           "tta_flip": int(args.tta_flip),
           "seeds": " ".join(str(s) for s in seeds),
           "accuracy": ens["accuracy"], "balanced_accuracy": ens["balanced_accuracy"],
           "macro_f1": ens["macro_f1"],
           "acc_own_video": by_source.get("own video", {}).get("accuracy", ""),
           "acc_own_360_photo": by_source.get("own 360 photo", {}).get("accuracy", ""),
           "worst_location": worst[0], "worst_location_acc": worst[1]["accuracy"],
           "n_errors": len(rows), "note": args.note, "run_folder": str(run_dir)}
    if seed_summary:
        row.update({"seed_accuracy_mean": seed_summary["mean"]["accuracy"],
                    "seed_accuracy_sd": seed_summary["sd"]["accuracy"],
                    "seed_balanced_mean": seed_summary["mean"]["balanced_accuracy"],
                    "seed_balanced_sd": seed_summary["sd"]["balanced_accuracy"],
                    "seed_f1_mean": seed_summary["mean"]["macro_f1"],
                    "seed_f1_sd": seed_summary["sd"]["macro_f1"]})
    row["minutes"] = round((time.time() - t_start) / 60, 1)
    cnn.append_summary(out_dir / "summary.csv", row)
    finish("Done; one line added to " + str(out_dir / "summary.csv"))


if __name__ == "__main__":
    main()
