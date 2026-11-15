#!/usr/bin/env python3
"""
train_eval.py - train a YOLO26 segmentation model, then evaluate best.pt on the TEST split
and save the results needed for the paper.

Uses the data.yaml written by make_split.py (either <root>/data_split.yaml or
<export>/data.yaml). The test set is only touched once, after training.

Example (own data only, nano model):
  python train_eval.py --data "D:/.../FSE-360_v1_split/data.yaml" --model yolo26n-seg.pt ^
                       --name EXP001_y26n-seg_own-only

Final model (D-25): --skip-test trains only. Use it when the training list also contains the
test images; the run is then logged as "not evaluated on test".

Outputs (in <project>/<name>/):
  weights/best.pt, results.csv, args.yaml, plots      (Ultralytics training output)
  test/                                               (test-set plots, confusion matrix)
  test_metrics_per_class.csv                          box + mask P, R, mAP50, mAP50-95
  test_metrics_overall.csv                            one row, also appended to the log
<project>/experiment_log.csv                          one row per run - your results table
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml from make_split.py")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--name", required=True, help="run name, e.g. EXP001_y26n-seg_own-only")
    p.add_argument("--project", default="experiments")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="lower to 8 or 4 on out-of-memory")
    p.add_argument("--patience", type=int, default=30, help="stop early if val stops improving")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0", help="0 = first GPU, cpu = CPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="none", choices=["none", "ram", "disk"],
                   help="cache decoded images: ram = fastest (needs free RAM), "
                        "disk = .npy files next to the images (needs a writable dataset folder)")
    p.add_argument("--notes", default="", help="free text for the experiment log")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training, evaluate <project>/<name>/weights/best.pt on test")
    p.add_argument("--skip-test", action="store_true",
                   help="train only, no test evaluation (final model whose training data "
                        "includes the test images, D-25)")
    a = p.parse_args()
    if a.skip_test and a.eval_only:
        p.error("--skip-test and --eval-only cannot be combined")
    return a


def f4(x):
    return f"{float(x):.4f}"


def main():
    a = get_args()
    run_dir = Path(a.project).resolve() / a.name
    t0 = time.time()

    # ---------- train ----------
    if not a.eval_only:
        if (run_dir / "weights").exists():
            raise SystemExit(f"Run folder already exists: {run_dir}\n"
                             f"Use a new --name, or --eval-only to re-evaluate that run.")
        model = YOLO(a.model)
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                    patience=a.patience, workers=a.workers, device=a.device, seed=a.seed,
                    deterministic=True, project=str(run_dir.parent), name=run_dir.name,
                    exist_ok=False, plots=True,
                    cache=False if a.cache == "none" else a.cache)
        run_dir = Path(model.trainer.save_dir).resolve()  # the folder Ultralytics really used
    train_min = (time.time() - t0) / 60

    # ---------- final model: test images are training data, so no test evaluation ----------
    if a.skip_test:
        row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "run": run_dir.name,
               "model": Path(a.model).name, "data": a.data, "epochs": a.epochs,
               "imgsz": a.imgsz, "batch": a.batch, "seed": a.seed, "cache": a.cache,
               "box_mAP50": "", "box_mAP50-95": "", "mask_mAP50": "", "mask_mAP50-95": "",
               "box_P": "", "box_R": "", "mask_P": "", "mask_R": "",
               "ms_per_img_inference": "", "train_minutes": f"{train_min:.1f}",
               "notes": ("NOT EVALUATED ON TEST: test images are in the training data. " + a.notes).strip()}
        with open(run_dir / "test_metrics_overall.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        log = run_dir.parent / "experiment_log.csv"
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"\nTraining done ({train_min:.1f} min). No test evaluation: the test images are "
              f"part of the training data.\nWeights: {run_dir / 'weights' / 'best.pt'}\nLog:     {log}")
        return

    # ---------- evaluate best.pt on the test split ----------
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"best.pt not found: {best}")
    m = YOLO(str(best)).val(data=a.data, split="test", imgsz=a.imgsz, batch=a.batch,
                            device=a.device, project=str(run_dir), name="test",
                            exist_ok=True, plots=True)
    names = m.names

    # per class (only classes that have test instances appear in ap_class_index)
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        c = int(c)
        rows.append([c, names[c],
                     f4(m.box.p[i]), f4(m.box.r[i]), f4(m.box.ap50[i]), f4(m.box.ap[i]),
                     f4(m.seg.p[i]), f4(m.seg.r[i]), f4(m.seg.ap50[i]), f4(m.seg.ap[i])])
    present = {r[0] for r in rows}
    for c in sorted(names):
        if c not in present:
            rows.append([c, names[c]] + ["no test instances"] + [""] * 7)
    rows.sort(key=lambda r: r[0])
    head = ["id", "class", "box_P", "box_R", "box_mAP50", "box_mAP50-95",
            "mask_P", "mask_R", "mask_mAP50", "mask_mAP50-95"]
    with open(run_dir / "test_metrics_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)

    # overall + log
    overall = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run": run_dir.name,
        "model": "best.pt (eval-only)" if a.eval_only else (
            f"{Path(a.model).parent.parent.name}/best.pt (fine-tune)"
            if Path(a.model).name in ("best.pt", "last.pt") else Path(a.model).name),
        "data": a.data,
        "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch, "seed": a.seed,
        "cache": a.cache,
        "box_mAP50": f4(m.box.map50), "box_mAP50-95": f4(m.box.map),
        "mask_mAP50": f4(m.seg.map50), "mask_mAP50-95": f4(m.seg.map),
        "box_P": f4(m.box.mp), "box_R": f4(m.box.mr),
        "mask_P": f4(m.seg.mp), "mask_R": f4(m.seg.mr),
        "ms_per_img_inference": f"{m.speed.get('inference', 0):.1f}",
        "train_minutes": "" if a.eval_only else f"{train_min:.1f}",
        "notes": a.notes,
    }
    with open(run_dir / "test_metrics_overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall))
        w.writeheader()
        w.writerow(overall)
    log = run_dir.parent / "experiment_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall))
        if new:
            w.writeheader()
        w.writerow(overall)

    print("\n=== TEST SET (held-out locations) ===")
    print(f"mask mAP50-95 {overall['mask_mAP50-95']}  mask mAP50 {overall['mask_mAP50']}  "
          f"box mAP50-95 {overall['box_mAP50-95']}")
    print(f"{'class':<24}{'mask mAP50':>11}{'mask mAP50-95':>15}")
    for r in rows:
        if r[2] == "no test instances":
            print(f"{r[1]:<24}{'-':>11}{'-':>15}   (no test instances)")
        else:
            print(f"{r[1]:<24}{r[8]:>11}{r[9]:>15}")
    print(f"\nSaved: {run_dir / 'test_metrics_per_class.csv'}\nLog:   {log}")


if __name__ == "__main__":  # required on Windows (data-loader workers)
    main()
