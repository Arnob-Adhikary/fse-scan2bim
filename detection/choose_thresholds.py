#!/usr/bin/env python3
"""
choose_thresholds.py - per-class confidence thresholds for the final model (D-25).

Rule (fixed before running):
  1. Each seed model predicts the held-out schools (the test split) down to confidence 0.05, the
     lowest threshold considered. Detections below 0.05 cannot change the result: they are matched
     after all higher-confidence ones and are never kept by any threshold in the search.
  2. A prediction is correct (TP) if its box overlaps a not-yet-matched label box of the same class
     with IoU >= 0.5 (greedy, highest confidence first, per image). Everything else is a false
     positive; unmatched labels are misses.
  3. For every class, TP / FP / misses of the three seed models are added up (pooled) and
     precision, recall and F1 are computed for thresholds 0.05, 0.06, ..., 0.95.
  4. The F1 curve is smoothed (moving average over 0.05) and the threshold with the highest
     smoothed F1 is chosen (ties: the higher threshold).
  5. Classes with fewer than --min-inst test instances (per model), or with no correct detection
     at any threshold, get the global threshold: the highest smoothed mean F1 over all classes
     (the same curve Ultralytics plots).
  The thresholds are then applied unchanged to the final model (trained on train + held-out schools).

Run in the yolo venv (GPU optional, about a minute per model on GPU):
  python choose_thresholds.py --data "D:/.../FSE-360_v2_split_public/data.yaml" ^
      --runs experiments/EXP007_y26s-seg_own+public_1024 ^
             experiments/EXP007_y26s-seg_own+public_1024_seed-1 ^
             experiments/EXP007_y26s-seg_own+public_1024_seed-2 ^
      --out thresholds_EXP007

Outputs (in --out):
  predictions_<run>.csv     all predictions (cached; --reuse skips inference next time)
  thresholds.csv            per class: test instances, threshold, source (class or global),
                            pooled precision / recall / F1 at that threshold and at 0.25 (default),
                            and the threshold each seed alone would have chosen (stability)
  thresholds.json           {class name: threshold} for inference
  f1_curves.png             pooled F1 against threshold per class, chosen thresholds marked

Using the thresholds at inference:
  r = model.predict(img, conf=min(thr.values()), imgsz=1024)[0]
  keep = [thr[r.names[int(c)]] <= float(p) for c, p in zip(r.boxes.cls, r.boxes.conf)]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRED_CONF = 0.05  # = lowest threshold in the search grid


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml of the exported split")
    p.add_argument("--runs", nargs="+", required=True,
                   help="run folders (with weights/best.pt) or .pt files, one per seed")
    p.add_argument("--split", default="test", help="split to use (default: test = held-out schools)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2, help="lower to 1 if the GPU runs out of memory")
    p.add_argument("--device", default=None, help="0 = first GPU, cpu = CPU (default: auto)")
    p.add_argument("--iou", type=float, default=0.5, help="box IoU for a correct detection")
    p.add_argument("--min-inst", type=int, default=15,
                   help="classes with fewer test instances get the global threshold")
    p.add_argument("--smooth", type=float, default=0.05, help="width of the F1 smoothing window")
    p.add_argument("--reuse", action="store_true", help="reuse cached predictions_<run>.csv")
    return p.parse_args()


def split_images(cfg, yaml_path, split):
    root = Path(cfg.get("path") or yaml_path.parent)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    imgs = []
    for e in entries:
        e = Path(e)
        e = e if e.is_absolute() else root / e
        if e.is_dir():
            imgs += sorted(p for p in e.rglob("*") if p.suffix.lower() in IMG_EXT)
        elif e.suffix == ".txt":
            for line in e.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    q = Path(line.strip())
                    imgs.append(q if q.is_absolute() else root / q)
        else:
            raise SystemExit(f"Cannot read split entry {e}")
    if not imgs:
        raise SystemExit(f"No images found for split '{split}'")
    return imgs


def label_path(img):
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(img):
    """YOLO-seg polygons -> list of (cls, x1, y1, x2, y2) in normalised coordinates."""
    lp = label_path(img)
    out = []
    if lp.exists():
        for line in lp.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c = int(v[0])
            xy = np.array(v[1:], dtype=float)
            if len(v) == 5:  # box label: cx cy w h
                cx, cy, w, h = xy
                out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            else:
                xs, ys = xy[0::2], xy[1::2]
                out.append((c, xs.min(), ys.min(), xs.max(), ys.max()))
    return out


def predict(weights, imgs, a):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    rows = []
    kw = dict(imgsz=a.imgsz, conf=PRED_CONF, stream=True, verbose=False, batch=a.batch)
    if a.device is not None:
        kw["device"] = a.device
    for k in range(0, len(imgs), 16):  # small chunks keep memory low
        chunk = [str(p) for p in imgs[k:k + 16]]
        for path, r in zip(chunk, model.predict(chunk, **kw)):
            b = r.boxes
            for (x1, y1, x2, y2), c, p in zip(b.xyxyn.tolist(), b.cls.tolist(), b.conf.tolist()):
                rows.append((path, int(c), float(p), x1, y1, x2, y2))
    return rows


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thr):
    """Greedy per image and class. Returns {cls: [(conf, tp), ...]}."""
    out = {}
    by_img = {}
    for path, c, p, *box in preds:
        by_img.setdefault((path, c), []).append((p, box))
    for (path, c), plist in by_img.items():
        g = [box for gc, *box in gts.get(path, []) if gc == c]
        used = [False] * len(g)
        for p, box in sorted(plist, key=lambda t: -t[0]):
            best, bj = iou_thr, -1
            for j, gb in enumerate(g):
                if not used[j]:
                    v = iou(box, gb)
                    if v >= best:
                        best, bj = v, j
            if bj >= 0:
                used[bj] = True
            out.setdefault(c, []).append((p, bj >= 0))
    return out


def curves(scored, n_gt, grid):
    """precision, recall, F1 over the grid from [(conf, tp)] and the number of labels."""
    if scored:
        conf = np.array([s[0] for s in scored])
        tp = np.array([s[1] for s in scored], dtype=bool)
    else:
        conf, tp = np.zeros(0), np.zeros(0, dtype=bool)
    P, R, F = [], [], []
    for t in grid:
        sel = conf >= t
        TP = int(tp[sel].sum())
        FP = int(sel.sum()) - TP
        p = TP / (TP + FP) if TP + FP else 0.0
        r = TP / n_gt if n_gt else 0.0
        P.append(p)
        R.append(r)
        F.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.array(P), np.array(R), np.array(F)


def smooth(f, k):
    if k <= 1:
        return f
    pad = np.pad(f, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def pick(fs, grid):
    best = fs.max()
    return float(grid[np.where(np.isclose(fs, best))[0][-1]])  # ties: highest threshold


def main():
    a = get_args()
    yaml_path = Path(a.data).resolve()
    cfg = yaml.safe_load(open(yaml_path, encoding="utf-8"))
    names = cfg["names"]
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    imgs = split_images(cfg, yaml_path, a.split)
    gts = {str(p): read_labels(p) for p in imgs}
    n_gt = {c: 0 for c in names}
    for lab in gts.values():
        for c, *_ in lab:
            n_gt[c] += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(imgs)} {a.split} images, {sum(n_gt.values())} labels, {len(a.runs)} models")

    grid = np.round(np.arange(0.05, 0.9501, 0.01), 2)
    k = max(1, int(round(a.smooth / 0.01)))
    per_run = []  # {cls: [(conf, tp)]} per model
    for run in a.runs:
        run = Path(run)
        w = run if run.suffix == ".pt" else run / "weights" / "best.pt"
        tag = run.stem if run.suffix == ".pt" else run.name
        cache = out / f"predictions_{tag}.csv"
        if a.reuse and cache.exists():
            with open(cache, newline="", encoding="utf-8") as f:
                preds = [(r[0], int(r[1]), float(r[2]), *map(float, r[3:])) for r in list(csv.reader(f))[1:]]
            print(f"{tag}: {len(preds)} cached predictions")
        else:
            if not w.exists():
                raise SystemExit(f"Weights not found: {w}")
            preds = predict(w, imgs, a)
            with open(cache, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["image", "cls", "conf", "x1", "y1", "x2", "y2"])
                wr.writerows(preds)
            print(f"{tag}: {len(preds)} predictions")
        per_run.append(match(preds, gts, a.iou))

    nr = len(per_run)
    pooled = {c: [s for m in per_run for s in m.get(c, [])] for c in names}
    cur = {c: curves(pooled[c], n_gt[c] * nr, grid) for c in names}
    present = [c for c in names if n_gt[c] > 0]
    mean_f1 = smooth(np.mean([cur[c][2] for c in present], axis=0), k)
    t_global = pick(mean_f1, grid)

    rows, thr = [], {}
    i25 = int(np.where(np.isclose(grid, 0.25))[0][0])
    for c in names:
        P, R, F = cur[c]
        if n_gt[c] == 0:
            t, src = t_global, "global (no test instances)"
        elif n_gt[c] < a.min_inst:
            t, src = t_global, f"global (< {a.min_inst} instances)"
        elif smooth(F, k).max() == 0:
            t, src = t_global, "global (no correct detections)"
        else:
            t, src = pick(smooth(F, k), grid), "class"
        i = int(np.where(np.isclose(grid, t))[0][0])
        seed_t = []
        for m in per_run:
            f1 = curves(m.get(c, []), n_gt[c], grid)[2]
            seed_t.append(f"{pick(smooth(f1, k), grid):.2f}" if n_gt[c] else "")
        thr[names[c]] = t
        rows.append([c, names[c], n_gt[c], f"{t:.2f}", src, f"{P[i]:.3f}", f"{R[i]:.3f}", f"{F[i]:.3f}",
                     f"{P[i25]:.3f}", f"{R[i25]:.3f}", f"{F[i25]:.3f}", " / ".join(seed_t)])

    head = ["id", "class", "test_instances", "threshold", "source", "P", "R", "F1",
            "P_at_0.25", "R_at_0.25", "F1_at_0.25", "threshold_per_seed"]
    with open(out / "thresholds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")

    print(f"\nGlobal threshold (highest smoothed mean F1): {t_global:.2f}")
    print(f"{'class':<24}{'n':>5}{'thr':>6}  {'P':>6}{'R':>6}{'F1':>6}   {'F1 @0.25':>8}   per seed")
    for r in rows:
        print(f"{r[1]:<24}{r[2]:>5}{r[3]:>6}  {r[5]:>6}{r[6]:>6}{r[7]:>6}   {r[10]:>8}   {r[11]}"
              + ("" if r[4] == "class" else f"   [{r[4]}]"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in present:
            line, = ax.plot(grid, smooth(cur[c][2], k), lw=1.2, label=f"{names[c]} ({thr[names[c]]:.2f})")
            ax.axvline(thr[names[c]], color=line.get_color(), lw=0.6, ls=":")
        ax.plot(grid, mean_f1, color="black", lw=2.5, label=f"mean ({t_global:.2f})")
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel(f"F1, pooled over {nr} models (smoothed)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(out / "f1_curves.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"\nSaved: {out / 'thresholds.csv'}, {out / 'thresholds.json'}, {out / 'f1_curves.png'}")


if __name__ == "__main__":  # required on Windows
    main()
