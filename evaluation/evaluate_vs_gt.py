"""
Evaluate the projected predictions (pano_seg_to_cloud.py) against the 3D ground truth
(cutout_instances.py / edit_gt.py) and write the tables and figures for the paper.

Both clouds carry per-point class_id / instance_id in the same point order (the
prediction was made on the GT's floor cloud). If the point order differs, the
prediction labels are moved to the GT points by nearest neighbour (--nn-tol), with a
warning.

What is measured (everything per class, plus overall):
  Object level   GT and predicted instances are matched greedily in order of confidence
                 (as in COCO / ScanNet), a pair counts when the point IoU >= --iou
                 (default 0.25). Class-aware: TP needs the same class.
                 -> precision, recall, F1, AP@0.25 and AP@0.5 (all-point interpolation),
                 also at IoU 0.5, at a centre distance <= --dist, and class-agnostic
                 (localisation only), for comparison with papers that count by distance.
  Error types    every FN is "missed" (no predicted point on it), "poor localisation"
                 (overlap below the IoU threshold) or "wrong class"; every FP is
                 "background", "poor localisation", "duplicate" or "wrong class".
  Confusion      class-agnostic matches: GT class x predicted class, plus missed / false
                 alarm.
  Localisation   for TP pairs: distance between box centres (3D; horizontal and vertical
                 parts per object in objects_gt.csv), and the instance mask IoU.
  Point level    per-class point IoU (semantic segmentation of the cloud), mIoU.
  Attributes     extinguisher type, exit-sign direction (any attribute column) on TP pairs
                 where the GT has one: accuracy, confusion; also for the confident ones
                 only (attribute_p >= --min-prob) with their coverage.
  Multi-view     F1 when dropping objects detected by too few of the nearby stations
                 that saw them (consistency sweep); recall vs distance to the nearest
                 scan station (--stations).
  Ablations      several prediction runs side by side (--run, first run = main).

Inputs
  --gt   CASE=GT_DIR ...           GT_DIR = cutout_instances.py --out-dir (gt_labelled.ply
                                   + instances_gt.csv): one folder with all floors merged,
                                   or one folder per floor (then give each its CASE name)
  --run  NAME CASE=PRED_DIR ...    one or more runs; PRED_DIR = pano_seg_to_cloud.py
                                   --out-dir (pred_labelled.ply + instances_pred.csv).
                                   The first run is the main result; all runs go into the
                                   ablation table.
  --stations stations.csv          optional, for recall vs distance to the nearest station

Outputs (in --out-dir)
  summary.md                       the key numbers in words, ready to quote
  tables.tex                       LaTeX (booktabs) tables: per class, criteria, errors,
                                   attributes, ablation
  csv/*.csv                        every table as CSV, plus objects_gt.csv / objects_pred.csv
                                   (one row per object with its match, IoU, error, type)
  figures/*.pdf + *.png            fig1 per-class P/R/F1, fig2 PR curves, fig3 confusion,
                                   fig4 localisation error + mask IoU, fig5 attributes,
                                   fig6 consistency sweep, fig7 recall vs range,
                                   fig8 error types, fig9 ablation, map_<case>[_floor<N>]
                                   floor plans (one per floor, also from a merged cloud)
  csv/per_floor.csv                results per floor (the 'floor' column of the instance
                                   CSVs; missing -> floor of the nearest station)
  clouds/errors_<case>.ply         object points coloured TP (blue) / FP (aqua) / FN (orange)
                                   with an error_type field (1 TP, 2 FP, 3 FN): load with
                                   the floor cloud in CloudCompare for qualitative figures

Usage (Command Prompt; ^ continues a line)
  both floors merged in one GT folder and one prediction folder:
  python evaluate_vs_gt.py ^
     --gt hotel=D:\\hotel\\gt\\cutout_all ^
     --run full hotel=D:\\hotel\\pred3d\\all ^
     --run per-class hotel=D:\\hotel\\pred3d_perclass\\all ^
     --stations D:\\hotel\\stations\\stations.csv --out-dir D:\\hotel\\eval
  one folder per floor:
  python evaluate_vs_gt.py ^
     --gt F1=D:\\hotel\\gt\\cutout_f1 F2=D:\\hotel\\gt\\cutout_f2 ^
     --run full F1=D:\\hotel\\pred3d\\floor1 F2=D:\\hotel\\pred3d\\floor2 ^
     --run per-class F1=D:\\hotel\\pred3d_perclass\\floor1 F2=D:\\hotel\\pred3d_perclass\\floor2 ^
     --stations D:\\hotel\\stations\\stations.csv --out-dir D:\\hotel\\eval

Requires: pip install numpy scipy matplotlib
"""
import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict

import numpy as np

CLASSES = ["Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
           "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED", "Fire Call Point",
           "Fire Blanket", "Emergency Door Release", "Evacuation Chair"]
PLY_TYPES = {"double": "<f8", "float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2", "char": "i1", "float32": "<f4",
             "float64": "<f8", "uint8": "u1", "int32": "<i4"}

# colours (validated: categorical slots 1-3 pass all-pairs CVD checks; ink for text)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SINGLE, DOUBLE = 3.54, 7.48            # Elsevier column widths (90 / 190 mm) in inches


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def norm_attr(s, alias):
    a = norm(s).replace("direction", "").replace("arrow", "")
    a = alias.get(a, a)
    return "" if a in ("", "unknown", "unk", "uncertain", "na", "none", "unclear", "tie") else a


# ---------------------------------------------------------------- reading
def read_ply(path):
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline()
            if not line:
                sys.exit(f"{path}: no end_header")
            line = line.decode("ascii", "replace").strip()
            head.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in head:
            sys.exit(f"{path}: expected a binary little-endian PLY (as our scripts write)")
        dt = []
        for l in head:
            if l.startswith("property"):
                _, t, name = l.split()[:3]
                dt.append((name, PLY_TYPES[t]))
        data = np.fromfile(f, dtype=dt)
    return data


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def instance_boxes(xyz, inst):
    """id -> (box centre, number of points) from the 0.5-99.5 percentile box, as in the
    GT and prediction scripts."""
    out = {}
    sel = np.flatnonzero(inst > 0)
    order = sel[np.argsort(inst[sel], kind="stable")]
    ids, starts = np.unique(inst[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    for i, a, b in zip(ids, starts, ends):
        Q = xyz[order[a:b]]
        lo, hi = np.percentile(Q, 0.5, axis=0), np.percentile(Q, 99.5, axis=0)
        out[int(i)] = ((lo + hi) / 2, b - a)
    return out


def load_case(case, gt_dir, pred_dir, a, alias, log):
    g = read_ply(os.path.join(gt_dir, "gt_labelled.ply"))
    p = read_ply(os.path.join(pred_dir, "pred_labelled.ply"))
    gxyz = np.c_[g["x"], g["y"], g["z"]]
    gi, gc = g["instance_id"].astype(np.int64), g["class_id"].astype(np.int64)
    if len(p) == len(g) and np.array_equal(p["x"], g["x"]) and np.array_equal(p["z"], g["z"]):
        pi, pc = p["instance_id"].astype(np.int64), p["class_id"].astype(np.int64)
    else:
        from scipy.spatial import cKDTree
        log(f"  WARNING {case}: prediction cloud has a different point order/count "
            f"({len(p):,} vs {len(g):,}); labels moved by nearest neighbour <= {a.nn_tol} m")
        pxyz = np.c_[p["x"], p["y"], p["z"]]
        obj = np.flatnonzero(p["instance_id"] > 0)
        d, k = cKDTree(pxyz[obj]).query(gxyz, distance_upper_bound=a.nn_tol)
        hit = np.isfinite(d)
        pi = np.zeros(len(g), np.int64)
        pc = np.full(len(g), -1, np.int64)
        pi[hit] = p["instance_id"][obj[k[hit]]]
        pc[hit] = p["class_id"][obj[k[hit]]]

    grows = {int(r["instance_id"]): r for r in read_csv(os.path.join(gt_dir, "instances_gt.csv"))}
    prows = {int(r["instance_id"]): r for r in read_csv(os.path.join(pred_dir, "instances_pred.csv"))}
    gbox, pbox = instance_boxes(gxyz, gi), instance_boxes(gxyz, pi)

    G = []
    for i, (cen, n) in sorted(gbox.items()):
        r = grows.get(i, {})
        cls = int(r["class_id"]) if r.get("class_id", "") != "" else \
            int(np.bincount(gc[gi == i][gc[gi == i] >= 0]).argmax())
        G.append({"case": case, "id": i, "cls": cls, "centre": cen, "n": n,
                  "attr": norm_attr(r.get("attribute", ""), alias),
                  "attr_raw": r.get("attribute", ""), "n_views_gt": r.get("n_views", ""),
                  "floor": str(r.get("floor", "") or "").strip()})
    missing = [i for i in grows if i not in gbox]
    if missing:
        log(f"  {case}: {len(missing)} GT rows without points skipped (e.g. FAILED cut-outs): "
            f"{missing[:10]}")
    P = []
    for j, (cen, n) in sorted(pbox.items()):
        r = prows.get(j, {})
        cls = int(r["class_id"]) if r.get("class_id", "") != "" else \
            int(np.bincount(pc[pi == j][pc[pi == j] >= 0]).argmax())
        P.append({"case": case, "id": j, "cls": cls, "centre": cen, "n": n,
                  "conf": fnum(r.get("conf_max"), 1.0),
                  "attr": norm_attr(r.get("attribute", ""), alias),
                  "attr_raw": r.get("attribute", ""),
                  "attr_p": fnum(r.get("attribute_p")),
                  "n_views": int(fnum(r.get("n_views"), 0)),
                  "seen_by": int(fnum(r.get("seen_by_near"), 0)),
                  "consistency": fnum(r.get("consistency")),
                  "floor": str(r.get("floor", "") or "").strip()})

    # intersections between GT and predicted instances
    both = (gi > 0) & (pi > 0)
    key = gi[both] * 1_000_000 + pi[both]
    k, cnt = np.unique(key, return_counts=True)
    gidx = {o["id"]: n for n, o in enumerate(G)}
    pidx = {o["id"]: n for n, o in enumerate(P)}
    inter = np.zeros((len(G), len(P)))
    for kk, c in zip(k, cnt):
        gg, pp = divmod(int(kk), 1_000_000)
        if gg in gidx and pp in pidx:
            inter[gidx[gg], pidx[pp]] = c
    gn = np.array([o["n"] for o in G], float)
    pn = np.array([o["n"] for o in P], float)
    iou = inter / np.maximum(gn[:, None] + pn[None, :] - inter, 1)

    # point-level (semantic) counts per class
    pts = {}
    for c in range(len(CLASSES)):
        gm, pm = gc == c, pc == c
        pts[c] = (int((gm & pm).sum()), int((~gm & pm).sum()), int((gm & ~pm).sum()))
    clouds = {"xyz": gxyz, "gi": gi, "pi": pi, "gc": gc, "pc": pc}
    return G, P, iou, pts, clouds


def combine(cases):
    """Stack several cases (floors) into one block-diagonal problem."""
    G, P, blocks, pts = [], [], [], defaultdict(lambda: [0, 0, 0])
    for c in cases:
        G += c["G"]
        P += c["P"]
        blocks.append(c["iou"])
        for k, v in c["pts"].items():
            for n in range(3):
                pts[k][n] += v[n]
    iou = np.zeros((len(G), len(P)))
    r = q = 0
    for b in blocks:
        iou[r:r + b.shape[0], q:q + b.shape[1]] = b
        r, q = r + b.shape[0], q + b.shape[1]
    dist = np.full((len(G), len(P)), np.inf)
    for i, go in enumerate(G):
        for j, po in enumerate(P):
            if go["case"] == po["case"]:
                dist[i, j] = np.linalg.norm(go["centre"] - po["centre"])
    return G, P, iou, dist, dict(pts)


# ---------------------------------------------------------------- matching
def greedy(G, P, iou, dist, crit, thr, class_aware=True, keep=None, only_cls=None):
    """Greedy one-to-one matching in order of prediction confidence.
    Returns gt->pred and pred->gt index arrays (-1 = unmatched)."""
    gcls = np.array([o["cls"] for o in G])
    conf = np.array([o["conf"] for o in P])
    g2p = np.full(len(G), -1)
    p2g = np.full(len(P), -1)
    used = np.zeros(len(G), bool)
    for j in np.argsort(-conf, kind="stable"):
        if keep is not None and not keep[j]:
            continue
        if only_cls is not None and P[j]["cls"] != only_cls:
            continue
        cand = ~used
        if class_aware:
            cand &= gcls == P[j]["cls"]
        if crit == "iou":
            s = np.where(cand & (iou[:, j] >= thr) & (iou[:, j] > 0), iou[:, j], -1.0)
            b = int(np.argmax(s)) if len(s) else -1
            ok = b >= 0 and s[b] >= 0
        else:
            s = np.where(cand & (dist[:, j] <= thr), dist[:, j], np.inf)
            b = int(np.argmin(s)) if len(s) else -1
            ok = b >= 0 and np.isfinite(s[b])
        if ok:
            used[b] = True
            g2p[b], p2g[j] = j, b
    return g2p, p2g


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (tp + fp) and (tp + fn) and (p + r) else (0.0 if tp + fn else np.nan)
    return p, r, f


def average_precision(G, P, iou, cls, thr):
    """All-point interpolated AP for one class; also returns the curve."""
    n_gt = sum(o["cls"] == cls for o in G)
    js = [j for j in np.argsort([-o["conf"] for o in P], kind="stable") if P[j]["cls"] == cls]
    if n_gt == 0:
        return np.nan, None
    used = np.zeros(len(G), bool)
    tp = []
    for j in js:
        cand = (~used) & np.array([o["cls"] == cls for o in G]) & (iou[:, j] >= thr) & (iou[:, j] > 0)
        if cand.any():
            b = int(np.argmax(np.where(cand, iou[:, j], -1)))
            used[b] = True
            tp.append(1)
        else:
            tp.append(0)
    if not tp:
        return 0.0, (np.array([0.0]), np.array([1.0]))
    tp = np.array(tp)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    mrec = np.r_[0, rec, rec[-1]]
    mpre = np.r_[1, prec, 0]
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, (rec, prec)


def error_types(G, P, iou, g2p, p2g, thr):
    gcls = np.array([o["cls"] for o in G])
    fn, fp = {}, {}
    for i in np.flatnonzero(g2p < 0):
        row = iou[i]
        if (row >= thr).any() and row.max() > 0:
            fn[i] = "wrong class"
        elif row.max() > 0:
            fn[i] = "poor localisation"
        else:
            fn[i] = "missed"
    for j in np.flatnonzero(p2g < 0):
        col = iou[:, j]
        good = np.flatnonzero((col >= thr) & (col > 0))
        if len(good):
            fp[j] = "duplicate" if (gcls[good] == P[j]["cls"]).any() else "wrong class"
        elif col.max() > 0:
            fp[j] = "poor localisation"
        else:
            fp[j] = "background"
    return fn, fp


def evaluate(G, P, iou, dist, pts, a, keep=None):
    """All headline numbers for one run (combined over cases)."""
    res = {}
    g2p, p2g = greedy(G, P, iou, dist, "iou", a.iou, True, keep)
    res["g2p"], res["p2g"] = g2p, p2g
    classes = sorted({o["cls"] for o in G} | {o["cls"] for o in P})
    rows = []
    for c in classes:
        gs = [i for i, o in enumerate(G) if o["cls"] == c]
        ps = [j for j, o in enumerate(P) if o["cls"] == c and (keep is None or keep[j])]
        tp = sum(g2p[i] >= 0 for i in gs)
        fp, fnn = len(ps) - tp, len(gs) - tp
        p, r, f = prf(tp, fp, fnn)
        ap25, _ = average_precision(G, P, iou, c, 0.25)
        ap50, _ = average_precision(G, P, iou, c, 0.5)
        d = [dist[i, g2p[i]] for i in gs if g2p[i] >= 0]
        miou = [iou[i, g2p[i]] for i in gs if g2p[i] >= 0]
        tpp, fpp, fnp = pts.get(c, (0, 0, 0))
        rows.append({"class_id": c, "class": CLASSES[c], "n_gt": len(gs), "n_pred": len(ps),
                     "TP": tp, "FP": fp, "FN": fnn, "precision": p, "recall": r, "F1": f,
                     "AP25": ap25, "AP50": ap50,
                     "loc_median_m": float(np.median(d)) if d else np.nan,
                     "loc_mean_m": float(np.mean(d)) if d else np.nan,
                     "mask_iou_mean": float(np.mean(miou)) if miou else np.nan,
                     "point_iou": tpp / (tpp + fpp + fnp) if tpp + fpp + fnp else np.nan})
    TP = sum(r["TP"] for r in rows)
    FP = sum(r["FP"] for r in rows)
    FN = sum(r["FN"] for r in rows)
    p, r, f = prf(TP, FP, FN)
    with_gt = [x for x in rows if x["n_gt"] > 0]
    d_all = [dist[i, g2p[i]] for i in range(len(G)) if g2p[i] >= 0]
    m_all = [iou[i, g2p[i]] for i in range(len(G)) if g2p[i] >= 0]
    res["per_class"] = rows
    res["overall"] = {"n_gt": len(G), "n_pred": int(sum(1 for j in range(len(P)) if keep is None or keep[j])),
                      "TP": TP, "FP": FP, "FN": FN, "precision": p, "recall": r, "F1": f,
                      "mAP25": np.nanmean([x["AP25"] for x in with_gt]) if with_gt else np.nan,
                      "mAP50": np.nanmean([x["AP50"] for x in with_gt]) if with_gt else np.nan,
                      "macro_F1": np.nanmean([x["F1"] for x in with_gt]) if with_gt else np.nan,
                      "loc_median_m": float(np.median(d_all)) if d_all else np.nan,
                      "loc_mean_m": float(np.mean(d_all)) if d_all else np.nan,
                      "loc_rmse_m": float(np.sqrt(np.mean(np.square(d_all)))) if d_all else np.nan,
                      "mask_iou_mean": float(np.mean(m_all)) if m_all else np.nan,
                      "mIoU_points": np.nanmean([x["point_iou"] for x in with_gt]) if with_gt else np.nan}
    # attributes
    att = defaultdict(list)
    for i, o in enumerate(G):
        j = g2p[i]
        if j >= 0 and o["attr"]:
            att[o["cls"]].append((o["attr"], P[j]["attr"], P[j]["attr_p"]))
    res["attr"] = att
    ok_all = [(g == q) for v in att.values() for g, q, _ in v]
    res["overall"]["attr_acc"] = float(np.mean(ok_all)) if ok_all else np.nan
    return res


# ---------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", SEQ)


def heat(ax, M, rows, cols, title=""):
    cm = seq_cmap()
    vmax = max(M.max(), 1)
    ax.imshow(M, cmap=cm, vmin=0, vmax=vmax, aspect="equal")
    for (r, c), v in np.ndenumerate(M):
        if v:
            ax.text(c, r, f"{int(v)}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title, loc="left")


def fig_per_class(plt, res, out):
    rows = [r for r in res["per_class"] if r["n_gt"] > 0]
    rows.sort(key=lambda r: -r["n_gt"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(SINGLE, 0.32 * n + 0.8))
    h = 0.26
    for k, (key, col, lab) in enumerate([("precision", BLUE, "Precision"),
                                         ("recall", ORANGE, "Recall"), ("F1", AQUA, "F1")]):
        y = np.arange(n) + (k - 1) * h
        v = np.nan_to_num([r[key] for r in rows])
        ax.barh(y, v, height=h - 0.03, color=col, label=lab, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n), [f"{r['class']} ({r['n_gt']})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Score")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1, columnspacing=1)
    save(fig, out, "fig1_per_class_prf")


def fig_pr(plt, G, P, iou, res, out, thr):
    rows = [r for r in res["per_class"] if r["n_gt"] > 0]
    rows.sort(key=lambda r: -r["n_gt"])
    n = len(rows)
    nc = min(4, n)
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(DOUBLE, 1.55 * nr + 0.2), squeeze=False,
                            sharex=True, sharey=True)
    for k, r in enumerate(rows):
        ax = axs[k // nc][k % nc]
        ap, curve = average_precision(G, P, iou, r["class_id"], thr)
        if curve is not None:
            rec, prec = curve
            ax.step(np.r_[0, rec], np.r_[prec[0] if len(prec) else 1, prec], where="post",
                    color=BLUE, linewidth=1.6)
        ax.set_title(f"{r['class']}  (n={r['n_gt']})", loc="left")
        ax.text(0.04, 0.06, f"AP = {ap:.2f}", transform=ax.transAxes, color=INK2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.set_axisbelow(True)
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("Recall")
    for row in axs:
        row[0].set_ylabel("Precision")
    fig.tight_layout()
    save(fig, out, "fig2_pr_curves")


def fig_confusion(plt, G, P, iou, dist, a, out, csvdir):
    g2p, p2g = greedy(G, P, iou, dist, "iou", a.iou, class_aware=False)
    gcls = sorted({o["cls"] for o in G})                            # rows: classes with GT
    cls = gcls + sorted({o["cls"] for o in P} - set(gcls))          # columns: same order first
    ix = {c: k for k, c in enumerate(cls)}
    iy = {c: k for k, c in enumerate(gcls)}
    M = np.zeros((len(gcls) + 1, len(cls) + 1), int)
    for i, o in enumerate(G):
        j = g2p[i]
        M[iy[o["cls"]], ix[P[j]["cls"]] if j >= 0 else len(cls)] += 1
    for j, o in enumerate(P):
        if p2g[j] < 0:
            M[len(gcls), ix[o["cls"]]] += 1
    rows = [CLASSES[c] for c in gcls] + ["False alarm"]
    cols = [CLASSES[c] for c in cls] + ["Missed"]
    with open(os.path.join(csvdir, "confusion_objects.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ predicted"] + cols)
        for r, line in zip(rows, M):
            w.writerow([r] + list(line))
    cell = min(0.36, (DOUBLE - 1.6) / len(cols))
    fig, ax = plt.subplots(figsize=(cell * len(cols) + 1.6, cell * len(rows) + 1.3))
    heat(ax, M, rows, cols)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    save(fig, out, "fig3_confusion")
    return M, rows, cols


def fig_localisation(plt, G, P, iou, dist, res, out):
    g2p = res["g2p"]
    cls = [r["class_id"] for r in sorted(res["per_class"], key=lambda r: -r["n_gt"]) if r["TP"] > 0]
    if not cls:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 0.3 * len(cls) + 1.0), sharey=True)
    rng = np.random.default_rng(0)
    for k, c in enumerate(cls):
        idx = [i for i, o in enumerate(G) if o["cls"] == c and g2p[i] >= 0]
        for ax, vals in ((axs[0], [dist[i, g2p[i]] * 100 for i in idx]),
                         (axs[1], [iou[i, g2p[i]] for i in idx])):
            y = k + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(vals, y, s=9, color=BLUE, alpha=0.75, linewidths=0)
            med = np.median(vals)
            ax.plot([med, med], [k - 0.32, k + 0.32], color=INK, linewidth=1.4)
    axs[0].set_yticks(range(len(cls)), [CLASSES[c] for c in cls])
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Centre distance to ground truth (cm)")
    axs[1].set_xlabel("Instance mask IoU")
    axs[1].set_xlim(0, 1)
    axs[0].set_xlim(left=0)
    for ax in axs:
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.text(0.995, 0.995, "dots: matched objects   bar: median", ha="right", va="top",
             color=MUTED, fontsize=6.5)
    save(fig, out, "fig4_localisation")


def fig_attributes(plt, res, out, disp):
    att = {c: v for c, v in res["attr"].items() if v}
    if not att:
        return
    fig, axs = plt.subplots(1, len(att), figsize=(min(DOUBLE, 2.4 * len(att) + 0.4), 2.2),
                            squeeze=False)
    for ax, (c, v) in zip(axs[0], sorted(att.items())):
        labs = sorted({g for g, _, _ in v})
        pl = sorted({q for _, q, _ in v if q} - set(labs))
        cols = labs + pl + (["none"] if any(not q for _, q, _ in v) else [])
        M = np.zeros((len(labs), len(cols)), int)
        for g, q, _ in v:
            M[labs.index(g), cols.index(q if q else "none")] += 1
        acc = np.mean([g == q for g, q, _ in v])
        heat(ax, M, [disp.get(x, x) for x in labs], [disp.get(x, x) for x in cols],
             f"{CLASSES[c]}: {acc * 100:.0f}% of {len(v)}")
        ax.set_xlabel("Predicted")
    axs[0][0].set_ylabel("Ground truth")
    fig.tight_layout()
    save(fig, out, "fig5_attributes")


def consistency_sweep(G, P, iou, dist, pts, a):
    out = []
    for t in np.round(np.arange(0, 1.0001, 0.1), 2):
        keep = np.array([not (o["seen_by"] >= 2 and np.isfinite(o["consistency"])
                              and o["consistency"] < t) for o in P])
        r = evaluate(G, P, iou, dist, pts, a, keep)["overall"]
        out.append({"threshold": t, "kept": int(keep.sum()), "TP": r["TP"], "FP": r["FP"],
                    "FN": r["FN"], "precision": r["precision"], "recall": r["recall"],
                    "F1": r["F1"]})
    return out


def fig_sweep(plt, sweep, out):
    t = [s["threshold"] for s in sweep]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    ends = []
    for key, col, lab in (("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"),
                          ("F1", AQUA, "F1")):
        v = [s[key] for s in sweep]
        ax.plot(t, v, color=col, linewidth=2, marker="o", markersize=3.5, label=lab)
        ends.append([np.nan_to_num(v[-1]), lab])
    ends.sort(key=lambda e: e[0])                       # spread end labels >= 0.07 apart
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.07)
    for y, lab in ends:
        ax.text(t[-1] + 0.03, y, lab, color=INK2, va="center", fontsize=7)
    best = max(sweep, key=lambda s: (np.nan_to_num(s["F1"]), -s["threshold"]))
    ax.axvline(best["threshold"], color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(best["threshold"], 0.02, f" best F1 at {best['threshold']:.1f}", color=INK2,
            fontsize=6.5, va="bottom")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, max(1.02, ends[-1][0] + 0.04))
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Minimum multi-view consistency (detected / saw)")
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1)
    save(fig, out, "fig6_consistency_sweep")


def range_table(G, res, stations):
    if stations is None:
        return None
    S = stations
    edges = [0, 2, 4, 6, 8, 10, np.inf]
    rows = []
    d = np.array([np.min(np.linalg.norm(S - o["centre"], axis=1)) for o in G])
    for o, dd in zip(G, d):
        o["nearest_station_m"] = float(dd)
    hit = res["g2p"] >= 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum():
            rows.append({"range": f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f">{lo:g} m",
                         "n_gt": int(m.sum()), "TP": int(hit[m].sum()),
                         "recall": float(hit[m].mean())})
    return rows


def fig_range(plt, rows, out):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    x = np.arange(len(rows))
    v = [r["recall"] for r in rows]
    ax.bar(x, v, width=0.6, color=BLUE, edgecolor="white")
    for k, r in enumerate(rows):
        ax.text(k, v[k] + 0.02, f"{r['TP']}/{r['n_gt']}", ha="center", va="bottom",
                fontsize=6.5, color=INK2)
    ax.set_xticks(x, [r["range"] for r in rows])
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Distance from object to the nearest scan station")
    ax.set_ylabel("Recall")
    save(fig, out, "fig7_recall_vs_range")


def fig_errors(plt, fn_t, fp_t, out):
    fn_cats = ["missed", "poor localisation", "wrong class"]
    fp_cats = ["background", "poor localisation", "duplicate", "wrong class"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 1.7))
    for ax, d, cats, title in ((axs[0], fn_t, fn_cats, "False negatives"),
                               (axs[1], fp_t, fp_cats, "False positives")):
        vals = [sum(v == c for v in d.values()) for c in cats]
        ax.barh(range(len(cats)), vals, height=0.55, color=BLUE, edgecolor="white")
        for k, v in enumerate(vals):
            ax.text(v + 0.1, k, str(v), va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(cats)), cats)
        ax.invert_yaxis()
        ax.set_title(f"{title} ({len(d)})", loc="left")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(vals + [1]) * 1.2)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Objects")
    fig.tight_layout()
    save(fig, out, "fig8_error_types")


def fig_ablation(plt, abl, out):
    if len(abl) < 2:
        return
    fig, ax = plt.subplots(figsize=(min(DOUBLE, 1.0 * len(abl) + 1.6), 2.4))
    x = np.arange(len(abl))
    w = 0.26
    for k, (key, col, lab) in enumerate((("precision", BLUE, "Precision"),
                                         ("recall", ORANGE, "Recall"), ("F1", AQUA, "F1"))):
        v = np.nan_to_num([r[key] for r in abl])
        ax.bar(x + (k - 1) * w, v, width=w - 0.03, color=col, label=lab, edgecolor="white")
    for k, r in enumerate(abl):
        ax.text(k + w, np.nan_to_num(r["F1"]) + 0.02, f"{r['F1']:.2f}", ha="center",
                fontsize=6.5, color=INK2)
    ax.set_xticks(x, [r["run"] for r in abl], rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1)
    save(fig, out, "fig9_ablation")


def assign_floors(G, P, stations, st_floor):
    """Objects without a floor in their CSV get the floor of the nearest station (or '1')."""
    for o in G + P:
        if not o["floor"]:
            if stations is not None and st_floor is not None:
                o["floor"] = st_floor[int(np.argmin(np.linalg.norm(stations - o["centre"], axis=1)))]
            else:
                o["floor"] = "1"


def floor_bands(G, P, stations, st_floor):
    """floor -> (z_lo, z_hi) of the wall band to draw on its map. From the scanner heights
    of that floor's stations (tripod ~1.5 m above the floor) when available, otherwise
    halfway between the floors' object heights."""
    floors = sorted({o["floor"] for o in G + P}, key=lambda f: (len(f), f))
    out = {}
    if stations is not None and st_floor is not None:
        for f in floors:
            z = stations[np.array([sf == f for sf in st_floor]), 2]
            if len(z):
                zs = float(np.median(z))
                out[f] = (zs - 0.9, zs + 0.7)
    missing = [f for f in floors if f not in out]
    if missing:
        zmed = {f: np.median([o["centre"][2] for o in G + P if o["floor"] == f]) for f in floors}
        order = sorted(floors, key=lambda f: zmed[f])
        for k, f in enumerate(order):
            if f in out:
                continue
            lo = (zmed[order[k - 1]] + zmed[f]) / 2 if k else -np.inf
            hi = (zmed[f] + zmed[order[k + 1]]) / 2 if k + 1 < len(order) else np.inf
            out[f] = (lo, hi, "slice")
    return out


def fig_map(plt, case, clouds, G, P, res, fn_t, fp_t, out, floor=None, band_z=None,
            max_pts=3_000_000):
    xyz = clouds["xyz"]
    if band_z is not None:
        in_z = (xyz[:, 2] > band_z[0]) & (xyz[:, 2] < band_z[1])
        cand = np.flatnonzero(in_z)
    else:
        cand = np.arange(len(xyz))
    sel = np.random.default_rng(0).choice(cand, min(len(cand), max_pts), replace=False)
    X = xyz[sel]
    if band_z is None or len(band_z) == 3:            # no scanner heights: find the slabs
        z0, z1 = np.percentile(X[:, 2], 2), np.percentile(X[:, 2], 98)
        band = X[(X[:, 2] > z0 + 0.6) & (X[:, 2] < z1 - 0.4)]          # walls, not slabs
        if len(band) > 1000:
            X = band
    lo, hi = X[:, :2].min(axis=0), X[:, :2].max(axis=0)
    span = hi - lo
    w = DOUBLE
    h = min(9.0, max(2.0, w * span[1] / max(span[0], 1e-6)))
    fig, ax = plt.subplots(figsize=(w, h + 0.4))
    bins = (int(span[0] / 0.05) + 1, int(span[1] / 0.05) + 1)
    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    ax.imshow(np.log1p(H.T), origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]),
              cmap="Greys", vmax=np.log1p(np.percentile(H[H > 0], 95)) if (H > 0).any() else 1,
              alpha=0.6, interpolation="nearest")
    gi = [i for i, o in enumerate(G) if o["case"] == case and (floor is None or o["floor"] == floor)]
    pj = [j for j, o in enumerate(P) if o["case"] == case and (floor is None or o["floor"] == floor)]
    groups = {"Correct": ([G[i]["centre"] for i in gi if res["g2p"][i] >= 0], "o", BLUE, BLUE),
              "Wrong class": ([G[i]["centre"] for i in gi if fn_t.get(i) == "wrong class"],
                              "D", "none", INK2),
              "Missed": ([G[i]["centre"] for i in gi if i in fn_t and fn_t[i] != "wrong class"],
                         "X", ORANGE, ORANGE),
              "False positive": ([P[j]["centre"] for j in pj if j in fp_t and
                                  fp_t[j] != "wrong class"], "^", AQUA, INK)}
    for lab, (pts, mk, fc, ec) in groups.items():
        if pts:
            Q = np.array(pts)
            ax.scatter(Q[:, 0], Q[:, 1], marker=mk, s=26, facecolors=fc, edgecolors=ec,
                       linewidths=0.9, label=f"{lab} ({len(pts)})", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.3,
              markerscale=1.0)
    save(fig, out, f"map_{case}" + (f"_floor{floor}" if floor is not None else ""))


# ---------------------------------------------------------------- tables
def fmt(v, d=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v * 100:.1f}" if pct else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def write_csv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else
                            ("" if isinstance(v, float) else v)) for k, v in r.items()})


def latex_table(caption, label, header, rows, align):
    s = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
         f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            s.append("\\midrule")
        else:
            s.append(" & ".join(str(x) for x in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(s)


def write_error_cloud(path, clouds, G, P, res, case):
    xyz, gi, pi = clouds["xyz"], clouds["gi"], clouds["pi"]
    # map case-local ids to matched pairs
    gid = {G[i]["id"]: i for i in range(len(G)) if G[i]["case"] == case}
    pid = {P[j]["id"]: j for j in range(len(P)) if P[j]["case"] == case}
    match_of_g = {G[i]["id"]: (P[res["g2p"][i]]["id"] if res["g2p"][i] >= 0 else -1) for i in gid.values()}
    m = (gi > 0) | (pi > 0)
    idx = np.flatnonzero(m)
    et = np.zeros(len(idx), np.int32)
    g, p = gi[idx], pi[idx]
    paired = np.array([match_of_g.get(int(a), -1) for a in g]) if len(g) else np.zeros(0)
    tp = (g > 0) & (p > 0) & (paired == p)
    et[tp] = 1
    et[(p > 0) & ~tp] = 2
    et[(g > 0) & ~tp & (et == 0)] = 3
    col = {1: (42, 120, 214), 2: (27, 175, 122), 3: (235, 104, 52)}
    rgb = np.array([col[e] for e in et], np.uint8).reshape(-1, 3)
    dt = [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
          ("error_type", "<i4"), ("gt_instance", "<i4"), ("pred_instance", "<i4")]
    a = np.empty(len(idx), dtype=dt)
    a["x"], a["y"], a["z"] = xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    a["error_type"], a["gt_instance"], a["pred_instance"] = et, g, p
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(idx)}\n"
            + "".join(f"property {t} {n}\n" for n, t in
                      (("x", "double"), ("y", "double"), ("z", "double"), ("red", "uchar"),
                       ("green", "uchar"), ("blue", "uchar"), ("error_type", "int"),
                       ("gt_instance", "int"), ("pred_instance", "int")))
            + "end_header\n")
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(a.tobytes())


# ---------------------------------------------------------------- main
def parse_pairs(items, what):
    out = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"{what}: expected CASE=FOLDER, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", nargs="+", required=True, metavar="CASE=GT_DIR")
    ap.add_argument("--run", nargs="+", action="append", required=True,
                    metavar="NAME CASE=PRED_DIR", help="repeat for each run; first = main")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stations", help="stations.csv (x, y, z of the scanners)")
    ap.add_argument("--iou", type=float, default=0.25,
                    help="point IoU for a match (default 0.25, the usual 3D instance threshold)")
    ap.add_argument("--dist", type=float, default=0.5,
                    help="centre distance (m) for the distance-based criterion (default 0.5)")
    ap.add_argument("--min-prob", type=float, default=0.8,
                    help="'confident' attributes: attribute_p >= this (default 0.8)")
    ap.add_argument("--attr-alias", nargs="*", default=[], metavar="FROM=TO",
                    help="map attribute names, e.g. foampowder=foam (compared after removing "
                         "case, spaces, punctuation and the word 'direction')")
    ap.add_argument("--nn-tol", type=float, default=0.003,
                    help="nearest-neighbour tolerance (m) if the point orders differ")
    ap.add_argument("--no-maps", action="store_true", help="skip the floor-plan maps")
    a = ap.parse_args()

    out = a.out_dir
    csvdir = os.path.join(out, "csv")
    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(os.path.join(out, "clouds"), exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    alias = {norm(k): norm(v) for k, v in (x.split("=", 1) for x in a.attr_alias)}
    gt_dirs = parse_pairs(a.gt, "--gt")
    runs = []
    for r in a.run:
        name, pairs = r[0], parse_pairs(r[1:], f"--run {r[0]}")
        unknown = set(pairs) - set(gt_dirs)
        if unknown:
            sys.exit(f"--run {name}: cases {sorted(unknown)} have no --gt")
        runs.append((name, pairs))
    stations = st_floor = None
    if a.stations:
        st = read_csv(a.stations)
        stations = np.array([[float(s["x"]), float(s["y"]), float(s["z"])] for s in st])
        st_floor = [str(s.get("floor", "")).strip() for s in st] if "floor" in st[0] else None

    plt = setup_mpl()
    ablation, main = [], None
    for k_run, (name, pairs) in enumerate(runs):
        log(f"\n=== run '{name}'" + (" (main)" if k_run == 0 else ""))
        cases = []
        for case, pdir in pairs.items():
            log(f"  {case}: GT {gt_dirs[case]}  |  prediction {pdir}")
            G, P, iou, pts, clouds = load_case(case, gt_dirs[case], pdir, a, alias, log)
            log(f"     {len(G)} GT objects, {len(P)} predicted objects")
            cases.append({"case": case, "G": G, "P": P, "iou": iou, "pts": pts,
                          "clouds": clouds if k_run == 0 else None})
        G, P, iou, dist, pts = combine(cases)
        res = evaluate(G, P, iou, dist, pts, a)
        o = res["overall"]
        log(f"  objects: P {fmt(o['precision'])}  R {fmt(o['recall'])}  F1 {fmt(o['F1'])}  "
            f"(TP {o['TP']}, FP {o['FP']}, FN {o['FN']})  mAP@0.25 {fmt(o['mAP25'])}  "
            f"mAP@0.5 {fmt(o['mAP50'])}  point mIoU {fmt(o['mIoU_points'])}  "
            f"median centre error {fmt(o['loc_median_m'] * 100, 1)} cm  "
            f"attribute acc. {fmt(o['attr_acc'])}")
        ablation.append({"run": name, **o})
        if k_run == 0:
            main = (cases, G, P, iou, dist, pts, res)

    cases, G, P, iou, dist, pts, res = main
    o = res["overall"]
    assign_floors(G, P, stations, st_floor)

    # ---- criteria table
    crit = []
    for lab, c_, thr, aware in ((f"IoU >= {a.iou:g}, class-aware", "iou", a.iou, True),
                                ("IoU >= 0.5, class-aware", "iou", 0.5, True),
                                (f"centre <= {a.dist:g} m, class-aware", "dist", a.dist, True),
                                (f"IoU >= {a.iou:g}, class-agnostic", "iou", a.iou, False)):
        g2p, p2g = greedy(G, P, iou, dist, c_, thr, aware)
        tp = int((g2p >= 0).sum())
        p, r, f = prf(tp, len(P) - tp, len(G) - tp)
        crit.append({"criterion": lab, "TP": tp, "FP": len(P) - tp, "FN": len(G) - tp,
                     "precision": p, "recall": r, "F1": f})

    # ---- error types, per-object tables
    fn_t, fp_t = error_types(G, P, iou, res["g2p"], res["p2g"], a.iou)
    rng_rows = range_table(G, res, stations)
    obj_g = []
    for i, go in enumerate(G):
        j = res["g2p"][i]
        best = int(np.argmax(iou[i])) if len(P) else -1
        obj_g.append({"case": go["case"], "gt_id": go["id"], "class": CLASSES[go["cls"]],
                      "n_points": go["n"], "result": "TP" if j >= 0 else f"FN: {fn_t[i]}",
                      "pred_id": P[j]["id"] if j >= 0 else "",
                      "iou": iou[i, j] if j >= 0 else (iou[i, best] if best >= 0 else np.nan),
                      "best_overlap_pred": P[best]["id"] if best >= 0 and iou[i, best] > 0 else "",
                      "best_overlap_class": CLASSES[P[best]["cls"]] if best >= 0 and iou[i, best] > 0 else "",
                      "centre_error_m": dist[i, j] if j >= 0 else np.nan,
                      "centre_error_xy_m": float(np.linalg.norm(go["centre"][:2] - P[j]["centre"][:2])) if j >= 0 else np.nan,
                      "centre_error_z_m": float(P[j]["centre"][2] - go["centre"][2]) if j >= 0 else np.nan,
                      "gt_attribute": go["attr_raw"],
                      "pred_attribute": P[j]["attr_raw"] if j >= 0 else "",
                      "attribute_correct": (go["attr"] == P[j]["attr"]) if j >= 0 and go["attr"] else "",
                      "nearest_station_m": go.get("nearest_station_m", np.nan),
                      "x": go["centre"][0], "y": go["centre"][1], "z": go["centre"][2]})
    obj_p = []
    for j, po in enumerate(P):
        i = res["p2g"][j]
        obj_p.append({"case": po["case"], "pred_id": po["id"], "class": CLASSES[po["cls"]],
                      "conf": po["conf"], "n_views": po["n_views"], "seen_by_near": po["seen_by"],
                      "consistency": po["consistency"], "n_points": po["n"],
                      "result": "TP" if i >= 0 else f"FP: {fp_t[j]}",
                      "gt_id": G[i]["id"] if i >= 0 else "",
                      "iou": iou[i, j] if i >= 0 else (float(iou[:, j].max()) if len(G) else np.nan),
                      "x": po["centre"][0], "y": po["centre"][1], "z": po["centre"][2]})

    # ---- multi-view analyses
    sweep = consistency_sweep(G, P, iou, dist, pts, a)
    tp_cons = [p_["consistency"] for p_, r in zip(P, res["p2g"]) if r >= 0 and np.isfinite(p_["consistency"])]
    fp_cons = [p_["consistency"] for p_, r in zip(P, res["p2g"]) if r < 0 and np.isfinite(p_["consistency"])]
    single_tp = sum(1 for p_, r in zip(P, res["p2g"]) if r >= 0 and p_["n_views"] == 1)
    single_fp = sum(1 for p_, r in zip(P, res["p2g"]) if r < 0 and p_["n_views"] == 1)

    # ---- attributes table
    att_rows = []
    for c, v in sorted(res["attr"].items()):
        acc = np.mean([g == q for g, q, _ in v])
        conf_v = [(g, q) for g, q, p_ in v if np.isfinite(p_) and p_ >= a.min_prob]
        att_rows.append({"class": CLASSES[c], "n_matched_with_gt_attribute": len(v),
                         "accuracy": acc, "n_confident": len(conf_v),
                         "coverage_confident": len(conf_v) / len(v),
                         "accuracy_confident": np.mean([g == q for g, q in conf_v]) if conf_v else np.nan,
                         "n_no_prediction": sum(1 for _, q, _ in v if not q)})

    # ---- write CSVs
    write_csv(os.path.join(csvdir, "detection_per_class.csv"), res["per_class"])
    write_csv(os.path.join(csvdir, "detection_overall.csv"), [o])
    write_csv(os.path.join(csvdir, "criteria.csv"), crit)
    write_csv(os.path.join(csvdir, "attributes.csv"), att_rows)
    write_csv(os.path.join(csvdir, "consistency_sweep.csv"), sweep)
    write_csv(os.path.join(csvdir, "ablation.csv"), ablation,
              ["run", "n_gt", "n_pred", "TP", "FP", "FN", "precision", "recall", "F1",
               "macro_F1", "mAP25", "mAP50", "mIoU_points", "mask_iou_mean", "loc_median_m",
               "loc_mean_m", "loc_rmse_m", "attr_acc"])
    write_csv(os.path.join(csvdir, "objects_gt.csv"), obj_g)
    write_csv(os.path.join(csvdir, "objects_pred.csv"), obj_p)
    if rng_rows:
        write_csv(os.path.join(csvdir, "recall_vs_range.csv"), rng_rows)
    err_rows = ([{"type": "FN", "category": c, "n": sum(v == c for v in fn_t.values())}
                 for c in ("missed", "poor localisation", "wrong class")] +
                [{"type": "FP", "category": c, "n": sum(v == c for v in fp_t.values())}
                 for c in ("background", "poor localisation", "duplicate", "wrong class")])
    write_csv(os.path.join(csvdir, "error_types.csv"), err_rows)
    per_case = []
    for c in cases:
        gi_ = [i for i, g_ in enumerate(G) if g_["case"] == c["case"]]
        pj_ = [j for j, p_ in enumerate(P) if p_["case"] == c["case"]]
        tp = sum(res["g2p"][i] >= 0 for i in gi_)
        p, r, f = prf(tp, len(pj_) - tp, len(gi_) - tp)
        per_case.append({"case": c["case"], "n_gt": len(gi_), "n_pred": len(pj_), "TP": tp,
                         "FP": len(pj_) - tp, "FN": len(gi_) - tp, "precision": p,
                         "recall": r, "F1": f})
    write_csv(os.path.join(csvdir, "per_case.csv"), per_case)
    floors = sorted({o_["floor"] for o_ in G + P}, key=lambda f: (len(f), f))
    per_floor = []
    for fl in floors:
        gi_ = [i for i, g_ in enumerate(G) if g_["floor"] == fl]
        pj_ = [j for j, p_ in enumerate(P) if p_["floor"] == fl]
        tp = sum(res["g2p"][i] >= 0 for i in gi_)
        fp = sum(res["p2g"][j] < 0 for j in pj_)
        p, r, f = prf(tp, fp, len(gi_) - tp)
        d_ = [dist[i, res["g2p"][i]] for i in gi_ if res["g2p"][i] >= 0]
        per_floor.append({"floor": fl, "n_gt": len(gi_), "n_pred": len(pj_), "TP": tp, "FP": fp,
                          "FN": len(gi_) - tp, "precision": p, "recall": r, "F1": f,
                          "loc_median_m": float(np.median(d_)) if d_ else np.nan})
    write_csv(os.path.join(csvdir, "per_floor.csv"), per_floor)

    # ---- figures
    fig_per_class(plt, res, out)
    fig_pr(plt, G, P, iou, res, out, a.iou)
    M, mrows, mcols = fig_confusion(plt, G, P, iou, dist, a, out, csvdir)
    fig_localisation(plt, G, P, iou, dist, res, out)
    disp = {}
    for o_ in G + P:
        k_ = norm_attr(o_["attr_raw"], alias)
        if k_ and k_ not in disp:
            disp[k_] = re.sub(r"(?i)^direction[\s_-]*", "", str(o_["attr_raw"]).strip())
    fig_attributes(plt, res, out, disp)
    fig_sweep(plt, sweep, out)
    fig_range(plt, rng_rows, out)
    fig_errors(plt, fn_t, fp_t, out)
    fig_ablation(plt, ablation, out)
    bands = floor_bands(G, P, stations, st_floor)
    for c in cases:
        if not a.no_maps:
            c_floors = sorted({o_["floor"] for o_ in G + P if o_["case"] == c["case"]},
                              key=lambda f: (len(f), f))
            if len(c_floors) > 1:              # several floors in one cloud: one map each
                for fl in c_floors:
                    fig_map(plt, c["case"], c["clouds"], G, P, res, fn_t, fp_t, out,
                            floor=fl, band_z=bands.get(fl))
            else:
                fig_map(plt, c["case"], c["clouds"], G, P, res, fn_t, fp_t, out)
        write_error_cloud(os.path.join(out, "clouds", f"errors_{c['case']}.ply"), c["clouds"],
                          G, P, res, c["case"])

    # ---- LaTeX
    tex = []
    body = []
    for r in sorted(res["per_class"], key=lambda r: -r["n_gt"]):
        body.append([r["class"], r["n_gt"], r["TP"], r["FP"], r["FN"], fmt(r["precision"]),
                     fmt(r["recall"]), fmt(r["F1"]), fmt(r["AP25"]), fmt(r["AP50"]),
                     fmt(r["loc_median_m"] * 100 if np.isfinite(r["loc_median_m"]) else np.nan, 1),
                     fmt(r["point_iou"])])
    body.append("MID")
    body.append(["Overall", o["n_gt"], o["TP"], o["FP"], o["FN"], fmt(o["precision"]),
                 fmt(o["recall"]), fmt(o["F1"]), fmt(o["mAP25"]), fmt(o["mAP50"]),
                 fmt(o["loc_median_m"] * 100 if np.isfinite(o["loc_median_m"]) else np.nan, 1),
                 fmt(o["mIoU_points"])])
    tex.append(latex_table(
        f"Object-level results against the 3D ground truth (point IoU $\\geq$ {a.iou:g}, "
        "class-aware). AP: all-point interpolated average precision; loc.: median distance "
        "between box centres of matched objects; pt IoU: point-level IoU. Overall: micro-"
        "averaged P/R/F1, mean AP and mIoU over classes.",
        "tab:detection", ["Class", "GT", "TP", "FP", "FN", "P", "R", "F1", "AP$_{25}$",
                          "AP$_{50}$", "Loc. (cm)", "Pt IoU"], body, "lrrrrrrrrrrr"))
    tex.append(latex_table("Overall results under different matching criteria.", "tab:criteria",
                           ["Criterion", "TP", "FP", "FN", "P", "R", "F1"],
                           [[c_["criterion"].replace(">=", "$\\geq$").replace("<=", "$\\leq$"),
                             c_["TP"], c_["FP"], c_["FN"], fmt(c_["precision"]),
                             fmt(c_["recall"]), fmt(c_["F1"])] for c_ in crit], "lrrrrrr"))
    if len(per_floor) > 1:
        tex.append(latex_table("Object-level results per floor.", "tab:floors",
                               ["Floor", "GT", "TP", "FP", "FN", "P", "R", "F1", "Loc. (cm)"],
                               [[r["floor"], r["n_gt"], r["TP"], r["FP"], r["FN"],
                                 fmt(r["precision"]), fmt(r["recall"]), fmt(r["F1"]),
                                 fmt(r["loc_median_m"] * 100 if np.isfinite(r["loc_median_m"]) else np.nan, 1)]
                                for r in per_floor], "lrrrrrrrr"))
    tex.append(latex_table("Error types of the false negatives and false positives.",
                           "tab:errors", ["", "Category", "Objects"],
                           [[r["type"], r["category"], r["n"]] for r in err_rows], "llr"))
    if att_rows:
        tex.append(latex_table(
            f"Attribute accuracy on correctly detected objects whose ground truth has the "
            f"attribute. Confident: summed stage-2 probability $\\geq$ {a.min_prob:g}.",
            "tab:attributes", ["Class", "n", "Accuracy", "Confident (coverage)", "Accuracy (confident)"],
            [[r["class"], r["n_matched_with_gt_attribute"], fmt(r["accuracy"], pct=True),
              f"{r['n_confident']} ({fmt(r['coverage_confident'], pct=True)}\\%)",
              fmt(r["accuracy_confident"], pct=True)] for r in att_rows], "lrrrr"))
    if len(ablation) > 1:
        tex.append(latex_table("Ablation: overall object-level results of each configuration.",
                               "tab:ablation",
                               ["Configuration", "P", "R", "F1", "mAP$_{25}$", "mAP$_{50}$",
                                "Pt mIoU", "Loc. (cm)", "Attr. acc."],
                               [[r["run"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["F1"]),
                                 fmt(r["mAP25"]), fmt(r["mAP50"]), fmt(r["mIoU_points"]),
                                 fmt(r["loc_median_m"] * 100 if np.isfinite(r["loc_median_m"]) else np.nan, 1),
                                 fmt(r["attr_acc"])] for r in ablation], "lrrrrrrrr"))
    with open(os.path.join(out, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("% needs \\usepackage{booktabs}\n" + "\n".join(tex))

    # ---- summary
    best = max(sweep, key=lambda s: (np.nan_to_num(s["F1"]), -s["threshold"]))
    base = sweep[0]
    s = [f"# Evaluation against the 3D ground truth", "",
         f"Main run: **{runs[0][0]}**; cases: " + ", ".join(
             f"{c['case']} ({pc['n_gt']} GT / {pc['n_pred']} predicted)"
             for c, pc in zip(cases, per_case)),
         f"Matching: greedy by confidence, point IoU >= {a.iou:g}, class-aware.", "",
         "## Headline",
         f"- {o['TP']} of {o['n_gt']} ground-truth objects found with the right class "
         f"(recall {fmt(o['recall'])}); {o['FP']} false positive(s) among {o['n_pred']} "
         f"predictions (precision {fmt(o['precision'])}); F1 {fmt(o['F1'])}, macro-F1 "
         f"{fmt(o['macro_F1'])}.",
         f"- mAP@0.25 {fmt(o['mAP25'])}, mAP@0.5 {fmt(o['mAP50'])}; point-level mIoU "
         f"{fmt(o['mIoU_points'])}; mean instance mask IoU of matched objects "
         f"{fmt(o['mask_iou_mean'])}.",
         f"- Localisation (box centres of matched objects): median "
         f"{fmt(o['loc_median_m'] * 100, 1)} cm, mean {fmt(o['loc_mean_m'] * 100, 1)} cm, "
         f"RMSE {fmt(o['loc_rmse_m'] * 100, 1)} cm.",
         f"- Attributes on matched objects: accuracy {fmt(o['attr_acc'], pct=True)} %" +
         "".join(f"; {r['class']} {fmt(r['accuracy'], pct=True)} % of "
                 f"{r['n_matched_with_gt_attribute']} (confident: "
                 f"{fmt(r['accuracy_confident'], pct=True)} % at "
                 f"{fmt(r['coverage_confident'], pct=True)} % coverage)" for r in att_rows) + ".",
         "", "## Per floor"] + [
        f"- floor {r['floor']}: {r['n_gt']} GT / {r['n_pred']} predicted, P {fmt(r['precision'])}, "
        f"R {fmt(r['recall'])}, F1 {fmt(r['F1'])}, median loc. "
        f"{fmt(r['loc_median_m'] * 100 if np.isfinite(r['loc_median_m']) else np.nan, 1)} cm"
        for r in per_floor] + [
         "", "## Matching criteria"] + [
        f"- {c_['criterion']}: P {fmt(c_['precision'])}, R {fmt(c_['recall'])}, F1 {fmt(c_['F1'])}"
        for c_ in crit] + [
        "", "## Errors",
        "- False negatives: " + ", ".join(f"{r['n']} {r['category']}" for r in err_rows if r["type"] == "FN"),
        "- False positives: " + ", ".join(f"{r['n']} {r['category']}" for r in err_rows if r["type"] == "FP"),
        "", "## Multi-view",
        f"- Single-view objects: {single_tp} correct, {single_fp} false positive.",
        f"- Consistency (detected / nearby stations that saw it): median "
        f"{fmt(float(np.median(tp_cons)) if tp_cons else np.nan)} for TP, "
        f"{fmt(float(np.median(fp_cons)) if fp_cons else np.nan)} for FP.",
        f"- Dropping objects below a consistency of {best['threshold']:.1f}: F1 "
        f"{fmt(base['F1'])} -> {fmt(best['F1'])} (P {fmt(base['precision'])} -> "
        f"{fmt(best['precision'])}, R {fmt(base['recall'])} -> {fmt(best['recall'])}). "
        f"Choose the threshold on a different building before reporting it as a result."]
    if rng_rows:
        s.append("- Recall by distance to the nearest station: " +
                 ", ".join(f"{r['range']} {fmt(r['recall'])} ({r['TP']}/{r['n_gt']})" for r in rng_rows))
    if len(ablation) > 1:
        s += ["", "## Ablation"] + [
            f"- {r['run']}: P {fmt(r['precision'])}, R {fmt(r['recall'])}, F1 {fmt(r['F1'])}, "
            f"mAP@0.25 {fmt(r['mAP25'])}, point mIoU {fmt(r['mIoU_points'])}, median loc. "
            f"{fmt(r['loc_median_m'] * 100, 1)} cm" for r in ablation]
    s += ["", "## Per class", "",
          "| class | GT | TP | FP | FN | P | R | F1 | AP@0.25 | loc. median (cm) |",
          "|---|---|---|---|---|---|---|---|---|---|"] + [
        f"| {r['class']} | {r['n_gt']} | {r['TP']} | {r['FP']} | {r['FN']} | {fmt(r['precision'])} | "
        f"{fmt(r['recall'])} | {fmt(r['F1'])} | {fmt(r['AP25'])} | "
        f"{fmt(r['loc_median_m'] * 100 if np.isfinite(r['loc_median_m']) else np.nan, 1)} |"
        for r in sorted(res["per_class"], key=lambda r: -r["n_gt"])]
    s += ["", "## Log", "```"] + lines + ["```"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")
    print(f"\nOutputs in {out}: summary.md, tables.tex, csv/, figures/, clouds/")


if __name__ == "__main__":
    sys.exit(main())
