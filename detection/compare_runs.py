"""
Paired comparison of two saved classifier runs (folders in cnn_runs/ with a
predictions.csv), e.g. the ResNet-18 baseline against YOLO26-cls.

Both runs must cover the same crops (same class folder). Compares the final
predictions (the seed ensemble when a run has several seeds):
  * accuracy overall, per source (video / 360 photo) and per location;
  * exact McNemar test on crops one run gets right and the other wrong
    (treats crops as independent -- optimistic, because crops from one
    building are related);
  * cluster bootstrap over whole locations (resamples buildings, not crops)
    for the difference in accuracy and balanced accuracy, with a 95% interval.
    This is the more honest test; report this one.

Usage:
    python compare_runs.py "...\\cnn_runs\\20261025-200659_baseline" ^
                           "...\\cnn_runs\\20261030-2200_yolo26n-cls"
    (add --out comparison.json to save the numbers)
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def load(run):
    p = Path(run) / "predictions.csv"
    if not p.is_file():
        sys.exit(f"No predictions.csv in {run}")
    with open(p, encoding="utf-8") as fh:
        return {r["crop"]: r for r in csv.DictReader(fh)}


def mcnemar_exact(n01, n10):
    """Two-sided exact binomial test on the discordant pairs."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def balanced_acc(y, p):
    cls = sorted(set(y))
    return float(np.mean([(p[y == c] == c).mean() for c in cls]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="reference run folder (e.g. the baseline)")
    ap.add_argument("run_b", help="run to compare with it")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save the numbers as JSON")
    args = ap.parse_args()

    A, B = load(args.run_a), load(args.run_b)
    if set(A) != set(B):
        only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
        sys.exit(f"The runs do not cover the same crops ({only_a} only in A, "
                 f"{only_b} only in B). Compare runs on the same class folder.")
    keys = sorted(A)
    y = np.array([A[k]["true"] for k in keys])
    pa = np.array([A[k]["predicted"] for k in keys])
    pb = np.array([B[k]["predicted"] for k in keys])
    loc = np.array([A[k]["location"] or A[k]["group"] for k in keys])
    src = np.array([A[k]["source"] for k in keys])
    na, nb = Path(args.run_a).name, Path(args.run_b).name
    print(f"A = {na}\nB = {nb}\n{len(keys)} crops\n")

    result = {"run_a": na, "run_b": nb, "n_crops": len(keys), "subsets": {}}
    print(f"{'subset':<20}{'n':>5}{'A':>8}{'B':>8}{'only A ok':>11}"
          f"{'only B ok':>11}{'McNemar p':>11}")
    subsets = [("all crops", np.ones(len(y), bool))]
    subsets += [(s, src == s) for s in ["own video", "own 360 photo",
                                        "own early frames", "other"]
                if (src == s).any()]
    for name, m in subsets:
        a_ok, b_ok = pa[m] == y[m], pb[m] == y[m]
        n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
        p = mcnemar_exact(n01, n10)
        print(f"{name:<20}{int(m.sum()):>5}{a_ok.mean():>8.3f}{b_ok.mean():>8.3f}"
              f"{n01:>11}{n10:>11}{p:>11.3f}")
        result["subsets"][name] = {"n": int(m.sum()), "acc_a": round(float(a_ok.mean()), 4),
                                   "acc_b": round(float(b_ok.mean()), 4),
                                   "only_a_correct": n01, "only_b_correct": n10,
                                   "mcnemar_p": round(p, 4)}

    rng = np.random.default_rng(args.seed)
    L = sorted(set(loc))
    idx = {l: np.where(loc == l)[0] for l in L}
    d_acc, d_bal = [], []
    for _ in range(args.n_boot):
        s = np.concatenate([idx[l] for l in rng.choice(L, len(L), replace=True)])
        d_acc.append((pb[s] == y[s]).mean() - (pa[s] == y[s]).mean())
        d_bal.append(balanced_acc(y[s], pb[s]) - balanced_acc(y[s], pa[s]))
    print(f"\nBootstrap over {len(L)} locations ({args.n_boot} samples), "
          f"difference B - A:")
    result["bootstrap"] = {"n_locations": len(L), "n_boot": args.n_boot}
    for name, d, obs in [("accuracy", np.array(d_acc),
                          (pb == y).mean() - (pa == y).mean()),
                         ("balanced accuracy", np.array(d_bal),
                          balanced_acc(y, pb) - balanced_acc(y, pa))]:
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {name:<18} {obs:+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]  "
              f"share of samples with B <= A: {np.mean(d <= 0):.3f}")
        result["bootstrap"][name] = {"observed": round(float(obs), 4),
                                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                     "p_b_not_better": round(float(np.mean(d <= 0)), 4)}

    print(f"\n{'location':<36}{'n':>5}{'A':>8}{'B':>8}")
    result["by_location"] = {}
    for l in L:
        m = loc == l
        a, b = float((pa[m] == y[m]).mean()), float((pb[m] == y[m]).mean())
        print(f"{l[:35]:<36}{int(m.sum()):>5}{a:>8.3f}{b:>8.3f}")
        result["by_location"][l] = {"n": int(m.sum()), "acc_a": round(a, 4),
                                    "acc_b": round(b, 4)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
