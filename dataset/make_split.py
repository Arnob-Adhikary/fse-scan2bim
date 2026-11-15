#!/usr/bin/env python3
"""
make_split.py - fixed train/val/test split for the combined fire-safety YOLO-seg dataset.

  Test : whole held-out locations (own data only). The script tries every
         combination of 1..N locations and keeps the one with the best class coverage.
  Val  : ~15% of own *videos* from the remaining locations (whole videos, so no
         near-duplicate frames/cube faces leak) + ~15% of public images.
         Clips name(1), name(2) ... of one recording count as one video; 360/smartphone
         photos taken < --photo-gap-min minutes apart count as one walk-through.
  Train: everything else, including own frames with no recognisable date/location.

Location = recording date found in the filename (e.g. 20240118). If one school was
recorded over several dates, pass --location-map with a CSV: date,location

Expects  <root>/images/<stem>.jpg  and  <root>/labels/<stem>.txt  (YOLO-seg format).

Writes to <root>:
  train.txt / val.txt / test.txt   image lists (./images/...), read directly by Ultralytics
  data_split.yaml                  dataset yaml pointing at those lists
  splits/                          reports: location stats, candidate ranking, class counts

Optional --drop-empty: images with no labels are left out of all splits (listed in
  splits/dropped_empty.txt). Use when unlabelled images may still contain equipment.

Optional --reuse-split <previous export folder>: keep the earlier split after correcting labels.
  Every image keeps its old split. Images that are new to the split (e.g. previously without
  labels) go to test if from a test location, to val if their video / walk-through is in val,
  otherwise to train. Reports: splits/reuse_changes.csv, splits/reuse_class_changes.csv.

Optional --export <new folder>: COPIES the files into per-split folders
  <export>/train/images   images + their X-AnyLabeling .json (same stem, same folder)
  <export>/train/labels   YOLO .txt labels
  (same for val/ and test/) + <export>/data.yaml + a copy of the reports.
  The original dataset folder is never changed. Needs free disk space ~ dataset size.

Run once, keep the outputs, and do not re-run with different settings later -
the test set must stay fixed across all experiments.

Example:
  python make_split.py --root "D:/fire_dataset"
  python make_split.py --root "D:/fire_dataset" --force-test 20240118,20240302
"""
import argparse
import csv
import itertools
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CLASS_NAMES = [
    "Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
    "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED",
    "Fire Call Point", "Fire Blanket", "Emergency Door Release", "Evacuation Chair",
]
NC = len(CLASS_NAMES)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="dataset folder containing images/ and labels/")
    p.add_argument("--public-prefixes", default="fireart_,firenet_",
                   help="comma-separated filename prefixes of public-dataset images")
    p.add_argument("--date-regex", default=r"(20\d{6})",
                   help="regex whose group 1 is the recording date in own filenames")
    p.add_argument("--location-map", default=None,
                   help="optional CSV with columns date,location")
    p.add_argument("--train-only", default=None,
                   help="comma-separated locations that are never used as test "
                        "(e.g. a mixed folder like 'Random smartphone image')")
    p.add_argument("--force-test", default=None,
                   help="comma-separated locations to use as test (skips the automatic choice)")
    p.add_argument("--max-locations", type=int, default=3, help="max locations in the test set")
    p.add_argument("--test-min", type=float, default=0.10, help="min share of own images in test")
    p.add_argument("--test-max", type=float, default=0.25, help="max share of own images in test")
    p.add_argument("--test-target", type=float, default=0.15, help="preferred share of own images")
    p.add_argument("--min-inst", type=int, default=20, help="target test instances per class")
    p.add_argument("--photo-gap-min", type=float, default=10,
                   help="360/smartphone photos of one location taken less than this many "
                        "minutes apart form one group for the val split (same walk-through)")
    p.add_argument("--max-class-share", type=float, default=0.45,
                   help="test may take at most this share of any class's total instances")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--public-val-frac", type=float, default=None,
                   help="share of public images put in val (default: same as --val-frac). "
                        "0 = all public images to train, so val is own data only")
    p.add_argument("--val-tries", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse-split", default=None,
                   help="previous export folder (with train/ val/ test/ images) or dataset folder "
                        "with train.txt/val.txt/test.txt: every image keeps its old split; new "
                        "images follow their location / video group (test location -> test, "
                        "group already in val -> val, otherwise train). No random choices.")
    p.add_argument("--drop-empty", action="store_true",
                   help="leave out images with no labels (empty or missing .txt) from all "
                        "splits; they are listed in splits/dropped_empty.txt")
    p.add_argument("--export", default=None,
                   help="optional NEW folder: copy each split into <export>/<split>/images "
                        "(images + X-AnyLabeling .json) and <export>/<split>/labels (YOLO .txt)")
    return p.parse_args()


def read_counts(label_path, bad_ids):
    c = Counter()
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if not 0 <= cls < NC:
                bad_ids.append((label_path.name, cls))
                continue
            c[cls] += 1
    return c


def video_id(stem):
    """Group key so all frames/cube faces of one video (or one 360 photo) stay together.
    Clips exported from one recording as name, name(1), name(2) ... are one video."""
    if "_frame_" in stem:
        return re.sub(r"\(\d+\)$", "", stem.split("_frame_")[0])
    m = re.search(r"20\d{6}_\d{6}", stem)  # 360 photos like 20240118_124305...
    return m.group(0) if m else stem


def load_old_split(path):
    """Return ({image file name: split}, {split: Counter of instances} or None)."""
    P = Path(path).resolve()
    old, old_counts = {}, None
    if (P / "train" / "images").is_dir():
        old_counts = {}
        for s in ("train", "val", "test"):
            d = P / s / "images"
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    old[f.name] = s
            c, ignore = Counter(), []
            for lp in (P / s / "labels").glob("*.txt"):
                c.update(read_counts(lp, ignore))
            old_counts[s] = c
    elif (P / "train.txt").exists():
        for s in ("train", "val", "test"):
            t = P / f"{s}.txt"
            if t.exists():
                for line in t.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        old[line.strip().replace("\\", "/").split("/")[-1]] = s
    else:
        sys.exit(f"--reuse-split: no train/images folder or train.txt found in {P}")
    if not old:
        sys.exit(f"--reuse-split: no images found in {P}")
    return old, old_counts


def csum(counters):
    total = Counter()
    for c in counters:
        total.update(c)
    return total


def main():
    a = get_args()
    root = Path(a.root).resolve()
    if a.export:
        exp0 = Path(a.export).resolve()
        if exp0.exists() and any(exp0.iterdir()):
            sys.exit(f"--export folder {exp0} is not empty - choose a new folder "
                     f"(this avoids mixing files from different split runs). Nothing was written.")
        if exp0 == root or root in exp0.parents:
            sys.exit("--export folder must be outside the dataset folder. Nothing was written.")
    img_dir, lbl_dir = root / "images", root / "labels"
    rep_dir = root / "splits"
    rep_dir.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(x.strip().lower() for x in a.public_prefixes.split(",") if x.strip())
    date_re = re.compile(a.date_regex)
    loc_map = {}
    if a.location_map:
        with open(a.location_map, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                d, L = (row.get("date") or "").strip(), (row.get("location") or "").strip()
                if d and L:
                    loc_map[d] = L
        if not loc_map:
            sys.exit(f"No date,location rows read from {a.location_map}")

    if not img_dir.is_dir():
        sys.exit(f"Folder not found: {img_dir}")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No images found in {img_dir}")

    # ---------- read everything ----------
    recs, bad_ids, missing = [], [], []
    for p in images:
        stem = p.stem
        lp = lbl_dir / f"{stem}.txt"
        if not lp.exists():
            missing.append(p.name)
        counts = read_counts(lp, bad_ids)
        if stem.lower().startswith(prefixes):
            kind, loc, vid = "public", None, None
        else:
            m = date_re.search(stem)
            if m:
                date = m.group(1)
                kind, loc, vid = "own", loc_map.get(date, date), video_id(stem)
            else:
                kind, loc, vid = "own_noloc", None, None
        recs.append(dict(path=p, kind=kind, loc=loc, vid=vid, counts=counts))

    dropped = []
    if a.drop_empty:
        dropped = [r for r in recs if not r["counts"]]
        recs = [r for r in recs if r["counts"]]
        by_loc = Counter(r["loc"] or ("public" if r["kind"] == "public" else "no-date")
                         for r in dropped)
        with open(rep_dir / "dropped_empty.txt", "w", encoding="utf-8") as f:
            f.write(f"# {len(dropped)} images without labels left out (--drop-empty)\n")
            for L, n in sorted(by_loc.items()):
                f.write(f"# {L}: {n}\n")
            f.write("\n".join(r["path"].name for r in dropped) + "\n")

    # photos (no _frame_): group shots taken close together into one walk-through session,
    # so neighbouring photos that show the same objects never end up in both train and val
    photos = defaultdict(list)
    for r in recs:
        if r["kind"] == "own" and "_frame_" not in r["path"].stem:
            m = re.search(r"(20\d{6})_(\d{6})", r["path"].stem)
            if m:
                t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                photos[r["loc"]].append((t, r))
    for L, items in photos.items():
        items.sort(key=lambda x: x[0])
        prev, sid = None, None
        for t, r in items:
            if prev is None or (t - prev).total_seconds() > a.photo_gap_min * 60:
                sid = f"photos_{t:%Y%m%d_%H%M%S}"
            r["vid"] = sid
            prev = t

    total = csum(r["counts"] for r in recs)
    own_all = [r for r in recs if r["kind"] != "public"]
    noloc = [r for r in recs if r["kind"] == "own_noloc"]
    public = [r for r in recs if r["kind"] == "public"]

    loc_imgs = defaultdict(list)
    for r in recs:
        if r["kind"] == "own":
            loc_imgs[r["loc"]].append(r)
    loc_counts = {L: csum(r["counts"] for r in rs) for L, rs in loc_imgs.items()}
    locs = sorted(loc_imgs)
    if len(locs) < 2:
        sys.exit(f"Found {len(locs)} location(s) - need at least 2. Check --date-regex.")

    with open(rep_dir / "location_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "images", "videos"] + CLASS_NAMES)
        for L in locs:
            vids = {r["vid"] for r in loc_imgs[L]}
            w.writerow([L, len(loc_imgs[L]), len(vids)] + [loc_counts[L][c] for c in range(NC)])

    old = old_counts = None
    if a.reuse_split:
        old, old_counts = load_old_split(a.reuse_split)
        if (Path(a.reuse_split).resolve() / "train" / "images") == img_dir:
            sys.exit("--reuse-split must point to the previous export, not this dataset's images.")

    # ---------- choose test locations ----------
    n_own = len(own_all)

    def evaluate(combo):
        n = sum(len(loc_imgs[L]) for L in combo)
        frac = n / n_own
        tc = csum(loc_counts[L] for L in combo)
        covered_min = sum(1 for c in range(NC) if tc[c] >= a.min_inst)
        covered_any = sum(1 for c in range(NC) if tc[c] > 0)
        capped = sum(min(tc[c], a.min_inst) for c in range(NC))
        worst_share = max((tc[c] / total[c] for c in range(NC) if total[c]), default=0.0)
        ok = a.test_min <= frac <= a.test_max and worst_share <= a.max_class_share
        return dict(combo=combo, n=n, frac=frac, counts=tc, covered_min=covered_min,
                    covered_any=covered_any, worst_share=worst_share, ok=ok,
                    # all classes present first, then as many instances (up to --min-inst
                    # per class) as possible, then closest to the target size
                    score=(ok, covered_any, capped, covered_min, -abs(frac - a.test_target)))

    if old is not None:
        combo = tuple(sorted({r["loc"] for r in recs
                              if r["kind"] == "own" and old.get(r["path"].name) == "test"}))
        if not combo:
            sys.exit("--reuse-split: the old split has no own images in test.")
        if a.force_test:
            forced = {x.strip() for x in a.force_test.split(",") if x.strip()}
            if forced != set(combo):
                sys.exit(f"--force-test {sorted(forced)} differs from the test locations of the "
                         f"reused split {list(combo)}.")
        cands = [evaluate(combo)]
    elif a.force_test:
        combo = tuple(x.strip() for x in a.force_test.split(",") if x.strip())
        unknown = [L for L in combo if L not in loc_imgs]
        if unknown:
            sys.exit(f"Unknown location(s) in --force-test: {unknown}. Known: {locs}")
        cands = [evaluate(combo)]
    else:
        train_only = {x.strip() for x in (a.train_only or "").split(",") if x.strip()}
        unknown = sorted(train_only - set(locs))
        if unknown:
            sys.exit(f"Unknown location(s) in --train-only: {unknown}. Known: {locs}")
        test_pool = [L for L in locs if L not in train_only]
        kmax = min(a.max_locations, len(test_pool) - 1)
        cands = [evaluate(c) for k in range(1, kmax + 1)
                 for c in itertools.combinations(test_pool, k)]
        cands.sort(key=lambda d: d["score"], reverse=True)
    best = cands[0]
    test_locs = set(best["combo"])

    with open(rep_dir / "test_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "locations", "images", "share_of_own", "meets_constraints",
                    f"classes_with_>={a.min_inst}", "classes_present", "worst_class_share"]
                   + CLASS_NAMES)
        for i, d in enumerate(cands[:30], 1):
            w.writerow([i, " + ".join(d["combo"]), d["n"], f"{d['frac']:.3f}", d["ok"],
                        d["covered_min"], d["covered_any"], f"{d['worst_share']:.2f}"]
                       + [d["counts"][c] for c in range(NC)])

    # ---------- choose val videos (whole videos, from non-test locations) ----------
    groups = defaultdict(list)
    for r in recs:
        if r["kind"] == "own" and r["loc"] not in test_locs:
            groups[(r["loc"], r["vid"])].append(r)
    pool_n = sum(len(v) for v in groups.values())
    pool_counts = csum(r["counts"] for v in groups.values() for r in v)
    pool_classes = [c for c in range(NC) if pool_counts[c] > 0]
    gids = sorted(groups)

    best_val, best_vscore = set(), None
    if old is not None:  # validation groups = groups that had images in the old val split
        best_val = {(r["loc"], r["vid"]) for r in recs
                    if r["kind"] == "own" and old.get(r["path"].name) == "val"}
    for t in (range(a.val_tries) if old is None else []):
        order = gids[:]
        random.Random(a.seed + t).shuffle(order)
        chosen, n = [], 0
        for g in order:
            if n >= a.val_frac * pool_n:
                break
            chosen.append(g)
            n += len(groups[g])
        vc = csum(r["counts"] for g in chosen for r in groups[g])
        present = sum(1 for c in pool_classes if vc[c] > 0)
        vscore = (present, -abs(n / max(pool_n, 1) - a.val_frac))
        if best_vscore is None or vscore > best_vscore:
            best_vscore, best_val = vscore, set(chosen)

    pub_sorted = sorted(public, key=lambda r: r["path"].name)
    random.Random(a.seed).shuffle(pub_sorted)
    pvf = a.val_frac if a.public_val_frac is None else a.public_val_frac
    pub_val = {r["path"] for r in pub_sorted[: round(pvf * len(pub_sorted))]}
    if old is not None:  # public images keep their old split; new public images go to train
        pub_val = {r["path"] for r in public if old.get(r["path"].name) == "val"}

    # ---------- assign ----------
    split_files = {"train": [], "val": [], "test": []}
    split_counts = {s: Counter() for s in split_files}
    split_src = {s: Counter() for s in split_files}
    assign = {}
    conflicts = []
    for r in recs:
        if old is not None and r["path"].name in old:
            s = old[r["path"].name]
            rule = ("test" if r["kind"] == "own" and r["loc"] in test_locs else
                    "val" if r["kind"] == "own" and (r["loc"], r["vid"]) in best_val else None)
            if rule and rule != s or (s == "test" and rule != "test"):
                conflicts.append((r["path"].name, s, rule or "train"))
        elif r["kind"] == "own" and r["loc"] in test_locs:
            s = "test"
        elif r["kind"] == "own" and (r["loc"], r["vid"]) in best_val:
            s = "val"
        elif r["kind"] == "public" and r["path"] in pub_val:
            s = "val"
        else:
            s = "train"
        assign[id(r)] = s
        split_files[s].append(f"./images/{r['path'].name}")
        split_counts[s].update(r["counts"])
        split_src[s][r["kind"]] += 1

    for s, files in split_files.items():
        (root / f"{s}.txt").write_text("\n".join(files) + "\n", encoding="utf-8")

    # ---------- reuse report ----------
    reuse_note = []
    if old is not None:
        now = {r["path"].name: assign[id(r)] for r in recs}
        added = sorted(n for n in now if n not in old)
        removed = sorted(n for n in old if n not in now)
        with open(rep_dir / "reuse_changes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "old_split", "new_split", "change"])
            for n in added:
                w.writerow([n, "", now[n], "added"])
            for n in removed:
                w.writerow([n, old[n], "", "removed"])
            for n, s_old, s_rule in conflicts:
                w.writerow([n, s_old, s_old, f"kept in old split (rules would say {s_rule})"])
        reuse_note.append(f"\nReused split from {Path(a.reuse_split).resolve()}")
        reuse_note.append("Split    kept  added  removed")
        for s in ("train", "val", "test"):
            kept = sum(1 for n, v in now.items() if v == s and n in old)
            add = sum(1 for n in added if now[n] == s)
            rem = sum(1 for n in removed if old[n] == s)
            reuse_note.append(f"{s:<8}{kept:>5}{add:>7}{rem:>9}")
        if conflicts:
            reuse_note.append(f"WARNING: {len(conflicts)} images kept their old split although "
                              f"location/group rules now differ (see splits/reuse_changes.csv)")
        if old_counts:
            with open(rep_dir / "reuse_class_changes.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "class"] + [f"{s}_{k}" for s in ("train", "val", "test")
                                              for k in ("old", "new", "change")])
                for c in range(NC):
                    row = [c, CLASS_NAMES[c]]
                    for s in ("train", "val", "test"):
                        o, nw = old_counts.get(s, Counter())[c], split_counts[s][c]
                        row += [o, nw, nw - o]
                    w.writerow(row)
            for s in ("train", "val", "test"):
                o = sum(old_counts.get(s, Counter()).values())
                nw = sum(split_counts[s].values())
                reuse_note.append(f"  {s:<6} instances {o:>6} -> {nw:>6} ({nw - o:+d})")
            reuse_note.append("  per class: splits/reuse_class_changes.csv")

    # ---------- optional: copy files into per-split folders ----------
    export_note = []
    if a.export:
        exp = Path(a.export).resolve()
        n_copied, n_json, n_lbl = Counter(), Counter(), Counter()
        for r in recs:
            s = assign[id(r)]
            di, dl = exp / s / "images", exp / s / "labels"
            di.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r["path"], di / r["path"].name)
            n_copied[s] += 1
            js = r["path"].with_suffix(".json")
            if js.exists():
                shutil.copy2(js, di / js.name)
                n_json[s] += 1
            lp = lbl_dir / f"{r['path'].stem}.txt"
            if lp.exists():
                shutil.copy2(lp, dl / lp.name)
                n_lbl[s] += 1
        names_e = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
        (exp / "data.yaml").write_text(
            f"# generated by make_split.py --export\npath: {exp.as_posix()}\n"
            f"train: train/images\nval: val/images\ntest: test/images\n\n"
            f"names:\n{names_e}\n", encoding="utf-8")
        export_note.append(f"\nExported copies to {exp}")
        for s in split_files:
            export_note.append(f"  {s:<6} images {n_copied[s]:>5}  json {n_json[s]:>5}  "
                               f"yolo labels {n_lbl[s]:>5}")
        missing_json = sum(n_copied.values()) - sum(n_json.values())
        if missing_json:
            export_note.append(f"  {missing_json} images had no .json next to them "
                               f"(fine for images annotated outside X-AnyLabeling)")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (root / "data_split.yaml").write_text(
        f"# generated by make_split.py - edit 'path' if the dataset moves to another computer\n"
        f"path: {root.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n\n"
        f"names:\n{names}\n", encoding="utf-8")

    with open(rep_dir / "class_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "train", "val", "test", "total", "test_share"])
        for c in range(NC):
            tot = total[c]
            w.writerow([c, CLASS_NAMES[c], split_counts["train"][c], split_counts["val"][c],
                        split_counts["test"][c], tot,
                        f"{split_counts['test'][c] / tot:.2f}" if tot else ""])

    with open(rep_dir / "val_videos.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(f"{L}\t{v}" for L, v in sorted(best_val)) + "\n")
    if noloc:
        (rep_dir / "no_location_files.txt").write_text(
            "\n".join(r["path"].name for r in noloc) + "\n", encoding="utf-8")

    # ---------- summary ----------
    lines = [
        f"Images: {len(recs)}  (own with location {n_own - len(noloc)}, "
        f"own without location {len(noloc)} -> train, public {len(public)})",
        f"Locations found ({len(locs)}): {', '.join(locs)}",
        "",
        f"TEST locations: {' + '.join(best['combo'])}",
        f"  {best['n']} images = {best['frac']:.1%} of own images | "
        f"classes with >={a.min_inst} instances: {best['covered_min']}/{NC} | "
        f"classes present: {best['covered_any']}/{NC}",
    ]
    if not best["ok"] and a.force_test:
        lines.append(f"  WARNING: forced test set is outside the size/share limits "
                     f"(worst class share {best['worst_share']:.0%}).")
    elif not best["ok"]:
        lines.append("  WARNING: no combination met all constraints - best compromise chosen. "
                     "See splits/test_candidates.csv")
    weak = [CLASS_NAMES[c] for c in range(NC) if best["counts"][c] < a.min_inst]
    if weak:
        lines.append("  Below target in test: " + ", ".join(
            f"{CLASS_NAMES[c]} ({best['counts'][c]})" for c in range(NC)
            if best["counts"][c] < a.min_inst))
    lines += [
        f"VAL: {len(best_val)} own videos + {len(pub_val)} public images",
        "",
        "Split    images   own  public  no-loc",
    ]
    for s in split_files:
        src = split_src[s]
        lines.append(f"{s:<8}{len(split_files[s]):>7}{src['own']:>6}{src['public']:>8}"
                     f"{src['own_noloc']:>8}")
    if dropped:
        lines.append(f"\nLeft out {len(dropped)} images without labels (--drop-empty), "
                     f"see splits/dropped_empty.txt")
    elif missing:
        lines.append(f"\nNote: {len(missing)} images have no label file (kept as background).")
    if bad_ids:
        lines.append(f"WARNING: {len(bad_ids)} label lines with class id outside 0-{NC - 1} "
                     f"were ignored, e.g. {bad_ids[:3]}")
    lines.append(f"\nReports in {rep_dir}. Train with: data='{(root / 'data_split.yaml').as_posix()}'")
    lines += reuse_note
    lines += export_note
    if a.export:
        lines.append(f"  or train from the copies with: data='{(Path(a.export).resolve() / 'data.yaml').as_posix()}'")
    summary = "\n".join(lines)
    print(summary)
    (rep_dir / "split_summary.txt").write_text(summary + "\n", encoding="utf-8")
    if a.export:
        shutil.copytree(rep_dir, Path(a.export).resolve() / "splits", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
