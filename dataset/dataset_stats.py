#!/usr/bin/env python3
"""
dataset_stats.py - read-only statistics for the fire-safety YOLO-seg dataset.

Nothing in the dataset is changed. Reports go to --out (default: ./dataset_stats).

Filename example (own data):
  VID_20251219_094741_00_053(1)_frame_000004_horizontal_05
    date  = 20251219                          (first 20YYMMDD in the name)
    clip  = VID_20251219_094741_00_053(1)     (everything before _frame_)
    video = VID_20251219_094741_00_053        (clip without the (n): all clips cut
                                               from one recording count as one video)
    frame = clip + frame_000004               (all cube faces of one frame)
  360 / smartphone photos (no _frame_): video = frame = one photo (its timestamp),
  e.g. IMG_20191223_093150_00_063_top_01 and _top_03 are one photo.
    image = one cube face = one file

Public images are recognised by prefix (default fireart_, firenet_).
Own files without a date are reported as "no-date".

Expects <root>/images and/or <root>/labels (YOLO-seg .txt).

Optional --location-map: CSV with header  date,location  (one row per date), e.g.
  date,location
  20251219,School_A
  20251220,School_A
  20260105,School_B
Dates missing from the map stay as their own location (a warning lists them).
The same CSV works with make_split.py --location-map.

Usage:
  python dataset_stats.py --root "D:/fire_dataset"
  python dataset_stats.py --root "D:/fire_dataset" --location-map locations.csv
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = [
    "Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
    "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED",
    "Fire Call Point", "Fire Blanket", "Emergency Door Release", "Evacuation Chair",
]
NC = len(CLASS_NAMES)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SHORT = ["Ext", "Alarm", "Hose", "Exit", "Ind", "Smoke", "FAid", "AED",
         "Call", "Blank", "Door", "Chair"]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="folder containing images/ and labels/")
    p.add_argument("--out", default="dataset_stats", help="folder for the CSV reports")
    p.add_argument("--public-prefixes", default="fireart_,firenet_")
    p.add_argument("--date-regex", default=r"(20\d{6})")
    p.add_argument("--location-map", default=None,
                   help="optional CSV with columns date,location to merge dates into locations")
    return p.parse_args()


PHOTO_RE = re.compile(r"20\d{6}_\d{6}")


def clip_of(stem):
    """Video clip name, or for 360/smartphone photos the photo's timestamp, so all cube
    faces of one photo (IMG_20191223_093150_00_063_top_01, ..._top_03) count once."""
    if "_frame_" in stem:
        return stem.split("_frame_")[0]
    m = PHOTO_RE.search(stem)
    return m.group(0) if m else stem


def video_of(stem):
    """Source video: clips exported as name(1), name(2) ... belong to one recording."""
    return re.sub(r"\(\d+\)$", "", clip_of(stem))


def frame_of(stem):
    m = re.match(r"(.*_frame_\d+)", stem)
    return m.group(1) if m else clip_of(stem)


def main():
    a = get_args()
    root = Path(a.root).resolve()
    img_dir, lbl_dir = root / "images", root / "labels"
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
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
            raise SystemExit(f"No date,location rows read from {a.location_map}")

    stems_img = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS} \
        if img_dir.is_dir() else set()
    stems_lbl = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()
    stems = sorted(stems_img | stems_lbl)
    if not stems:
        raise SystemExit(f"No images or labels found under {root}")

    # per group (date / "no-date" / "public")
    g_imgs = Counter()
    g_empty = Counter()
    g_inst = defaultdict(Counter)
    g_imgs_with = defaultdict(Counter)       # images containing class c
    g_videos = defaultdict(set)
    g_frames = defaultdict(set)
    g_clips = defaultdict(set)
    g_dates = defaultdict(set)
    unmapped = set()
    # per video
    v_imgs, v_frames, v_inst, v_group = Counter(), defaultdict(set), defaultdict(Counter), {}
    v_date = {}
    v_clips = defaultdict(set)
    bad_ids, no_label, no_image = [], [], []

    for s in stems:
        if s not in stems_lbl:
            no_label.append(s)
        if stems_img and s not in stems_img:
            no_image.append(s)

        date = ""
        if s.lower().startswith(prefixes):
            g = "public"
        else:
            m = date_re.search(s)
            if m:
                date = m.group(1)
                if loc_map and date not in loc_map:
                    unmapped.add(date)
                g = loc_map.get(date, date)
                g_dates[g].add(date)
            else:
                g = "no-date"

        counts = Counter()
        lp = lbl_dir / f"{s}.txt"
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                c = int(float(parts[0]))
                if 0 <= c < NC:
                    counts[c] += 1
                else:
                    bad_ids.append((s, c))

        g_imgs[g] += 1
        if not counts:
            g_empty[g] += 1
        g_inst[g].update(counts)
        for c in counts:
            g_imgs_with[g][c] += 1

        if g != "public":
            v, f, cl = video_of(s), frame_of(s), clip_of(s)
            g_videos[g].add(v)
            g_clips[g].add(cl)
            g_frames[g].add(f)
            v_clips[v].add(cl)
            v_imgs[v] += 1
            v_frames[v].add(f)
            v_inst[v].update(counts)
            v_group[v] = g
            v_date[v] = date

    dates = sorted(k for k in g_imgs if k not in ("public", "no-date"))
    order = dates + [k for k in ("no-date", "public") if k in g_imgs]
    key = "location" if loc_map else "date"
    suffix = "location" if loc_map else "date"

    # ---------- per date / location ----------
    with open(out / f"stats_by_{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([key, "dates", "videos", "clips", "frames", "images", "empty_images",
                    "instances", "classes_present"] + CLASS_NAMES)
        for g in order + ["TOTAL_OWN", "TOTAL_ALL"]:
            if g.startswith("TOTAL"):
                keys = [k for k in order if k != "public"] if g == "TOTAL_OWN" else order
                inst = sum((g_inst[k] for k in keys), Counter())
                row = [g, sum(len(g_dates[k]) for k in keys),
                       sum(len(g_videos[k]) for k in keys if k != "public"),
                       sum(len(g_clips[k]) for k in keys if k != "public"),
                       sum(len(g_frames[k]) for k in keys if k != "public"),
                       sum(g_imgs[k] for k in keys), sum(g_empty[k] for k in keys)]
            else:
                inst = g_inst[g]
                vids = "" if g == "public" else len(g_videos[g])
                clps = "" if g == "public" else len(g_clips[g])
                frms = "" if g == "public" else len(g_frames[g])
                row = [g, " ".join(sorted(g_dates[g])), vids, clps, frms, g_imgs[g],
                       g_empty[g]]
            row += [sum(inst.values()), sum(1 for c in range(NC) if inst[c])]
            row += [inst[c] for c in range(NC)]
            w.writerow(row)

    # ---------- images containing each class, per date ----------
    with open(out / f"images_per_class_by_{suffix}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([key] + CLASS_NAMES)
        for g in order:
            w.writerow([g] + [g_imgs_with[g][c] for c in range(NC)])

    # ---------- per video ----------
    with open(out / "stats_by_video.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "date", "video", "clips", "frames", "images", "instances"]
                   + CLASS_NAMES)
        for v in sorted(v_imgs, key=lambda x: (v_group[x], x)):
            w.writerow([v_group[v], v_date[v], v, len(v_clips[v]), len(v_frames[v]), v_imgs[v],
                        sum(v_inst[v].values())] + [v_inst[v][c] for c in range(NC)])

    # ---------- console summary ----------
    W = max([10] + [len(g) + 2 for g in order])
    head = f"{key:<{W}}{'vid':>5}{'frm':>6}{'img':>6}{'inst':>6} | " + \
           "".join(f"{s:>6}" for s in SHORT)
    lines = [f"Instances per class by {key}", head, "-" * len(head)]
    for g in order:
        inst = g_inst[g]
        vids = "-" if g == "public" else len(g_videos[g])
        frms = "-" if g == "public" else len(g_frames[g])
        lines.append(f"{g:<{W}}{vids:>5}{frms:>6}{g_imgs[g]:>6}{sum(inst.values()):>6} | "
                     + "".join(f"{inst[c]:>6}" for c in range(NC)))
    own = sum((g_inst[k] for k in order if k != "public"), Counter())
    lines.append("-" * len(head))
    lines.append(f"{'OWN':<{W}}{sum(len(g_videos[k]) for k in order if k != 'public'):>5}"
                 f"{sum(len(g_frames[k]) for k in order if k != 'public'):>6}"
                 f"{sum(g_imgs[k] for k in order if k != 'public'):>6}"
                 f"{sum(own.values()):>6} | " + "".join(f"{own[c]:>6}" for c in range(NC)))
    lines.append("")
    n_dates = len(set().union(*g_dates.values())) if g_dates else 0
    if loc_map:
        lines.append(f"Dates found: {n_dates}, grouped into {len(dates)} locations")
        multi = [f"{g} ({', '.join(sorted(g_dates[g]))})" for g in dates if len(g_dates[g]) > 1]
        if multi:
            lines.append("Locations with several dates: " + "; ".join(multi))
        if unmapped:
            lines.append(f"WARNING: {len(unmapped)} date(s) not in the location map, kept as "
                         f"their own location: {', '.join(sorted(unmapped))}")
    else:
        lines.append(f"Dates found: {n_dates}")
    multi_clip = sum(1 for v in v_clips if len(v_clips[v]) > 1)
    n_clips = sum(len(v_clips[v]) for v in v_clips)
    lines.append(f"{n_clips} clips from {len(v_clips)} source videos "
                 f"({multi_clip} videos were cut into several clips: name, name(1), name(2) ...)")
    if no_label:
        lines.append(f"{len(no_label)} images have no label file (counted as empty).")
    if no_image:
        lines.append(f"{len(no_image)} label files have no matching image.")
    if bad_ids:
        lines.append(f"WARNING: {len(bad_ids)} label lines with class id outside 0-{NC - 1}, "
                     f"e.g. {bad_ids[:3]}")
    lines.append(f"\nCSV reports written to {out}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
