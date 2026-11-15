"""
STEP: Extract frames from 360 video walkthroughs and cut the pile down to a
labelable set BEFORE a human looks at any of it.

Accepts EITHER a single video file OR a directory of videos. Every video is
processed on its own, all kept frames land in ONE shared folder
(<output_dir>/kept/), and a per-video results table is printed at the end.

Pipeline (run separately for each video):
  1. ffmpeg extracts frames at a sparse, fixed rate (not the video's native fps).
  2. Each frame is scored for blur (variance of Laplacian), measured ONLY in a
     horizontal band around the middle of the frame.
  3. Frames are dropped based on where they fall in THAT video's own score
     distribution -- self-calibrating per video.
  4. Remaining frames are compared against the last KEPT frame via a
     perceptual hash -- near-duplicates of a frame you already have are
     skipped.

NAMING: kept frames are written as "<video_stem>_frame_000123.jpg", so every
kept frame's name is unique across all videos and tells you which video it
came from. Because of this, two videos with the same file name (e.g. in
different sub-folders when using --recursive) are refused up front -- they
would overwrite each other.

Usage:
    # whole directory of videos
    python extract_and_filter_frames_batch.py videos_folder/ output_folder/ \\
        --fps 1.0 --blur-drop-percentile 10 --hash-distance 6

    # include videos in sub-folders too
    python extract_and_filter_frames_batch.py videos_folder/ output_folder/ --recursive

    # a single video still works
    python extract_and_filter_frames_batch.py walkthrough.mp4 output_folder/
"""
import argparse
import csv
import subprocess
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import imagehash
from PIL import Image

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_videos(path, recursive=False):
    """Return a sorted list of video files from a file path or a directory."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        sys.exit(f"Error: '{path}' is not a file or directory.")
    candidates = p.rglob("*") if recursive else p.iterdir()
    return sorted(
        f for f in candidates if f.is_file() and f.suffix.lower() in VIDEO_EXTS
    )


def check_unique_stems(videos):
    """Frames are named by video stem, so two videos sharing a stem would collide."""
    counts = Counter(v.stem for v in videos)
    clashes = {stem: n for stem, n in counts.items() if n > 1}
    if clashes:
        lines = "\n".join(
            f"  '{stem}' appears {n} times:\n"
            + "\n".join(f"      {v}" for v in videos if v.stem == stem)
            for stem, n in clashes.items()
        )
        sys.exit(
            "Error: multiple videos share the same file name (without extension), "
            "so their frames would overwrite each other in kept/.\n"
            f"{lines}\nRename them and re-run."
        )


def extract_raw_frames(video_path, raw_dir, fps):
    raw_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(raw_dir / "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}",
         "-qscale:v", "2", pattern],
        check=True, capture_output=True,
    )
    return sorted(raw_dir.glob("frame_*.jpg"))


def blur_score(image_path, band_fraction=0.5):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h = img.shape[0]
    band_h = max(int(h * band_fraction), 1)
    top = (h - band_h) // 2
    band = img[top:top + band_h, :]
    return cv2.Laplacian(band, cv2.CV_64F).var()


def summarize(scores):
    arr = np.array(scores)
    pct = lambda p: np.percentile(arr, p)
    print(f"  blur score distribution: min={arr.min():.0f}  "
          f"p10={pct(10):.0f}  median={pct(50):.0f}  "
          f"p90={pct(90):.0f}  max={arr.max():.0f}")


# --------------------------------------------------------------------------- #
# Per-video processing
# --------------------------------------------------------------------------- #
def process_video(video_path, out_dir, args):
    """Run the full pipeline on ONE video. Returns a dict of counts."""
    video_stem = video_path.stem
    raw_dir = out_dir / f"_raw_{video_stem}"
    kept_dir = out_dir / "kept"
    kept_dir.mkdir(parents=True, exist_ok=True)

    # Clear this video's frames from any previous run so counts stay accurate
    # (frames from other videos are never touched).
    stale = list(kept_dir.glob(f"{video_stem}_frame_*.jpg"))
    if stale:
        print(f"  Note: removing {len(stale)} frame(s) from an earlier run of "
              f"'{video_stem}' (other videos' frames are untouched).")
        for f in stale:
            f.unlink()

    try:
        print(f"  Extracting frames at {args.fps} fps...")
        raw_frames = extract_raw_frames(video_path, raw_dir, args.fps)
        n_raw = len(raw_frames)
        print(f"  {n_raw} raw frames extracted")
        if n_raw == 0:
            raise RuntimeError("ffmpeg produced no frames")

        scored = [(f, blur_score(f, args.blur_band)) for f in raw_frames]
        summarize([s for _, s in scored])

        if args.blur_threshold is not None:
            cutoff = args.blur_threshold
            print(f"  using fixed threshold: {cutoff}")
        else:
            cutoff = np.percentile([s for _, s in scored], args.blur_drop_percentile)
            print(f"  using self-calibrated cutoff "
                  f"(p{args.blur_drop_percentile:.0f}): {cutoff:.0f}")

        sharp_frames = [f for f, s in scored if s >= cutoff]
        n_blur_dropped = n_raw - len(sharp_frames)
        print(f"  {len(sharp_frames)} survived the blur filter "
              f"({n_blur_dropped} dropped as too blurry)")

        kept = []
        last_hash = None
        for f in sharp_frames:
            with Image.open(f) as im:
                h = imagehash.phash(im)
            if last_hash is None or (h - last_hash) >= args.hash_distance:
                dest = kept_dir / f"{video_stem}_{f.name}"
                shutil.copy(f, dest)
                kept.append(dest)
                last_hash = h
        n_dup_dropped = len(sharp_frames) - len(kept)
        print(f"  {len(kept)} survived de-duplication "
              f"({n_dup_dropped} dropped as near-duplicates of a kept frame)")

        return {
            "video": video_path.name,
            "status": "ok",
            "raw": n_raw,
            "blur_dropped": n_blur_dropped,
            "dup_dropped": n_dup_dropped,
            "kept": len(kept),
        }
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def pct_str(kept, raw):
    return f"{100 * kept / raw:.1f}%" if raw else "-"


def print_summary(results):
    name_w = min(max(len(r["video"]) for r in results + [{"video": "TOTAL"}]), 45)
    header = (f"{'Video':<{name_w}}  {'Raw':>6}  {'Blurry':>7}  "
              f"{'Dupes':>6}  {'Dropped':>7}  {'Kept':>6}  {'Kept %':>7}")
    line = "-" * len(header)

    print("\n" + "=" * len(header))
    print("RESULTS PER VIDEO")
    print("=" * len(header))
    print(header)
    print(line)

    totals = {"raw": 0, "blur_dropped": 0, "dup_dropped": 0, "kept": 0}
    failed = []
    for r in results:
        name = r["video"] if len(r["video"]) <= name_w else r["video"][:name_w - 3] + "..."
        if r["status"] != "ok":
            print(f"{name:<{name_w}}  FAILED: {r['status']}")
            failed.append(r["video"])
            continue
        dropped = r["blur_dropped"] + r["dup_dropped"]
        print(f"{name:<{name_w}}  {r['raw']:>6}  {r['blur_dropped']:>7}  "
              f"{r['dup_dropped']:>6}  {dropped:>7}  {r['kept']:>6}  "
              f"{pct_str(r['kept'], r['raw']):>7}")
        for k in totals:
            totals[k] += r[k]

    total_dropped = totals["blur_dropped"] + totals["dup_dropped"]
    print(line)
    print(f"{'TOTAL':<{name_w}}  {totals['raw']:>6}  {totals['blur_dropped']:>7}  "
          f"{totals['dup_dropped']:>6}  {total_dropped:>7}  {totals['kept']:>6}  "
          f"{pct_str(totals['kept'], totals['raw']):>7}")
    print("(Blurry = dropped by blur filter; Dupes = dropped as near-duplicates)")

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")


def write_csv(results, csv_path):
    fields = ["video", "status", "raw", "blur_dropped", "dup_dropped", "kept"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Extract + filter frames from one video or a folder of videos."
    )
    ap.add_argument("input", help="A video file, or a directory containing videos")
    ap.add_argument("output_dir")
    ap.add_argument("--recursive", action="store_true",
                    help="When input is a directory, also search sub-folders")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--blur-band", type=float, default=0.5)
    ap.add_argument("--blur-drop-percentile", type=float, default=10.0)
    ap.add_argument("--blur-threshold", type=float, default=None)
    ap.add_argument("--hash-distance", type=int, default=6)
    args = ap.parse_args()

    videos = find_videos(args.input, args.recursive)
    if not videos:
        sys.exit(f"No videos ({', '.join(sorted(VIDEO_EXTS))}) found in '{args.input}'.")
    check_unique_stems(videos)

    out_dir = Path(args.output_dir)
    kept_dir = out_dir / "kept"
    kept_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} video(s). Frames will be saved to {kept_dir}/\n")

    results = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        try:
            results.append(process_video(video, out_dir, args))
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or b"").decode(errors="ignore").strip().splitlines()
            reason = msg[-1] if msg else "ffmpeg error"
            print(f"  ERROR: {reason}")
            results.append({"video": video.name, "status": reason})
        except Exception as e:  # keep going so one bad video doesn't stop the batch
            print(f"  ERROR: {e}")
            results.append({"video": video.name, "status": str(e)})
        print()

    print_summary(results)

    csv_path = out_dir / "frame_filter_summary.csv"
    write_csv(results, csv_path)
    print(f"\nSummary also saved to {csv_path}")

    total_in_folder = sum(1 for _ in kept_dir.glob("*.jpg"))
    print(f"Total frames now in {kept_dir}/: {total_in_folder}")


if __name__ == "__main__":
    main()
