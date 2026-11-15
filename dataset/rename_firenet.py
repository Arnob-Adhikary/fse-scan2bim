#!/usr/bin/env python3
"""
rename_firenet.py - give FireNet files their own prefix.

FireNet images were named 0001.jpg ... and prefixed to fireart_0001.jpg. This renames every
file whose name is exactly  fireart_<4 digits>.<ext>  to  firenet_<4 digits>.<ext>
in <root>/images (image + X-AnyLabeling .json) and <root>/labels (YOLO .txt).

Default is a PREVIEW (nothing renamed). Add --apply to rename.
Every rename is written to <root>/rename_firenet_log.csv; --undo reverses them.

  python rename_firenet.py --root "D:/.../FSE-360_v1"            # preview
  python rename_firenet.py --root "D:/.../FSE-360_v1" --apply    # rename
  python rename_firenet.py --root "D:/.../FSE-360_v1" --undo     # put names back
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PATTERN = re.compile(r"^fireart_(\d{4})$")  # stem must be exactly fireart_ + 4 digits


def fix_image_path(json_file: Path, old_prefix: str, new_prefix: str):
    """Update X-AnyLabeling's imagePath inside the .json so it matches the renamed image."""
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    ip = data.get("imagePath")
    if isinstance(ip, str):
        name = ip.replace("\\", "/").split("/")[-1]
        if name.startswith(old_prefix):
            data["imagePath"] = ip[: len(ip) - len(name)] + new_prefix + name[len(old_prefix):]
            json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset folder with images/ and labels/")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: preview)")
    ap.add_argument("--undo", action="store_true", help="reverse the renames in the log")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    log = root / "rename_firenet_log.csv"

    if a.undo:
        if not log.exists():
            sys.exit(f"No log found: {log}")
        rows = list(csv.DictReader(open(log, encoding="utf-8")))
        n = 0
        for r in rows:
            new, old = Path(r["new"]), Path(r["old"])
            if new.exists() and not old.exists():
                new.rename(old)
                if old.suffix.lower() == ".json":
                    fix_image_path(old, "firenet_", "fireart_")
                n += 1
        log.rename(log.with_suffix(".undone.csv"))
        print(f"Undone: {n} of {len(rows)} files renamed back.")
        return

    folders = [root / "images", root / "labels"]
    for f in folders:
        if not f.is_dir():
            sys.exit(f"Folder not found: {f}")

    plan, kinds = [], Counter()
    other_fireart = 0
    for folder in folders:
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            m = PATTERN.match(p.stem)
            if m:
                plan.append((p, p.with_name(f"firenet_{m.group(1)}{p.suffix}")))
                kinds[(folder.name, p.suffix.lower())] += 1
            elif p.stem.startswith("fireart_"):
                other_fireart += 1

    if not plan:
        sys.exit("No files named fireart_<4 digits> found - nothing to do.")
    clashes = [new for _, new in plan if new.exists()]
    if clashes:
        sys.exit(f"{len(clashes)} target names already exist, e.g. {clashes[0]}. Nothing renamed.")

    print("Files matching fireart_<4 digits> (-> firenet_):")
    for (folder, ext), n in sorted(kinds.items()):
        print(f"  {folder:<7} {ext:<6} {n}")
    print(f"Other fireart_ files left unchanged: {other_fireart}")
    print("Examples:")
    for old, new in plan[:5]:
        print(f"  {old.parent.name}/{old.name}  ->  {new.name}")

    imgs = {o.stem for o, _ in plan if o.parent.name == "images" and o.suffix.lower() != ".json"}
    lbls = {o.stem for o, _ in plan if o.parent.name == "labels"}
    if imgs - lbls:
        print(f"Note: {len(imgs - lbls)} FireNet images have no label file (fine if they are empty).")

    if not a.apply:
        print("\nPREVIEW only - check that the counts look like FireNet only, then add --apply.")
        return

    with open(log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old", "new"])
        for old, new in plan:
            old.rename(new)
            if new.suffix.lower() == ".json":
                fix_image_path(new, "fireart_", "firenet_")
            w.writerow([str(old), str(new)])
    print(f"\nRenamed {len(plan)} files. Log: {log}  (undo with --undo)")


if __name__ == "__main__":
    main()
