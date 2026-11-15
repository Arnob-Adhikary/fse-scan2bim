"""
Filter the Fire-ART dataset down to the classes that exist in YOUR class list,
and renumber the kept classes to YOUR class indices.

How classes are matched
-----------------------
By NAME (case-insensitive), between the two class files you provide:
    Fire-ART "Fire exit sign" (id 1)  ->  yours "Fire Exit Sign" (id 3)
Fire-ART classes with no name match in your list are removed. Your classes
with no match in Fire-ART (e.g. AED, First Aid Kit) simply get no Fire-ART
instances. The full mapping is printed before anything is written, and
saved to class_mapping.txt next to the output. If two names should be treated
as the same class, use --alias (see below).

What happens to images
----------------------
- Label lines of removed classes are deleted; kept lines are renumbered.
- An image that still has at least one label line is kept.
- An image that ONLY contained removed classes (or had no labels at all) is
  dropped by default. Use --keep-background to keep such images as
  background/negative images (empty label file) instead.
- The original dataset is never modified; everything is written to output_dir.

Expected input layout (both common layouts work, subfolders like train/val
are mirrored to the output):
    root/images/<split>/x.jpg   +  root/labels/<split>/x.txt
    root/<split>/images/x.jpg   +  root/<split>/labels/x.txt
    root/images/x.jpg           +  root/labels/x.txt
Label lines may be segmentation polygons or boxes; only the first number
(class id) is changed.

Usage:
    python filter_remap_fireart.py "Fire-ART_Part 2" fireart_filtered ^
        --src-classes classes_fireart.txt --dst-classes classes_fse.txt

    # preview only, writes nothing
    python filter_remap_fireart.py "Fire-ART_Part 2" fireart_filtered ^
        --src-classes classes_fireart.txt --dst-classes classes_fse.txt --dry-run

    # flat images/ + labels/ output (e.g. for split_dataset.py), keep negatives
    python filter_remap_fireart.py "Fire-ART_Part 2" fireart_filtered ^
        --src-classes classes_fireart.txt --dst-classes classes_fse.txt ^
        --flatten --keep-background

    # map a Fire-ART class onto a differently named class of yours
    ... --alias "Fire alarm=Fire Indicator"
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
def norm(name):
    return " ".join(name.lower().split())


def parse_classes(path):
    """Read lines like '3 = Fire Exit Sign' -> {3: 'Fire Exit Sign'}."""
    names = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            sys.exit(f"Cannot parse line in {path}: '{line}' (expected 'id = name')")
        idx, name = line.split("=", 1)
        names[int(idx.strip())] = name.strip()
    return names


def parse_aliases(alias_args, dst):
    dst_names = {norm(n) for n in dst.values()}
    aliases = {}
    for a in alias_args or []:
        if "=" not in a:
            sys.exit(f"--alias must look like 'Source name=Target name', got '{a}'")
        left, right = a.split("=", 1)
        if norm(right) not in dst_names:
            sys.exit(f"--alias target '{right}' is not in your class list.")
        aliases[norm(left)] = norm(right)
    return aliases


def build_mapping(src, dst, aliases):
    """{old_id: new_id} for every source class that has a counterpart."""
    dst_by_name = {norm(n): i for i, n in dst.items()}
    mapping = {}
    for sid, sname in src.items():
        key = aliases.get(norm(sname), norm(sname))
        if key in dst_by_name:
            mapping[sid] = dst_by_name[key]
    return mapping


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def find_label_files(root):
    """All .txt files that live under a folder named 'labels'."""
    out = []
    for p in sorted(root.rglob("*.txt")):
        rel = p.relative_to(root)
        if any(part.lower() == "labels" for part in rel.parts[:-1]):
            out.append(p)
    return out


def find_image_for_label(root, label_path):
    """Swap the last 'labels' folder in the path for 'images' and look for an
    image with the same stem (any common extension)."""
    rel_parts = list(label_path.relative_to(root).parts)
    for i in range(len(rel_parts) - 2, -1, -1):
        if rel_parts[i].lower() == "labels":
            rel_parts[i] = "images"
            break
    img_dir = root.joinpath(*rel_parts[:-1])
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        for cand in (img_dir / f"{stem}{ext}", img_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_root", help="Fire-ART folder (contains images/ and labels/)")
    ap.add_argument("output_dir")
    ap.add_argument("--src-classes", required=True, help="Fire-ART class file ('id = name')")
    ap.add_argument("--dst-classes", required=True, help="Your class file ('id = name')")
    ap.add_argument("--alias", action="append",
                    help="Manual match 'Fire-ART name=Your name' (repeatable)")
    ap.add_argument("--keep-background", action="store_true",
                    help="Keep images left with no labels as background images")
    ap.add_argument("--flatten", action="store_true",
                    help="Write everything to output_dir/images and output_dir/labels "
                         "(no train/val subfolders)")
    ap.add_argument("--prefix", default="",
                    help="Prefix added to every output file name, e.g. 'fireart_'")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without writing anything")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    if not root.is_dir():
        sys.exit(f"Dataset folder not found: {root}")
    if out.resolve() == root.resolve() or root.resolve() in out.resolve().parents:
        sys.exit("output_dir must be outside the dataset folder.")

    src = parse_classes(args.src_classes)
    dst = parse_classes(args.dst_classes)
    aliases = parse_aliases(args.alias, dst)
    mapping = build_mapping(src, dst, aliases)

    # ---- show the mapping first ------------------------------------------
    print("Class mapping (Fire-ART -> yours):")
    for sid in sorted(src):
        if sid in mapping:
            print(f"  {sid:>2} {src[sid]:<28} -> {mapping[sid]:>2} {dst[mapping[sid]]}")
        else:
            print(f"  {sid:>2} {src[sid]:<28} -> REMOVED")
    no_source = [i for i in sorted(dst) if i not in mapping.values()]
    if no_source:
        print("\nYour classes with no Fire-ART counterpart (will come from your own data only):")
        for i in no_source:
            print(f"  {i:>2} {dst[i]}")
    if not mapping:
        sys.exit("\nNo class names matched between the two files -- nothing to do. "
                 "Check the class files or use --alias.")
    print()

    # ---- process ---------------------------------------------------------
    label_files = find_label_files(root)
    if not label_files:
        sys.exit(f"No label .txt files found under a 'labels' folder in {root}.")

    before, removed, after = Counter(), Counter(), Counter()
    n_kept = n_dropped_only_removed = n_dropped_empty = 0
    n_bg_kept = n_partially_stripped = n_missing_img = n_bad_lines = 0
    unknown_ids = Counter()
    seen_outputs = {}

    for label_path in label_files:
        img_path = find_image_for_label(root, label_path)
        if img_path is None:
            n_missing_img += 1
            continue

        new_lines, n_removed_here, n_orig = [], 0, 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                n_bad_lines += 1
                continue
            n_orig += 1
            if cid not in src:
                unknown_ids[cid] += 1
                n_removed_here += 1
                continue
            before[cid] += 1
            if cid in mapping:
                parts[0] = str(mapping[cid])
                new_lines.append(" ".join(parts))
                after[mapping[cid]] += 1
            else:
                removed[cid] += 1
                n_removed_here += 1

        if new_lines:
            n_kept += 1
            if n_removed_here:
                n_partially_stripped += 1
        elif args.keep_background:
            n_bg_kept += 1
        else:
            if n_orig:
                n_dropped_only_removed += 1
            else:
                n_dropped_empty += 1
            continue

        # ---- output paths ------------------------------------------------
        if args.flatten:
            img_dest = out / "images" / (args.prefix + img_path.name)
            lbl_dest = out / "labels" / (args.prefix + label_path.name)
        else:
            img_dest = out / img_path.relative_to(root).parent / (args.prefix + img_path.name)
            lbl_dest = out / label_path.relative_to(root).parent / (args.prefix + label_path.name)

        key = str(img_dest).lower()
        if key in seen_outputs:
            sys.exit(f"Name collision: '{img_path}' and '{seen_outputs[key]}' would both be "
                     f"written to '{img_dest}'. This happens with --flatten when two splits "
                     f"contain the same file name. Re-run without --flatten, or rename one.")
        seen_outputs[key] = img_path

        if not args.dry_run:
            img_dest.parent.mkdir(parents=True, exist_ok=True)
            lbl_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, img_dest)
            lbl_dest.write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                                encoding="utf-8")

    # ---- report ----------------------------------------------------------
    print(f"Scanned {len(label_files)} label files.")
    print(f"  images kept (with labels):            {n_kept}"
          f"   ({n_partially_stripped} of them had some labels removed)")
    if args.keep_background:
        print(f"  images kept as background:            {n_bg_kept}")
    else:
        print(f"  dropped (only removed classes):       {n_dropped_only_removed}")
        print(f"  dropped (no labels to begin with):    {n_dropped_empty}")
    if n_missing_img:
        print(f"  label files with no matching image:   {n_missing_img}  (skipped)")
    if n_bad_lines:
        print(f"  malformed label lines skipped:        {n_bad_lines}")
    if unknown_ids:
        print(f"  WARNING: class ids not in --src-classes were removed: {dict(unknown_ids)}")

    print("\nInstances per class (Fire-ART original -> after filtering):")
    for sid in sorted(src):
        if sid in mapping:
            print(f"  {src[sid]:<28} {before[sid]:>6} -> {after[mapping[sid]]:>6}"
                  f"   (new id {mapping[sid]})")
    print("Removed classes:")
    for sid in sorted(removed):
        print(f"  {src[sid]:<28} {removed[sid]:>6} instances deleted")

    if args.dry_run:
        print("\nDry run: nothing was written.")
        return

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "class_mapping.txt", "w", encoding="utf-8") as f:
        f.write("Fire-ART id | Fire-ART name | -> your id | your name\n")
        for sid in sorted(src):
            if sid in mapping:
                f.write(f"{sid} | {src[sid]} | {mapping[sid]} | {dst[mapping[sid]]}\n")
            else:
                f.write(f"{sid} | {src[sid]} | REMOVED\n")
    print(f"\nWritten to {out}/ (mapping saved as class_mapping.txt)")


if __name__ == "__main__":
    main()
