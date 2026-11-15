"""
STEP 1 — Look before you extract.

E57 is a standard, but different scanners/software populate it slightly
differently. Before writing extraction code, run this on YOUR OWN file to see
exactly what's inside it: how many scans, how many embedded images, and which
representation each image uses (spherical / pinhole / visual reference).

Usage:
    python inspect_e57.py path/to/your_scan.e57
"""
import sys
import pye57


def describe_node(node, indent=0):
    """Recursively print an E57 node's structure (names only, not big data)."""
    prefix = "  " * indent
    try:
        name = node.elementName()
    except Exception:
        name = "?"
    print(f"{prefix}- {name} ({type(node).__name__})")
    # only descend into structures/vectors, not into blobs or big arrays
    if hasattr(node, "childCount"):
        for i in range(min(node.childCount(), 20)):  # cap for sanity
            describe_node(node[i], indent + 1)


def main(path):
    e57 = pye57.E57(path)
    imf = e57.image_file
    root = imf.root()

    print(f"E57 library version: {e57.header}" if hasattr(e57, "header") else "")

    # ---- point cloud scans ----
    data3d = root["data3D"]
    print(f"\n== data3D: {len(data3d)} scan(s) ==")
    for i in range(len(data3d)):
        header = e57.get_header(i)
        print(f"  scan {i}: {header.point_count} points")
        print(f"    rotation_matrix=\n{header.rotation_matrix}")
        print(f"    translation={header.translation}")

    # ---- embedded images ----
    images2d = root["images2D"]
    print(f"\n== images2D: {len(images2d)} image(s) ==")
    for i in range(len(images2d)):
        img_node = images2d[i]
        print(f"\n  image {i}:")
        describe_node(img_node, indent=2)

    print(
        "\nLook at the 'image N' sections above. Note which representation "
        "each one has (sphericalRepresentation = full 360 pano, "
        "pinholeRepresentation = normal perspective photo, "
        "cubeFaceRepresentation-style fields = separate cube-map faces you'd "
        "need to stitch). That tells extract_e57.py exactly what to pull out."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_e57.py path/to/your_scan.e57")
        sys.exit(1)
    main(sys.argv[1])
