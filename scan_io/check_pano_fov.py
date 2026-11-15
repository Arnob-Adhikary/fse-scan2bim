"""
DEBUG STEP 3 -- Find the panorama's real angular field of view.

A 5084x2083 image isn't 2:1, meaning it doesn't cover the full sphere --
almost certainly the scanner crops what's straight below it (blocked by the
tripod/scan head). The row<->elevation formula needs the ACTUAL angular
extent, not an assumed full +/-90 degrees.

sphericalRepresentation stores pixelWidth/pixelHeight as the angular size of
one pixel, in radians. Multiplying by imageWidth/imageHeight gives the true
total field of view.

Usage:
    python check_pano_fov.py path/to/your_scan.e57
"""
import sys
import numpy as np
import pye57


def main(e57_path):
    e57 = pye57.E57(e57_path)
    images2d = e57.image_file.root()["images2D"]

    for i in range(len(images2d)):
        node = images2d[i]
        if not node.isDefined("sphericalRepresentation"):
            print(f"image {i}: no sphericalRepresentation, skipping")
            continue

        rep = node["sphericalRepresentation"]
        img_w = rep["imageWidth"].value()
        img_h = rep["imageHeight"].value()
        px_w = rep["pixelWidth"].value()   # radians per pixel, horizontally
        px_h = rep["pixelHeight"].value()  # radians per pixel, vertically

        total_az_range = px_w * img_w
        total_el_range = px_h * img_h

        print(f"image {i}:")
        print(f"  imageWidth={img_w}, imageHeight={img_h}")
        print(f"  pixelWidth={px_w:.8f} rad/px, pixelHeight={px_h:.8f} rad/px")
        print(f"  => total azimuth range: {np.degrees(total_az_range):.2f} deg "
              f"(expect ~360)")
        print(f"  => total elevation range: {np.degrees(total_el_range):.2f} deg "
              f"(this is the number we actually need)")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_pano_fov.py path/to/your_scan.e57")
        sys.exit(1)
    main(sys.argv[1])
