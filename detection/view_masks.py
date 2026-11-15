r"""
view_masks.py -- look at the masks saved by pano_inference.py, at full resolution.

Reads <out>\settings.json (to find the panos), <out>\json\<pano>.json and
<out>\masks\<pano>.npz, and writes into <out>\mask_views\<pano>\ :

    instances.jpg     full-resolution pano, every instance mask filled and outlined,
                      labelled I<id> class conf | sub-type (n views)
    detections.jpg    the same for the face detections BEFORE merging
                      (D<id>, colour = instance it went to), to compare with the above
    I<id>.jpg         close-up of one instance, three panels:
                        pano crop | instance mask | outlines of every view merged into it
                      (dashed = cut off at a face border), with a legend

Nothing in the inference output is changed.

Usage:
    python view_masks.py "D:\...\test_panos_inference_20261029-1500"
    python view_masks.py "<out>" --pano pano_003.jpg pano_017.jpg   only these panos
    python view_masks.py "<out>" --show       also open instances.jpg in a zoomable window
    python view_masks.py "<out>" --pano-dir "D:\...\test_panos"     if the panos moved
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
           (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
           (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]
ALPHA = 0.45


def colour(i):
    return PALETTE[int(i) % len(PALETTE)]


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        p.write_bytes(buf.tobytes())


def label(img, txt, x, y, col, scale=0.7, taken=None):
    t = max(1, round(2 * scale / 0.6))
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    x = int(min(max(0, x), img.shape[1] - tw - 4))
    y = int(min(max(th + 6, y), img.shape[0] - 2))
    if taken is not None:
        for _ in range(50):
            box = (x, y - th - 6, x + tw + 4, y)
            hit = [b for b in taken if box[0] < b[2] and b[0] < box[2]
                   and box[1] < b[3] and b[1] < box[3]]
            if not hit:
                break
            y = min(b[1] for b in hit) - 1
            if y - th - 6 < 0:
                y = max(b[3] for b in hit) + th + 7
        taken.append((x, y - th - 6, x + tw + 4, y))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), col, -1)
    cv2.putText(img, txt, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), t, cv2.LINE_AA)


def to_bool(pix, w, h):
    m = np.zeros(h * w, bool)
    m[pix] = True
    return m.reshape(h, w)


def contours(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    return cnts


def dashed(img, cnt, col, thick=2, on=8):
    pts = cnt.reshape(-1, 2)
    for i in range(0, len(pts) - 1, 2 * on):
        seg = pts[i:i + on + 1]
        if len(seg) > 1:
            cv2.polylines(img, [seg.reshape(-1, 1, 2)], False, col, thick, cv2.LINE_AA)


def sub_txt(x):
    if not x.get("subtype"):
        return ""
    unc = "? " if x.get("subtype_status") == "uncertain" else ""
    return f' | {unc}{x["subtype"]} {float(x["p_subtype"]):.2f}'


def overlay(pano, items, key_col, text_of, w, h):
    """Fill + outline every pixel set in items on a copy of the pano."""
    over = pano.copy()
    flat = over.reshape(-1, 3)
    for it in items:
        flat[it["pix"]] = key_col(it)
    img = cv2.addWeighted(over, ALPHA, pano, 1 - ALPHA, 0)
    taken = []
    for it in items:
        m = to_bool(it["pix"], w, h)
        cv2.drawContours(img, contours(m), -1, key_col(it), 2, cv2.LINE_AA)
    for it in items:
        v, u = np.divmod(it["pix"], w)
        x = int(u.min()) if u.max() - u.min() < w / 2 else int(u[u > w / 2].min())
        label(img, text_of(it), x, int(v.min()) - 3, key_col(it), 0.6, taken)
    return img


def crop_wrap(img, u0, u1, v0, v1):
    """Columns u0..u1 (may run past the seam), rows v0..v1."""
    cols = np.arange(u0, u1 + 1) % img.shape[1]
    return img[v0:v1 + 1][:, cols]


def instance_tile(pano, inst, dets_by_id, npz, w, h, margin=40, min_side=260):
    pix = npz[f'inst_{inst["instance"]}']
    members = [dets_by_id[i] for i in inst["detections"]]
    allpix = np.concatenate([pix] + [npz[f'det_{d["detection"]}'] for d in members])
    v, u = np.divmod(allpix, w)
    ref = int(u[0])
    uu = ref + (u - ref + w // 2) % w - w // 2
    u0, u1 = int(uu.min()) - margin, int(uu.max()) + margin
    v0, v1 = max(0, int(v.min()) - margin), min(h - 1, int(v.max()) + margin)

    def local(p):
        pv, pu = np.divmod(p, w)
        pu = ref + (pu - ref + w // 2) % w - w // 2
        m = np.zeros((v1 - v0 + 1, u1 - u0 + 1), np.uint8)
        m[pv - v0, pu - u0] = 1
        return m

    base = crop_wrap(pano, u0, u1, v0, v1)
    s = max(1.0, min_side / min(base.shape[:2]))
    size = (round(base.shape[1] * s), round(base.shape[0] * s))
    up = lambda a, interp=cv2.INTER_LINEAR: cv2.resize(a, size, interpolation=interp)

    p1 = up(base)
    mi = up(local(pix), cv2.INTER_NEAREST).astype(bool)
    col = colour(inst["class_id"])
    p2 = p1.copy()
    p2[mi] = (0.5 * p2[mi] + 0.5 * np.array(col)).astype(np.uint8)
    cv2.drawContours(p2, contours(mi), -1, col, 2, cv2.LINE_AA)
    p3 = (p1 * 0.6).astype(np.uint8)
    legend = []
    for n, d in enumerate(members):
        c = PALETTE[(n * 3 + 1) % len(PALETTE)]
        md = up(local(npz[f'det_{d["detection"]}']), cv2.INTER_NEAREST)
        for cnt in contours(md):
            if int(d["cut_off"]):
                dashed(p3, cnt, c)
            else:
                cv2.drawContours(p3, [cnt], -1, c, 2, cv2.LINE_AA)
        legend.append((c, f'D{d["detection"]} {d["face"]} {d["class_name"]} '
                          f'{float(d["conf"]):.2f}{" (edge)" if int(d["cut_off"]) else ""}'))
    gap = np.full((p1.shape[0], 6, 3), 255, np.uint8)
    tile = np.hstack([p1, gap, p2, gap, p3])
    head = np.full((34, tile.shape[1], 3), 40, np.uint8)
    label(head, f'I{inst["instance"]} {inst["class_name"]} {float(inst["conf"]):.2f}'
                f'{sub_txt(inst)}  {inst["n_views"]} views, votes: '
                f'{", ".join(f"{k} {v:.2f}" for k, v in inst["votes"].items())}',
          4, 28, (40, 40, 40), 0.55)
    foot = np.full((24 * len(legend) + 8, tile.shape[1], 3), 255, np.uint8)
    for n, (c, txt) in enumerate(legend):
        y = 22 + 24 * n
        cv2.line(foot, (8, y - 6), (40, y - 6), c, 3)
        cv2.putText(foot, txt, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return np.vstack([head, tile, foot])


def main():
    ap = argparse.ArgumentParser(description="Draw the masks saved by pano_inference.py.")
    ap.add_argument("out", type=Path, help="Output folder of pano_inference.py")
    ap.add_argument("--pano", nargs="+", default=None, help="Only these panos (file names)")
    ap.add_argument("--pano-dir", type=Path, default=None,
                    help="Pano folder, if it moved since the inference run")
    ap.add_argument("--show", action="store_true",
                    help="Also open instances.jpg of each pano in a zoomable window")
    a = ap.parse_args()

    settings = json.loads((a.out / "settings.json").read_text(encoding="utf-8"))
    pano_dir = a.pano_dir or Path(settings["pano_dir"])
    jsons = sorted((a.out / "json").glob("*.json"))
    if a.pano:
        want = {Path(p).stem for p in a.pano}
        jsons = [j for j in jsons if j.stem in want]
    if not jsons:
        sys.exit(f"No matching results in {a.out / 'json'}")
    if a.show:
        import matplotlib.pyplot as plt

    for jp in jsons:
        res = json.loads(jp.read_text(encoding="utf-8"))
        pano = imread(pano_dir / res["pano"])
        if pano is None:
            print(f"{res['pano']}: pano not found in {pano_dir} (use --pano-dir)")
            continue
        h, w = pano.shape[:2]
        npz = np.load(a.out / "masks" / f"{jp.stem}.npz")
        dest = a.out / "mask_views" / jp.stem

        insts = [{**k, "pix": npz[f'inst_{k["instance"]}']} for k in res["instances"]]
        dets = [{**d, "pix": npz[f'det_{d["detection"]}']} for d in res["detections"]]
        imwrite(dest / "instances.jpg", overlay(
            pano, insts, lambda k: colour(k["class_id"]),
            lambda k: f'I{k["instance"]} {k["class_name"]} {float(k["conf"]):.2f}'
                      f'{sub_txt(k)} ({k["n_views"]} views)', w, h))
        imwrite(dest / "detections.jpg", overlay(
            pano, dets, lambda d: colour(3 * int(d["instance"]) + 1),
            lambda d: f'D{d["detection"]} {d["class_name"]} {float(d["conf"]):.2f} '
                      f'-> I{d["instance"]}{" edge" if int(d["cut_off"]) else ""}', w, h))
        by_id = {d["detection"]: d for d in res["detections"]}
        for k in res["instances"]:
            imwrite(dest / f'I{k["instance"]}.jpg',
                    instance_tile(pano, k, by_id, npz, w, h))
        print(f"{res['pano']}: {len(insts)} instances, {len(dets)} detections -> {dest}")

        if a.show:
            img = imread(dest / "instances.jpg")
            plt.figure(figsize=(18, 8))
            plt.imshow(img[:, :, ::-1])
            plt.title(f"{res['pano']} (zoom with the toolbar)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
