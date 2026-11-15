"""
Project panorama segmentation predictions (YOLO26-seg + stage-2 attributes) onto the
floor point cloud, fuse the views into 3D object instances, and write a per-point
prediction in the SAME point order as the ground truth (gt_labelled.ply), so the two
can be compared point by point.

Per panorama (scan station):
  1. take the floor-cloud points within --max-range of the station and project them into
     the panorama's equirectangular grid (same convention as project_mask_to_3d.py,
     including the mirrored azimuth);
  2. build a depth image with point splatting (each point covers the pixels its 5 mm
     footprint covers, so the 5 mm cloud has no holes even close to the scanner) and keep
     only the points that are visible from this station (range <= depth + tolerance).
     Points behind walls, or behind the object, never get the label;
  3. every visible point that falls inside a predicted mask gets that detection.
     Mask edges can be eroded (--erode px); leaked background (e.g. through a door
     next to the object) is removed by splitting the detection's points at range gaps
     > --range-gap and keeping the nearest part that covers >= 30% of the mask; bleed onto a
     wall/ceiling behind a small object (seen at a grazing angle, no range gap) is removed
     by dropping points farther than 0.6 x the class's max size from the nearest quarter
     of the mask (the object sticks out towards the scanner);
  4. overlapping detections of different classes on the SAME object in one panorama
     (YOLO runs NMS per class) are merged into the most confident one; the others
     become class votes.
Across panoramas:
  5. detections from different panoramas are grouped into one object when their 3D
     points overlap (>= --overlap of the smaller one) or their centres lie closer than a
     class-specific distance (double-sided hanging signs: front and back share no points).
     Two detections from the same panorama are never grouped (two different objects);
  6. class = confidence-weighted vote over the views (so a Fire Alarm/Fire Indicator
     confusion in one view is out-voted by the others); attribute (subclass) = weighted
     vote over the views that gave one, "unknown" ignored;
  7. a point belongs to the object if at least --point-vote of the views that SAW it
     (inside a dilated mask) put it inside their mask; points claimed by two objects go
     to the one with more support; the largest connected part is kept (DBSCAN);
  8. multi-view consistency: how many stations within --seen-range saw the object's
     points, and how many of them detected it (a detection in 1 of 6 nearby views is
     likely a false positive; use --min-consistency to drop those, default off).

Predictions (auto-detected per file in --pred-dir, one file per panorama;
file names start with S001_pano... or pano_000... like the click annotations):
  pano_inference.py output folder (give the folder itself as --pred-dir):
          json/<pano>.json (instances: class, conf, subtype + subtype_probs) and
          masks/<pano>.npz (inst_<id> = exact pano pixels). The INSTANCES are used
          (copies across cube faces already merged); the per-class thresholds are
          already applied, so --min-conf defaults to 0. Stage-2 probabilities are summed
          over the stations, so a sub-type that was 'uncertain' in each pano can still be
          settled in 3D (attribute_p = summed share; flagged below --min-prob).
          Without the npz, the instance outlines in the JSON are used.
  *.json  X-AnyLabeling / LabelMe: polygon (or rectangle / rotation) shapes;
          label "5 = Smoke Detector", "Smoke Detector", "smoke_detector" or "5";
          attribute after "|" (e.g. "0 = Fire Extinguisher | CO2") or in the
          description / attributes field; confidence in "score" (missing = 1.0).
  *.txt   YOLO segmentation txt (Ultralytics save_txt, with or without save_conf):
          "cls x1 y1 x2 y2 ... [conf]" normalised. Image size from the JSON-less
          panorama in --pano-dir or from --image-size W H.
  *.json  list of detections (e.g. 03_segment_image.py's *_detections.json):
          [{"class" or "class_id", "confidence", "subclass" (optional),
            "mask_file": "x.npy|x.png"  or  "polygon": [[u, v], ...]}, ...]
Optional --attributes CSV (file,det_index,subclass) adds stage-2 attributes to
YOLO txt predictions (det_index = line number, starting at 0).

Inputs
  --pred-dir    prediction files (above)
  --pose-dir    S001_pano_pose.json / pano_000_pose.json from extract_e57.py
  --stations    stations.csv from station_map.py
  --cloud       the floor cloud the GT is defined on: floorN_5mm.ply, or gt_labelled.ply
                itself (then the point order is guaranteed to match)
  --floor       only panoramas of stations on this floor (stations.csv)

Outputs (in --out-dir)
  pred_labelled.ply       whole cloud + class_id (-1 = background), instance_id (0 = bg),
                          conf (float). Same point order as --cloud.
  pred_objects_only.ply   only the predicted object points
  instances_pred.csv      one row per predicted 3D object: class, attribute, centre, box,
                          n_points, views, confidence, class/attribute votes,
                          multi-view consistency, flags
  instances_pred.ply      one coloured point per object (load on top of instances_3d.ply)
  detections_3d.csv       every 2D detection: station, class, conf, mask px, 3D points,
                          range, centre, which object it went into (or why it was dropped)
  views/<pano>.ply        (--save-views) the points each panorama labelled, for checking

Usage
  python pano_seg_to_cloud.py --pred-dir D:\\hotel\\test_panos_inference_20261101-093000
         --pose-dir D:\\hotel\\extract
         --stations D:\\hotel\\stations\\stations.csv --cloud D:\\hotel\\floor1_5mm.ply
         --floor 1 --out-dir D:\\hotel\\pred3d\\floor1
  quick check on two stations first:  add  --only S012 S013 --save-views

Requires: pip install numpy scipy opencv-python open3d
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

CLASSES = ["Fire Extinguisher", "Fire Alarm", "Fire Hose Reel", "Fire Exit Sign",
           "Fire Indicator", "Smoke Detector", "First Aid Kit", "AED", "Fire Call Point",
           "Fire Blanket", "Emergency Door Release", "Evacuation Chair"]
COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
           (0, 128, 128), (170, 110, 40)]
# detections from two panoramas whose centres are closer than this (m) are one object,
# even without shared points (each view sees a different side). About half the object size.
MERGE_DIST = {0: 0.15, 1: 0.08, 2: 0.35, 3: 0.20, 4: 0.08, 5: 0.08, 6: 0.20, 7: 0.20,
              8: 0.08, 9: 0.15, 10: 0.08, 11: 0.40}


# largest plausible object size (m), as in cutout_instances.py; detection points farther
# than 0.6 x this from the object's core are dropped (mask bleed onto walls/ceilings behind)
MAX_SIZE = {0: 0.90, 1: 0.30, 2: 1.20, 3: 0.70, 4: 0.25, 5: 0.25, 6: 0.70, 7: 0.80,
            8: 0.25, 9: 0.50, 10: 0.25, 11: 1.60}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


NAME_TO_ID = {norm(c): i for i, c in enumerate(CLASSES)}
NAME_TO_ID.update({str(i): i for i in range(len(CLASSES))})


def class_id(label):
    """Accepts 'Smoke Detector', 'smoke_detector', '5', and '5 = Smoke Detector'."""
    lab = str(label).strip()
    if norm(lab) in NAME_TO_ID:
        return NAME_TO_ID[norm(lab)]
    m = re.match(r"^\s*(\d+)\s*[=:\-_.)\s]", lab + " ")
    if m and int(m.group(1)) < len(CLASSES):
        return int(m.group(1))
    tail = re.split(r"[=:]", lab)[-1]
    return NAME_TO_ID.get(norm(tail), -1)


def cname(cid):
    return CLASSES[cid] if 0 <= cid < len(CLASSES) else "UNKNOWN"


def known(sub):
    return norm(sub) not in ("", "unknown", "unk", "na", "none", "unclear")


# ---------------------------------------------------------------- stations and poses
def read_stations(path):
    rows = list(csv.DictReader(open(path, newline="")))
    by_station = {int(r["station"]): r for r in rows}
    by_image = {int(r["image_index"]): r for r in rows if r.get("image_index") not in ("", None)}
    return by_station, by_image


def station_of_file(fn, by_image):
    stem = os.path.basename(fn)
    m = re.match(r"^S(\d{3})", stem)
    if m:
        return int(m.group(1))
    m = re.match(r"^pano_(\d{3})", stem)
    if m and int(m.group(1)) in by_image:
        return int(by_image[int(m.group(1))]["station"])
    return None


def first_existing(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_pose(path):
    p = json.load(open(path))
    rot, tr = p.get("rotation"), p.get("translation")
    if isinstance(rot, dict):
        R = quat_to_R(rot["w"], rot["x"], rot["y"], rot["z"])
    elif "rotation_matrix" in p:
        R = np.asarray(p["rotation_matrix"], dtype=float)
    else:
        raise ValueError(f"no rotation in {path}")
    t = np.array([tr["x"], tr["y"], tr["z"]] if isinstance(tr, dict) else tr, dtype=float)
    return R, t


def image_size(path):
    """(W, H) of an image file; reads only the header when Pillow is installed."""
    try:
        from PIL import Image
        with Image.open(path) as f:
            return f.size
    except Exception:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None if im is None else (im.shape[1], im.shape[0])


# ---------------------------------------------------------------- reading predictions
def split_label(label):
    label, sub = str(label), ""
    if "|" in label:
        label, sub = [x.strip() for x in label.split("|", 1)]
    return label, sub


def read_attributes(path):
    out = {}
    if not path:
        return out
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        key = (os.path.splitext(os.path.basename(r["file"]))[0], int(r["det_index"]))
        out[key] = r.get("subclass", "") or r.get("attribute", "")
    return out


def read_predictions(path, size_hint):
    """Returns (W, H, list of raw detections) or raises ValueError.
    Raw detection: label, cls, conf, sub, polys (list of Nx2 pixel arrays) or mask (HxW)."""
    if path.lower().endswith(".txt"):
        W, H = size_hint() or (None, None)
        if W is None:
            raise ValueError("YOLO txt needs the image size: give --pano-dir or --image-size")
        dets = []
        for k, line in enumerate(open(path)):
            vals = line.split()
            if len(vals) < 5:
                continue
            cid = int(float(vals[0]))
            rest = [float(v) for v in vals[1:]]
            conf = 1.0
            if len(rest) % 2 == 1:
                conf, rest = rest[-1], rest[:-1]
            if len(rest) == 4:                     # box format: cx cy w h
                cx, cy, bw, bh = rest
                poly = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                                 [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
                kind = "box"
            else:
                poly, kind = np.asarray(rest).reshape(-1, 2), "polygon"
            poly = poly * [W, H]
            dets.append({"line": k, "label": str(cid), "cls": cid if cid < len(CLASSES) else -1,
                         "conf": conf, "sub": "", "polys": [poly], "kind": kind})
        return W, H, dets

    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict) and "instances" in d and "width" in d:        # pano_inference.py
        W, H = int(d["width"]), int(d["height"])
        stem = os.path.splitext(os.path.basename(path))[0]
        npz_path = first_existing([os.path.join(os.path.dirname(path), "..", "masks", stem + ".npz"),
                                   os.path.join(os.path.dirname(path), stem + ".npz")])
        npz = np.load(npz_path) if npz_path else None
        dets = []
        for it in d["instances"]:
            cid = class_id(it.get("class_name", ""))
            if cid < 0:
                cid = class_id(str(it.get("class_id", "")))
            sub = str(it.get("subtype") or "")
            probs = it.get("subtype_probs") or None
            det = {"line": int(it["instance"]), "label": it.get("class_name", ""), "cls": cid,
                   "conf": float(it["conf"]), "sub": sub, "sub_probs": probs,
                   "sub_status": it.get("subtype_status", ""), "kind": "inference mask"}
            key = f"inst_{it['instance']}"
            if npz is not None and key in npz.files:
                pix = npz[key].astype(np.int64)
                det["pix"] = (pix // W, pix % W)
            elif it.get("outline_pano"):
                det["polys"] = [np.asarray(c, float).reshape(-1, 2) for c in it["outline_pano"]]
                det["kind"] = "inference outline (no npz)"
            else:
                continue
            dets.append(det)
        return W, H, dets

    if isinstance(d, dict) and "shapes" in d:                          # X-AnyLabeling / LabelMe
        W, H = d.get("imageWidth"), d.get("imageHeight")
        if not W or not H:
            W, H = size_hint() or (None, None)
        dets = []
        for k, s in enumerate(d["shapes"]):
            pts = np.asarray(s.get("points", []), dtype=float).reshape(-1, 2)
            stype = s.get("shape_type", "polygon")
            if stype == "rectangle" and len(pts) == 2:
                (u0, v0), (u1, v1) = pts
                pts = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
            elif stype not in ("polygon", "rectangle", "rotation") or len(pts) < 3:
                continue                                                 # points, lines
            label, sub = split_label(s.get("label", ""))
            if not sub and str(s.get("description") or "").strip():
                sub = str(s["description"]).strip()
            if not sub and isinstance(s.get("attributes"), dict) and s["attributes"]:
                sub = " ".join(str(v) for v in s["attributes"].values() if str(v).strip())
            score = s.get("score")
            dets.append({"line": k, "label": label, "cls": class_id(label),
                         "conf": 1.0 if score is None else float(score), "sub": sub,
                         "polys": [pts], "kind": stype})
        return W, H, dets

    items = d.get("detections", d) if isinstance(d, dict) else d     # generic list
    if not isinstance(items, list):
        raise ValueError("unknown JSON layout")
    W = H = None
    if isinstance(d, dict):
        W, H = d.get("width") or d.get("imageWidth"), d.get("height") or d.get("imageHeight")
    if not W:
        W, H = size_hint() or (None, None)
    dets = []
    for k, it in enumerate(items):
        lab = it.get("class", it.get("class_id", it.get("label", "")))
        label, sub = split_label(lab)
        sub = sub or str(it.get("subclass") or it.get("attribute") or "")
        conf = float(it.get("confidence", it.get("conf", it.get("score", 1.0))))
        det = {"line": k, "label": label, "cls": class_id(label), "conf": conf, "sub": sub,
               "kind": "mask"}
        if it.get("mask_file"):
            mf = it["mask_file"]
            if not os.path.isabs(mf) and not os.path.exists(mf):
                mf = os.path.join(os.path.dirname(path), os.path.basename(mf))
            if mf.lower().endswith(".npy"):
                m = np.load(mf)
            else:
                import cv2
                m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            det["mask"] = np.asarray(m) > 0
            if W is None:
                H, W = det["mask"].shape
        else:
            poly = it.get("polygon") or it.get("points") or it.get("segments")
            if poly is None:
                continue
            det["polys"] = [np.asarray(poly, dtype=float).reshape(-1, 2)]
            det["kind"] = "polygon"
        dets.append(det)
    if W is None:
        raise ValueError("image size unknown: give --pano-dir or --image-size")
    return W, H, dets


def rasterise(det, W, H, erode, context):
    """Crop masks for one detection. Columns may run past W (object across the seam);
    lookups compare a point's column with both col and col + W.
    Returns dict(r0, c0, inner, ctx, n_px) or None."""
    import cv2
    pad = context + erode + 2
    if "mask" in det or "pix" in det:
        if "pix" in det:
            rr, cc = det["pix"]
        else:
            m = det["mask"]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            rr, cc = np.nonzero(m)
        if len(rr) == 0:
            return None
        if cc.max() - cc.min() > W / 2:              # touches both ends: across the seam
            cc = np.where(cc < W / 2, cc + W, cc)
        r0, c0 = max(rr.min() - pad, 0), cc.min() - pad
        crop = np.zeros((min(rr.max() + pad + 1, H) - r0, cc.max() + pad + 1 - c0), np.uint8)
        crop[rr - r0, cc - c0] = 1
    else:
        polys = []
        for p in det["polys"]:
            p = np.asarray(p, float).copy()
            if np.ptp(p[:, 0]) > W / 2:
                p[:, 0] = np.where(p[:, 0] < W / 2, p[:, 0] + W, p[:, 0])
            polys.append(p)
        if min(p[:, 0].min() for p in polys) < 0:      # unwrapped outline left of column 0
            polys = [p + [W, 0] for p in polys]
        allp = np.vstack(polys)
        r0 = int(max(np.floor(allp[:, 1].min()) - pad, 0))
        r1 = int(min(np.ceil(allp[:, 1].max()) + pad + 1, H))
        c0 = int(np.floor(allp[:, 0].min()) - pad)
        c1 = int(np.ceil(allp[:, 0].max()) + pad + 1)
        if r1 <= r0 or c1 <= c0:
            return None
        crop = np.zeros((r1 - r0, c1 - c0), np.uint8)
        for p in polys:
            q = np.round((p - [c0, r0]) * 16).astype(np.int32)       # 1/16 px precision
            cv2.fillPoly(crop, [q], 1, lineType=cv2.LINE_8, shift=4)
    n_px = int(crop.sum())
    if n_px == 0:
        return None
    inner = crop
    if erode > 0:
        e = cv2.erode(crop, np.ones((3, 3), np.uint8), iterations=erode)
        if e.sum() >= 0.3 * n_px:                    # small, far objects: don't erode away
            inner = e
    ctx = cv2.dilate(crop, np.ones((3, 3), np.uint8), iterations=context) if context else crop
    cols = (np.flatnonzero(crop.any(axis=0)) + c0) % W      # wrapped image columns
    return {"r0": r0, "c0": c0, "inner": inner.astype(bool), "ctx": ctx.astype(bool), "n_px": n_px,
            "touch_l": bool(cols.min() <= 2), "touch_r": bool(cols.max() >= W - 3)}


# ---------------------------------------------------------------- geometry
class Grid2D:
    """Points bucketed in square xy cells, to fetch the neighbourhood of a station fast."""

    def __init__(self, P, cell=2.0):
        self.cell = cell
        self.lo = P[:, :2].min(axis=0)
        ij = np.floor((P[:, :2] - self.lo) / cell).astype(np.int64)
        self.nj = int(ij[:, 1].max()) + 1
        self.ni = int(ij[:, 0].max()) + 1
        key = ij[:, 0] * self.nj + ij[:, 1]
        self.order = np.argsort(key, kind="stable")
        self.starts = np.searchsorted(key[self.order], np.arange(self.ni * self.nj + 1))

    def near(self, xy, radius):
        i0, j0 = np.floor((xy - self.lo - radius) / self.cell).astype(int)
        i1, j1 = np.floor((xy - self.lo + radius) / self.cell).astype(int)
        parts = []
        for i in range(max(i0, 0), min(i1, self.ni - 1) + 1):
            a = i * self.nj + max(j0, 0)
            b = i * self.nj + min(j1, self.nj - 1) + 1
            if b > a:
                parts.append(self.order[self.starts[a]:self.starts[b]])
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def visible_points(P, idx, R, t, W, H, a):
    """Project points idx into the station's panorama; keep those visible from it.
    Returns (idx, row, col, range) of the visible points and the depth image."""
    from scipy.ndimage import minimum_filter
    rel = P[idx] - t.astype(P.dtype)
    loc = rel @ R.astype(P.dtype)                 # = (R.T @ rel.T).T: world -> scanner frame
    r = np.linalg.norm(loc, axis=1)
    ok = (r > a.min_range) & (r < a.max_range)
    idx, loc, r = idx[ok], loc[ok], r[ok]
    # NOTE: mirrored azimuth, as in project_mask_to_3d.py - do not change to atan2(y, x)
    az = np.arctan2(-loc[:, 1], loc[:, 0])
    el = np.arcsin(np.clip(loc[:, 2] / r, -1, 1))
    col = ((az + np.pi) / (2 * np.pi) * W).astype(np.int64) % W
    row = np.clip(((np.pi / 2 - el) / np.pi * H).astype(np.int64), 0, H - 1)

    # depth image with splatting: footprint half-width of each point in pixels
    half = 0.5 * a.spacing / r
    kx = np.clip(np.rint(half / (np.maximum(np.cos(el), 0.05) * 2 * np.pi / W)), 0, a.max_splat)
    ky = np.clip(np.rint(half / (np.pi / H)), 0, a.max_splat)
    key = (kx * (a.max_splat + 1) + ky).astype(np.int64)
    depth = np.full((H, W), np.inf, np.float32)
    for k in np.unique(key):
        s = np.flatnonzero(key == k)
        hx, hy = divmod(int(k), a.max_splat + 1)
        o = s[np.argsort(-r[s])]                  # nearest written last
        rr, cc = row[o], col[o]
        if hx == 0 and hy == 0:
            img = np.full((H, W), np.inf, np.float32)
            img[rr, cc] = r[o]
            np.minimum(depth, img, out=depth)
            continue
        # only filter the band of rows this group touches (+ margin), full width (wraps)
        ra, rb = max(int(rr.min()) - hy, 0), min(int(rr.max()) + hy + 1, H)
        img = np.full((rb - ra, W), np.inf, np.float32)
        img[rr - ra, cc] = r[o]
        img = minimum_filter(img, size=(2 * hy + 1, 2 * hx + 1), mode=("nearest", "wrap"))
        np.minimum(depth[ra:rb], img, out=depth[ra:rb])
    tol = a.depth_tol + a.depth_tol_rel * r
    vis = r <= depth[row, col] + tol
    return idx[vis], row[vis], col[vis], r[vis].astype(np.float32), depth


POOL = 4


def pooled(depth):
    """4x4 min-pooled depth image (float16) kept per station for the consistency check."""
    H, W = depth.shape
    h, w = H // POOL, W // POOL
    d = depth[:h * POOL, :w * POOL].reshape(h, POOL, w, POOL).min(axis=(1, 3))
    return np.minimum(d, 6e4).astype(np.float16)


def sees(station, Q, a, frac=0.2):
    """Does this station see at least `frac` of the object points Q (scanner frame test
    against its pooled depth image)?"""
    R, t, W, H, dsm = station
    loc = (Q - t) @ R
    r = np.linalg.norm(loc, axis=1)
    az = np.arctan2(-loc[:, 1], loc[:, 0])
    el = np.arcsin(np.clip(loc[:, 2] / np.maximum(r, 1e-9), -1, 1))
    col = ((az + np.pi) / (2 * np.pi) * W).astype(np.int64) % W // POOL
    row = np.clip(((np.pi / 2 - el) / np.pi * H).astype(np.int64), 0, H - 1) // POOL
    col = np.minimum(col, dsm.shape[1] - 1)
    row = np.minimum(row, dsm.shape[0] - 1)
    # pooled = nearest surface in a 4x4 block, so allow the object's own depth (+10 cm)
    ok = (r > a.min_range) & (r <= dsm[row, col].astype(np.float32) + 0.10 + a.depth_tol_rel * r)
    return ok.mean() >= frac


def points_in_mask(m, vrow, vcol, vr, W):
    """Visible points inside a detection's crop masks."""
    h, w = m["inner"].shape
    c0 = m["c0"]
    inr = (vrow >= m["r0"]) & (vrow < m["r0"] + h)
    c = np.where(vcol >= c0, vcol, vcol + W) - c0          # seam: also try col + W
    inc = (c >= 0) & (c < w)
    s = np.flatnonzero(inr & inc)
    rr, cc = vrow[s] - m["r0"], c[s]
    return s[m["inner"][rr, cc]], s[m["ctx"][rr, cc]]


def range_filter(sel, vr, pix, gap, min_share=0.3):
    """Split the detection's points at range jumps > gap (object vs. what is seen behind it
    through bleed at the mask edge, or through a door next to it). Keep the NEAREST part
    that covers at least `min_share` of the mask's pixels (objects stick out towards the
    scanner); if none does, the part covering most pixels. Pixels, not points, are counted:
    far surfaces have more points per pixel."""
    if len(sel) < 2:
        return sel
    o = np.argsort(vr[sel])
    rs = vr[sel][o]
    cut = np.flatnonzero(np.diff(rs) > gap)
    if len(cut) == 0:
        return sel
    bounds = np.r_[0, cut + 1, len(rs)]
    parts = [o[a_:b_] for a_, b_ in zip(bounds[:-1], bounds[1:])]      # nearest first
    npx = np.array([len(np.unique(pix[sel][p])) for p in parts])
    total = len(np.unique(pix[sel]))
    ok = np.flatnonzero(npx >= min_share * total)
    best = parts[ok[0]] if len(ok) else parts[int(np.argmax(npx))]
    return sel[np.sort(best)]


# ---------------------------------------------------------------- fusion
class Groups:
    def __init__(self, n, files):
        self.p = list(range(n))
        self.files = [{f} for f in files]

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a == b:
            return True
        if self.files[a] & self.files[b]:
            return False                      # would put 2 detections of one panorama together
        self.p[b] = a
        self.files[a] |= self.files[b]
        return True


def overlap(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(np.intersect1d(a, b, assume_unique=True)) / min(len(a), len(b))


def min_dist(P, a, b):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return np.inf
    small, big = (a, b) if len(a) < len(b) else (b, a)
    return float(cKDTree(P[big]).query(P[small], k=1)[0].min())


def dedup_in_pano(dets, thr, P):
    """Within one panorama:
    - YOLO's NMS is per class: one object can get e.g. Fire Alarm 0.4 + Fire Indicator 0.6.
      Such duplicates are merged into the most confident one; the others become class votes.
    - an object across the image seam (left/right edge) comes as two masks of the same
      class; they are joined when their 3D points touch."""
    by_file = defaultdict(list)
    for d in dets:
        by_file[d["file"]].append(d)
    for fl, ds in by_file.items():
        ds.sort(key=lambda d: -d["conf"])
        kept = []
        for d in ds:
            for k in kept:
                if overlap(d["pts"], k["pts"]) >= thr:
                    k["votes"][d["cls"]] = max(k["votes"].get(d["cls"], 0), d["conf"])
                    if known(d["sub"]):
                        k["sub_votes"][d["sub"]] = max(k["sub_votes"].get(d["sub"], 0), d["conf"])
                    d["status"] = f"duplicate of {k['det_id']} (same panorama)"
                    break
                seam = (d.get("touch_l") and k.get("touch_r")) or (d.get("touch_r") and k.get("touch_l"))
                if seam and d["cls"] == k["cls"] and min_dist(P, d["pts"], k["pts"]) < 0.05:
                    k["pts"] = np.union1d(k["pts"], d["pts"])
                    k["ctx"] = np.union1d(k["ctx"], d["ctx"])
                    k["centre"] = np.median(P[k["pts"]], axis=0).astype(np.float64)
                    k["n_px"] += d["n_px"]
                    d["status"] = f"joined with {k['det_id']} (other half across the image seam)"
                    break
            else:
                kept.append(d)
    return [d for d in dets if not d["status"]]


def weighted_vote(votes):
    if not votes:
        return None, ""
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    summary = " ".join(f"{k}:{v:.2f}" for k, v in ranked)
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 1e-9:
        return "TIE", summary
    return ranked[0][0], summary


# ---------------------------------------------------------------- output
def write_ply(path, P, C, extra):
    n = len(P)
    dt = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    head = ["property double x", "property double y", "property double z"]
    if C is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    for name, arr in extra.items():
        if np.issubdtype(arr.dtype, np.floating):
            dt.append((name, "<f4"))
            head.append(f"property float {name}")
        else:
            dt.append((name, "<i4"))
            head.append(f"property int {name}")
    a = np.empty(n, dtype=dt)
    a["x"], a["y"], a["z"] = P[:, 0], P[:, 1], P[:, 2]
    if C is not None:
        a["red"], a["green"], a["blue"] = C[:, 0], C[:, 1], C[:, 2]
    for name, arr in extra.items():
        a[name] = arr
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n" + "\n".join(head) + "\nend_header\n").encode("ascii"))
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--floor", help="only panoramas of stations on this floor")
    ap.add_argument("--only", nargs="*", help="only these stations, e.g. S012 S013 (quick check)")
    ap.add_argument("--pano-dir", help="panorama images, only to read the image size for YOLO txt")
    ap.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--attributes", help="CSV file,det_index,subclass for YOLO txt predictions")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="ignore detections below this confidence (default 0: pano_inference.py "
                         "already applied the per-class thresholds)")
    ap.add_argument("--min-prob", type=float, default=0.8,
                    help="flag an object's attribute as uncertain below this share of the "
                         "summed votes/probabilities (default 0.8, as in pano_inference.py)")
    ap.add_argument("--max-range", type=float, default=15.0,
                    help="ignore cloud points farther than this from the station (m, default 15)")
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--spacing", type=float, default=0.008,
                    help="point footprint for the depth image (m); a bit above the 5 mm "
                         "subsampling so surfaces have no holes (default 0.008)")
    ap.add_argument("--max-splat", type=int, default=8, help="largest footprint half-width (px)")
    ap.add_argument("--depth-tol", type=float, default=0.02,
                    help="a point is visible if range <= depth + depth-tol + depth-tol-rel * range")
    ap.add_argument("--depth-tol-rel", type=float, default=0.005)
    ap.add_argument("--erode", type=int, default=0,
                    help="erode masks by this many px before projecting (default 0; try 1-2 if "
                         "masks spill over the object edges - tune against the GT)")
    ap.add_argument("--context", type=int, default=6,
                    help="dilate masks by this many px for 'saw this point' (default 6)")
    ap.add_argument("--range-gap", type=float, default=0.10,
                    help="split a detection's points at range jumps larger than this (m)")
    ap.add_argument("--no-size-filter", dest="size_filter", action="store_false",
                    help="keep detection points far from the object's core (default: drop "
                         "points farther than 0.6 x the class's max size, see MAX_SIZE)")
    ap.add_argument("--size-scale", type=float, default=1.0,
                    help="multiply MAX_SIZE for that filter (e.g. 1.3 if objects get cut)")
    ap.add_argument("--dedup", type=float, default=0.5,
                    help="same-panorama detections sharing this fraction of points are one object")
    ap.add_argument("--overlap", type=float, default=0.2,
                    help="detections from two panoramas sharing this fraction of points "
                         "(of the smaller one) are one object (default 0.2)")
    ap.add_argument("--per-class", action="store_true",
                    help="only group detections of the same class (default: any class, "
                         "then a weighted class vote)")
    ap.add_argument("--point-vote", type=float, default=0.5,
                    help="keep a point if this fraction of the views that saw it labelled it")
    ap.add_argument("--clean-eps", type=float, default=0.03,
                    help="DBSCAN distance (m) to keep the largest connected part; 0 = off")
    ap.add_argument("--min-points", type=int, default=10, help="drop objects with fewer points")
    ap.add_argument("--seen-range", type=float, default=6.0,
                    help="stations within this distance that see an object count for the "
                         "multi-view consistency (m, default 6)")
    ap.add_argument("--min-consistency", type=float, default=0.0,
                    help="drop objects detected by less than this fraction of the nearby "
                         "stations that saw them (default 0 = keep all; try 0.3)")
    ap.add_argument("--save-views", action="store_true", help="write views/<pano>.ply per panorama")
    a = ap.parse_args()

    import open3d as o3d
    os.makedirs(a.out_dir, exist_ok=True)
    by_station, by_image = read_stations(a.stations)
    attrs = read_attributes(a.attributes)
    only = {int(re.sub(r"\D", "", s)) for s in a.only} if a.only else None

    pred_dir = a.pred_dir
    if os.path.isdir(os.path.join(pred_dir, "json")) and os.path.isdir(os.path.join(pred_dir, "masks")):
        pred_dir = os.path.join(pred_dir, "json")          # pano_inference.py output folder
        print(f"pano_inference.py output: reading {pred_dir} + masks")
    files = sorted(glob.glob(os.path.join(pred_dir, "*.json")) +
                   glob.glob(os.path.join(pred_dir, "*.txt")))
    files = [f for f in files if not f.endswith("_pose.json")
             and os.path.basename(f) != "settings.json"]
    jobs, warnings = [], []
    for f in files:
        st = station_of_file(f, by_image)
        if st is None or st not in by_station:
            warnings.append(f"{os.path.basename(f)}: cannot tell which station - skipped")
            continue
        if a.floor and str(by_station[st].get("floor", "")) != str(a.floor):
            continue
        if only and st not in only:
            continue
        jobs.append((st, f))
    if not jobs:
        sys.exit("no prediction files for the chosen floor/stations - check --pred-dir names "
                 "(S001_pano... or pano_000...) and --floor")
    print(f"{len(jobs)} panoramas to project")

    print(f"loading {a.cloud} ...")
    pcd = o3d.io.read_point_cloud(a.cloud)
    P64 = np.asarray(pcd.points, dtype=np.float64)
    if len(P64) == 0:
        sys.exit(f"no points read from {a.cloud}")
    C = (np.asarray(pcd.colors) * 255).round().astype(np.uint8) if pcd.has_colors() else None
    del pcd
    shift = P64.mean(axis=0)                 # float32 is exact enough around the cloud centre
    P = (P64 - shift).astype(np.float32)
    print(f"  {len(P):,} points")
    grid = Grid2D(P)

    dets, station_xyz, station_view = [], {}, {}
    t_start = time.time()
    for n_job, (st, f) in enumerate(jobs, 1):
        name = os.path.basename(f)
        info = by_station[st]
        ii = int(info["image_index"]) if info.get("image_index") not in ("", None) else -1
        pose = first_existing([os.path.join(a.pose_dir, f"S{st:03d}_pano_pose.json"),
                               os.path.join(a.pose_dir, f"pano_{ii:03d}_pose.json")])
        if not pose:
            warnings.append(f"{name}: no pose file for station S{st:03d} - skipped")
            continue

        def size_hint(st=st, ii=ii):
            if a.image_size:
                return tuple(a.image_size)
            if a.pano_dir:
                img = first_existing([os.path.join(a.pano_dir, f"S{st:03d}_pano.*"),
                                      os.path.join(a.pano_dir, f"pano_{ii:03d}.*")])
                if img:
                    return image_size(img)
            return None

        try:
            W, H, raw = read_predictions(f, size_hint)
        except Exception as e:
            warnings.append(f"{name}: {e} - skipped")
            continue
        stem = os.path.splitext(name)[0]
        raw = [d for d in raw if d["conf"] >= a.min_conf]
        R, t = load_pose(pose)
        t_loc = t - shift
        station_xyz[st] = t_loc
        idx = grid.near(t_loc[:2], a.max_range)
        vidx, vrow, vcol, vr, depth = visible_points(P, idx, R, t_loc, W, H, a)
        station_view[st] = (R.astype(np.float32), t_loc.astype(np.float32), W, H, pooled(depth))
        del depth
        pix = vrow * W + vcol
        n_hit = 0
        view_lab = []
        for d in raw:
            det_id = len(dets) + 1
            if (stem, d["line"]) in attrs and not d["sub"]:
                d["sub"] = attrs[(stem, d["line"])]
            rec = {"det_id": det_id, "file": name, "station": st, "floor": info.get("floor", ""),
                   "label": d["label"], "cls": d["cls"], "conf": d["conf"], "sub": d["sub"],
                   "kind": d["kind"], "n_px": 0, "pts": np.zeros(0, np.int64),
                   "ctx": np.zeros(0, np.int64), "range": np.nan, "centre": None, "status": "",
                   "votes": {d["cls"]: d["conf"]},
                   "sub_votes": {d["sub"]: d["conf"]} if known(d["sub"]) else {}}
            if d.get("sub_probs"):
                # stage-2 probabilities (pano_inference.py): sum them over the views, so
                # a sub-type that is 'uncertain' in each pano can still be settled in 3D
                rec["sub_votes"] = {k: float(v) for k, v in d["sub_probs"].items()}
            dets.append(rec)
            if d["cls"] < 0:
                rec["status"] = f"unknown label '{d['label']}'"
                continue
            m = rasterise(d, W, H, a.erode, a.context)
            if m is None:
                rec["status"] = "empty mask"
                continue
            rec["n_px"] = m["n_px"]
            rec["touch_l"], rec["touch_r"] = m["touch_l"], m["touch_r"]
            sel, ctx = points_in_mask(m, vrow, vcol, vr, W)
            sel = range_filter(sel, vr, pix, a.range_gap)
            if a.size_filter and len(sel) and d["cls"] in MAX_SIZE:
                # anchor = the nearest quarter of the mask's pixels: the object sticks out from
                # the wall/ceiling towards the scanner, bleed lands on surfaces behind it
                o_ = np.lexsort((vr[sel], pix[sel]))
                first = o_[np.r_[True, pix[sel][o_][1:] != pix[sel][o_][:-1]]]
                pmin = vr[sel][first]
                near_px = sel[first[pmin <= np.percentile(pmin, 25)]]
                anchor = np.median(P[vidx[near_px]], axis=0)
                dd = np.linalg.norm(P[vidx[sel]] - anchor, axis=1)
                sel = sel[dd <= 0.6 * MAX_SIZE[d["cls"]] * a.size_scale]
            if len(sel) == 0:
                rec["status"] = "no visible cloud points in the mask"
                continue
            rec["pts"] = np.sort(vidx[sel])
            rec["ctx"] = np.unique(np.r_[vidx[ctx], rec["pts"]])
            rec["range"] = float(np.median(vr[sel]))
            rec["centre"] = np.median(P[rec["pts"]], axis=0).astype(np.float64)
            n_hit += 1
            view_lab.append((rec["pts"], d["cls"]))
        if a.save_views and view_lab:
            os.makedirs(os.path.join(a.out_dir, "views"), exist_ok=True)
            ii_ = np.concatenate([p for p, _ in view_lab])
            cc_ = np.concatenate([np.full(len(p), c) for p, c in view_lab])
            rgb = np.array([COLOURS[c] for c in cc_], np.uint8)
            write_ply(os.path.join(a.out_dir, "views", f"{stem}.ply"), P64[ii_], rgb,
                      {"class_id": cc_.astype(np.int32)})
        el = time.time() - t_start
        print(f"  [{n_job}/{len(jobs)}] {name}: S{st:03d}, {len(raw)} detections >= {a.min_conf}, "
              f"{n_hit} with 3D points, {len(vidx):,} visible points  ({el:.0f} s)")

    # ---- fuse
    live = dedup_in_pano([d for d in dets if len(d["pts"]) and not d["status"]], a.dedup, P)
    print(f"\n{len(live)} detections with 3D points after same-panorama de-duplication")
    from scipy.spatial import cKDTree
    groups = Groups(len(live), [d["file"] for d in live])
    if len(live) > 1:
        cen = np.array([d["centre"] for d in live])
        pairs = []
        for i, j in cKDTree(cen).query_pairs(1.5):
            di, dj = live[i], live[j]
            if di["file"] == dj["file"]:
                continue
            if a.per_class and di["cls"] != dj["cls"]:
                continue
            ov = overlap(di["pts"], dj["pts"])
            dist = float(np.linalg.norm(cen[i] - cen[j]))
            near = dist < min(MERGE_DIST.get(di["cls"], 0.1), MERGE_DIST.get(dj["cls"], 0.1))
            if ov >= a.overlap or near:
                pairs.append((-ov, dist, i, j))
        pairs.sort()
        refused = 0
        for _, _, i, j in pairs:
            refused += not groups.union(i, j)
        if refused:
            print(f"  {refused} links refused (would join two detections of one panorama)")
    members = defaultdict(list)
    for i in range(len(live)):
        members[groups.find(i)].append(i)

    objs = []
    for g in members.values():
        ds = [live[i] for i in g]
        votes, sub_votes = defaultdict(float), defaultdict(float)
        for d in ds:
            for c, v in d["votes"].items():
                votes[c] += v
            for s_, v in d["sub_votes"].items():
                sub_votes[s_] += v
        cls, cls_summary = weighted_vote(dict(votes))
        if cls == "TIE":
            cls = max(ds, key=lambda d: d["conf"])["cls"]
            cls_summary += " (tie: most confident view)"
        sub, sub_summary = weighted_vote(dict(sub_votes))
        sub_share = (sub_votes[sub] / sum(sub_votes.values())) if sub not in (None, "TIE") else None
        allp = np.concatenate([d["pts"] for d in ds])
        allc = np.concatenate([d["ctx"] for d in ds])
        up, pos = np.unique(allp, return_counts=True)
        uc, cnt = np.unique(allc, return_counts=True)
        saw = np.maximum(cnt[np.searchsorted(uc, up)], pos)
        keep = up[pos >= a.point_vote * saw - 1e-9]
        support = pos[pos >= a.point_vote * saw - 1e-9]
        objs.append({"dets": ds, "cls": cls, "cls_votes": cls_summary,
                     "sub": sub or ("unknown" if any(d["sub"] for d in ds) else ""),
                     "sub_votes": sub_summary, "sub_share": sub_share,
                     "pts": keep, "support": support,
                     "conf_sum": sum(max(d["votes"].values()) for d in ds)})

    # points claimed by two objects -> more views (then more confidence) wins
    if objs:
        pid = np.concatenate([o["pts"] for o in objs])
        oid = np.concatenate([np.full(len(o["pts"]), k) for k, o in enumerate(objs)])
        sc = np.concatenate([o["support"] + o["conf_sum"] / 1000 for o in objs])
        order = np.lexsort((-sc, pid))            # by point, best object first
        pid, oid = pid[order], oid[order]
        first = np.r_[True, pid[1:] != pid[:-1]]
        winner = oid[np.maximum.accumulate(np.where(first, np.arange(len(pid)), 0))]
        for o in objs:
            o["shared_with"] = set()
        if (~first).any():
            for o_lose, o_win in np.unique(np.c_[oid[~first], winner[~first]], axis=0):
                objs[o_lose]["shared_with"].add(int(o_win))
                objs[o_win]["shared_with"].add(int(o_lose))
        wp, wo = pid[first], oid[first]
        o_order = np.argsort(wo, kind="stable")
        splits = np.searchsorted(wo[o_order], np.arange(len(objs) + 1))
        wp = wp[o_order]
        for k, o in enumerate(objs):
            o["pts"] = wp[splits[k]:splits[k + 1]]

    # largest connected part
    for o in objs:
        o["note"] = []
        if a.clean_eps > 0 and len(o["pts"]) >= 20:
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[o["pts"]].astype(np.float64)))
            lab = np.asarray(pc.cluster_dbscan(eps=a.clean_eps, min_points=5))
            if (lab >= 0).any():
                big = np.bincount(lab[lab >= 0]).argmax()
                dropped = int((lab != big).sum())
                if dropped:
                    o["pts"] = o["pts"][lab == big]
                    if dropped > 0.2 * len(lab):
                        o["note"].append(f"{dropped} loose points removed")

    # multi-view consistency, final table
    for o in objs:
        if len(o["pts"]) == 0:
            o["centre"] = np.full(3, np.nan)
            continue
        Q = P[o["pts"]].astype(np.float64)
        lo, hi = np.percentile(Q, 0.5, axis=0), np.percentile(Q, 99.5, axis=0)
        o["centre"], o["size"] = (lo + hi) / 2, hi - lo
        o["stations"] = sorted({d["station"] for d in o["dets"]})
        near = [s for s, xyz in station_xyz.items()
                if np.linalg.norm(xyz - o["centre"]) < a.seen_range]
        sub = P[o["pts"][::max(1, len(o["pts"]) // 500)]]
        saw = {s for s in near if s in o["stations"] or sees(station_view[s], sub, a)}
        o["seen_by"] = len(saw)
        o["detected_near"] = len([s for s in o["stations"] if s in saw])
        o["consistency"] = (o["detected_near"] / o["seen_by"]) if o["seen_by"] else np.nan

    kept = []
    for o in objs:
        why = ""
        if len(o["pts"]) < a.min_points:
            why = f"dropped: {len(o['pts'])} points after fusion"
        elif a.min_consistency > 0 and o["seen_by"] >= 2 and o["consistency"] < a.min_consistency:
            why = (f"dropped: detected by {o['detected_near']} of {o['seen_by']} "
                   f"stations that saw it")
        if why:
            for d in o["dets"]:
                d["status"] = why
        else:
            kept.append(o)
    floor_of = lambda o: max({str(d["floor"]) for d in o["dets"]},
                             key=lambda fl: sum(str(d["floor"]) == fl for d in o["dets"]))
    kept.sort(key=lambda o: (floor_of(o), o["cls"], o["centre"][0], o["centre"][1]))
    old_to_new = {}
    for new_id, o in enumerate(kept, 1):
        o["id"] = new_id
        old_to_new[id(o)] = new_id
    for o in kept:
        for d in o["dets"]:
            d["status"] = f"object #{o['id']}"

    cls_f = np.full(len(P), -1, np.int32)
    ins_f = np.zeros(len(P), np.int32)
    conf_f = np.zeros(len(P), np.float32)
    rows = []
    for o in kept:
        ds = o["dets"]
        confs = [max(d["votes"].values()) for d in ds]
        cls_f[o["pts"]] = o["cls"]
        ins_f[o["pts"]] = o["id"]
        conf_f[o["pts"]] = max(confs)
        flags = list(o["note"])
        if len(ds) == 1:
            flags.append("single view")
        if len(o["cls_votes"].split()) > 1:
            flags.append("views disagree on class")
        if len({norm(d["sub"]) for d in ds if known(d["sub"])}) > 1:
            flags.append("views disagree on attribute")
        if o["sub_share"] is not None and o["sub_share"] < a.min_prob:
            flags.append(f"attribute uncertain ({o['sub_share']:.2f})")
        if o["seen_by"] >= 2 and o["consistency"] < 0.3:
            flags.append(f"detected by {o['detected_near']} of {o['seen_by']} nearby stations")
        if len({str(d["floor"]) for d in ds}) > 1:
            flags.append("seen from stations on different floors")
        shared = [old_to_new[id(objs[k])] for k in o.get("shared_with", ()) if id(objs[k]) in old_to_new]
        if shared:
            flags.append("touches #" + ",#".join(str(s) for s in sorted(shared)))
        c = o["centre"] + shift
        rows.append({"instance_id": o["id"], "class_id": o["cls"], "class": cname(o["cls"]),
                     "floor": floor_of(o), "x": f"{c[0]:.3f}", "y": f"{c[1]:.3f}", "z": f"{c[2]:.3f}",
                     "size_x": f"{o['size'][0]:.3f}", "size_y": f"{o['size'][1]:.3f}",
                     "size_z": f"{o['size'][2]:.3f}", "n_points": len(o["pts"]),
                     "n_views": len(ds), "stations": " ".join(f"S{s:03d}" for s in o["stations"]),
                     "conf_max": f"{max(confs):.3f}", "conf_mean": f"{np.mean(confs):.3f}",
                     "class_votes": " ".join(f"{cname(int(k))}:{v}" for k, v in
                                             (x.split(":") for x in o["cls_votes"].split()
                                              if ":" in x)),
                     "attribute": o["sub"],
                     "attribute_p": "" if o["sub_share"] is None else f"{o['sub_share']:.3f}",
                     "attribute_votes": o["sub_votes"],
                     "seen_by_near": o["seen_by"], "detected_by_near": o["detected_near"],
                     "consistency": "" if np.isnan(o["consistency"]) else f"{o['consistency']:.2f}",
                     "flags": "; ".join(flags)})

    # ---- write
    with open(os.path.join(a.out_dir, "instances_pred.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["instance_id", "class_id", "class", "floor", "x", "y", "z", "size_x", "size_y",
                "size_z", "n_points", "n_views", "stations", "conf_max", "conf_mean",
                "class_votes", "attribute", "attribute_p", "attribute_votes", "seen_by_near",
                "detected_by_near", "consistency", "flags"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out_dir, "detections_3d.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["det_id", "file", "station", "floor", "label", "class_id", "class", "conf",
                    "attribute", "shape", "mask_px", "n_points", "median_range_m",
                    "x", "y", "z", "result"])
        for d in dets:
            c = d["centre"] + shift if d["centre"] is not None else [np.nan] * 3
            w.writerow([d["det_id"], d["file"], d["station"], d["floor"], d["label"], d["cls"],
                        cname(d["cls"]), f"{d['conf']:.3f}", d["sub"], d["kind"], d["n_px"],
                        len(d["pts"]), "" if np.isnan(d["range"]) else f"{d['range']:.2f}",
                        *(f"{v:.3f}" for v in c), d["status"]])
    write_ply(os.path.join(a.out_dir, "pred_labelled.ply"), P64, C,
              {"class_id": cls_f, "instance_id": ins_f, "conf": conf_f})
    obj = np.flatnonzero(ins_f > 0)
    write_ply(os.path.join(a.out_dir, "pred_objects_only.ply"), P64[obj],
              C[obj] if C is not None else None,
              {"class_id": cls_f[obj], "instance_id": ins_f[obj], "conf": conf_f[obj]})
    cen = np.array([o["centre"] + shift for o in kept]).reshape(-1, 3)
    rgb = np.array([COLOURS[o["cls"]] for o in kept], np.uint8).reshape(-1, 3)
    write_ply(os.path.join(a.out_dir, "instances_pred.ply"), cen, rgb,
              {"class_id": np.array([o["cls"] for o in kept], np.int32),
               "instance_id": np.array([o["id"] for o in kept], np.int32)})

    # ---- summary
    print(f"{len(objs)} fused objects, {len(kept)} kept")
    per = defaultdict(int)
    for o in kept:
        per[o["cls"]] += 1
    for c in sorted(per):
        print(f"  {c:>2} {cname(c):<24} {per[c]}")
    nv = defaultdict(int)
    for o in kept:
        nv[min(len(o["dets"]), 5)] += 1
    print("  views per object: " + ", ".join(f"{k if k < 5 else '5+'}: {v}" for k, v in sorted(nv.items())))
    bad = [d for d in dets if not d["status"].startswith("object")]
    if bad:
        reasons = defaultdict(int)
        for d in bad:
            reasons[re.sub(r"\d+", "N", d["status"])] += 1
        print(f"{len(bad)} detections not in any object:")
        for r_, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>4}  {r_}")
    for wmsg in warnings:
        print("WARNING:", wmsg)
    print(f"{len(obj):,} points labelled. Outputs in {a.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
