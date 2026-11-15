r"""
pano_inference.py -- final case-study inference on scanner panoramas (D-19, D-22, D-26).

Per pano:
  1. 18 faces, cut with the training code (cubemap_convert_for_training.py,
     6 yaws x horizontal / +30 deg / -30 deg, 1024 px).
  2. YOLO26s-seg on every face. The per-class confidence thresholds from
     choose_thresholds.py (thresholds.json, D-26) are applied to each FACE detection,
     because that is the unit they were calibrated on.
  3. Every face mask is turned into a full-resolution pano mask: each pano pixel in the
     object's region looks up the face pixel it was sampled from (nearest, no blurring;
     objects across the pano seam are handled).
  4. Stage 2 on every Fire Extinguisher and Fire Exit Sign face detection:
     - extinguisher type: frozen DINOv2 probe (loaded as in classify_subtypes.py);
     - exit-sign direction, chosen by what --exit-model points to:
         PREFIX.json from train_yolo_cls_cv.py --final-only --save-final PREFIX
             -> YOLO classification (final choice); each crop is letterboxed to a
                grey square of the training imgsz, exactly as in training;
         a .pt from train_attribute_cnn_v2.py --save-final
             -> ResNet-18 (ablation);
       all seed models of the bundle are averaged; flip TTA only if the bundle was
       trained with it.
  5. Copies of one object in overlapping faces are merged into one INSTANCE:
       - detections are visited best first: complete views (box not touching the face
         border) before cut-off views, then by confidence;
       - each is compared with the representative (first) detection of every instance;
         it joins an instance when
           same class       and overlap / smaller mask  > --same-overlap (0.5)
           different class  and IoU                      > --cross-iou    (0.5)
           different class, cut off at the face border, and overlap / its own mask
                                                         > --same-overlap (0.5)
         (the last rule absorbs wrong labels on cut-off copies at face edges);
         same-class matches are preferred, then the largest overlap;
       - instance class   = largest sum of confidences over its views (votes kept);
         instance conf    = highest confidence among views of that class;
         instance mask    = best complete view of that class (all view masks are saved);
         instance subtype = stage-2 probabilities averaged over complete views of that
                            class (over all its views if none is complete).
     Adjacent objects (e.g. paired extinguishers) stay separate because their masks do
     not overlap.

Output folder (default: <pano-dir>_inference_<date-time>, next to --pano-dir):
    overview\<pano>.jpg        pano with instance masks and labels (I<id>)
    faces\<pano>__<face>.jpg   every face with detections; each detection labelled
                               D<id> and the instance it went to (-> I<id>), "edge" = cut off
    crops\<class>\[<subtype>|uncertain\]<pano>__I<id>.jpg   one crop per instance
    instances.csv              one row per instance (the pipeline output)
    detections.csv             one row per face detection after thresholds, with its
                               instance (for the before/after-merge comparison)
    json\<pano>.json           instances + detections with votes, probabilities, faces,
                               and instance outlines in pano pixels
    masks\<pano>.npz           pano masks as flat pixel indices (v * width + u):
                               inst_<id> and det_<id>, plus width/height (for 3D lifting
                               and for scoring against pano ground truth)
    settings.json              paths, SHA-256 of every model file, thresholds, merge
                               settings, library versions
    log.txt                    everything printed on screen

Keep this file next to cubemap_convert_for_training.py, classify_subtypes.py,
probe_separability_v4.py and train_attribute_cnn_v2.py (their functions are imported).

Usage (Command Prompt; ^ continues a line):
    python pano_inference.py ^
        --pano-dir   "D:\...\case_study\hotel\test_panos" ^
        --yolo       "D:\...\models\FINAL_y26s-seg_own+public_1024_train+heldout_best.pt" ^
        --thresholds "D:\...\experiments\thresholds_EXP007\thresholds.json" ^
        --probe      "D:\...\models\extinguisher_dinov2s_448.joblib" ^
        --exit-model "D:\...\models\exit_sign_yolo26n_cls.json"
    (the PREFIX_seed<N>.pt files are found through the .json, or next to it if moved;
    for the ResNet ablation pass its .pt instead)
    python pano_inference.py -h   lists every option.
"""
# Repository layout: helpers live in ../attributes and ../imagery.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
for _d in ("attributes", "imagery"):
    if str(_ROOT / _d) not in _sys.path:
        _sys.path.insert(0, str(_ROOT / _d))

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from cubemap_convert_for_training import (compute_face_map, face_configs,
                                              get_rotation_matrix)
    from classify_subtypes import crop_box, load_probe
    from probe_separability_v4 import letterbox, load_extractor
    from train_attribute_cnn_v2 import (build_model, flip_permutation,
                                        letterbox_rect, to_array)
except ImportError as e:
    sys.exit(f"Missing helper script ({e}). Put this file next to "
             "cubemap_convert_for_training.py, classify_subtypes.py, "
             "probe_separability_v4.py and train_attribute_cnn_v2.py.")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
NB_SPLITS = 6               # as in cubemap_convert_for_training.py
FACE_RES = 1024             # as in training (--resolution 1024)
EXT_NAME, EXIT_NAME = "Fire Extinguisher", "Fire Exit Sign"
OVERVIEW_WIDTH = 4000
FILL_ALPHA = 0.40
JPEG_QUALITY = 92


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Final case-study inference: YOLO26-seg with per-class thresholds, "
                    "stage-2 sub-types, merging of copies across cube faces.")
    g = ap.add_argument_group("paths")
    g.add_argument("--pano-dir", required=True, type=Path,
                   help="Folder with the panos (every jpg/png in it is used)")
    g.add_argument("--yolo", required=True, type=Path, help="Final YOLO best.pt")
    g.add_argument("--thresholds", required=True, type=Path,
                   help="thresholds.json from choose_thresholds.py (or its folder)")
    g.add_argument("--probe", required=True, type=Path,
                   help="Extinguisher probe .joblib (probe_separability_v4.py --save-probe)")
    g.add_argument("--exit-model", required=True, type=Path,
                   help="Exit-sign direction model: PREFIX.json from train_yolo_cls_cv.py "
                        "--save-final (YOLO-cls), or the .pt from "
                        "train_attribute_cnn_v2.py --save-final (ResNet)")
    g.add_argument("--exit-json", type=Path, default=None,
                   help="ResNet only: class list, if the .pt has none inside")
    g.add_argument("--out", type=Path, default=None,
                   help="Output folder (default: <pano-dir>_inference_<date-time>)")
    m = ap.add_argument_group("settings (defaults match training and the logbook)")
    m.add_argument("--imgsz", type=int, default=1024, help="YOLO input size (1024, D-17)")
    m.add_argument("--batch", type=int, default=6,
                   help="Faces per YOLO pass; lower if GPU memory runs out (6)")
    m.add_argument("--device", default=None, help="0 or cpu (default: auto)")
    m.add_argument("--min-prob", type=float, default=0.8,
                   help="Sub-types below this averaged probability are 'uncertain' (0.8)")
    m.add_argument("--pad", type=float, default=0.0,
                   help="Stage-2 crop margin, fraction of box; match training crops (0)")
    m.add_argument("--pad-px", type=float, default=0.0, help="Extra crop margin in px (0)")
    m.add_argument("--min-size", type=float, default=0.0,
                   help="Drop face detections whose shorter box side is below this many "
                        "px (D-16); 0 = off (default)")
    mg = ap.add_argument_group("merging copies across faces")
    mg.add_argument("--same-overlap", type=float, default=0.5,
                    help="Same class: overlap / smaller mask needed to merge (0.5)")
    mg.add_argument("--cross-iou", type=float, default=0.5,
                    help="Different class: IoU needed to merge (0.5)")
    mg.add_argument("--border-px", type=int, default=3,
                    help="A box within this many px of the face border is 'cut off' (3)")
    a = ap.parse_args()

    if a.thresholds.is_dir():
        a.thresholds = a.thresholds / "thresholds.json"
    for path, what in ((a.pano_dir, "--pano-dir"), (a.yolo, "--yolo"),
                       (a.thresholds, "--thresholds"), (a.probe, "--probe"),
                       (a.exit_model, "--exit-model")):
        if not path.exists():
            sys.exit(f"{what} not found: {path}")
    if a.device is not None and a.device.isdigit():
        a.device = int(a.device)
    return a


# =============================================================================
# Geometry: faces <-> pano (same convention as compute_face_map)
# =============================================================================
_MAP_CACHE = {}


def make_faces(equi):
    h, w = equi.shape[:2]
    faces = []
    for ftype, idx in face_configs(NB_SPLITS):
        key = (ftype, idx, h, w)
        if key not in _MAP_CACHE:
            _MAP_CACHE[key] = compute_face_map(ftype, idx, NB_SPLITS, FACE_RES, h, w)
        mx, my = _MAP_CACHE[key]
        faces.append({"name": f"{ftype}_{idx:02d}", "type": ftype, "idx": idx,
                      "bgr": cv2.remap(equi, mx, my, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_WRAP)})
    return faces


def face_to_pano(x, y, ftype, idx, w, h):
    """Face pixel(s) -> pano pixel(s), exactly as compute_face_map samples them."""
    R = get_rotation_matrix(ftype, idx, NB_SPLITS)
    f = FACE_RES / 2.0
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.stack([(x - FACE_RES / 2.0) / f, (y - FACE_RES / 2.0) / f,
                  np.ones_like(x)], -1) @ R.T
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    lon = np.arctan2(v[..., 0], v[..., 2])
    lat = np.arcsin(np.clip(v[..., 1], -1, 1))
    return (lon + math.pi) / (2 * math.pi) * w, (lat + math.pi / 2) / math.pi * h


def pano_to_face(u, v, ftype, idx, w, h):
    """Pano pixel(s) -> face pixel(s) (inverse of face_to_pano); ok = in front of the face."""
    lon = np.asarray(u, float) / w * 2 * math.pi - math.pi
    lat = np.asarray(v, float) / h * math.pi - math.pi / 2
    d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    c = d @ get_rotation_matrix(ftype, idx, NB_SPLITS)     # = R^T d
    z = c[..., 2]
    ok = z > 1e-9
    zs = np.where(ok, z, 1.0)
    f = FACE_RES / 2.0
    return c[..., 0] / zs * f + FACE_RES / 2.0, c[..., 1] / zs * f + FACE_RES / 2.0, ok


def unwrap_u(u, ref, w):
    """Column numbers made continuous around ref (for objects across the pano seam)."""
    return ref + (np.asarray(u, float) - ref + w / 2) % w - w / 2


def face_mask_to_pano(mask, ftype, idx, w, h):
    """Face mask (bool, FACE_RES x FACE_RES) -> sorted flat pano indices v * w + u."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.empty(0, np.int32)
    pts = np.concatenate([c.reshape(-1, 2) for c in cnts]).astype(float)
    u, v = face_to_pano(pts[:, 0], pts[:, 1], ftype, idx, w, h)
    uu = unwrap_u(u, u[0], w)
    u0, u1 = int(math.floor(uu.min())) - 2, int(math.ceil(uu.max())) + 2
    v0, v1 = max(0, int(math.floor(v.min())) - 2), min(h - 1, int(math.ceil(v.max())) + 2)
    gu, gv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    x, y, ok = pano_to_face(gu % w, gv, ftype, idx, w, h)
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    ok &= (xi >= 0) & (xi < FACE_RES) & (yi >= 0) & (yi < FACE_RES)
    hit = np.zeros(gu.shape, bool)
    hit[ok] = mask[yi[ok], xi[ok]]
    return np.unique(gv[hit] * w + gu[hit] % w).astype(np.int32)


def pix_stats(pix, w):
    """Centroid (u, v) and unwrapped column / row range of a pano pixel set."""
    u, v = pix % w, pix // w
    ref = float(u[0])
    uu = unwrap_u(u, ref, w)
    return {"cu": float(uu.mean() % w), "cv": float(v.mean()),
            "u0": float(uu.min()), "u1": float(uu.max()),
            "v0": int(v.min()), "v1": int(v.max())}


def ranges_overlap(a, b, w):
    if a["v1"] < b["v0"] or b["v1"] < a["v0"]:
        return False
    return any(a["u0"] + s <= b["u1"] and b["u0"] <= a["u1"] + s for s in (-w, 0, w))


def outline(pix, w):
    """Outer contour(s) of a pano pixel set in pano pixels (columns unwrapped, so an
    object across the seam can have u < 0 or u >= width)."""
    st = pix_stats(pix, w)
    u = np.rint(unwrap_u(pix % w, st["u0"] + 1, w)).astype(int)
    v = pix // w
    u0, v0 = u.min(), v.min()
    img = np.zeros((v.max() - v0 + 1, u.max() - u0 + 1), np.uint8)
    img[v - v0, u - u0] = 1
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(c.reshape(-1, 2) + [u0, v0]).tolist() for c in cnts]


# =============================================================================
# Stage 2 models (return full probability vectors)
# =============================================================================
class ExtinguisherProbe:
    def __init__(self, path):
        self.bundle = load_probe(path)
        self.clf, self.classes = self.bundle["classifier"], list(self.bundle["classes"])
        self.extract = load_extractor(self.bundle["model"], self.bundle["img_size"])

    def predict_proba(self, crops):
        return np.asarray(self.clf.predict_proba(self.extract(crops)), float)


class ExitSignModel:
    def __init__(self, path, json_path, device):
        import torch
        self.torch = torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "classes" in obj:
            meta = obj
        else:
            jp = Path(json_path) if json_path else Path(str(path) + ".classes.json")
            if not jp.is_file():
                sys.exit(f"{path} has no class list inside; pass --exit-json.")
            meta = json.loads(jp.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and obj.get("state_dicts"):
            sds = obj["state_dicts"]
        elif isinstance(obj, dict) and "state_dict" in obj:
            sds = [obj["state_dict"]]
        else:
            sds = [obj]
        self.classes = list(meta["classes"])
        self.h, self.w = int(meta.get("img_h", 192)), int(meta.get("img_w", 384))
        self.arch = meta.get("arch", "resnet18")
        self.device = device
        self.models = []
        for sd in sds:
            m, _, _ = build_model(self.arch, len(self.classes), pretrained=False)
            m.load_state_dict(sd)
            self.models.append(m.eval().to(device))
        self.flip_perm = None
        if meta.get("tta_flip"):
            swap = meta.get("swap_flip")
            swap_map = None
            if swap:
                a, b = (self.classes.index(s) for s in swap)
                swap_map = {a: b, b: a}
            self.flip_perm = flip_permutation(len(self.classes), swap_map)
        self.weight_files = []          # the bundle is the one .pt (hashed as exit_model)
        print(f"Exit-sign model (ResNet): {path}\n  {self.arch} {self.h}x{self.w} (HxW) "
              f"letterbox, {len(self.models)} model(s) averaged, TTA "
              f"{'on' if self.flip_perm else 'off'}, classes {self.classes}")

    def describe(self):
        return {"backend": "resnet (train_attribute_cnn_v2.py)", "arch": self.arch,
                "input_hw": [self.h, self.w], "preprocess": "letterbox grey to HxW",
                "classes": self.classes, "n_models": len(self.models),
                "tta_flip": self.flip_perm is not None}

    def predict_proba(self, crops):
        torch = self.torch
        x = torch.from_numpy(np.stack([to_array(letterbox_rect(c, self.w, self.h))
                                       for c in crops])).to(self.device)
        with torch.no_grad():
            p = 0
            for m in self.models:
                pm = torch.softmax(m(x), dim=1)
                if self.flip_perm is not None:
                    pf = torch.softmax(m(torch.flip(x, dims=[3])), dim=1)
                    pm = (pm + pf[:, self.flip_perm]) / 2
                p = p + pm
        return (p / len(self.models)).cpu().numpy().astype(float)


class YoloClsExitSign:
    """YOLO classification bundle from train_yolo_cls_cv.py --final-only --save-final
    PREFIX: PREFIX.json (classes, imgsz, letterbox, swap_flip, tta_flip, seeds, weights)
    plus PREFIX_seed<N>.pt. Preprocessing repeats training: the crop is letterboxed to
    an imgsz x imgsz grey square (probe_separability_v4.letterbox) and given to
    Ultralytics in BGR, as cv2 reads the training PNGs; mirrored copies for TTA are
    flipped after letterboxing, as in training."""

    def __init__(self, json_path, device):
        from ultralytics import YOLO
        json_path = Path(json_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("classes", "imgsz"):
            if key not in meta:
                sys.exit(f"{json_path} has no '{key}'; is it the .json written by "
                         "train_yolo_cls_cv.py --save-final?")
        self.meta = meta
        self.classes = list(meta["classes"])
        self.imgsz = int(meta["imgsz"])
        self.use_letterbox = bool(meta.get("letterbox", True))
        self.arch = str(meta.get("model", "yolo-cls"))
        self.device = device
        prefix = json_path.with_suffix("")
        files = []
        seeds = meta.get("seeds") or []
        listed = meta.get("weights") or []
        for n, sd in enumerate(seeds):
            p = Path(listed[n]) if n < len(listed) else None
            if p is None or not p.is_file():      # moved: look next to the .json
                p = prefix.parent / f"{prefix.name}_seed{sd}.pt"
            if not p.is_file():
                sys.exit(f"Exit-sign weights for seed {sd} not found ({p}). Keep the "
                         f"{prefix.name}_seed<N>.pt files next to {json_path.name}.")
            files.append(p)
        if not files:
            files = sorted(prefix.parent.glob(f"{prefix.name}_seed*.pt"))
        if not files:
            sys.exit(f"No {prefix.name}_seed<N>.pt files next to {json_path}")
        self.weight_files = files
        self.models, self.cols = [], []
        for p in files:
            m = YOLO(str(p))
            idx = {v: k for k, v in m.names.items()}
            missing = [c for c in self.classes if c not in idx]
            if missing:
                sys.exit(f"{p.name} has classes {list(m.names.values())}, but "
                         f"{json_path.name} lists {self.classes}")
            self.models.append(m)
            self.cols.append([idx[c] for c in self.classes])
        self.perm = None
        if meta.get("tta_flip"):
            swap = meta.get("swap_flip")
            names = {swap[0]: swap[1], swap[1]: swap[0]} if swap else {}
            self.perm = [self.classes.index(names.get(c, c)) for c in self.classes]
        print(f"Exit-sign model (YOLO-cls): {json_path}\n  {self.arch}, {self.imgsz} px "
              f"{'letterbox' if self.use_letterbox else 'no letterbox'}, "
              f"{len(self.models)} model(s) averaged ({', '.join(p.name for p in files)}), "
              f"TTA {'on' if self.perm else 'off'}, classes {self.classes}")

    def _prep(self, crop):
        im = crop.convert("RGB")
        if self.use_letterbox:
            im = letterbox(im, self.imgsz)
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])      # RGB -> BGR

    def _run(self, model, cols, arrs):
        out = []
        for i in range(0, len(arrs), 64):
            res = model.predict(source=arrs[i:i + 64], imgsz=self.imgsz,
                                device=self.device, verbose=False)
            out += [np.asarray(r.probs.data.cpu().numpy(), float)[cols] for r in res]
        return np.array(out)

    def predict_proba(self, crops):
        arrs = [self._prep(c) for c in crops]
        mirrored = [np.ascontiguousarray(a[:, ::-1]) for a in arrs] if self.perm else None
        p = 0
        for m, cols in zip(self.models, self.cols):
            pm = self._run(m, cols, arrs)
            if mirrored is not None:
                pm = (pm + self._run(m, cols, mirrored)[:, self.perm]) / 2
            p = p + pm
        return p / len(self.models)

    def describe(self):
        return {"backend": "yolo-cls (train_yolo_cls_cv.py)", "arch": self.arch,
                "imgsz": self.imgsz,
                "preprocess": "letterbox grey to square, BGR" if self.use_letterbox
                else "Ultralytics resize + centre crop",
                "classes": self.classes, "n_models": len(self.models),
                "seeds": self.meta.get("seeds"), "swap_flip": self.meta.get("swap_flip"),
                "tta_flip": self.perm is not None,
                "bundle_note": self.meta.get("note", "")}


def load_exit_model(path, json_path, device):
    if Path(path).suffix.lower() == ".json":
        return YoloClsExitSign(path, device)
    return ExitSignModel(path, json_path, device)


# =============================================================================
# Merging copies across faces
# =============================================================================
def merge(dets, args, w):
    order = sorted(range(len(dets)), key=lambda i: (dets[i]["cut"], -dets[i]["conf"]))
    groups = []                                   # [{"rep": i, "members": [...]}, ...]
    for i in order:
        d = dets[i]
        best = None
        for gi, g in enumerate(groups):
            r = dets[g["rep"]]
            if not ranges_overlap(d["st"], r["st"], w):
                continue
            inter = np.intersect1d(d["pix"], r["pix"], assume_unique=True).size
            if not inter:
                continue
            a, b = d["pix"].size, r["pix"].size
            iou = inter / (a + b - inter)
            if d["cls"] == r["cls"]:
                key = (2, inter / min(a, b), iou) if inter / min(a, b) > args.same_overlap \
                    else None
            elif iou > args.cross_iou:
                key = (1, iou, iou)
            elif d["cut"] and inter / a > args.same_overlap:
                key = (0, round(inter / a, 2), iou)   # ties -> the closer-fitting instance
            else:
                key = None
            if key and (best is None or key > best[0]):
                best = (key, gi)
        if best is None:
            groups.append({"rep": i, "members": [i]})
        else:
            groups[best[1]]["members"].append(i)
    return groups


def summarise(group, dets, names, stage2, min_prob):
    mem = [dets[i] for i in group["members"]]
    votes = {}
    for d in mem:
        votes[d["cls"]] = votes.get(d["cls"], 0.0) + d["conf"]
    cls = max(votes, key=lambda c: (votes[c], c == dets[group["rep"]]["cls"]))
    same = [d for d in mem if d["cls"] == cls]
    complete = [d for d in same if not d["cut"]]
    rep = max(complete or same, key=lambda d: d["conf"])
    inst = {"cls": cls, "name": names[cls], "conf": max(d["conf"] for d in same),
            "votes": {names[c]: round(v, 4) for c, v in
                      sorted(votes.items(), key=lambda kv: -kv[1])},
            "agreement": len(same) / len(mem), "n_views": len(mem),
            "n_complete": sum(not d["cut"] for d in mem),
            "faces": sorted({d["face"] for d in mem}),
            "members": [d["id"] for d in mem], "rep": rep,
            "sub": None, "sub_p": None, "sub_probs": None, "sub_status": "",
            "n_sub_views": 0}
    if cls in stage2:
        use = [d for d in complete if d["probs"] is not None] or \
              [d for d in same if d["probs"] is not None]
        if use:
            p = np.mean([d["probs"] for d in use], axis=0)
            k = int(p.argmax())
            classes = stage2[cls].classes
            inst.update(sub=classes[k], sub_p=float(p[k]), n_sub_views=len(use),
                        sub_probs={c: round(float(x), 4) for c, x in zip(classes, p)},
                        sub_status="ok" if p[k] >= min_prob else "uncertain")
    return inst


# =============================================================================
# Drawing
# =============================================================================
def put_label(img, txt, x, y, col, scale, taken=None):
    """Draw a filled label; with `taken` (list of placed boxes) it moves up until it
    no longer covers another label."""
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


def sub_txt(x, min_prob):
    if not x.get("sub"):
        return ""
    return f' | {"? " if x["sub_p"] < min_prob else ""}{x["sub"]} {x["sub_p"]:.2f}'


def draw_face(face_bgr, fdets, palette, min_prob):
    img = face_bgr.copy()
    over = img.copy()
    for d in fdets:
        over[d["mask"]] = palette(d["cls"], True)
    img = cv2.addWeighted(over, FILL_ALPHA, img, 1 - FILL_ALPHA, 0)
    taken = []
    for d in fdets:
        col = palette(d["cls"], True)
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        txt = (f'D{d["id"]} {d["name"]} {d["conf"]:.2f}{sub_txt(d, min_prob)} '
               f'-> I{d["inst"]}{" edge" if d["cut"] else ""}')
        put_label(img, txt, x1, y1, col, 0.55, taken)
    return img


def draw_overview(equi, insts, palette, min_prob):
    h, w = equi.shape[:2]
    over = equi.copy().reshape(-1, 3)
    for k in insts:
        over[k["rep"]["pix"]] = palette(k["cls"], True)
    full = cv2.addWeighted(over.reshape(h, w, 3), FILL_ALPHA, equi, 1 - FILL_ALPHA, 0)
    s = OVERVIEW_WIDTH / w
    img = cv2.resize(full, (OVERVIEW_WIDTH, round(h * s)), interpolation=cv2.INTER_AREA)
    taken = []
    for k in insts:
        col = palette(k["cls"], True)
        st = k["rep"]["st"]
        bx0, bx1 = int(st["u0"] * s), int(st["u1"] * s)
        by0, by1 = int(st["v0"] * s), int(st["v1"] * s)
        for shift in (-OVERVIEW_WIDTH, 0, OVERVIEW_WIDTH):
            cv2.rectangle(img, (bx0 + shift, by0), (bx1 + shift, by1), col, 2)
        txt = (f'I{k["id"]} {k["name"]} {k["conf"]:.2f}{sub_txt(k, min_prob)} '
               f'({k["n_views"]} views)')
        put_label(img, txt, (bx0 % OVERVIEW_WIDTH), by0 - 2, col, 0.55, taken)
    return img


# =============================================================================
# Output helpers
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


INST_FIELDS = ["pano", "instance", "class_id", "class_name", "conf", "votes",
               "class_agreement", "n_views", "n_complete_views", "faces", "detections",
               "rep_detection", "rep_face", "centroid_u", "centroid_v", "area_px",
               "subtype", "p_subtype", "subtype_status", "n_subtype_views", "crop"]
DET_FIELDS = ["pano", "detection", "face", "class_id", "class_name", "conf",
              "threshold", "x1", "y1", "x2", "y2", "cut_off", "centroid_u",
              "centroid_v", "area_px", "subtype", "p_subtype", "instance"]


def safe(name):
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in str(name)).strip()


def imwrite(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        path.write_bytes(buf.tobytes())


def file_info(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": st.st_size,
            "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(
                timespec="seconds"), "sha256": h.hexdigest()}


def load_thresholds(path, names):
    thr = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [n for n in names.values() if n not in thr]
    if missing:
        sys.exit(f"{path} has no threshold for {missing}; its classes are {list(thr)}")
    extra = [k for k in thr if k not in names.values()]
    if extra:
        print(f"  note: thresholds for classes not in the model are ignored: {extra}")
    return {c: float(thr[n]) for c, n in names.items()}


def class_id(names, wanted):
    for k, v in names.items():
        if v == wanted:
            return k
    sys.exit(f"Class {wanted!r} is not in the YOLO model: {names}")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.plotting import colors as palette

    panos = sorted(p for p in args.pano_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not panos:
        sys.exit(f"No jpg/png images in {args.pano_dir}")
    started = datetime.datetime.now()
    out = args.out or args.pano_dir.parent / (
        f"{args.pano_dir.name}_inference_{started:%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    log_fh = open(out / "log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Output: {out}\nDevice: {device}\n{len(panos)} pano(s) in {args.pano_dir}")

    yolo = YOLO(str(args.yolo))
    names = {int(k): v for k, v in yolo.names.items()}
    print(f"YOLO: {args.yolo}\n  classes {names}")
    thr = load_thresholds(args.thresholds, names)
    floor = min(thr.values())
    print("Per-class thresholds (" + str(args.thresholds) + "):\n  "
          + ", ".join(f"{names[c]} {t:.2f}" for c, t in thr.items()))
    ext_id, exit_id = class_id(names, EXT_NAME), class_id(names, EXIT_NAME)
    stage2 = {ext_id: ExtinguisherProbe(args.probe),
              exit_id: load_exit_model(args.exit_model, args.exit_json, device)}

    import sklearn
    import torchvision
    import ultralytics
    print("Hashing model files...")
    settings = {
        "started": started.isoformat(timespec="seconds"), "command": sys.argv,
        "pano_dir": str(args.pano_dir.resolve()), "panos": [p.name for p in panos],
        "models": {"yolo": file_info(args.yolo), "probe": file_info(args.probe),
                   "exit_model": file_info(args.exit_model),
                   "exit_weights": [file_info(p) for p in stage2[exit_id].weight_files]},
        "thresholds_file": file_info(args.thresholds),
        "thresholds": {names[c]: t for c, t in thr.items()},
        "yolo_classes": names, "imgsz": args.imgsz, "yolo_conf_floor": floor,
        "retina_masks": True, "nb_splits": NB_SPLITS, "face_res": FACE_RES,
        "probe_meta": {k: v for k, v in stage2[ext_id].bundle.items()
                       if k != "classifier"},
        "exit_meta": stage2[exit_id].describe(),
        "stage2_crop": {"pad": args.pad, "pad_px": args.pad_px},
        "min_prob": args.min_prob, "min_size": args.min_size,
        "merge": {"same_overlap": args.same_overlap, "cross_iou": args.cross_iou,
                  "border_px": args.border_px,
                  "rule": "complete views first, then confidence; compare with each "
                          "instance's representative; class by summed confidence"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__,
                     "ultralytics": ultralytics.__version__,
                     "scikit_learn": sklearn.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__,
                     "device": torch.cuda.get_device_name(0)
                     if str(device) != "cpu" and torch.cuda.is_available() else "cpu"},
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False,
                                                  default=str), encoding="utf-8")

    inst_fh = open(out / "instances.csv", "w", newline="", encoding="utf-8")
    det_fh = open(out / "detections.csv", "w", newline="", encoding="utf-8")
    inst_w = csv.DictWriter(inst_fh, fieldnames=INST_FIELDS)
    det_w = csv.DictWriter(det_fh, fieldnames=DET_FIELDS)
    inst_w.writeheader()
    det_w.writeheader()
    tot_det, tot_inst = {}, {}
    b = args.border_px

    for n_pano, pano_path in enumerate(panos, 1):
        equi = cv2.imdecode(np.fromfile(str(pano_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if equi is None:
            print(f"\n[{n_pano}/{len(panos)}] could not read {pano_path.name}, skipping")
            continue
        H, W = equi.shape[:2]
        stem = pano_path.stem
        faces = make_faces(equi)

        results = []
        for i in range(0, len(faces), args.batch):
            results += yolo.predict([f["bgr"] for f in faces[i:i + args.batch]],
                                    imgsz=args.imgsz, conf=floor, retina_masks=True,
                                    device=device, verbose=False)

        # ---- face detections above their class threshold ----
        dets, n_raw, n_small = [], 0, 0
        for fi, r in enumerate(results):
            if r.boxes is None or not len(r.boxes) or r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy() > 0.5
            for box, c, conf, mk in zip(r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy().astype(int),
                                        r.boxes.conf.cpu().numpy(), masks):
                n_raw += 1
                c, conf = int(c), float(conf)
                if conf < thr[c]:
                    continue
                if args.min_size and min(box[2] - box[0], box[3] - box[1]) < args.min_size:
                    n_small += 1
                    continue
                if mk.shape != (FACE_RES, FACE_RES):
                    mk = cv2.resize(mk.astype(np.uint8), (FACE_RES, FACE_RES),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                f = faces[fi]
                pix = face_mask_to_pano(mk, f["type"], f["idx"], W, H)
                if not pix.size:
                    continue
                cut = bool(box[0] <= b or box[1] <= b or box[2] >= FACE_RES - 1 - b
                           or box[3] >= FACE_RES - 1 - b)
                dets.append(dict(id=len(dets), fi=fi, face=f["name"], cls=c,
                                 name=names[c], conf=conf, box=[float(x) for x in box],
                                 mask=mk, pix=pix, st=pix_stats(pix, W), cut=cut,
                                 probs=None, sub=None, sub_p=None, crop=None, inst=None))

        # ---- stage 2 on every extinguisher / exit-sign view ----
        rgb = {}
        for cls_id, model in stage2.items():
            todo = []
            for d in dets:
                if d["cls"] != cls_id:
                    continue
                if d["fi"] not in rgb:
                    rgb[d["fi"]] = Image.fromarray(faces[d["fi"]]["bgr"][:, :, ::-1].copy())
                crop, _ = crop_box(rgb[d["fi"]], d["box"], args.pad, args.pad_px)
                if crop.size[0] >= 2 and crop.size[1] >= 2:
                    d["crop"] = crop
                    todo.append(d)
            if todo:
                for d, p in zip(todo, model.predict_proba([d["crop"] for d in todo])):
                    k = int(np.argmax(p))
                    d.update(probs=p, sub=model.classes[k], sub_p=float(p[k]))

        # ---- merge copies across faces into instances ----
        groups = merge(dets, args, W)
        insts = []
        for gi, g in enumerate(groups):
            k = summarise(g, dets, names, stage2, args.min_prob)
            k["id"] = gi
            for i in g["members"]:
                dets[i]["inst"] = gi
            insts.append(k)
        insts.sort(key=lambda k: (k["rep"]["st"]["cu"]))
        remap = {k["id"]: n for n, k in enumerate(insts)}
        for n, k in enumerate(insts):
            k["id"] = n
        for d in dets:
            d["inst"] = remap[d["inst"]]

        # ---- save ----
        face_files = {}
        for fi in sorted({d["fi"] for d in dets}):
            fp = out / "faces" / f"{stem}__{faces[fi]['name']}.jpg"
            imwrite(fp, draw_face(faces[fi]["bgr"], [d for d in dets if d["fi"] == fi],
                                  palette, args.min_prob))
        imwrite(out / "overview" / f"{stem}.jpg",
                draw_overview(equi, insts, palette, args.min_prob))

        npz = {"width": np.int32(W), "height": np.int32(H)}
        json_insts, json_dets = [], []
        for k in insts:
            rep = k["rep"]
            crop_rel = ""
            img = rep["crop"]
            if img is None:
                if rep["fi"] not in rgb:
                    rgb[rep["fi"]] = Image.fromarray(
                        faces[rep["fi"]]["bgr"][:, :, ::-1].copy())
                img, _ = crop_box(rgb[rep["fi"]], rep["box"], 0.0, 0.0)
            if img.size[0] >= 2 and img.size[1] >= 2:
                folder = out / "crops" / safe(k["name"])
                if k["sub"]:
                    folder = folder / (safe(k["sub"]) if k["sub_status"] == "ok"
                                       else "uncertain")
                folder.mkdir(parents=True, exist_ok=True)
                cp = folder / f"{stem}__I{k['id']}.jpg"
                img.save(cp, quality=95)
                crop_rel = str(cp.relative_to(out))
            row = {"pano": pano_path.name, "instance": k["id"], "class_id": k["cls"],
                   "class_name": k["name"], "conf": round(k["conf"], 4),
                   "votes": "; ".join(f"{n} {v:.2f}" for n, v in k["votes"].items()),
                   "class_agreement": round(k["agreement"], 3),
                   "n_views": k["n_views"], "n_complete_views": k["n_complete"],
                   "faces": " ".join(k["faces"]),
                   "detections": " ".join(str(dets[i]["id"]) for i in k["members"]),
                   "rep_detection": rep["id"], "rep_face": rep["face"],
                   "centroid_u": round(rep["st"]["cu"], 1),
                   "centroid_v": round(rep["st"]["cv"], 1),
                   "area_px": int(rep["pix"].size), "subtype": k["sub"] or "",
                   "p_subtype": round(k["sub_p"], 4) if k["sub"] else "",
                   "subtype_status": k["sub_status"],
                   "n_subtype_views": k["n_sub_views"], "crop": crop_rel}
            inst_w.writerow(row)
            json_insts.append({**row, "faces": k["faces"],
                               "detections": [dets[i]["id"] for i in k["members"]],
                               "votes": k["votes"], "subtype_probs": k["sub_probs"],
                               "outline_pano": outline(rep["pix"], W)})
            npz[f"inst_{k['id']}"] = rep["pix"]
            tot_inst[k["name"]] = tot_inst.get(k["name"], 0) + 1
        for d in dets:
            row = {"pano": pano_path.name, "detection": d["id"], "face": d["face"],
                   "class_id": d["cls"], "class_name": d["name"],
                   "conf": round(d["conf"], 4), "threshold": thr[d["cls"]],
                   "x1": round(d["box"][0], 1), "y1": round(d["box"][1], 1),
                   "x2": round(d["box"][2], 1), "y2": round(d["box"][3], 1),
                   "cut_off": int(d["cut"]), "centroid_u": round(d["st"]["cu"], 1),
                   "centroid_v": round(d["st"]["cv"], 1), "area_px": int(d["pix"].size),
                   "subtype": d["sub"] or "",
                   "p_subtype": round(d["sub_p"], 4) if d["sub"] else "",
                   "instance": d["inst"]}
            det_w.writerow(row)
            json_dets.append({**row, "subtype_probs": None if d["probs"] is None else
                              {c: round(float(x), 4) for c, x in
                               zip(stage2[d["cls"]].classes, d["probs"])}})
            npz[f"det_{d['id']}"] = d["pix"]
            tot_det[d["name"]] = tot_det.get(d["name"], 0) + 1
        inst_fh.flush()
        det_fh.flush()
        (out / "masks").mkdir(exist_ok=True)
        np.savez_compressed(out / "masks" / f"{stem}.npz", **npz)
        (out / "json").mkdir(exist_ok=True)
        (out / "json" / f"{stem}.json").write_text(json.dumps({
            "pano": pano_path.name, "width": W, "height": H,
            "n_raw_detections": n_raw, "n_after_thresholds": len(dets),
            "n_dropped_small": n_small, "instances": json_insts,
            "detections": json_dets}, indent=1), encoding="utf-8")

        print(f"\n[{n_pano}/{len(panos)}] {pano_path.name}: {n_raw} raw face detections, "
              f"{len(dets)} above thresholds, {len(insts)} instances")
        for k in insts:
            print(f'  I{k["id"]:<3} {k["name"]:<23} {k["conf"]:.2f}{sub_txt(k, args.min_prob)}'
                  f'  views {k["n_views"]} ({k["n_complete"]} complete), agreement '
                  f'{k["agreement"]:.2f}, faces {" ".join(k["faces"])}')

    inst_fh.close()
    det_fh.close()
    print("\nAll panos, per class: face detections -> instances")
    for n in sorted(set(tot_det) | set(tot_inst)):
        print(f"  {n:<23}{tot_det.get(n, 0):>6} -> {tot_inst.get(n, 0)}")
    print(f"Results saved in {out}")
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    log_fh.close()


if __name__ == "__main__":
    main()
