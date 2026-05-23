
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import os

from PIL import Image
from ultralytics import YOLO

_YOLO_PATH = os.environ.get("YOLO_MODEL", "yolov8s.pt")
_YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.4"))
_CLIP_MIN_CONF = float(os.environ.get("CLIP_MIN_CONF", "0.4"))
_CLIP_MAX_REFINES = int(os.environ.get("CLIP_MAX_REFINES", "3"))
_CLIP_REFINE_IF_YOLO_BELOW = float(os.environ.get("CLIP_REFINE_IF_YOLO_BELOW", "0.75"))

_yolo_model: YOLO | None = None
_clip_model = None
_clip_processor = None

CLIP_LABELS = [
    # Animals (YOLO-COCO lacks many species like "tiger")
    "tiger",
    "lion",
    "leopard",
    "cheetah",
    "jaguar",
    "panther",
    "cat",
    "dog",
    "horse",
    "zebra",
    "cow",
    "sheep",
    "goat",
    "deer",
    "elephant",
    "bear",
    "monkey",
    "giraffe",
    "wolf",
    "fox",
    "rabbit",
    "bird",

    # Plants / flowers
    "flower",
    "rose",
    "sunflower",
    "tulip",
    "daisy",
    "bouquet",
    "plant",
    "tree",
    "leaf",

    "coffee cup",
    "mug",
    "water bottle",
    "plastic bottle",
    "glass bottle",
    "smartphone",
    "mobile phone",
    "laptop",
    "tablet",
    "book",
    "notebook",
    "document",
    "paper",
    "receipt",
    "shoe",
    "sneaker",
    "watch",
    "glasses",
    "bag",
    "backpack",
    "t-shirt",
    "keyboard",
    "mouse",
    "headphones",
    "barcode",
    "QR code",
    "math equation",
    "printed text",
    "handwritten text",
    "pizza",
    "burger",
    "banana",
    "apple",
]

SCENE_FALLBACK_LABELS = [
    # Only used when YOLO finds no boxes (fast "what is this?" fallback)
    "flower",
    "rose",
    "sunflower",
    "plant",
    "tree",
    "leaf",
    "person",
    "dog",
    "cat",
    "car",
    "bicycle",
    "food",
    "coffee cup",
    "bottle",
    "book",
    "document",
]


def _get_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(_YOLO_PATH)
    return _yolo_model


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def _clip_refine(crop: Image.Image, labels: list[str]) -> tuple[str, float]:
    try:
        import torch

        model, processor = _get_clip()
        inputs = processor(text=labels, images=crop, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0]
        idx = int(probs.argmax().item())
        return labels[idx], float(probs[idx])
    except Exception:
        return "", 0.0


def _clip_classify_full_image(img: Image.Image) -> tuple[str, float]:
    """
    Fast fallback used only when YOLO finds no boxes.
    """
    try:
        # Resize to keep CLIP fast on high-res photos
        w, h = img.size
        max_side = 384
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        return _clip_refine(img, SCENE_FALLBACK_LABELS)
    except Exception:
        return "", 0.0


def _clip_label_subset_for_yolo(yolo_label: str) -> list[str]:
    yl = (yolo_label or "").lower()
    # Keep CLIP label list small per crop for speed/accuracy.
    if any(k in yl for k in ("cat", "dog", "horse", "zebra", "cow", "sheep", "bear", "bird", "elephant", "giraffe", "deer")):
        return [
            "tiger",
            "lion",
            "leopard",
            "cheetah",
            "jaguar",
            "panther",
            "cat",
            "dog",
            "horse",
            "zebra",
            "cow",
            "sheep",
            "goat",
            "deer",
            "elephant",
            "bear",
            "monkey",
            "giraffe",
            "wolf",
            "fox",
            "rabbit",
            "bird",
        ]
    if any(k in yl for k in ("cell phone", "mobile", "laptop", "keyboard", "mouse", "tablet", "book", "backpack", "handbag", "shoe", "sneaker", "watch", "glasses", "bottle")):
        return [
            "smartphone",
            "mobile phone",
            "laptop",
            "tablet",
            "keyboard",
            "mouse",
            "headphones",
            "watch",
            "glasses",
            "bag",
            "backpack",
            "shoe",
            "sneaker",
            "water bottle",
            "plastic bottle",
            "glass bottle",
            "book",
            "notebook",
            "document",
            "receipt",
        ]
    if any(k in yl for k in ("apple", "banana", "pizza", "burger", "bowl", "sandwich", "cake", "orange")):
        return ["apple", "banana", "pizza", "burger", "coffee cup", "mug", "water bottle"]
    if any(k in yl for k in ("book", "paper", "document")):
        return ["document", "paper", "receipt", "printed text", "handwritten text", "barcode", "QR code", "math equation"]
    return CLIP_LABELS


def detect_objects(
    image_path: str,
    conf_threshold: float = _YOLO_CONF,
    *,
    enable_clip: bool = True,
    max_clip_refines: int = _CLIP_MAX_REFINES,
) -> list[dict]:
    """
    Returns:
    [
      { "label": "...", "confidence": 0.92, "box": [x1,y1,x2,y2] }
    ]
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return []

    model = _get_yolo()

    try:
        results = model(img, conf=conf_threshold, iou=0.45, verbose=False)
    except Exception:
        return []

    objects: list[dict] = []
    clip_used = 0
    for r in results:
        names = getattr(r, "names", {}) or {}
        for b in getattr(r, "boxes", []) or []:
            try:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls_id = int(b.cls[0])
                yolo_label = names.get(cls_id, f"class_{cls_id}")
                conf = float(b.conf[0])

                x1i, y1i, x2i, y2i = [int(round(v)) for v in (x1, y1, x2, y2)]
                x1i = max(0, min(x1i, img.width - 1))
                y1i = max(0, min(y1i, img.height - 1))
                x2i = max(0, min(x2i, img.width))
                y2i = max(0, min(y2i, img.height))
                if x2i <= x1i or y2i <= y1i:
                    continue

                used_clip = False
                clip_label, clip_conf = "", 0.0
                # Only refine a few low-confidence boxes to keep speed.
                if enable_clip and clip_used < max_clip_refines and conf < _CLIP_REFINE_IF_YOLO_BELOW:
                    crop = img.crop((x1i, y1i, x2i, y2i))
                    labels = _clip_label_subset_for_yolo(yolo_label)
                    clip_label, clip_conf = _clip_refine(crop, labels)
                    used_clip = bool(clip_label and clip_conf >= _CLIP_MIN_CONF)
                    if used_clip:
                        clip_used += 1
                label = clip_label if used_clip else yolo_label
                conf_out = clip_conf if used_clip else conf

                objects.append(
                    {
                        "label": str(label),
                        "confidence": round(float(conf_out), 4),
                        "box": [x1i, y1i, x2i, y2i],
                    }
                )
            except Exception:
                continue

    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(a_area + b_area - inter)

    # Dedupe near-identical boxes per label (helps cases like banana appearing 3×).
    objects.sort(key=lambda o: o["confidence"], reverse=True)
    deduped: list[dict] = []
    for o in objects:
        box = o.get("box")
        if not box:
            deduped.append(o)
            continue
        keep = True
        for k in deduped:
            if k.get("label") != o.get("label"):
                continue
            kbox = k.get("box")
            if not kbox:
                continue
            if _iou(box, kbox) >= 0.85:
                keep = False
                break
        if keep:
            deduped.append(o)
    objects = deduped

    # If YOLO found nothing (common for flowers / niche categories), do a quick CLIP scene fallback.
    if not objects and enable_clip:
        label, c = _clip_classify_full_image(img)
        if label and c >= _CLIP_MIN_CONF:
            return [{"label": str(label), "confidence": round(float(c), 4), "box": None}]
    return objects


# Backward-compat (old UI might call run_detection)
def run_detection(image_path: str, conf_threshold: float = _YOLO_CONF) -> list[dict]:
    return [
        {
            "label": o["label"],
            "confidence": o["confidence"],
            "bbox": [o["box"][0], o["box"][1], o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]],
        }
        for o in detect_objects(image_path, conf_threshold=conf_threshold)
    ]