
import os
import cv2
import numpy as np
import re

_BACKEND              = os.environ.get("OCR_BACKEND", "easyocr").lower()
_DO_PERSPECTIVE       = os.environ.get("OCR_CORRECT_PERSPECTIVE", "0") == "1"
_OCR_MAX_SIDE         = int(os.environ.get("OCR_MAX_SIDE", "1200"))

# ── EASYOCR SETUP ─────────────────────────────────────────────────────────────
_easy_reader = None

def _get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        langs = os.environ.get("OCR_LANGS", "en").split(",")
        _easy_reader = easyocr.Reader(langs, gpu=False, verbose=False)
    return _easy_reader


# ── OPTIONAL PERSPECTIVE CORRECTION ──────────────────────────────────────────
def _perspective_correct(img: np.ndarray) -> np.ndarray:
    grey  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur  = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:5]:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts  = approx.reshape(4, 2).astype(np.float32)
            rect = _order_points(pts)
            tl, tr, br, bl = rect
            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
            if w < 50 or h < 50:
                continue
            dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
            M   = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (w, h))
    return img

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1);  diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)];   rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect


# ── FAST DESKEW ───────────────────────────────────────────────────────────────
def _deskew(grey: np.ndarray) -> np.ndarray:
    """
    Detect skew angle on a DOWNSCALED copy (much faster Hough),
    then rotate the full-res image.
    """
    h, w = grey.shape[:2]
    scale = min(1.0, 400 / max(h, w))   # work on ≤400px copy for speed
    small = cv2.resize(grey, (int(w*scale), int(h*scale)),
                       interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                             minLineLength=20, maxLineGap=5)
    if lines is None:
        return grey

    angles = [np.degrees(np.arctan2(y2-y1, x2-x1))
              for line in lines
              for x1, y1, x2, y2 in [line[0]]
              if -45 < np.degrees(np.arctan2(y2-y1, x2-x1)) < 45]

    if not angles:
        return grey
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return grey

    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(grey, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def _preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (binary_grey, colour) for tesseract / easyocr respectively."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")

    if _DO_PERSPECTIVE:
        img = _perspective_correct(img)

    h, w = img.shape[:2]
    # Cap size — biggest single speedup for large phone photos
    if max(h, w) > _OCR_MAX_SIDE:
        scale = _OCR_MAX_SIDE / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_AREA)
    # Upscale tiny images
    elif max(h, w) < 640:
        scale = 640 / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_CUBIC)

    grey  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey  = cv2.fastNlMeansDenoising(grey, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grey  = clahe.apply(grey)
    grey  = _deskew(grey)

    # Binarise
    var = float(grey.var())
    if var < 800:
        _, binary = cv2.threshold(grey, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(grey, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 10)
    return binary, img


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def run_ocr(image_path: str) -> str:
    binary, colour = _preprocess(image_path)
    if _BACKEND == "easyocr":
        return _clean_text(_ocr_easyocr(colour, binary))
    return _clean_text(_ocr_tesseract(binary))


def extract_text(image_path: str) -> str:
    """
    Mandatory OCR entrypoint for the SmartLens pipeline.
    Returns a cleaned combined string. On failure returns "".
    """
    try:
        t = run_ocr(image_path)
        if not t or t.strip().lower() == "no text detected.":
            return ""
        return t.strip()
    except Exception:
        return ""


# ── EASYOCR RESULT PARSER ─────────────────────────────────────────────────────
def _parse_easyocr(results: list) -> list[tuple]:
    """
    EasyOCR returns 3-tuples (bbox, text, conf) when detail=1, paragraph=False.
    With paragraph=True it returns 2-tuples (bbox, text) — no confidence score.
    This helper normalises both formats so unpacking never crashes.
    """
    parsed = []
    for item in results:
        if len(item) == 3:
            bbox, text, conf = item
        elif len(item) == 2:
            bbox, text = item
            conf = 1.0   # paragraph mode merges high-conf regions; treat as trusted
        else:
            continue
        parsed.append((bbox, text, conf))
    return parsed


# ── EASYOCR (lazy rotation retry) ─────────────────────────────────────────────
def _ocr_easyocr(colour: np.ndarray, binary: np.ndarray) -> str:
    reader = _get_easy_reader()

    # First pass — normal orientation
    # NOTE: paragraph=True returns (bbox, text) 2-tuples (no confidence).
    #       _parse_easyocr() normalises both formats safely.
    results = reader.readtext(
        colour,
        detail=1,
        paragraph=True,   # merges nearby text regions → faster
        width_ths=0.7,    # reduces over-segmentation
        add_margin=0.05,
    )
    lines = [(bbox, text, conf)
             for (bbox, text, conf) in _parse_easyocr(results)
             if conf > 0.3 and text.strip()]
    lines.sort(key=lambda t: (t[0][0][1], t[0][0][0]))
    text = "\n".join(t[1] for t in lines).strip()

    # Lazy rotation retry — only if we got fewer than 3 words
    if len(text.split()) < 3:
        for k in [2, 1, 3]:   # 180°, 90°, 270°
            rot = np.rot90(colour, k=k)
            res = reader.readtext(rot, detail=1, paragraph=True,
                                  width_ths=0.7, add_margin=0.05)
            candidate = "\n".join(
                text for _, text, conf in _parse_easyocr(res)
                if conf > 0.3 and text.strip()
            ).strip()
            if len(candidate.split()) > len(text.split()):
                text = candidate
            if len(text.split()) >= 3:
                break

    return text or "No text detected."


# ── TESSERACT ─────────────────────────────────────────────────────────────────
def _ocr_tesseract(binary: np.ndarray) -> str:
    try:
        import pytesseract
    except ImportError:
        return "pytesseract not installed. Run: pip install pytesseract"

    config = "--oem 3 --psm 6"
    text   = pytesseract.image_to_string(binary, config=config).strip()

    if len(text.split()) < 3:
        for k in [2, 1, 3]:
            rot       = np.rot90(binary, k=k)
            candidate = pytesseract.image_to_string(rot, config=config).strip()
            if len(candidate.split()) > len(text.split()):
                text = candidate
            if len(text.split()) >= 3:
                break

    return text or "No text detected."


_NOISE_LINES = {
    "|",
    "||",
    "|||",
    "l",
    "I",
    "1",
    "—",
    "-",
    "_",
}


def _clean_text(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s in _NOISE_LINES:
            continue
        if len(s) == 1 and not s.isalnum():
            continue
        # drop lines that are almost all punctuation
        punct = sum(1 for c in s if not c.isalnum() and c != " ")
        if punct / max(1, len(s)) > 0.55:
            continue
        lines.append(s)

    out = "\n".join(lines).strip()
    out = re.sub(r"[^\S\n]{2,}", " ", out)
    return out


# ── CLI QUICK TEST ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    print(run_ocr(path))