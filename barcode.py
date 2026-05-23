
import cv2
import numpy as np


# ── FAST BASE VARIANTS ────────────────────────────────────────────────────────
def _prepare_variants(image_path: str) -> list[np.ndarray]:
    img  = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape

    variants: list[np.ndarray] = [grey]

    # Upscale small images (capped at 800px for speed)
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        up    = cv2.resize(grey, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_LANCZOS4)
        variants.append(up)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(binary)

    # Inverted
    variants.append(cv2.bitwise_not(binary))

    # CLAHE (handles uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(grey))

    # Morphological closing (fills gaps in barcode bars)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    variants.append(cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel))

    return variants


def _rotated_variants(variants: list[np.ndarray], angle: float) -> list[np.ndarray]:
    out = []
    for img in variants:
        h, w = img.shape[:2]
        M    = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        out.append(cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


# ── DECODERS ──────────────────────────────────────────────────────────────────
def _decode_pyzbar(variants: list[np.ndarray]) -> dict | None:
    try:
        from pyzbar.pyzbar import decode as pz_decode
    except (ImportError, OSError):
        # ImportError → pyzbar not installed
        # OSError     → libzbar-64.dll missing on Windows.
        # Fix: download libzbar-64.dll from https://github.com/NaturalHistoryMuseum/pyzbar#windows
        # and place it next to python.exe, or add its folder to PATH.
        return None
    for img in variants:
        barcodes = pz_decode(img)
        if barcodes:
            bc   = barcodes[0]
            data = bc.data.decode("utf-8", errors="replace").strip()
            typ  = bc.type
            return {
                "value":    data,
                "type":     _fmt_type(typ),
                "raw_type": typ,
                "count":    len(barcodes),
                "all":      [{"value": b.data.decode("utf-8", errors="replace"),
                              "type":  _fmt_type(b.type)} for b in barcodes],
            }
    return None


def _decode_opencv_qr(variants: list[np.ndarray]) -> dict | None:
    detector = cv2.QRCodeDetector()
    for img in variants:
        data, _, _ = detector.detectAndDecode(img)
        if data:
            return {"value": data.strip(), "type": "QR Code",
                    "raw_type": "QRCODE", "count": 1,
                    "all": [{"value": data.strip(), "type": "QR Code"}]}
    return None


def _decode_wechat_qr(variants: list[np.ndarray]) -> dict | None:
    try:
        detector = cv2.wechat_qrcode_WeChatQRCode()
        for img in variants:
            texts, _ = detector.detectAndDecode(img)
            if texts:
                return {"value": texts[0].strip(), "type": "QR Code",
                        "raw_type": "QRCODE", "count": len(texts),
                        "all": [{"value": t.strip(), "type": "QR Code"}
                                for t in texts]}
    except Exception:
        pass
    return None


import re
import urllib.parse

def classify_barcode_value(value: str, barcode_type: str) -> dict:
    """
    Inspect a decoded barcode value and return link/action metadata.
    Returns a dict with keys: kind, label, url, display
      - kind:    'url' | 'upi' | 'email' | 'phone' | 'wifi' | 'sms' | 'geo' | 'text'
      - label:   human-friendly button label e.g. "Open Link", "Send Email"
      - url:     the href to open (may be a constructed one for non-URLs)
      - display: short display text shown under the value
    """
    v = (value or "").strip()
    vl = v.lower()

    # ── URL ──────────────────────────────────────────────────────────────────
    if re.match(r"https?://", vl) or re.match(r"www\.", vl):
        url = v if v.lower().startswith("http") else "https://" + v
        try:
            domain = urllib.parse.urlparse(url).netloc or url
        except Exception:
            domain = url
        return {"kind": "url", "label": "Open Link", "url": url, "display": domain}

    # ── UPI payment ──────────────────────────────────────────────────────────
    if vl.startswith("upi://") or re.match(r"[\w.\-]+@[\w.\-]+", v):
        upi_url = v if vl.startswith("upi://") else f"upi://pay?pa={urllib.parse.quote(v)}"
        return {"kind": "upi", "label": "Pay via UPI", "url": upi_url, "display": v}

    # ── Email ─────────────────────────────────────────────────────────────────
    if vl.startswith("mailto:") or re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
        email = v if vl.startswith("mailto:") else f"mailto:{v}"
        return {"kind": "email", "label": "Send Email", "url": email, "display": v.replace("mailto:", "")}

    # ── Phone ─────────────────────────────────────────────────────────────────
    if vl.startswith("tel:") or re.match(r"^\+?[\d\s\-()]{7,15}$", v):
        tel = v if vl.startswith("tel:") else f"tel:{re.sub(r'[^+\d]', '', v)}"
        return {"kind": "phone", "label": "Call Number", "url": tel, "display": v.replace("tel:", "")}

    # ── SMS ───────────────────────────────────────────────────────────────────
    if vl.startswith("sms:") or vl.startswith("smsto:"):
        return {"kind": "sms", "label": "Send SMS", "url": v, "display": v}

    # ── WiFi ──────────────────────────────────────────────────────────────────
    if vl.startswith("wifi:"):
        ssid_m = re.search(r"S:([^;]+)", v)
        ssid = ssid_m.group(1) if ssid_m else "WiFi Network"
        return {"kind": "wifi", "label": "Connect to WiFi", "url": "", "display": f"Network: {ssid}"}

    # ── Geo / Maps ────────────────────────────────────────────────────────────
    if vl.startswith("geo:"):
        coords = v[4:].split("?")[0]
        maps_url = f"https://maps.google.com/?q={urllib.parse.quote(coords)}"
        return {"kind": "geo", "label": "Open in Maps", "url": maps_url, "display": coords}

    # ── App / deep-link schemes ───────────────────────────────────────────────
    if re.match(r"^[a-z][a-z0-9+\-.]+://", vl):
        return {"kind": "url", "label": "Open Link", "url": v, "display": v}

    # ── Plain product barcode (EAN/UPC/etc.) ─────────────────────────────────
    if barcode_type.upper() not in ("QRCODE", "QR CODE") and re.match(r"^\d{6,14}$", v):
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(v)}"
        return {"kind": "product", "label": "Search Product", "url": search_url, "display": f"Barcode: {v}"}

    # ── Generic text — offer Google Search ───────────────────────────────────
    search_url = f"https://www.google.com/search?q={urllib.parse.quote(v)}"
    return {"kind": "text", "label": "Search on Google", "url": search_url, "display": v[:80]}


_TYPE_MAP = {
    "QRCODE":"QR Code","EAN13":"EAN-13","EAN8":"EAN-8","UPCA":"UPC-A",
    "UPCE":"UPC-E","CODE128":"Code 128","CODE39":"Code 39","CODE93":"Code 93",
    "ITF":"ITF","CODABAR":"Codabar","PDF417":"PDF-417",
    "DATAMATRIX":"Data Matrix","AZTEC":"Aztec","I25":"Interleaved 2of5",
}
def _fmt_type(raw: str) -> str:
    return _TYPE_MAP.get(raw.upper(), raw)


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def run_barcode(image_path: str) -> dict:
    base = _prepare_variants(image_path)

    # Fast path: try base variants first (no rotation overhead)
    result = (_decode_pyzbar(base)
              or _decode_opencv_qr(base)
              or _decode_wechat_qr(base))
    if result:
        return _attach_link(result)

    # Slow path: rotation retry only when base fails
    for angle in [15, -15, 30, -30]:
        rot    = _rotated_variants(base, angle)
        result = (_decode_pyzbar(rot)
                  or _decode_opencv_qr(rot)
                  or _decode_wechat_qr(rot))
        if result:
            return _attach_link(result)

    return {"value": None, "type": "No barcode or QR code detected",
            "count": 0, "all": [], "link": None}


def _attach_link(result: dict) -> dict:
    """Enrich a decoded barcode result with link/action metadata."""
    if result.get("value"):
        result["link"] = classify_barcode_value(result["value"], result.get("raw_type", ""))
        for item in result.get("all", []):
            if item.get("value"):
                item["link"] = classify_barcode_value(item["value"], item.get("type", ""))
    else:
        result["link"] = None
    return result


# ── CLI QUICK TEST ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    print(json.dumps(run_barcode(path), indent=2))