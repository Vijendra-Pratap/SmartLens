from flask import Flask, render_template, request, jsonify
import os
import uuid
import traceback

from detect import detect_objects
from ocr import extract_text
from translator import translate_to_english
from analyze import generate_ai_response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED = {".jpg",".jpeg",".png",".webp",".bmp",".gif"}

def _save(f):
    ext = os.path.splitext(f.filename)[-1].lower() or ".jpg"
    if ext not in ALLOWED: raise ValueError(f"Unsupported: {ext}")
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    f.save(path)
    return path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file  = request.files.get("image")
        query = request.form.get("query", "")
        if not file or not file.filename:
            return jsonify({"error": "No image provided"}), 400

        image_path = _save(file)

        qlow = (query or "").lower()
        # For math-heavy queries, skip CLIP refinement (saves time; math comes from OCR anyway).
        enable_clip = not any(k in qlow for k in ("solve", "math", "equation", "value of x", "value of y"))
        objects = detect_objects(image_path, enable_clip=enable_clip)
        ocr_text = extract_text(image_path)

        # Always attempt barcode/QR scan — fast and non-blocking
        barcode_result = None
        try:
            from barcode import run_barcode
            br = run_barcode(image_path)
            if br.get("value"):
                barcode_result = br
        except Exception:
            barcode_result = None

        translated_text = ""
        try:
            translated_text = translate_to_english(ocr_text) if ocr_text else ""
        except Exception:
            translated_text = ""

        ai = generate_ai_response(
            # Keep label signal clean: unique labels, and avoid "person" dominating captions.
            label=", ".join(
                [
                    l
                    for l in dict.fromkeys(
                        [o.get("label", "") for o in (objects or []) if o.get("label")]
                    )
                    if l and (l != "person" or len(objects or []) == 1)
                ][:6]
            )
            if objects
            else "",
            ocr_text=ocr_text,
            query=query or "",
            translated_text=translated_text,
        )

        return jsonify(
            {
                "image_path": image_path.replace("\\", "/"),
                "objects": objects,
                "ocr_text": ocr_text,
                "translated_text": translated_text,
                "query": query or "",
                "ai": ai,
                "barcode": barcode_result,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/detect",    methods=["POST"])
def detect():
    from detect import detect_objects
    try: return jsonify({"detections": detect_objects(_save(request.files["image"]))})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/ocr",       methods=["POST"])
def ocr():
    from ocr import extract_text
    try: return jsonify({"text": extract_text(_save(request.files["image"]))})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/barcode",   methods=["POST"])
def barcode():
    from barcode import run_barcode
    try: return jsonify(run_barcode(_save(request.files["image"])))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate():
    from translator import run_translation
    try:
        return jsonify({"translated": run_translation(
            request.form.get("text",""), request.form.get("to","en"))})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
