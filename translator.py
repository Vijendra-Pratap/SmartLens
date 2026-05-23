
from __future__ import annotations

import functools
import re
from typing import Dict, Tuple

# ── LANGUAGE CODE MAP ─────────────────────────────────────────────────────────
SUPPORTED_LANGS = {
    'auto': 'auto',
    'en':   'english',
    'hi':   'hindi',
    'zh':   'chinese (simplified)',
    'es':   'spanish',
    'fr':   'french',
    'ar':   'arabic',
    'de':   'german',
    'ja':   'japanese',
    'ko':   'korean',
    'pt':   'portuguese',
    'ru':   'russian',
    'it':   'italian',
    'tr':   'turkish',
    'nl':   'dutch',
    'pl':   'polish',
    'bn':   'bengali',
    'ur':   'urdu',
    'pa':   'punjabi',
    'ta':   'tamil',
    'te':   'telugu',
}


def detect_language(text: str) -> str:
    """
    Returns ISO-ish code like 'en','hi','ta',... or 'auto' when unknown.
    """
    s = (text or "").strip()
    if not s:
        return "auto"
    try:
        from langdetect import detect  # type: ignore
        code = detect(s)
        return code if code in SUPPORTED_LANGS else "auto"
    except Exception:
        return "auto"


_OPUS_TO_EN: Dict[str, str] = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "ta": "Helsinki-NLP/opus-mt-ta-en",
    "te": "Helsinki-NLP/opus-mt-te-en",
    "bn": "Helsinki-NLP/opus-mt-bn-en",
    "mr": "Helsinki-NLP/opus-mt-mr-en",
    "gu": "Helsinki-NLP/opus-mt-gu-en",
    "kn": "Helsinki-NLP/opus-mt-kn-en",
    "ml": "Helsinki-NLP/opus-mt-ml-en",
    "pa": "Helsinki-NLP/opus-mt-pa-en",
    "ur": "Helsinki-NLP/opus-mt-ur-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "tr": "Helsinki-NLP/opus-mt-tr-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "pl": "Helsinki-NLP/opus-mt-pl-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
}


@functools.lru_cache(maxsize=8)
def _get_translator(model_name: str):
    from transformers import pipeline

    return pipeline("translation", model=model_name)


def run_translation(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if target_lang != "en":
        # This project’s core requirement is OCR->English; keep API stable.
        target_lang = "en"

    src = source_lang if source_lang in SUPPORTED_LANGS else "auto"
    if src == "auto":
        src = detect_language(s)

    if src == "en":
        return s

    model = _OPUS_TO_EN.get(src) or "Helsinki-NLP/opus-mt-mul-en"
    try:
        tr = _get_translator(model)
        # chunk to avoid very long sequences
        chunks = _chunk_text(s, 900)
        out = []
        for c in chunks:
            r = tr(c, max_length=256)
            out.append((r[0] or {}).get("translation_text", "") if r else "")
        return _post_clean(" ".join(out).strip())
    except Exception:
        return ""


def translate_to_english(text: str) -> str:
    return run_translation(text, target_lang="en", source_lang="auto")


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(\n+|[.!?]\s+)", text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) <= max_chars:
            buf += p
        else:
            if buf.strip():
                out.append(buf.strip())
            buf = p
    if buf.strip():
        out.append(buf.strip())
    return out or [text[:max_chars]]


def _post_clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── CLI QUICK TEST ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else 'Hello, world!'
    tgt  = sys.argv[2] if len(sys.argv) > 2 else 'hi'
    print(run_translation(text, tgt))
