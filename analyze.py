
from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, Dict, List


_GEN = None


def _get_generator():
    global _GEN
    if _GEN is None:
        from transformers import pipeline

        _GEN = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
        )
    return _GEN


_PRODUCT_HINTS = {"shoe", "phone", "laptop", "bottle", "smartphone", "sneaker", "watch"}


def _looks_like_math(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if "solve" in t.lower():
        return True
    if re.search(r"[\d]\s*[\+\-\*/=×÷]", t):
        return True
    if re.search(r"(?:\b(x|y|z)\b.*=)|(?:=\s*[\d])", t, re.I):
        return True
    return False


def _solve_math(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {"steps": [], "answer": ""}
    try:
        from sympy import Eq, Symbol
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        from sympy.solvers import solve

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")
        T = standard_transformations + (implicit_multiplication_application,)
        cleaned = (
            s.replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("^", "**")
        )

        # Split into candidate equations/expressions from OCR output.
        parts = [p.strip() for p in re.split(r"[\n;]+", cleaned) if p.strip()]
        # Some OCR returns multiple equations on one line.
        if len(parts) == 1 and re.search(r"\b(?:x|y|z)\b", parts[0]) and parts[0].count("=") >= 2:
            parts = [p.strip() for p in re.split(r"\s+(?=(?:[A-Za-z].*?=))", parts[0]) if p.strip()]

        local = {"x": x, "y": y, "z": z}
        eqs = []
        exprs: list[str] = []
        for p in parts:
            p = p.strip().rstrip(".")
            # Treat "=?"/"= ?" as "evaluate left hand side".
            p = re.sub(r"=\s*\?$", "", p)
            if "=" in p:
                lhs_s, rhs_s = [t.strip() for t in p.split("=", 1)]
                # If rhs is blank or a lone '?', also evaluate lhs.
                if not rhs_s or rhs_s == "?":
                    exprs.append(lhs_s)
                else:
                    lhs = parse_expr(lhs_s, transformations=T, local_dict=local)
                    rhs = parse_expr(rhs_s, transformations=T, local_dict=local)
                    eqs.append(Eq(lhs, rhs))
            else:
                # Not an equation; keep as expression candidate.
                exprs.append(p)

        if eqs:
            # System solve when multiple equations are present.
            vars_involved = []
            if any(e.has(x) for e in eqs):
                vars_involved.append(x)
            if any(e.has(y) for e in eqs):
                vars_involved.append(y)
            if any(e.has(z) for e in eqs):
                vars_involved.append(z)

            sols = solve(eqs, vars_involved or [x], dict=True)
            if sols:
                # Prefer a real-valued solution when available.
                sol_pick = None
                for sol in sols:
                    if not isinstance(sol, dict) or not sol:
                        continue
                    if all(getattr(v, "is_real", None) is True for v in sol.values()):
                        sol_pick = sol
                        break
                sol0 = sol_pick or (sols[0] if isinstance(sols[0], dict) else None)

                if isinstance(sol0, dict) and sol0:
                    ans = ", ".join(f"{str(k)} = {sol0[k]}" for k in sol0)
                    if sol_pick is None:
                        ans = f"No real solution. Complex solution: {ans}"
                else:
                    ans = "No real solution."
                return {
                    "steps": [
                        "Recognize equation(s)",
                        "Convert to symbolic form",
                        "Solve the system",
                    ],
                    "answer": ans,
                }

        # Otherwise evaluate as expression (after stripping trailing '?')
        expr_s = (exprs[0] if exprs else (parts[0] if parts else cleaned)).strip()
        expr_s = re.sub(r"\?$", "", expr_s).strip()
        expr = parse_expr(expr_s, transformations=T, local_dict=local)
        val = expr.evalf()
        return {"steps": ["Parse expression", "Evaluate"], "answer": str(val)}
    except Exception:
        return {"steps": [], "answer": ""}


def _safe_json_fallback(label: str, ocr_text: str, query: str, translated_text: str) -> Dict[str, Any]:
    t = (translated_text or ocr_text or "").strip()
    return {
        "type": "general object",
        "summary": (query.strip() or "Analysis complete.").strip(),
        "details": f"Detected objects: {label or ''}\nDetected text: {t or ''}".strip(),
        "actions": [],
        "extra": {"steps": [], "products": [], "translated_text": translated_text or ""},
    }


def generate_ai_response(label: str, ocr_text: str, query: str, translated_text: str = "") -> Dict[str, Any]:
    """
    Returns ONLY JSON:
    {
      "type": "...",
      "summary": "...",
      "details": "...",
      "actions": [...],
      "extra": { "steps": [...], "products": [...], "translated_text": "..." }
    }
    """
    label_s = (label or "").strip()
    ocr_s = (ocr_text or "").strip()
    query_s = (query or "").strip()
    trans_s = (translated_text or "").strip()

    # Rule-based intent overrides
    qlow = query_s.lower()
    product_trigger = any(k in qlow for k in ("price", "buy", "cost")) or any(h in label_s.lower() for h in _PRODUCT_HINTS)
    doc_trigger = len(ocr_s) > 350 or len(trans_s) > 350
    math_trigger = _looks_like_math(ocr_s) or _looks_like_math(query_s)

    if math_trigger:
        src = ocr_s if _looks_like_math(ocr_s) else query_s
        solved = _solve_math(src)
        if solved.get("answer"):
            return {
                "type": "math",
                "summary": solved["answer"],
                "details": src,
                "actions": ["Check the steps", "Try another problem"],
                "extra": {
                    "steps": solved.get("steps", []),
                    "products": [],
                    "translated_text": trans_s or "",
                },
            }

    if product_trigger:
        name = label_s or "product"
        q = urllib.parse.quote(name)
        products = [
            {
                "name": name,
                "price_range": "₹500–₹5,000",
                "links": [
                    f"https://www.amazon.in/s?k={q}",
                    f"https://www.flipkart.com/search?q={q}",
                    f"https://www.google.com/search?tbm=shop&q={q}",
                ],
            }
        ]
        return {
            "type": "product",
            "summary": f"Likely product: {name}",
            "details": "Estimated price range and quick links are provided.",
            "actions": ["Compare prices", "Search by brand/model", "Check reviews"],
            "extra": {"steps": [], "products": products, "translated_text": trans_s or ""},
        }

    if doc_trigger:
        text_for_doc = trans_s or ocr_s
        prompt = f"""You are an advanced AI visual assistant like Google Lens.

Detected objects: {label_s}
Detected text: {text_for_doc}
User query: {query_s}

Tasks:
1. Detect intent: document
2. Return ONLY JSON:
{{
  "type": "document",
  "summary": "...",
  "details": "...",
  "actions": [...],
  "extra": {{
    "steps": [...],
    "products": [...],
    "translated_text": "{trans_s}"
  }}
}}
"""
        return _flan_json(prompt, fallback=_safe_json_fallback(label_s, ocr_s, query_s, trans_s), force_type="document")

    prompt = f"""You are an advanced AI visual assistant like Google Lens.

Detected objects: {label_s}
Detected text: {trans_s or ocr_s}
User query: {query_s}

Tasks:

1. Detect intent:
   - product
   - math
   - document
   - general object

2. Return ONLY JSON:
{{
  "type": "...",
  "summary": "...",
  "details": "...",
  "actions": [...],
  "extra": {{
    "steps": [...],
    "products": [...],
    "translated_text": "{trans_s}"
  }}
}}
"""
    return _flan_json(prompt, fallback=_safe_json_fallback(label_s, ocr_s, query_s, trans_s))


def _flan_json(prompt: str, fallback: Dict[str, Any], force_type: str | None = None) -> Dict[str, Any]:
    try:
        gen = _get_generator()
        out = gen(prompt, max_length=512, do_sample=False)[0]["generated_text"]
        parsed = _extract_json(out)
        if not isinstance(parsed, dict):
            return fallback

        # enforce schema keys
        parsed.setdefault("type", force_type or "general object")
        parsed.setdefault("summary", "")
        parsed.setdefault("details", "")
        parsed.setdefault("actions", [])
        parsed.setdefault("extra", {})
        if not isinstance(parsed["actions"], list):
            parsed["actions"] = []
        if not isinstance(parsed["extra"], dict):
            parsed["extra"] = {}
        parsed["extra"].setdefault("steps", [])
        parsed["extra"].setdefault("products", [])
        parsed["extra"].setdefault("translated_text", "")
        if force_type:
            parsed["type"] = force_type
        return parsed
    except Exception:
        return fallback


def _extract_json(text: str) -> Any:
    if not text:
        return None
    t = text.strip()
    # Try exact json first
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        # As a last resort, attempt to quote keys minimally is risky; skip.
        return None
