"""extract_v1 — versioned final-answer extraction.

One public entry point:

    extract_answer(text, answer_type, scope="full", labels=None) -> dict

Every call returns a dict, always:

    {
      "extraction_version": "extract_v1",
      "status":   "parsed" | "unparsed" | "ambiguous" | "malformed",
      "prediction": <normalized answer or None>,
      "strategy": <which rule produced it, or None>,
      "candidates": [ ... ],     # all values found by the winning strategy
      "unit": <str or None>,     # numeric answers only
      "flags": [ ... ],
    }

Rules worth knowing:

- scope is "full" (whole completion; used for zero-shot-style prose answers)
  or "first_paragraph" (used for completion-style prompts whose answers come
  first and then drift, matching the old grade.py few-shot scoping).
- Numbers: precedence is \\boxed{X} -> "answer is X"/"answer: X" ->
  "final answer ... X" -> **X** -> "= X" -> last number in scope. Within the
  winning strategy the LAST value wins; multiple distinct values add a
  "multiple_distinct_values" flag. Two different \\boxed{} values are genuinely
  ambiguous -> status "ambiguous" (boxed is an explicit final marker).
- Number tokens accept $, thousand separators, decimals, negatives, and a/b
  fractions. A trailing unit ("cm^2", "apples", "%") is captured as metadata,
  not discarded silently.
- Nothing is ever silently discarded: statuses unparsed/ambiguous/malformed
  are returned with flags, and harness.grade additionally logs them to
  parse_failures.jsonl.
"""

import re

EXTRACTION_VERSION = "extract_v1"

NUM_TOKEN = r"(?<!\^)-?\$?\d[\d,]*\.?\d*(?:\s*/\s*\d+)?"
NUM_TOKEN_RE = re.compile(NUM_TOKEN)
UNIT_RE = re.compile(r"^\s*([A-Za-z²³%][\w²³%/^\- ]{0,19})")

ANSWER_IS_RE = re.compile(r"answer\s+(?:is\s+|=+\s*)?(?:approximately\s+)?(" + NUM_TOKEN + ")", re.IGNORECASE)
ANSWER_COLON_RE = re.compile(r"answer\s*:\s*(" + NUM_TOKEN + ")", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"final\s+answer[^\d\-]*(" + NUM_TOKEN + ")", re.IGNORECASE)
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
EQUALS_RE = re.compile(r"=\s*(" + NUM_TOKEN + ")")
YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def first_paragraph(text):
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            return para.strip()
    return ""


def nonempty_lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def normalize_text(s):
    return re.sub(r"\s+", " ", str(s).strip().strip('"').strip().rstrip(".")).lower()


def parse_number_token(tok):
    """'$1,234.50', '1,234', '-3.', '3/4' -> float, or None."""
    tok = tok.replace("$", "").replace(",", "").strip().rstrip(".")
    if "/" in tok:
        num, _, den = tok.partition("/")
        try:
            den = float(den.strip())
            if den == 0:
                return None
            return float(num.strip()) / den
        except ValueError:
            return None
    try:
        return float(tok)
    except ValueError:
        return None


def find_boxed(text):
    """Brace-balanced scan for \\boxed{...}; returns list of inner strings."""
    inners = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            inners.append(text[m.end(): i - 1])
        else:
            inners.append(None)  # unbalanced braces -> malformed
    return inners


def _result(status, prediction, strategy, candidates, flags, unit=None):
    return {
        "extraction_version": EXTRACTION_VERSION,
        "status": status,
        "prediction": prediction,
        "strategy": strategy,
        "candidates": candidates,
        "unit": unit,
        "flags": flags,
    }


def _capture_unit(scope, num_match_end):
    """Trailing unit right after the winning number token, else None."""
    m = UNIT_RE.match(scope[num_match_end:])
    if not m:
        return None
    unit = m.group(1).strip()
    return unit or None


def _extract_number(scope):
    flags = []

    # 1. \boxed{...}
    boxed = find_boxed(scope)
    if boxed:
        parsed = []
        for b in boxed:
            if b is None:
                continue
            v = parse_number_token(b.strip())
            if v is None:  # e.g. "24 \text{ cm}^2": take the first number inside
                inner = NUM_TOKEN_RE.search(b)
                v = parse_number_token(inner.group()) if inner else None
            if v is not None:
                parsed.append(v)
        if any(b is None for b in boxed) or (boxed and len(parsed) < len(boxed) and not parsed):
            if not parsed:
                return _result("malformed", None, "boxed", [], flags + ["boxed_malformed"])
            flags.append("boxed_malformed")
        distinct = sorted(set(parsed))
        if len(distinct) > 1:
            return _result("ambiguous", None, "boxed", distinct, flags)
        if parsed:
            last_box = [b for b in boxed if b is not None][-1]
            end = scope.rfind(last_box) + len(last_box)
            return _result("parsed", parsed[-1], "boxed", parsed, flags,
                           unit=_capture_unit(scope, end))

    # 2-5. marker strategies in precedence order; last match wins.
    strategies = [
        ("answer_is", ANSWER_IS_RE),
        ("answer_colon", ANSWER_COLON_RE),
        ("final_answer", FINAL_ANSWER_RE),
    ]
    for name, pattern in strategies:
        matches = list(pattern.finditer(scope))
        values = [parse_number_token(m.group(1)) for m in matches]
        values = [v for v in values if v is not None]
        if values:
            if len(set(values)) > 1:
                flags.append("multiple_distinct_values")
            return _result("parsed", values[-1], name, values, flags,
                           unit=_capture_unit(scope, matches[-1].end()))

    bold = [m for m in BOLD_RE.finditer(scope)]
    bold_hits = []
    for m in bold:
        inner = NUM_TOKEN_RE.search(m.group(1))
        if inner:
            v = parse_number_token(inner.group())
            if v is not None:
                bold_hits.append((v, m.start() + inner.start()))
    if bold_hits:
        values = [v for v, _ in bold_hits]
        if len(set(values)) > 1:
            flags.append("multiple_distinct_values")
        return _result("parsed", values[-1], "bold", values, flags)

    matches = list(EQUALS_RE.finditer(scope))
    values = [parse_number_token(m.group(1)) for m in matches]
    values = [v for v in values if v is not None]
    if values:
        if len(set(values)) > 1:
            flags.append("multiple_distinct_values")
        return _result("parsed", values[-1], "equals", values, flags,
                       unit=_capture_unit(scope, matches[-1].end()))

    matches = list(NUM_TOKEN_RE.finditer(scope))
    values = [parse_number_token(m.group()) for m in matches]
    values = [v for v in values if v is not None]
    if values:
        if len(set(values)) > 1:
            flags.append("multiple_distinct_values")
        return _result("parsed", values[-1], "last_number", values, flags,
                       unit=_capture_unit(scope, matches[-1].end()))

    return _result("unparsed", None, None, [], flags + ["no_numeric_candidate"])


def _scope_text(text, scope):
    return first_paragraph(text) if scope == "first_paragraph" else text


def _extract_yesno(scope):
    m = YESNO_RE.search(scope)
    if not m:
        return _result("unparsed", None, None, [], ["no_yesno_candidate"])
    return _result("parsed", m.group(1).lower(), "first_yesno", [m.group(1).lower()], [])


def _extract_label(scope, labels):
    low = scope.lower()
    hits = []
    masked = low
    for label in sorted(labels, key=len, reverse=True):
        idx = masked.find(label.lower())
        if idx != -1:
            hits.append((idx, label.lower()))
            masked = masked.replace(label.lower(), "#" * len(label))
    if not hits:
        return _result("unparsed", None, None, [], ["no_label_candidate"])
    hits.sort()
    return _result("parsed", min(hits)[1], "first_label", [h[1] for h in hits], [])


def _extract_containment(scope, gold, answer_type):
    """name/string: normalized containment of the gold string in scope.

    Honest caveat: this strategy uses the gold answer at grading time, so it
    measures whether the gold appears in the conclusion, not free-form
    extraction ability. Prediction is the matched gold on success.
    """
    gold_norm = normalize_text(gold)
    scope_norm = normalize_text(scope)
    if gold_norm in scope_norm:
        return _result("parsed", gold_norm, "gold_containment", [gold_norm], [])
    return _result("unparsed", None, "gold_containment", [],
                   ["gold_not_found_in_scope"])


def extract_answer(text, answer_type, scope="full", labels=None, gold=None):
    if not text or not text.strip():
        return _result("unparsed", None, None, [], ["empty_output"])

    scoped = _scope_text(text, scope)

    if answer_type == "number":
        return _extract_number(scoped)
    if answer_type == "yesno":
        return _extract_yesno(scoped)
    if answer_type == "label":
        if not labels:
            raise ValueError("label extraction requires labels")
        return _extract_label(scoped, labels)
    if answer_type in ("name", "string"):
        if gold is None:
            raise ValueError(f"{answer_type} extraction requires gold")
        return _extract_containment(scoped, gold, answer_type)
    if answer_type == "code":
        raise ValueError("code answers are graded by execution, not extraction")
    raise ValueError(f"unknown answer_type {answer_type!r}")


def answers_match(extraction, gold, answer_type):
    """Comparison after extraction. Numbers compare numerically (1e-6);
    yes/no, label, name, string compare normalized strings."""
    if extraction["status"] != "parsed":
        return False
    if answer_type == "number":
        gold_num = parse_number_token(str(gold))
        pred = extraction["prediction"]
        if gold_num is None or not isinstance(pred, (int, float)):
            return False
        return abs(pred - gold_num) < 1e-6
    return normalize_text(extraction["prediction"]) == normalize_text(gold)
