"""
QA Utilities — Shared answer extraction, comparison, and normalization functions.

This module consolidates the duplicated logic previously found in step3 and step6,
providing a single source of truth for all QA-task answer handling.

Key design decisions:
  - Regex patterns are compiled once at module level (performance)
  - All public functions accept `answer_type` ("multiple_choice" | "numerical")
  - `ast.literal_eval` is used instead of `eval()` for safety
  - Ground truth access is centralized to handle both old/new formats
"""

import re
import ast
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Precompiled regex patterns (compiled once, reused across calls)
# ---------------------------------------------------------------------------

# Multiple-choice patterns — ordered by specificity (first match wins)
_PATTERNS_MC = [
    re.compile(r'(?:the\s+)?answer\s+(?:is|:)\s*\**([A-E])\b', re.IGNORECASE),
    re.compile(r'(?:correct\s+)?(?:choice|answer)\s+(?:is|:)?\s*([A-E])\b', re.IGNORECASE),
    re.compile(r'(?:^|\n)\s*([A-E])\s*[\.\)\,]', re.MULTILINE),
    re.compile(r'\(([A-E])\)\s*[^A-E]*$'),
    re.compile(r'\b([A-E])\s*\.?\s*$', re.MULTILINE),
]

# Numerical patterns — ordered by specificity
_PATTERNS_NUM = [
    re.compile(r'\$\$?\s*([\d,]+(?:\.\d+)?)\s*\$\$?'),
    re.compile(r'#{2,}\s*([\d,]+(?:\.\d+)?)'),
    re.compile(r'(?:the\s+)?answer\s+(?:is|:)\s*\**([\d,]+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'(?:therefore|so|thus|finally),?\s+(?:the\s+)?answer\s+(?:is|:)\s*([\d,]+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'=\s*([\d,]+(?:\.\d+)?)\s*\.?\s*$', re.MULTILINE),
    re.compile(r'([\d,]+(?:\.\d+)?)\s*[\.\s]*$', re.MULTILINE),
]


# ---------------------------------------------------------------------------
# 1. Answer extraction
# ---------------------------------------------------------------------------

# Patterns that signal the start of the assistant response in Qwen chat format
_ASSISTANT_MARKERS = [
    re.compile(r'<\|im_start\|>assistant', re.IGNORECASE),
    re.compile(r'\nassistant\s*:?\s*\n', re.IGNORECASE),
]


def _strip_prompt(text: str) -> str:
    """Strip the prompt/context from generated text, keeping only the assistant response.

    The generated_text from step2 contains the full sequence (prompt + response).
    Answer-extraction regexes can falsely match choice labels (A., B., C.) inside
    the prompt.  This helper finds the last assistant marker and returns everything
    after it.
    """
    last_pos = -1
    for pat in _ASSISTANT_MARKERS:
        for m in pat.finditer(text):
            last_pos = max(last_pos, m.end())
    if last_pos >= 0:
        return text[last_pos:].strip()
    return text


def extract_answer(generated_text: str, answer_type: str) -> Optional[str]:
    """Extract the model's answer from generated text.

    For multiple-choice: returns a single uppercase letter (A-E).
    For numerical: returns a string of digits (commas stripped).

    Args:
        generated_text: Full decoded text from model generation (prompt + response).
        answer_type: "multiple_choice" or "numerical".

    Returns:
        Extracted answer string, or None if not found.
    """
    if not generated_text:
        return None

    # Only search the assistant's response, not the prompt containing choices
    text = _strip_prompt(generated_text.strip())

    if answer_type == "multiple_choice":
        for pattern in _PATTERNS_MC:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()
        # Fallback: first standalone capital letter A-E
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1).upper()

    elif answer_type == "numerical":
        for pattern in _PATTERNS_NUM:
            match = pattern.search(text)
            if match:
                return match.group(1).replace(",", "")

    return None


def get_clean_answer(model_output: Dict) -> Tuple[str, str]:
    """Extract the model's clean answer from a step2 output dict.

    For multiple-choice questions, ``generated_text`` contains the full
    prompt (including choice labels A/B/C/D).  Naively searching the full
    text for a letter can match a choice label instead of the actual answer.

    This function provides a **robust, centralized** answer extraction:

    Priority:
      1. ``generated_answer_only`` — pure answer text (no prompt), added by
         step2 V6+.  This is the cleanest source.
      2. ``extract_answer(generated_text, answer_type)`` — falls back to the
         regex extractor which internally calls ``_strip_prompt()`` to remove
         everything before the last ``<|im_start|>assistant`` marker.

    Args:
        model_output: Dict produced by ``step2_run_inference.py``.

    Returns:
        (answer_str, answer_type) — answer_str may be "" if extraction fails.
    """
    # --- Priority 1: pre-computed clean answer (step2 V6+) ---
    clean = model_output.get("generated_answer_only", "")
    if clean:
        clean = clean.strip()
        # For multiple-choice, the clean answer is typically just "C" or "C\n"
        # Strip whitespace/newlines
        clean = clean.split("\n")[0].strip()
        if clean:
            answer_type = get_answer_type(model_output.get("ground_truth", {}))
            return clean, answer_type

    # --- Priority 2: extract from full text via _strip_prompt() ---
    generated_text = model_output.get("generated_text", "")
    answer_type = get_answer_type(model_output.get("ground_truth", {}))
    answer = extract_answer(generated_text, answer_type)
    return answer or "", answer_type


# ---------------------------------------------------------------------------
# 2. Ground truth helpers
# ---------------------------------------------------------------------------

def get_gt_answer(ground_truth: Any) -> Optional[str]:
    """Extract the ground truth answer from the unified QA format.

    Handles:
      - Dict with "answer" key (unified QA format)
      - Raw string answers (legacy)
    """
    if not ground_truth:
        return None

    if isinstance(ground_truth, dict):
        answer = ground_truth.get("answer")
        if answer is not None:
            return str(answer).strip()

    return None


def get_answer_type(ground_truth: Any) -> str:
    """Get the answer type from ground truth.

    Returns "multiple_choice" or "numerical".
    """
    if isinstance(ground_truth, dict):
        return ground_truth.get("answer_type", "multiple_choice")
    return "multiple_choice"


# ---------------------------------------------------------------------------
# 3. Numerical answer normalization
# ---------------------------------------------------------------------------

def normalize_numerical_answer(pred: str, gt: str) -> Tuple[str, str]:
    """Normalize numerical answers for comparison.

    Handles cases like "1.0" vs "1", "$42" vs "42", "1,024" vs "1024".

    Returns:
        Tuple of (normalized_pred, normalized_gt) as strings.
    """
    pred = pred.strip()
    gt = gt.strip()

    for char in ["$", "%", ",", " "]:
        pred = pred.replace(char, "")
        gt = gt.replace(char, "")

    try:
        pred_float = float(pred)
        gt_float = float(gt)
        if pred_float == int(pred_float) and gt_float == int(gt_float):
            return str(int(pred_float)), str(int(gt_float))
        return str(pred_float), str(gt_float)
    except (ValueError, OverflowError):
        return pred, gt


# ---------------------------------------------------------------------------
# 4. Answer comparison
# ---------------------------------------------------------------------------

def compare_answers(
    predicted_answer: Optional[str],
    ground_truth: Any,
) -> Dict[str, Any]:
    """Compare predicted answer against ground truth.

    Returns a dict with:
      - match: bool
      - has_error: bool
      - error_subtype: str | None
      - predicted_answer, gt_answer, answer_type
    """
    gt_answer = get_gt_answer(ground_truth)
    answer_type = get_answer_type(ground_truth)

    result = {
        "match": False,
        "has_error": True,
        "error_subtype": None,
        "predicted_answer": predicted_answer,
        "gt_answer": gt_answer,
        "answer_type": answer_type,
    }

    if gt_answer is None:
        result["has_error"] = False
        result["error_subtype"] = "gt_unavailable"
        return result

    if predicted_answer is None:
        result["error_subtype"] = "no_output"
        return result

    if answer_type == "numerical":
        pred_norm, gt_norm = normalize_numerical_answer(predicted_answer, gt_answer)
    else:
        pred_norm = predicted_answer.strip().upper()
        gt_norm = gt_answer.strip().upper()

    if pred_norm == gt_norm:
        result["match"] = True
        result["has_error"] = False
    else:
        result["has_error"] = True
        result["error_subtype"] = "wrong_answer"

    return result


# ---------------------------------------------------------------------------
# 5. Token ID helpers
# ---------------------------------------------------------------------------

def get_answer_token_id(
    tokenizer, answer: str, answer_type: str
) -> Optional[int]:
    """Get the first token ID for an answer string."""
    if not answer:
        return None
    if answer_type == "multiple_choice":
        tokens = tokenizer.encode(answer.upper(), add_special_tokens=False)
    else:
        tokens = tokenizer.encode(answer, add_special_tokens=False)
    return tokens[0] if tokens else None


def find_answer_step(
    generated_ids, input_len, tokenizer, predicted_answer, answer_type
) -> Tuple[int, Optional[int]]:
    """Find the generation step where the answer token first appears.

    Returns:
        (step_index, token_id) — step_index is 0-based from first generated token.
    """
    new_ids = generated_ids[input_len:].tolist()
    token_id = get_answer_token_id(tokenizer, predicted_answer, answer_type)

    if token_id is None:
        return 0, None

    for step in range(len(new_ids)):
        if new_ids[step] == token_id:
            return step, token_id

    return 0, token_id


# ---------------------------------------------------------------------------
# 6. Safe details/dict parsing (replaces eval())
# ---------------------------------------------------------------------------

def parse_details(raw) -> Dict[str, Any]:
    """Safely parse a details value that may be a dict, JSON string, or invalid.

    Replaces unsafe `eval()` with `ast.literal_eval()`.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw not in ("{}", "", "nan", "None"):
        try:
            return ast.literal_eval(raw)
        except (ValueError, TypeError, SyntaxError):
            pass
    return {}
