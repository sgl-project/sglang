"""Minimal math scorer adapted from Nano-vLLM-UNO's Eval360 grader."""

from __future__ import annotations

import re
from typing import Any

try:
    from math_verify import parse, verify
except ModuleNotFoundError:  # Allow CLI help without optional eval dependencies.
    parse = None
    verify = None


def _require_math_verify() -> None:
    if parse is None or verify is None:
        raise RuntimeError("math scoring requires the 'math_verify' package")


def extract_last_boxed_content(text: str | None) -> str | None:
    if not text:
        return None
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text, re.IGNORECASE))
    for match in reversed(matches):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            depth += (text[index] == "{") - (text[index] == "}")
            index += 1
        if depth == 0:
            return text[start : index - 1].strip().strip("$").strip()
    return None


def _generation_list(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("generations"), list):
        return [str(value or "") for value in row["generations"]]
    return [str(row.get("generation") or "")]


def _get_accuracy(correct: list[bool]) -> float:
    return sum(correct) / len(correct) if correct else float("nan")


def _source_key(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("source_row", row.get("row", fallback)))


def normalize_answer_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().strip("$").strip()
    gsm8k_answer = re.search(r"####\s*([^\n]+)", text)
    if gsm8k_answer:
        text = gsm8k_answer.group(1).strip()
    text = re.sub(r"\\(?:boxed|fbox)\s*\{(.+)\}\s*$", r"\1", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)", "", text)
    text = text.replace("\\displaystyle", "")
    text = re.sub(r"\\mathbf\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\s+([A-Za-z])", r"\1", text)
    text = re.sub(r"\\\\\s*\[[^\]]+\]", r"\\\\", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\frac\s*([+-]?\d+)\s*([+-]?\d+)", r"\1/\2", text)
    text = text.replace("{,}", "")
    text = re.sub(r"\\(?:,|;|!|:)", "", text)
    text = text.replace("\\%", "%")
    text = text.replace("\\$", "").replace("$", "")
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\{([^{}])\}", r"^\1", text)
    text = re.sub(r"_\{([^{}])\}", r"_\1", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return re.sub(r"^[A-Za-z]+=(?=\\begin\{pmatrix\})", "", text)


def _has_plain_variable(text: str) -> bool:
    text = re.sub(r"\\(?:text|mathrm)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    return bool(re.search(r"[A-Za-z]", text))


def _parse_boxed_content(boxed: str) -> Any:
    _require_math_verify()
    boxed = normalize_answer_text(boxed)
    leading_number = re.match(r"\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", boxed)
    if leading_number and re.fullmatch(r"[A-Za-z]+", boxed[leading_number.end() :]):
        answer = leading_number.group(1).replace(",", "")
        try:
            return parse(answer)
        except Exception:
            return answer
    if _has_plain_variable(boxed):
        return boxed
    try:
        parsed = parse(f"${boxed}$") or parse(boxed)
        if parsed:
            return parsed
    except Exception:
        pass
    if leading_number:
        answer = leading_number.group(1).replace(",", "")
        try:
            return parse(answer)
        except Exception:
            return answer
    return boxed


def _parse_unboxed_answer(text: str) -> Any:
    _require_math_verify()
    try:
        parsed = parse(f"${text}$") or parse(text)
        if parsed:
            return parsed
    except Exception:
        pass

    patterns = [
        r"The answer is:?\s*\$?([\-0-9\.,]+)",
        r"#### ?\$?([\-0-9\.,]+)",
        r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Final answer:?\s*\$?([\-0-9\.,]+)",
        r"The final answer is:?\s*\$?([\-0-9\.,]+)",
        r"The answer is:?\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?",
        (
            r"=\s*\$?([\-0-9\.,]+)"
            r"\s*(?:miles?|minutes?|hours?|dollars?|GB)?"
            r"\.?\s*(?:The answer|$)"
        ),
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer = matches[-1].replace(",", "").strip().rstrip(".")
            try:
                return parse(answer)
            except Exception:
                return answer

    sentence_end = (
        r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)"
        r"\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?"
        r"\.?\s*$"
    )
    match = re.search(sentence_end, text, re.MULTILINE | re.IGNORECASE)
    if match:
        answer = match.group(1).replace(",", "").strip().rstrip(".")
        try:
            return parse(answer)
        except Exception:
            return answer

    for sentence in reversed(text.split(".")):
        if "Human:" in sentence or "Assistant:" in sentence:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sentence)
        if numbers:
            answer = numbers[-1].lstrip("0") or "0"
            try:
                return parse(answer)
            except Exception:
                return answer
    return None


def _parse_answer(text: str) -> Any:
    boxed = extract_last_boxed_content(text)
    return _parse_boxed_content(boxed) if boxed else _parse_unboxed_answer(text)


def _answer_candidates(text: str) -> list[Any]:
    boxed = extract_last_boxed_content(text)
    candidates = (
        [_parse_boxed_content(boxed), normalize_answer_text(boxed), boxed]
        if boxed
        else [_parse_unboxed_answer(text)]
    )
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate is not None and repr(candidate) not in seen:
            seen.add(repr(candidate))
            unique.append(candidate)
    return unique


def _vector_components(text: str) -> tuple[str, ...] | None:
    matrix = re.fullmatch(r"\\begin\{pmatrix\}(.+)\\end\{pmatrix\}", text)
    if matrix:
        return tuple(part for part in matrix.group(1).split(r"\\") if part)
    tuple_match = re.fullmatch(r"\(([^()]+)\)", text)
    if tuple_match and "," in tuple_match.group(1):
        return tuple(part for part in tuple_match.group(1).split(",") if part)
    return None


def _text_answers_match(answer: str, gold: str) -> bool:
    answer_norm = normalize_answer_text(answer)
    gold_norm = normalize_answer_text(gold)
    if answer_norm == gold_norm:
        return True
    try:
        if float(answer_norm.rstrip("%")) == float(gold_norm.rstrip("%")):
            return True
    except ValueError:
        pass
    if re.fullmatch(r"\([A-Za-z]\)", gold_norm) and answer_norm == gold_norm[1:-1]:
        return True
    answer_parts = _vector_components(answer_norm)
    gold_parts = _vector_components(gold_norm)
    return bool(answer_parts and answer_parts == gold_parts)


def _compare_answers(answer: Any, gold: str | None) -> bool:
    _require_math_verify()
    if not answer or gold is None:
        return False
    if isinstance(answer, str) and _text_answers_match(answer, gold):
        return True
    try:
        if verify(answer, gold):
            return True
        gold_answer = _parse_answer(gold)
        if gold_answer:
            return bool(verify(gold_answer, answer))
    except Exception:
        return isinstance(answer, str) and _text_answers_match(answer, gold)


def grade_math_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("ground_truth")
    expected_values = expected if isinstance(expected, list) else [expected]
    correct = []
    parsed_generations = []
    for generation in _generation_list(row):
        candidates = _answer_candidates(generation)
        parsed_generations.append([str(candidate) for candidate in candidates])
        correct.append(
            any(
                _compare_answers(candidate, str(gold))
                for candidate in candidates
                for gold in expected_values
                if gold is not None
            )
        )
    return {
        **row,
        "parsed_generations": parsed_generations,
        "correct": correct,
        "accuracy": _get_accuracy(correct),
        "grader": "math",
    }


def score_math(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graded = [grade_math_row(row) for row in rows]
    correct = [bool(value) for row in graded for value in row["correct"]]
    by_source: dict[str, list[tuple[int, bool]]] = {}
    for index, row in enumerate(graded):
        try:
            sample_index = int(row.get("sample_index", 0))
        except (TypeError, ValueError):
            sample_index = 0
        samples = by_source.setdefault(_source_key(row, index), [])
        samples.extend(
            (sample_index + offset, bool(value))
            for offset, value in enumerate(row["correct"])
        )

    per_problem = [
        [value for _, value in sorted(samples)]
        for samples in by_source.values()
        if samples
    ]
    samples_per_problem = max((len(samples) for samples in per_problem), default=0)
    sample0 = [samples[0] for samples in per_problem]
    summary = {
        "grader": "math",
        "num_rows": len(graded),
        "num_problems": len(per_problem),
        "samples_per_problem": samples_per_problem,
        "num_correct": sum(correct),
        "accuracy": _get_accuracy(correct),
    }
    if per_problem:
        summary.update(
            avg_at_1=_get_accuracy(sample0),
            pass_at_1=_get_accuracy(sample0),
            num_correct_at_1=sum(sample0),
        )
    if samples_per_problem > 1:
        summary.update(
            {
                f"avg_at_{samples_per_problem}": _get_accuracy(correct),
                f"pass_at_{samples_per_problem}": _get_accuracy(
                    [any(samples) for samples in per_problem]
                ),
                f"num_pass_at_{samples_per_problem}": sum(
                    any(samples) for samples in per_problem
                ),
            }
        )
    return graded, summary
