"""
MMMU evaluation for VLMs using the run_eval simple-evals interface.

"""

from __future__ import annotations

import base64
import io
from typing import List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset
from PIL import Image

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    map_with_progress,
)


class MMMUVLMEval(Eval):
    DOMAIN_CAT2SUB_CAT = {
        "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
        "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
        "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
        "Health and Medicine": [
            "Basic_Medical_Science",
            "Clinical_Medicine",
            "Diagnostics_and_Laboratory_Medicine",
            "Pharmacy",
            "Public_Health",
        ],
        "Humanities and Social Science": [
            "History",
            "Literature",
            "Sociology",
            "Psychology",
        ],
        "Tech and Engineering": [
            "Agriculture",
            "Architecture_and_Engineering",
            "Computer_Science",
            "Electronics",
            "Energy_and_Power",
            "Materials",
            "Mechanical_Engineering",
        ],
    }

    def __init__(
        self, num_examples: Optional[int] = 100, num_threads: int = 32, seed: int = 42
    ):
        """Create MMMU VLM eval (Math subset, 100 fixed samples by default)."""
        self.num_examples = num_examples
        self.num_threads = num_threads
        self.seed = seed
        # Prepare samples deterministically across all MMMU subjects (validation split)
        self.samples = self._prepare_mmmu_samples(self.num_examples)

    @staticmethod
    def _to_data_uri(image: Image.Image) -> str:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _build_mc_mapping(options: List[str]) -> Tuple[dict, List[str]]:
        index2ans = {}
        all_choices = []
        ch = ord("A")
        for opt in options:
            letter = chr(ch)
            index2ans[letter] = opt
            all_choices.append(letter)
            ch += 1
        return index2ans, all_choices

    def _prepare_mmmu_samples(self, k: int) -> List[dict]:
        # Subjects and domains copied from MMMU data_utils to categorize results
        subjects: List[str] = []
        for subs in self.DOMAIN_CAT2SUB_CAT.values():
            subjects.extend(subs)

        # Load validation split of each subject
        datasets = []
        for subj in subjects:
            try:
                d = load_dataset("MMMU/MMMU", subj, split="validation")
                # attach subject info via transform
                d = d.add_column("__subject__", [subj] * len(d))
                datasets.append(d)
            except Exception:
                continue
        if not datasets:
            raise RuntimeError("Failed to load MMMU datasets")

        merged = concatenate_datasets(datasets)

        # Deterministic selection: sort by id (fallback to subject+index)
        def _key(idx):
            ex = merged[idx]
            return str(ex.get("id", f"{ex['__subject__']}:{idx}"))

        order = sorted(range(len(merged)), key=_key)
        picked_indices = order[:k]

        samples: List[dict] = []
        for idx in picked_indices:
            ex = merged[idx]
            subject = ex["__subject__"]
            image = ex.get("image_1")
            if image is None or not hasattr(image, "convert"):
                continue
            data_uri = self._to_data_uri(image)
            question = ex.get("question", "")
            answer = ex.get("answer")
            raw_options = ex.get("options")
            question_type = "open"
            index2ans = None
            all_choices = None
            options = None
            if raw_options:
                try:
                    options = (
                        raw_options
                        if isinstance(raw_options, list)
                        else list(eval(raw_options))
                    )
                    if isinstance(options, list) and len(options) > 0:
                        index2ans, all_choices = self._build_mc_mapping(options)
                        question_type = "multiple-choice"
                except Exception:
                    options = None

            # Build final textual prompt; include choices if MC
            prompt_text = f"Question: {question}\n\n"
            if options:
                letters = [chr(ord("A") + i) for i in range(len(options))]
                for letter, opt in zip(letters, options):
                    prompt_text += f"{letter}) {opt}\n"
            prompt_text += "\nAnswer: "

            samples.append(
                {
                    "id": ex.get("id", f"{subject}:{idx}"),
                    "final_input_prompt": prompt_text,
                    "image_data": data_uri,
                    "answer": answer,
                    "question_type": question_type,
                    "index2ans": index2ans,
                    "all_choices": all_choices,
                    "category": subject,
                }
            )

        return samples

    @staticmethod
    def _split_prompt_for_image(prompt: str) -> tuple[str, str]:
        """Split a prompt containing an inline image tag into prefix and suffix.

        If no tag is present, treat the whole prompt as prefix and empty suffix.
        """
        if "<" in prompt and ">" in prompt:
            prefix = prompt.split("<")[0]
            suffix = prompt.split(">", 1)[1]
            return prefix, suffix
        return prompt, ""

    @staticmethod
    def build_chat_messages_from_prompt(prompt: str, image_data) -> List:
        """Split a prompt containing an inline image tag into prefix and suffix.

        If no tag is present, treat the whole prompt as prefix and empty suffix.
        """
        # Build a vision+text message for OpenAI-compatible API
        prefix, suffix = MMMUVLMEval._split_prompt_for_image(prompt)

        content: List[dict] = []
        if prefix:
            content.append({"type": "text", "text": prefix})
        content.append({"type": "image_url", "image_url": {"url": image_data}})
        if suffix:
            content.append({"type": "text", "text": suffix})
        prompt_messages = [{"role": "user", "content": content}]

        return prompt_messages

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(sample: dict):
            prompt = sample["final_input_prompt"]
            image_data = sample["image_data"]
            prompt_messages = MMMUVLMEval.build_chat_messages_from_prompt(
                prompt, image_data
            )

            # Sample
            response_text = sampler(prompt_messages)
            response_text = response_text or ""

            # Parse and score
            gold = sample["answer"]
            if (
                sample["question_type"] == "multiple-choice"
                and sample["all_choices"]
                and sample["index2ans"]
            ):
                pred = _parse_multi_choice_response(
                    response_text, sample["all_choices"], sample["index2ans"]
                )
                score = 1.0 if (gold is not None and pred == gold) else 0.0
                extracted_answer = pred
            else:
                parsed_list = _parse_open_response(response_text)
                score = (
                    1.0 if (gold is not None and _eval_open(gold, parsed_list)) else 0.0
                )
                extracted_answer = ", ".join(map(str, parsed_list))

            html_rendered = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=gold,
                extracted_answer=extracted_answer,
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html_rendered,
                score=score,
                metrics={"__category__": sample["category"]},
                convo=convo,
            )

        results = map_with_progress(fn, self.samples, self.num_threads)

        # Build category table and overall accuracy
        # Gather per-sample correctness and category
        per_cat_total: dict[str, int] = {}
        per_cat_correct: dict[str, int] = {}
        htmls = []
        convos = []
        scores: List[float] = []
        for r in results:
            # __category__ stored under metrics
            cat = r.metrics.get("__category__") if r.metrics else None
            if cat is None:
                cat = "Unknown"
            per_cat_total[cat] = per_cat_total.get(cat, 0) + 1
            if r.score:
                per_cat_correct[cat] = per_cat_correct.get(cat, 0) + 1
            htmls.append(r.html)
            convos.append(r.convo)
            if r.score is not None:
                scores.append(r.score)

        evaluation_result = {}
        for cat, tot in per_cat_total.items():
            corr = per_cat_correct.get(cat, 0)
            acc = (corr / tot) if tot > 0 else 0.0
            evaluation_result[cat] = {"acc": round(acc, 3), "num_example": tot}

        printable_results = {}
        # Domains first
        for domain, cats in self.DOMAIN_CAT2SUB_CAT.items():
            acc_sum = 0.0
            num_sum = 0
            for cat in cats:
                if cat in evaluation_result:
                    acc_sum += (
                        evaluation_result[cat]["acc"]
                        * evaluation_result[cat]["num_example"]
                    )
                    num_sum += evaluation_result[cat]["num_example"]
            if num_sum > 0:
                printable_results[f"Overall-{domain}"] = {
                    "num": num_sum,
                    "acc": round(acc_sum / num_sum, 3),
                }
            # add each sub-category row if present
            for cat in cats:
                if cat in evaluation_result:
                    printable_results[cat] = {
                        "num": evaluation_result[cat]["num_example"],
                        "acc": evaluation_result[cat]["acc"],
                    }

        # Overall
        total_num = sum(v["num_example"] for v in evaluation_result.values())
        overall_acc = (
            sum(v["acc"] * v["num_example"] for v in evaluation_result.values())
            / total_num
            if total_num > 0
            else 0.0
        )
        printable_results["Overall"] = {"num": total_num, "acc": round(overall_acc, 3)}

        # Build EvalResult
        return EvalResult(
            score=overall_acc, metrics=printable_results, htmls=htmls, convos=convos
        )


def _parse_multi_choice_response(
    response: str, all_choices: List[str], index2ans: dict
) -> str:
    # loosely adapted from benchmark mmmu eval
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    # Prefer explicit letter with bracket e.g. (A)
    candidates: List[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        # try match by option text
        for idx, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(idx)
    if not candidates:
        # fallback to first choice
        return all_choices[0]
    if len(candidates) == 1:
        return candidates[0]
    # choose the last occurrence
    starts = []
    for can in candidates:
        pos = response.rfind(f"({can})")
        if pos == -1:
            pos = response.rfind(f" {can} ")
        if pos == -1 and index2ans.get(can):
            pos = response.lower().rfind(index2ans[can].lower())
        starts.append(pos)
    return candidates[int(max(range(len(starts)), key=lambda i: starts[i]))]


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False


def _normalize_str(s: str):
    s = s.strip()
    if _check_is_number(s):
        s = s.replace(",", "")
        try:
            v = round(float(s), 2)
            return [v]
        except Exception:
            return [s.lower()]
    return [s.lower()] if len(s) > 1 else [" " + s, s + " "]


def _extract_numbers(s: str) -> List[str]:
    import re as _re

    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        _re.findall(pattern_commas, s)
        + _re.findall(pattern_scientific, s)
        + _re.findall(pattern_simple, s)
    )


def _parse_open_response(response: str) -> List[str]:
    import re as _re

    def get_key_subresponses(resp: str) -> List[str]:
        resp = resp.strip().strip(".").lower()
        subs = _re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        keys = []
        for i, s in enumerate(subs):
            cands = [*indicators]
            if i == len(subs) - 1:
                cands.append("=")
            shortest = None
            for ind in cands:
                if ind in s:
                    part = s.split(ind)[-1].strip()
                    if not shortest or len(part) < len(shortest):
                        shortest = part
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                keys.append(shortest)
        return keys or [resp]

    key_resps = get_key_subresponses(response)
    pred_list = key_resps.copy()
    for r in key_resps:
        pred_list.extend(_extract_numbers(r))
    out = []
    for x in pred_list:
        out.extend(_normalize_str(x))
    # dedup
    return list(dict.fromkeys(out))


def _eval_open(gold, preds: List[str]) -> bool:
    if isinstance(gold, list):
        norm_answers = []
        for ans in gold:
            norm_answers.extend(_normalize_str(ans))
    else:
        norm_answers = _normalize_str(gold)
    for p in preds:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False
