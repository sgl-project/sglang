"""Prompt loading for branch_compare.

Two paths:
  1. --eval-name {gpqa, mmlu, gsm8k, math, humaneval, mgsm, aime25,
     longbench_v2}: instantiate the eval class from sglang.test.simple_eval_*
     and run it with a RecordingSampler (returns dummy text, captures the
     message_list). The captured messages become the prompts.
  2. --prompts-file FILE: JSONL with `{"prompt": "..."}` per line.

For --api chat we render the captured messages through the model's chat
template via HF AutoTokenizer (client-side); the server receives plain
text on /generate. This guarantees byte-identical input across phases.
For --api completion, we send the last message's content unmodified (or
the JSONL "prompt" field).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.test.simple_eval_common import SamplerBase


@dataclass
class Prompt:
    text: str  # what we send to /generate (chat-templated if --api=chat)
    messages: Optional[List[Dict[str, Any]]] = (
        None  # original chat messages for traceability
    )


class RecordingSampler(SamplerBase):
    """Captures every message_list passed in __call__, returns empty text.

    Used to extract prompts from sglang.test.simple_eval_* classes without
    actually running their grading.
    """

    def __init__(self):
        self.captured: List[List[Dict[str, Any]]] = []
        self.model = "branch-compare-recording-sampler"
        # Some evals look up `_completion_tokens` on the sampler.
        self._completion_tokens: List[int] = []

    def __call__(self, message_list):
        self.captured.append(message_list)
        return ""

    # Some evals expect ChatCompletionSampler-shaped helpers:
    def _pack_message(self, role: str, content: Any) -> Dict[str, Any]:
        return {"role": role, "content": content}


def _load_eval(eval_name: str, num_examples: int, num_threads: int = 1):
    """Mirror of the eval-name dispatch in sglang.test.run_eval."""
    if eval_name == "mmlu":
        from sglang.test.simple_eval_mmlu import MMLUEval

        return MMLUEval(
            "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
            num_examples,
            num_threads,
        )
    if eval_name == "gpqa":
        from sglang.test.simple_eval_gpqa import GPQAEval

        return GPQAEval(
            "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
            num_examples,
            num_threads,
        )
    if eval_name == "math":
        from sglang.test.simple_eval_common import ChatCompletionSampler
        from sglang.test.simple_eval_math import MathEval

        # Math eval needs a checker; supply a stub that always says equal.
        # We never run the grading anyway (RecordingSampler captures and bails).
        return MathEval(
            "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv",
            ChatCompletionSampler(model="gpt-4-turbo"),
            num_examples,
            num_threads,
        )
    if eval_name == "gsm8k":
        from sglang.test.simple_eval_gsm8k import GSM8KEval

        return GSM8KEval(num_examples=num_examples, num_threads=num_threads)
    if eval_name == "humaneval":
        from sglang.test.simple_eval_humaneval import HumanEval

        return HumanEval(num_examples, num_threads)
    if eval_name == "mgsm":
        from sglang.test.simple_eval_mgsm import MGSMEval

        return MGSMEval(num_examples, num_threads)
    if eval_name == "aime25":
        from sglang.test.simple_eval_aime25 import AIME25Eval

        return AIME25Eval(num_examples, num_threads)
    if eval_name == "longbench_v2":
        from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval

        return LongBenchV2Eval(
            data_source="THUDM/LongBench-v2",
            num_examples=num_examples,
            num_threads=num_threads,
        )
    raise ValueError(f"Unsupported eval_name: {eval_name}")


def _capture_messages_via_eval(
    eval_name: str, num_examples: int
) -> List[List[Dict[str, Any]]]:
    """Run the eval with a RecordingSampler. Grading runs on dummy responses
    and may print warnings; we only need the captured message_lists."""
    eval_obj = _load_eval(eval_name, num_examples, num_threads=1)
    sampler = RecordingSampler()
    try:
        eval_obj(sampler)
    except Exception:
        # Some evals try to grade dummy outputs and crash on the empty string;
        # whatever was captured before the crash is still usable.
        pass
    return sampler.captured[:num_examples]


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]]) -> str:
    """Render messages to a string the same way the server would on
    /v1/chat/completions, so that record/verify see byte-identical inputs.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _last_user_content(messages: List[Dict[str, Any]]) -> str:
    """For --api completion: pull the last user message's content as a
    plain string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            # Multimodal content list
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "".join(parts)
    raise ValueError(f"No user message in {messages!r}")


def load_prompts(
    *,
    eval_name: Optional[str],
    num_examples: Optional[int],
    prompts_file: Optional[str],
    api: str,
    tokenizer=None,
) -> List[Prompt]:
    if (eval_name is None) == (prompts_file is None):
        raise ValueError("Exactly one of --eval-name or --prompts-file must be set")

    if prompts_file is not None:
        prompts: List[Prompt] = []
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["prompt"]
                prompts.append(Prompt(text=text, messages=None))
        return prompts

    assert eval_name is not None and num_examples is not None
    captured = _capture_messages_via_eval(eval_name, num_examples)
    if not captured:
        raise RuntimeError(
            f"RecordingSampler captured no messages for eval={eval_name!r}; "
            f"the eval may have crashed before invoking the sampler."
        )

    prompts = []
    for messages in captured:
        if api == "chat":
            if tokenizer is None:
                raise ValueError("--api chat requires a tokenizer")
            text = _apply_chat_template(tokenizer, messages)
        elif api == "completion":
            text = _last_user_content(messages)
        else:
            raise ValueError(f"Unsupported --api: {api}")
        prompts.append(Prompt(text=text, messages=messages))
    return prompts
