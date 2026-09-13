import json
import math
import random
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.benchmark.datasets.generated_shared_prefix import (
    sample_generated_shared_prefix_requests,
)
from sglang.benchmark.datasets.random import sample_random_requests

SOURCES = ("random-ids", "sharegpt", "generated-shared-prefix")
ARRIVAL_PATTERNS = ("constant", "poisson")


def _positive_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return value


def _positive_int(value: Any, field: str) -> int:
    number = _positive_number(value, field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def load_workload_plan(path: str) -> dict:
    plan_path = Path(path)
    try:
        plan = json.loads(plan_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"dynamic workload does not exist: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dynamic workload JSON: {exc}") from exc

    if not isinstance(plan, dict) or not isinstance(plan.get("phases"), list):
        raise ValueError("dynamic workload must contain a phases list")
    if not plan["phases"]:
        raise ValueError("dynamic workload must contain at least one phase")
    if plan.get("prompt_pool_size") is not None:
        _positive_int(plan["prompt_pool_size"], "prompt_pool_size")

    names = set()
    for index, phase in enumerate(plan["phases"]):
        prefix = f"phases[{index}]"
        if not isinstance(phase, dict):
            raise ValueError(f"{prefix} must be an object")
        name = phase.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{prefix}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate dynamic workload phase: {name}")
        names.add(name)
        _positive_number(phase.get("duration"), f"{prefix}.duration")
        _positive_number(phase.get("request_rate"), f"{prefix}.request_rate")
        _positive_int(phase.get("input_len"), f"{prefix}.input_len")
        _positive_int(phase.get("output_len"), f"{prefix}.output_len")
        if phase.get("max_concurrency") is not None:
            _positive_int(phase.get("max_concurrency"), f"{prefix}.max_concurrency")
        range_ratio = phase.get("range_ratio", 1.0)
        _positive_number(range_ratio, f"{prefix}.range_ratio")
        if range_ratio > 1:
            raise ValueError(f"{prefix}.range_ratio must be at most 1")
        if phase.get("extra_request_body") is not None and not isinstance(
            phase["extra_request_body"], dict
        ):
            raise ValueError(f"{prefix}.extra_request_body must be an object")
        source = phase.get("source", plan.get("source", "random-ids"))
        if source not in SOURCES:
            raise ValueError(f"{prefix}.source must be one of {SOURCES}")
        pattern = phase.get("arrival_pattern", plan.get("arrival_pattern", "constant"))
        if pattern not in ARRIVAL_PATTERNS:
            raise ValueError(
                f"{prefix}.arrival_pattern must be one of {ARRIVAL_PATTERNS}"
            )
    return plan


def generate_arrival_offsets(
    duration: float,
    request_rate: float,
    pattern: str,
    rng: np.random.Generator,
) -> List[float]:
    if pattern == "constant":
        count = max(1, math.ceil(duration * request_rate))
        return [
            index / request_rate
            for index in range(count)
            if index / request_rate < duration
        ]

    offsets = [0.0]
    elapsed = 0.0
    while True:
        elapsed += float(rng.exponential(1.0 / request_rate))
        if elapsed >= duration:
            break
        offsets.append(elapsed)
    return offsets


def _generate_requests(
    *,
    source: str,
    input_len: int,
    output_len: int,
    count: int,
    range_ratio: float,
    tokenizer: Any,
    dataset_path: str,
    seed: int,
) -> List[DatasetRow]:
    if source in ("random-ids", "sharegpt"):
        return sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=count,
            range_ratio=range_ratio,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            random_sample=source == "sharegpt",
            return_text=True,
        )

    num_groups = min(4, count)
    prompts_per_group = math.ceil(count / num_groups)
    # Leave room for decode/encode drift and the separator inserted by the
    # existing shared-prefix generator.
    prompt_budget = max(2, input_len * 9 // 10)
    system_prompt_len = max(1, prompt_budget * 3 // 4)
    question_len = max(1, prompt_budget - system_prompt_len)
    requests = sample_generated_shared_prefix_requests(
        num_groups=num_groups,
        prompts_per_group=prompts_per_group,
        system_prompt_len=system_prompt_len,
        question_len=question_len,
        output_len=output_len,
        range_ratio=range_ratio,
        tokenizer=tokenizer,
        seed=seed,
        ordered=True,
    )
    return requests[:count]


def _encode(tokenizer: Any, prompt: str, *, add_special_tokens: bool) -> list[int]:
    try:
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TypeError:
        return tokenizer.encode(prompt)


def fit_prompt_length(tokenizer: Any, prompt: str, target: int) -> tuple[str, int]:
    """Best-effort text round-trip with an exact tokenizer-visible length."""
    special_tokens = int(tokenizer.num_special_tokens_to_add())
    target_content = max(1, target - special_tokens)
    token_ids = _encode(tokenizer, prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [next(iter(tokenizer.get_vocab().values()))]
    while len(token_ids) < target_content:
        token_ids.extend(token_ids[: target_content - len(token_ids)])
    token_ids = token_ids[:target_content]

    for _ in range(8):
        prompt = tokenizer.decode(token_ids)
        actual = len(_encode(tokenizer, prompt, add_special_tokens=True))
        if actual == target:
            return prompt, actual
        target_content = max(1, target_content + target - actual)
        while len(token_ids) < target_content:
            token_ids.extend(token_ids[: target_content - len(token_ids)])
        token_ids = token_ids[:target_content]
    return prompt, actual


@dataclass
class DynamicDataset(BaseDataset):
    workload_path: str
    dataset_path: str
    seed: int

    @classmethod
    def from_args(cls, args: Namespace) -> "DynamicDataset":
        return cls(
            workload_path=args.dynamic_workload_path,
            dataset_path=args.dataset_path,
            seed=args.seed,
        )

    def load(self, tokenizer: Any, model_id: Optional[str] = None) -> List[DatasetRow]:
        del model_id
        plan = load_workload_plan(self.workload_path)
        random.seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        phase_start = 0.0
        requests = []
        request_pools = {}
        prompt_pool_size = int(plan.get("prompt_pool_size", 8))

        for index, phase in enumerate(plan["phases"]):
            duration = float(phase["duration"])
            request_rate = float(phase["request_rate"])
            source = phase.get("source", plan.get("source", "random-ids"))
            pattern = phase.get(
                "arrival_pattern", plan.get("arrival_pattern", "constant")
            )
            offsets = generate_arrival_offsets(duration, request_rate, pattern, rng)
            input_len = int(phase["input_len"])
            output_len = int(phase["output_len"])
            range_ratio = float(phase.get("range_ratio", 1.0))
            pool_key = (source, input_len, output_len, range_ratio)
            pool = request_pools.get(pool_key)
            if pool is None:
                pool = _generate_requests(
                    source=source,
                    input_len=input_len,
                    output_len=output_len,
                    count=prompt_pool_size,
                    range_ratio=range_ratio,
                    tokenizer=tokenizer,
                    dataset_path=self.dataset_path,
                    seed=self.seed + index,
                )
                for request in pool:
                    target_len = (
                        input_len
                        if range_ratio == 1
                        else request.prompt_len
                        + int(tokenizer.num_special_tokens_to_add())
                    )
                    request.prompt, request.prompt_len = fit_prompt_length(
                        tokenizer, request.prompt, target_len
                    )
                    request.text_prompt_len = request.prompt_len
                request_pools[pool_key] = pool
            for request_index, offset in enumerate(offsets):
                request = replace(pool[request_index % len(pool)])
                request.timestamp = (phase_start + offset) * 1000
                request.phase = phase["name"]
                request.phase_duration = duration
                request.phase_request_rate = request_rate
                request.phase_max_concurrency = phase.get("max_concurrency")
                request.extra_request_body = dict(
                    phase.get("extra_request_body") or request.extra_request_body
                )
                requests.append(request)
            phase_start += duration

        return requests
