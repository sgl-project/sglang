import argparse
import math
import pickle
import random
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt,
)


def _finite_positive_float(value) -> float:
    """argparse-compatible type for a finite, strictly positive float.

    Rejects NaN, infinities, zero, and negatives. Also used as a
    defensive check in GeneratedSharedPrefixDataset.from_args.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected a finite float > 0, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a finite float > 0, got {value!r}")
    return parsed


def _zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray:
    """Rank-based Zipf probability vector with rank starting at 1.

    weight(rank)      = 1 / rank ** alpha       (rank in 1..num_groups)
    probability(rank) = weight(rank) / sum_over_all_ranks(weight)

    The returned array has length num_groups; element i corresponds to
    group index i (rank i + 1), so group 0 is the hottest.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = 1.0 / (ranks**alpha)
    return weights / weights.sum()


@dataclass
class GeneratedSharedPrefixDataset(BaseDataset):
    num_groups: int
    prompts_per_group: int
    system_prompt_len: int
    question_len: int
    output_len: int
    range_ratio: float
    seed: int
    fast_prepare: bool
    send_routing_key: bool
    num_turns: int
    ordered: bool
    group_distribution: str = "uniform"
    zipf_alpha: Optional[float] = None

    @classmethod
    def from_args(cls, args: Namespace) -> "GeneratedSharedPrefixDataset":
        assert not getattr(args, "tokenize_prompt", False)
        group_distribution = getattr(args, "gsp_group_distribution", "uniform")
        zipf_alpha = getattr(args, "gsp_zipf_alpha", None)

        if group_distribution not in ("uniform", "zipf"):
            raise ValueError(
                f"--gsp-group-distribution must be 'uniform' or 'zipf', "
                f"got {group_distribution!r}"
            )
        if group_distribution == "zipf":
            if zipf_alpha is None:
                raise ValueError(
                    "--gsp-group-distribution=zipf requires --gsp-zipf-alpha "
                    "(a finite float > 0)"
                )
            if not math.isfinite(zipf_alpha) or zipf_alpha <= 0:
                raise ValueError(
                    f"--gsp-zipf-alpha must be a finite float > 0, got {zipf_alpha!r}"
                )
        else:
            if zipf_alpha is not None:
                raise ValueError(
                    "--gsp-zipf-alpha is only meaningful with "
                    "--gsp-group-distribution=zipf; remove --gsp-zipf-alpha "
                    "or set --gsp-group-distribution=zipf"
                )

        return cls(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            seed=args.seed,
            fast_prepare=getattr(args, "gsp_fast_prepare", False),
            send_routing_key=getattr(args, "gsp_send_routing_key", False),
            num_turns=getattr(args, "gsp_num_turns", 1),
            ordered=getattr(args, "gsp_ordered", False),
            group_distribution=group_distribution,
            zipf_alpha=zipf_alpha,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_generated_shared_prefix_requests(
            num_groups=self.num_groups,
            prompts_per_group=self.prompts_per_group,
            system_prompt_len=self.system_prompt_len,
            question_len=self.question_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            seed=self.seed,
            send_routing_key=self.send_routing_key,
            num_turns=self.num_turns,
            fast_prepare=self.fast_prepare,
            ordered=self.ordered,
            group_distribution=self.group_distribution,
            zipf_alpha=self.zipf_alpha,
        )


def get_gen_prefix_cache_path(
    seed: int,
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer,
):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    cache_key = (
        f"gen_shared_prefix_{seed}_{num_groups}_{prompts_per_group}_"
        f"{system_prompt_len}_{question_len}_{output_len}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
    send_routing_key: bool = False,
    num_turns: int = 1,
    fast_prepare: bool = False,
    ordered: bool = False,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
) -> List[DatasetRow]:
    """Generate benchmark requests with shared system prompts using random tokens and caching.

    When group_distribution is "uniform" (default), each group receives exactly
    prompts_per_group requests; behavior matches the legacy generator.

    When group_distribution is "zipf", each request's group is sampled by rank
    with probability 1/rank**zipf_alpha / sum_k(1/k**zipf_alpha); rank starts at
    1 and group index 0 is the hottest. The on-disk cache is bypassed in this
    mode. Sampling uses an isolated numpy.random.default_rng(seed) so the
    shared question/system-prompt pool stays byte-identical to uniform mode for
    the same seed and other args.
    """
    if group_distribution not in ("uniform", "zipf"):
        raise ValueError(
            f"group_distribution must be 'uniform' or 'zipf', got {group_distribution!r}"
        )
    if group_distribution == "zipf":
        if zipf_alpha is None:
            raise ValueError(
                "group_distribution='zipf' requires zipf_alpha (a finite float > 0)"
            )
        if not math.isfinite(zipf_alpha) or zipf_alpha <= 0:
            raise ValueError(
                f"zipf_alpha must be a finite float > 0, got {zipf_alpha!r}"
            )
    elif zipf_alpha is not None:
        raise ValueError("zipf_alpha is only meaningful with group_distribution='zipf'")

    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
    )
    should_cache = (
        group_distribution == "uniform"
        and range_ratio == 1
        and not send_routing_key
        and num_turns == 1
    )

    # Try to load from cache first (only when caching is enabled).
    if should_cache and cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(
        f"\nGenerating new input data... "
        f"({num_groups=}, {prompts_per_group}, {system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=}, {num_turns=}, {group_distribution=}, {zipf_alpha=})"
    )

    run_random_str = uuid.uuid4().hex[:8]
    run_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    system_prompt_lens = compute_random_lens(
        full_len=system_prompt_len,
        range_ratio=range_ratio,
        num=num_groups,
    )
    question_lens = np.array(
        compute_random_lens(
            full_len=question_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group * num_turns,
        )
    ).reshape(num_groups, prompts_per_group, num_turns)
    output_lens = np.array(
        compute_random_lens(
            full_len=output_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group,
        )
    ).reshape(num_groups, prompts_per_group)
    del system_prompt_len, question_len, output_len

    # Generate system prompts for each group
    system_prompts = [
        gen_prompt(tokenizer, system_prompt_lens[i]) for i in range(num_groups)
    ]

    # Generate questions: shape (num_groups, prompts_per_group, num_turns)
    questions = [
        [
            [
                gen_prompt(tokenizer, int(question_lens[g, p, t]))
                for t in range(num_turns)
            ]
            for p in range(prompts_per_group)
        ]
        for g in range(num_groups)
    ]

    # Combine system prompts with questions
    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0

    if group_distribution == "zipf":
        # Isolated RNG: never perturbs the module-level random / numpy.random
        # state that compute_random_lens / gen_prompt rely on. This keeps the
        # generated system-prompt and question pool byte-identical to uniform
        # mode for the same seed and other args.
        rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        total_slots = num_groups * prompts_per_group
        sampled_groups = rng.choice(num_groups, size=total_slots, replace=True, p=probs)

        for slot_idx in tqdm(
            range(total_slots), desc="Generating zipf-sampled prompts"
        ):
            # Source slot index in the uniform enumeration: walks the question
            # pool in the same order uniform mode does, so the per-slot
            # question text is reproducibly identical.
            src_g = slot_idx // prompts_per_group
            src_p = slot_idx % prompts_per_group
            sampled_g = int(sampled_groups[slot_idx])

            system_prompt = system_prompts[sampled_g]
            routing_key = (
                f"{run_random_str}_{run_start_timestamp}_{sampled_g}"
                if send_routing_key
                else None
            )
            turn_questions = questions[src_g][src_p]
            turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[
                1:
            ]
            full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
            prompt_len = 1 if fast_prepare else len(tokenizer.encode(turn_prompts[0]))
            output_len_val = int(output_lens[src_g, src_p])

            input_requests.append(
                DatasetRow(
                    prompt=full_prompt,
                    prompt_len=prompt_len,
                    output_len=output_len_val,
                    routing_key=routing_key,
                )
            )
            total_input_tokens += prompt_len
            total_output_tokens += output_len_val
    else:
        for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
            system_prompt = system_prompts[group_idx]
            routing_key = (
                f"{run_random_str}_{run_start_timestamp}_{group_idx}"
                if send_routing_key
                else None
            )
            for prompt_idx in tqdm(
                range(prompts_per_group), desc="Generating questions", leave=False
            ):
                turn_questions = questions[group_idx][prompt_idx]
                turn_prompts = [
                    f"{system_prompt}\n\n{turn_questions[0]}"
                ] + turn_questions[1:]
                full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
                prompt_len = (
                    1 if fast_prepare else len(tokenizer.encode(turn_prompts[0]))
                )
                output_len_val = int(output_lens[group_idx, prompt_idx])

                input_requests.append(
                    DatasetRow(
                        prompt=full_prompt,
                        prompt_len=prompt_len,
                        output_len=output_len_val,
                        routing_key=routing_key,
                    )
                )
                total_input_tokens += prompt_len
                total_output_tokens += output_len_val

    if not ordered:
        random.shuffle(input_requests)

    # Print statistics
    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Number of turns: {num_turns}")
    print(f"Group distribution: {group_distribution}")
    if group_distribution == "zipf":
        print(f"Zipf alpha: {zipf_alpha}")
    print(f"Total prompts: {len(input_requests)}")
    if not fast_prepare:
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(
            f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
        )
        all_questions = [q for group in questions for conv in group for q in conv]
        print(
            f"Average question length: {sum(len(tokenizer.encode(q)) for q in all_questions) / len(all_questions):.1f} tokens\n"
        )

    # Save to cache (only when caching is enabled).
    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(input_requests, f)

    return input_requests
