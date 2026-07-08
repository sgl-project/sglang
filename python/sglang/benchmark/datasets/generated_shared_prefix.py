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

        # Defensive validation for in-process callers that construct a
        # Namespace by hand and bypass the argparse boundary in
        # serving.py. The CLI hook enforces the same rules first.
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
        elif zipf_alpha is not None:
            raise ValueError(
                "--gsp-zipf-alpha is only meaningful with "
                "--gsp-group-distribution=zipf; remove --gsp-zipf-alpha "
                "or set --gsp-group-distribution=zipf"
            )

        prompts_per_group = args.gsp_prompts_per_group
        warmup_requests = getattr(args, "warmup_requests", 0) or 0
        num_groups = args.gsp_num_groups
        if warmup_requests > 0 and num_groups > 0:
            extra_per_group = warmup_requests // num_groups
            remainder = warmup_requests % num_groups
            prompts_per_group += extra_per_group
            if remainder > 0:
                prompts_per_group += 1  # one extra in first few groups
            print(
                f"[gsp] warmup_requests={warmup_requests}: "
                f"generating {prompts_per_group} prompts per group "
                f"(original {args.gsp_prompts_per_group} + {warmup_requests} warmup). "
                f"First {warmup_requests} prompts will be consumed by warmup, "
                f"remaining {args.gsp_prompts_per_group * num_groups} for benchmark.",
                flush=True,
            )

        return cls(
            num_groups=num_groups,
            prompts_per_group=prompts_per_group,
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
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
):
    """Create cache directory under ~/.cache/sglang/benchmark.

    The uniform-mode filename is preserved exactly as before so existing
    on-disk caches remain valid. Non-default sampling modes get an extra
    suffix encoding the parameters that affect the cached payload.
    """
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    suffix = ""
    if group_distribution != "uniform":
        suffix = f"_{group_distribution}_{zipf_alpha}"

    cache_key = (
        f"gen_shared_prefix_{seed}_{num_groups}_{prompts_per_group}_"
        f"{system_prompt_len}_{question_len}_{output_len}{suffix}_"
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
    1 and group index 0 is the hottest. Sampling uses an isolated
    numpy.random.default_rng(seed) so the shared question/system-prompt pool
    stays byte-identical to uniform mode for the same seed and other args.
    Zipf mode is cached on disk under a distinct key per (group_distribution,
    zipf_alpha) value.
    """
    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
        group_distribution=group_distribution,
        zipf_alpha=zipf_alpha,
    )
    # range_ratio != 1 / num_turns > 1 perturb the payload but are not in the
    # cache key; send_routing_key embeds a per-run uuid + timestamp that is
    # meaningless to cache. Bypass for these pre-existing reasons only.
    should_cache = range_ratio == 1 and not send_routing_key and num_turns == 1

    if should_cache and cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not should_cache:
        print(f"\nCache bypassed ({range_ratio=}, {send_routing_key=}, {num_turns=})")

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

    system_prompts = [
        gen_prompt(tokenizer, system_prompt_lens[i]) for i in range(num_groups)
    ]

    # shape: (num_groups, prompts_per_group, num_turns)
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

    # Per-slot group assignment. Uniform mode is the identity assignment
    # [0,0,...,1,1,...,N-1,N-1]; zipf mode samples from the rank distribution
    # using an isolated RNG so the module-level random / numpy.random state
    # that compute_random_lens / gen_prompt rely on is never perturbed -- this
    # keeps the system-prompt and question pool byte-identical to uniform mode
    # for the same seed and other args.
    total_slots = num_groups * prompts_per_group
    if group_distribution == "uniform":
        assignment = np.repeat(np.arange(num_groups), prompts_per_group)
    else:  # "zipf"
        rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        assignment = rng.choice(num_groups, size=total_slots, replace=True, p=probs)

    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0
    for slot_idx, sampled_g in enumerate(
        tqdm(assignment, desc="Generating shared-prefix prompts")
    ):
        # src_(g,p) walks the question pool in uniform-enumeration order, so
        # per-slot question text is reproducibly identical across modes.
        src_g, src_p = divmod(slot_idx, prompts_per_group)
        sampled_g = int(sampled_g)

        system_prompt = system_prompts[sampled_g]
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{sampled_g}"
            if send_routing_key
            else None
        )
        turn_questions = questions[src_g][src_p]
        turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[1:]
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

    if not ordered:
        random.shuffle(input_requests)

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

    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(input_requests, f)

    return input_requests
