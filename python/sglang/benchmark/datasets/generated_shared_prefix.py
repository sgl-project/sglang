import pickle
import random
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt,
)


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

    @classmethod
    def from_args(cls, args: Namespace) -> "GeneratedSharedPrefixDataset":
        assert not getattr(args, "tokenize_prompt", False)
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
) -> List[DatasetRow]:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
    )
    should_cache = (range_ratio == 1) and not send_routing_key and num_turns == 1

    # Try to load from cache first
    if cache_path.exists() and should_cache:
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(
        f"\nGenerating new input data... "
        f"({num_groups=}, {prompts_per_group}, {system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=}, {num_turns=})"
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
            turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[
                1:
            ]
            full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
            prompt_len = 1 if fast_prepare else len(tokenizer.encode(turn_prompts[0]))
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

    # Save to cache
    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(input_requests, f)

    return input_requests
