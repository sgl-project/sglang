"""GSM8K with a deterministic 4-way mix of few-shot prefix patterns.

idx % 4: 0=standard / 1=cluster / 2=random sample / 3=zero-shot.
Per-mode scores reported as score_standard / score_cluster / score_random /
score_zero_shot in addition to overall score.
"""

import random
from typing import Optional

from sglang.test.simple_eval_gsm8k import (
    GSM8KEval,
    get_few_shot_examples,
    get_one_example,
)

_MODE_LABELS = ("standard", "cluster", "random", "zero_shot")


class MixedPrefixGSM8KEval(GSM8KEval):
    def __init__(
        self,
        num_examples: Optional[int] = 100,
        num_threads: int = 128,
        num_shots: int = 10,
        num_clusters: int = 5,
        random_pool_size: int = 50,
        data_path: Optional[str] = None,
        seed: int = 42,
    ):
        self._num_clusters = num_clusters
        self._random_pool_size = random_pool_size
        self._seed = seed
        super().__init__(
            num_examples=num_examples,
            num_threads=num_threads,
            num_shots=num_shots,
            data_path=data_path,
        )

    def _setup_prefix_pool(self, all_lines: list, num_shots: int) -> int:
        cluster_block = num_shots * self._num_clusters
        pool_size = num_shots + cluster_block + self._random_pool_size
        if len(all_lines) < pool_size + 1:
            raise ValueError(
                f"GSM8K dataset has {len(all_lines)} examples but mixed-prefix "
                f"eval needs at least {pool_size + 1} (pool {pool_size} + 1 test)."
            )
        self._standard_prefix = get_few_shot_examples(all_lines[:num_shots], num_shots)
        self._cluster_prefixes = [
            get_few_shot_examples(
                all_lines[num_shots + k * num_shots : num_shots + (k + 1) * num_shots],
                num_shots,
            )
            for k in range(self._num_clusters)
        ]
        self._random_pool = all_lines[num_shots + cluster_block : pool_size]
        return pool_size

    def _build_prefix(self, idx: int) -> str:
        mode = idx % 4
        if mode == 0:
            return self._standard_prefix
        if mode == 1:
            return self._cluster_prefixes[(idx // 4) % self._num_clusters]
        if mode == 2:
            rng = random.Random(self._seed + idx)
            sampled = rng.sample(range(len(self._random_pool)), self._num_shots)
            return "".join(
                get_one_example(self._random_pool, i, include_answer=True) + "\n\n"
                for i in sampled
            )
        return ""

    def _extra_sample_metrics(self, idx: int, score: float) -> dict:
        return {f"score_{_MODE_LABELS[idx % 4]}": score}
