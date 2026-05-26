import random
from typing import Optional

from sglang.test.simple_eval_gsm8k import GSM8KEval, get_one_example


class MixedPrefixGSM8KEval(GSM8KEval):
    """GSM8K with a per-question variable-length primary + random secondary prefix.

    Each query's prefix =
        primary_shots[:k]
      + random_subset(secondary_pool)

    where k ~ Uniform{0..num_shots} and the secondary subset is itself a
    random sample of random size in {0..secondary_pool_size}. The primary
    portion is order-stable across queries, so two queries picking k1 <= k2
    primary shots share the first k1 examples as a radix-cacheable prefix.
    The secondary tail makes each full prefix unique.
    """

    def __init__(
        self,
        num_examples: Optional[int],
        num_threads: int,
        num_shots: int,
        secondary_pool_size: int,
        data_path: Optional[str],
        seed: int,
    ):
        self._secondary_pool_size = secondary_pool_size
        self._seed = seed
        super().__init__(
            num_examples=num_examples,
            num_threads=num_threads,
            num_shots=num_shots,
            data_path=data_path,
        )

    def _setup_prefix_pool(self, all_lines: list, num_shots: int) -> int:
        overall_pool_size = num_shots + self._secondary_pool_size
        if len(all_lines) < overall_pool_size + 1:
            raise ValueError(
                f"GSM8K dataset has {len(all_lines)} examples but mixed-prefix "
                f"eval needs at least {overall_pool_size + 1} "
                f"(num_shots {num_shots} + secondary "
                f"{self._secondary_pool_size} + 1 test)."
            )
        self._primary_shots = all_lines[:num_shots]
        self._secondary_pool = all_lines[num_shots:overall_pool_size]
        return overall_pool_size

    def _build_prefix(self, idx: int) -> str:
        rng = random.Random(self._seed + idx)
        num_primary = rng.randint(0, self._num_shots)
        primary = self._primary_shots[:num_primary]
        secondary_size = rng.randint(0, self._secondary_pool_size)
        secondary = rng.sample(self._secondary_pool, secondary_size)
        combined = primary + secondary
        return "".join(
            get_one_example(combined, i, include_answer=True) + "\n\n"
            for i in range(len(combined))
        )
