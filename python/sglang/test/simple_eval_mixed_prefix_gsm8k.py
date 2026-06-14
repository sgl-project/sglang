import random
from typing import Optional

from sglang.test.simple_eval_gsm8k import GSM8KEval, get_one_example


class MixedPrefixGSM8KEval(GSM8KEval):

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
        secondary_size = rng.randint(0, self._secondary_pool_size)
        secondary_indices = rng.sample(range(len(self._secondary_pool)), secondary_size)
        primary = self._primary_shots[:num_primary]
        secondary = [self._secondary_pool[i] for i in secondary_indices]
        combined = primary + secondary
        return "".join(
            get_one_example(combined, i, include_answer=True) + "\n\n"
            for i in range(len(combined))
        )
