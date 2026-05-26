"""GSM8K with a deterministic 4-way mix of few-shot prefix patterns.

idx % 4: 0=standard / 1=cluster / 2=random sample / 3=zero-shot.
Per-mode scores reported as score_standard / score_cluster / score_random /
score_zero_shot in addition to overall score.
"""

import random
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)
from sglang.test.simple_eval_gsm8k import (
    GSM8K_URL,
    GSM8KEval,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.utils import download_and_cache_file, read_jsonl

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
        # Bypass GSM8KEval.__init__: it builds a single ``_few_shot_prompt``
        # we don't use, and its training-prefix slicing conflicts with our
        # multi-block pool reservation.
        self._num_threads = num_threads
        self._num_shots = num_shots
        self._num_clusters = num_clusters
        self._seed = seed

        if data_path:
            filename = data_path
        else:
            filename = download_and_cache_file(GSM8K_URL)

        all_lines = list(read_jsonl(filename))

        cluster_block = num_shots * num_clusters
        pool_size = num_shots + cluster_block + random_pool_size
        if len(all_lines) < pool_size + 1:
            raise ValueError(
                f"GSM8K dataset has {len(all_lines)} examples but mixed-prefix "
                f"eval needs at least {pool_size + 1} (pool {pool_size} + 1 test)."
            )

        self._train_pool = all_lines[:pool_size]
        self._lines = all_lines[pool_size:]
        if num_examples is not None:
            self._lines = self._lines[:num_examples]

        self._standard_prefix = get_few_shot_examples(
            self._train_pool[:num_shots], num_shots
        )
        self._cluster_prefixes = [
            get_few_shot_examples(
                self._train_pool[
                    num_shots + k * num_shots : num_shots + (k + 1) * num_shots
                ],
                num_shots,
            )
            for k in range(num_clusters)
        ]
        self._random_pool = self._train_pool[num_shots + cluster_block :]

    def _pick_prefix(self, idx: int) -> str:
        mode = idx % 4
        if mode == 0:
            return self._standard_prefix
        if mode == 1:
            cluster_idx = (idx // 4) % self._num_clusters
            return self._cluster_prefixes[cluster_idx]
        if mode == 2:
            rng = random.Random(self._seed + idx)
            sampled_indices = rng.sample(range(len(self._random_pool)), self._num_shots)
            return "".join(
                get_one_example(self._random_pool, i, include_answer=True) + "\n\n"
                for i in sampled_indices
            )
        return ""

    @staticmethod
    def _mode_label(idx: int) -> str:
        return _MODE_LABELS[idx % 4]

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(idx: int) -> SingleEvalResult:
            question = get_one_example(self._lines, idx, include_answer=False)
            correct_answer = get_answer_value(self._lines[idx]["answer"])

            prefix = self._pick_prefix(idx)
            mode_label = self._mode_label(idx)
            prompt_content = prefix + question

            prompt_messages = [
                sampler._pack_message(content=prompt_content, role="user")
            ]

            try:
                response_text = sampler(prompt_messages)
            except Exception:
                response_text = ""

            extracted_answer = get_answer_value(response_text)
            score = float(extracted_answer == correct_answer)

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            # One per-mode key per sample so aggregate_results averages each
            # ``score_<mode>`` only over samples in that mode.
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={f"score_{mode_label}": score},
            )

        results = common.map_with_progress(
            fn, list(range(len(self._lines))), num_threads=self._num_threads
        )
        return common.aggregate_results(results, default_stats=("mean", "std"))
