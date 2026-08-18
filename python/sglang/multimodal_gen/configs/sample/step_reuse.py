# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class StepReuseParams(CacheParams):
    """
    Parameters for the step-reuse skip-forward strategy
    (see ``sglang.multimodal_gen.runtime.cache.step_reuse``).

    Unlike TeaCache's model-tuned polynomial rescaling, step-reuse uses a
    plain relative-L1 threshold on the modulated input, with an explicit cap
    on how many consecutive steps may reuse one real prediction.

    Attributes:
        cache_type: A string labeling these parameters as belonging to
            step-reuse.
        threshold: Relative L1 distance below which a new real prediction is
            considered similar enough to open a reuse window.
        max_skip_steps: Max consecutive steps that may reuse one real
            prediction before a real forward is forced again.
        history_size: Max number of past observations kept for the
            similarity decision.
    """

    cache_type: str = "step_reuse"
    threshold: float = 0.1
    max_skip_steps: int = 1
    history_size: int = 1
