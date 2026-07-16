# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ShapeKey — typed identifier for one captured CUDA-graph shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape across all runners.

    size: the per-phase capture size — what the runner iterates over.
        - prefill: num_tokens
        - decode:  bs
    stream_idx:   pdmux stream index, or None for single-stream runners.
    variant_label: LoRA-variant label ("lora" / "nolora"), or None
        for runners that don't record per-variant graphs.
    extra_label: backend-specific graph variant label, or None. Used when a
        backend needs multiple captures for the same logical shape.
    """

    size: int
    stream_idx: Optional[int] = None
    variant_label: Optional[str] = None
    extra_label: Optional[str] = None
