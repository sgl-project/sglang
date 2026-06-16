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
"""Z-Image breakable CUDA graph (BCG) prompt padding.

Z-Image pads the masked text streams like the generic path but must also
rebuild its caption rotary-embedding cache for the padded length, since that
cache is the ``cap`` half of the ``freqs_cis`` tuple. Registered with the base
denoising stage's padder registry.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages import bcg_utils


def is_zimage_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return bcg_utils.transformer_class_name_matches(current_model, "zimage")


def build_zimage_cap_freqs(current_model: Any, target: int, device) -> Any:
    rotary_emb = getattr(current_model, "rotary_emb", None)
    if rotary_emb is None:
        return None

    axes = [
        torch.arange(1, target + 1, dtype=torch.int32, device=device),
        torch.zeros(target, dtype=torch.int32, device=device),
        torch.zeros(target, dtype=torch.int32, device=device),
    ]
    cap_pos_ids = torch.stack(axes, dim=-1)
    return rotary_emb(cap_pos_ids)


def pad_zimage_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    seq_and_dim = bcg_utils.prompt_seq_and_dim(call_kwargs)
    if seq_and_dim is None:
        return call_kwargs
    seq, seq_dim = seq_and_dim
    bucket = bcg_utils.select_text_bucket(seq, buckets)
    if bucket is None or seq == bucket:
        return call_kwargs

    out = dict(call_kwargs)
    for key in bcg_utils.TEXT_DIM1_KEYS:
        if key in out and out[key] is not None:
            out[key] = bcg_utils.pad_nested_text_dim(
                out[key], source=seq, target=bucket, preferred_dim=seq_dim
            )

    freqs_cis = out.get("freqs_cis")
    if isinstance(freqs_cis, tuple) and len(freqs_cis) == 2:
        cap_cache, image_cache = freqs_cis
        cap_tensor = bcg_utils.first_tensor(cap_cache)
        if torch.is_tensor(cap_tensor):
            cap_cache = (
                build_zimage_cap_freqs(current_model, bucket, cap_tensor.device)
                or cap_cache
            )
        out["freqs_cis"] = (cap_cache, image_cache)
    elif isinstance(freqs_cis, list) and len(freqs_cis) == 2:
        cap_cache, image_cache = freqs_cis
        cap_tensor = bcg_utils.first_tensor(cap_cache)
        if torch.is_tensor(cap_tensor):
            cap_cache = (
                build_zimage_cap_freqs(current_model, bucket, cap_tensor.device)
                or cap_cache
            )
        out["freqs_cis"] = [cap_cache, image_cache]

    return out


bcg_utils.register_prompt_padder(is_zimage_transformer, pad_zimage_prompt_kwargs)
