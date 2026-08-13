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
"""Qwen-Image breakable CUDA graph (BCG) prompt padding.

Qwen-Image / Qwen-Image-Edit carry text length on dim 1 of
``encoder_hidden_states`` and a separate ``freqs_cis`` text-rope cache plus
``txt_seq_lens``; they may not pass an explicit prompt mask, so this padder
synthesizes one. Registered with the base denoising stage's padder registry.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)


def is_qwen_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "qwen")
        and "txt_seq_lens" in call_kwargs
        and "freqs_cis" in call_kwargs
    )


def pad_qwen_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    ehs = call_kwargs.get("encoder_hidden_states")
    ehs_tensor = bcg_utils.first_tensor(ehs)
    if not torch.is_tensor(ehs_tensor) or ehs_tensor.dim() < 2:
        return call_kwargs

    seq = ehs_tensor.shape[1]
    bucket = bcg_utils.select_text_bucket(seq, buckets)
    if bucket is None:
        return call_kwargs

    out = dict(call_kwargs)
    if seq < bucket:
        out["encoder_hidden_states"] = bcg_utils.pad_nested_dim(
            ehs, dim=1, source=seq, target=bucket
        )
        if (
            "encoder_hidden_states_2" in out
            and out["encoder_hidden_states_2"] is not None
        ):
            out["encoder_hidden_states_2"] = bcg_utils.pad_nested_dim(
                out["encoder_hidden_states_2"], dim=1, source=seq, target=bucket
            )

    mask = out.get("encoder_hidden_states_mask")
    if mask is None:
        mask = torch.ones(
            ehs_tensor.shape[:2],
            device=ehs_tensor.device,
            dtype=torch.bool,
        )
    if mask is not None:
        out["encoder_hidden_states_mask"] = bcg_utils.pad_nested_dim(
            mask, dim=1, source=seq, target=bucket
        )

    if "encoder_attention_mask" in out and out["encoder_attention_mask"] is not None:
        out["encoder_attention_mask"] = bcg_utils.pad_nested_dim(
            out["encoder_attention_mask"], dim=1, source=seq, target=bucket
        )

    freqs_cis = out.get("freqs_cis")
    if isinstance(freqs_cis, tuple) and len(freqs_cis) == 2:
        img_cache, txt_cache = freqs_cis
        txt_cache = bcg_utils.pad_nested_dim(
            txt_cache, dim=0, source=seq, target=bucket
        )
        out["freqs_cis"] = (img_cache, txt_cache)
    elif isinstance(freqs_cis, list) and len(freqs_cis) == 2:
        img_cache, txt_cache = freqs_cis
        txt_cache = bcg_utils.pad_nested_dim(
            txt_cache, dim=0, source=seq, target=bucket
        )
        out["freqs_cis"] = [img_cache, txt_cache]

    out["txt_seq_lens"] = bcg_utils.bucket_txt_seq_lens(out.get("txt_seq_lens"), bucket)
    return out


bcg_utils.register_prompt_padder(is_qwen_transformer, pad_qwen_prompt_kwargs)
