# Copyright 2023-2024 SGLang Team
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

"""Pure-Python RoPE correction helpers for fuzzy KV reuse.

Pulled out of ``model_runner.py`` so they can be unit-tested without
importing the full model runtime (which transitively requires torch
extensions like torchvision that aren't present in pure CPU CI).

Two helpers:

* ``copy_kv_with_rope_correction`` - per-layer K/V copy that reverses RoPE
  at the donor position and reapplies it at the target position. When a
  ``layer_recompute_mask`` is provided, flagged layers are zeroed instead
  of copied (the prefill compute path will produce fresh K/V for those
  layers).

* ``as_long_tensor`` - coerce list/numpy/torch input to a long-typed
  torch.Tensor on a target device. Used for segment-pos plumbing where
  Python providers may produce plain lists.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def as_long_tensor(obj, device) -> torch.Tensor:
    """Coerce list/numpy/torch input to a long-typed torch.Tensor on ``device``."""
    if obj is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=torch.long, non_blocking=True)
    return torch.as_tensor(list(obj), dtype=torch.long, device=device)


def copy_kv_with_rope_correction(
    pool,
    rotary_emb,
    old_locs: torch.Tensor,
    new_locs: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    layer_recompute_mask: Optional[List[bool]] = None,
    *,
    apply_rotary_emb=None,
    reverse_rotary_emb=None,
) -> None:
    """Copy donor K/V into ``new_locs`` with RoPE re-indexed to ``new_positions``.

    Args:
        pool: KV pool with ``k_buffer`` / ``v_buffer`` lists, ``layer_num``.
        rotary_emb: Rotary embedding object exposing ``cos_sin_cache``,
            ``is_neox_style``, ``rotary_dim``.
        old_locs: Source slot indices in the donor's KV pool.
        new_locs: Destination slot indices.
        old_positions: Donor-side absolute positions for ``old_locs``.
        new_positions: Target-side absolute positions for ``new_locs``.
        layer_recompute_mask: Optional list of bools; when ``mask[i]`` is
            True, layer ``i`` is zeroed instead of copied (the model recomputes).
        apply_rotary_emb / reverse_rotary_emb: Optional overrides for the
            rotary embedding kernels. Default: lazy-imported from SGLang's
            ``layers.rotary_embedding.utils``. The override hook lets unit
            tests run without pulling in SGLang's heavyweight rotary stack
            (which uses ``@torch.compile``).
    """
    if apply_rotary_emb is None or reverse_rotary_emb is None:
        from sglang.srt.layers.rotary_embedding.utils import (
            apply_rotary_emb as _apply,
            reverse_rotary_emb as _reverse,
        )
        apply_rotary_emb = apply_rotary_emb or _apply
        reverse_rotary_emb = reverse_rotary_emb or _reverse

    cos_sin_cache = rotary_emb.cos_sin_cache
    is_neox_style = rotary_emb.is_neox_style
    rotary_dim = rotary_emb.rotary_dim

    old_cos_sin = cos_sin_cache.index_select(0, old_positions)
    new_cos_sin = cos_sin_cache.index_select(0, new_positions)
    old_cos, old_sin = old_cos_sin.chunk(2, dim=-1)
    new_cos, new_sin = new_cos_sin.chunk(2, dim=-1)

    for layer_id in range(pool.layer_num):
        if layer_recompute_mask is not None and layer_id < len(layer_recompute_mask):
            if layer_recompute_mask[layer_id]:
                pool.v_buffer[layer_id][new_locs] = 0
                pool.k_buffer[layer_id][new_locs] = 0
                continue

        pool.v_buffer[layer_id][new_locs] = pool.v_buffer[layer_id][old_locs]

        k = pool.k_buffer[layer_id][old_locs]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]

        k_raw = reverse_rotary_emb(k_rot, old_cos, old_sin, is_neox_style)
        k_new = apply_rotary_emb(k_raw, new_cos, new_sin, is_neox_style)

        pool.k_buffer[layer_id][new_locs] = torch.cat((k_new, k_pass), dim=-1)
