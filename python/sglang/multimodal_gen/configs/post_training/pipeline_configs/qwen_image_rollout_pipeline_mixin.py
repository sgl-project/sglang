# SPDX-License-Identifier: Apache-2.0
"""Rollout / RL hooks for Qwen-Image pipeline configs."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    all_gather_if_sp_sharded,
    maybe_trim_sp_rope_seq_for_batch,
)


class QwenImageRolloutPipelineMixin:

    def gather_dit_env_static_for_sp(self, batch, cond_kwargs: dict | None):
        if cond_kwargs is None:
            return None
        out = dict(cond_kwargs)
        freqs = out.get("freqs_cis")
        if freqs is not None:
            img_cache, txt_cache = freqs[0], freqs[1]
            if isinstance(img_cache, torch.Tensor) and img_cache.dim() == 2:
                img_g = all_gather_if_sp_sharded(batch, img_cache, dim=0)
                img_g = maybe_trim_sp_rope_seq_for_batch(batch, img_g)
                out["freqs_cis"] = (img_g, txt_cache)
        return out
