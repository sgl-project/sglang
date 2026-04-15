# SPDX-License-Identifier: Apache-2.0
"""Rollout / RL hooks for Z-Image pipeline configs."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    all_gather_if_sp_sharded,
    maybe_trim_sp_rope_seq_for_batch,
)


class ZImageRolloutPipelineMixin:

    def gather_dit_env_static_for_sp(self, batch, cond_kwargs: dict | None):
        if cond_kwargs is None:
            return None
        out = dict(cond_kwargs)
        freqs = out.get("freqs_cis")
        if freqs is not None:
            cap_freqs, x_freqs = freqs[0], freqs[1]
            if isinstance(x_freqs, torch.Tensor) and x_freqs.dim() >= 2:
                x_g = all_gather_if_sp_sharded(batch, x_freqs, dim=0)
                x_g = maybe_trim_sp_rope_seq_for_batch(batch, x_g)
                out["freqs_cis"] = (cap_freqs, x_g)
        return out
