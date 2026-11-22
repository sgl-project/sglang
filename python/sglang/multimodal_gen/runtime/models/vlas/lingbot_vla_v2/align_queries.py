# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
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

# Adapted for inference from huggingface/lerobot at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
"""Learned alignment query tokens that LingBot-VLA 2.0 appends after the language prefix."""

from __future__ import annotations

import torch
from torch import nn


class AlignQueries:
    """Policy mixin; the query parameters stay on the policy module, so checkpoint keys are unchanged."""

    def _init_align(self, params: dict, llm_hidden_size: int):
        """Build the learned query tokens and projections used during action inference."""
        self.use_align = bool(params)
        self.num_task_tokens = 0
        self.use_future_depth = self.use_future_video = self.use_future_video_patch = (
            False
        )
        self.use_current_shared_task_proj = self.use_current_video_patch = (
            self.use_future_video_cls
        ) = False
        self.future_video_share_future_depth_query = (
            self.use_shared_future_task_proj
        ) = False
        self.block_future_depth_to_action = self.block_suffix_to_future_video = False
        if not params:
            return
        if params.get("mode") != "query":
            raise ValueError(
                f"align_params.mode must be 'query', got {params.get('mode')!r}."
            )
        dim = params["llm"]["dim_out"]
        if dim != llm_hidden_size:
            raise ValueError(
                f"align_params.llm.dim_out ({dim}) must equal the VLM hidden size ({llm_hidden_size})."
            )
        depth = params["depth"]
        self.num_task_tokens = params["num_task_tokens"]
        if depth["num_backbone_tokens"] % self.num_task_tokens:
            raise ValueError(
                "align_params.depth.num_backbone_tokens must be divisible by num_task_tokens."
            )
        self.use_future_depth = bool(depth.get("use_future_depth", False))
        self.block_future_depth_to_action = bool(
            depth.get("block_future_depth_to_action", False)
        )
        self.depth_align_embs = nn.Parameter(
            torch.randn(depth["num_backbone_tokens"], dim)
        )
        if self.use_future_depth:
            self.future_depth_align_embs = nn.Parameter(
                torch.randn(depth["num_backbone_tokens"], dim)
            )

        self.use_future_video = bool(params.get("use_future_video", False))
        if not self.use_future_video:
            return
        video = {**depth, **params.get("video", {})}
        self.use_future_video_patch = bool(video.get("use_patch_loss", True))
        self.use_current_video_patch = use_current_patch = bool(
            video.get("use_current_patch_loss", False)
        )
        self.use_current_shared_task_proj = bool(
            video.get("use_current_shared_task_proj", use_current_patch)
        )
        self.use_future_video_cls = bool(video.get("use_cls_loss", False))
        self.future_video_share_future_depth_query = bool(
            video.get("share_future_depth_query", False)
        )
        self.use_shared_future_task_proj = bool(
            video.get("use_shared_future_task_proj", False)
        )
        self.block_suffix_to_future_video = bool(
            video.get("block_suffix_to_future_video", False)
        )
        if use_current_patch and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_current_patch_loss requires use_patch_loss."
            )
        if self.use_current_shared_task_proj and not use_current_patch:
            raise ValueError(
                "align_params.video.use_current_shared_task_proj requires use_current_patch_loss."
            )
        if self.use_shared_future_task_proj and not (
            self.use_future_video_patch and self.future_video_share_future_depth_query
        ):
            raise ValueError(
                "align_params.video.use_shared_future_task_proj requires use_patch_loss and share_future_depth_query."
            )
        if self.future_video_share_future_depth_query and (
            not self.use_future_depth
            or video["num_backbone_tokens"] != depth["num_backbone_tokens"]
        ):
            raise ValueError(
                "align_params.video.share_future_depth_query requires depth.use_future_depth and equal "
                "num_backbone_tokens."
            )
        if self.use_future_video_patch:
            if use_current_patch:
                self.current_video_align_embs = nn.Parameter(
                    torch.randn(video["num_backbone_tokens"], dim)
                )
                if self.use_current_shared_task_proj:
                    self.current_shared_task_proj = nn.Linear(dim * 2, dim)
            if (
                not self.future_video_share_future_depth_query
                or self.use_shared_future_task_proj
            ):
                self.future_video_align_embs = nn.Parameter(
                    torch.randn(video["num_backbone_tokens"], dim)
                )
            if self.use_shared_future_task_proj:
                self.future_shared_task_proj = nn.Linear(dim * 2, dim)
        if self.use_future_video_cls:
            self.future_video_cls_align_emb = nn.Embedding(1, dim)

    def _align_query_tokens(self):
        """[n, D] query tokens appended after the language tokens, in upstream ``prefix_query_segments``
        order: current (depth, mixed with current video), future-video cls, future video, future depth
        (mixed with future video when the future query is shared)."""

        def pool(embs):  # group consecutive rows into num_task_tokens tokens
            return embs.view(self.num_task_tokens, -1, embs.shape[-1]).mean(dim=1)

        current = pool(self.depth_align_embs)
        if self.use_current_shared_task_proj:
            current = self.current_shared_task_proj(
                torch.cat([current, pool(self.current_video_align_embs)], -1)
            )
        tokens = [current]
        if self.use_future_video_cls:
            tokens.append(self.future_video_cls_align_emb.weight)
        if (
            self.use_future_video_patch
            and not self.future_video_share_future_depth_query
        ):
            tokens.append(pool(self.future_video_align_embs))
        if self.use_future_depth:
            future = pool(self.future_depth_align_embs)
            if self.use_shared_future_task_proj:
                future = self.future_shared_task_proj(
                    torch.cat([future, pool(self.future_video_align_embs)], -1)
                )
            tokens.append(future)
        return torch.cat(tokens, dim=0)

    def _block_suffix_to_queries_(self, att_2d_masks, suffix_row_start, prefix_len):
        """Optional upstream masks hiding the future query tokens from the suffix (off in released recipes)."""
        n = self.num_task_tokens
        if (
            self.block_future_depth_to_action
        ):  # upstream block_suffix_to_fv_: the last n prefix tokens
            att_2d_masks[:, suffix_row_start:, prefix_len - n : prefix_len] = False
        if self.block_suffix_to_future_video:
            own = int(self.use_future_video_cls) + (
                n
                if self.use_future_video_patch
                and not self.future_video_share_future_depth_query
                else 0
            )
            end = prefix_len - (n if self.use_future_depth else 0)
            att_2d_masks[:, suffix_row_start:, end - own : end] = False
        return att_2d_masks
