# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePool,
    DreamZeroCachePoolManager,
    DreamZeroRequestCache,
    apply_request_lifecycle_resets,
    resolve_request_cache,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.utils import (
    infer_dreamzero_batch_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class DreamZeroTextEncodingStage(PipelineStage):
    """Custom DreamZero text stage using the local compatible encoder."""

    def __init__(
        self,
        text_encoder: torch.nn.Module | None = None,
        cache_manager: DreamZeroCachePoolManager | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.cache_manager = cache_manager

    @property
    def parallelism_type(self) -> StageParallelismType:
        if getattr(self.server_args, "enable_cfg_parallel", False):
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "text_encoder",
                target_dtype=PRECISION_TO_TYPE[
                    server_args.pipeline_config.text_encoder_precisions[0]
                ],
            )
        ]

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_inputs",
            getattr(batch, "dreamzero_inputs", None),
            lambda value: isinstance(value, dict),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_prompt_embs",
            getattr(batch, "dreamzero_prompt_embs", None),
            lambda value: isinstance(value, list) and len(value) > 0,
        )
        return result

    @staticmethod
    def _fit_text_len(tensor: torch.Tensor, text_len: int) -> torch.Tensor:
        if tensor.shape[1] == text_len:
            return tensor
        if tensor.shape[1] > text_len:
            return tensor[:, :text_len]
        pad = tensor.new_zeros(
            tensor.shape[0], text_len - tensor.shape[1], tensor.shape[2]
        )
        return torch.cat([tensor, pad], dim=1)

    @staticmethod
    def _set_prompt_metadata(
        batch: Req,
        prompt_embs: list[torch.Tensor],
        *,
        cfg_parallel: bool,
        cfg_rank: int | None,
    ) -> None:
        batch.dreamzero_cfg_branch_index = cfg_rank if cfg_parallel else None
        batch.prompt_embeds = prompt_embs[0]
        if cfg_parallel and cfg_rank == 1:
            batch.negative_prompt_embeds = prompt_embs[0]
        elif len(prompt_embs) > 1:
            batch.negative_prompt_embeds = prompt_embs[1]
        batch.dreamzero_prompt_embs = prompt_embs

    def _encode_prompt(
        self,
        encoder: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        text_len: int,
    ) -> torch.Tensor:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = input_ids.device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        prompt_output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_emb = prompt_output.last_hidden_state
        prompt_emb = self._fit_text_len(prompt_emb.clone(), text_len)
        attention_mask = attention_mask[:, : prompt_emb.shape[1]]
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        positions = torch.arange(prompt_emb.shape[1], device=prompt_emb.device)
        valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        prompt_emb = prompt_emb.masked_fill(~valid.unsqueeze(-1), 0)
        return prompt_emb.to(dtype=torch.bfloat16)

    @staticmethod
    def _local_attn_size(server_args: ServerArgs) -> int:
        arch = server_args.pipeline_config.dit_config.arch_config
        max_chunk_size = int(getattr(arch, "max_chunk_size", -1))
        if max_chunk_size == -1:
            return -1
        return max_chunk_size * int(arch.num_frame_per_block) + 1

    @staticmethod
    def _video_frame_count(inputs: dict[str, Any]) -> int | None:
        videos = inputs.get("images")
        if videos is None:
            videos = inputs.get("videos")
        if not torch.is_tensor(videos) or videos.ndim != 5:
            return None
        if videos.shape[-1] in (1, 3):
            return int(videos.shape[1])
        if videos.shape[2] in (1, 3) and videos.shape[1] != 3:
            return int(videos.shape[1])
        if videos.shape[1] in (1, 3):
            return int(videos.shape[2])
        return int(videos.shape[2])

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        inputs: dict[str, Any] = batch.dreamzero_inputs
        cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
        cfg_rank = 0
        if cfg_parallel:
            cfg_world_size = get_classifier_free_guidance_world_size()
            if cfg_world_size != 2:
                raise ValueError(
                    "DreamZero CFG parallel requires cfg_parallel_degree=2, "
                    f"got {cfg_world_size}"
                )
            cfg_rank = get_classifier_free_guidance_rank()
        request_cache = resolve_request_cache(
            batch,
            self.cache_manager,
            local_attn_size=self._local_attn_size(server_args),
            batch_size=infer_dreamzero_batch_size(
                inputs,
                error_message="DreamZero text stage cannot infer batch size",
            ),
        )
        return self._forward_cache_manager(
            batch,
            server_args,
            request_cache,
            cfg_parallel=cfg_parallel,
            cfg_rank=cfg_rank,
        )

    def _forward_cache_manager(
        self,
        batch: Req,
        server_args: ServerArgs,
        request_cache: DreamZeroRequestCache,
        *,
        cfg_parallel: bool = False,
        cfg_rank: int = 0,
    ):
        if self.cache_manager is None:
            raise RuntimeError("DreamZero text stage requires a cache manager")
        state: DreamZeroCachePool = self.cache_manager.pool
        slots = request_cache.slot_indices
        inputs: dict[str, Any] = batch.dreamzero_inputs
        input_ids = inputs.get("text")
        neg_ids = inputs.get("text_negative")
        if input_ids is not None and not torch.is_tensor(input_ids):
            raise ValueError("DreamZero batched text input must be a tensor")
        if neg_ids is not None and not torch.is_tensor(neg_ids):
            raise ValueError("DreamZero batched negative text input must be a tensor")
        reset_reasons: list[str | None] = [None] * request_cache.batch_size
        lifecycle_reset_mask: list[bool] = [False] * request_cache.batch_size
        lifecycle_preserve_text: list[bool] = [True] * request_cache.batch_size
        frame_count = self._video_frame_count(inputs)
        for index, slot in enumerate(slots):
            if request_cache.reset_mask[index]:
                continue
            cond_hash = request_cache.prompt_hashes[index]
            neg_hash = request_cache.neg_prompt_hashes[index]
            language_changed = (
                state.prompt_hashes[BRANCH_COND][slot] is not None
                and cond_hash is not None
                and state.prompt_hashes[BRANCH_COND][slot] != cond_hash
            ) or (
                state.prompt_hashes[BRANCH_UNCOND][slot] is not None
                and neg_hash is not None
                and state.prompt_hashes[BRANCH_UNCOND][slot] != neg_hash
            )
            if language_changed:
                lifecycle_reset_mask[index] = True
                lifecycle_preserve_text[index] = False
                reset_reasons[index] = "language_changed"
                continue
            first_observation = state.current_start_frames[slot] == 0
            if not first_observation:
                window_full = (
                    state.local_attn_size != -1
                    and state.current_start_frames[slot] >= state.local_attn_size
                )
                if frame_count == 1:
                    lifecycle_reset_mask[index] = True
                    lifecycle_preserve_text[index] = True
                    reset_reasons[index] = "single_frame"
                elif window_full:
                    lifecycle_reset_mask[index] = True
                    lifecycle_preserve_text[index] = True
                    reset_reasons[index] = "local_attention_window_full"
        batch.dreamzero_lifecycle_reset_mask = lifecycle_reset_mask
        batch.dreamzero_lifecycle_reset_preserve_text = lifecycle_preserve_text
        batch.dreamzero_session_reset_reason = reset_reasons
        apply_request_lifecycle_resets(batch, self.cache_manager, request_cache)
        reset_reasons = batch.dreamzero_session_reset_reason
        prompt_reusable = [
            bool(reusable and not (reset and not preserve_text))
            for reusable, reset, preserve_text in zip(
                request_cache.prompt_reusable,
                lifecycle_reset_mask,
                lifecycle_preserve_text,
                strict=True,
            )
        ]
        neg_prompt_reusable = [
            bool(reusable and not (reset and not preserve_text))
            for reusable, reset, preserve_text in zip(
                request_cache.neg_prompt_reusable,
                lifecycle_reset_mask,
                lifecycle_preserve_text,
                strict=True,
            )
        ]
        # Text stage owns the lifecycle-adjusted reusable view. Later stages
        # resolve a fresh request cache and should not depend on these fields.
        if cfg_parallel:
            # Negative prompts are static for DreamZero eval. Use positive
            # prompt metadata to keep CFG ranks aligned on encode-vs-cache.
            prompt_reusable = [
                bool(
                    cache_hit
                    and prompt_hash is not None
                    and state.prompt_hashes[BRANCH_COND][slot] == prompt_hash
                )
                for slot, cache_hit, prompt_hash in zip(
                    slots,
                    request_cache.cache_hit,
                    request_cache.prompt_hashes,
                    strict=True,
                )
            ]
            neg_prompt_reusable = prompt_reusable
        request_cache.prompt_reusable = prompt_reusable
        request_cache.neg_prompt_reusable = neg_prompt_reusable

        attention_mask = inputs.get("text_attention_mask")
        text_len = server_args.pipeline_config.dit_config.arch_config.text_len
        prompt_embs: list[torch.Tensor] = []

        def get_branch_prompt(
            branch: int, *, ids, mask, hashes, reusable
        ) -> torch.Tensor:
            if all(reusable):
                prompt_pool = state.cached_prompt_embs[branch]
                if (
                    prompt_pool is not None
                    and (not slots or prompt_pool.shape[0] > max(slots))
                    and all(state.prompt_valid[branch][slot] for slot in slots)
                ):
                    cached = state.gather_prompt(branch, slots)
                    if cached is not None:
                        return cached
                if cfg_parallel:
                    raise RuntimeError(
                        "DreamZero CFG prompt cache metadata is reusable but "
                        f"branch {branch} embeddings are missing"
                    )
            if ids is None:
                raise ValueError("DreamZero text stage requires tokenized text")
            with self.use_declared_component(
                component_name="text_encoder", module=self.text_encoder
            ) as encoder:
                if encoder is None:
                    raise ValueError("DreamZero text encoder module is not loaded")
                self.text_encoder = encoder
                prompt = self._encode_prompt(
                    encoder,
                    ids,
                    mask,
                    text_len=text_len,
                )
            state.scatter_prompt(branch, slots, prompt, hashes)
            return prompt

        if cfg_parallel:
            if neg_ids is None:
                raise ValueError(
                    "DreamZero CFG parallel requires tokenized text_negative"
                )
            if cfg_rank == 0:
                prompt_embs = [
                    get_branch_prompt(
                        BRANCH_COND,
                        ids=input_ids,
                        mask=attention_mask,
                        hashes=request_cache.prompt_hashes,
                        reusable=prompt_reusable,
                    )
                ]
            else:
                prompt_embs = [
                    get_branch_prompt(
                        BRANCH_UNCOND,
                        ids=neg_ids,
                        mask=inputs.get("text_attention_mask_negative"),
                        hashes=request_cache.neg_prompt_hashes,
                        reusable=neg_prompt_reusable,
                    )
                ]
        else:
            prompt_embs = [
                get_branch_prompt(
                    BRANCH_COND,
                    ids=input_ids,
                    mask=attention_mask,
                    hashes=request_cache.prompt_hashes,
                    reusable=prompt_reusable,
                )
            ]
            if server_args.pipeline_config.should_use_guidance and (
                neg_ids is not None
            ):
                prompt_embs.append(
                    get_branch_prompt(
                        BRANCH_UNCOND,
                        ids=neg_ids,
                        mask=inputs.get("text_attention_mask_negative"),
                        hashes=request_cache.neg_prompt_hashes,
                        reusable=neg_prompt_reusable,
                    )
                )
        for slot, prompt_hash, neg_prompt_hash in zip(
            slots,
            request_cache.prompt_hashes,
            request_cache.neg_prompt_hashes,
            strict=True,
        ):
            state.prompt_hashes[BRANCH_COND][slot] = prompt_hash
            state.prompt_hashes[BRANCH_UNCOND][slot] = neg_prompt_hash
        self._set_prompt_metadata(
            batch,
            prompt_embs,
            cfg_parallel=cfg_parallel,
            cfg_rank=cfg_rank,
        )
        return batch
