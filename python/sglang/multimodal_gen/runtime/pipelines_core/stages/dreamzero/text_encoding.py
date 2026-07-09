# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.dreamzero_session_store import (
    SessionStore,
    get_request_session_state,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class DreamZeroTextEncodingStage(PipelineStage):
    """Custom DreamZero text stage using the Phase 3 Groot fallback encoder."""

    def __init__(
        self,
        text_encoder: torch.nn.Module | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.session_store = session_store

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
        pad = tensor.new_zeros(tensor.shape[0], text_len - tensor.shape[1], tensor.shape[2])
        return torch.cat([tensor, pad], dim=1)

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
        prompt_emb = encoder(input_ids, attention_mask)
        prompt_emb = self._fit_text_len(prompt_emb.clone(), text_len)
        attention_mask = attention_mask[:, : prompt_emb.shape[1]]
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        for batch_index, seq_len in enumerate(seq_lens):
            prompt_emb[batch_index, int(seq_len) :] = 0
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
        return int(videos.shape[1] if videos.shape[-1] in (1, 3) else videos.shape[2])

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
            # ParallelExecutor broadcasts rank 0's batch before CFG stages.
            # Resolve the worker-local session once, here at the first
            # session-aware CFG stage, so rank 1 never adopts rank 0's caches.
            if hasattr(batch, "dreamzero_session_state"):
                delattr(batch, "dreamzero_session_state")
        state = get_request_session_state(
            batch,
            self.session_store,
            local_attn_size=self._local_attn_size(server_args),
        )
        input_ids = inputs.get("text")
        neg_ids = inputs.get("text_negative")
        language_changed = (
            (
                torch.is_tensor(input_ids)
                and state.language is not None
                and not torch.equal(state.language, input_ids)
            )
            or (
                torch.is_tensor(neg_ids)
                and state.negative_language is not None
                and not torch.equal(state.negative_language, neg_ids)
            )
        )
        first_observation = state.language is None
        reset_reason = None
        if language_changed:
            state.reset_stream(preserve_text=False)
            reset_reason = "language_changed"
        elif not first_observation:
            frame_count = self._video_frame_count(inputs)
            window_full = (
                state.local_attn_size != -1
                and state.current_start_frame >= state.local_attn_size
            )
            if frame_count == 1:
                state.reset_stream(preserve_text=True)
                reset_reason = "single_frame"
            elif window_full:
                state.reset_stream(preserve_text=True)
                reset_reason = "local_attention_window_full"

        if torch.is_tensor(input_ids) and (
            state.language is None or language_changed
        ):
            state.language = input_ids.detach().clone()
        if torch.is_tensor(neg_ids) and (
            state.negative_language is None or language_changed
        ):
            state.negative_language = neg_ids.detach().clone()

        precomputed = inputs.get("prompt_embs")
        if precomputed is None:
            precomputed = batch.extra.get("dreamzero_prompt_embs")
        if state.cached_prompt_embs is not None:
            prompt_embs = state.cached_prompt_embs
        elif precomputed is not None:
            prompt_embs = list(precomputed)
            if cfg_parallel:
                if len(prompt_embs) < 2:
                    raise ValueError(
                        "DreamZero CFG parallel requires positive and negative "
                        "precomputed prompt embeddings"
                    )
                prompt_embs = [prompt_embs[cfg_rank]]
        else:
            attention_mask = inputs.get("text_attention_mask")
            if input_ids is None:
                raise ValueError(
                    "DreamZero text stage requires tokenized 'text' or precomputed "
                    "'prompt_embs'"
                )
            text_len = server_args.pipeline_config.dit_config.arch_config.text_len
            with self.use_declared_component(
                component_name="text_encoder", module=self.text_encoder
            ) as encoder:
                if encoder is None:
                    raise ValueError("DreamZero text encoder module is not loaded")
                self.text_encoder = encoder
                if cfg_parallel:
                    if neg_ids is None:
                        raise ValueError(
                            "DreamZero CFG parallel requires tokenized "
                            "'text_negative'"
                        )
                    branch_ids = input_ids if cfg_rank == 0 else neg_ids
                    branch_mask = (
                        attention_mask
                        if cfg_rank == 0
                        else inputs.get("text_attention_mask_negative")
                    )
                    prompt_embs = [
                        self._encode_prompt(
                            encoder,
                            branch_ids,
                            branch_mask,
                            text_len=text_len,
                        )
                    ]
                else:
                    prompt_embs = [
                        self._encode_prompt(
                            encoder,
                            input_ids,
                            attention_mask,
                            text_len=text_len,
                        )
                    ]
                    if (
                        server_args.pipeline_config.should_use_guidance
                        and neg_ids is not None
                    ):
                        prompt_embs.append(
                            self._encode_prompt(
                                encoder,
                                neg_ids,
                                inputs.get("text_attention_mask_negative"),
                                text_len=text_len,
                            )
                        )

        state.cached_prompt_embs = prompt_embs
        batch.dreamzero_session_reset_reason = reset_reason
        batch.dreamzero_prompt_embs = prompt_embs
        batch.dreamzero_cfg_branch_index = cfg_rank if cfg_parallel else None
        batch.prompt_embeds = prompt_embs[0]
        if cfg_parallel and cfg_rank == 1:
            batch.negative_prompt_embeds = prompt_embs[0]
        elif len(prompt_embs) > 1:
            batch.negative_prompt_embeds = prompt_embs[1]
        return batch
