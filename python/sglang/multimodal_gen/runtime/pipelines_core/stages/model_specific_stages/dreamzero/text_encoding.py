# SPDX-License-Identifier: Apache-2.0
"""DreamZero text encoding stage.

Text-encoding flow from ``batch.prompt`` to tokenizer and text encoder, with branch-specific prompt caching for CFG.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePool,
    DreamZeroCachePoolManager,
    DreamZeroRequestCache,
    apply_request_lifecycle_resets,
    resolve_request_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DreamZeroTextEncodingStage(TextEncodingStage):
    """Encode and cache DreamZero prompt branches for a session batch.

    The stage inherits ``TextEncodingStage.encode_text`` for tokenizer and
    encoder execution, then stores cond/uncond embeddings in the DreamZero
    session cache.
    """

    def __init__(
        self,
        text_encoder: torch.nn.Module | None = None,
        tokenizer: Any | None = None,
        cache_manager: DreamZeroCachePoolManager | None = None,
    ) -> None:
        super().__init__([text_encoder], [tokenizer])
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_inputs",
            getattr(batch, "dreamzero_inputs", None),
            lambda value: isinstance(value, dict),
        )
        result.add_check(
            "prompt",
            getattr(batch, "prompt", None),
            V.string_or_list_strings,
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
    def _fit_mask_len(mask: torch.Tensor, text_len: int) -> torch.Tensor:
        if mask.shape[1] == text_len:
            return mask
        if mask.shape[1] > text_len:
            return mask[:, :text_len]
        pad = mask.new_zeros(mask.shape[0], text_len - mask.shape[1])
        return torch.cat([mask, pad], dim=1)

    @staticmethod
    def _mask_prompt_padding(
        prompt_emb: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Zero out padded UMT5 positions to match DreamZero conditioning."""
        if attention_mask is None:
            return prompt_emb
        attention_mask = attention_mask.to(device=prompt_emb.device, dtype=torch.long)
        attention_mask = DreamZeroTextEncodingStage._fit_mask_len(
            attention_mask, prompt_emb.shape[1]
        )
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        positions = torch.arange(prompt_emb.shape[1], device=prompt_emb.device)
        valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        return prompt_emb.masked_fill(~valid.unsqueeze(-1), 0)

    @staticmethod
    def _batched_texts(
        value: Any,
        batch_size: int,
        field_name: str,
        *,
        default: str | None = None,
    ) -> list[str]:
        if value is None:
            if default is None:
                raise ValueError(f"DreamZero {field_name} is required")
            return [default] * batch_size
        if isinstance(value, str):
            return [value] * batch_size
        if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
            raise TypeError(
                f"DreamZero {field_name} must be a string or list of strings"
            )
        texts = list(value)
        if not all(isinstance(item, str) for item in texts):
            raise TypeError(
                f"DreamZero {field_name} must be a string or list of strings"
            )
        if len(texts) != batch_size:
            raise ValueError(
                f"DreamZero {field_name} batch size mismatch: "
                f"got {len(texts)}, expected {batch_size}"
            )
        return texts

    def _ensure_prompt_extra(self, batch: Req, batch_size: int) -> None:
        extra = batch.extra
        if extra.get("dreamzero_prompts") is None:
            extra["dreamzero_prompts"] = self._batched_texts(
                batch.prompt,
                batch_size,
                "prompt",
            )
        if extra.get("dreamzero_negative_prompts") is None:
            extra["dreamzero_negative_prompts"] = self._batched_texts(
                batch.negative_prompt,
                batch_size,
                "negative_prompt",
                default="",
            )

    @staticmethod
    def _set_prompt_metadata(
        batch: Req,
        prompt_embs: list[torch.Tensor],
    ) -> None:
        batch.dreamzero_cfg_branch_index = None
        batch.prompt_embeds = prompt_embs[0]
        batch.negative_prompt_embeds = prompt_embs[1] if len(prompt_embs) > 1 else None
        batch.dreamzero_prompt_embs = prompt_embs

    def _encode_prompt_texts(
        self,
        texts: list[str],
        server_args: ServerArgs,
        *,
        text_len: int,
    ) -> torch.Tensor:
        """Tokenize and encode prompt texts through the native text stage helper."""
        if self.text_encoders[0] is None:
            raise ValueError("DreamZero text encoder module is not loaded")
        if self.tokenizers[0] is None:
            raise ValueError("DreamZero tokenizer module is not loaded")
        (
            prompt_embeds_list,
            prompt_masks_list,
            _pooler_embeds_list,
            _prompt_embeds_masks_list,
            _prompt_seq_lens_list,
        ) = self.encode_text(
            texts,
            server_args,
            encoder_index=0,
            return_attention_mask=True,
            dtype=torch.bfloat16,
            max_length=text_len,
        )
        prompt = self._fit_text_len(prompt_embeds_list[0], text_len)
        attention_mask = prompt_masks_list[0] if prompt_masks_list else None
        return self._mask_prompt_padding(prompt, attention_mask).to(
            dtype=torch.bfloat16
        )

    @staticmethod
    def _local_attn_size(server_args: ServerArgs) -> int:
        arch = server_args.pipeline_config.dit_config.arch_config
        max_chunk_size = int(arch.max_chunk_size)
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
        batch_size = next(
            int(value.shape[0])
            for value in inputs.values()
            if torch.is_tensor(value) and value.ndim > 0
        )
        self._ensure_prompt_extra(batch, batch_size)
        request_cache = resolve_request_cache(
            batch,
            self.cache_manager,
            local_attn_size=self._local_attn_size(server_args),
            batch_size=batch_size,
        )
        return self._forward_cache_manager(
            batch,
            server_args,
            request_cache,
        )

    def _forward_cache_manager(
        self,
        batch: Req,
        server_args: ServerArgs,
        request_cache: DreamZeroRequestCache,
    ):
        """Resolve prompt cache hits and apply language/window lifecycle resets."""
        if self.cache_manager is None:
            raise RuntimeError("DreamZero text stage requires a cache manager")
        state: DreamZeroCachePool = self.cache_manager.pool
        slots = request_cache.slot_indices
        inputs: dict[str, Any] = batch.dreamzero_inputs
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
                # Prompt changes invalidate text plus downstream visual/KV state.
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
                    # New single-frame anchors reset visual/KV state but reuse text.
                    lifecycle_reset_mask[index] = True
                    lifecycle_preserve_text[index] = True
                    reset_reasons[index] = "single_frame"
                elif window_full:
                    # Full local-attention windows roll over without re-encoding text.
                    lifecycle_reset_mask[index] = True
                    lifecycle_preserve_text[index] = True
                    reset_reasons[index] = "local_attention_window_full"
        batch.dreamzero_lifecycle_reset_mask = lifecycle_reset_mask
        batch.dreamzero_lifecycle_reset_preserve_text = lifecycle_preserve_text
        batch.dreamzero_session_reset_reason = reset_reasons
        apply_request_lifecycle_resets(batch, self.cache_manager, request_cache)
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
        request_cache.prompt_reusable = prompt_reusable
        request_cache.neg_prompt_reusable = neg_prompt_reusable

        text_len = server_args.pipeline_config.dit_config.arch_config.text_len
        prompt_texts = batch.extra["dreamzero_prompts"]
        negative_prompt_texts = batch.extra["dreamzero_negative_prompts"]

        def get_branch_prompt(
            branch: int, *, texts: list[str], hashes, reusable
        ) -> torch.Tensor:
            """Load one CFG branch from cache or encode and scatter it."""
            if all(reusable):
                cached = state.gather_prompt(branch, slots)
                if cached is not None:
                    return cached
            prompt = self._encode_prompt_texts(
                texts,
                server_args,
                text_len=text_len,
            )
            state.scatter_prompt(branch, slots, prompt, hashes)
            return prompt

        prompt_embs = [
            get_branch_prompt(
                BRANCH_COND,
                texts=prompt_texts,
                hashes=request_cache.prompt_hashes,
                reusable=prompt_reusable,
            )
        ]
        if server_args.pipeline_config.should_use_guidance:
            prompt_embs.append(
                get_branch_prompt(
                    BRANCH_UNCOND,
                    texts=negative_prompt_texts,
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
        self._set_prompt_metadata(batch, prompt_embs)
        return batch
