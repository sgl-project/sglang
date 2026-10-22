# Copyright 2026 SGLang Team
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
"""SynthID sampling and fused state coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.sampling.watermarking.synthid_target_graph import MixedTargetGraphs

from sglang.srt.watermarking.synthid.state import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
    SynthIDState,
    SynthIDStatePool,
    compute_context_hash_torch,
    compute_g_values_torch,
    compute_synthid_key_iv,
    get_synthid_config,
    load_synthid_keys,
    watermark_topk_scores_torch,
)


@dataclass(frozen=True)
class SynthIDSparseOutput:
    scores: torch.Tensor
    token_ids: torch.Tensor

    def argmax_token_ids(self, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        selected = self.scores.argmax(dim=-1, keepdim=True)
        return self.token_ids.gather(1, selected).view(-1).to(dtype)


@dataclass
class _SynthIDSampleBuffer:
    scores: torch.Tensor
    token_ids: torch.Tensor
    exponential_noise: torch.Tensor
    event: torch.cuda.Event
    in_flight: bool = False


class SynthIDWatermarkProcessor:
    """Apply SynthID through Triton, with PyTorch retained as a test oracle."""

    def __init__(
        self,
        config: SynthIDConfig,
        device: str | torch.device,
        num_slots: int,
    ):
        self.config = config
        self.keys = torch.tensor(config.keys, dtype=torch.int64, device=device)
        if torch.device(device).type == "cuda":
            from sglang.kernels.ops.sampling.synthid.triton import (
                watermark_topk_scores_triton,
            )

            self._watermark_topk_scores = watermark_topk_scores_triton
        else:
            self._watermark_topk_scores = watermark_topk_scores_torch
        self.state_pool = SynthIDStatePool(config, num_slots, device)
        self._spec_commit_event = (
            torch.cuda.Event() if torch.device(device).type == "cuda" else None
        )
        self._sampling_stream = (
            torch.cuda.Stream(device=device)
            if torch.device(device).type == "cuda"
            else None
        )
        self._has_spec_commit = False
        self._mixed_target_graphs: dict[int, MixedTargetGraphs] = {}
        self._mixed_draft_graphs: MixedTargetGraphs | None = None
        self._sample_buffer_cursor = 0
        self._sample_buffers = self._allocate_sample_buffers(
            num_slots,
            device,
        )

    @classmethod
    def from_runtime(
        cls, device: str | torch.device, num_slots: Optional[int]
    ) -> Optional[SynthIDWatermarkProcessor]:
        config = get_synthid_config()
        if config is None:
            return None
        if num_slots is None:
            raise RuntimeError(
                "SynthID watermarking requires an initialized request-token pool."
            )
        return cls(config, device, num_slots)

    def apply(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        *,
        verify_tokens: Optional[torch.Tensor] = None,
        scatter_to_logits: bool = True,
        filter_probs: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthIDSparseOutput:
        """Apply temperature, the candidate cap, optional request filters, and SynthID."""
        if logits.ndim != 2:
            raise ValueError(f"SynthID logits must be 2D, got {tuple(logits.shape)}.")

        batch_size = len(state)
        if verify_tokens is None:
            if logits.shape[0] != batch_size:
                raise ValueError(
                    f"SynthID state has {batch_size} rows for {logits.shape[0]} logits."
                )
            rows_per_request = 1
            context_hashes, apply_mask = self.state_pool.current_hashes_and_mask(state)
        else:
            rows_per_request = verify_tokens.shape[1]
            if logits.shape[0] != batch_size * rows_per_request:
                raise ValueError(
                    "SynthID speculative logits row count mismatch: "
                    f"{logits.shape[0]} != {batch_size} * {rows_per_request}."
                )
            context_hashes, apply_mask = self._project_speculative(state, verify_tokens)

        return self._apply_with_hashes(
            logits,
            temperatures,
            context_hashes,
            apply_mask,
            batch_size=batch_size,
            rows_per_request=rows_per_request,
            scatter_to_logits=scatter_to_logits,
            keys=self.keys,
            filter_probs=filter_probs,
        )

    def apply_draft(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        prior_draft_tokens: Optional[torch.Tensor] = None,
        *,
        scatter_to_logits: bool = True,
        filter_probs: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthIDSparseOutput:
        """Watermark draft logits without committing provisional state."""
        if logits.shape[0] != len(state):
            raise ValueError(
                f"Draft logits have {logits.shape[0]} rows for batch size {len(state)}."
            )
        context_hashes, apply_mask = self._draft_hashes_and_mask(
            state,
            prior_draft_tokens,
        )
        return self._apply_with_hashes(
            logits,
            temperatures,
            context_hashes,
            apply_mask,
            batch_size=len(state),
            rows_per_request=1,
            scatter_to_logits=scatter_to_logits,
            keys=self.keys,
            filter_probs=filter_probs,
        )

    def apply_draft_probs_and_sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        prior_draft_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return sparse draft probabilities and one fused proposal sample."""
        if not self._supports_fused_state(logits):
            raise RuntimeError("Fused draft sampling requires the Triton backend.")
        if logits.shape[0] != len(state):
            raise ValueError(
                f"Draft logits have {logits.shape[0]} rows for batch size {len(state)}."
            )
        context_hashes, apply_mask = self._draft_hashes_and_mask(
            state,
            prior_draft_tokens,
        )
        buffer = None
        if torch.cuda.is_current_stream_capturing():
            topk_scores, topk_indices = self._topk_candidates(logits)
            noise = torch.empty_like(topk_scores).exponential_(1.0)
        else:
            (
                topk_scores,
                topk_indices,
                noise,
                buffer,
            ) = self._buffered_topk_candidates(
                logits,
                with_exponential_noise=True,
            )
        from sglang.kernels.ops.sampling.synthid.triton import (
            watermark_topk_probs_and_sample_triton,
        )

        sorted_tokens = (
            torch.empty_like(topk_indices) if topk_scores.shape[-1] <= 1024 else None
        )
        probs, sampled_token_ids, sampled_probs = (
            watermark_topk_probs_and_sample_triton(
                topk_scores,
                topk_indices,
                context_hashes.contiguous(),
                self.keys,
                apply_mask.contiguous(),
                noise,
                temperatures=temperatures,
                sorted_tokens=sorted_tokens,
            )
        )
        if sorted_tokens is None:
            topk_indices, order = topk_indices.sort(dim=-1)
            probs = probs.gather(1, order)
        else:
            topk_indices = sorted_tokens
        if buffer is not None:
            self._release_sample_buffer(buffer, logits.device)
        return probs, topk_indices, sampled_probs, sampled_token_ids

    def _draft_hashes_and_mask(
        self,
        state: SynthIDState,
        prior_draft_tokens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(state)
        if prior_draft_tokens is None or prior_draft_tokens.shape[1] == 0:
            if self._supports_fused_state(state.slot_indices):
                # Projection column zero ignores the root and hashes current state.
                root = torch.zeros_like(state.slot_indices[:, None])
                return self._project_speculative(state, root)
            self.wait_for_speculative_commit()
            return self.state_pool.current_hashes_and_mask(state)
        if prior_draft_tokens.ndim != 2 or prior_draft_tokens.shape[0] != batch_size:
            raise ValueError("prior_draft_tokens must have shape [batch, draft_steps].")
        placeholder_root = torch.zeros_like(prior_draft_tokens[:, :1])
        projected_hashes, projected_mask = self._project_speculative(
            state,
            torch.cat((placeholder_root, prior_draft_tokens), dim=1),
        )
        projection_width = prior_draft_tokens.shape[1] + 1
        return (
            projected_hashes.view(batch_size, projection_width)[:, -1],
            projected_mask.view(batch_size, projection_width)[:, -1],
        )

    def apply_sparse_probs(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        *,
        verify_tokens: Optional[torch.Tensor] = None,
        sort_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sparse watermarked probabilities without dense softmax."""
        if not self._supports_fused_state(logits):
            raise RuntimeError("Fused probability output requires the Triton backend.")
        batch_size = len(state)
        if verify_tokens is None:
            if logits.shape[0] != batch_size:
                raise ValueError(
                    f"SynthID state has {batch_size} rows for {logits.shape[0]} logits."
                )
            rows_per_request = 1
            context_hashes, apply_mask = self.state_pool.current_hashes_and_mask(state)
        else:
            rows_per_request = verify_tokens.shape[1]
            if logits.shape[0] != batch_size * rows_per_request:
                raise ValueError(
                    "SynthID speculative logits row count mismatch: "
                    f"{logits.shape[0]} != {batch_size} * {rows_per_request}."
                )
            context_hashes, apply_mask = self._project_speculative(state, verify_tokens)

        topk_scores, topk_indices = self._topk_candidates(logits)
        if topk_scores.shape[-1] > 1024:
            topk_scores.view(batch_size, rows_per_request, -1).div_(
                temperatures.view(batch_size, 1, 1)
            )

        from sglang.kernels.ops.sampling.synthid.triton import (
            watermark_topk_probs_and_sample_triton,
        )

        sorted_tokens = (
            torch.empty_like(topk_indices)
            if sort_tokens and topk_scores.shape[-1] <= 1024
            else None
        )
        probs, _, _ = watermark_topk_probs_and_sample_triton(
            topk_scores.contiguous(),
            topk_indices.contiguous(),
            context_hashes.contiguous(),
            self.keys,
            apply_mask.contiguous(),
            temperatures=temperatures if topk_scores.shape[-1] <= 1024 else None,
            rows_per_request=rows_per_request,
            sorted_tokens=sorted_tokens,
        )
        if sorted_tokens is not None:
            return probs, sorted_tokens
        if sort_tokens:
            topk_indices, order = topk_indices.sort(dim=-1)
            probs = probs.gather(1, order)
        return probs, topk_indices

    def capture_mixed_target_graphs(
        self, *, capture_bs: list[int], rows_per_request: int, vocab_size: int
    ) -> None:
        from sglang.srt.sampling.watermarking.synthid_target_graph import (
            MixedTargetGraphs,
        )

        if rows_per_request not in self._mixed_target_graphs:
            self._mixed_target_graphs[rows_per_request] = MixedTargetGraphs(
                processor=self,
                capture_bs=capture_bs,
                rows_per_request=rows_per_request,
                vocab_size=vocab_size,
            )
        if self._mixed_draft_graphs is None:
            self._mixed_draft_graphs = MixedTargetGraphs(
                processor=self,
                capture_bs=capture_bs,
                rows_per_request=1,
                vocab_size=vocab_size,
                sample_draft=True,
            )

    def try_sample_mixed_draft(
        self,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        prior_draft_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        family = self._mixed_draft_graphs
        if family is None or torch.cuda.is_current_stream_capturing():
            return None
        if state.request_enabled is None:
            raise RuntimeError("Mixed draft replay requires live request flags.")
        hashes, apply_mask = self._draft_hashes_and_mask(state, prior_draft_tokens)
        return family.sample(
            logits=logits,
            temperatures=temperatures,
            hashes=hashes,
            apply_mask=apply_mask,
            enabled=state.request_enabled,
        )

    def try_apply_mixed_target(
        self,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        verify_tokens: torch.Tensor,
    ) -> bool:
        family = self._mixed_target_graphs.get(verify_tokens.shape[1])
        if family is None:
            return False
        if state.request_enabled is None:
            raise RuntimeError("Mixed target replay requires live request flags.")
        hashes, apply_mask = self._project_speculative(state, verify_tokens)
        return family.apply(
            logits=logits,
            temperatures=temperatures,
            hashes=hashes,
            apply_mask=apply_mask,
            enabled=state.request_enabled,
        )

    def _project_speculative(
        self,
        state: SynthIDState,
        verify_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.wait_for_speculative_commit()
        use_triton = (
            self._supports_fused_state(verify_tokens) and verify_tokens.shape[1] <= 8
        )
        if not use_triton:
            return self.state_pool.project_linear_speculative(state, verify_tokens)

        self.state_pool.initialize_pending(state)

        from sglang.kernels.ops.sampling.synthid.triton import (
            project_synthid_speculative_state_triton,
        )

        return project_synthid_speculative_state_triton(
            state.slot_indices,
            verify_tokens.contiguous(),
            self.state_pool.context,
            self.state_pool.context_history,
            self.state_pool.history_cursor,
            self.state_pool.num_calls,
            hash_iv=self.config.hash_iv,
            skip_first_ngram_calls=self.config.skip_first_ngram_calls,
        )

    def _apply_with_hashes(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        context_hashes: torch.Tensor,
        apply_mask: torch.Tensor,
        *,
        batch_size: int,
        rows_per_request: int,
        scatter_to_logits: bool,
        keys: torch.Tensor,
        filter_probs: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthIDSparseOutput:
        topk_scores, topk_indices = self._topk_candidates(logits)
        top_k = topk_scores.shape[-1]
        topk_scores_3d = topk_scores.view(batch_size, rows_per_request, top_k)
        topk_scores_3d.div_(temperatures.view(batch_size, 1, 1))

        filtered_probs = None
        if filter_probs is not None:
            filtered_probs = filter_probs(torch.softmax(topk_scores, dim=-1))
            topk_scores = filtered_probs.log()

        topk_scores = topk_scores.contiguous()
        topk_indices = topk_indices.contiguous()
        context_hashes = context_hashes.contiguous()
        apply_mask = apply_mask.contiguous()

        updated_scores = self._watermark_topk_scores(
            topk_scores,
            topk_indices,
            context_hashes,
            keys,
            apply_mask,
        )
        if filtered_probs is not None:
            updated_scores = updated_scores.masked_fill(
                filtered_probs == 0, -float("inf")
            )

        if scatter_to_logits:
            logits.fill_(-float("inf"))
            logits.scatter_(1, topk_indices, updated_scores)
        return SynthIDSparseOutput(updated_scores, topk_indices)

    def _topk_candidates(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(
            logits,
            k=min(self.config.top_k, logits.shape[-1]),
            dim=-1,
            sorted=False,
        )

    def _allocate_sample_buffers(
        self,
        num_slots: int,
        device: str | torch.device,
    ) -> list[_SynthIDSampleBuffer]:
        if torch.device(device).type != "cuda" or self.config.top_k > 1024:
            return []
        return [
            _SynthIDSampleBuffer(
                scores=torch.empty(
                    (num_slots, self.config.top_k),
                    dtype=torch.float32,
                    device=device,
                ),
                token_ids=torch.empty(
                    (num_slots, self.config.top_k),
                    dtype=torch.int64,
                    device=device,
                ),
                exponential_noise=torch.empty(
                    (num_slots, self.config.top_k),
                    dtype=torch.float32,
                    device=device,
                ),
                event=torch.cuda.Event(),
            )
            for _ in range(4)
        ]

    def _buffered_topk_candidates(
        self,
        logits: torch.Tensor,
        *,
        with_exponential_noise: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[_SynthIDSampleBuffer],
    ]:
        if not self._sample_buffers:
            topk_scores, topk_indices = self._topk_candidates(logits)
            exponential_noise = (
                torch.empty_like(topk_scores).exponential_(1.0)
                if with_exponential_noise
                else None
            )
            return topk_scores, topk_indices, exponential_noise, None

        buffer = self._sample_buffers[self._sample_buffer_cursor]
        self._sample_buffer_cursor = (self._sample_buffer_cursor + 1) % len(
            self._sample_buffers
        )
        stream = torch.cuda.current_stream(device=logits.device)
        if buffer.in_flight:
            stream.wait_event(buffer.event)

        batch_size = logits.shape[0]
        top_k = min(self.config.top_k, logits.shape[-1])
        if top_k != self.config.top_k:
            topk_scores, topk_indices = torch.topk(
                logits,
                k=top_k,
                dim=-1,
                sorted=False,
            )
            exponential_noise = (
                torch.empty_like(topk_scores).exponential_(1.0)
                if with_exponential_noise
                else None
            )
            return topk_scores, topk_indices, exponential_noise, None

        scores = buffer.scores[:batch_size, :top_k]
        token_ids = buffer.token_ids[:batch_size, :top_k]
        torch.topk(
            logits,
            k=top_k,
            dim=-1,
            sorted=False,
            out=(scores, token_ids),
        )
        exponential_noise = None
        if with_exponential_noise:
            exponential_noise = buffer.exponential_noise[:batch_size, :top_k]
            exponential_noise.exponential_(1.0)
        return scores, token_ids, exponential_noise, buffer

    @staticmethod
    def _release_sample_buffer(
        buffer: _SynthIDSampleBuffer,
        device: torch.device,
    ) -> None:
        buffer.event.record(torch.cuda.current_stream(device=device))
        buffer.in_flight = True

    def _supports_fused_state(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.is_cuda
            and self.config.context_history_size <= 2048
            and self.config.context_width <= 16
        )

    def supports_fused_sampling(self, logits: torch.Tensor) -> bool:
        return (
            self._supports_fused_state(logits)
            and logits.dtype == torch.float32
            and self.config.top_k <= 1024
        )

    def supports_wide_sampling(self, logits: torch.Tensor) -> bool:
        return (
            self._supports_fused_state(logits)
            and logits.dtype == torch.float32
            and self.config.top_k > 1024
        )

    def sample_wide_stochastic_and_commit(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        exponential_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the wide Triton distribution on its owned CUDA stream."""
        if not self.supports_wide_sampling(logits):
            raise RuntimeError("The wide SynthID sampling path is not supported.")
        if logits.shape[0] != len(state):
            raise ValueError(
                f"SynthID state has {len(state)} rows for {logits.shape[0]} logits."
            )

        caller_stream = torch.cuda.current_stream(device=logits.device)
        self._sampling_stream.wait_stream(caller_stream)
        logits.record_stream(self._sampling_stream)
        temperatures.record_stream(self._sampling_stream)
        for field_name in SynthIDState._BATCH_FIELDS:
            getattr(state, field_name).record_stream(self._sampling_stream)
        if state.pending_init is not None:
            state.pending_init.record_stream(self._sampling_stream)
        if exponential_noise is not None:
            exponential_noise.record_stream(self._sampling_stream)

        with torch.cuda.stream(self._sampling_stream):
            self.state_pool.initialize_pending(state)
            topk_scores, topk_indices = self._topk_candidates(logits)
            topk_scores.div_(temperatures)
            topk_scores = topk_scores.contiguous()
            topk_indices = topk_indices.contiguous()
            context_hashes, apply_mask = self.state_pool.current_hashes_and_mask(state)
            updated_scores = self._watermark_topk_scores(
                topk_scores,
                topk_indices,
                context_hashes.contiguous(),
                self.keys,
                apply_mask.contiguous(),
            )
            probs = torch.softmax(updated_scores, dim=-1)
            noise = exponential_noise
            if noise is None:
                noise = torch.empty_like(probs).exponential_(1.0)
            elif noise.shape != probs.shape:
                raise ValueError(
                    "Wide SynthID exponential noise must match candidate scores."
                )
            selected = torch.argmax(probs / noise, dim=-1, keepdim=True)
            token_ids = topk_indices.gather(1, selected).view(-1).to(torch.int32)
            self.state_pool.commit(state, token_ids)

        caller_stream.wait_stream(self._sampling_stream)
        token_ids.record_stream(caller_stream)
        return token_ids

    def sample_stochastic_and_commit(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        state: SynthIDState,
        exponential_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.supports_fused_sampling(logits):
            raise RuntimeError("The fused SynthID stochastic path is not supported.")
        if logits.shape[0] != len(state):
            raise ValueError(
                f"SynthID state has {len(state)} rows for {logits.shape[0]} logits."
            )

        caller_stream = torch.cuda.current_stream(device=logits.device)
        self._sampling_stream.wait_stream(caller_stream)
        logits.record_stream(self._sampling_stream)
        temperatures.record_stream(self._sampling_stream)
        state.slot_indices.record_stream(self._sampling_stream)
        with torch.cuda.stream(self._sampling_stream):
            self.state_pool.initialize_pending(state)
            buffer = None
            if exponential_noise is None:
                (
                    topk_scores,
                    topk_indices,
                    exponential_noise,
                    buffer,
                ) = self._buffered_topk_candidates(
                    logits,
                    with_exponential_noise=True,
                )
            else:
                exponential_noise.record_stream(self._sampling_stream)
                topk_scores, topk_indices = self._topk_candidates(logits)

            from sglang.kernels.ops.sampling.synthid.triton import (
                synthid_sample_and_commit_triton,
            )

            token_ids = synthid_sample_and_commit_triton(
                topk_scores,
                topk_indices,
                temperatures,
                state.slot_indices,
                self.state_pool.context,
                self.state_pool.context_history,
                self.state_pool.history_cursor,
                self.state_pool.num_calls,
                self.keys,
                state.commit_mask,
                exponential_noise,
                hash_iv=self.config.hash_iv,
                skip_first_ngram_calls=self.config.skip_first_ngram_calls,
            )
            if buffer is not None:
                self._release_sample_buffer(buffer, token_ids.device)
        caller_stream.wait_stream(self._sampling_stream)
        token_ids.record_stream(caller_stream)
        return token_ids

    def _record_commit_event(self, token_ids: torch.Tensor) -> None:
        if self._spec_commit_event is None:
            return
        self._spec_commit_event.record(
            torch.cuda.current_stream(device=token_ids.device)
        )
        self._has_spec_commit = True

    def _supports_fused_commit(self, token_ids: torch.Tensor) -> bool:
        return (
            self._supports_fused_state(token_ids)
            and token_ids.ndim == 2
            and token_ids.shape[1] <= 8
        )

    def wait_for_speculative_commit(self) -> None:
        if (
            self._spec_commit_event is None
            or not self._has_spec_commit
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        torch.cuda.current_stream(device=self.state_pool.context.device).wait_event(
            self._spec_commit_event
        )

    def invalidate_on_cache_flush(self) -> None:
        self.wait_for_speculative_commit()
        self.state_pool.generations.fill_(-1)

    def commit(
        self,
        state: SynthIDState,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> None:
        if token_ids.ndim == 1 and self._supports_fused_state(token_ids):
            self.wait_for_speculative_commit()
            self.state_pool.initialize_pending(state)
            from sglang.kernels.ops.sampling.synthid.triton import (
                commit_synthid_token_triton,
            )

            commit_synthid_token_triton(
                token_ids,
                state.slot_indices,
                state.commit_mask,
                self.state_pool.context,
                self.state_pool.context_history,
                self.state_pool.history_cursor,
                self.state_pool.num_calls,
                hash_iv=self.config.hash_iv,
                skip_first_ngram_calls=self.config.skip_first_ngram_calls,
            )
            self._record_commit_event(token_ids)
            return
        if lengths is not None and self._supports_fused_commit(token_ids):
            self.state_pool.initialize_pending(state)
            from sglang.kernels.ops.sampling.synthid.triton import (
                commit_synthid_state_triton,
            )

            commit_synthid_state_triton(
                token_ids,
                torch.where(state.commit_mask, lengths, 0),
                state.slot_indices,
                self.state_pool.context,
                self.state_pool.context_history,
                self.state_pool.history_cursor,
                self.state_pool.num_calls,
                hash_iv=self.config.hash_iv,
                skip_first_ngram_calls=self.config.skip_first_ngram_calls,
            )
            self._record_commit_event(token_ids)
            return
        self.state_pool.commit(state, token_ids, lengths)


__all__ = [
    "REFERENCE_SYNTHID_KEYS",
    "SynthIDConfig",
    "SynthIDState",
    "SynthIDStatePool",
    "SynthIDSparseOutput",
    "SynthIDWatermarkProcessor",
    "compute_context_hash_torch",
    "compute_g_values_torch",
    "compute_synthid_key_iv",
    "get_synthid_config",
    "load_synthid_keys",
    "watermark_topk_scores_torch",
]
