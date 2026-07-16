from enum import IntEnum
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange
from torch.nn.parameter import Parameter

from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from sglang.srt.models.inkling_common.kernels.sconv import (
    HIS_ONES,
    HIS_PREFIX,
    HIS_SEQ_MINUS_EXT,
    HIS_ZEROS,
    SconvDecodeMetadata,
    SconvExtendMetadata,
    causal_conv1d,
    fused_causal_conv1d_update_decode,
    fused_decode_sconv_metadata,
    fused_extend_sconv_metadata,
    precompute_helion_extend_metadata,
    save_intermediate_conv_windows,
    update_sconv_cache,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.utils import is_cuda, set_weight_attrs


class SconvType(IntEnum):
    K_FULL = 0
    V_FULL = 1
    K_LOCAL = 2
    V_LOCAL = 3
    ATTN = 4
    MLP = 5


# Module-level cache for sconv metadata (shared across layers in the same forward pass)
_metadata_cache: dict = {}


class ShortConvolution(nn.Module):
    """Short convolution layer for efficient causal convolution operations.

    This class implements a depthwise separable 1D convolution with causal padding,
    designed for efficient sequence processing using Triton.

    Args:
        hidden_size (int): Number of input/output channels (must be equal for depthwise conv)
        kernel_size (int): Size of the convolution kernel
        activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to 'silu'.
        use_residual (bool, optional): Whether to add residual connection (y = conv(x) + x). Defaults to False.
        device (Optional[torch.device], optional): Device to place the layer on. Defaults to None.
        dtype (Optional[torch.dtype], optional): Data type for layer parameters. Defaults to None.
        param_config (ParameterConfig | None, optional): Parameter configuration for mixed precision. Defaults to None.
        **kwargs: Additional keyword arguments (deprecated 'use_fast_conv1d' supported for compatibility)

    Note:
        - Uses depthwise convolution (groups=hidden_size) for efficiency
        - Applies causal padding (kernel_size-1) to ensure no future information leakage
        - Uses Triton for efficient GPU execution
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        sconv_type: SconvType,
        activation: str | None = None,
        use_residual: bool = True,
        layer_id: int | None = None,
        tp_rank: int | None = None,
    ):
        super().__init__()

        self.kernel_size = (
            (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        )
        self.use_residual = use_residual
        self.layer_id = layer_id
        self.sconv_type: SconvType | None = sconv_type

        if tp_rank is None:
            tp_rank = get_parallel().attn_tp_rank
        self.tp_rank = tp_rank

        # Initialize weight parameter (will be initialized in reset_parameters)
        self.weight = nn.Parameter(
            torch.empty(
                hidden_size,
                1,
                kernel_size,
            ),
            requires_grad=False,
        )
        # Register the weight_loader method for this parameter
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

        self.activation = None
        if activation is not None:
            assert activation in [
                "silu",
                "swish",
            ], f"Activation `{activation}` not supported yet."
            self.activation = activation

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """
        For TP-sharded sconv layers (e.g., k_sconv and v_sconv in attention blocks),
        the parameter size is the sharded size while the checkpoint contains the
        full unsharded weight. This method narrows the loaded weight to the correct
        shard based on the current TP rank.

        Non-TP-sharded sconv layers (e.g., attn_sconv and mlp_sconv) already have
        parameter shapes matching the checkpoint, so the narrowing branch is skipped.
        """
        param_data = param.data

        if loaded_weight.shape[0] != param_data.shape[0]:
            # This weight is TP-sharded, need to narrow to the correct shard
            shard_size = param_data.shape[0]
            start_idx = self.tp_rank * shard_size

            # Narrow the loaded weight to the correct shard on dim 0
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape, (
            f"Shape mismatch after narrowing: param {param_data.shape} vs "
            f"loaded {loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)

    def _owns_extend_metadata(self, forward_batch: ForwardBatch) -> bool:
        # layer 0 computes the shared _metadata_cache for all layers within one
        # forward. Under de-tied draft_extend_v2 each STEP is its own forward
        # against its own pool, and only step 0's model carries layer_id == 0 —
        # steps 1..N-1 would silently reuse a previous forward's cached (freed
        # or wrong-pool) tensors, so every step must own its metadata.
        return self.layer_id == 0 or forward_batch.forward_mode.is_draft_extend_v2()

    def _prepare_extend_common_metadata(
        self, forward_batch: ForwardBatch, cache_indices: torch.Tensor
    ):
        """Compute ALL extend sconv metadata (query_start_loc, has_initial_state,
        and the SconvExtendMetadata) in one fused launch and stash it in
        _metadata_cache; _prepare_extend_sconv_metadata is then a cache read.
        Falls back to the original unfused op sequence off-CUDA or past the
        fused kernel's batch bound."""
        if self._owns_extend_metadata(forward_batch):
            B = forward_batch.batch_size
            is_verify = forward_batch.forward_mode.is_target_verify()
            if is_verify:
                # target_verify does not populate extend_seq_lens/extend_prefix_lens;
                # the lens are a constant draft_token_num per request.
                draft_token_num = forward_batch.spec_info.draft_token_num
                num_tokens = B * draft_token_num
                fused = fused_extend_sconv_metadata(
                    B=B,
                    T=num_tokens,
                    cache_indices=cache_indices,
                    his_mode=HIS_ONES,
                    draft_token_num=draft_token_num,
                )
            else:
                num_tokens = forward_batch.extend_num_tokens
                spec_info = forward_batch.spec_info
                if (
                    isinstance(spec_info, EagleDraftExtendInput)
                    and spec_info.num_front_tokens > 0
                ):
                    # Boundary-KV fix: run conv fresh so warm-up rows rebuild
                    # the window.
                    his_mode, his_src = HIS_ZEROS, None
                elif forward_batch.extend_prefix_lens is not None:
                    his_mode, his_src = HIS_PREFIX, forward_batch.extend_prefix_lens
                else:
                    # draft_extend_v2 capture has no extend_prefix_lens.
                    his_mode, his_src = HIS_SEQ_MINUS_EXT, forward_batch.seq_lens
                fused = fused_extend_sconv_metadata(
                    B=B,
                    T=num_tokens,
                    cache_indices=cache_indices,
                    his_mode=his_mode,
                    extend_seq_lens=forward_batch.extend_seq_lens,
                    his_src=his_src,
                )
            if fused is not None:
                query_start_loc, has_initial_state, precomputed = fused
            else:
                query_start_loc, has_initial_state = (
                    self._unfused_extend_common_metadata(forward_batch)
                )
                precomputed = precompute_helion_extend_metadata(
                    B=B,
                    T=num_tokens,
                    W=self.kernel_size[0],
                    cache_indices=cache_indices,
                    has_initial_state=has_initial_state,
                    query_start_loc=query_start_loc,
                )
            _metadata_cache["query_start_loc"] = query_start_loc
            _metadata_cache["has_initial_state"] = has_initial_state
            _metadata_cache["helion_precomputed_extend"] = precomputed
        return _metadata_cache["query_start_loc"], _metadata_cache["has_initial_state"]

    def _unfused_extend_common_metadata(self, forward_batch: ForwardBatch):
        """Original multi-kernel query_start_loc/has_initial_state prep; fused
        fallback only."""
        device = forward_batch.req_pool_indices.device
        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = forward_batch.spec_info.draft_token_num
            query_start_loc = torch.arange(
                0,
                (forward_batch.batch_size + 1) * draft_token_num,
                draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            has_initial_state = torch.ones(
                forward_batch.batch_size, dtype=torch.bool, device=device
            )
            return query_start_loc, has_initial_state
        query_start_loc = torch.zeros(
            forward_batch.batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        query_start_loc[1:] = forward_batch.extend_seq_lens.cumsum(dim=0)
        spec_info = forward_batch.spec_info
        if (
            isinstance(spec_info, EagleDraftExtendInput)
            and spec_info.num_front_tokens > 0
        ):
            has_initial_state = torch.zeros(
                forward_batch.batch_size, dtype=torch.bool, device=device
            )
        elif forward_batch.extend_prefix_lens is not None:
            has_initial_state = forward_batch.extend_prefix_lens > 0
        else:
            has_initial_state = (
                forward_batch.seq_lens[: forward_batch.batch_size]
                - forward_batch.extend_seq_lens
            ) > 0
        return query_start_loc, has_initial_state

    def _prepare_extend_sconv_metadata(
        self, forward_batch: ForwardBatch, cache_indices: torch.Tensor
    ) -> SconvExtendMetadata | Any:
        # Filled by _prepare_extend_common_metadata, which every caller invokes
        # first with the same cache_indices (the fused kernel produces the
        # whole metadata set in one launch).
        del forward_batch, cache_indices
        return _metadata_cache["helion_precomputed_extend"]

    def _prepare_decode_sconv_metadata(
        self, forward_batch: ForwardBatch, cache_indices: torch.Tensor
    ):
        if self.layer_id == 0:
            query_start_loc, has_initial_state, precomputed = (
                fused_decode_sconv_metadata(
                    B=forward_batch.batch_size, cache_indices=cache_indices
                )
            )
            _metadata_cache["query_start_loc_decode"] = query_start_loc
            _metadata_cache["has_initial_state_decode"] = has_initial_state
            _metadata_cache["helion_precomputed_decode"] = precomputed
        return (
            _metadata_cache["query_start_loc_decode"],
            _metadata_cache["has_initial_state_decode"],
            _metadata_cache["helion_precomputed_decode"],
        )

    def _apply_training_sconv_kernel(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor,
        precomputed: SconvDecodeMetadata | SconvExtendMetadata,
        is_decode: bool = False,
    ) -> torch.Tensor:
        y = causal_conv1d(
            x=hidden_states,
            weight=weight,
            sconv_cache=sconv_cache,
            activation=self.activation,
            use_residual=self.use_residual,
            is_decode=is_decode,
            **precomputed,
        )
        update_sconv_cache(
            x=hidden_states,
            sconv_cache=sconv_cache,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            query_start_loc=query_start_loc,
        )
        return y

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) processes sequences in chunks
        of FLA_CHUNK_SIZE. We only track the conv state up to the last complete chunk boundary
        (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        conv_state_len = self.kernel_size[0] - 1

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        chunk_aligned_lens_to_track = (
            lens_to_track // mamba_cache_chunk_size
        ) * mamba_cache_chunk_size
        start_indices = (
            query_start_loc[:-1] + chunk_aligned_lens_to_track - conv_state_len
        )

        # Create indices: [batch_size, conv_state_len] or padded batch_size in prefill cudagraph
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=forward_batch.req_pool_indices.device,
            dtype=start_indices.dtype,
        )

        # Use slice [-1:] instead of [-1] to avoid 0-d tensor -> scalar conversion during graph capture
        return torch.clamp(
            indices,
            min=torch.zeros(
                (1,),
                dtype=start_indices.dtype,
                device=forward_batch.req_pool_indices.device,
            ),
            max=query_start_loc[-1:] - 1,
        )

    def _prepare_extend_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self.layer_id == 0:
            track_conv_indices = self._init_track_conv_indices(
                query_start_loc, forward_batch
            )
            _metadata_cache["track_conv_indices_extend"] = track_conv_indices
        return _metadata_cache["track_conv_indices_extend"]

    def _prepare_cache_indices(
        self, req_to_token_pool, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Resolve the per-request mamba slot indices ONCE per forward step.

        ``get_mamba_indices`` is a GPU gather
        (``req_index_to_mamba_index_mapping[req_pool_indices]``) that depends
        only on ``forward_batch.req_pool_indices``, which is invariant across
        every sconv layer within a step. Computing it in each layer's
        ``forward`` launched one redundant gather kernel per k_sconv/v_sconv
        (``2 * num_attn_layers`` per step). Cache the layer-0 result in the
        shared per-step metadata cache and hand it back to subsequent layers,
        so all layers reuse the same resolved indices.

        Cuda-graph-safe: on capture only layer 0's gather is recorded and
        subsequent layers read that captured tensor; on replay layer 0's gather
        re-runs into the same address, keeping it current -- the same mechanism
        the other ``_metadata_cache`` entries already rely on.

        Under de-tied DRAFT_EXTEND_V2 each per-step forward runs with
        layer_id != 0 against its own draft pool, so every step must own its
        gather instead of reusing another forward's cached tensor (same rule
        as ``_owns_extend_metadata``).
        """
        if self._owns_extend_metadata(forward_batch):
            _metadata_cache["cache_indices"] = (
                req_to_token_pool.translate_mamba_indices(
                    req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
                )
            )
        return _metadata_cache["cache_indices"]

    def _prepare_extend_sconv_cache(
        self,
        forward_batch: ForwardBatch,
        sconv_cache: torch.Tensor,
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor,
    ):
        if forward_batch.mamba_track_mask is not None:
            # Track conv state for prefix caching. Fused gather→scatter writes
            # directly into sconv_cache without an intermediate [B, W-1, D] buffer.
            conv_dst = forward_batch.mamba_track_indices
            # [B, W - 1]
            track_conv_indices = self._prepare_extend_track_conv_indices(
                query_start_loc, forward_batch
            )
            fused_gather_scatter_to_sconv_cache(
                hidden_states=hidden_states,
                sconv_cache=sconv_cache,
                track_conv_indices=track_conv_indices,
                mask=forward_batch.mamba_track_mask,
                dst_indices=conv_dst,
            )

    def _save_intermediate_conv_windows(
        self,
        forward_batch: ForwardBatch,
        cache: MambaPool.SpeculativeState,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """Save intermediate conv windows per draft token for speculative decoding.

        Builds a padded sequence [initial_conv_state | draft_tokens] and extracts
        sliding windows of size (kernel_size - 1) after each draft token position.
        These intermediate states are consumed by
        InklingForConditionalGeneration.update_conv_state_after_mtp_verify
        to restore the correct conv state for the number of accepted tokens.
        """
        save_intermediate_conv_windows(
            sconv_cache=sconv_cache,
            hidden_states=hidden_states,
            cache_indices=cache_indices,
            intermediate_out=cache.intermediate_conv_window[self.sconv_type.value],
            batch_size=forward_batch.batch_size,
            draft_token_num=forward_batch.spec_info.draft_token_num,
        )

    def _update_sconv_cache_for_draft_extend(
        self,
        forward_batch: ForwardBatch,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """Write the correct conv state based on how many tokens were accepted.

        During DRAFT_EXTEND_V2 the draft model processes all num_draft_tokens
        through its sconv layers, but only num_accept_tokens of them should be
        reflected in the final conv state.  We reconstruct the sliding-window
        state after exactly num_accept_tokens and write it to the cache,
        replacing the normal update_sconv_cache call.

        If mamba persistent caching is enabled and the accepted range crosses a
        mamba_track_interval boundary, also writes the conv state at that boundary
        to the persistent ping-pong cache (mamba_track_indices).
        """
        num_accept_tokens = forward_batch.spec_info.num_accept_tokens
        batch_size = forward_batch.batch_size
        if len(hidden_states.shape) == 2:
            draft_token_num = hidden_states.shape[0] // batch_size
        else:
            draft_token_num = hidden_states.shape[1]

        mamba_track_indices = getattr(forward_batch, "mamba_track_indices", None)
        do_tracking = (
            mamba_track_indices is not None
            and get_global_server_args().enable_mamba_extra_buffer()
        )

        crossed = track_step = None
        if do_tracking:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            pre_seqlen = forward_batch.seq_lens[:batch_size] - draft_token_num
            post_seqlen = pre_seqlen + num_accept_tokens
            crossed = (pre_seqlen // mamba_track_interval) != (
                post_seqlen // mamba_track_interval
            )
            tracking_boundary = (
                post_seqlen // mamba_track_interval
            ) * mamba_track_interval
            track_step = (tracking_boundary - pre_seqlen - 1).clamp(0, draft_token_num)

        fused_draft_extend_sconv_cache(
            hidden_states=hidden_states,
            sconv_cache=sconv_cache,
            cache_indices=cache_indices[:batch_size],
            num_accept_tokens=num_accept_tokens,
            draft_token_num=draft_token_num,
            do_tracking=do_tracking,
            crossed=crossed,
            track_step=track_step,
            mamba_track_indices=(
                mamba_track_indices[:batch_size] if do_tracking else None
            ),
        )

    def decode_fused_ar_inputs(self, forward_batch: ForwardBatch):
        """Return inputs for fused decode all-reduce, convolution, and norm.

        These match the fused decode branch of ``forward``, including its
        per-step metadata cache behavior. Returns
        ``(sconv_cache, cache_indices, cache_mask, weight_2d)``."""
        req_to_token_pool = get_req_to_token_pool()
        cache = req_to_token_pool.mamba2_layer_cache(self.layer_id)
        sconv_cache = cache.conv[self.sconv_type.value]
        cache_indices = self._prepare_cache_indices(req_to_token_pool, forward_batch)
        _, _, precomputed = self._prepare_decode_sconv_metadata(
            forward_batch, cache_indices
        )
        weight = rearrange(self.weight, "d 1 w -> d w")
        return sconv_cache, cache_indices, precomputed["cache_mask"], weight

    def verify_fused_ar_inputs(self, forward_batch: ForwardBatch):
        """Return inputs for fused target-verify convolution and norm.

        These mirror the target-verify branch of ``forward``. Returns ``(sconv_cache,
        cache_indices[B], has_initial_state[B], weight_2d, inter_out)``."""
        req_to_token_pool = get_req_to_token_pool()
        cache = req_to_token_pool.mamba2_layer_cache(self.layer_id)
        sconv_cache = cache.conv[self.sconv_type.value]
        cache_indices = self._prepare_cache_indices(req_to_token_pool, forward_batch)
        _, has_initial_state = self._prepare_extend_common_metadata(
            forward_batch, cache_indices
        )
        weight = rearrange(self.weight, "d 1 w -> d w")
        inter_out = cache.intermediate_conv_window[self.sconv_type.value]
        b = forward_batch.batch_size
        return sconv_cache, cache_indices[:b], has_initial_state, weight, inter_out

    def extend_fused_ar_inputs(self, forward_batch: ForwardBatch):
        """Return inputs for fused extend all-reduce and scattered convolution.

        These mirror the extend branch of ``forward`` with convolution fused.
        Returns ``(sconv_cache, safe_idx[B], cache_mask[B], cu[B+1], si[T],
        weight_2d, query_start_loc, cache_indices, has_initial_state,
        track_rows, track_mask, track_dst)``. ``query_start_loc`` is unused by
        the fused-AR caller (kept for symmetry with the unfused metadata
        prep); ``cache_indices``/``has_initial_state`` feed the in-kernel
        cache update, and ``track_rows``/``track_mask``/``track_dst`` feed the
        in-kernel prefix-cache track."""
        req_to_token_pool = get_req_to_token_pool()
        cache = req_to_token_pool.mamba2_layer_cache(self.layer_id)
        sconv_cache = cache.conv[self.sconv_type.value]
        cache_indices = self._prepare_cache_indices(req_to_token_pool, forward_batch)
        weight = rearrange(self.weight, "d 1 w -> d w")
        if forward_batch.forward_mode.is_decode():
            # Decode: every token its own sequence (arange qsl, has_init=ones).
            query_start_loc, has_initial_state, precomputed = (
                self._prepare_decode_sconv_metadata(forward_batch, cache_indices)
            )
        else:
            query_start_loc, has_initial_state = self._prepare_extend_common_metadata(
                forward_batch, cache_indices
            )
            precomputed = self._prepare_extend_sconv_metadata(
                forward_batch, cache_indices
            )
        # Prefix-cache track inputs (extend only; the kernel fuses the write).
        dev = cache_indices.device
        if (
            forward_batch.mamba_track_mask is not None
            and not forward_batch.forward_mode.is_decode()
        ):
            track_rows = self._prepare_extend_track_conv_indices(
                query_start_loc, forward_batch
            ).long()
            track_mask = forward_batch.mamba_track_mask
            track_dst = forward_batch.mamba_track_indices
        else:
            w1 = self.kernel_size[0] - 1
            track_rows = torch.empty((0, w1), dtype=torch.int64, device=dev)
            track_mask = torch.empty((0,), dtype=torch.bool, device=dev)
            track_dst = torch.empty((0,), dtype=torch.int64, device=dev)
        return (
            sconv_cache,
            precomputed["safe_idx"],
            precomputed["cache_mask"].view(-1),
            precomputed["cu"],
            precomputed["si"],
            weight,
            query_start_loc,
            cache_indices,
            has_initial_state,
            track_rows,
            track_mask,
            track_dst,
        )

    def verify_fused_ar_finish(
        self,
        forward_batch: ForwardBatch,
        x_scratch: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> None:
        """Target-verify finish for the fused {AR + scattered sconv} path: no
        working-cache update; save the per-position windows (consumed by
        update_conv_state_after_mtp_verify), exactly as the verify branch of
        ``forward`` does -- on the reduced pre-conv x."""
        req_to_token_pool = get_req_to_token_pool()
        cache = req_to_token_pool.mamba2_layer_cache(self.layer_id)
        sconv_cache = cache.conv[self.sconv_type.value]
        self._save_intermediate_conv_windows(
            forward_batch=forward_batch,
            cache=cache,
            sconv_cache=sconv_cache,
            cache_indices=cache_indices,
            hidden_states=x_scratch,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]` or `[B*T, D]`.
            sequence_info (SequenceInfo):
                Sequence info for handling batch/sequence dimensions. Required.

        Returns:
            Output tensor with same shape as input x.
        """
        del positions

        req_to_token_pool = get_req_to_token_pool()
        cache = req_to_token_pool.mamba2_layer_cache(self.layer_id)
        sconv_cache = cache.conv[self.sconv_type.value]
        cache_indices = self._prepare_cache_indices(req_to_token_pool, forward_batch)

        weight = rearrange(self.weight, "d 1 w -> d w")

        if forward_batch.forward_mode.is_target_verify():
            query_start_loc, has_initial_state = self._prepare_extend_common_metadata(
                forward_batch, cache_indices
            )
            precomputed = self._prepare_extend_sconv_metadata(
                forward_batch, cache_indices
            )
            y = causal_conv1d(
                x=hidden_states,
                weight=weight,
                sconv_cache=sconv_cache,
                activation=self.activation,
                use_residual=self.use_residual,
                is_decode=False,
                **precomputed,
            )
            self._save_intermediate_conv_windows(
                forward_batch=forward_batch,
                cache=cache,
                sconv_cache=sconv_cache,
                cache_indices=cache_indices,
                hidden_states=hidden_states,
            )

        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            query_start_loc, has_initial_state = self._prepare_extend_common_metadata(
                forward_batch, cache_indices
            )
            self._prepare_extend_sconv_cache(
                forward_batch, sconv_cache, hidden_states, query_start_loc
            )

            precomputed = self._prepare_extend_sconv_metadata(
                forward_batch, cache_indices
            )
            if forward_batch.forward_mode.is_draft_extend_v2():
                y = causal_conv1d(
                    x=hidden_states,
                    weight=weight,
                    sconv_cache=sconv_cache,
                    activation=self.activation,
                    use_residual=self.use_residual,
                    is_decode=False,
                    **precomputed,
                )
                self._update_sconv_cache_for_draft_extend(
                    forward_batch,
                    sconv_cache,
                    cache_indices,
                    hidden_states,
                )
            else:
                y = self._apply_training_sconv_kernel(
                    hidden_states=hidden_states,
                    weight=weight,
                    sconv_cache=sconv_cache,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    has_initial_state=has_initial_state,
                    precomputed=precomputed,
                    is_decode=False,
                )
        else:
            # Fused decode: prefix construction + conv + cache update + prefix-cache
            # track-copy in a single Triton kernel. Reads sconv_cache directly (no
            # intermediate prefix tensor) and snapshots the post-update conv window
            # into the persistent ping-pong slot in-register (no separate
            # copy_if_needed launch). track_mask is None when prefix caching with the
            # mamba extra buffer is disabled, which disables the track-copy path.
            _query_start_loc, _has_initial_state, precomputed = (
                self._prepare_decode_sconv_metadata(forward_batch, cache_indices)
            )
            y = fused_causal_conv1d_update_decode(
                x=hidden_states,
                weight=weight,
                sconv_cache=sconv_cache,
                cache_indices=cache_indices,
                cache_mask=precomputed["cache_mask"],
                activation=self.activation,
                use_residual=self.use_residual,
                track_mask=forward_batch.mamba_track_mask,
                track_indices=forward_batch.mamba_track_indices,
            )

        return y


# ---------------------------------------------------------------------------
# Fused gather→scatter: replaces hidden_states[indices].contiguous() + copy_if_needed
# ---------------------------------------------------------------------------


@triton.jit
def _fused_gather_scatter_to_sconv_cache_kernel(
    hidden_ptr,  # [T, D]
    sconv_cache_ptr,  # [pool, W-1, D]
    track_idx_ptr,  # [B, W-1] any int dtype, any strides
    mask_ptr,  # [B] bool
    dst_ptr,  # [B] any int dtype, any stride
    stride_hs_t,  # hidden_states row stride
    stride_cache_slot,  # sconv_cache outer (pool) stride
    stride_cache_w,  # sconv_cache w-position stride
    stride_track_b,  # track_idx row (batch) stride
    stride_track_w,  # track_idx w-position stride
    stride_dst_b,  # dst_indices element stride
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """For each batch element b where mask[b] is True, copy W-1 token rows
    from hidden_states into sconv_cache[dst[b]], reading token positions from
    track_idx[b, w].  Eliminates the intermediate [B, W-1, D] gather buffer
    and the subsequent copy_if_needed call.

    Index tensors are read at their native dtype/strides and cast to int64
    in-kernel, so the host-side `.to(int32/int64).contiguous()` casts that
    previously launched separate kernels are folded away.
    """
    bid = tl.program_id(0)
    pid_d = tl.program_id(1)

    if not tl.load(mask_ptr + bid):
        return

    dst_slot = tl.load(dst_ptr + bid * stride_dst_b).to(tl.int64)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    for w in tl.static_range(W_MINUS_1):
        src_t = tl.load(track_idx_ptr + bid * stride_track_b + w * stride_track_w).to(
            tl.int64
        )
        val = tl.load(hidden_ptr + src_t * stride_hs_t + d_off, mask=d_mask, other=0.0)
        tl.store(
            sconv_cache_ptr + dst_slot * stride_cache_slot + w * stride_cache_w + d_off,
            val,
            mask=d_mask,
        )


def fused_gather_scatter_to_sconv_cache(
    hidden_states: torch.Tensor,  # [T, D]
    sconv_cache: torch.Tensor,  # [pool, W-1, D]
    track_conv_indices: torch.Tensor,  # [B, W-1] int (any int dtype/strides)
    mask: torch.Tensor,  # [B] bool
    dst_indices: torch.Tensor,  # [B] int (any int dtype/strides)
) -> None:
    """Fused replacement for: hidden_states[track_conv_indices].contiguous() + copy_if_needed.

    Writes masked rows from hidden_states directly into sconv_cache without
    allocating the intermediate [B, W-1, D] gather buffer.

    The index-tensor dtype casts (`.to(int32/int64)`) and `.contiguous()` are
    folded INTO the kernel: it reads track_conv_indices / dst_indices at their
    native dtype and strides and casts to int64 internally, eliminating the
    separate cast/contiguous kernel launches.
    """
    D = hidden_states.shape[-1]
    W_minus_1 = sconv_cache.shape[1]
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    B = mask.shape[0]

    if (
        is_cuda()
        and hidden_states.dtype == torch.bfloat16
        and D % 2 == 0
        and hidden_states.stride(-1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.jit_kernel.inkling_sconv import (
            fused_gather_scatter_to_sconv_cache as _cuda_gs,
        )

        _cuda_gs(
            hidden_states,
            sconv_cache,
            track_conv_indices.to(torch.int32),
            mask,
            dst_indices.to(torch.int64),
        )
        return

    grid = (B, triton.cdiv(D, BLOCK_D))
    _fused_gather_scatter_to_sconv_cache_kernel[grid](
        hidden_states,
        sconv_cache,
        track_conv_indices,
        mask,
        dst_indices,
        hidden_states.stride(0),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        track_conv_indices.stride(0),
        track_conv_indices.stride(1),
        dst_indices.stride(0),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# Fused draft-extend sconv cache update
# Replaces: initial_state gather + padded cat + windows unfold +
#           track gather/transpose/contiguous/copy_if_needed +
#           accepted gather/transpose/contiguous/scatter
# ---------------------------------------------------------------------------


@triton.jit
def _fused_draft_extend_sconv_cache_kernel(
    hidden_ptr,  # [B*T, D] or [B, T, D] — non-contiguous ok
    sconv_cache_ptr,  # [pool, W-1, D]
    cache_indices_ptr,  # [B] int32  – working cache slots
    num_accept_ptr,  # [B] int32  – accepted token count per seq
    crossed_ptr,  # [B] bool   – tracking boundary crossed
    track_step_ptr,  # [B] int32  – position in [0,T] for tracking
    mamba_track_indices_ptr,  # [B] int64  – persistent cache slots
    stride_hs_b,  # per-batch stride in hidden_states
    stride_hs_t,  # per-token stride in hidden_states
    stride_cache_slot,  # sconv_cache dim-0 stride (slot)
    stride_cache_w,  # sconv_cache dim-1 stride (w-position)
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DO_TRACKING: tl.constexpr,
):
    """Single-kernel replacement for the whole _update_sconv_cache_for_draft_extend body.

    Reads from the "virtual padded" sequence without materialising it:
      padded[b, j] = sconv_cache[ci, j, :]   for j <  W_MINUS_1   (initial state)
      padded[b, j] = hidden_states[b, j-W_MINUS_1, :]  otherwise  (draft tokens)

    Writes (in order, to avoid RAW conflicts on sconv_cache[ci]):
      1. Tracking window → sconv_cache[mamba_track_indices[b]] (if DO_TRACKING & crossed)
      2. Accepted window → sconv_cache[cache_indices[b]]

    RAW safety: at accepted-write iteration w, we read padded position n_acc+w.
    Since n_acc >= 0, that position is always > w-1 (the latest slot written so far),
    so we never read a slot overwritten by an earlier iteration.
    """
    bid = tl.program_id(0)
    pid_d = tl.program_id(1)

    # CUDA-graph padded rows carry a stale cache index over a possibly-live
    # slot; they are marked with a negative accept count and must not write.
    n_acc = tl.load(num_accept_ptr + bid).to(tl.int64)
    if n_acc < 0:
        return

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    ci = tl.load(cache_indices_ptr + bid).to(tl.int64)

    # -- 1. Tracking write (conditional, must precede accepted write) -----------
    if DO_TRACKING:
        do_track = tl.load(crossed_ptr + bid)
        if do_track:
            step = tl.load(track_step_ptr + bid).to(tl.int64)
            dst_slot = tl.load(mamba_track_indices_ptr + bid)
            for w in tl.static_range(W_MINUS_1):
                pos = step + w
                if pos < W_MINUS_1:
                    val = tl.load(
                        sconv_cache_ptr
                        + ci * stride_cache_slot
                        + pos * stride_cache_w
                        + d_off,
                        mask=d_mask,
                        other=0.0,
                    )
                else:
                    val = tl.load(
                        hidden_ptr
                        + bid * stride_hs_b
                        + (pos - W_MINUS_1) * stride_hs_t
                        + d_off,
                        mask=d_mask,
                        other=0.0,
                    )
                tl.store(
                    sconv_cache_ptr
                    + dst_slot * stride_cache_slot
                    + w * stride_cache_w
                    + d_off,
                    val,
                    mask=d_mask,
                )

    # -- 2. Accepted-window write ------------------------------------------------
    for w in tl.static_range(W_MINUS_1):
        pos = n_acc + w
        if pos < W_MINUS_1:
            val = tl.load(
                sconv_cache_ptr + ci * stride_cache_slot + pos * stride_cache_w + d_off,
                mask=d_mask,
                other=0.0,
            )
        else:
            val = tl.load(
                hidden_ptr
                + bid * stride_hs_b
                + (pos - W_MINUS_1) * stride_hs_t
                + d_off,
                mask=d_mask,
                other=0.0,
            )
        tl.store(
            sconv_cache_ptr + ci * stride_cache_slot + w * stride_cache_w + d_off,
            val,
            mask=d_mask,
        )


def fused_draft_extend_sconv_cache(
    hidden_states: torch.Tensor,  # [B*T, D] or [B, T, D]
    sconv_cache: torch.Tensor,  # [pool, W-1, D]
    cache_indices: torch.Tensor,  # [B] int32
    num_accept_tokens: torch.Tensor,  # [B] int32
    draft_token_num: int,
    do_tracking: bool = False,
    crossed: torch.Tensor | None = None,  # [B] bool
    track_step: torch.Tensor | None = None,  # [B] int32
    mamba_track_indices: torch.Tensor | None = None,  # [B] int64
) -> None:
    """Fused replacement for _update_sconv_cache_for_draft_extend.

    Eliminates: initial_state gather, padded cat, windows unfold,
    track_selected gather/transpose/contiguous/copy_if_needed,
    selected gather/transpose/contiguous/scatter — all in one kernel.
    """
    B = cache_indices.shape[0]
    D = sconv_cache.shape[2]
    W_minus_1 = sconv_cache.shape[1]

    if (
        is_cuda()
        and hidden_states.ndim == 2
        and hidden_states.dtype == torch.bfloat16
        and D % 2 == 0
        and hidden_states.stride(-1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.jit_kernel.inkling_sconv import (
            fused_draft_extend_sconv_cache as _cuda_de,
        )

        _cuda_de(
            hidden_states,
            sconv_cache,
            cache_indices.to(torch.int32),
            num_accept_tokens.to(torch.int32),
            draft_token_num,
            do_tracking,
            crossed,
            track_step.to(torch.int32) if track_step is not None else None,
            (
                mamba_track_indices.to(torch.int64)
                if mamba_track_indices is not None
                else None
            ),
        )
        return

    BLOCK_D = min(triton.next_power_of_2(D), 1024)

    if hidden_states.ndim == 2:
        stride_hs_b = hidden_states.stride(0) * draft_token_num
        stride_hs_t = hidden_states.stride(0)
    else:
        stride_hs_b = hidden_states.stride(0)
        stride_hs_t = hidden_states.stride(1)

    # Sentinel tensors for the no-tracking path (never dereferenced).
    _dummy_bool = torch.zeros(1, dtype=torch.bool, device=sconv_cache.device)
    _dummy_int = torch.zeros(1, dtype=torch.int32, device=sconv_cache.device)
    _dummy_int64 = torch.zeros(1, dtype=torch.int64, device=sconv_cache.device)

    grid = (B, triton.cdiv(D, BLOCK_D))
    _fused_draft_extend_sconv_cache_kernel[grid](
        hidden_states,
        sconv_cache,
        cache_indices.to(torch.int32).contiguous(),
        num_accept_tokens.to(torch.int32).contiguous(),
        crossed if do_tracking else _dummy_bool,
        track_step.to(torch.int32).contiguous() if do_tracking else _dummy_int,
        (
            mamba_track_indices.to(torch.int64).contiguous()
            if do_tracking
            else _dummy_int64
        ),
        stride_hs_b,
        stride_hs_t,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
        DO_TRACKING=do_tracking,
    )
