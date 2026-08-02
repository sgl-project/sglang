"""DVR GDN state adapter and attention backend."""

import math
from typing import Any, Optional, Tuple, Union

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.layers.attention.dvr.gdn_kernels import (
    _compact_gdn_transition_windows,
    _gather_verify_output,
    _pack_verify_window_pair,
    _rebuild_gdn_self_draft_state,
    _scatter_prefill_transitions,
    dvr_chunk_gated_delta_rule,
    dvr_scatter_conv_window,
    dvr_scatter_state,
)
from sglang.srt.layers.attention.linear import gdn_backend as base_gdn
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.nvtx_utils import profile_range

__all__ = [
    "DVRGDNAttnBackend",
    "DVRGDNStateAdapter",
    "dvr_gdn_workspace_state_slots",
]


def _local_gdn_dimensions(state_shape):
    local_value_heads, value_dim, key_dim = state_shape.temporal
    local_key_heads = state_shape.num_k_heads_per_tp
    assert local_key_heads > 0 and local_value_heads > 0
    return local_key_heads, local_value_heads, key_dim, value_dim


def dvr_gdn_workspace_state_slots(
    cache_params,
    num_draft_tokens: int,
    *,
    num_layers: int,
) -> int:
    """Conservatively express DVR workspace as whole Mamba-state slots."""

    intermediate_conv_numel = sum(
        conv_dim * (num_draft_tokens + window_size - 1)
        for conv_dim, window_size in cache_params.shape.conv
    )
    total = num_layers * (
        intermediate_conv_numel * cache_params.dtype.conv.itemsize
        + math.prod(cache_params.shape.temporal) * cache_params.dtype.temporal.itemsize
    )
    # Target verify uses a request-owned conv scratch so the ordinary Mamba
    # kernel can keep its in-place contract without touching accepted state.
    total += (
        num_layers
        * sum(math.prod(shape) for shape in cache_params.shape.conv)
        * cache_params.dtype.conv.itemsize
    )
    local_key_heads, local_value_heads, key_dim, value_dim = _local_gdn_dimensions(
        cache_params.shape
    )
    values_per_token = (
        local_key_heads * key_dim * cache_params.dtype.conv.itemsize
        + local_value_heads * value_dim * cache_params.dtype.conv.itemsize
        + 2 * local_value_heads * torch.float32.itemsize
    )
    workspace_bytes = (
        total
        + num_layers * (FLA_CHUNK_SIZE + num_draft_tokens) * values_per_token
        # Cover request-local boundary metadata without coupling pool sizing to
        # the selected Radix strategy.
        + 6 * torch.int64.itemsize
    )
    return max(1, math.ceil(workspace_bytes / cache_params.mamba_cache_per_req))


class DVRGDNStateAdapter:
    """Adapter for DVR state replay in gated linear-state layers.

    The adapter owns the rolling-window and commit mechanics so model backends
    do not need to know the verify/post-verify lifecycle details.
    """

    def __init__(
        self,
        *,
        state_cache: Any,
        recurrent_workspace: torch.Tensor,
        verify_conv_windows: torch.Tensor,
        transition_cache: tuple[torch.Tensor, ...],
        uses_self_draft: bool,
        chunk_size: int = FLA_CHUNK_SIZE,
    ):
        self.state_cache = state_cache
        self.recurrent_workspace = recurrent_workspace
        self.verify_conv_windows = verify_conv_windows
        self.transition_cache = transition_cache
        self.uses_self_draft = uses_self_draft
        self.chunk_size = chunk_size
        self.verify_state_plan: Optional[tuple[torch.Tensor, ...]] = None
        self.private_conv_state = state_cache.conv[0].new_zeros(
            (
                state_cache.conv[0].shape[0],
                recurrent_workspace.shape[1],
                *state_cache.conv[0].shape[2:],
            )
        )

    @classmethod
    def for_gdn(
        cls,
        *,
        model_runner: Any,
    ) -> "DVRGDNStateAdapter":
        config = hybrid_gdn_config(model_runner.model_config)
        if config is None:
            raise RuntimeError("DVR GDN state adapter requires a GDN model.")
        mamba_cache_params = config.mamba2_cache_params
        state_cache = model_runner.req_to_token_pool.mamba_pool.mamba_cache
        if len(state_cache.conv) != 1:
            raise RuntimeError(
                "DVR GDN currently supports exactly one convolution state per layer."
            )
        state_tensors = (state_cache.temporal, *getattr(state_cache, "conv", ()))
        if any(not tensor.is_contiguous() for tensor in state_tensors):
            raise RuntimeError(
                "DVR GDN verify requires contiguous recurrent-state storage."
            )
        if any(
            getattr(state_cache, name, None) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        ):
            raise RuntimeError("DVR GDN rollback does not support ReplaySSM state.")
        if state_cache.temporal.dtype != torch.float32:
            raise RuntimeError("DVR GDN verify requires fp32 recurrent states.")
        num_layers = state_cache.temporal.shape[0]
        num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        if num_draft_tokens is None:
            raise RuntimeError("DVR requires speculative_num_draft_tokens.")

        state_shape = mamba_cache_params.shape
        local_key_heads, local_value_heads, key_dim, value_dim = _local_gdn_dimensions(
            state_shape
        )
        # Request row 0 is the CUDA-graph padding row. DVR workspaces use request
        # rows, not Mamba slots, so Radix may donate/rebind cache slots normally.
        num_slots = model_runner.req_to_token_pool.req_to_token.shape[0]
        recurrent_workspace = torch.zeros(
            num_layers,
            num_slots,
            1,
            *state_cache.temporal.shape[2:],
            dtype=state_cache.temporal.dtype,
            device=model_runner.device,
        )
        conv_dim, conv_window = state_cache.conv[0].shape[2:]
        intermediate_conv_storage = torch.zeros(
            num_layers,
            num_slots,
            conv_dim,
            num_draft_tokens + conv_window - 1,
            dtype=state_cache.conv[0].dtype,
            device=model_runner.device,
        )
        verify_conv_windows = intermediate_conv_storage.as_strided(
            (
                num_layers,
                num_slots,
                num_draft_tokens,
                conv_dim,
                conv_window,
            ),
            (
                intermediate_conv_storage.stride(0),
                intermediate_conv_storage.stride(1),
                intermediate_conv_storage.stride(3),
                intermediate_conv_storage.stride(2),
                intermediate_conv_storage.stride(3),
            ),
        )
        window_len = FLA_CHUNK_SIZE + num_draft_tokens
        k = torch.zeros(
            num_layers,
            num_slots,
            window_len,
            local_key_heads,
            key_dim,
            dtype=model_runner.dtype,
            device=model_runner.device,
        )
        v = torch.zeros(
            num_layers,
            num_slots,
            window_len,
            local_value_heads,
            value_dim,
            dtype=model_runner.dtype,
            device=model_runner.device,
        )
        gate = torch.zeros(
            num_layers,
            num_slots,
            window_len,
            local_value_heads,
            dtype=torch.float32,
            device=model_runner.device,
        )

        return cls(
            state_cache=state_cache,
            recurrent_workspace=recurrent_workspace,
            verify_conv_windows=verify_conv_windows,
            transition_cache=(k, v, gate, torch.zeros_like(gate)),
            uses_self_draft=model_runner.spec_algorithm.is_dvr_self_draft(),
        )

    def resolve_request_slots(self, *, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve request rows and accepted-endpoint convolution slots."""

        pool = batch.req_to_token_pool
        target_cache_slots = pool.get_mamba_indices(batch.req_pool_indices).to(
            torch.long
        )
        request_rows = batch.req_pool_indices.to(
            device=target_cache_slots.device, dtype=torch.long
        )
        return request_rows, target_cache_slots

    def zero_boundary_state(self, *, indices: torch.Tensor):
        indices = indices.to(device=self.state_cache.temporal.device, dtype=torch.long)
        self.state_cache.temporal[:, indices] = 0

    def initialize_self_draft_state(
        self,
        *,
        request_rows: torch.Tensor,
        target_cache_slots: torch.Tensor,
        tail_lens: torch.Tensor,
    ) -> None:
        """Seed private self-draft state after target EXTEND.

        Target live temporal state remains at the latest exact chunk boundary.
        Replaying only its accepted tail gives self draft the current endpoint
        without letting provisional decode mutate target or Radix-owned state.
        """

        if not self.uses_self_draft:
            return
        target_cache_slots = target_cache_slots.to(
            device=self.state_cache.temporal.device, dtype=torch.long
        )
        request_rows = request_rows.to(
            device=self.state_cache.temporal.device, dtype=torch.long
        )
        with profile_range("draft_state_copy"):
            self.private_conv_state[:, request_rows] = self.state_cache.conv[0][
                :, target_cache_slots
            ]
            _rebuild_gdn_self_draft_state(
                self.transition_cache,
                boundary_state=self.state_cache.temporal,
                draft_state=self.recurrent_workspace,
                request_rows=request_rows,
                boundary_slots=target_cache_slots,
                token_count=tail_lens,
            )

    def decode_state(self, *, layer_cache, forward_batch, layer_idx: int):
        """Return request-owned state for provisional DVR-self decode."""

        if not self.uses_self_draft or not forward_batch.forward_mode.is_decode():
            return None
        request_rows = forward_batch.req_pool_indices.to(
            device=layer_cache.temporal.device, dtype=torch.long
        )
        return (
            self.private_conv_state[layer_idx],
            self.recurrent_workspace[layer_idx, :, 0],
            request_rows,
        )

    def cache_prefill_transitions(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        layer_idx: int,
        forward_batch,
    ):
        prefill_metadata = (
            forward_batch.req_pool_indices,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
        )
        if any(tensor is None for tensor in prefill_metadata):
            raise RuntimeError("DVR GDN target EXTEND metadata was not prepared.")

        # q affects only the output at its own token; it does not participate in
        # the recurrent transition. Target verify receives candidate q directly.
        prefill_transitions = (
            k.reshape(k.shape[1], k.shape[2], k.shape[3]),
            v.reshape(v.shape[1], v.shape[2], v.shape[3]),
            g.reshape(-1, g.shape[-1]),
            beta.reshape(-1, beta.shape[-1]),
        )
        layer_transition_cache = tuple(
            cache[layer_idx] for cache in self.transition_cache
        )
        request_rows, prefix_lens, extend_lens, extend_start_loc = prefill_metadata
        for cache, value in zip(
            layer_transition_cache, prefill_transitions, strict=True
        ):
            _scatter_prefill_transitions(
                cache,
                value,
                request_rows=request_rows,
                prefix_lens=prefix_lens,
                extend_lens=extend_lens,
                extend_start_loc=extend_start_loc,
                chunk_size=self.chunk_size,
            )

    def prepare_forward(self, *, forward_batch, forward_metadata):
        """Prepare request metadata once before all GDN layers execute."""

        self.verify_state_plan = None
        if forward_batch.forward_mode.is_target_verify():
            self.prepare_target_verify(
                forward_batch=forward_batch,
                cache_indices=forward_metadata.mamba_cache_indices,
            )
            return None
        if not forward_batch.forward_mode.is_extend():
            return None

        prefix_lens = forward_batch.extend_prefix_lens
        extend_lens = forward_batch.extend_seq_lens
        if prefix_lens is None or extend_lens is None:
            raise RuntimeError("DVR GDN target EXTEND metadata was not prepared.")
        aligned_prefix = prefix_lens.remainder(self.chunk_size).eq(0)
        boundary_offsets = (
            prefix_lens + extend_lens
        ) // self.chunk_size * self.chunk_size - prefix_lens
        boundary_steps = torch.where(
            aligned_prefix & (boundary_offsets > 0),
            boundary_offsets // self.chunk_size,
            torch.full_like(boundary_offsets, -1),
        )
        return forward_metadata.mamba_cache_indices, boundary_steps

    def prepare_target_verify(self, *, forward_batch, cache_indices: torch.Tensor):
        """Resolve graph-stable request, boundary, tail, and padding metadata once."""

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = forward_batch.input_ids.shape[0] // draft_token_num
        request_rows = forward_batch.req_pool_indices[:batch_size].to(
            device=cache_indices.device, dtype=torch.long
        )
        valid_mask = cache_indices[:batch_size] != PAD_SLOT_ID
        request_rows = torch.where(
            valid_mask, request_rows, torch.zeros_like(request_rows)
        )
        boundary_slots = torch.where(
            valid_mask,
            cache_indices[:batch_size],
            torch.zeros_like(cache_indices[:batch_size]),
        )
        # The stock conv kernel is intentionally in-place. Run it on this tiny
        # request-owned copy so target verify cannot mutate accepted endpoints.
        self.private_conv_state[:, request_rows] = self.state_cache.conv[0][
            :, boundary_slots
        ]
        # TARGET_VERIFY keeps seq_lens at the accepted endpoint while its
        # speculative cache rows are appended separately.
        accepted_tail_lens = (
            forward_batch.seq_lens[:batch_size].to(
                device=cache_indices.device, dtype=torch.long
            )
            % self.chunk_size
        )
        accepted_tail_lens = torch.where(
            valid_mask,
            accepted_tail_lens,
            torch.zeros_like(accepted_tail_lens),
        )
        boundary_state_steps = torch.where(
            valid_mask & (accepted_tail_lens + draft_token_num >= self.chunk_size),
            torch.ones_like(accepted_tail_lens),
            torch.full_like(accepted_tail_lens, -1),
        )
        self.verify_state_plan = (
            boundary_slots,
            request_rows,
            accepted_tail_lens,
            valid_mask,
            boundary_state_steps,
        )

    def verify_conv_state(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return request-owned verify convolution state and request rows."""

        if self.verify_state_plan is None:
            raise RuntimeError("DVR target-verify metadata was not prepared.")
        request_rows = self.verify_state_plan[1]
        return self.private_conv_state[layer_idx], request_rows

    def forward_target_verify(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Replay GDN state from cached transitions and candidate q/k/v/g/beta."""

        if self.verify_state_plan is None:
            raise RuntimeError("DVR target-verify metadata was not prepared.")
        if not query.is_cuda:
            raise RuntimeError("DVR GDN target verify requires CUDA tensors.")
        (
            boundary_slots,
            request_rows,
            accepted_tail_lens,
            valid_mask,
            boundary_state_steps,
        ) = self.verify_state_plan
        batch_size = boundary_slots.shape[0]
        draft_token_num = query.shape[1] // batch_size
        candidate_inputs = (
            query.reshape(batch_size, draft_token_num, *query.shape[2:]),
            key.reshape(batch_size, draft_token_num, *key.shape[2:]),
            value.reshape(batch_size, draft_token_num, *value.shape[2:]),
            g.reshape(batch_size, draft_token_num, g.shape[-1]),
            beta.reshape(batch_size, draft_token_num, beta.shape[-1]),
        )
        k_cache, v_cache, g_cache, beta_cache = (
            cache[layer_idx] for cache in self.transition_cache
        )
        # The fixed window may retain values after accepted_tail_lens + draft_token_num.
        # They are causally after every returned logit, a boundary is exported
        # only after all chunk rows are valid, and state rebuilds use token_count.
        with profile_range("verify_state_pack"):
            # Cached q rows would produce only discarded prefix outputs.
            # Keep them zero while packing candidate q and persistent k in
            # the same launch.
            q, k = _pack_verify_window_pair(
                k_cache,
                candidate_inputs[0],
                cache1=k_cache,
                candidate1=candidate_inputs[1],
                request_rows=request_rows,
                accepted_tail_lens=accepted_tail_lens,
                valid_mask=valid_mask,
                read_cache0=False,
                persist_cache0=False,
            )
            v, _ = _pack_verify_window_pair(
                v_cache,
                candidate_inputs[2],
                request_rows=request_rows,
                accepted_tail_lens=accepted_tail_lens,
                valid_mask=valid_mask,
            )
            cached_g, cached_beta = _pack_verify_window_pair(
                g_cache,
                candidate_inputs[3],
                cache1=beta_cache,
                candidate1=candidate_inputs[4],
                request_rows=request_rows,
                accepted_tail_lens=accepted_tail_lens,
                valid_mask=valid_mask,
            )
        core_attn_out, _, _ = dvr_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=cached_g,
            beta=cached_beta,
            initial_state=self.state_cache.temporal[layer_idx],
            initial_state_indices=boundary_slots,
            boundary_state=self.recurrent_workspace[layer_idx, :, 0],
            boundary_state_indices=request_rows,
            boundary_state_steps=boundary_state_steps,
        )

        with profile_range("verify_output_gather"):
            return _gather_verify_output(
                core_attn_out,
                accepted_tail_lens=accepted_tail_lens,
                draft_tokens=draft_token_num,
            )

    def publish_boundary_state(
        self,
        *,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
        publish_mask: torch.Tensor,
    ) -> None:
        """Copy exact live boundaries to the ordinary Radix tracking slots."""

        dvr_scatter_state(
            self.state_cache.temporal,
            self.state_cache.temporal.unsqueeze(2),
            source_rows=source_slots,
            destination_rows=destination_slots,
            source_steps=torch.where(
                publish_mask,
                torch.zeros_like(source_slots),
                torch.full_like(source_slots, -1),
            ),
        )

    def stage_boundary_state(
        self,
        *,
        request_rows: torch.Tensor,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
        boundary_conv_steps: torch.Tensor,
    ) -> None:
        """Stage one exact live boundary in ordinary Radix tracking storage."""

        dvr_scatter_state(
            self.state_cache.temporal,
            self.state_cache.temporal.unsqueeze(2),
            source_rows=source_slots,
            destination_rows=destination_slots,
            source_steps=torch.where(
                boundary_conv_steps >= 0,
                torch.zeros_like(boundary_conv_steps),
                torch.full_like(boundary_conv_steps, -1),
            ),
        )
        dvr_scatter_conv_window(
            self.state_cache.conv[0],
            self.verify_conv_windows,
            source_rows=request_rows,
            destination_rows=destination_slots,
            source_steps=boundary_conv_steps,
        )

    def commit_accepted_state(
        self,
        *,
        request_rows: torch.Tensor,
        target_cache_slots: torch.Tensor,
        tail_lens_before: torch.Tensor,
        accepted_token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tail_lens_before = tail_lens_before.to(
            device=target_cache_slots.device, dtype=torch.long
        )
        accepted_token_counts = accepted_token_counts.to(
            device=target_cache_slots.device, dtype=torch.long
        )
        tail_lens_after = tail_lens_before + accepted_token_counts
        crosses_chunk_boundary = tail_lens_after >= self.chunk_size
        no_commit_step = torch.full_like(tail_lens_before, -1)
        boundary_conv_steps = torch.where(
            crosses_chunk_boundary,
            self.chunk_size - 1 - tail_lens_before,
            no_commit_step,
        )
        # accept_lens includes the bonus token, so every non-idle request has a
        # valid accepted step. The fused scatter already handles the empty batch.
        dvr_scatter_conv_window(
            self.state_cache.conv[0],
            self.verify_conv_windows,
            source_rows=request_rows,
            destination_rows=target_cache_slots,
            source_steps=accepted_token_counts - 1,
        )
        # Convolution state is committed at the accepted endpoint; unlike
        # temporal state, it is cheap and necessary to resume the next conv.
        if self.uses_self_draft:
            dvr_scatter_conv_window(
                self.private_conv_state,
                self.verify_conv_windows,
                source_rows=request_rows,
                destination_rows=request_rows,
                source_steps=accepted_token_counts - 1,
            )

        # The target live slot owns the newest computed boundary. Radix staging
        # is delayed until the scheduler has either continued or finalized the
        # request, because CPU stop handling may trim the accepted suffix.
        with profile_range("target_boundary_commit"):
            dvr_scatter_state(
                self.state_cache.temporal,
                self.recurrent_workspace,
                source_rows=request_rows,
                destination_rows=target_cache_slots,
                source_steps=torch.where(
                    crosses_chunk_boundary,
                    torch.zeros_like(tail_lens_before),
                    no_commit_step,
                ),
            )
        new_tail_lens = tail_lens_after - self.chunk_size
        tail_lens_after = torch.where(
            crosses_chunk_boundary, new_tail_lens, tail_lens_after
        )
        with profile_range("state_window_compact"):
            _compact_gdn_transition_windows(
                self.transition_cache,
                indices=request_rows,
                crosses_chunk_boundary=crosses_chunk_boundary,
                chunk_size=self.chunk_size,
                accepted_tail_lens=tail_lens_after,
            )
        if self.uses_self_draft:
            # Reconstruct only self-draft's private endpoint. Target recurrent
            # state remains boundary-owned and target verify reads it directly.
            with profile_range("draft_state_rebuild"):
                _rebuild_gdn_self_draft_state(
                    self.transition_cache,
                    boundary_state=self.state_cache.temporal,
                    draft_state=self.recurrent_workspace,
                    request_rows=request_rows,
                    boundary_slots=target_cache_slots,
                    token_count=tail_lens_after,
                )
        # EAGLE/MTP owns separate draft state and skips this reconstruction.

        return crosses_chunk_boundary, boundary_conv_steps


class DVRGDNAttnBackend(GDNAttnBackend):
    """GDN backend whose provisional state and deterministic verify are DVR-owned."""

    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.dvr_state_adapter = DVRGDNStateAdapter.for_gdn(
            model_runner=model_runner,
        )
        self.dvr_extend_boundary_metadata = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self.prepare_dvr_forward(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        super().init_forward_metadata_in_graph(forward_batch)
        self.prepare_dvr_forward(forward_batch)

    def prepare_dvr_forward(self, forward_batch: ForwardBatch):
        self.dvr_extend_boundary_metadata = self.dvr_state_adapter.prepare_forward(
            forward_batch=forward_batch,
            forward_metadata=self.forward_metadata,
        )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        draft_state = self.dvr_state_adapter.decode_state(
            layer_cache=layer_cache,
            forward_batch=forward_batch,
            layer_idx=self.req_to_token_pool.mamba_map[layer.layer_id],
        )
        if draft_state is None:
            return super().forward_decode(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )

        conv_states, ssm_states, cache_indices = draft_state
        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = base_gdn.causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )
        if self.kernel_dispatcher.supports_packed_decode:
            return self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        batch_size = forward_batch.batch_size
        query = query.view(1, batch_size, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, batch_size, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, batch_size, layer.num_v_heads, layer.head_v_dim)
        return self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=self.forward_metadata.query_start_loc,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        metadata = self.forward_metadata
        query_start_loc = metadata.query_start_loc
        cache_indices = metadata.mamba_cache_indices
        retrieve_next_token = metadata.retrieve_next_token
        retrieve_next_sibling = metadata.retrieve_next_sibling
        retrieve_parent_token = metadata.retrieve_parent_token

        layer_idx = self.req_to_token_pool.mamba_map[layer.layer_id]
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        if is_target_verify:
            intermediate_conv_window = self.dvr_state_adapter.verify_conv_windows[
                layer_idx
            ]
            conv_states, conv_indices = self.dvr_state_adapter.verify_conv_state(
                layer_idx
            )
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_tokens = forward_batch.spec_info.draft_token_num
            mixed_qkv = (
                base_gdn.causal_conv1d_update(
                    mixed_qkv.view(batch_size, draft_tokens, -1).transpose(1, 2),
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    layer.activation,
                    conv_state_indices=conv_indices,
                    intermediate_conv_window=intermediate_conv_window,
                    intermediate_state_indices=conv_indices,
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                )
                .transpose(1, 2)
                .reshape(seq_len, -1)
            )
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if metadata.has_mamba_track_mask:
                tracked_conv = mixed_qkv[:, metadata.track_conv_indices].transpose(0, 1)
                conv_states[metadata.conv_states_mask_indices] = tracked_conv
            mixed_qkv = base_gdn.causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=forward_batch.extend_prefix_lens > 0,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if qkv_dim <= base_gdn.MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = base_gdn.fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim)

        g, beta = base_gdn.fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        if is_target_verify:
            return self.dvr_state_adapter.forward_target_verify(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                layer_idx=layer_idx,
            )

        if self.dvr_extend_boundary_metadata is None:
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
        else:
            boundary_indices, boundary_steps = self.dvr_extend_boundary_metadata
            core_attn_out, last_recurrent_state, h = dvr_chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                boundary_state=ssm_states,
                boundary_state_indices=boundary_indices,
                boundary_state_steps=boundary_steps,
            )
        self.dvr_state_adapter.cache_prefill_transitions(
            k=key,
            v=value,
            g=g,
            beta=beta,
            layer_idx=layer_idx,
            forward_batch=forward_batch,
        )
        if h is not None:
            if self.dvr_extend_boundary_metadata is None:
                self._track_mamba_state_extend(forward_batch, h, ssm_states, metadata)
        return core_attn_out
