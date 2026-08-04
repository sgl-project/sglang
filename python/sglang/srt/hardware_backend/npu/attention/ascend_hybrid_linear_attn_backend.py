import logging
import os
from typing import Optional, Union

import torch

from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
)

from sglang.srt.hardware_backend.npu.kernels.mamba_state_move import (
    move_intermediate_cache,
)
from sglang.srt.hardware_backend.npu.kernels.speculative_state_scatter import (
    speculative_state_scatter_npu,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


def _state_trace_enabled() -> bool:
    marker = os.environ.get("SGLANG_K3_TRACE_STATE_FILE") or os.environ.get(
        "SGLANG_K3_TRACE_HIDDEN_FILE"
    )
    if not marker or not os.path.exists(marker):
        return False
    try:
        return get_parallel().attn_tp_group.rank_in_group == 0
    except (AttributeError, RuntimeError):
        return True


def _trace_selected_state(
    *,
    stage: str,
    tensor_name: str,
    tensor: torch.Tensor,
    slot_indices: torch.Tensor,
    step_indices: Optional[torch.Tensor] = None,
) -> None:
    """Log per-state-layer checksums without changing the cache contents.

    State tracing is marker-gated and limited to local NPU 0.  The diagnostic
    requests use batch size one, but a small request cap keeps accidental larger
    batches from producing unbounded logs.  Reductions happen only while the
    marker exists, after graph warmup.
    """
    if not _state_trace_enabled():
        return
    max_reqs = int(os.environ.get("SGLANG_K3_TRACE_STATE_MAX_REQS", "4"))
    slots = slot_indices.detach().to(torch.int64)[:max_reqs]
    if slots.numel() == 0:
        return

    steps = None
    if step_indices is not None:
        steps = step_indices.detach().to(torch.int64)[: slots.numel()]
        valid = steps >= 0
        slots = slots[valid]
        steps = steps[valid]
        if slots.numel() == 0:
            return

    with torch.no_grad():
        if steps is None:
            selected = tensor[:, slots]
        else:
            selected = tensor[:, slots, steps]
        flat = selected.detach().float().reshape(
            selected.shape[0], slots.numel(), -1
        )
        abs_sums = flat.abs().sum(dim=-1).cpu().tolist()
        signed_sums = flat.sum(dim=-1).cpu().tolist()
        max_abs = flat.abs().amax(dim=-1).cpu().tolist()
        slots_cpu = slots.cpu().tolist()
        steps_cpu = None if steps is None else steps.cpu().tolist()

    for state_layer, (layer_abs, layer_sum, layer_max) in enumerate(
        zip(abs_sums, signed_sums, max_abs)
    ):
        logger.warning(
            "K3_STATE_TRACE stage=%s tensor=%s state_layer=%d "
            "slots=%s steps=%s selected_shape=%s abs_sum=%s sum=%s max_abs=%s",
            stage,
            tensor_name,
            state_layer,
            slots_cpu,
            steps_cpu,
            tuple(selected.shape),
            layer_abs,
            layer_sum,
            layer_max,
        )


def _trace_state_pair(
    *,
    stage: str,
    tensor_name: str,
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    src_slot_indices: torch.Tensor,
    dst_slot_indices: torch.Tensor,
    step_indices: Optional[torch.Tensor] = None,
) -> None:
    """Log exact elementwise agreement for a state commit."""
    if not _state_trace_enabled():
        return
    max_reqs = int(os.environ.get("SGLANG_K3_TRACE_STATE_MAX_REQS", "4"))
    src_slots = src_slot_indices.detach().to(torch.int64)[:max_reqs]
    dst_slots = dst_slot_indices.detach().to(torch.int64)[:max_reqs]
    count = min(src_slots.numel(), dst_slots.numel())
    if count == 0:
        return
    src_slots = src_slots[:count]
    dst_slots = dst_slots[:count]

    steps = None
    if step_indices is not None:
        steps = step_indices.detach().to(torch.int64)[:count]
        valid = steps >= 0
        src_slots = src_slots[valid]
        dst_slots = dst_slots[valid]
        steps = steps[valid]
        if steps.numel() == 0:
            return

    with torch.no_grad():
        src = (
            src_tensor[:, src_slots]
            if steps is None
            else src_tensor[:, src_slots, steps]
        )
        dst = dst_tensor[:, dst_slots]
        if src.shape != dst.shape:
            logger.warning(
                "K3_STATE_PAIR stage=%s tensor=%s shape_mismatch src=%s dst=%s",
                stage,
                tensor_name,
                tuple(src.shape),
                tuple(dst.shape),
            )
            return
        diff = (src.detach().float() - dst.detach().float()).abs().reshape(
            src.shape[0], -1
        )
        max_diff = diff.amax(dim=-1).cpu().tolist()
        mismatch_count = (src != dst).reshape(src.shape[0], -1).sum(dim=-1)
        mismatch_count = mismatch_count.cpu().tolist()

    for state_layer, (layer_max_diff, layer_mismatches) in enumerate(
        zip(max_diff, mismatch_count)
    ):
        logger.warning(
            "K3_STATE_PAIR stage=%s tensor=%s state_layer=%d "
            "src_slots=%s dst_slots=%s steps=%s max_abs_diff=%s mismatches=%s",
            stage,
            tensor_name,
            state_layer,
            src_slots.cpu().tolist(),
            dst_slots.cpu().tolist(),
            None if steps is None else steps.cpu().tolist(),
            layer_max_diff,
            layer_mismatches,
        )


class AscendMambaAttnBackendBase(MambaAttnBackendBase):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.state_indices_list_gdn = []

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.state_indices_list_gdn.append(
                torch.full(
                    ((i + 1) * draft_token_num,),
                    self.pad_slot_id,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
            ssm_state_indices = torch.arange(
                mamba_indices.shape[0] * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1][
                : len(mamba_indices) * spec_info.draft_token_num
            ].copy_(ssm_state_indices)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        # out_graph passes seq_lens_cpu=None at capture; mirror the base guard.
        if seq_lens_cpu is None:
            num_padding = 0
        else:
            num_padding = torch.count_nonzero(
                seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
            )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = 0
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        track_buf = None
        if mamba_track_indices is not None:
            track_buf = mamba_track_indices
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            ssm_state_indices = torch.arange(
                bs * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1].copy_(ssm_state_indices)
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        if not in_capture and _state_trace_enabled():
            trace_stage = (
                "replay_decode_read"
                if forward_mode.is_decode_or_idle()
                else "replay_verify_read"
            )
            trace_caches = (
                self.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            _trace_selected_state(
                stage=trace_stage,
                tensor_name="ssm_persistent",
                tensor=trace_caches.temporal,
                slot_indices=mamba_indices,
            )
            _trace_selected_state(
                stage=trace_stage,
                tensor_name="conv_persistent",
                tensor=trace_caches.conv[0],
                slot_indices=mamba_indices,
            )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
                mamba_track_indices=track_buf,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0  # Mamba attn does not use seq lens to index kv cache


class AscendMamba2AttnBackend(AscendMambaAttnBackendBase):
    pass


class AscendHybridLinearAttnBackend(HybridLinearAttnBackend):
    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AscendMambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        super().__init__(full_attn_backend, linear_attn_backend, full_attn_layers)

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """
        Commit accepted verify states with the fused NPU state-move kernels.

        This avoids the original advanced-indexing kernels and emits marker-gated
        diagnostics for commit-to-replay validation:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        del req_pool_indices  # accepted for hook parity; slots come from metadata
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        dst_indices_tensor = state_indices_tensor.to(torch.int32)
        src_indices_tensor = torch.arange(
            dst_indices_tensor.shape[0],
            device=dst_indices_tensor.device,
            dtype=torch.int32,
        )
        last_steps = last_correct_step_indices.to(torch.int32)

        _trace_selected_state(
            stage="verify_commit_src_before",
            tensor_name="ssm_intermediate",
            tensor=intermediate_state_cache,
            slot_indices=src_indices_tensor,
            step_indices=last_steps,
        )
        _trace_selected_state(
            stage="verify_commit_dst_before",
            tensor_name="ssm_persistent",
            tensor=ssm_states,
            slot_indices=dst_indices_tensor,
        )
        move_intermediate_cache(
            ssm_states,
            intermediate_state_cache,
            dst_indices_tensor,
            src_indices_tensor,
            last_steps,
            h_block_size=1,
        )
        _trace_selected_state(
            stage="verify_commit_dst_after",
            tensor_name="ssm_persistent",
            tensor=ssm_states,
            slot_indices=dst_indices_tensor,
        )
        _trace_state_pair(
            stage="verify_commit_exact",
            tensor_name="ssm",
            src_tensor=intermediate_state_cache,
            dst_tensor=ssm_states,
            src_slot_indices=src_indices_tensor,
            dst_slot_indices=dst_indices_tensor,
            step_indices=last_steps,
        )

        draft_token_num = intermediate_state_cache.shape[2]
        has_conv_snapshots = getattr(
            self.linear_attn_backend,
            "supports_speculative_conv_state_snapshots",
            False,
        )
        if has_conv_snapshots:
            intermediate_conv_window_cache = (
                mamba_caches.intermediate_conv_window[0]
            )
            _trace_selected_state(
                stage="verify_commit_src_before",
                tensor_name="conv_intermediate",
                tensor=intermediate_conv_window_cache,
                slot_indices=src_indices_tensor,
                step_indices=last_steps,
            )
            _trace_selected_state(
                stage="verify_commit_dst_before",
                tensor_name="conv_persistent",
                tensor=conv_states,
                slot_indices=dst_indices_tensor,
            )
            speculative_state_scatter_npu(
                conv_states,
                intermediate_conv_window_cache,
                dst_indices_tensor,
                src_indices_tensor,
                last_steps,
            )
            _trace_selected_state(
                stage="verify_commit_dst_after",
                tensor_name="conv_persistent",
                tensor=conv_states,
                slot_indices=dst_indices_tensor,
            )
            _trace_state_pair(
                stage="verify_commit_exact",
                tensor_name="conv",
                src_tensor=intermediate_conv_window_cache,
                dst_tensor=conv_states,
                src_slot_indices=src_indices_tensor,
                dst_slot_indices=dst_indices_tensor,
                step_indices=last_steps,
            )

        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            mamba_track_indices = mamba_track_indices.to(torch.int32)
            mamba_steps_to_track = mamba_steps_to_track.to(torch.int32)

            move_intermediate_cache(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                src_indices_tensor,
                mamba_steps_to_track,
                h_block_size=1,
            )

            if has_conv_snapshots:
                speculative_state_scatter_npu(
                    conv_states,
                    intermediate_conv_window_cache,
                    mamba_track_indices,
                    src_indices_tensor,
                    mamba_steps_to_track,
                )
            else:
                track_mask = mamba_steps_to_track >= 0
                # Legacy NPU backends do not keep per-step conv intermediates.
                # Preserve their verify-time window before rolling back the
                # working slot.
                track_indices = mamba_track_indices[track_mask]
                if track_indices.numel() > 0:
                    conv_states[:, track_indices] = conv_states[
                        :, dst_indices_tensor[track_mask]
                    ]

        if not has_conv_snapshots:
            if dst_indices_tensor.numel() > 0:
                conv_state_rollback(
                    conv_states,
                    dst_indices_tensor,
                    last_steps,
                    draft_token_num,
                )

            if mamba_track_indices is not None and mamba_track_indices.numel() > 0:
                conv_state_rollback(
                    conv_states,
                    mamba_track_indices,
                    mamba_steps_to_track,
                    draft_token_num,
                )

        return

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass
