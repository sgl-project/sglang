import logging
from typing import Optional, Union

import torch
import triton
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    move_intermediate_cache,
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
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

# Ascend Unified Buffer budget for the mamba gather-scatter kernel. The kernel
# tile is H_BLOCK_SIZE * next_pow2(V) * next_pow2(K) elements, and the Bisheng
# multi-buffer pass doubles that allocation, so a tile larger than half the UB
# fails to compile ("ub overflow"). 192KB UB is the smallest across supported
# SoCs; keep the tile under half of it and let H_BLOCK_SIZE absorb the rest.
_ASCEND_UB_BYTES = 192 * 1024


def _mamba_scatter_h_block_size(intermediate_state_cache: torch.Tensor) -> int:
    """Largest H_BLOCK_SIZE whose (double-buffered) kernel tile fits in UB."""
    _, _, _, _, dim_v, dim_k = intermediate_state_cache.shape
    tile_elems = triton.next_power_of_2(dim_v) * triton.next_power_of_2(dim_k)
    tile_bytes = tile_elems * intermediate_state_cache.element_size()
    # Halve the budget to leave room for the multi-buffer duplicate.
    h_block_size = (_ASCEND_UB_BYTES // 2) // max(tile_bytes, 1)
    if h_block_size < 1:
        # A single (V, K) plane already overflows UB; the kernel would need to
        # tile V/K too. Warn instead of letting Bisheng fail with a raw
        # "ub overflow" during the first verify step.
        logger.warning_once(
            f"Mamba state shape (V={dim_v}, K={dim_k}, "
            f"dtype={intermediate_state_cache.dtype}) needs {tile_bytes} bytes per "
            f"plane, which exceeds the {_ASCEND_UB_BYTES // 2} byte Ascend UB tile "
            "budget; the gather-scatter kernel may fail to compile. Consider "
            "--mamba-ssm-dtype bfloat16."
        )
    return max(h_block_size, 1)


def _commit_conv_windows(
    conv_states: torch.Tensor,
    state_indices: torch.Tensor,
    step_indices: torch.Tensor,
    window_size: int,
):
    """Restore each slot's conv window to the accepted step.

    The NPU verify conv runs with num_accepted_tokens=1, which makes the
    ascendc op read the pre-verify window at offset 0 and write the sliding
    buffer [P1, P2, x0, x1, ..., x_{D-1}] back into the slot (P = pre-verify
    window, x_s = conv input of verify token s). The window that must survive
    the commit for a request that accepted step s is the slice [s, s+window_size)
    of that buffer, so we copy it back over the base window.

    This replaces `conv_state_rollback`, which tried to reconstruct the window
    from the post-verify final state alone and so could not recover the draft
    inputs it no longer contained.
    """
    num_valid = state_indices.shape[0]
    if num_valid == 0:
        return
    valid = step_indices >= 0
    if not valid.any():
        return
    dst = state_indices[valid].to(torch.int64)
    steps = step_indices[valid].to(torch.int64)
    num_valid = dst.shape[0]
    if num_valid == 0:
        return
    src_rows = dst[:, None].expand(num_valid, window_size)
    src_cols = steps[:, None] + torch.arange(window_size, device=dst.device)
    src = conv_states[:, src_rows, src_cols, :].clone()
    dst_rows = dst[:, None].expand(num_valid, window_size)
    dst_cols = torch.arange(window_size, device=dst.device)[None, :].expand(
        num_valid, window_size
    )
    conv_states[:, dst_rows, dst_cols, :] = src


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
                bs * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1].copy_(ssm_state_indices)
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
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
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
        dst_indices_tensor = state_indices_tensor.to(torch.int64)  # [N]
        src_indices_tensor = torch.arange(
            dst_indices_tensor.shape[0],
            device=dst_indices_tensor.device,
            dtype=torch.int64,
        )
        last_steps = last_correct_step_indices.to(torch.int64)  # [N]

        h_block_size = _mamba_scatter_h_block_size(intermediate_state_cache)

        move_intermediate_cache(
            ssm_states,
            intermediate_state_cache,
            dst_indices_tensor,
            src_indices_tensor,
            last_steps,
            h_block_size=h_block_size,
        )

        draft_token_num = intermediate_state_cache.shape[2]
        # conv window: [W + D - 1] buffer, W slots are one window.
        window_size = conv_states.shape[2] - draft_token_num + 1
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            mamba_track_indices = mamba_track_indices.to(torch.int64)
            mamba_steps_to_track = mamba_steps_to_track.to(torch.int64)

            move_intermediate_cache(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                src_indices_tensor,
                mamba_steps_to_track,
                h_block_size=h_block_size,
            )

            track_mask = mamba_steps_to_track >= 0
            # Copy the verify-time sliding buffer (base window + draft inputs)
            # into the track slot before the working slot's window is committed;
            # the per-step conv windows are then sliced out of it the same way.
            track_indices = mamba_track_indices[track_mask]
            if track_indices.numel() > 0:
                conv_states[:, track_indices] = conv_states[
                    :, dst_indices_tensor[track_mask]
                ]

        if dst_indices_tensor.numel() > 0:
            _commit_conv_windows(
                conv_states,
                dst_indices_tensor,
                last_steps,
                window_size,
            )

        if mamba_track_indices is not None and mamba_track_indices.numel() > 0:
            _commit_conv_windows(
                conv_states,
                mamba_track_indices,
                mamba_steps_to_track,
                window_size,
            )

        return

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass
