from typing import Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)

# Lazy import for K-last GDN verify kernel (CuTe DSL)
_cutedsl_gdn_verify_available = None
_cutedsl_gdn_verify_k_last = None

# DEBUG: Global step counters for logging
_debug_extend_step = 0
_debug_verify_step = 0
# Set to False for CUDA Graph compatibility; use env var SGLANG_GDN_DEBUG=1 to enable
import os
_debug_enabled = os.environ.get("SGLANG_GDN_DEBUG", "0") == "1"

def reset_debug_counters():
    """Reset debug counters - call after warmup to start fresh."""
    global _debug_extend_step, _debug_verify_step
    _debug_extend_step = 0
    _debug_verify_step = 0

def enable_debug_logging(enabled=True):
    """Enable or disable debug logging."""
    global _debug_enabled
    _debug_enabled = enabled

def _log_tensor(phase, step, layer, name, t, k_last, transpose_to_vlast=False):
    """Log tensor values in a structured way.
    
    Args:
        phase: "extend" or "verify"
        step: step number (0 for extend, 0/1/2... for verify steps)
        layer: layer index
        name: tensor name
        t: tensor
        k_last: whether using K-last layout
        transpose_to_vlast: whether to transpose for V-last comparison
    """
    if not _debug_enabled or t is None:
        return
    
    # Only log on TP rank 0 to avoid log ordering issues
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except:
        pass
    
    import json
    t_cmp = t.transpose(-1, -2) if transpose_to_vlast else t
    vals = t_cmp.flatten()[:8].tolist()
    open('/lustre/raplab/client/xutingz/workspace/gdn_log/sglang_debug.log', 'a').write(json.dumps({
        "phase": phase, "step": step, "layer": layer, 
        "name": name, "k_last": k_last,
        "shape": list(t.shape), "vals": vals
    }) + '\n')


def _get_cutedsl_gdn_verify():
    """Lazy import for CuTe DSL GDN verify kernel."""
    global _cutedsl_gdn_verify_available, _cutedsl_gdn_verify_k_last
    if _cutedsl_gdn_verify_available is None:
        try:
            from sglang.jit_kernel.cutedsl_gdn_verify import (
                cutedsl_gdn_verify_k_last,
                is_cutedsl_gdn_verify_available,
            )
            _cutedsl_gdn_verify_available = is_cutedsl_gdn_verify_available()
            if _cutedsl_gdn_verify_available:
                _cutedsl_gdn_verify_k_last = cutedsl_gdn_verify_k_last
        except ImportError:
            _cutedsl_gdn_verify_available = False
    return _cutedsl_gdn_verify_available, _cutedsl_gdn_verify_k_last
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cuda, is_npu

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu


# Kernel to track mamba states if needed based on track mask
@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    if not track_mask:
        return

    # Load source and destination indices
    src_idx = tl.load(cache_indices_ptr + batch_idx)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx)

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
    )


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )

        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if forward_batch.spec_info.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrive_next_token
                    retrieve_next_sibling = forward_batch.spec_info.retrive_next_sibling
                    # retrieve_next_token is None during dummy run so skip tensor creation
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

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
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The FLA kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % FLA_CHUNK_SIZE == 0): In this case, FLA will store the to-cache
            state in the last_recurrent_state.
          - Unaligned case (len % FLA_CHUNK_SIZE != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example:
        1. If chunked prefill length is < 64, then only final state has value. In this case we
           cache `final` state.
        2. if chunked prefill length == 64, then only final state has value. In this case we
           cache pos 64, from `final` state
        3. if chunked prefill length >64 and < 128, then both h and final state have value.
           We cache pos 64 from `h` state
        4. if chunked prefill length ==128, then both h and final state have value. We cache
           pos 128 from `final` state. Note `h` doesn't include the pos 128.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        # Move to CPU to avoid kernel launches for masking operations
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        num_h_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1

        # Calculate the starting offset for each sequence in the packed batch
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        is_aligned = (lens_masked % FLA_CHUNK_SIZE) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support FLA_CHUNK_SIZE % page size != 0, then need to modify this
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // FLA_CHUNK_SIZE
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self.forward_metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )

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
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

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
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
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
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.
        """
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_indices,
                forward_batch.batch_size,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the FLA chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to FLA_CHUNK_SIZE. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        if (
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        ):
            h = h.squeeze(0)
            # Check if K-last layout is used (for GDNAttnBackend with MTP optimization)
            ssm_k_last = getattr(self, 'ssm_k_last', False)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                h_to_store = h[forward_metadata.track_ssm_h_src]
                if ssm_k_last:
                    # h is V-last from kernel, transpose to K-last for pool
                    h_to_store = h_to_store.transpose(-1, -2)
                ssm_states[forward_metadata.track_ssm_h_dst] = h_to_store.to(
                    ssm_states.dtype, copy=False
                )
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class KimiLinearAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        q_conv_state = q_conv_state.transpose(-1, -2)
        k_conv_state = k_conv_state.transpose(-1, -2)
        v_conv_state = v_conv_state.transpose(-1, -2)

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        initial_state = ssm_states[cache_indices].contiguous()
        (
            core_attn_out,
            last_recurrent_state,
        ) = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )
        ssm_states[cache_indices] = last_recurrent_state
        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )

        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_state_q, conv_state_k, conv_state_v = mamba_cache_params.conv
        # deal with strides
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        q_proj_states = q_proj_states.transpose(0, 1)
        k_proj_states = k_proj_states.transpose(0, 1)
        v_proj_states = v_proj_states.transpose(0, 1)

        q = causal_conv1d_fn(
            q_proj_states,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=conv_state_q,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        k = causal_conv1d_fn(
            k_proj_states,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=conv_state_k,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        v = causal_conv1d_fn(
            v_proj_states,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=conv_state_v,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        core_attn_out = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )

        return core_attn_out


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        assert (
            self.conv_states_shape[-1] < FLA_CHUNK_SIZE
        ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"
        # Check if SSM states use K-last layout (HV, V, K) for MTP kernel optimization
        # Priority: server_args > model config
        self.ssm_k_last = get_global_server_args().mamba_ssm_k_last

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        # Decode kernel expects V-last layout
        # DEBUG: Compare K-last vs V-last tensor values (all tensors logged in V-last format)
        def _debug_tensor(name, t, transpose_to_vlast=False):
            if not _debug_enabled:  # Skip during CUDA Graph capture
                return
            import json
            if t is None:
                return
            try:
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_rank() != 0:
                    return
            except:
                pass
            t_cmp = t.transpose(-1, -2) if transpose_to_vlast else t
            # Sample first batch, first head, corner values
            if t_cmp.dim() >= 3:
                vals = t_cmp[0, 0, :4, :4].flatten()[:8].tolist() if t_cmp.dim() == 4 else t_cmp[0, :4, :4].flatten()[:8].tolist()
            else:
                vals = t_cmp.flatten()[:8].tolist()
            open('/lustre/raplab/client/xutingz/workspace/gdn_log/sglang_debug.log', 'a').write(json.dumps({
                "loc": "decode", "name": name, "k_last": self.ssm_k_last, 
                "shape": list(t.shape), "vals": vals
            }) + '\n')
        
        if self.ssm_k_last:
            # K-last -> V-last: transpose for kernel
            # ssm_states shape: (pool_size+1, HV, V, K) for K-last
            ssm_input = ssm_states[cache_indices]  # [batch, HV, V, K]
            # Log as V-last for comparison: transpose K-last to V-last
            _debug_tensor("ssm_input", ssm_input, transpose_to_vlast=True)
            ssm_states_v_last = ssm_input.transpose(-1, -2).contiguous()  # [batch, HV, K, V]
            
            core_attn_out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                initial_state_source=ssm_states_v_last,
                initial_state_indices=torch.arange(len(cache_indices), dtype=torch.int32, device=cache_indices.device),
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )
            _debug_tensor("core_attn_out", core_attn_out)
            # Log updated state as V-last
            _debug_tensor("ssm_updated", ssm_states_v_last)
            
            # V-last -> K-last: transpose back and write to pool
            ssm_states[cache_indices] = ssm_states_v_last.transpose(-1, -2)
            # Log writeback as V-last for comparison
            _debug_tensor("ssm_writeback", ssm_states[cache_indices], transpose_to_vlast=True)
        else:
            # V-last mode: ssm_states shape is (pool_size+1, HV, K, V)
            ssm_input = ssm_states[cache_indices] if ssm_states.dim() > 3 else ssm_states
            _debug_tensor("ssm_input", ssm_input)
            
            core_attn_out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )
            _debug_tensor("core_attn_out", core_attn_out)
            _debug_tensor("ssm_updated", ssm_states[cache_indices])
            _debug_tensor("ssm_writeback", ssm_states[cache_indices])

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                # Gather all slices at once: [:, track_conv_indices] -> [d, num_masked, slice_len]
                # track_conv_indices is already filtered and clamped in _init_track_conv_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                # Apply mask and assign to destinations
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        if is_target_verify:
            # Check if K-last GDN verify kernel should be used
            use_k_last_gdn = self.ssm_k_last and retrieve_parent_token is None
            if use_k_last_gdn:
                available, cutedsl_gdn_verify_k_last = _get_cutedsl_gdn_verify()
                use_k_last_gdn = available

            # DEBUG: Log verify tensors with step counter
            global _debug_verify_step
            if layer_id == 0:
                _debug_verify_step += 1  # Increment at layer 0
            cur_verify_step = _debug_verify_step
            
            if use_k_last_gdn:
                # Use K-last CuTe DSL GDN verify kernel (no transpose needed)
                batch_size = seq_len // forward_batch.spec_info.draft_token_num
                draft_token_num = forward_batch.spec_info.draft_token_num
                query_mtp = query.view(batch_size, draft_token_num, num_heads, head_k_dim)
                key_mtp = key.view(batch_size, draft_token_num, num_heads, head_k_dim)
                value_mtp = value.view(batch_size, draft_token_num, num_value_heads, head_v_dim)
                a_mtp = a.view(batch_size, draft_token_num, num_value_heads)
                b_mtp = b.view(batch_size, draft_token_num, num_value_heads)

                # Log input tensors (K-last transposed to V-last for comparison)
                _log_tensor("verify", cur_verify_step, layer_id, "query", query_mtp, use_k_last_gdn)
                _log_tensor("verify", cur_verify_step, layer_id, "value", value_mtp, use_k_last_gdn)
                # DEBUG: Log cache_indices info (only on rank 0)
                if _debug_enabled and layer_id == 0:
                    try:
                        import torch.distributed as dist
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            import json
                            open('/lustre/raplab/client/xutingz/workspace/gdn_log/sglang_debug.log', 'a').write(json.dumps({
                                "phase": "verify", "step": cur_verify_step, "layer": layer_id,
                                "name": "cache_indices_info", "k_last": True,
                                "cache_indices": cache_indices[:min(5, len(cache_indices))].tolist(),
                                "batch_size": batch_size, "used_indices": cache_indices[:batch_size].tolist(),
                            }) + '\n')
                    except:
                        pass
                ssm_input = ssm_states[cache_indices[:batch_size]]
                _log_tensor("verify", cur_verify_step, layer_id, "ssm_input", ssm_input, use_k_last_gdn, transpose_to_vlast=True)

                core_attn_out = cutedsl_gdn_verify_k_last(
                    A_log=A_log,
                    a=a_mtp,
                    dt_bias=dt_bias,
                    q=query_mtp,
                    k=key_mtp,
                    v=value_mtp,
                    b=b_mtp,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices[:batch_size],
                    intermediate_states_buffer=intermediate_state_cache,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,
                    cu_seqlens=query_start_loc,
                    cache_steps=draft_token_num,
                )
                # Reshape output: [B, T, HV, V] -> [1, seq_len, HV, V]
                core_attn_out_reshaped = core_attn_out.view(1, seq_len, num_value_heads, head_v_dim)
                _log_tensor("verify", cur_verify_step, layer_id, "core_attn_out", core_attn_out_reshaped, use_k_last_gdn)
                core_attn_out = core_attn_out_reshaped
            else:
                # Use original V-last kernel with pre-computed gates
                g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
                
                # Log input tensors
                _log_tensor("verify", cur_verify_step, layer_id, "query", query, use_k_last_gdn)
                _log_tensor("verify", cur_verify_step, layer_id, "value", value, use_k_last_gdn)
                # DEBUG: Log cache_indices info (only on rank 0)
                if _debug_enabled and layer_id == 0:
                    try:
                        import torch.distributed as dist
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            import json
                            open('/lustre/raplab/client/xutingz/workspace/gdn_log/sglang_debug.log', 'a').write(json.dumps({
                                "phase": "verify", "step": cur_verify_step, "layer": layer_id,
                                "name": "cache_indices_info", "k_last": False,
                                "cache_indices": cache_indices[:min(5, len(cache_indices))].tolist(),
                            }) + '\n')
                    except:
                        pass
                ssm_input = ssm_states[cache_indices] if ssm_states.dim() > 3 else ssm_states
                _log_tensor("verify", cur_verify_step, layer_id, "ssm_input", ssm_input, use_k_last_gdn)
                
                core_attn_out = fused_recurrent_gated_delta_rule_update(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices,
                    cu_seqlens=query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                )
                _log_tensor("verify", cur_verify_step, layer_id, "core_attn_out", core_attn_out, use_k_last_gdn)
        else:
            # Prefill/Extend path
            g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
            
            # DEBUG: Log extend tensors with step counter
            global _debug_extend_step
            if layer_id == 0:
                _debug_extend_step += 1  # Increment at layer 0
            cur_extend_step = _debug_extend_step
            
            # Only cuda env uses fuse ssm_states update
            # For K-last layout, we need to handle V-last kernel with K-last pool
            if self.ssm_k_last:
                batch_size = cache_indices.shape[0]
                # Check if any request has prefix cache (non-zero initial state)
                has_prefix_cache = (forward_batch.extend_prefix_lens > 0).any()
                
                if has_prefix_cache:
                    # Has prefix cache: need to read and transpose existing state
                    ssm_input = ssm_states[cache_indices]  # [batch, HV, V, K]
                    _log_tensor("extend", cur_extend_step, layer_id, "ssm_input", ssm_input, self.ssm_k_last, transpose_to_vlast=True)
                    recurrent_state = ssm_input.transpose(-1, -2).contiguous()  # [batch, HV, K, V]
                    sequential_indices = torch.arange(batch_size, device=cache_indices.device, dtype=cache_indices.dtype)
                    recurrent_state_indices_args = {"initial_state_indices": sequential_indices}
                else:
                    # No prefix cache: initial state is all zeros
                    # Optimization: pass ssm_states directly, kernel writes in-place
                    # Since initial is 0, reading with wrong stride is fine (0 == 0)
                    # After kernel, transpose the modified slots to fix semantics
                    # Note: This works because K == V (stride values are identical)
                    _log_tensor("extend", cur_extend_step, layer_id, "ssm_input", ssm_states[cache_indices], self.ssm_k_last, transpose_to_vlast=True)
                    recurrent_state = ssm_states  # Direct reference, no copy
                    recurrent_state_indices_args = {"initial_state_indices": cache_indices}
            else:
                ssm_input = ssm_states[cache_indices] if ssm_states.dim() > 3 else ssm_states
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_input", ssm_input, self.ssm_k_last)
                recurrent_state = ssm_states
                recurrent_state_indices_args = {"initial_state_indices": cache_indices}
                if is_npu():
                    recurrent_state = ssm_states[cache_indices]
                    recurrent_state_indices_args = {}

            _log_tensor("extend", cur_extend_step, layer_id, "query", query, self.ssm_k_last)
            _log_tensor("extend", cur_extend_step, layer_id, "value", value, self.ssm_k_last)
            
            # DEBUG: Log cache_indices for extend (only on rank 0)
            if _debug_enabled and layer_id == 0:
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        import json
                        open('/lustre/raplab/client/xutingz/workspace/gdn_log/sglang_debug.log', 'a').write(json.dumps({
                            "phase": "extend", "step": cur_extend_step, "layer": layer_id,
                            "name": "cache_indices_info", "k_last": self.ssm_k_last,
                            "cache_indices": cache_indices[:min(5, len(cache_indices))].tolist(),
                        }) + '\n')
                except:
                    pass
            
            core_attn_out, last_recurrent_state, h = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                **recurrent_state_indices_args,
            )
            
            _log_tensor("extend", cur_extend_step, layer_id, "core_attn_out", core_attn_out, self.ssm_k_last)

            if self.ssm_k_last:
                # V-last -> K-last: transpose kernel output back to pool layout
                # Note: INPLACE_UPDATE=True in kernel, so recurrent_state is updated in-place
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_updated", recurrent_state, self.ssm_k_last)
                if recurrent_state is ssm_states:
                    # Optimization path: kernel wrote directly to ssm_states with V-last semantics
                    # Need to fix by reading, transposing, and writing back
                    # Since K == V, stride values are identical, only semantics differ
                    ssm_states[cache_indices] = ssm_states[cache_indices].transpose(-1, -2).contiguous()
                else:
                    # Standard path: recurrent_state is a separate V-last buffer
                    ssm_states[cache_indices] = recurrent_state.to(ssm_states.dtype, copy=False).transpose(-1, -2)
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_writeback", ssm_states[cache_indices], self.ssm_k_last, transpose_to_vlast=True)
            elif is_npu():
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_updated", last_recurrent_state, self.ssm_k_last)
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_writeback", ssm_states[cache_indices], self.ssm_k_last)
            else:
                # CUDA V-last mode: kernel updates in-place via cache_indices
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_updated", ssm_states[cache_indices], self.ssm_k_last)
                _log_tensor("extend", cur_extend_step, layer_id, "ssm_writeback", ssm_states[cache_indices], self.ssm_k_last)

            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, forward_metadata
            )

        return core_attn_out


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_id: int,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        return mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        accepted_steps: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        request_number = accepted_steps.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )
        intermediate_state_indices = torch.arange(
            request_number, dtype=torch.int32, device=state_indices_tensor.device
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]

        # Compute common indices once to avoid duplication
        valid_mask = accepted_steps >= 0
        dst_state_indices = state_indices_tensor[valid_mask].to(torch.int64)  # [N]
        src_state_indices = intermediate_state_indices[valid_mask].to(
            torch.int64
        )  # [N]
        last_steps = accepted_steps[valid_mask].to(torch.int64)  # [N]

        # scatter into ssm_states at the chosen cache lines
        ssm_states[:, dst_state_indices, :] = intermediate_state_cache[
            :, src_state_indices, last_steps
        ].to(ssm_states.dtype, copy=False)

        # Scatter into conv_states at the chosen cache lines
        conv_states[:, dst_state_indices, :] = intermediate_conv_window_cache[
            :, src_state_indices, last_steps
        ].to(conv_states.dtype, copy=False)

        # Track indices used for tracking mamba states for prefix cache
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            track_mask = mamba_steps_to_track >= 0
            track_steps = mamba_steps_to_track[track_mask].to(torch.int64)  # [N]
            if track_steps.numel() == 0:
                # No track indices to update
                return
            dst_track_indices = mamba_track_indices[track_mask].to(torch.int64)
            src_track_indices = intermediate_state_indices[track_mask].to(torch.int64)

            # scatter into ssm_states at the chosen track states
            ssm_states[:, dst_track_indices, :] = intermediate_state_cache[
                :, src_track_indices, track_steps
            ].to(ssm_states.dtype, copy=False)

            # scatter into conv_states at the chosen track states
            conv_states[:, dst_track_indices, :] = intermediate_conv_window_cache[
                :, src_track_indices, track_steps
            ].to(conv_states.dtype, copy=False)
