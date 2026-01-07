from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

logger = logging.getLogger(__name__)

# Note: Sparse bias correction is used instead of dense PyTorch fallback
# This allows attention steering to work efficiently at any sequence length
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor


class TritonAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            build_unified_kv_indices,
            extend_attention_fwd,
            extend_attention_fwd_unified,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        if (
            model_runner.hybrid_gdn_config is not None
            or model_runner.kimi_linear_config is not None
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits

        # Decide whether enable deterministic inference with batch-invariant operations
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        # Configure deterministic inference settings
        if self.enable_deterministic:
            # Use fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            # Set static_kv_splits to False to use deterministic logic instead
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        # Check arguments
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # Initialize buffers
        # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                # When provided a buffer, create a clone for the second buffer
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        # Initialize forward metadata
        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        # Legacy dynamic splitting logic (non-deterministic)
        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        # deterministic
        if self.split_tile_size is not None and self.enable_deterministic:
            # expand seq_lens to match num_token
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = (
                expanded_seq_lens + self.split_tile_size - 1
            ) // self.split_tile_size
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                # Sliding window
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool_allocator,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int64, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets is used to calculate the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        elif forward_batch.forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            kv_indices = kv_indices.to(torch.int64)
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.accept_length_cpu)`.
            # It might have been forgotten to update somewhere.
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                sum(forward_batch.extend_prefix_lens_cpu),
                dtype=torch.int64,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            # Sliding window
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indptr, window_kv_indices, _, _ = (
                    update_sliding_window_buffer(
                        self.window_kv_indptr,
                        self.req_to_token,
                        self.sliding_window_size,
                        forward_batch.extend_prefix_lens,
                        forward_batch.req_pool_indices,
                        bs,
                        self.device,
                        self.token_to_kv_pool_allocator,
                    )
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            num_kv_splits = None

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

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
        assert encoder_lens is None, "Not supported"
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indptr, window_kv_indices, _, _ = (
                        update_sliding_window_buffer_cuda_graph(
                            self.window_kv_indptr,
                            window_kv_indices,
                            self.req_to_token,
                            self.sliding_window_size,
                            seq_lens[:bs],
                            req_pool_indices,
                            bs,
                            self.token_to_kv_pool_allocator,
                        )
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            attn_lse = self.cuda_graph_attn_lse
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )

            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            custom_mask = None
            mask_indptr = None
            max_extend_len = num_tokens_per_bs
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
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
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                num_token = bs
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    _, _, window_kv_lens, _ = update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices[:bs],
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                    self.get_num_kv_splits(
                        window_num_kv_splits[:num_token], window_kv_lens[:bs]
                    )

            else:
                assert False, "Multi-step cuda graph init is not done here."
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])

        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                _, _, window_kv_lens, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        elif forward_mode.is_draft_extend(include_v2=True):
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        # Save KV cache first (must do this before unified kernel)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        # Deterministic mode: use unified 1-stage kernel
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Normal mode: use original 2-stage kernel
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            else:
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(
                    forward_batch.spec_info, "draft_token_num"
                ):
                    extend_prefix_lens = (
                        forward_batch.seq_lens[:bs]
                        - forward_batch.spec_info.draft_token_num
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens
                else:
                    window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        # Build unified kv_indices using fused Triton kernel
        extend_kv_indices = forward_batch.out_cache_loc

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        if forward_batch.extend_start_loc is None:
            # Compute extend_start_loc from extend_seq_lens
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        # Check if this layer has attention biases for steering
        has_sparse_biases = forward_batch.has_attention_biases_for_layer(layer.layer_id)
        sparse_biases = None
        if has_sparse_biases:
            sparse_biases = forward_batch.get_sparse_attention_biases(layer.layer_id)

        # Always run the fast Triton kernel first
        # If sparse biases exist, we'll apply correction afterward
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
        )

        # Apply sparse bias correction if biases exist
        # This is O(|S| * head_dim) where |S| is number of biased positions
        # Much faster than the old dense PyTorch fallback for sparse biases
        if sparse_biases is not None:
            batch_indices, token_positions, bias_values = sparse_biases
            self._apply_sparse_bias_correction(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                kv_indptr,
                kv_indices,
                layer.scaling,
                self.forward_metadata.attn_lse,
                self.forward_metadata.num_kv_splits,
                batch_indices,
                token_positions,
                bias_values,
                logit_cap=logits_soft_cap,
            )

        # Compute top-k attention tokens for interpretability
        # Multi-layer capture: compute on all specified layers
        if (
            forward_batch.capture_attention_tokens
            and forward_batch.attention_capture_layer_ids
            and layer.layer_id in forward_batch.attention_capture_layer_ids
        ):
            self._compute_attention_token_info(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                kv_indptr,
                kv_indices,
                layer.scaling,
                forward_batch,
                layer.layer_id,
            )

        return o

    def _apply_sparse_bias_correction(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        o: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
        attn_lse: torch.Tensor,
        num_kv_splits: torch.Tensor,
        batch_indices: torch.Tensor,
        token_positions: torch.Tensor,
        bias_values: torch.Tensor,
        logit_cap: float = 0.0,
    ):
        """
        Apply sparse bias correction to unbiased attention output.

        This is much more efficient than the dense PyTorch fallback for sparse biases.
        Math: Given unbiased output O0 with LSE0, and sparse biases b_j for j∈S:
          - Z0 = exp(LSE0)
          - Z = Z0 + Σ_{j∈S} exp(l_j) * (exp(b_j)-1)
          - O = (O0*Z0 + Σ_{j∈S} exp(l_j) * v_j * (exp(b_j)-1)) / Z

        Complexity: O(|S| * head_dim) per head per token, where |S| is number of biased positions.

        Args:
            q: [batch, num_heads, head_dim] query
            k_buffer: [max_tokens, num_kv_heads, head_dim] key cache
            v_buffer: [max_tokens, num_kv_heads, head_dim] value cache
            o: [batch, num_heads, head_dim] output tensor (modified in-place)
            kv_indptr: [batch + 1] CSR format indptr for KV cache
            kv_indices: [total_kv_tokens] KV cache locations
            sm_scale: attention scale factor
            attn_lse: [batch, num_heads, max_kv_splits] log-sum-exp from Triton kernel
            num_kv_splits: [batch] number of splits per batch element
            batch_indices: [num_biases] batch index for each bias
            token_positions: [num_biases] token position for each bias
            bias_values: [num_biases] bias values
            logit_cap: optional logit capping value (0 = disabled)
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        num_kv_heads = k_buffer.shape[1]
        head_dim = q.shape[2]
        v_head_dim = v_buffer.shape[2]

        # GQA: heads per KV head
        heads_per_kv = num_heads // num_kv_heads

        # Compute final LSE by reducing across splits (log-sum-exp reduction)
        # attn_lse: [batch, num_heads, max_kv_splits]
        # We need to reduce to [batch, num_heads]
        max_splits = attn_lse.shape[2]
        final_lse = torch.full(
            (batch_size, num_heads), float('-inf'),
            dtype=attn_lse.dtype, device=attn_lse.device
        )
        for b in range(batch_size):
            n_splits = num_kv_splits[b].item()
            if n_splits > 0:
                # Log-sum-exp reduction: LSE = log(sum(exp(lse_i)))
                final_lse[b] = torch.logsumexp(attn_lse[b, :, :n_splits], dim=-1)

        # Z0 = exp(LSE0) - the normalization constant
        z0 = torch.exp(final_lse)  # [batch, num_heads]

        # Group biases by batch index for efficient processing
        unique_batches = batch_indices.unique()

        for b_idx in unique_batches:
            b = b_idx.item()

            # Get biases for this batch element
            mask = batch_indices == b_idx
            positions = token_positions[mask]
            biases = bias_values[mask]

            if len(positions) == 0:
                continue

            # Get KV cache range for this batch
            kv_start = kv_indptr[b].item()
            kv_end = kv_indptr[b + 1].item()
            seq_len = kv_end - kv_start

            if seq_len == 0:
                continue

            # Filter positions that are within sequence length
            valid_mask = positions < seq_len
            if not valid_mask.any():
                continue

            positions = positions[valid_mask]
            biases = biases[valid_mask]

            # Get KV cache locations for biased positions
            kv_locs = kv_indices[kv_start:kv_end]
            biased_kv_locs = kv_locs[positions]

            # Gather K, V for biased positions only
            k_biased = k_buffer[biased_kv_locs]  # [num_biased, num_kv_heads, head_dim]
            v_biased = v_buffer[biased_kv_locs]  # [num_biased, num_kv_heads, v_head_dim]

            # Get query for this batch
            q_b = q[b]  # [num_heads, head_dim]

            # exp(b_j) - 1 for each biased position
            exp_bias_minus_1 = torch.exp(biases) - 1  # [num_biased]

            # Process each KV head group
            for kv_h in range(num_kv_heads):
                k_h = k_biased[:, kv_h, :]  # [num_biased, head_dim]
                v_h = v_biased[:, kv_h, :]  # [num_biased, v_head_dim]

                # Process all heads that share this KV head
                for h_offset in range(heads_per_kv):
                    h = kv_h * heads_per_kv + h_offset
                    q_h = q_b[h]  # [head_dim]

                    # Compute logits for biased positions: l_j = q @ k_j * scale
                    logits = torch.matmul(k_h, q_h) * sm_scale  # [num_biased]

                    # Apply logit capping if enabled
                    if logit_cap > 0:
                        logits = logit_cap * torch.tanh(logits / logit_cap)

                    # exp(l_j) for biased positions
                    exp_logits = torch.exp(logits)  # [num_biased]

                    # Correction terms:
                    # weight_delta_j = exp(l_j) * (exp(b_j) - 1)
                    weight_delta = exp_logits * exp_bias_minus_1  # [num_biased]

                    # Z_delta = sum of weight deltas
                    z_delta = weight_delta.sum()

                    # Value correction: sum of weight_delta_j * v_j
                    # [num_biased] @ [num_biased, v_head_dim] -> [v_head_dim]
                    v_correction = torch.matmul(weight_delta, v_h)

                    # Apply correction: O = (O0 * Z0 + v_correction) / (Z0 + Z_delta)
                    z0_h = z0[b, h]
                    o0_h = o[b, h].clone()

                    # New output
                    o[b, h] = (o0_h * z0_h + v_correction) / (z0_h + z_delta)

    def _decode_attention_with_bias_pytorch(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        o: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
        attention_bias: torch.Tensor,
        logit_cap: float = 0.0,
    ):
        """
        PyTorch fallback for decode attention with attention bias.

        DEPRECATED: Use _apply_sparse_bias_correction instead for better performance.
        This is kept for backwards compatibility and very dense bias patterns.

        Args:
            q: [batch, num_heads, head_dim]
            k_buffer: [max_tokens, num_kv_heads, head_dim]
            v_buffer: [max_tokens, num_kv_heads, head_dim]
            o: [batch, num_heads, head_dim] output tensor
            kv_indptr: [batch + 1] CSR format indptr
            kv_indices: [total_kv_tokens] KV cache locations
            sm_scale: attention scale factor
            attention_bias: [batch, max_seq_len] additive bias tensor
            logit_cap: optional logit capping value (0 = disabled)
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        num_kv_heads = k_buffer.shape[1]
        head_dim = q.shape[2]

        # GQA: heads per KV head
        heads_per_kv = num_heads // num_kv_heads

        for b in range(batch_size):
            # Get KV cache indices for this batch element
            kv_start = kv_indptr[b].item()
            kv_end = kv_indptr[b + 1].item()
            seq_len = kv_end - kv_start

            if seq_len == 0:
                continue

            # Gather K, V from cache
            kv_locs = kv_indices[kv_start:kv_end]
            k = k_buffer[kv_locs]  # [seq_len, num_kv_heads, head_dim]
            v = v_buffer[kv_locs]  # [seq_len, num_kv_heads, head_dim]

            # Get query for this batch element
            q_b = q[b]  # [num_heads, head_dim]

            # Get bias for this batch element (truncate to seq_len)
            bias_b = attention_bias[b, :seq_len]  # [seq_len]

            # Compute attention per head (or per KV head group for GQA)
            for kv_h in range(num_kv_heads):
                k_h = k[:, kv_h, :]  # [seq_len, head_dim]
                v_h = v[:, kv_h, :]  # [seq_len, head_dim]

                # Process all heads that share this KV head
                for h_offset in range(heads_per_kv):
                    h = kv_h * heads_per_kv + h_offset
                    q_h = q_b[h]  # [head_dim]

                    # Compute attention logits: q @ k^T
                    attn_logits = torch.matmul(q_h, k_h.t()) * sm_scale  # [seq_len]

                    # Apply logit capping if enabled
                    if logit_cap > 0:
                        attn_logits = logit_cap * torch.tanh(attn_logits / logit_cap)

                    # Add steering bias before softmax
                    attn_logits = attn_logits + bias_b

                    # Softmax
                    attn_weights = F.softmax(attn_logits, dim=-1)  # [seq_len]

                    # Compute output: weights @ v
                    o[b, h, :] = torch.matmul(attn_weights, v_h)  # [head_dim]

    def _compute_attention_token_info(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
        forward_batch,
        layer_id: int,
    ):
        """
        Compute top-k attention tokens for interpretability.

        Supports two modes:
        1. Debug Mode: Returns raw indices/scores (for visualization)
        2. Fingerprint Mode: Returns 20D feature vector (for production routing)

        Uses memory-efficient chunked approach:
        - For small sequences (<2K tokens): Direct PyTorch computation
        - For large sequences: Chunked Triton kernel

        Memory usage: O(batch × heads × num_chunks) instead of O(batch × heads × seq_len)
        For 1M context: ~125KB vs ~256MB
        """
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
            compute_fingerprint_gpu,
            compute_fingerprint_features,
            classify_manifold_gpu,
        )
        from sglang.srt.model_executor.forward_batch_info import AttentionTokenInfo

        try:
            # Validate inputs to prevent CUDA device-side asserts
            batch_size = q.shape[0]
            if batch_size == 0:
                return  # Nothing to do for empty batch

            # Synchronize to catch any pending CUDA errors before we start
            import torch
            torch.cuda.synchronize()

            # Check kv_indptr has correct size
            if kv_indptr.shape[0] != batch_size + 1:
                import logging
                logging.getLogger(__name__).warning(
                    f"Skipping attention capture: kv_indptr size mismatch "
                    f"(expected {batch_size + 1}, got {kv_indptr.shape[0]})"
                )
                return

            # Validate kv_indices bounds to prevent out-of-bounds access
            max_kv_idx = kv_indptr[-1].item()
            if max_kv_idx > 0 and kv_indices.shape[0] > 0:
                if max_kv_idx > kv_indices.shape[0]:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Skipping attention capture: kv_indptr max ({max_kv_idx}) > "
                        f"kv_indices size ({kv_indices.shape[0]})"
                    )
                    return

                # Check kv_indices values are within k_buffer bounds (sync first to get accurate values)
                indices_to_check = kv_indices[:max_kv_idx]
                max_idx = indices_to_check.max().item()
                min_idx = indices_to_check.min().item()
                if max_idx >= k_buffer.shape[0] or min_idx < 0:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Skipping attention capture: kv_indices out of bounds "
                        f"(min={min_idx}, max={max_idx}, k_buffer_size={k_buffer.shape[0]})"
                    )
                    return

            topk_scores, topk_indices, topk_logits, logsumexp_candidates = compute_topk_attention_chunked(
                q,
                k_buffer,
                kv_indptr,
                kv_indices,
                sm_scale,
                top_k=forward_batch.attention_top_k,
                chunk_size=forward_batch.attention_chunk_size,
                window=forward_batch.attention_window,
            )

            # Compute fingerprint if fingerprint mode is enabled (production path)
            # This computes the 20D feature vector ON GPU, avoiding CPU export bottleneck
            if getattr(forward_batch, 'attention_fingerprint_mode', False):
                # Get current positions for fingerprint computation
                # Use seq_lens as current positions (decode position)
                current_pos = forward_batch.seq_lens.clone()

                # Compute fingerprint: 16-bin log-distance histogram
                fingerprint = compute_fingerprint_gpu(
                    topk_indices, topk_scores, current_pos
                )
                # Extract 20D feature vector for clustering
                features = compute_fingerprint_features(fingerprint)
                # Classify manifold zone
                manifolds, _ = classify_manifold_gpu(fingerprint)

                # Store fingerprint (replaces raw indices in production)
                # Only store once (last layer typically) to avoid redundancy
                if forward_batch.attention_fingerprint is None:
                    forward_batch.attention_fingerprint = features
                    forward_batch.attention_manifold = manifolds

            # Store raw attention info (for debug mode / backward compatibility)
            info = AttentionTokenInfo(
                token_positions=topk_indices,
                attention_scores=topk_scores,
                topk_logits=topk_logits,
                logsumexp_candidates=logsumexp_candidates,
                layer_id=layer_id,
            )

            # Store in multi-layer dict
            if forward_batch.attention_token_infos is not None:
                forward_batch.attention_token_infos[layer_id] = info

            # Also set single-layer field for backward compatibility
            # (keeps the last layer's info, which is typically the most useful)
            forward_batch.attention_token_info = info

            # Synchronize to catch any CUDA errors before they propagate
            torch.cuda.synchronize()

        except (RuntimeError, torch.cuda.CudaError) as e:
            # Catch CUDA errors specifically - they can be fatal
            import logging
            logging.getLogger(__name__).error(
                f"CUDA error during attention capture (disabling for this request): {e}"
            )
            # Try to recover by clearing CUDA error state
            try:
                torch.cuda.synchronize()
            except:
                pass
        except Exception as e:
            # Don't let attention capture errors break inference
            import logging
            import traceback
            logging.getLogger(__name__).warning(
                f"Failed to compute attention tokens: {e}\n{traceback.format_exc()}"
            )


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int64,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, None, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        self.common_template(forward_batch, None, None)

        # NOTE: Multi-step's attention backends use the slice of
        # - kv_indptr buffer (cuda graph and non-cuda graph)
        # - kv_indices buffer (cuda graph only)
        # So we don't need to assign the KV indices inside the attention backend.

        # Compute num_kv_splits only once
        num_token = forward_batch.batch_size * self.topk
        self.attn_backends[-1].get_num_kv_splits(
            self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
            forward_batch.seq_lens[:bs],
        )


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int64, device=device
    )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx


def update_sliding_window_buffer_cuda_graph(
    window_kv_indptr,
    window_kv_indices,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
