from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu
from sgl_kernel_npu.attention.sinks_attention import (
    attention_sinks_prefill_triton,
    attention_sinks_triton,
)

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
)
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_kv_cache
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var, get_current_device_stream_fast

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import logging

import numpy as np

logger = logging.getLogger(__name__)
FULL_ATTENTION_WINDOW = 2147483647


def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -> torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # mapped block_tables for swa
    block_tables_swa: Optional[torch.Tensor] = None

    # pre-translated full->SWA write target for SWAKVPool.set_kv_buffer
    swa_out_cache_loc: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_q_pa: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

    # swa attention mask for graph mode decode
    swa_mask: Optional[torch.Tensor] = None

    # prefix cache
    prefix_lens: Optional[torch.Tensor] = None
    flatten_prefix_block_tables: Optional[torch.Tensor] = None


class AscendAttnMaskBuilder:
    def __init__(self, model_runner: ModelRunner, device, use_fia, use_mla):
        """
        Initialize the AscendAttnMaskBuilder class.

        :param model_runner: ModelRunner instance for model execution.
        :param device: Device to run the model on (e.g., 'cuda', 'npu').
        :param use_fia: Boolean flag to indicate if environment variable ASCEND_USE_FIA is set to 1.
        """
        self.use_fia = use_fia
        self.model_runner = model_runner
        self.device = device

        # Initialize mask
        mask_len = 128
        self.mask = self.generate_attn_mask(mask_len, "norm", model_runner.dtype).to(
            self.device
        )

        # Initialize FIA mask
        fia_mask_len = 2048
        self.fia_mask = self.generate_mask_flag(fia_mask_len).to(self.device)

        # Initialize MTP mask
        mtp_mask_len = 2048
        self.mtp_mask = self.generate_mask_flag(mtp_mask_len).to(self.device)

        # Initialize mixed chunk mask cache
        mixed_mask_len = 2048
        self.mixed_chunk_attn_mask = self.get_splitfuse_attn_mask(mixed_mask_len)

        if use_mla:
            # Initialize RingMla mask
            ringmla_mask_len = 512
            self.ringmla_mask = self.generate_attn_mask(
                ringmla_mask_len, "norm", torch.bfloat16
            ).to(self.device)

    @staticmethod
    def generate_mask_flag(max_seq_len):
        """
        Generate a mask flag for attention masks.

        :param max_seq_len: Maximum sequence length for the mask.
        :return: A boolean tensor representing the mask flag.
        """
        # Construct lower triangle matrix.
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Create upper triangle matrix used to mark mask positions.
        mask_flag = ~mask_flag
        return mask_flag

    @staticmethod
    def generate_attn_mask(max_seq_len, mode, dtype=torch.float16):
        """
        Generate an attention mask.

        :param max_seq_len: Maximum sequence length for the mask.
        :param mode: Mode of the mask ('mix' or 'norm').
        :param dtype: Data type of the mask tensor.
        :return: A tensor representing the attention mask.
        """
        mask_flag = AscendAttnMaskBuilder.generate_mask_flag(max_seq_len)
        if mode == "mix":
            mask_value = (
                float("-inf") if dtype in [torch.float16, torch.bfloat16] else 1
            )
        else:
            mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
        attn_mask = (
            torch.zeros(size=(max_seq_len, max_seq_len))
            .masked_fill_(mask_flag, mask_value)
            .to(dtype)
        )
        return attn_mask

    @staticmethod
    def get_attention_mask_id(seq_lens, extend_lens):
        """
        Generate attention mask IDs based on sequence lengths and extended lengths.

        :param seq_lens: Sequence lengths.
        :param extend_lens: Extended lengths.
        :return: A tensor containing the attention mask IDs.
        """
        starts = seq_lens - extend_lens
        ends = seq_lens

        # Use torch.stack to stack the start and end indices together
        ranges = torch.stack((starts, ends), dim=-1)

        # Use list comprehension to generate tensors for each range and concatenate them
        attn_mask_id = torch.cat([torch.arange(start, end) for start, end in ranges])
        return attn_mask_id

    def update_attn_cache(
        self,
        seqlen: int,
        mask_cache: torch.Tensor,
        seq_len_cached: int,
        dtype: torch.dtype,
        mode,
    ):
        """
        Update the attention mask cache.

        :param seqlen: Maximum sequence length.
        :param mask_cache: Current attention mask cache.
        :param seq_len_cached: Cached sequence length.
        :param dtype: Data type of the mask tensor.
        :param mode: Mode of the mask ('mix' or 'norm').
        :return: Updated mask cache and sequence length cache.
        """
        if seqlen > seq_len_cached:
            seq_len_cached = seqlen
            mask_cache = self.generate_attn_mask(seqlen, mode, dtype)
        if mask_cache.dtype != dtype:
            mask_cache = mask_cache.to(dtype)
        return mask_cache, seq_len_cached

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate a splitfuse attention mask.

        :param seq_lens: Sequence lengths.
        :return: A tensor representing the splitfuse attention mask.
        """
        attn_mask = (
            torch.triu(torch.ones(seq_lens, seq_lens), diagonal=1)
            .to(torch.int8)
            .to(self.device)
        )
        return attn_mask

    def get_swa_mask(self, seq_lens: torch.Tensor, s2: int, left_context=512):
        if seq_lens.dim() == 1:
            seq_lens = seq_lens.unsqueeze(1)
        b = seq_lens.size(0)
        device = seq_lens.device
        indices = torch.arange(s2, device=device).unsqueeze(0).expand(b, -1)
        start_indices = torch.clamp(seq_lens - left_context, min=0)
        mask = (indices < start_indices) | (indices >= seq_lens)
        return mask.unsqueeze(1).to(self.device, non_blocking=True)


def _cp_allgather_and_save_kv_npu(
    forward_batch, layer, k, v, cp_size, token_to_kv_pool, swa_loc=None
):
    """NPU-compatible CP KV all-gather with merged K/V communication.

    Merges K and V along the feature dimension so only one all-gather is
    needed instead of two, halving communication latency.

    k shape: [S_local, tp_k_head_num, qk_head_dim]
    v shape: [S_local, tp_v_head_num, v_head_dim]

    Equivalent to cp_allgather_and_save_kv_cache() in cp_utils.py, but uses
    a single all-gather for both K and V.

    swa_loc is the pre-translated full->SWA write target for hybrid SWA pools
    (None for non-SWA pools); set_kv_buffer never translates internally.
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    # Save original trailing shapes for reshape after gather.
    k_tail = k.shape[1:]  # (tp_k_head_num, qk_head_dim)
    v_tail = v.shape[1:]  # (tp_v_head_num, v_head_dim)

    # Flatten trailing dims then concat → one all-gather instead of two.
    # Works for GQA where tp_k_head_num != tp_v_head_num.
    k_flat = k.contiguous().reshape(k.shape[0], -1)  # [S_local, k_feat]
    v_flat = v.contiguous().reshape(v.shape[0], -1)  # [S_local, v_feat]
    k_feat_size = k_flat.shape[-1]
    kv_flat = torch.cat([k_flat, v_flat], dim=-1)  # [S_local, k_feat + v_feat]

    kv_full = cp_all_gather_rerange_kv_cache(
        kv_flat, cp_size, forward_batch, get_current_device_stream_fast()
    )  # [S_full, k_feat + v_feat]

    key_cache_full = kv_full[..., :k_feat_size].reshape(-1, *k_tail)
    value_cache_full = kv_full[..., k_feat_size:].reshape(-1, *v_tail)

    token_to_kv_pool.set_kv_buffer(
        layer,
        KVWriteLoc(cache_loc, swa_loc),
        key_cache_full,
        value_cache_full,
    )


class AscendAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.speculative_step_id = speculative_step_id
        self.speculative_step_offset_npu = torch.tensor(
            speculative_step_id + 1, device="npu"
        )
        self.page_size = model_runner.page_size
        self.model_dtype = model_runner.model_config.dtype
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            if (
                "MiniCPM3ForCausalLM"
                in model_runner.model_config.hf_config.architectures
            ):
                self.qk_nope_head_dim = (
                    model_runner.model_config.hf_config.qk_nope_head_dim
                )
            else:
                self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.q_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        else:
            self.use_alibi = getattr(model_runner.model_config, "use_alibi", False)
            if (
                "Gemma2ForSequenceClassification"
                in model_runner.model_config.hf_config.architectures
            ):
                self.use_native_sdpa = True
        self.native_attn = AscendTorchNativeAttnBackend()
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fa = get_bool_env_var("ASCEND_USE_FA", "False")
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.ascend_attn_mask_builder = AscendAttnMaskBuilder(
            model_runner, self.device, self.use_fia, self.use_mla
        )
        self.mask, self.fia_mask, self.mtp_mask, self.mix_mask = (
            self.ascend_attn_mask_builder.mask,
            self.ascend_attn_mask_builder.fia_mask,
            self.ascend_attn_mask_builder.mtp_mask,
            self.ascend_attn_mask_builder.mixed_chunk_attn_mask,
        )
        if self.use_mla:
            self.ringmla_mask = self.ascend_attn_mask_builder.ringmla_mask
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        if self.is_hybrid_swa:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
            self.sliding_window_size = model_runner.sliding_window_size
        self.use_sliding_window_kv_pool = (
            isinstance(self.token_to_kv_pool, SWAKVPool)
            and self.token_to_kv_pool.swa_layer_nums > 0
        )

        # head num padding
        self.padding_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
        self.q_head_num_padding = None
        if hasattr(model_runner.model_config, "num_attention_heads") and self.use_mla:
            self.tp_q_head_num = (
                model_runner.model_config.num_attention_heads // get_attention_tp_size()
            )
            for num in self.padding_size_list:
                if num >= self.tp_q_head_num:
                    self.q_head_num_padding = num
                    break

        # dllm model config
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = False
        if self.dllm_config is not None:
            self.is_dllm_model = True
            self.dllm_block_size = self.dllm_config.block_size

        self.attn_cp_size = model_runner.attn_cp_size

    def _is_swa_layer(self, layer: RadixAttention) -> bool:
        return (
            self.is_hybrid_swa
            and layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
        )

    @staticmethod
    def _can_use_tnd(layer: RadixAttention) -> bool:
        """Check if TND layout is supported."""
        d = layer.qk_head_dim
        v = layer.v_head_dim
        return (d == v and d in (128, 192, 256)) or (d == 192 and v == 128)

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        if in_capture:
            self._init_cuda_graph_metadata(
                bs,
                forward_batch.forward_mode,
                forward_batch.seq_lens,
                forward_batch.out_cache_loc,
            )
        self._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=(
                forward_batch.seq_lens.cpu()
                if in_capture
                else forward_batch.seq_lens_cpu
            ),
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=forward_batch.out_cache_loc,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        seq_lens_max = forward_batch.seq_lens.max()
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            seq_lens_max += self.speculative_step_id + 1
        self.forward_metadata.block_tables = (
            self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if self.is_hybrid_swa:
            self.forward_metadata.block_tables_swa = (
                (
                    self.full_to_swa_index_mapping[
                        self.req_to_token_pool.req_to_token[
                            forward_batch.req_pool_indices, :seq_lens_max
                        ]
                    ][:, :: self.page_size]
                    // self.page_size
                )
                .to(torch.int32)
                .contiguous()
            )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens = forward_batch.extend_seq_lens
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        if forward_batch.seq_lens is not None:
            self.forward_metadata.seq_lens = forward_batch.seq_lens.int()
        else:
            self.forward_metadata.seq_lens = forward_batch.seq_lens_cpu.to(
                self.device
            ).int()

        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            self.forward_metadata.seq_lens_cpu_int += self.speculative_step_id + 1

        if (
            self.use_mla
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and sum(forward_batch.extend_prefix_lens_cpu) > 0
        ):
            self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens.to(
                "cpu"
            )
            seq_prefix_lens = self.forward_metadata.prefix_lens.tolist()
            self.forward_metadata.flatten_prefix_block_tables = torch.empty(
                0, dtype=torch.int32
            ).to(self.device)
            for req_idx, seq_len in zip(
                forward_batch.req_pool_indices.tolist(), seq_prefix_lens
            ):
                req_indices = self.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = (
                    req_indices[:seq_len][:: self.page_size] // self.page_size
                )
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.forward_metadata.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        total_context_len = self.max_context_len + self.page_size - 1
        if self.speculative_num_draft_tokens is not None:
            total_context_len += self.speculative_num_draft_tokens
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, total_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }
        if self.is_hybrid_swa:
            self.graph_metadata["block_tables_swa"] = torch.empty(
                (max_bs, total_context_len // self.page_size),
                dtype=torch.int32,
                device=self.device,
            )
            # SWA mask: True = masked out (don't attend), False = attend.
            # Pre-allocated at max size, sliced per batch size during capture,
            # content updated via copy_() during replay.
            self.graph_metadata["swa_mask"] = torch.ones(
                (max_bs, 1, total_context_len),
                dtype=torch.bool,
                device=self.device,
            )
            # Pre-allocated index buffer for mask generation during replay,
            # avoids torch.arange allocation on every replay step.
            self.graph_metadata["swa_indices"] = torch.arange(
                total_context_len, device=self.device, dtype=torch.int32
            )
        if self.use_sliding_window_kv_pool:
            # refilled in place at replay; the captured graph reads this storage
            self.swa_out_cache_loc_buf = torch.zeros(
                max_num_tokens,
                dtype=torch.int64,
                device=self.device,
            )
        # V4-specific extra graph buffers. Default no-op on the base class;
        # DeepseekV4AscendAttnBackend overrides.
        self._init_dsv4_graph_buffers(max_bs=max_bs, max_num_tokens=max_num_tokens)

    def _init_dsv4_graph_buffers(self, *, max_bs: int, max_num_tokens: int) -> None:
        """Hook for V4-Flash to preallocate dsv4-specific graph buffers.

        Default no-op. Overridden by DeepseekV4AscendAttnBackend.
        """
        pass

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> ForwardMetadata:
        """Create and store the per-bs ForwardMetadata for CUDA graph capture."""
        metadata = ForwardMetadata()
        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        if self.is_hybrid_swa:
            metadata.block_tables_swa = self.graph_metadata["block_tables_swa"][:bs, :]
            metadata.swa_mask = self.graph_metadata["swa_mask"][:bs, :, :]
        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            num_tokens = out_cache_loc.shape[0]
            metadata.swa_out_cache_loc = self.swa_out_cache_loc_buf[:num_tokens]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens
                + bs * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )
        if forward_mode.is_dllm_extend():
            extend_seq_lens_cpu_int = torch.tensor(
                [self.dllm_block_size for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )
            metadata.seq_lens_list_cumsum = (
                torch.cumsum(extend_seq_lens_cpu_int, dim=0).int().tolist()
            )
        if (
            self.q_head_num_padding is not None
            and self.q_head_num_padding > self.tp_q_head_num
        ):
            dtype = self.model_dtype if self.model_dtype is not None else torch.bfloat16
            metadata.nope_padding = torch.empty(
                [
                    bs,
                    1,
                    self.q_head_num_padding - self.tp_q_head_num,
                    self.kv_lora_rank,
                ],
                dtype=dtype,
                device=seq_lens.device,
            )
            metadata.rope_padding = torch.empty(
                [
                    bs,
                    1,
                    self.q_head_num_padding - self.tp_q_head_num,
                    self.qk_rope_head_dim,
                ],
                dtype=dtype,
                device=seq_lens.device,
            )
        self.graph_metadata[bs] = metadata
        return metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.
        """
        metadata = self.graph_metadata[bs]

        # refill the captured SWA write-target buffer in place from the live loc
        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            n = out_cache_loc.shape[0]
            self.swa_out_cache_loc_buf[n:].zero_()
            self.swa_out_cache_loc_buf[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc)
            )
        max_len = seq_lens_cpu[:bs].max().item()
        if forward_mode.is_target_verify():
            max_len += self.speculative_num_draft_tokens
        elif forward_mode.is_decode_or_idle() and spec_info is not None:
            max_len += self.speculative_step_id + 1
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        if self.is_hybrid_swa:
            metadata.block_tables_swa[:bs, :max_seq_pages].copy_(
                self.full_to_swa_index_mapping[
                    self.req_to_token[
                        req_pool_indices[:bs], 0 : max_len : self.page_size
                    ]
                ]
                // self.page_size
            )
            metadata.block_tables_swa[:bs, max_seq_pages:].fill_(0)
            metadata.block_tables_swa[bs:, :].fill_(0)

            # Update SWA mask: True = masked out (don't attend), False = attend
            seq_lens_int = seq_lens_cpu[:bs].int()
            starts = torch.clamp(seq_lens_int - self.sliding_window_size, min=0)
            indices = self.graph_metadata["swa_indices"]
            start_exp = starts.unsqueeze(1).to(self.device)
            seq_exp = seq_lens_int.unsqueeze(1).to(self.device)
            mask = (indices.unsqueeze(0) < start_exp) | (
                indices.unsqueeze(0) >= seq_exp
            )
            metadata.swa_mask[:bs, 0, :].copy_(mask)
            metadata.swa_mask[bs:, :, :].fill_(True)
        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], 0 : max_len : self.page_size]
            // self.page_size
        )

        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        elif forward_mode.is_decode_or_idle() and spec_info is not None:
            seq_lens = seq_lens + self.speculative_step_offset_npu
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def _generate_alibi_bias(
        self,
        seq_len: int,
        slopes: torch.Tensor,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        position_point = (
            torch.arange(seq_len).view(1, 1, -1).expand(num_heads, -1, -1).to(device)
        )
        alibi = slopes.view(-1, 1, 1) * position_point
        alibi_bias = alibi.view(num_heads, 1, seq_len).to(device).to(dtype)
        return alibi_bias

    def generate_alibi_bias(
        self,
        q_seq_len: int,
        kv_seq_len: int,
        slopes: torch.Tensor,
        num_heads: int,
        device: torch.device,
        is_extend: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        MAX_LEN_ALB = 5000
        max_seq_len = max(kv_seq_len, q_seq_len, MAX_LEN_ALB)
        if getattr(self, "alibi_bias", None) is None:
            self.alibi_bias = self._generate_alibi_bias(
                max_seq_len, slopes, num_heads, device, dtype
            )

        if getattr(self, "super_mask", None) is None:
            super_mask = torch.ones(size=(1, max_seq_len, max_seq_len), dtype=dtype)
            super_mask = super_mask.float().fill_(float("-inf")).type_as(super_mask)
            super_mask = torch.triu(super_mask, 1).to(device)
            self.super_mask = super_mask
        if is_extend:
            return (
                self.alibi_bias[:, :q_seq_len, :kv_seq_len]
                + self.super_mask[:, :q_seq_len, :kv_seq_len]
            )
        else:
            return self.alibi_bias[:, :q_seq_len, :kv_seq_len]

    def attn_alibi(
        self,
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        query_lens,
        scale_value,
        num_heads,
        slopes,
        is_extend,
    ):
        curr = 0
        num_prompts = query_lens.shape[0]
        head_size = k_cache.shape[3]
        head_size_v = v_cache.shape[3]
        block_size = k_cache.shape[1]
        attn_output = []
        for i in range(num_prompts):
            seq_len = seq_lens[i].item()
            block_table = block_tables[i]

            j = torch.arange(seq_len, device=block_table.device)

            block_number = block_table[j // block_size]
            block_offset = j % block_size

            k = k_cache[block_number, block_offset]
            v = v_cache[block_number, block_offset]
            k = k.view(seq_len, num_heads, head_size)
            v = v.view(seq_len, num_heads, head_size_v)

            if is_extend:
                q_len = query_lens[i].item()
                query = q[curr : curr + q_len]
            else:
                q_len = 1
                query = q[curr : curr + 1]

            query = query.to(torch.float32)
            query = query * scale_value
            query = query.permute(1, 0, 2)
            k = k.permute(1, 2, 0)

            score = torch.bmm(query, k)
            score = score.to(torch.float32)
            if slopes is not None:
                alibi_bias = self.generate_alibi_bias(
                    q_seq_len=q_len,
                    kv_seq_len=seq_len,
                    slopes=slopes,
                    num_heads=num_heads,
                    device=q.device,
                    is_extend=is_extend,
                    dtype=query.dtype,
                )
                score = score + alibi_bias
            score = torch.max(score, torch.tensor(torch.finfo(score.dtype).min))
            p = torch.nn.functional.softmax(score, dim=-1)
            v = v.permute(1, 0, 2)
            out = torch.bmm(p, v)
            out = out.permute(1, 0, 2)
            out = out.reshape(-1, num_heads * head_size_v)
            attn_output.append(out)
            curr += q_len
        attn_output = torch.cat(attn_output, dim=0).to(q.dtype).to(q.device)
        attn_output = attn_output.view(-1, num_heads * head_size)
        return attn_output

    def do_cp_balance_attn(
        self,
        q_nope,
        k_nope,
        q_pe,
        k_pe,
        topk_indices,
        layer,
        actual_seq_qlen,
        actual_seq_lengths_kv,
    ):
        seq_len = q_nope.shape[0]
        split_len = (seq_len + 1) // 2
        q_nope_prev, q_nope_next = torch.split(q_nope, split_len, dim=0)
        q_rope_prev, q_rope_next = torch.split(q_pe, split_len, dim=0)
        q_nope_prev = q_nope_prev.contiguous()
        q_nope_next = q_nope_next.contiguous()
        q_rope_prev = q_rope_prev.contiguous()
        q_rope_next = q_rope_next.contiguous()
        topk_indices_prev, topk_indices_next = topk_indices

        actual_seq_qlen_prev, actual_seq_qlen_next = actual_seq_qlen
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        attn_out_prev, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope_prev,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope_prev,
            key_rope=k_pe,
            sparse_indices=topk_indices_prev,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen_prev.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=actual_seq_lengths_kv_prev.to(
                device=q_nope.device, dtype=torch.int32
            ),
            block_table=self.forward_metadata.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
        attn_out_next, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope_next,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope_next,
            key_rope=k_pe,
            sparse_indices=topk_indices_next,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen_next.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=actual_seq_lengths_kv_next.to(
                device=q_nope.device, dtype=torch.int32
            ),
            block_table=self.forward_metadata.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
        return torch.cat([attn_out_prev, attn_out_next], dim=0)

    def do_cp_attn_fia(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """CP-aware attention for standard (non-MLA) models using FIA on Ascend NPU.

        Uses npu_fused_infer_attention_score with paged KV cache (block_table).
        The KV cache must already contain the full gathered sequence
        (written by _cp_allgather_and_save_kv_npu before this call).

        Args:
            q:            Query tensor, shape [total_q_tokens, tp_q_head_num * qk_head_dim]
            k_cache:      Full key cache from token_to_kv_pool
            v_cache:      Full value cache from token_to_kv_pool
            layer:        RadixAttention layer
            forward_batch: ForwardBatch with attn_cp_metadata populated

        Returns:
            attn_output [total_q_tokens, tp_q_head_num * v_head_dim]
        """
        cp_meta = forward_batch.attn_cp_metadata

        # Local tokens are laid out [all_seqs_prev, all_seqs_next]; split at
        # total_q_prev_tokens rather than the midpoint to support bs > 1.
        split = cp_meta.total_q_prev_tokens
        q_prev = (
            q[:split].contiguous().reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )
        q_next = (
            q[split:].contiguous().reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )

        k_cache_paged = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
        )
        v_cache_paged = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
        )

        attn_out_prev, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_prev,
            k_cache_paged,
            v_cache_paged,
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.fia_mask,
            sparse_mode=3,
            next_tokens=0,
            scale=layer.scaling,
            actual_seq_lengths=np.cumsum(cp_meta.actual_seq_q_prev_list).tolist(),
            actual_seq_lengths_kv=cp_meta.kv_len_prev_list,
        )

        attn_out_next, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_next,
            k_cache_paged,
            v_cache_paged,
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.fia_mask,
            sparse_mode=3,
            next_tokens=0,
            scale=layer.scaling,
            actual_seq_lengths=np.cumsum(cp_meta.actual_seq_q_next_list).tolist(),
            actual_seq_lengths_kv=cp_meta.kv_len_next_list,
        )

        attn_out = torch.cat([attn_out_prev, attn_out_next], dim=0)
        return attn_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
    ):

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        k_nope, k_pe = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        if is_prefill:
            if self.forward_metadata.actual_seq_lengths_q is not None:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q
            else:
                actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        else:
            if self.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    actual_seq_qlen = (
                        torch.arange(
                            self.speculative_num_draft_tokens,
                            self.speculative_num_draft_tokens + q.shape[0],
                            self.speculative_num_draft_tokens,
                            dtype=torch.int32,
                        )
                        .to(q.device)
                        .to(torch.int32)
                    )
                else:
                    actual_seq_qlen = (
                        torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
                    )
            else:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q

        if self.forward_metadata.actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = self.forward_metadata.actual_seq_lengths_kv
        elif self.forward_metadata.seq_lens_cpu_int is not None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens

        if (
            is_prefill
            and is_dsa_enable_prefill_cp()
            and forward_batch.attn_cp_metadata is not None
        ):
            attn_out = self.do_cp_balance_attn(
                q_nope,
                k_nope,
                q_pe,
                k_pe,
                topk_indices,
                layer,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            )
        else:
            attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
                query=q_nope,
                key=k_nope,
                value=k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                sparse_indices=topk_indices,
                scale_value=layer.scaling,
                actual_seq_lengths_query=actual_seq_qlen.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )

        return attn_out

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
    ):
        if is_mla_preprocess_enabled() and self.use_mla:
            # MLAPO and MLAPROLOG do save kv_cache
            save_kv_cache = False
        if self.is_dllm_model:
            return self.forward_dllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return self.forward_mtp(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if not self.use_mla:
            # Detect CP mode for prefill (context parallel)
            is_cp_mode = (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            )

            # In cross attention layer, when there is no vision input,the values of k and v is None
            if save_kv_cache and k is not None and v is not None:
                if is_cp_mode:
                    # All-gather K/V from all CP ranks and write full sequence to KV pool
                    _cp_allgather_and_save_kv_npu(
                        forward_batch,
                        layer,
                        k,
                        v,
                        self.attn_cp_size,
                        self.token_to_kv_pool,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    )
                else:
                    # support cross attention
                    cache_loc = (
                        forward_batch.out_cache_loc
                        if not layer.is_cross_attention
                        else forward_batch.encoder_out_cache_loc
                    )
                    swa_loc = (
                        self.forward_metadata.swa_out_cache_loc
                        if not layer.is_cross_attention
                        else None
                    )
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, KVWriteLoc(cache_loc, swa_loc), k, v
                    )

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None or (self._is_swa_layer(layer) and self.use_fia):
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self._is_swa_layer(layer):
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
                if self.use_fia:
                    if self._can_use_tnd(layer):
                        num_token_padding = q.shape[0]
                        if num_token_padding > forward_batch.num_token_non_padded_cpu:
                            q, k, v = [
                                data[: forward_batch.num_token_non_padded_cpu]
                                for data in [q, k, v]
                            ]
                        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                        block_size = self.page_size
                        attn_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                            query=q,
                            key=k_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_k_head_num * layer.qk_head_dim,
                            ),
                            value=v_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_v_head_num * layer.v_head_dim,
                            ),
                            pre_tokens=(
                                layer.sliding_window_size
                                if layer.sliding_window_size != -1
                                else FULL_ATTENTION_WINDOW
                            ),
                            next_tokens=(
                                0
                                if layer.sliding_window_size != -1
                                else FULL_ATTENTION_WINDOW
                            ),
                            atten_mask=self.fia_mask,
                            block_table=block_tables,
                            input_layout="TND",
                            block_size=block_size,
                            num_query_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            actual_seq_qlen=self.forward_metadata.seq_lens_list_cumsum,
                            actual_seq_kvlen=self.forward_metadata.seq_lens_cpu_int,
                            softmax_scale=layer.scaling,
                            sparse_mode=4 if layer.sliding_window_size != -1 else 3,
                            learnable_sink=sinks,
                        )
                        attn_out = attn_out.view(
                            -1, layer.tp_q_head_num * layer.v_head_dim
                        )
                        if num_token_padding != forward_batch.num_token_non_padded_cpu:
                            attn_out = torch.cat(
                                [
                                    attn_out,
                                    attn_out.new_zeros(
                                        num_token_padding - attn_out.shape[0],
                                        *attn_out.shape[1:],
                                    ),
                                ],
                                dim=0,
                            )
                    else:
                        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

                        # FIA BSND with paged KV cache (reads prefix tokens from cache)
                        seq_lens_cpu = forward_batch.seq_lens.cpu().tolist()
                        attn_out = torch.empty(
                            (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                            device=q.device,
                            dtype=q.dtype,
                        )
                        q_len_offset = 0
                        for seq_idx, q_len in enumerate(
                            forward_batch.extend_seq_lens_cpu
                        ):
                            if q_len == 0:
                                continue
                            total_kv_len = seq_lens_cpu[seq_idx]
                            result, _ = torch_npu.npu_fused_infer_attention_score_v2(
                                query=q[None, q_len_offset : q_len_offset + q_len],
                                key=k_cache.view(
                                    -1,
                                    self.page_size,
                                    layer.tp_k_head_num * layer.qk_head_dim,
                                ),
                                value=v_cache.view(
                                    -1,
                                    self.page_size,
                                    layer.tp_v_head_num * layer.v_head_dim,
                                ),
                                num_query_heads=layer.tp_q_head_num,
                                num_key_value_heads=layer.tp_k_head_num,
                                input_layout="BSND",
                                block_table=block_tables[seq_idx : seq_idx + 1],
                                block_size=self.page_size,
                                actual_seq_qlen=[q_len],
                                actual_seq_kvlen=[total_kv_len],
                                atten_mask=self.fia_mask.unsqueeze(0),
                                sparse_mode=4,
                                softmax_scale=layer.scaling,
                                pre_tokens=layer.sliding_window_size,
                                next_tokens=0,
                            )
                            attn_out[q_len_offset : q_len_offset + q_len] = result[0]
                            q_len_offset += q_len

                        attn_out = attn_out.view(
                            -1, layer.tp_q_head_num * layer.v_head_dim
                        )

                else:
                    attn_out = attention_sinks_prefill_triton(
                        q,
                        k_cache,
                        v_cache,
                        sinks,
                        self.forward_metadata.extend_seq_lens,
                        block_tables,
                        self.forward_metadata.seq_lens,
                        layer.scaling,
                        layer.sliding_window_size,
                        layer.tp_q_head_num,
                        layer.tp_k_head_num,
                    )
                return attn_out

            if is_cp_mode:
                if self.use_fia:
                    attn_output = self.do_cp_attn_fia(
                        q, k_cache, v_cache, layer, forward_batch
                    )
                else:
                    raise NotImplementedError(
                        "CP attention for non-FIA path on Ascend is not yet implemented. "
                        "Set ASCEND_USE_FIA=1 to use FIA-based CP attention."
                    )
                return attn_output

            if self.use_fia:
                if self._can_use_tnd(layer):
                    """FIA supports multi-bs in the current version of CANN"""
                    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    num_token_padding = q.shape[0]
                    if num_token_padding > forward_batch.num_token_non_padded_cpu:
                        q, k, v = [
                            data[: forward_batch.num_token_non_padded_cpu]
                            for data in [q, k, v]
                        ]
                    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                        query=q,
                        key=k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                        ),
                        value=v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                        ),
                        block_table=self.forward_metadata.block_tables,
                        block_size=self.page_size,
                        atten_mask=self.fia_mask,
                        input_layout="TND",
                        actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                        actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                        num_key_value_heads=layer.tp_k_head_num,
                        num_heads=layer.tp_q_head_num,
                        scale=layer.scaling,
                        sparse_mode=3,
                    )
                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )

                    if num_token_padding != forward_batch.num_token_non_padded_cpu:
                        attn_output = torch.cat(
                            [
                                attn_output,
                                attn_output.new_zeros(
                                    num_token_padding - attn_output.shape[0],
                                    *attn_output.shape[1:],
                                ),
                            ],
                            dim=0,
                        )
                else:
                    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

                    # FIA BSND with paged KV cache (reads prefix tokens from cache)
                    seq_lens_cpu = forward_batch.seq_lens.cpu().tolist()
                    attn_output = torch.empty(
                        (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                        device=q.device,
                        dtype=q.dtype,
                    )
                    q_len_offset = 0
                    for seq_idx, q_len in enumerate(forward_batch.extend_seq_lens_cpu):
                        if q_len == 0:
                            continue
                        total_kv_len = seq_lens_cpu[seq_idx]
                        result, _ = torch_npu.npu_fused_infer_attention_score_v2(
                            query=q[None, q_len_offset : q_len_offset + q_len],
                            key=k_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_k_head_num * layer.qk_head_dim,
                            ),
                            value=v_cache.view(
                                -1,
                                self.page_size,
                                layer.tp_v_head_num * layer.v_head_dim,
                            ),
                            num_query_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",
                            block_table=self.forward_metadata.block_tables[
                                seq_idx : seq_idx + 1
                            ],
                            block_size=self.page_size,
                            actual_seq_qlen=[q_len],
                            actual_seq_kvlen=[total_kv_len],
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3,
                            softmax_scale=layer.scaling,
                        )
                        attn_output[q_len_offset : q_len_offset + q_len] = result[0]
                        q_len_offset += q_len

                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )
            elif self.use_fa:
                from flash_attn_npu_v3 import flash_attn_with_kvcache

                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                k = k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                )
                v = v_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                extend_seq_lens = self.forward_metadata.extend_seq_lens_cpu_int.npu()
                cu_seqlens_q = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32).npu(),
                        extend_seq_lens.cumsum(0).to(torch.int32),
                    ]
                )
                max_seqlen_q = extend_seq_lens.max().item()
                attn_output = flash_attn_with_kvcache(
                    q,
                    k,
                    v,
                    cache_seqlens=self.forward_metadata.seq_lens,
                    page_table=self.forward_metadata.block_tables,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=[-1, -1],
                    softcap=0.0,
                    rotary_interleaved=False,
                    num_splits=0,
                    sm_margin=0,
                    return_softmax_lse=False,
                )
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            else:
                causal = True
                if (
                    layer.is_cross_attention
                    or layer.attn_type == AttentionType.ENCODER_ONLY
                ):
                    causal = False
                # there are some accuracy issues in cross attention scene to use torch_npu._npu_flash_attention_qlens
                # forward_batch.encoder_lens is not None in cross attention scend, we add native attn to solve accuracy issues
                # Model skywork-reward-gemma2-2-27B also suffers from precision anomalies, thus the torch native backend becomes beneficial approach.
                if (
                    layer.qk_head_dim <= 128
                    and causal
                    and forward_batch.encoder_lens is None
                    and layer.logit_cap == 0
                    and not getattr(self, "use_native_sdpa", False)
                ):
                    if not self.use_alibi:
                        query = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
                        attn_output = torch.empty(
                            (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                            dtype=query.dtype,
                            device=query.device,
                        )
                        torch_npu._npu_flash_attention_qlens(
                            query=query,
                            key_cache=k_cache,
                            value_cache=v_cache,
                            mask=self.mask,
                            block_table=self.forward_metadata.block_tables,
                            seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                            context_lens=self.forward_metadata.seq_lens_cpu_int,
                            scale_value=layer.scaling,
                            num_heads=layer.tp_q_head_num,
                            num_kv_heads=layer.tp_k_head_num,
                            out=attn_output,
                        )
                    else:
                        attn_output = self.attn_alibi(
                            q=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
                            k_cache=k_cache,
                            v_cache=v_cache,
                            block_tables=self.forward_metadata.block_tables,
                            seq_lens=self.forward_metadata.seq_lens_cpu_int,
                            query_lens=self.forward_metadata.extend_seq_lens_cpu_int,
                            scale_value=layer.scaling,
                            num_heads=layer.tp_q_head_num,
                            slopes=slopes,
                            is_extend=True,
                        )
                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        attn_output = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        attn_output = torch.empty_like(q)

                    use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                    q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                    # add forward_batch.encoder_lens and is_cross_attention arguments for cross attention scene
                    attn_output = self.native_attn.run_sdpa_forward_extend(
                        q_,
                        o_,
                        k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        self.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        forward_batch.encoder_lens,
                        is_cross_attention=layer.is_cross_attention,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
                        sliding_window_size=layer.sliding_window_size,
                        full_to_swa_mapping=(
                            self.full_to_swa_index_mapping
                            if self._is_swa_layer(layer)
                            else None
                        ),
                        logit_cap=layer.logit_cap,
                        logit_capping_method=layer.logit_capping_method,
                    )
                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )
        elif sum(forward_batch.extend_prefix_lens_cpu) > 0:
            # This branch adds support for prefix cache for GLM-4.7-Flash.
            # When using the MLA architecture, if qk head dim equals v head dim and the head count is not a power of 2,
            # we use the FIA kernel for computation.
            if layer.qk_head_dim == layer.v_head_dim:
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

                k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                kv_cached = torch.index_select(
                    k_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                )
                k_rope_cached = torch.index_select(
                    v_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                ).flatten(0, 1)

                assert layer.kv_b_proj is not None
                kv = layer.kv_b_proj(kv_cached)[0].view(
                    -1, layer.tp_k_head_num, self.qk_nope_head_dim + layer.v_head_dim
                )
                k_nope, v_pre = kv.split(
                    [self.qk_nope_head_dim, layer.v_head_dim], dim=-1
                )

                k_rope = k_rope_cached.expand(-1, layer.tp_k_head_num, -1)
                k_pre = torch.cat([k_nope, k_rope], dim=-1)

                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                prefix_len_offset = 0
                for q_len, prefix_len in zip(
                    self.forward_metadata.extend_seq_lens_cpu_int,
                    self.forward_metadata.prefix_lens,
                ):
                    k_cur_slice = k[None, q_len_offset : q_len_offset + q_len]
                    v_cur_slice = v[None, q_len_offset : q_len_offset + q_len]
                    k_pre_slice = k_pre[
                        None, prefix_len_offset : prefix_len_offset + prefix_len
                    ]
                    v_pre_slice = v_pre[
                        None, prefix_len_offset : prefix_len_offset + prefix_len
                    ]

                    k_full = torch.cat([k_pre_slice, k_cur_slice], dim=1)
                    v_full = torch.cat([v_pre_slice, v_cur_slice], dim=1)

                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k_full,
                            v_full,
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask,
                            sparse_mode=3,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                    prefix_len_offset += prefix_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            else:
                num_token_padding = q.shape[0]
                q, k, v = [
                    data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
                ]
                q_nope, q_rope = q.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )
                k_nope, k_rope = k.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )

                # 1st, compute extend tokens to get attn_output and attn_lse
                num_tokens = q_nope.size(0)
                attn_output = torch.zeros(
                    num_tokens,
                    layer.tp_q_head_num,
                    layer.v_head_dim,
                    dtype=q_nope.dtype,
                    device=q_nope.device,
                )
                attn_lse = torch.zeros(
                    layer.tp_q_head_num,
                    num_tokens,
                    dtype=torch.float32,
                    device=q_nope.device,
                )
                torch_npu.atb.npu_ring_mla(
                    q_nope=q_nope,
                    q_rope=q_rope,
                    k_nope=k_nope,
                    k_rope=k_rope,
                    value=v,
                    mask=self.ringmla_mask,
                    seqlen=self.forward_metadata.extend_seq_lens_cpu_int,
                    head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                    pre_out=None,
                    prev_lse=None,
                    qk_scale=layer.scaling,
                    kernel_type="kernel_type_high_precision",
                    mask_type="mask_type_triu",
                    calc_type="calc_type_first_ring",
                    output=attn_output,
                    softmax_lse=attn_lse,
                )

                # 2nd, load history kvcache(kv_a and k_pe) and calculate k_nope
                k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                kv_cached = torch.index_select(
                    k_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                )
                k_rope_cached = torch.index_select(
                    v_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                ).flatten(0, 1)

                assert layer.kv_b_proj is not None
                kv = layer.kv_b_proj(kv_cached)[0].view(
                    -1, layer.tp_k_head_num, self.qk_nope_head_dim + layer.v_head_dim
                )
                k_nope, v = kv.split([self.qk_nope_head_dim, layer.v_head_dim], dim=-1)

                # 3rd, compute history kv to attn_out
                k_rope = k_rope_cached.expand(-1, layer.tp_k_head_num, -1)
                seq_len = torch.stack(
                    [
                        self.forward_metadata.extend_seq_lens_cpu_int,
                        self.forward_metadata.prefix_lens,
                    ]
                )
                torch_npu.atb.npu_ring_mla(
                    q_nope=q_nope,
                    q_rope=q_rope,
                    k_nope=k_nope,
                    k_rope=k_rope,
                    value=v,
                    mask=self.ringmla_mask,
                    seqlen=seq_len,
                    head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                    pre_out=attn_output,
                    prev_lse=attn_lse,
                    qk_scale=layer.scaling,
                    kernel_type="kernel_type_high_precision",
                    mask_type="no_mask",
                    calc_type="calc_type_default",
                    output=attn_output,
                    softmax_lse=attn_lse,
                )
                attn_output = attn_output.reshape(
                    [-1, layer.tp_q_head_num, layer.v_head_dim]
                )
                if num_token_padding != forward_batch.num_token_non_padded_cpu:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - attn_output.shape[0],
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )
        else:
            if layer.qk_head_dim == layer.v_head_dim:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3 if q_len != 1 else 0,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            elif layer.v_head_dim in [256]:
                """Currently, in NO_QUANT situation, qk_nope_head_dim == v_head_dim, and rope exists, v_head_dim only support 512 and 128"""
                kv_lora_rank = k.shape[-1] - self.qk_rope_head_dim
                kv_c, k_rope = k.split([kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, kv_c, k_rope
                    )
                attn_output = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, kv_lora_rank)
                )
                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                kv_cache = torch.cat([k_cache, v_cache], dim=-1)
                attn_output = self.native_attn.run_sdpa_forward_extend(
                    q,
                    attn_output,
                    kv_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    k_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.extend_prefix_lens,
                    forward_batch.extend_seq_lens,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=True,
                )
            else:
                num_token_padding = q.shape[0]
                q, k, v = [
                    data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
                ]

                q_nope, q_rope = q.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )
                k_nope, k_rope = k.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )

                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_nope,
                    v,
                    query_rope=q_rope,
                    key_rope=k_rope,
                    num_heads=layer.tp_q_head_num,
                    input_layout="TND",
                    atten_mask=self.fia_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                    scale=layer.scaling,
                    next_tokens=0,
                )

                attn_output = attn_output.reshape(
                    -1, layer.tp_q_head_num, layer.v_head_dim
                )
                if num_token_padding != forward_batch.num_token_non_padded_cpu:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - attn_output.shape[0],
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )

        return attn_output

    def forward_dllm(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                ),
                k,
                v,
            )

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if self.forward_metadata.seq_lens_cpu_int is None:
            # capture
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
        else:
            # eagle
            actual_seq_lengths_kv = (
                self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            )

        if self.forward_metadata.extend_seq_lens_cpu_int is None:
            # capture & replay
            actual_seq_lengths = self.forward_metadata.seq_lens_list_cumsum
        else:
            actual_seq_lengths = (
                torch.cumsum(self.forward_metadata.extend_seq_lens_cpu_int, dim=0)
                .int()
                .tolist()
            )

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim),
            v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim),
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=None,
            scale=layer.scaling,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        return attn_output

    def forward_mtp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                )

        if not self.use_mla:
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
            )
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
            )
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()

            if not self.graph_mode:
                num_token_padding = query.shape[0]
                query = query[: forward_batch.num_token_non_padded_cpu]

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            if forward_batch.forward_mode.is_draft_extend_v2():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + query.shape[0],
                    self.speculative_num_draft_tokens,
                )

            is_swa_layer = layer.sliding_window_size != -1
            if (
                is_swa_layer
                and self.is_hybrid_swa
                and hasattr(self.forward_metadata, "block_tables_swa")
            ):
                block_table = self.forward_metadata.block_tables_swa
            else:
                block_table = self.forward_metadata.block_tables

            if layer.attn_type == AttentionType.ENCODER_ONLY:
                mask = None
                sparse_mode = 0
            else:
                mask = self.mtp_mask
                sparse_mode = 4 if is_swa_layer else 3

            if self.is_hybrid_swa:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                    query,
                    k_cache,
                    v_cache,
                    block_table=block_table,
                    block_size=self.page_size,
                    num_query_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    atten_mask=mask,
                    softmax_scale=layer.scaling,
                    actual_seq_qlen=actual_seq_lengths,
                    actual_seq_kvlen=actual_seq_lengths_kv,
                    sparse_mode=sparse_mode,
                    pre_tokens=(
                        layer.sliding_window_size
                        if is_swa_layer
                        else FULL_ATTENTION_WINDOW
                    ),
                    next_tokens=0 if is_swa_layer else FULL_ATTENTION_WINDOW,
                    learnable_sink=sinks,
                )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    atten_mask=mask,
                    scale=layer.scaling,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    sparse_mode=sparse_mode,
                )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu is not None
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - forward_batch.num_token_non_padded_cpu,
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output
        else:
            c_kv, k_rope = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(
                    -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
                )

            q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            if not self.graph_mode:
                num_token_padding = q.shape[0]
                q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
                q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            actual_seq_lengths = np.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens + q_nope.shape[0],
                self.speculative_num_draft_tokens,
            )

            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > self.tp_q_head_num
            ):
                nope_padding = torch.empty(
                    [
                        q_nope.shape[0],
                        self.q_head_num_padding - self.tp_q_head_num,
                        self.kv_lora_rank,
                    ],
                    dtype=(
                        self.model_dtype
                        if self.model_dtype is not None
                        else torch.bfloat16
                    ),
                    device=q_nope.device,
                )
                rope_padding = torch.empty(
                    [
                        q_rope.shape[0],
                        self.q_head_num_padding - self.tp_q_head_num,
                        self.qk_rope_head_dim,
                    ],
                    dtype=(
                        self.model_dtype
                        if self.model_dtype is not None
                        else torch.bfloat16
                    ),
                    device=q_rope.device,
                )
                q_nope = torch.cat([q_nope, nope_padding], dim=1).contiguous()
                q_rope = torch.cat([q_rope, rope_padding], dim=1).contiguous()

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
            )
            attn_output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                workspace=workspace,
                out=[attn_output, softmax_lse],
            )
            attn_output = attn_output[:, : layer.tp_q_head_num, :]
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                )

        if sinks is not None or self.is_hybrid_swa:
            # Use SWA block tables if hybrid SWA is enabled for this layer
            if self._is_swa_layer(layer):
                block_tables = self.forward_metadata.block_tables_swa
            else:
                block_tables = self.forward_metadata.block_tables
            if self.use_fia:
                k_cache = (
                    self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    .view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
                    .contiguous()
                )
                v_cache = (
                    self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                    .view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
                    .contiguous()
                )
                query = q.reshape(
                    -1, layer.tp_q_head_num, layer.qk_head_dim
                ).contiguous()

                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_lengths_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                seq_lens_list = (
                    self.forward_metadata.seq_lens_cpu_list
                    if self.forward_metadata.seq_lens_cpu_int is None
                    else self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
                actual_seq_lengths = (
                    torch.tensor([1] * len(seq_lens_list), dtype=torch.int32)
                    .cumsum(dim=0)
                    .tolist()
                )
                if layer.sliding_window_size != -1:
                    sparse_mode = 4
                else:
                    sparse_mode = 3

                attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                    query,
                    k_cache,
                    v_cache,
                    num_query_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    pre_tokens=(
                        layer.sliding_window_size
                        if layer.sliding_window_size != -1
                        else FULL_ATTENTION_WINDOW
                    ),
                    next_tokens=(
                        0 if layer.sliding_window_size == -1 else FULL_ATTENTION_WINDOW
                    ),
                    atten_mask=self.fia_mask.to(torch.int8),
                    sparse_mode=sparse_mode,
                    softmax_scale=layer.scaling,
                    block_table=block_tables,
                    block_size=self.page_size,
                    actual_seq_qlen=actual_seq_lengths,
                    actual_seq_kvlen=actual_seq_lengths_kv,
                    learnable_sink=sinks,
                )
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
                return attn_output
            else:
                k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                attn_out = attention_sinks_triton(
                    q,
                    k_cache,
                    v_cache,
                    sinks,
                    block_tables,
                    self.forward_metadata.seq_lens,
                    layer.scaling,
                    layer.sliding_window_size,
                    layer.tp_q_head_num,
                    layer.tp_k_head_num,
                )
                return attn_out

        if not self.use_mla:
            seq_lens_cpu_int = self.forward_metadata.seq_lens_cpu_int
            seq_lens_cpu_list = self.forward_metadata.seq_lens_cpu_list
            if self._is_swa_layer(layer):
                # CUDA/NPU graph capture uses seq_len fill value 0 on Ascend.
                # Avoid dynamic window block-table construction during capture,
                # because it can create a zero-width block table and break tiling.
                block_tables = self.forward_metadata.block_tables_swa
                attn_mask = self.forward_metadata.swa_mask
            else:
                block_tables = self.forward_metadata.block_tables
                attn_mask = None
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
            )
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
            )
            query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
            if seq_lens_cpu_int is None:
                actual_seq_len_kv = seq_lens_cpu_list
            else:
                actual_seq_len_kv = seq_lens_cpu_int.cpu().int().tolist()

            num_tokens = query.shape[0]
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query,
                k_cache,
                v_cache,
                block_table=block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                atten_mask=attn_mask,
                sparse_mode=0,
            )
            output = torch.empty(
                (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                query,
                k_cache,
                v_cache,
                block_table=block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                atten_mask=attn_mask,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            c_kv, k_rope = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                )

            q_nope = q.view(-1, 1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, 1, layer.tp_q_head_num, self.qk_rope_head_dim)

            assert (
                self.q_head_num_padding is None
                or self.q_head_num_padding >= layer.tp_q_head_num
            )

            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > layer.tp_q_head_num
            ):
                # The FIA kernel only supports head counts that are powers of 2.
                # Therefore, we pad the head dimension when it is not a power of 2.
                q_nope = torch.cat(
                    [q_nope, self.forward_metadata.nope_padding], dim=2
                ).contiguous()
                q_rope = torch.cat(
                    [q_rope, self.forward_metadata.rope_padding], dim=2
                ).contiguous()

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
            )
            output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )

            output = output[:, :, : layer.tp_q_head_num, :]
            return output.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if is_mla_preprocess_enabled() and self.use_mla:
            # MLAPO does saving kv_cache
            save_kv_cache = False
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )

        if self.graph_mode and (not self.enable_torch_compile):
            return self.forward_decode_graph(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if not self.use_mla:
            # In cross attention layer, when there is no vision input,the values of k and v is None
            if save_kv_cache and k is not None and v is not None:
                # support cross attention
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                # swa_out_cache_loc is the full->SWA write target, derived from
                # out_cache_loc; it must not be applied to cross-attention writes
                # (which target encoder_out_cache_loc) and is None for non-SWA pools.
                swa_loc = (
                    self.forward_metadata.swa_out_cache_loc
                    if not layer.is_cross_attention
                    else None
                )
                self.token_to_kv_pool.set_kv_buffer(
                    layer, KVWriteLoc(cache_loc, swa_loc), k, v
                )
            num_tokens = q.shape[0]
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None or (self._is_swa_layer(layer) and self.use_fia):
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self._is_swa_layer(layer):
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
                if self.use_fia:
                    if self.forward_metadata.seq_lens_cpu_int is None:
                        actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                    else:
                        actual_seq_len_kv = (
                            self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                        )
                    block_size = self.page_size

                    if sinks is not None:
                        mask = self.fia_mask
                    else:
                        max_model_len = block_tables.shape[-1] * block_size
                        mask = self.ascend_attn_mask_builder.get_swa_mask(
                            self.forward_metadata.seq_lens,
                            max_model_len,
                            layer.sliding_window_size,
                        )

                    attn_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                        q.view(
                            forward_batch.batch_size,
                            -1,
                            layer.tp_q_head_num,
                            layer.qk_head_dim,
                        ),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                        ),
                        num_query_heads=layer.tp_q_head_num,
                        num_key_value_heads=layer.tp_k_head_num,
                        input_layout="BSND",
                        block_size=block_size,
                        atten_mask=(mask if layer.sliding_window_size != -1 else None),
                        sparse_mode=4 if layer.sliding_window_size != -1 else 0,
                        softmax_scale=layer.scaling,
                        block_table=block_tables,
                        actual_seq_qlen=[1] * len(self.forward_metadata.seq_lens),
                        actual_seq_kvlen=actual_seq_len_kv,
                        pre_tokens=layer.sliding_window_size,
                        next_tokens=0,
                        learnable_sink=sinks,
                    )
                    attn_out = attn_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
                else:
                    attn_out = attention_sinks_triton(
                        q,
                        k_cache,
                        v_cache,
                        sinks,
                        block_tables,
                        self.forward_metadata.seq_lens,
                        layer.scaling,
                        layer.sliding_window_size,
                        layer.tp_q_head_num,
                        layer.tp_k_head_num,
                    )
                return attn_out

            if self.use_fia:
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                num_token_padding = q.shape[0]
                actual_bs = self.forward_metadata.block_tables.shape[0]
                q = q[:actual_bs]
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q.view(
                        -1,
                        1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    ),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                    ),
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=self.page_size,
                    block_table=self.forward_metadata.block_tables,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    scale=layer.scaling,
                )
                if actual_bs != num_token_padding:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - actual_bs,
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )
            elif self.use_fa:
                from flash_attn_npu_v3 import flash_attn_with_kvcache

                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, layer.qk_head_dim
                )
                k = k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                )
                v = v_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                attn_output = flash_attn_with_kvcache(
                    q,
                    k,
                    v,
                    page_table=self.forward_metadata.block_tables,
                    cache_seqlens=self.forward_metadata.seq_lens,
                    softmax_scale=layer.scaling,
                )
            # there are some accuracy issues in cross attention scene to use torch_npu._npu_flash_attention_qlens
            # forward_batch.encoder_lens is not None in cross attention scend, we add native attn to solve accuracy issues
            elif forward_batch.encoder_lens is None and layer.logit_cap == 0:
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                if not self.use_alibi:
                    attn_output = torch.empty(
                        (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                        dtype=query.dtype,
                        device=query.device,
                    )

                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=k_cache,
                        value_cache=v_cache,
                        num_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        scale_value=layer.scaling,
                        block_table=self.forward_metadata.block_tables,
                        context_lens=self.forward_metadata.seq_lens_cpu_int,
                        out=attn_output,
                    )
                else:
                    attn_output = self.attn_alibi(
                        q=query,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        block_tables=self.forward_metadata.block_tables,
                        seq_lens=self.forward_metadata.seq_lens_cpu_int,
                        query_lens=torch.ones(num_tokens, dtype=torch.int32),
                        scale_value=layer.scaling,
                        num_heads=layer.tp_q_head_num,
                        slopes=slopes,
                        is_extend=False,
                    )
            else:
                if layer.qk_head_dim != layer.v_head_dim:
                    attn_output = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    attn_output = torch.empty_like(q)

                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                attn_output = self.native_attn.run_sdpa_forward_decode(
                    q_,
                    o_,
                    k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.encoder_lens,
                    is_cross_attention=layer.is_cross_attention,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=False,
                    sliding_window_size=layer.sliding_window_size,
                    full_to_swa_mapping=(
                        self.full_to_swa_index_mapping
                        if self._is_swa_layer(layer)
                        else None
                    ),
                    logit_cap=layer.logit_cap,
                    logit_capping_method=layer.logit_capping_method,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
                """layer.tp_q_head_num // layer.tp_k_head_num < 8 will support in the later version of CANN"""
                if is_fia_nz():
                    kv_c = _reshape_kv_for_fia_nz(
                        kv_c, layer.tp_k_head_num, self.kv_lora_rank, self.page_size
                    )
                    k_pe = _reshape_kv_for_fia_nz(
                        k_pe, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                    )
                else:
                    kv_c = kv_c.view(
                        -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                    )
                    k_pe = k_pe.view(
                        -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                    )
                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank
                )
                q_rope = q_rope.view(
                    forward_batch.batch_size,
                    -1,
                    layer.tp_q_head_num,
                    self.qk_rope_head_dim,
                )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    kv_c,
                    kv_c,
                    query_rope=q_rope,
                    key_rope=k_pe,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    sparse_mode=0,
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                )
            else:
                assert (
                    self.graph_mode == False
                )  # _npu_paged_attention_mla not support graph mode
                if q_rope is not None:
                    q = torch.cat([q, q_rope], dim=-1)
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                kv_c_and_k_pe_cache = torch.cat([kv_c, k_pe], dim=-1)
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                attn_output = torch.empty(
                    [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_c_and_k_pe_cache,
                    num_kv_heads=layer.tp_k_head_num,
                    num_heads=layer.tp_q_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if (
            topk_indices is not None
            or self.use_mla
            or (not self.use_fia and layer.qk_head_dim > 128)
        ):
            raise NotImplementedError(
                "The 'enable-mixed-chunk' feature is currently unsupported in the following scenarios: "
                "1. When using the MLA backend on Ascend NPU devices, "
                "2. When using the deepseekv3.2 model on Ascend NPU devices, "
                "3. When the environment variable ASCEND_USE_FIA is set to 0 and qk_head_dim exceeds 128 on Ascend NPU devices."
            )
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                ),
                k,
                v,
            )
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        num_block, block_size, _, _ = k_cache.shape
        key = k_cache.view(num_block, block_size, -1)
        value = v_cache.view(num_block, block_size, -1)

        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            block_size=block_size,
            block_table=self.forward_metadata.block_tables,
            atten_mask=self.mix_mask,
            sparse_mode=3,
            actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
            actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
            scale=layer.scaling,
        )

        return attn_output.view(
            attn_output.shape[0], layer.tp_q_head_num * layer.v_head_dim
        )


class AscendAttnMultiStepDraftBackend:
    """
    Wrap multiple Ascend attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for step_id in range(self.speculative_num_steps):
            self.attn_backends.append(
                AscendAttnBackend(model_runner, speculative_step_id=step_id)
            )

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_in_graph(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)
