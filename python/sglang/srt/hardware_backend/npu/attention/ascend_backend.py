from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    OutCacheLoc,
)
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import logging

import numpy as np


def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -> torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)


logger = logging.getLogger(__name__)


def dsv4_sparse_attn(
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
):
    query_states = query_states.transpose(1, 2)
    kv_states = kv_states.unsqueeze(1)
    attn_weights = torch.matmul(query_states, kv_states.transpose(2, 3)) * softmax_scale
    topk_idxs = topk_idxs.to(query_states.device)
    index_mask = torch.full(
        (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),
        fill_value=torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device="npu",
    ).scatter_(-1, topk_idxs.unsqueeze(1), 0)

    attn_weights = attn_weights + index_mask[..., :-1]

    sinks = sinks.reshape(1, -1, 1, 1).expand(
        query_states.shape[0], -1, query_states.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    del combined_logits
    scores = probs[..., :-1].to(kv_states.dtype)
    attn_output = torch.matmul(scores, kv_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def get_kv_indices(forward_batch, kv_len, page_table, idx, seqlen):
    logic_start = max(0, seqlen - kv_len)
    logic_end = seqlen
    if forward_batch.attn_backend.page_size == 1:
        kv_indices = page_table[idx, logic_start:logic_end]
    else:
        page_size = forward_batch.attn_backend.page_size
        logic_pos = torch.arange(logic_start, logic_end, device=page_table.device)
        block_id = logic_pos // page_size
        offset_in_block = logic_pos % page_size
        physical_block_id = page_table[idx, block_id]
        kv_indices = physical_block_id * page_size + offset_in_block
    return kv_indices


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # mapped block_tables for swa
    block_tables_swa: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

    # prefix cache
    prefix_lens: Optional[torch.Tensor] = None
    flatten_prefix_block_tables: Optional[torch.Tensor] = None

    # dsv4
    kernel_metadata = None
    start_pos = None

    swa_page_table = None
    c4_page_table = None
    c128_page_table = None
    c4_state_page_table = None
    c128_state_page_table = None

    swa_loc = None
    swa_loc_local = None
    swa_kv_tobe_scatter_index = None
    c4_loc = None
    c128_loc = None
    c128_state_loc = None
    c4_state_loc = None

    actual_seq_lengths_q_pa = None
    positions_cmp_padding_c4 = None
    positions_cmp_padding_c128 = None
    seqused = None
    actual_seq_lengths_q_cmp = None


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


class AscendAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.speculative_step_id = speculative_step_id
        self.speculative_step_offset = speculative_step_id + 1
        self.speculative_step_offset_npu = torch.tensor(
            speculative_step_id + 1, device="npu"
        )
        self.page_size = model_runner.page_size
        self.model_dtype = model_runner.model_config.dtype
        self.is_dsv4 = (
            "DeepseekV4ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "DeepseekV4ForCausalLMNextN"
            in model_runner.model_config.hf_config.architectures
        )
        self.is_dsv4_nextn = (
            "DeepseekV4ForCausalLMNextN"
            in model_runner.model_config.hf_config.architectures
        )

        self.config = model_runner.model_config
        assert (self.is_dsv4 and self.page_size in [1, 128]) or (not self.is_dsv4)
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
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_token_swa = model_runner.req_to_token_pool.req_to_token_swa
        self.req_to_token_c4 = model_runner.req_to_token_pool.req_to_token_c4
        self.req_to_token_c128 = model_runner.req_to_token_pool.req_to_token_c128
        self.req_to_token_c4_state = (
            model_runner.req_to_token_pool.req_to_token_c4_state
        )
        self.req_to_token_c128_state = (
            model_runner.req_to_token_pool.req_to_token_c128_state
        )
        self.graph_mode = False
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
        if self.is_dsv4:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
            self.compress_ratios = model_runner.model_config.hf_config.compress_ratios
            self.index_topk = model_runner.model_config.hf_config.index_topk
            self.index_n_heads = model_runner.model_config.hf_config.index_n_heads
            self.index_head_dim = model_runner.model_config.hf_config.index_head_dim
            assert (
                self.index_topk == 512
                and self.index_n_heads == 64
                and self.index_head_dim == 128
            )
            assert self.use_fia

        # dllm model config
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = False
        if self.dllm_config is not None:
            self.is_dllm_model = True
            self.dllm_block_size = self.dllm_config.block_size

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        positions = spec_info.positions
        self.forward_metadata.positions_cmp_padding_c4.fill_(0)
        self.forward_metadata.positions_cmp_padding_c128.fill_(0)
        if positions.numel() > 0:
            request_num = positions.shape[0] // self.speculative_num_draft_tokens
            start_positions = (
                self.forward_metadata.seq_lens_cpu[:request_num]
                - self.speculative_num_draft_tokens
                + 1
            )
            abs_positions = start_positions.view(-1, 1) + torch.arange(
                self.speculative_num_draft_tokens,
            ).view(1, -1)
            mask_c4 = (abs_positions % 4) != 0
            mask_c128 = (abs_positions % 128) != 0
            gather_shape_c4 = min(
                positions.shape[0],
                self.forward_metadata.positions_cmp_padding_c4.shape[0],
            )
            gather_shape_c128 = min(
                positions.shape[0],
                self.forward_metadata.positions_cmp_padding_c128.shape[0],
            )
            sorted_indices_c4 = (
                torch.argsort(mask_c4.flatten(), dim=0, stable=True)[:gather_shape_c4]
                .pin_memory()
                .to(device=positions.device, non_blocking=True)
            )
            sorted_indices_c128 = (
                torch.argsort(mask_c128.flatten(), dim=0, stable=True)[
                    :gather_shape_c128
                ]
                .pin_memory()
                .to(device=positions.device, non_blocking=True)
            )
            self.forward_metadata.positions_cmp_padding_c4[:gather_shape_c4] = (
                torch.gather(positions, 0, sorted_indices_c4)
            )
            self.forward_metadata.positions_cmp_padding_c128[:gather_shape_c128] = (
                torch.gather(positions, 0, sorted_indices_c128)
            )

    def compute_kernel_metadata(
        self,
        batch_size: int,
        forward_metadata: ForwardMetadata,
        max_seqlen_q: int,
        is_nextn=False,
    ):
        fa_common_kwargs = {
            "cu_seqlens_q": forward_metadata.actual_seq_lengths_q_pa,  # just for TND: 0,1,2,3,4,5; B+1
            "seqused_kv": forward_metadata.actual_seq_lengths_kv,  # num of key elements used, DT_INT32, kv_len TODO: 确认压缩的怎么填
            "cmp_ratio": 1,  # no support 1, None.   TODO repair this after package updated
            "ori_mask_mode": 4,  # sliding window
            "cmp_mask_mode": 3,  # causal
            "ori_win_left": self.config.sliding_window_size - 1,
            "ori_win_right": 0,
            "layout_q": "TND",  # "BSND" , "TND"
            "layout_kv": "PA_ND",  # "PA_ND"
        }
        tp_size = get_attention_tp_size()
        q_head_num = self.config.get_num_attention_heads(tp_size)
        kv_head_num = self.config.get_total_num_kv_heads()
        c1a_metadata_kwargs = {
            "batch_size": batch_size,  # If tnd layout, set None. TODO repair this after package updated
            "num_heads_q": q_head_num,
            "num_heads_kv": kv_head_num,
            "head_dim": self.config.head_dim,  # TODO: qzd
            "has_ori_kv": True,
            "has_cmp_kv": False,  # True, False; False means no compressor kv cache
        }
        kernel_metadata = {}
        c1a_metadata_kwargs = c1a_metadata_kwargs | fa_common_kwargs
        c1a_metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
            **c1a_metadata_kwargs
        )
        kernel_metadata.update({"c1a_metadata": c1a_metadata})
        if not is_nextn:
            # scfa_metadata
            c4a_metadata_kwargs = {
                "cmp_ratio": 4,
                "has_cmp_kv": True,
                "cmp_topk": self.index_topk,
            }
            c4a_metadata_kwargs = c1a_metadata_kwargs | c4a_metadata_kwargs
            c4a_metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                **c4a_metadata_kwargs
            )
            kernel_metadata.update({"c4a_metadata": c4a_metadata})

            # cfa_metadata
            c128a_metadata_kwargs = {"cmp_ratio": 128, "has_cmp_kv": True}
            c128a_metadata_kwargs = c1a_metadata_kwargs | c128a_metadata_kwargs

            c128a_metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                **c128a_metadata_kwargs
            )
            kernel_metadata.update({"c128a_metadata": c128a_metadata})

            # li_quant_metadata
            li_quant_metadata = torch.ops.custom.npu_quant_lightning_indexer_metadata(
                device=str(forward_metadata.actual_seq_lengths_q.device),
                actual_seq_lengths_query=forward_metadata.actual_seq_lengths_q,
                actual_seq_lengths_key=forward_metadata.actual_seq_lengths_kv,
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                layout_query="TND",
                cmp_ratio=4,  # only c4a have li module
                key_quant_mode=0,  # 0:per-token-head
                query_quant_mode=0,  # 0:per-token-head
                num_heads_q=self.index_n_heads,
                num_heads_k=1,  # MQA: num_heads_kv=1
                head_dim=self.index_head_dim,
            )
            kernel_metadata.update({"li_quant_metadata": li_quant_metadata})
        return kernel_metadata

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

        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens = forward_batch.extend_seq_lens
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
            self.forward_metadata.actual_seq_lengths_q = (
                forward_batch.extend_seq_lens.cumsum(0).int()
            )
            if self.is_dsv4:
                self.forward_metadata.actual_seq_lengths_q_pa = torch.cat(
                    [
                        torch.tensor([0], dtype=torch.int32, device="npu"),
                        self.forward_metadata.actual_seq_lengths_q,
                    ],
                    dim=0,
                )  # [0,seq1,seq1+seq2,...]
                self.forward_metadata.actual_seq_lengths_q_cmp = (
                    self.forward_metadata.actual_seq_lengths_q_pa.clone()
                )
                max_seqlen_q = seq_lens_max

        if forward_batch.seq_lens is not None:
            self.forward_metadata.seq_lens = forward_batch.seq_lens.int()
        else:
            self.forward_metadata.seq_lens = forward_batch.seq_lens_cpu.to(
                self.device
            ).int()

        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum
        if forward_batch.forward_mode.is_decode():
            self.forward_metadata.actual_seq_lengths_q = torch.arange(
                1, forward_batch.batch_size + 1, device="npu", dtype=torch.int32
            )
            if self.is_dsv4:
                self.forward_metadata.actual_seq_lengths_q_pa = torch.arange(
                    0, forward_batch.batch_size + 1, device="npu", dtype=torch.int32
                )
                self.forward_metadata.actual_seq_lengths_q_cmp = (
                    self.forward_metadata.actual_seq_lengths_q_pa.clone()
                )
                max_seqlen_q = 1
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            self.forward_metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens * forward_batch.batch_size
                + self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device="npu",
                dtype=torch.int32,
            )
            if self.is_dsv4:
                self.forward_metadata.actual_seq_lengths_q_pa = torch.arange(
                    0,
                    forward_batch.batch_size * self.speculative_num_draft_tokens
                    + self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    device="npu",
                    dtype=torch.int32,
                )
                self.forward_metadata.actual_seq_lengths_q_cmp = (
                    self.forward_metadata.actual_seq_lengths_q_pa.clone()
                )
                max_seqlen_q = self.speculative_num_draft_tokens

        self.forward_metadata.actual_seq_lengths_kv = forward_batch.seq_lens.clone().to(
            torch.int32
        )

        # compute page_table
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ]
        )

        if self.is_dsv4:
            if forward_batch.forward_mode.is_extend():
                num_pages_swa = (
                    forward_batch.seq_lens_cpu + (self.page_size - 1)
                ) // self.page_size
                cum_num_pages_swa = num_pages_swa.cumsum(0)
                page_offset = 0
                swa_kv_tobe_scatter_index = []
                self.forward_metadata.swa_page_table = torch.zeros_like(
                    self.forward_metadata.block_tables
                )
                for idx, seq_len in enumerate(forward_batch.seq_lens_cpu):
                    if seq_len == 0:
                        continue
                    positions_per_bs = (
                        torch.arange(seq_len, dtype=torch.int32, device="npu")
                        + page_offset * self.page_size
                    )
                    swa_kv_tobe_scatter_index.append(positions_per_bs)
                    page_offset = cum_num_pages_swa[idx]
                    self.forward_metadata.swa_page_table[
                        idx, : positions_per_bs.numel()
                    ].copy_(positions_per_bs)
                self.forward_metadata.swa_kv_tobe_scatter_index = torch.cat(
                    swa_kv_tobe_scatter_index
                )
            else:
                self.forward_metadata.swa_page_table = (
                    forward_batch.req_to_token_pool.req_to_token_swa[
                        forward_batch.req_pool_indices, :seq_lens_max
                    ]
                )

            # compute loc for compressor and swa
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
            ):  # prefill
                c4_offset = 0
                swa_offset = 0
                full_offset = 0
                c4_out_loc_list = []
                swa_out_loc_list = []
                swa_kv_local_list = []
                save_swa_lens = forward_batch.seq_lens.clip(
                    max=self.config.sliding_window_size
                )
                for idx, (seqlen, c4_state_seqlen, swa_seqlen) in enumerate(
                    zip(
                        forward_batch.seq_lens,
                        forward_batch.kv_seq_lens_cpu.c4_state_kv_len,
                        forward_batch.kv_seq_lens_cpu.swa_kv_len,
                    )
                ):
                    c4_end = c4_offset + c4_state_seqlen
                    c4_state_out_cache_loc = (
                        forward_batch.out_cache_loc_dsv4.out_c4_state_loc[
                            c4_offset:c4_end
                        ]
                    )
                    ratio = 4
                    remainder = seqlen % ratio
                    cutoff = seqlen - remainder
                    last_num = 0
                    if cutoff >= ratio:
                        last_num += ratio
                    last_num += remainder
                    c4_out_loc_list.append(c4_state_out_cache_loc[-last_num:])
                    c4_offset += c4_state_seqlen

                    swa_end = swa_offset + swa_seqlen
                    swa_out_cache_loc = forward_batch.out_cache_loc_dsv4.out_swa_loc[
                        swa_offset:swa_end
                    ]
                    if seqlen > self.config.sliding_window_size:
                        swa_out_cache_loc = swa_out_cache_loc[
                            -self.config.sliding_window_size :
                        ]
                    swa_out_loc_list.append(swa_out_cache_loc)

                    end = full_offset + seqlen
                    swa_kv_local_list.append(
                        torch.arange(
                            end - save_swa_lens[idx],
                            end,
                            dtype=torch.int64,
                            device="npu",
                        )
                    )
                    swa_offset += swa_seqlen
                    full_offset += seqlen
                c4_out_cache_loc = torch.cat(c4_out_loc_list, dim=0)
                swa_out_cache_loc = torch.cat(swa_out_loc_list, dim=0)
                swa_loc_local = torch.cat(swa_kv_local_list, dim=0)
                self.forward_metadata.swa_loc_local = (
                    swa_loc_local  # for prefill swa cache gather index
                )

                self.forward_metadata.swa_loc = swa_out_cache_loc
                self.forward_metadata.c4_state_loc = c4_out_cache_loc
            else:  # decoder/verify
                self.forward_metadata.swa_loc = (
                    forward_batch.out_cache_loc_dsv4.out_swa_loc
                )
                self.forward_metadata.c4_state_loc = (
                    forward_batch.out_cache_loc_dsv4.out_c4_state_loc
                )

            if not self.is_dsv4_nextn:
                self.forward_metadata.c4_page_table = (
                    forward_batch.req_to_token_pool.req_to_token_c4[
                        forward_batch.req_pool_indices, : max(1, seq_lens_max // 4)
                    ]
                )

                self.forward_metadata.c128_page_table = (
                    forward_batch.req_to_token_pool.req_to_token_c128[
                        forward_batch.req_pool_indices, : max(1, seq_lens_max // 128)
                    ]
                )

                self.forward_metadata.c4_state_page_table = (
                    forward_batch.req_to_token_pool.req_to_token_c4_state[
                        forward_batch.req_pool_indices, :seq_lens_max
                    ]
                )

                self.forward_metadata.c128_state_page_table = (
                    forward_batch.req_to_token_pool.req_to_token_c128_state[
                        forward_batch.req_pool_indices, :seq_lens_max
                    ]
                )
                self.forward_metadata.c4_loc = (
                    forward_batch.out_cache_loc_dsv4.out_c4_loc
                )
                self.forward_metadata.c128_loc = (
                    forward_batch.out_cache_loc_dsv4.out_c128_loc
                )
                self.forward_metadata.c128_state_loc = (
                    forward_batch.out_cache_loc_dsv4.out_c128_state_loc
                )

            # compute compressor metadata
            if get_bool_env_var("USE_FUSED_COMPRESSOR") and not self.is_dsv4_nextn:
                if forward_batch.forward_mode.is_decode():
                    # pad to (bs,) shape
                    self.forward_metadata.c4_loc = F.pad(
                        self.forward_metadata.c4_loc,
                        (
                            0,
                            forward_batch.batch_size
                            - self.forward_metadata.c4_loc.numel(),
                        ),
                        value=0,
                    )
                    self.forward_metadata.c128_loc = F.pad(
                        self.forward_metadata.c128_loc,
                        (
                            0,
                            forward_batch.batch_size
                            - self.forward_metadata.c128_loc.numel(),
                        ),
                        value=0,
                    )

                    t = self.forward_metadata.actual_seq_lengths_q[-1]
                    self.forward_metadata.positions_cmp_padding_c4 = torch.zeros(
                        min(t, t // 4 + forward_batch.batch_size),
                        dtype=torch.int64,
                        device="npu",
                    )
                    self.forward_metadata.positions_cmp_padding_c128 = torch.zeros(
                        min(t, t // 128 + forward_batch.batch_size),
                        dtype=torch.int64,
                        device="npu",
                    )

                    valid = forward_batch.seq_lens > 0
                    positions_last = torch.clamp(forward_batch.seq_lens - 1, min=0)
                    should_compress = ((forward_batch.seq_lens % 4) == 0) & valid
                    positions_cmp = positions_last[should_compress].to(torch.int64) + (
                        1 - 4
                    )
                    self.forward_metadata.positions_cmp_padding_c4[
                        : positions_cmp.shape[0]
                    ].copy_(positions_cmp)

                    should_compress = ((forward_batch.seq_lens % 128) == 0) & valid
                    positions_cmp = positions_last[should_compress].to(torch.int64) + (
                        1 - 128
                    )
                    self.forward_metadata.positions_cmp_padding_c128[
                        : positions_cmp.shape[0]
                    ].copy_(positions_cmp)
                    self.forward_metadata.start_pos = positions_last.to(torch.int32)
                    self.forward_metadata.seqused = valid.to(torch.int32)
                elif forward_batch.forward_mode.is_prefill():
                    t = self.forward_metadata.actual_seq_lengths_q[-1]
                    self.forward_metadata.positions_cmp_padding_c4 = torch.zeros(
                        min(t, t // 4 + forward_batch.batch_size),
                        dtype=torch.int64,
                        device="npu",
                    )
                    self.forward_metadata.positions_cmp_padding_c128 = torch.zeros(
                        min(t, t // 128 + forward_batch.batch_size),
                        dtype=torch.int64,
                        device="npu",
                    )
                    positions_cmp_dict = {
                        "4": [],
                        "128": [],
                    }
                    for idx in range(
                        self.forward_metadata.actual_seq_lengths_q_cmp.numel() - 1
                    ):
                        start = self.forward_metadata.actual_seq_lengths_q_cmp[idx]
                        end = self.forward_metadata.actual_seq_lengths_q_cmp[idx + 1]
                        seq_len = end - start
                        if seq_len == 0:
                            continue
                        positions = forward_batch.positions[start:end]
                        for ratio, positions_cmp_list in positions_cmp_dict.items():
                            ratio = int(ratio)
                            cutoff = seq_len - (seq_len % ratio)
                            if cutoff > 0:
                                positions_cmp_list.append(positions[:cutoff:ratio])
                    for key, value in positions_cmp_dict.items():
                        if len(value) > 0:
                            value = torch.cat(value, dim=0).long()
                            if key == "4":
                                positions_cmp_padding = (
                                    self.forward_metadata.positions_cmp_padding_c4
                                )
                            else:
                                positions_cmp_padding = (
                                    self.forward_metadata.positions_cmp_padding_c128
                                )
                            assert value.numel() <= positions_cmp_padding.numel()
                            positions_cmp_padding[: value.shape[0]].copy_(value)

                    self.forward_metadata.start_pos = torch.zeros(
                        forward_batch.batch_size, device="npu", dtype=torch.int32
                    )  # TODO support chunk prefill
                    self.forward_metadata.seqused = None

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.forward_metadata.block_tables = (
                self.forward_metadata.block_tables[:, :: self.page_size]
                // self.page_size
            )
            if self.is_hybrid_swa:
                self.forward_metadata.block_tables_swa = (
                    (
                        self.full_to_swa_index_mapping[
                            forward_batch.req_to_token_pool.req_to_token[
                                forward_batch.req_pool_indices, :seq_lens_max
                            ]
                        ][:, :: self.page_size]
                        // self.page_size
                    )
                    .to(torch.int32)
                    .contiguous()
                )
            if self.is_dsv4:
                self.forward_metadata.swa_page_table = (
                    self.forward_metadata.swa_page_table[:, :: self.page_size]
                    // self.page_size
                )
                if not self.is_dsv4_nextn:
                    self.forward_metadata.c4_page_table = (
                        self.forward_metadata.c4_page_table[:, :: self.page_size]
                        // self.page_size
                    )
                    self.forward_metadata.c128_page_table = (
                        self.forward_metadata.c128_page_table[:, :: self.page_size]
                        // self.page_size
                    )
                    self.forward_metadata.c4_state_page_table = (
                        self.forward_metadata.c4_state_page_table[:, :: self.page_size]
                        // self.page_size
                    )
                    self.forward_metadata.c128_state_page_table = (
                        self.forward_metadata.c128_state_page_table[
                            :, :: self.page_size
                        ]
                        // self.page_size
                    )

        if self.is_dsv4:
            self.forward_metadata.kernel_metadata = self.compute_kernel_metadata(
                forward_batch.batch_size,
                self.forward_metadata,
                max_seqlen_q,
                is_nextn=self.is_dsv4_nextn,
            )

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
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
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
                req_indices = forward_batch.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = (
                    req_indices[:seq_len][:: self.page_size] // self.page_size
                )
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.zeros(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
            "swa_page_table": torch.zeros(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ).fill_(-1),
            "c4_page_table": torch.zeros(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ).fill_(-1),
            "c128_page_table": torch.zeros(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ).fill_(-1),
            "c4_state_page_table": torch.zeros(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ).fill_(-1),
            "c128_state_page_table": torch.zeros(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ).fill_(-1),
        }
        if self.is_hybrid_swa:
            self.graph_metadata["block_tables_swa"] = torch.empty(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
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
        metadata = ForwardMetadata()

        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        if self.is_dllm_model:
            max_len = int(seq_lens[:bs].max().item())
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            metadata.block_tables[:bs, :max_seq_pages].copy_(
                (
                    self.req_to_token[req_pool_indices[:bs], :max_len][
                        :, :: self.page_size
                    ]
                    // self.page_size
                ).to(torch.int32)
            )
            metadata.block_tables[:bs, max_seq_pages:].fill_(0)
            metadata.block_tables[bs:, :].fill_(0)

        if self.is_hybrid_swa:
            metadata.block_tables_swa = self.graph_metadata["block_tables_swa"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
            tokens_per_bs = self.speculative_num_draft_tokens
        else:
            tokens_per_bs = 1

        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
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
                [1 + i * 1 for i in range(bs)],
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
            # In the MLA architecture, the FIA kernel requires the head count to be a power of 2.
            # Therefore, we pad the head dimension accordingly and initialize an empty tensor for padding.
            metadata.nope_padding = torch.empty(
                [
                    bs,
                    1,
                    self.q_head_num_padding - self.tp_q_head_num,
                    self.kv_lora_rank,
                ],
                dtype=(
                    self.model_dtype if self.model_dtype is not None else torch.bfloat16
                ),
                device=seq_lens.device,
            )
            metadata.rope_padding = torch.empty(
                [
                    bs,
                    1,
                    self.q_head_num_padding - self.tp_q_head_num,
                    self.qk_rope_head_dim,
                ],
                dtype=(
                    self.model_dtype if self.model_dtype is not None else torch.bfloat16
                ),
                device=seq_lens.device,
            )

        if self.is_dsv4:
            metadata.swa_page_table = self.graph_metadata["swa_page_table"][:bs, :]
            c4_padding_num = min(
                bs * tokens_per_bs,
                bs * tokens_per_bs // 4 + bs,
            )
            c128_padding_num = min(
                bs * tokens_per_bs,
                bs * tokens_per_bs // 128 + bs,
            )
            metadata.swa_loc = torch.zeros(bs * tokens_per_bs, dtype=torch.int64).npu()
            if not self.is_dsv4_nextn:
                metadata.c4_page_table = self.graph_metadata["c4_page_table"][:bs, :]
                metadata.c128_page_table = self.graph_metadata["c128_page_table"][
                    :bs, :
                ]
                metadata.c4_state_page_table = self.graph_metadata[
                    "c4_state_page_table"
                ][:bs, :]
                metadata.c128_state_page_table = self.graph_metadata[
                    "c128_state_page_table"
                ][:bs, :]
                metadata.c4_loc = torch.zeros(c4_padding_num, dtype=torch.int64).npu()
                metadata.c4_state_loc = torch.zeros(
                    bs * tokens_per_bs, dtype=torch.int64
                ).npu()
                metadata.c128_loc = torch.zeros(
                    c128_padding_num, dtype=torch.int64
                ).npu()
                metadata.c128_state_loc = torch.zeros(
                    bs * tokens_per_bs, dtype=torch.int64
                ).npu()

            metadata.actual_seq_lengths_q = torch.arange(
                tokens_per_bs,
                tokens_per_bs * bs + tokens_per_bs,
                tokens_per_bs,
                device="npu",
                dtype=torch.int32,
            )
            metadata.actual_seq_lengths_q_pa = torch.arange(
                0,
                bs * tokens_per_bs + tokens_per_bs,
                tokens_per_bs,
                device="npu",
                dtype=torch.int32,
            )
            metadata.actual_seq_lengths_kv = seq_lens.clone().to(torch.int32)

            metadata.actual_seq_lengths_q_cmp = torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            )

            kernel_metadata = {}
            c_metadata = torch.zeros(1024, dtype=torch.int32).npu()

            kernel_metadata["c1a_metadata"] = c_metadata.clone()
            if not self.is_dsv4_nextn:
                kernel_metadata["c4a_metadata"] = c_metadata.clone()
                kernel_metadata["c128a_metadata"] = c_metadata.clone()
                kernel_metadata["li_quant_metadata"] = c_metadata.clone()

                metadata.start_pos = torch.zeros(bs, dtype=torch.int32).npu()
                metadata.seqused = torch.zeros(bs, dtype=torch.int32).npu()
                metadata.positions_cmp_padding_c4 = torch.zeros(
                    c4_padding_num, dtype=torch.int32
                ).npu()
                metadata.positions_cmp_padding_c128 = torch.zeros(
                    c128_padding_num, dtype=torch.int32
                ).npu()

            metadata.kernel_metadata = kernel_metadata

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

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
        out_cache_loc_dsv4: OutCacheLoc = None,
        positions: torch.Tensor = None,
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        if forward_mode.is_target_verify():
            max_len += self.speculative_num_draft_tokens
        elif forward_mode.is_decode_or_idle() and spec_info is not None:
            max_len += self.speculative_step_id + 1
        if not self.is_dsv4:
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size

            kv_indices = self.req_to_token[req_pool_indices[:bs], :max_len]
            if self.is_hybrid_swa:
                metadata.block_tables_swa[:bs, :max_seq_pages].copy_(
                    self.full_to_swa_index_mapping[
                        self.req_to_token[req_pool_indices[:bs], :max_len]
                    ][:, :: self.page_size]
                    // self.page_size
                )
                metadata.block_tables_swa[:bs, max_seq_pages:].fill_(0)
                metadata.block_tables_swa[bs:, :].fill_(0)
            metadata.block_tables[:bs, :max_seq_pages].copy_(
                kv_indices[:, :: self.page_size] // self.page_size
            )
            metadata.block_tables[:bs, max_seq_pages:].fill_(0)
            metadata.block_tables[bs:, :].fill_(0)

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
            seq_lens_cpu = seq_lens_cpu + self.speculative_num_draft_tokens
        elif forward_mode.is_decode_or_idle() and spec_info is not None:
            seq_lens = seq_lens + self.speculative_step_offset
            seq_lens_cpu = seq_lens_cpu + self.speculative_step_offset

        metadata.seq_lens_cpu = seq_lens_cpu

        seq_len_for_this_batch = seq_lens[:bs].to(torch.int32)
        metadata.seq_lens[:bs].copy_(seq_len_for_this_batch)
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
            tokens_per_bs = self.speculative_num_draft_tokens
        else:
            tokens_per_bs = 1
        if self.is_dsv4:
            metadata.actual_seq_lengths_kv.fill_(0)
            metadata.actual_seq_lengths_kv[:bs].copy_(seq_len_for_this_batch)
            if not self.is_dsv4_nextn:
                if positions.numel() > 0:
                    metadata.start_pos.copy_(
                        torch.clamp(seq_lens - tokens_per_bs, min=0).to(torch.int32)
                    )
                else:
                    metadata.start_pos.fill_(0)
                last_token_idx = metadata.actual_seq_lengths_q_pa[:-1]

                if forward_mode.is_target_verify():
                    valid = (
                        seq_lens - self.speculative_num_draft_tokens
                    ) > 0  # pre-seq_lens
                else:
                    valid = seq_lens > 0

                metadata.seqused.copy_((valid.to(torch.int32)) * tokens_per_bs)

                metadata.actual_seq_lengths_q_cmp.copy_(
                    metadata.actual_seq_lengths_q_pa
                )

                if not forward_mode.is_target_verify():
                    metadata.positions_cmp_padding_c4.fill_(0)
                    if positions.numel() > 0:
                        positions_last = positions[last_token_idx]
                        should_compress = ((seq_lens % 4) == 0) & valid
                        positions_cmp = positions_last[should_compress] + (1 - 4)
                        metadata.positions_cmp_padding_c4[
                            : positions_cmp.shape[0]
                        ].copy_(positions_cmp)

                    metadata.positions_cmp_padding_c128.fill_(0)
                    if positions.numel() > 0:
                        should_compress = ((seq_lens % 128) == 0) & valid
                        positions_cmp = positions_last[should_compress] + (1 - 128)
                        metadata.positions_cmp_padding_c128[
                            : positions_cmp.shape[0]
                        ].copy_(positions_cmp)
                assert req_pool_indices.numel() == bs

            req_pool_indices = req_pool_indices[:bs]
            src = (
                self.req_to_token_swa[req_pool_indices, :max_len][:, :: self.page_size]
                // self.page_size
            )
            metadata.swa_page_table.fill_(-1)
            metadata.swa_page_table[: src.shape[0], : src.shape[1]].copy_(src)

            swa_loc = out_cache_loc_dsv4.out_swa_loc
            metadata.swa_loc[: swa_loc.shape[0]].copy_(swa_loc)
            metadata.swa_loc[swa_loc.shape[0] :].fill_(0)

            if not self.is_dsv4_nextn:
                src = (
                    self.req_to_token_c4[req_pool_indices, : max_len // 4][
                        :, :: self.page_size
                    ]
                    // self.page_size
                )
                metadata.c4_page_table.fill_(-1)
                metadata.c4_page_table[: src.shape[0], : src.shape[1]].copy_(src)

                src = (
                    self.req_to_token_c128[req_pool_indices, : max_len // 128][
                        :, :: self.page_size
                    ]
                    // self.page_size
                )
                metadata.c128_page_table.fill_(-1)
                metadata.c128_page_table[: src.shape[0], : src.shape[1]].copy_(src)

                src = (
                    self.req_to_token_c4_state[req_pool_indices, :max_len][
                        :, :: self.page_size
                    ]
                    // self.page_size
                )
                metadata.c4_state_page_table.fill_(-1)
                metadata.c4_state_page_table[: src.shape[0], : src.shape[1]].copy_(src)

                src = (
                    self.req_to_token_c128_state[req_pool_indices[:bs], :max_len][
                        :, :: self.page_size
                    ]
                    // self.page_size
                )
                metadata.c128_state_page_table.fill_(-1)
                metadata.c128_state_page_table[: src.shape[0], : src.shape[1]].copy_(
                    src
                )

                c4_loc = out_cache_loc_dsv4.out_c4_loc
                metadata.c4_loc[: c4_loc.shape[0]].copy_(c4_loc)
                metadata.c4_loc[c4_loc.shape[0] :].fill_(
                    0
                )  # 0: put padding data at the first location, which is unused position

                c128_loc = out_cache_loc_dsv4.out_c128_loc
                metadata.c128_loc[: c128_loc.shape[0]].copy_(c128_loc)
                metadata.c128_loc[c128_loc.shape[0] :].fill_(0)

                c4_state_loc = out_cache_loc_dsv4.out_c4_state_loc
                metadata.c4_state_loc[: c4_state_loc.shape[0]].copy_(c4_state_loc)
                metadata.c4_state_loc[c4_state_loc.shape[0] :].fill_(0)

                c128_state_loc = out_cache_loc_dsv4.out_c128_state_loc
                metadata.c128_state_loc[: c128_state_loc.shape[0]].copy_(c128_state_loc)
                metadata.c128_state_loc[c128_state_loc.shape[0] :].fill_(0)

            kernel_metadata = self.compute_kernel_metadata(
                bs, metadata, max_seqlen_q=tokens_per_bs, is_nextn=self.is_dsv4_nextn
            )
            metadata.kernel_metadata["c1a_metadata"].copy_(
                kernel_metadata["c1a_metadata"]
            )
            if not self.is_dsv4_nextn:
                metadata.kernel_metadata["c4a_metadata"].copy_(
                    kernel_metadata["c4a_metadata"]
                )
                metadata.kernel_metadata["c128a_metadata"].copy_(
                    kernel_metadata["c128a_metadata"]
                )
                metadata.kernel_metadata["li_quant_metadata"].copy_(
                    kernel_metadata["li_quant_metadata"]
                )

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

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: Union[torch.Tensor, List[torch.Tensor]],
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
        sinks: Optional[torch.Tensor] = None,
    ):

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )
        if self.is_dsv4:
            assert save_kv_cache == False
            compress_ratio = self.compress_ratios[layer.layer_id]
            if self.is_dsv4_nextn:
                compress_page_table = None
            elif compress_ratio == 4:
                compress_page_table = (
                    forward_batch.attn_backend.forward_metadata.c4_page_table
                )
            elif compress_ratio == 128:
                compress_page_table = (
                    forward_batch.attn_backend.forward_metadata.c128_page_table
                )
            else:
                compress_page_table = None
            swa_page_table = forward_batch.attn_backend.forward_metadata.swa_page_table

            if is_prefill:
                use_pa_prefill = get_bool_env_var("USE_PA_PREFILL")
                if use_pa_prefill:
                    kv_pad = k.unflatten(0, (-1, self.page_size))
                    cmp_kv = forward_batch.token_to_kv_pool.get_compress_buffer(
                        layer.layer_id, False
                    )  #
                    ori_kv = forward_batch.token_to_kv_pool.get_swa_buffer(
                        layer.layer_id
                    )  # [num_block, page_size, 1, dim]
                    metadata = self.forward_metadata.kernel_metadata[
                        f"c{compress_ratio}a_metadata"
                    ]
                    q = q.squeeze(0)
                    attn_kwargs = {
                        "cu_seqlens_q": self.forward_metadata.actual_seq_lengths_q_pa,  # just for TND
                        "seqused_kv": self.forward_metadata.actual_seq_lengths_kv,  # num of key elements used, DT_INT32, Tentative: kv_len 非压缩
                        "ori_mask_mode": 4,  # sliding window
                        "ori_win_left": self.config.sliding_window_size - 1,
                        "ori_win_right": 0,
                        "layout_q": "TND",  # "BSND" , "TND"
                        "layout_kv": "PA_ND",  # "PA_ND"
                        "q": q,
                        "ori_kv": kv_pad,  # get from past_key_values, prefill is full cache, transfer bsnd to bbnd TODO： full cacheTODO：这里需要从0开始（最小index）
                        "ori_block_table": swa_page_table,  # TODO full cache in prefill stage
                        "sinks": sinks,
                        "metadata": metadata,  # get from operator ?----sparse_attn_sharedkv_metadata
                        "softmax_scale": layer.scaling,
                    }

                    if compress_ratio == 4:
                        topk_indices = topk_indices.view(-1, 1, topk_indices.shape[-1])
                        c_kwargs = {
                            "cmp_ratio": compress_ratio,  # no support 1, None
                            "cmp_mask_mode": 3,  # causal
                            "cmp_kv": cmp_kv,  # get from past_key_values, TODO when no cmp_kv , swa：None
                            "cmp_sparse_indices": topk_indices,  # for LI, need to set TODO：c128传None， C4需要view一下 TODO：这里需要从0开始（最小index）
                            "cmp_block_table": compress_page_table,
                        }
                        attn_kwargs = attn_kwargs | c_kwargs
                    elif compress_ratio == 128:
                        c_kwargs = {
                            "cmp_ratio": compress_ratio,  # no support 1, None
                            "cmp_mask_mode": 3,  # causal
                            "cmp_kv": cmp_kv,  # get from past_key_values, TODO when no cmp_kv , swa：None
                            "cmp_sparse_indices": None,  # for LI, need to set TODO：c128传None， C4需要view一下 TODO：这里需要从0开始（最小index）
                            "cmp_block_table": compress_page_table,
                        }
                        attn_kwargs = attn_kwargs | c_kwargs

                    o, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
                else:
                    split_seq = forward_batch.seq_lens_cpu
                    split_seq = split_seq.tolist()

                    if isinstance(k, (list, tuple)):
                        k_list = k
                    else:
                        k_list = k.split(split_seq, dim=0)
                    o = q.new_empty(q.shape)
                    q_list = q.split(split_seq, dim=0)
                    topk_indices_list = topk_indices.split(split_seq, dim=0)

                    offset = 0
                    for q_i, k_i, topk_indices_i in zip(
                        q_list, k_list, topk_indices_list
                    ):
                        q_i = q_i.unsqueeze(0)  # BSND
                        k_i = k_i.view(1, k_i.shape[0], k_i.shape[-1])  # T1D --> BSD
                        topk_indices_i = topk_indices_i.unsqueeze(0)  # BSK
                        o[offset : offset + q_i.shape[1]] = dsv4_sparse_attn(
                            q_i, k_i, sinks, topk_indices_i, layer.scaling
                        ).flatten(0, 1)
                        offset += q_i.shape[1]

            else:
                o = q.new_empty((q.shape[0], q.shape[1], layer.head_dim))

                use_pa_decode = get_bool_env_var("USE_PA_DECODE")
                if use_pa_decode:
                    cmp_kv = forward_batch.token_to_kv_pool.get_compress_buffer(
                        layer.layer_id, False
                    )
                    ori_kv = forward_batch.token_to_kv_pool.get_swa_buffer(
                        layer.layer_id
                    )  # [T, 1, D]
                    metadata = self.forward_metadata.kernel_metadata[
                        f"c{compress_ratio}a_metadata"
                    ]

                    attn_kwargs = {
                        "cu_seqlens_q": self.forward_metadata.actual_seq_lengths_q_pa,  # just for TND
                        "seqused_kv": self.forward_metadata.actual_seq_lengths_kv,  # num of key elements used, DT_INT32, Tentative: kv_len 非压缩
                        "ori_mask_mode": 4,  # sliding window
                        "ori_win_left": self.config.sliding_window_size - 1,
                        "ori_win_right": 0,
                        "layout_q": "TND",  # "BSND" , "TND"
                        "layout_kv": "PA_ND",  # "PA_ND"
                        "q": q,
                        "ori_kv": ori_kv,  # get from past_key_values, prefill is full cache, transfer bsnd to bbnd TODO： full cacheTODO：这里需要从0开始（最小index）
                        "ori_block_table": swa_page_table,  # TODO full cache in prefill stage
                        "sinks": sinks,
                        "metadata": metadata,  # get from operator ?----sparse_attn_sharedkv_metadata
                        "softmax_scale": layer.scaling,
                    }

                    if compress_ratio == 4:
                        topk_indices = topk_indices.view(-1, 1, topk_indices.shape[-1])
                        c_kwargs = {
                            "cmp_ratio": compress_ratio,  # no support 1, None
                            "cmp_mask_mode": 3,  # causal
                            "cmp_kv": cmp_kv,  # get from past_key_values, TODO when no cmp_kv , swa：None
                            "cmp_sparse_indices": topk_indices,  # for LI, need to set TODO：c128传None， C4需要view一下 TODO：这里需要从0开始（最小index）
                            "cmp_block_table": compress_page_table,
                        }
                        attn_kwargs = attn_kwargs | c_kwargs
                    elif compress_ratio == 128:
                        c_kwargs = {
                            "cmp_ratio": compress_ratio,  # no support 1, None
                            "cmp_mask_mode": 3,  # causal
                            "cmp_kv": cmp_kv,  # get from past_key_values, TODO when no cmp_kv , swa：None
                            "cmp_sparse_indices": None,  # for LI, need to set TODO：c128传None， C4需要view一下 TODO：这里需要从0开始（最小index）
                            "cmp_block_table": compress_page_table,
                        }
                        attn_kwargs = attn_kwargs | c_kwargs
                    o, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
                else:
                    topk_indices = topk_indices.unflatten(
                        0, (forward_batch.batch_size, -1)
                    )
                    q = q.unflatten(0, (forward_batch.batch_size, -1))  # BSND
                    for idx, seq_len in enumerate(forward_batch.seq_lens_cpu):
                        if seq_len <= 0:
                            continue
                        q_i = q[idx : idx + 1]
                        topk_indices_i = topk_indices[idx : idx + 1]

                        win_indices = get_kv_indices(
                            forward_batch,
                            layer.sliding_window_size,
                            swa_page_table,
                            idx,
                            seq_len,
                        )
                        buffer_2 = None
                        if compress_ratio > 1:
                            compress_indices = get_kv_indices(
                                forward_batch,
                                seq_len // compress_ratio,
                                compress_page_table,
                                idx,
                                seq_len // compress_ratio,
                            )
                            buffer_2 = (
                                forward_batch.token_to_kv_pool.get_compress_buffer(
                                    layer.layer_id, False, compress_indices
                                )
                            )  # [T//4, 1, D] or [T//128, 1, D]

                        buffer_1 = forward_batch.token_to_kv_pool.get_swa_buffer(
                            layer.layer_id, win_indices
                        )

                        if buffer_2 is not None and buffer_2.shape[0] > 0:
                            k_i = (
                                torch.cat([buffer_1, buffer_2], dim=0)
                                .squeeze(1)
                                .unsqueeze(0)
                            )  # BSD
                        else:
                            k_i = buffer_1.squeeze(1).unsqueeze(0)  # BSD

                        o[idx : idx + 1] = dsv4_sparse_attn(
                            q_i, k_i, sinks, topk_indices_i, layer.scaling
                        ).flatten(0, 1)

            return o

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

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
                elif forward_batch.forward_mode.is_draft_extend():
                    actual_seq_qlen = (
                        forward_batch.extend_seq_lens.cumsum()
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
            and is_nsa_enable_prefill_cp()
            and forward_batch.nsa_cp_metadata is not None
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
        if self.is_dsv4 or topk_indices is not None:
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
                sinks,
            )
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
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
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None:
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self.is_hybrid_swa and layer.sliding_window_size != -1:
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
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

            if self.use_fia:
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
                        forward_batch.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        forward_batch.encoder_lens,
                        is_cross_attention=layer.is_cross_attention,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
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

                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
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
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
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
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, kv_c, k_rope
                    )
                attn_output = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, kv_lora_rank)
                )
                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                kv_cache = torch.cat([k_cache, v_cache], dim=-1)
                attn_output = self.native_attn.run_sdpa_forward_extend(
                    q,
                    attn_output,
                    kv_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    k_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    forward_batch.req_to_token_pool.req_to_token,
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
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
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
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
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
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + query.shape[0],
                    self.speculative_num_draft_tokens,
                )

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                atten_mask=self.mtp_mask,
                scale=layer.scaling,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                sparse_mode=3,
            )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
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
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
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
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + q_nope.shape[0],
                    self.speculative_num_draft_tokens,
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
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
                num_heads=layer.tp_q_head_num,
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
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if sinks is not None:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            # Use SWA block tables if hybrid SWA is enabled for this layer
            if self.is_hybrid_swa and layer.sliding_window_size != -1:
                block_tables = self.forward_metadata.block_tables_swa
            else:
                block_tables = self.forward_metadata.block_tables
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
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            num_tokens = query.shape[0]
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
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
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
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
    ):
        if is_mla_preprocess_enabled() and self.use_mla:
            # MLAPO does saving kv_cache
            save_kv_cache = False
        if self.is_dsv4 or topk_indices is not None:
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
                sinks,
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
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            num_tokens = q.shape[0]
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None:
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self.is_hybrid_swa and layer.sliding_window_size != -1:
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
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
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
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
                        -1, self.page_size, layer.tp_v_head_num * layer.qk_head_dim
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
                    forward_batch.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.encoder_lens,
                    is_cross_attention=layer.is_cross_attention,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=False,
                    logit_cap=layer.logit_cap,
                    logit_capping_method=layer.logit_capping_method,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

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
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
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

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

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

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        out_cache_loc_dsv4 = forward_batch.out_cache_loc_dsv4.reshape_(
            self.topk, self.speculative_num_steps
        )

        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc_dsv4=out_cache_loc_dsv4.index_(i),  # todo
            )

        self.common_template(forward_batch, call_fn)
