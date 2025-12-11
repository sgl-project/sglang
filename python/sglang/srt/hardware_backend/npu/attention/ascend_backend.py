from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating_v3
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update_npu,
)
from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu

logger = logging.getLogger(__name__)


@dataclass
class ForwardMetadata:

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

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
        mixed_chunk_cache_len = 8192
        self.mix_mask_cache = self.generate_attn_mask(mixed_chunk_cache_len, "mix")
        self.mix_seq_len_cached = self.mix_mask_cache.shape[0]

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
        position: torch.Tensor = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate a splitfuse attention mask.

        :param seq_lens: Sequence lengths.
        :param position: Position indices for the mask.
        :param dtype: Data type of the mask tensor.
        :param device: Device to run the model on.
        :return: A tensor representing the splitfuse attention mask.
        """
        if dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("splitfuse_attn_mask now only supports bf16 and fp16")
        max_seq_len = max(seq_lens, default=0)
        self.mix_mask_cache, self.mix_seq_len_cached = self.update_attn_cache(
            max_seq_len, self.mix_mask_cache, self.mix_seq_len_cached, dtype, mode="mix"
        )
        attn_mask = torch.index_select(self.mix_mask_cache, dim=0, index=position)[
            :, :max_seq_len
        ]
        return attn_mask.contiguous().to(device, non_blocking=True)

    def update_mask(self, forward_metadata):
        """
        Update the splitfuse attention mask based on forward metadata.

        :param forward_metadata: Forward metadata containing sequence lengths and extended lengths.
        :return: Updated splitfuse attention mask.
        """
        attn_mask_id = self.get_attention_mask_id(
            forward_metadata.seq_lens_cpu_int,
            forward_metadata.extend_seq_lens_cpu_int,
        )
        mix_mask = self.get_splitfuse_attn_mask(
            seq_lens=forward_metadata.seq_lens_cpu_int,
            position=attn_mask_id,
            dtype=torch.float16,
            device=self.device,
        ).to(torch.bfloat16)
        return mix_mask


class AscendAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.q_head_dim = (
                self.qk_rope_head_dim + model_runner.model_config.qk_nope_head_dim
            )
        self.native_attn = TorchNativeAttnBackend(model_runner)
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
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
            self.ascend_attn_mask_builder.mix_mask_cache,
        )
        if self.use_mla:
            self.ringmla_mask = self.ascend_attn_mask_builder.ringmla_mask

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

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        seq_lens_max = forward_batch.seq_lens.max()
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens

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

        if forward_batch.forward_mode.is_mixed():
            self.mix_mask = self.ascend_attn_mask_builder.update_mask(
                self.forward_metadata
            )

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            if self.forward_metadata.seq_lens_cpu_int is None:
                self.forward_metadata.actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_list
                )
            else:
                self.forward_metadata.actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            if forward_batch.forward_mode.is_draft_extend():
                self.forward_metadata.actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_len = forward_batch.input_ids[
                    : forward_batch.num_token_non_padded_cpu
                ].shape[0]
                self.forward_metadata.actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + actual_len,
                    self.speculative_num_draft_tokens,
                ).tolist()
        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }

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
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
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
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        if forward_mode.is_target_verify():
            max_len += self.speculative_num_draft_tokens
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)
        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

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
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        block_table = self.forward_metadata.block_tables
        if is_prefill:
            actual_seq_qlen = torch.cumsum(forward_batch.seq_lens, dim=0)
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
        if self.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int

        attn_out = torch.ops.custom.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope,
            value=k_nope,
            query_rope=q_pe,
            key_rope=k_pe,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen.to(torch.int32),
            actual_seq_lengths_kv=actual_seq_lengths_kv.to(q.device),
            block_table=block_table,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
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
    ):
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
            or forward_batch.forward_mode.is_draft_extend()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):

            if is_mla_preprocess_enabled():
                save_kv_cache = False
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
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

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
                if layer.qk_head_dim <= 128:
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
                    if layer.qk_head_dim != layer.v_head_dim:
                        attn_output = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        attn_output = torch.empty_like(q)

                    use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                    q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                    causal = True
                    if (
                        layer.is_cross_attention
                        or layer.attn_type == AttentionType.ENCODER_ONLY
                    ):
                        causal = False

                    self.native_attn._run_sdpa_forward_extend(
                        q_,
                        o_,
                        k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        forward_batch.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
                    )
        elif sum(forward_batch.extend_prefix_lens_cpu) > 0:
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]
            q_nope, q_rope = q.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)

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
            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
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
        else:
            assert (
                layer.qk_head_dim != layer.v_head_dim
            ), "FIA only supports qk_head_dim != v_head_dim"

            num_token_padding = q.shape[0]
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]

            q_nope, q_rope = q.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)

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

            attn_output = attn_output.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)
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
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.to(dtype=torch.int32)
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

        if self.use_mla:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
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
        else:
            q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            num_token_padding = q.shape[0]
            seq_len = num_token_padding // forward_batch.batch_size
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                q = q[: forward_batch.num_token_non_padded_cpu]

            k, v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_cache = k.view(
                -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
            )
            v_cache = v.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            q = q.view(-1, seq_len, layer.tp_q_head_num, layer.qk_head_dim)
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                q,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=self.forward_metadata.actual_seq_lengths_kv,
            )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if (
                not self.graph_mode
            ) and forward_batch.num_token_non_padded_cpu != num_token_padding:
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
            num_tokens = q.shape[0]
            """PA will support bs<tp in the later version of CANN"""
            if self.use_fia or num_tokens < get_attention_tp_size():
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
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
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
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = torch.from_numpy(
                        np.array(self.forward_metadata.seq_lens_cpu_list).astype(
                            np.int32
                        )
                    )
                else:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_int

                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=actual_seq_len_kv,
                    out=attn_output,
                )
                return attn_output.view(
                    num_tokens, layer.tp_q_head_num * layer.v_head_dim
                )
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_rope_cache = k_rope.view(
                -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
            )
            c_kv_cache = c_kv.view(
                -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
            )

            q_nope = q.view(-1, layer.tp_q_head_num, 1, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, 1, self.qk_rope_head_dim)
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
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BNSD",
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
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BNSD",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
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
    ):
        if is_mla_preprocess_enabled():
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

        if self.graph_mode:
            return self.forward_decode_graph(
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
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            num_tokens = q.shape[0]
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            if self.use_fia:
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
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                    scale=layer.scaling,
                )
            else:
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
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

        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        # Initialize the output tensor for attention results
        attn_output = torch.empty(
            (query.shape[0], layer.tp_q_head_num, layer.v_head_dim),
            dtype=query.dtype,
            device=query.device,
        )

        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=k_cache,
            value_cache=v_cache,
            block_table=self.forward_metadata.block_tables,
            context_lens=self.forward_metadata.seq_lens_cpu_int,
            mask=self.mix_mask,
            seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
            scale_value=layer.scaling,
            num_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            out=attn_output,
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
        for _ in range(self.speculative_num_steps):
            self.attn_backends.append(AscendAttnBackend(model_runner))

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
            )

        self.common_template(forward_batch, call_fn)


class AscendGDNAttnBackend(MambaAttnBackendBase):
    """
    Attention backend using Mamba kernel.
    Adapted from layers.attention.hybrid_linear_attn_backend.GDNAttnBackend
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.enable_ascendc_fusion_gdn = get_bool_env_var(
            "ENABLE_ASCENDC_FUSION_GDN", "false"
        )
        self.num_accepted_tokens = None
        self.actual_seq_lengths = None
        self.ssm_state_indices = None
        self.graph_mode = False

    def prepare_gdn_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accepted_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            # indices
            start_indices = cache_indices * seq_len
            offset = torch.arange(seq_len, device=start_indices.device)
            ranges = start_indices.unsqueeze(1) + offset
            self.ssm_state_indices = ranges.flatten().to(torch.int32)
        else:
            self.ssm_state_indices = cache_indices

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend(True):
            return
        super().init_forward_metadata(forward_batch)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self.graph_mode = False

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
        if forward_mode.is_draft_extend(True):
            return
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        self.prepare_gdn_inputs(bs, forward_mode, spec_info)
        self.graph_mode = True

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
        if forward_mode.is_draft_extend(True):
            return
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )
        self.prepare_gdn_inputs(bs, forward_mode, spec_info)
        self.graph_mode = True

    def torch_causal_conv1d_update_npu(
        self,
        hidden_state: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_update: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        bsz, hidden_size, seq_len = hidden_state.shape
        state_len = conv_state.shape[-1]
        hidden_states_new = torch.cat([conv_state, hidden_state], dim=-1).to(
            weight.dtype
        )
        if conv_state_update is not None:
            for i in range(seq_len):
                end = i - seq_len + 1
                start = end - state_len
                slice_range = slice(start, end if end != 0 else None)
                conv_state_update[:, i] = hidden_states_new[:, :, slice_range]
        else:
            conv_state_update = hidden_states_new[:, :, -state_len:]
        out = F.conv1d(
            hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
        )
        out = F.silu(out[:, :, -seq_len:])
        out = out.to(hidden_state.dtype)
        return out, conv_state_update

    def fused_recurrent_gated_delta_rule_update_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state: Optional[torch.Tensor] = None,
    ):
        _, num_heads, head_k_dim = query.shape  # T, N, D
        _, num_value_heads, head_v_dim = value.shape
        beta = beta.view(-1, num_value_heads).to(torch.bfloat16)
        g = g.view(-1, num_value_heads).to(torch.float32)
        batch_size = cache_indices.shape[0]
        seq_len = query.shape[0] // batch_size
        scale = 1 / (head_k_dim**0.5)

        if intermediate_state is not None:
            # MTP
            intermediate_state[cache_indices, 0] = recurrent_state[cache_indices]
            ssm_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )
        else:
            ssm_state = recurrent_state

        attn_core_out = torch_npu.npu_recurrent_gated_delta_rule(
            query,
            key,
            value,
            ssm_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=self.actual_seq_lengths,
            ssm_state_indices=self.ssm_state_indices,
            num_accepted_tokens=self.num_accepted_tokens,
            g=g,
        )

        if intermediate_state is not None:
            intermediate_state = ssm_state.view(
                -1, seq_len, num_value_heads, head_k_dim, head_v_dim
            )
        return attn_core_out

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

        mixed_qkv = mixed_qkv.unsqueeze(-1)
        conv_state_update = conv_states[cache_indices]
        mixed_qkv, conv_states[cache_indices] = (
            self.torch_causal_conv1d_update_npu(  # todo
                mixed_qkv,
                conv_state_update,
                conv_weights,
                bias=bias,
            )
        )
        mixed_qkv = mixed_qkv.squeeze(-1)
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
        num_value_heads = value.shape[1] // head_v_dim
        g, beta = fused_gdn_gating_v3(A_log, a, b, dt_bias)

        if self.enable_ascendc_fusion_gdn:
            query = query.view(-1, num_heads, head_k_dim)
            key = key.view(-1, num_heads, head_k_dim)
            value = value.view(-1, num_value_heads, head_v_dim)
            query = l2norm_fwd(
                query.contiguous(), eps=1e-6, output_dtype=torch.bfloat16
            )
            key = l2norm_fwd(key.contiguous(), eps=1e-6, output_dtype=torch.bfloat16)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update_npu(
                query,
                key,
                value,
                recurrent_state=ssm_states,
                beta=beta,
                g=g,
                cache_indices=cache_indices,
            )
        else:
            query = query.view(1, seq_len, num_heads, head_k_dim)
            key = key.view(1, seq_len, num_heads, head_k_dim)
            value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

            core_attn_out = fused_sigmoid_gating_delta_rule_update_npu(
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

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        batch_size = cache_indices.shape[0]
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            assert self.enable_ascendc_fusion_gdn, (
                "When enabling MTP, you must enable the Ascendc fusion "
                + 'GDN operator. Please set the environment variable using `export ENABLE_ASCENDC_FUSION_GDN="true"`.'
            )
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            num_token_padding = mixed_qkv.shape[0]
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                mixed_qkv = mixed_qkv[: forward_batch.num_token_non_padded_cpu]
                a = a[: forward_batch.num_token_non_padded_cpu]
                b = b[: forward_batch.num_token_non_padded_cpu]
                seq_len = forward_batch.num_token_non_padded_cpu
            head_dim = mixed_qkv.shape[-1]
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = (
                mixed_qkv.view(-1, draft_token_num, head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            conv_states_to_use = conv_states[cache_indices]
            intermediate_conv_cache = intermediate_conv_window_cache[cache_indices]
            mixed_qkv_processed, conv_state_update = (
                self.torch_causal_conv1d_update_npu(
                    mixed_qkv_reshaped,
                    conv_states_to_use,
                    conv_weights,
                    intermediate_conv_cache,
                    bias,
                )
            )
            intermediate_conv_window_cache[cache_indices] = conv_state_update
            mixed_qkv = (
                mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
            )
        else:
            mixed_qkv = causal_conv1d_fn_npu(
                mixed_qkv.transpose(0, 1),
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

        g, beta = fused_gdn_gating_v3(A_log, a, b, dt_bias)

        if is_target_verify:
            query = query.view(-1, num_heads, head_k_dim)
            key = key.view(-1, num_heads, head_k_dim)
            value = value.view(-1, num_value_heads, head_v_dim)
            query = l2norm_fwd(
                query.contiguous(), eps=1e-6, output_dtype=torch.bfloat16
            )
            key = l2norm_fwd(key.contiguous(), eps=1e-6, output_dtype=torch.bfloat16)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update_npu(
                query,
                key,
                value,
                recurrent_state=ssm_states,
                beta=beta,
                g=g,
                cache_indices=cache_indices,
                intermediate_state=intermediate_state_cache,
            )
            if (not self.graph_mode) and core_attn_out.shape[0] < num_token_padding:
                core_attn_out = torch.cat(
                    [
                        core_attn_out,
                        core_attn_out.new_zeros(
                            num_token_padding - core_attn_out.shape[0],
                            *core_attn_out.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        else:
            recurrent_state = ssm_states[cache_indices]
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule_npu(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            if self.enable_ascendc_fusion_gdn:
                # AscendC GDN fusion operator requires the shape of the recurrent_state to be [b, s, Dv, Dk]
                last_recurrent_state = last_recurrent_state.transpose(-1, -2).to(
                    ssm_states.dtype, copy=False
                )
            else:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
            ssm_states[cache_indices] = last_recurrent_state

        return core_attn_out
