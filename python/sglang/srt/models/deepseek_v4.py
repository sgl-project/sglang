from __future__ import annotations

import concurrent.futures
import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import fused_rope, linear_bf16_fp32
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
    deepseek_v4_moe_code_path_checker,
)
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import LayerScatterModes, get_attn_tp_context
from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.compress_state import (
    CompressStatePool,
    KVAndScore,
    KVAndScoreOld,
)
from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    get_bool_env_var,
    log_info_on_rank0,
    make_layers,
    maybe_torch_compile,
)

logger = logging.getLogger(__name__)

from sglang.srt.environ import envs

MOE_BIT_WISE_EQUAL_MODE = False
ATTN_BIT_WISE_EQUAL_MODE = False
COMPRESSOR_BIT_WISE_EQUAL_MODE = False
_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4Backend
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class DeepseekRefRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        out = rms_normalize_triton(x, self.eps, self.weight)
        return out


@maybe_torch_compile
def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    x *= torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # x / sqrt(mean(x^2) + eps)
    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    """RMS normalize with optional weight.

    Args:
        x: Input tensor of shape (..., dim), normalizes over last dimension
        eps: Epsilon for numerical stability
        weight: Optional weight tensor of shape (dim,)
    """
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        rotary_emb: RotaryEmbedding,
        freqs_cis: torch.Tensor,  # TODO: remove it after using rotary embedding
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.nope_head_dim = head_dim - self.rope_head_dim
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # fuse wkv and wgate into wkv_gate, merge the last dim
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        # self.norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.norm = DeepseekRefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    @cached_property
    def use_fused_compress(self) -> bool:
        if (
            envs.SGLANG_OPT_USE_FUSED_PAGED_COMPRESS.get()
            and envs.SGLANG_OPT_DPSK_V4_RADIX.get()
        ):
            return True
        return (
            envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()
            and not envs.SGLANG_OPT_DPSK_V4_RADIX.get()
        )

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        # ========== copied from the hotfix in "260119-updated" of ref code ==========
        is_model_2604 = envs.SGLANG_DSV4_MODE.get() == "2604"
        if (
            self.overlap
            and not is_model_2604
            and get_bool_env_var("SGLANG_ENABLE_APE_HOTFIX", "1")
        ):
            # NOTE: We reorder the parameters here to match the layout of the provided checkpoint.
            # This is only required for compatibility with this checkpoint; the official version
            # does not need this reordering.
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            if self.use_fused_compress:
                ape = torch.cat([ape[1], ape[0]], dim=0)
            else:
                ape = torch.cat([ape[1], ape[0]], dim=-1)
            self.ape.data.copy_(ape.view(self.ratio, -1))
            # ============================================================================

    def _get_states(self, forward_batch: ForwardBatch) -> KVAndScore:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def _get_state_pool(self, forward_batch: ForwardBatch) -> CompressStatePool:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)

        assert isinstance(ret, CompressStatePool)

        return ret

    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        # tensor: [block_num, r, 2 * d]
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        # NOTE: the default value has been initialized when creating the states
        # tensor: [bs, 2 * r, 2 * d]
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        """Tailing length for the valid states in kv cache.
        When overlap is enabled, there is always an extra block: [extra block, compressing part]
        """
        return seq_len % ratio + (ratio == 4) * ratio

    @staticmethod
    def compute_state_len_indices(seq_len: int, ratio: int):
        state_len = seq_len % ratio + (ratio == 4) * ratio
        # NOTE: -1 here means invalid position
        return torch.arange(seq_len - state_len, seq_len).clamp(min=-1)

    def print_tensor(self, y: torch.Tensor, name: str):
        enable = int(os.environ.get("SGLANG_ENABLE_PRINT_TENSOR", 0))
        if enable:
            print(f"[sgl] {name}: shape={y.shape}, dtype={y.dtype}, device={y.device}")
            print(f"{y.flatten()[:10]}...{y.flatten()[-10:]}")

    def compress_extend_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ):
        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4Backend)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        # extract some info
        state_pool = self._get_state_pool(forward_batch)
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        assert not self.forward_mode.is_target_verify()

        assert extend_lens is not None and prefix_lens is not None
        device = kv_and_scores.kv.device

        # Deliberately fill w/ huge values, s.t. when misuse and access the unfilled values,
        # we have higher probability to see something very weird
        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=device,
        )

        bs = forward_batch.batch_size
        pt = 0
        for i in range(bs):
            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            pre_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i], ratio=self.ratio
            ).to(device)
            raw_loc = torch.where(
                pre_state_indices < 0,
                -1,
                req_to_token[req_pool_indices[i], pre_state_indices],
            )
            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(raw_loc)
            state_loc = state_pool.translate_from_swa_loc_to_state_loc(swa_loc)
            pre_kv_state = state_pool.get_state_by_state_loc(state_loc)
            kv_and_score_buffer = KVAndScore.cat([pre_kv_state, kv_and_score], dim=0)
            valid_kv_len = kv_and_score_buffer.kv.size(0)

            post_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i] + extend_lens[i], ratio=self.ratio
            ).to(device)
            post_state_len = post_state_indices.size(0)

            # write to kv_and_score_states
            assert post_state_len <= valid_kv_len
            post_raw_loc = torch.where(
                post_state_indices < 0,
                -1,
                req_to_token[req_pool_indices[i], post_state_indices],
            )
            post_swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(post_raw_loc)
            post_state_loc = state_pool.translate_from_swa_loc_to_state_loc(
                post_swa_loc
            )
            post_state_to_set = kv_and_score_buffer[valid_kv_len - post_state_len :]
            state_pool.set_state_by_state_loc(post_state_loc, post_state_to_set)

            # Get the part that can be compressed (ratio-aligned)
            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                # Nothing to compress yet, just update pointers
                pt += extend_lens[i]
                continue

            # kv to compress: [compressed_len, ratio, head_dim * coff]
            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            # NOTE: apply ape only when compressing
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

            # Apply overlap transformation if enabled
            if self.overlap:
                new_kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, fill_value=0
                )
                new_score = self.overlap_transform(
                    kv_and_score_to_compress.score, fill_value=float("-inf")
                )
                kv_and_score_to_compress = KVAndScore.from_kv_score(
                    kv=new_kv, score=new_score
                )
                del new_kv, new_score
                # remove the first block before compression
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)

            # NOTE: ref code requires dtype as the same as hidden states (float32)
            # the raw output of kv_compressed is float32 already
            assert kv_compressed.dtype == torch.float32
            kv_compressed = self.norm(kv_compressed)

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            # get all the pos: ratio * n + (ratio - 1) > prefix_len - 1
            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    def compress_decode_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ):
        """Paged and cudagraph compatible version of compress_decode"""
        assert self.ape_converted
        state_pool = self._get_state_pool(forward_batch)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_target_verify():
            draft_tokens = forward_batch.attn_backend.speculative_num_draft_tokens
            offsets = torch.arange(1, draft_tokens + 1, device=seq_lens.device)
            seq_lens_2d = seq_lens[:, None] + offsets[None, :]
            seq_lens = seq_lens_2d.view(-1)
            req_pool_indices = req_pool_indices.repeat_interleave(draft_tokens)

        raw_locs = req_to_token[req_pool_indices, seq_lens - 1]

        # Update the new decode states
        swa_locs = token_to_kv_pool.translate_loc_from_full_to_swa(raw_locs)
        state_locs = state_pool.translate_from_swa_loc_to_state_loc(swa_locs)
        state_pool.set_state_by_state_loc(state_locs, kv_and_scores)

        compress_bulk_len = self.ratio * self.coff
        compress_indices = seq_lens[:, None] + torch.arange(
            -compress_bulk_len, 0, device=seq_lens.device
        )
        compress_indices.clamp_(min=-1)
        compress_indices_raw = torch.where(
            compress_indices < 0,
            -1,
            req_to_token[req_pool_indices[:, None], compress_indices],
        )
        compress_indices_swa = token_to_kv_pool.translate_loc_from_full_to_swa(
            compress_indices_raw
        )
        compress_indices_state = state_pool.translate_from_swa_loc_to_state_loc(
            compress_indices_swa
        )
        kv_and_score_to_compress = state_pool.get_state_by_state_loc(
            compress_indices_state.view(-1)
        ).view(-1, self.ratio, self.coff * self.head_dim)
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        bs = seq_lens.size(0)
        if self.overlap:
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        self.print_tensor(kv_and_score_to_compress.kv, "kv_to_compress")
        self.print_tensor(kv_and_score_to_compress.score, "score_to_compress")

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        kv_compressed = (
            kv_and_score_to_compress.kv * kv_and_score_to_compress.score.softmax(dim=1)
        ).sum(dim=1)
        self.print_tensor(kv_compressed, "kv_before_norm")
        kv_compressed = self.norm(kv_compressed)
        self.print_tensor(kv_compressed, "kv_after_norm")
        freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        self.print_tensor(freqs_cis, "freqs_cis")
        apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        self.print_tensor(kv_compressed, "kv_after_rope")
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # `new_compressed_list` format is only used for testing
        self.print_tensor(kv_compressed, "compressed_kv_output")
        return kv_compressed

    def compress_extend(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion

        # kv_and_score_states: [max_num_reqs, compress_ratio * coff, head_dim * coff]
        kv_and_score_states = self._get_states(forward_batch)
        _, _, head_dim_times_coff = kv_and_score_states.kv.shape

        # extract some info
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        assert extend_lens is not None and prefix_lens is not None

        # compress info
        # TODO: reuse the buffer across layers and reduce the sizes
        max_buffer_size = 2 * kv_and_score_states.shape[1] + kv_and_scores.shape[0]
        temp_buffer_shape = [max_buffer_size, head_dim_times_coff]
        temp_buffer = kv_and_scores.new_empty(temp_buffer_shape)

        # Deliberately fill w/ huge values, s.t. when misuse and access the unfilled values,
        # we have higher probability to see something very weird
        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=kv_and_scores.kv.device,
        )

        bs = forward_batch.batch_size
        pt = 0
        for i in range(bs):
            # Definitions of variables
            #
            # kv_and_score_state: (compress_ratio * coff, head_dim * coff)
            #     only it[:old_valid_state_len] has valid data
            #
            # kv_and_score_buffer: (old_valid_state_len + valid_kv_len, head_dim * coff)
            #     content is cat(kv_and_score_state[:old_valid_state_len], kv_and_score)

            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            kv_and_score_state = kv_and_score_states[req_pool_indices[i]]
            if prefix_lens[i] == 0:
                # NOTE: padding with default values for overlap
                kv_and_score_state.clear()

            # Create kv_and_score_buffer
            pre_state_len = self.compute_state_len(
                seq_len=prefix_lens[i], ratio=self.ratio
            )
            valid_kv_len = pre_state_len + extend_lens[i]
            kv_and_score_buffer = temp_buffer[:valid_kv_len]
            kv_and_score_buffer[:pre_state_len] = kv_and_score_state[:pre_state_len]
            kv_and_score_buffer[pre_state_len:valid_kv_len] = kv_and_score

            # Write to kv_and_score_states
            post_state_len = self.compute_state_len(
                seq_len=valid_kv_len, ratio=self.ratio
            )
            kv_and_score_state[:post_state_len] = kv_and_score_buffer[
                valid_kv_len - post_state_len : valid_kv_len
            ]

            # Get the part that can be compressed (ratio-aligned)
            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                # Nothing to compress yet, just update pointers
                pt += extend_lens[i]
                continue

            # kv to compress: [compressed_len, ratio, head_dim * coff]
            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            # NOTE: apply ape only when compressing
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

            # Apply overlap transformation if enabled
            if self.overlap:
                new_kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, fill_value=0
                )
                new_score = self.overlap_transform(
                    kv_and_score_to_compress.score, fill_value=float("-inf")
                )
                kv_and_score_to_compress = KVAndScore.from_kv_score(
                    kv=new_kv, score=new_score
                )
                del new_kv, new_score
                # remove the first block before compression
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)

            # NOTE: ref code requires dtype as the same as hidden states (float32)
            # the raw output of kv_compressed is float32 already
            assert kv_compressed.dtype == torch.float32
            kv_compressed = self.norm(kv_compressed)

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            # get all the pos: ratio * n + (ratio - 1) > prefix_len - 1
            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    @maybe_torch_compile
    def compress_decode(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion

        seq_lens = forward_batch.seq_lens
        kv_and_score_states_pool = self._get_states(forward_batch)
        req_pool_indices = forward_batch.req_pool_indices

        # NOTE: first, write to the states
        bs = kv_and_scores.kv.size(0)
        write_pos = (seq_lens - 1) % self.ratio + self.overlap * self.ratio
        kv_and_score_states_pool[req_pool_indices, write_pos] = kv_and_scores

        # NOTE: need to copy out before modifying overlap states
        # kv_states: [bs, coff * ratio, coff * head_dim]
        kv_and_score_to_compress = kv_and_score_states_pool[req_pool_indices]

        # Shift just compressed kv states left by ratio
        if self.overlap:
            should_shift = seq_lens % self.ratio == 0
            kv_and_score_states_pool[req_pool_indices, : self.ratio] = KVAndScore(
                kv_score=torch.where(
                    should_shift[:, None, None],
                    kv_and_score_to_compress.kv_score[:, self.ratio :],
                    kv_and_score_to_compress.kv_score[:, : self.ratio],
                )
            )

        # shape: [bs * coff, ratio, coff * head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            -1, self.ratio, self.coff * self.head_dim
        )
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        if self.overlap:
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        self.print_tensor(kv_and_score_to_compress.kv, "kv_to_compress")
        self.print_tensor(kv_and_score_to_compress.score, "score_to_compress")

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        kv_compressed = (
            kv_and_score_to_compress.kv * kv_and_score_to_compress.score.softmax(dim=1)
        ).sum(dim=1)
        self.print_tensor(kv_compressed, "kv_before_norm")
        kv_compressed = self.norm(kv_compressed)
        self.print_tensor(kv_compressed, "kv_after_norm")
        freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        self.print_tensor(freqs_cis, "freqs_cis")
        apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        self.print_tensor(kv_compressed, "kv_after_rope")
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # `new_compressed_list` format is only used for testing
        self.print_tensor(kv_compressed, "compressed_kv_output")
        return kv_compressed

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # TODO: this should be the final implementation after verifying correctness
        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4Backend)
        is_paged = envs.SGLANG_OPT_DPSK_V4_RADIX.get()
        if is_paged:
            kv_score_buffer = self._get_state_pool(forward_batch)
            kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score
        else:
            kv_score_buffer = self._get_states(forward_batch).kv_score
        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=is_paged,
        )

    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.use_fused_compress:
            return self.compress_fused(kv_score, forward_batch)

        if envs.SGLANG_OPT_USE_OLD_COMPRESSOR.get():
            kv = kv_score[:, : self.coff * self.head_dim]
            score = kv_score[:, self.coff * self.head_dim :]
            kv_and_scores = KVAndScoreOld(kv=kv, score=score)
            self.compress_decode = self.compress_decode_old
            self.compress_extend = self.compress_extend_old
        else:
            if envs.SGLANG_OPT_DPSK_V4_RADIX.get():
                self.compress_decode = self.compress_decode_paged
                self.compress_extend = self.compress_extend_paged
            kv_and_scores = KVAndScore(kv_score)
        if TYPE_CHECKING:
            assert isinstance(kv_and_scores, KVAndScore)

        if (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            result = self.compress_decode(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
            )
        elif forward_batch.forward_mode.is_extend():
            result = self.compress_extend(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
            )
        else:
            msg = f"Forward mode {forward_batch.forward_mode} not supported in Compressor."
            raise NotImplementedError(msg)

        return result

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        self.forward_mode = forward_batch.forward_mode

        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
        return self.compress_dispatch(kv_score, forward_batch)

    def compress_extend_old(
        self, kv_and_scores: KVAndScore, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion
        KVAndScore = KVAndScoreOld

        # kv_and_score_states: [max_num_reqs, compress_ratio * coff, head_dim * coff]
        kv_and_score_states = self._get_states(forward_batch)
        _, _, head_dim_times_coff = kv_and_score_states.kv.shape

        # extract some info
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices

        # compress info
        # TODO: reuse the buffer across layers and reduce the sizes
        max_buffer_size = 2 * kv_and_score_states.shape[1] + kv_and_scores.shape[0]
        temp_buffer_shape = [max_buffer_size, head_dim_times_coff]
        temp_buffer = KVAndScore.empty_like(temp_buffer_shape, old=kv_and_scores)

        # Deliberately fill w/ huge values, s.t. when misuse and access the unfilled values,
        # we have higher probability to see something very weird
        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=kv_and_scores.kv.device,
        )

        bs = forward_batch.batch_size
        pt = 0
        for i in range(bs):
            # Definitions of variables
            #
            # kv_and_score_state: (compress_ratio * coff, head_dim * coff)
            #     only it[:old_valid_state_len] has valid data
            #
            # kv_and_score_buffer: (old_valid_state_len + valid_kv_len, head_dim * coff)
            #     content is cat(kv_and_score_state[:old_valid_state_len], kv_and_score)

            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            kv_and_score_state = kv_and_score_states[req_pool_indices[i]]
            if prefix_lens[i] == 0:
                # NOTE: padding with default values for overlap
                kv_and_score_state.clear()

            # Create kv_and_score_buffer
            pre_state_len = self.compute_state_len(
                seq_len=prefix_lens[i], ratio=self.ratio
            )
            valid_kv_len = pre_state_len + extend_lens[i]
            kv_and_score_buffer = temp_buffer[:valid_kv_len]
            kv_and_score_buffer[:pre_state_len] = kv_and_score_state[:pre_state_len]
            kv_and_score_buffer[pre_state_len:valid_kv_len] = kv_and_score

            # Write to kv_and_score_states
            post_state_len = self.compute_state_len(
                seq_len=valid_kv_len, ratio=self.ratio
            )
            kv_and_score_state[:post_state_len] = kv_and_score_buffer[
                valid_kv_len - post_state_len : valid_kv_len
            ]

            # Get the part that can be compressed (ratio-aligned)
            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                # Nothing to compress yet, just update pointers
                pt += extend_lens[i]
                continue

            # kv to compress: [compressed_len, ratio, head_dim * coff]
            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            # NOTE: apply ape only when compressing
            kv_and_score_to_compress.score = (
                kv_and_score_to_compress.score + self.ape.unsqueeze(0)
            )

            # Apply overlap transformation if enabled
            if self.overlap:
                kv_and_score_to_compress.kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, 0
                )
                kv_and_score_to_compress.score = self.overlap_transform(
                    kv_and_score_to_compress.score, float("-inf")
                )

                # remove the first block before compression
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)

            # NOTE: ref code requires dtype as the same as hidden states (float32)
            # the raw output of kv_compressed is float32 already
            assert kv_compressed.dtype == torch.float32
            kv_compressed = self.norm(kv_compressed)

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            # get all the pos: ratio * n + (ratio - 1) > prefix_len - 1
            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    def compress_decode_old(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion
        KVAndScore = KVAndScoreOld

        seq_lens = forward_batch.seq_lens
        kv_and_score_states_pool = self._get_states(forward_batch)
        req_pool_indices = forward_batch.req_pool_indices

        bs = kv_and_scores.kv.size(0)
        write_pos = (seq_lens - 1) % self.ratio + self.overlap * self.ratio
        kv_and_score_states_pool[req_pool_indices, write_pos] = kv_and_scores

        # NOTE: need to copy out before modifying overlap states
        # kv_states: [bs, coff * ratio, coff * head_dim]
        kv_and_score_to_compress = kv_and_score_states_pool[req_pool_indices]

        if self.overlap:
            # Shift just compressed kv states left by ratio
            should_shift = seq_lens % self.ratio == 0
            kv_and_score_states_pool[req_pool_indices, : self.ratio] = KVAndScore(
                kv=torch.where(
                    should_shift[:, None, None],
                    kv_and_score_to_compress.kv[:, self.ratio :],
                    kv_and_score_to_compress.kv[:, : self.ratio],
                ),
                score=torch.where(
                    should_shift[:, None, None],
                    kv_and_score_to_compress.score[:, self.ratio :],
                    kv_and_score_to_compress.score[:, : self.ratio],
                ),
            )

        # shape: [bs * coff, ratio, coff * head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            -1, self.ratio, self.coff * self.head_dim
        )
        kv_and_score_to_compress.score = (
            kv_and_score_to_compress.score + self.ape.unsqueeze(0)
        )

        if self.overlap:
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress.kv = self.overlap_transform_decode(
                kv_and_score_to_compress.kv
            )
            kv_and_score_to_compress.score = self.overlap_transform_decode(
                kv_and_score_to_compress.score
            )

        self.print_tensor(kv_and_score_to_compress.kv, "kv_to_compress")
        self.print_tensor(kv_and_score_to_compress.score, "score_to_compress")

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        kv_compressed = (
            kv_and_score_to_compress.kv * kv_and_score_to_compress.score.softmax(dim=1)
        ).sum(dim=1)
        self.print_tensor(kv_compressed, "kv_before_norm")
        kv_compressed = self.norm(kv_compressed)
        self.print_tensor(kv_compressed, "kv_after_norm")
        freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        self.print_tensor(freqs_cis, "freqs_cis")
        apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        self.print_tensor(kv_compressed, "kv_after_rope")
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # `new_compressed_list` format is only used for testing
        new_compressed_list = None
        self.print_tensor(kv_compressed, "compressed_kv_output")
        return kv_compressed


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        rotary_emb: RotaryEmbedding,
        freqs_cis: torch.Tensor,  # TODO: remove it after using rotary embedding
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        # TODO: do we need to support TP indexer?
        # currently, we duplicate indexer on all TP ranks
        self.n_local_heads = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,  # is_in_indexer
            rotary_emb,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        self.alt_streams = alt_streams

    def compute_q(self, q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # [bs, n_heads, head_dim]
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        fused_rope(
            q[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
        )
        q = rotate_activation(q)
        return q

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        out, _ = self.weights_proj(x)
        if not skip_scale:
            out = out * self.weight_scale
        return out

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        x_for_compressor: Optional[torch.Tensor] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DeepseekV4Backend)
        return forward_batch.attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            x_for_compressor=x_for_compressor if x_for_compressor is not None else x,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
        )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        else:
            self.qk_nope_head_dim = config.qk_nope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = config.compress_ratios[layer_id]
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio  # type: ignore

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert self.head_dim == config.head_dim
        else:
            assert self.head_dim == config.v_head_dim
        assert config.num_key_value_heads == 1

        # need a indexer for compress ratio = 4
        rope_scaling = config.rope_scaling
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # Please keep this assertion and not remove it
        # NOTE:
        # 1. 2601
        #    The `260119-updated` code changed compress_rope_theta
        # 2. 2604
        #    `official_code_0409/code/config.json` is 160000
        #    while `official_code_0409/config.json` is 40000
        #    maybe the latter is buggy? b/c dpsk's official generate.py uses `code/config.json`
        expected_compress_rope_theta = os.environ.get(
            "SGLANG_HACK_ASSERT_COMPRESS_ROPE_THETA"
        )
        if expected_compress_rope_theta is None:
            expected_compress_rope_theta = "160000"
        expected_compress_rope_theta = int(expected_compress_rope_theta)
        assert (
            config.compress_rope_theta == expected_compress_rope_theta
        ), f"{config.compress_rope_theta=} {expected_compress_rope_theta=}"
        rope_base = (
            config.compress_rope_theta if self.compress_ratio else config.rope_theta
        )

        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=get_global_server_args().device,
        )

        # naive impl: copy from reference code
        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert rope_scaling["factor"] == 16
        elif envs.SGLANG_DSV4_MODE.get() == "2601":
            assert rope_scaling["factor"] == 4
        else:
            raise NotImplementedError

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert self.compress_ratio in {0, 4, 128}
            if self.compress_ratio:
                original_seq_len = rope_scaling["original_max_position_embeddings"]
                assert original_seq_len == 65536
            else:
                original_seq_len = 0
        else:
            original_seq_len = rope_scaling["original_max_position_embeddings"]

        rope_scaling = config.rope_scaling
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]  # use first 3 streams for mqa layer
            self.alt_streams_indexer = alt_streams[
                -2:
            ]  # use last 2 streams for indexer
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                rotary_emb=self.rotary_emb,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    rotary_emb=self.rotary_emb,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                )

        # Note: attention sink should be replicated
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_a", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            # fp8_einsum handles scale transform internally — skip UE8M0 conversion
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.overlap_store_cache = envs.SGLANG_OPT_USE_OVERLAP_STORE_CACHE.get()

    def _compute_q_a(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # [bs, q_lora_rank]
        q, _ = self.wq_a(x)
        # [bs, q_lora_rank]
        q = self.q_norm(q)
        q_lora = q  # only used for indexer
        return q_lora

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [bs, n_local_heads, head_dim]
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q = rms_normalize_triton(q, self.eps)

        if positions is not None:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return q

    def _compute_kv(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [bs, head_dim]
        kv, _ = self.wkv(x)
        # [bs, head_dim]
        kv = self.kv_norm(kv)
        if positions is not None:
            fused_rope(
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(kv[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4Backend,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        # main stream: compute q
        q_lora = self._compute_q_a(x)
        q_lora_ready = current_stream.record_event()
        q = self._compute_q_b(q_lora, positions, freqs_cis)
        if q_out is not None:
            q_out.copy_(q)

        # alt stream 2: compute indexer
        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        # alt stream 0: compute kv
        with torch.cuda.stream(stream_kv):
            kv = self._compute_kv(x, positions, freqs_cis)
            if self.overlap_store_cache:
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )

        # alt stream 1: compute compressor
        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q, kv

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4Backend,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [bs, q_lora_rank]
        q, _ = self.wq_a(x)
        # [bs, q_lora_rank]
        q = self.q_norm(q)
        q_lora = q  # only used for indexer
        # [bs, n_local_heads, head_dim]
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        # [bs, n_local_heads, head_dim]
        q = rms_normalize_triton(q, self.eps)

        # [bs, head_dim]
        kv, _ = self.wkv(x)
        # [bs, head_dim]
        kv = self.kv_norm(kv)

        fused_rope(
            q[..., -self.qk_rope_head_dim :],
            kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
            self.freqs_cis,
            positions=positions,
        )

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        if _use_cp:
            kv = cp_all_gather_rerange_output(
                kv.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            x_for_compressor = (
                cp_all_gather_rerange_output(
                    x.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                if self.compressor is not None
                else x
            )
        else:
            x_for_compressor = x

        if self.overlap_store_cache:
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )

        if self.indexer is not None:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                x_for_compressor=x_for_compressor if _use_cp else None,
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x_for_compressor, forward_batch, self.layer_id, self.compressor
            )

        if q_out is not None:
            q_out.copy_(q)
        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        debug_return_kv: bool = False,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4Backend)

        freqs_cis = None

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            # pad the q to [batch_size, n_heads]
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            q, kv = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )

        # for TP attention, use the padded q, since q_out is set to the correct slice
        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=kv,
            v=kv,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=not self.overlap_store_cache,
        )
        # NOTE: no-op for pure DP-attention
        o = o[:, tp_slice, :]
        fused_rope(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
            )
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
        )
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        # TODO: check whether the implementation matches
        # TODO: make necessary changes if possible
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=alt_streams[0] if alt_streams is not None else None,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # self.layer_communicator = LayerCommunicator(
        #     layer_scatter_modes=self.layer_scatter_modes,
        #     input_layernorm=self.input_layernorm,
        #     post_attention_layernorm=self.post_attention_layernorm,
        #     allow_reduce_scatter=True,
        #     is_last_layer=(
        #         is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
        #     ),
        # )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            first_k_dense_replace = 0
            moe_layer_freq = 1
        else:
            first_k_dense_replace = self.config.first_k_dense_replace
            moe_layer_freq = self.config.moe_layer_freq
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= first_k_dense_replace
            and layer_id % moe_layer_freq == 0
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        @maybe_torch_compile
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        # x: [n,hc,d] -> y: [n,d], where n=b*s
        shape, dtype = x.size(), x.dtype

        # Handle empty batch
        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            # returned post should be [n, hc_mult]
            return y, post.squeeze(-1), comb

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            # DeepGEMM implementation
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            # TODO: maybe remove the contiguity requirement?
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            # Naive Torch implementation
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        # x: [n,d], residual: [n,hc,d] -> y: [n,hc,d]
        # post: [n,hc], comb: [n,hc,hc]

        # Handle empty batch
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            result = mhc_post(x, residual, post, comb)
            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @maybe_torch_compile
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        result = hc_post_torch_impl(x, residual, post, comb)
        return result

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert deepseek_v4_moe_code_path_checker.observed == 0

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )  # -> [n, d]
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )

        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states  # [n, hc, d]
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )  # -> [n, d]
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Communication logic (equivalent to LayerCommunicator):
        #
        # ======================== i. TP MoE ========================
        # DP attn + TP moe (moe_a2a_backend=none):
        # * mlp_mode = FULL (each-rank-has-whole-world-tokens)
        # * prepare_mlp -> _gather_hidden_states_and_residual -> dp_gather_partial
        # * postprocess_layer -> _scatter_hidden_states -> dp_scatter
        # Need Gather before MoE and Scatter after MoE.
        #
        # ======================== ii. DeepEP MoE ========================
        # DP attn + DeepEP moe (moe_a2a_backend=deepep/flashinfer/etc):
        # * mlp_mode = SCATTERED (each-rank-only-has-this-rank-tokens)
        # * prepare_mlp -> _simple (just layernorm, no gather)
        # * postprocess_layer -> _trivial (no scatter)
        # Because attn_tp_size==1 when tp==dp==ep, SCATTERED and TP_ATTN_FULL
        # have the same group_size. Token dispatch/combine is handled by
        # DeepEP inside MoE forward. No Gather/Scatter around MoE.
        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        # ----------------------------------- CP: fix input_ids to LOCAL ----------------
        if _use_cp:
            # CP requires DeepEP — TP MoE's all-reduce assumes identical tokens
            # across ranks, which CP violates. (Analogous to NSACPLayerCommunicator's
            # assert mlp_mode==SCATTERED when dp_size>1.)
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            # DeepEP handles cross-rank MoE dispatch/combine internally.
            # No gather/scatter needed — tokens stay LOCAL (SCATTERED).
            # This matches DSV3.2's mlp_mode=SCATTERED behavior with DeepEP + CP.
            #
            # Hash gating (n_hash_layers=3) needs input_ids[i] to correspond to
            # hidden_states[i]. hidden_states is LOCAL [N/cp_size] (round-robin).
            # input_ids is ORIGINAL [N] on every rank (never CP-split).
            # Slice to LOCAL to match hidden_states.
            cp_rank = get_attention_tp_rank()
            cp_size = get_attention_tp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            # TODO: improve the name - it is indeed local in CP, but is only used by e.g. Hash gating
            input_ids_global = input_ids
        # ----------------------------------- DP: gather for TP MoE --------------------
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = get_global_dp_buffer(), hidden_states
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        # ----------------------------------- MoE ------------------------------------
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        # ----------------------------------- Scatter (DP only, not CP) ----------------
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        hidden_states = self.hc_post(
            hidden_states, residual, post, comb
        )  # [n, d] -> [n, hc, d]

        # if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B" and not _is_hip:
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert deepseek_v4_moe_code_path_checker.observed == 1
            deepseek_v4_moe_code_path_checker.observed = 0

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.first_k_dense_replace = config.first_k_dense_replace
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)] if (_is_cuda or _is_hip) else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gemm_output_zero_allocator_size = 0
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    # TODO
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> torch.Tensor:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )
        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )
        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        for i in range(self.start_layer, self.end_layer):
            # TODO: ctx?
            layer = self.layers[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                # zero_allocator,
                # gemm_output_zero_allocator,
            )

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        pre_hc_head = (
            hidden_states.flatten(1)
            if envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
            else None
        )

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        if pre_hc_head is not None:
            return hidden_states, pre_hc_head
        return hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        # TODO: is this true that compress is kind of NSA
        get_attn_tp_context().init_context(config.q_lora_rank, is_nsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE)
            }
        )

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if self.config.n_routed_experts != 256 or self.config.n_shared_experts != 1:
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."
        elif (
            envs.SGLANG_DSV4_MODE.get() == "2604" and envs.SGLANG_DSV4_FP4_EXPERTS.get()
        ):
            disable_reason = "2604 routed experts use FP4 while shared experts remain FP8; fusion would incorrectly apply FP4 to shared experts."

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            disable_reason = "2604B checkpoint requires different clamping for shared and routed experts"

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:

        if self.nsa_enable_prefill_cp:
            if can_nsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        pre_hc_head = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if (
            envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
        ):
            hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            # TODO: indeed ours is "hidden_states_before_hc_head" instead of "before norm"
            #       abuse the existing field temporarily to minimize code diff
            #       should rename and generalize later, e.g. "hidden_states_for_spec"
            hidden_states_before_norm=pre_hc_head,
        )

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        layers = self.model.layers
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            # Pre-transform weight scale to DeepGEMM required layout (TMA-aligned / UE8M0 packed)
            # fp8_einsum('bhr,hdr->bhd') maps B=[h,d,r]=[G,R,D], so N=R, K=D for the B-side scale
            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        # ================ apply_ape_hotfix, should not be needed for final ckpt ================
        if is_nextn:
            return
        for layer in self.model.layers:
            self_attn = layer.self_attn
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    # This is used externally, please try to keep the API mostly unchanged
    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str, is_nextn: bool = False, num_hidden_layers: Optional[int] = None
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if not ATTN_BIT_WISE_EQUAL_MODE:
            if "self_attn" in name and (
                "compressor" not in name or not COMPRESSOR_BIT_WISE_EQUAL_MODE
            ):
                name = name.replace(".scale", ".weight_scale_inv")

        if not MOE_BIT_WISE_EQUAL_MODE:
            name = name.replace(".gate.tid2eid", ".topk.tid2eid")
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            if "mlp" in name:
                name = name.replace(".scale", ".weight_scale_inv")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        assert envs.SGLANG_DSV4_MODE.get() in ["2601", "2604"]
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() in ["2604A", "2604B"]
        else:
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() == ""

        if MOE_BIT_WISE_EQUAL_MODE:
            assert (
                self.num_fused_shared_experts == 0
            ), "use --disable-shared-experts-fusion for MoE bit-wise equal mode"

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        # Ignore this, b/c it is for nvfp4 ckpt
        # weights = self._maybe_quant_weights_to_fp8_ue8m0(
        #     weights, NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, is_nextn
        # )

        if (
            envs.SGLANG_DSV4_MODE.get() == "2604"
            and not envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
        ):
            if envs.SGLANG_DSV4_FP4_EXPERTS.get():
                weights = _dequant_fp8_wo_a(weights)
            else:
                # Converted FP8 checkpoint: wo_a is already bf16; drop stale wo_a.scale if present
                weights = ((n, t) for n, t in weights if not n.endswith(".wo_a.scale"))

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Params for special naming rules in mixed-precision models, for example:
        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # fuse compressor wkv and wgate weights into wkv_gate
        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"  # match wkv and wgate, skip ape

        # use default weight loader if module has no custom weight_loader
        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",  # Note that we need a . here to avoid confusion with gate_proj
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    # remap reference's temp ckpt weight -> deepseek hf format
                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        # Use shared head and embed weights from target model
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        # Skip target model weights
                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        # For nextn specific weights (out of layer)
                        # The nextn layer prefix of these weights has been removed
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        # For decoder layer weights
                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        # Skip non-stacked layers and experts (experts handled below).
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        # We have mlp.experts[0].gate_proj in the checkpoint.
                        # Since we handle the experts below in expert_params_mapping,
                        # we need to skip here BEFORE we update the name, otherwise
                        # name will be updated to mlp.experts[0].gate_up_proj, which
                        # will then be updated below in expert_params_mapping
                        # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):  # TODO
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            if MOE_BIT_WISE_EQUAL_MODE:
                                continue
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            # Skip loading extra bias for GPTQ models.
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            # Skip loading embed_tokens if not first rank in pipeline parallelism
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            # Skip loading norm if not last rank in pipeline parallelism
                            if ".norm." in name and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate  # exactly one is true
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    # modelopt attn kv scale is named differently
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    # modelopt ckpt contains not needed weights for MTP module:
                                    # model.decoder.self_attn.attn_mqa.v_scale and
                                    # model.decoder.self_attn.attn_mqa.k_scale
                                    if not name.startswith("mtp"):  # TODO: mtp
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                # if "attn_sink" in name:
                                #     attn_tp_rank = get_attention_tp_rank()
                                #     start = attn_tp_rank * param.numel()
                                #     param.data.copy_(
                                #         loaded_weight[start : start + param.numel()]
                                #     )
                                #     loaded_params.add(name)
                                #     continue

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            # hack to skip checking these in default ckpt. should have more rigorous check.
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if os.environ.get("SGLANG_SKIP_CHECKPOINT_LOAD_CHECK", "0") == "0":
            if unloaded_params:
                raise RuntimeError(
                    f"Some weights are not initialized from checkpoints: {unloaded_params}"
                )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequant fp8 block-quantized wo_a weight: bf16 = fp8_weight * e8m0_scale.

    Specifically for wo_a in 2604 checkpoint:
      weight: [8192, 4096] fp8_e4m3fn   (64*128 x 32*128)
      scale:  [64, 32]     fp8_e8m0fnu  (per 128x128 block)
    """
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert (
        scale.dtype == torch.float8_e8m0fnu
    ), f"expected fp8_e8m0fnu, got {scale.dtype}"
    assert weight.shape == (8192, 4096), f"unexpected weight shape {weight.shape}"
    assert scale.shape == (64, 32), f"unexpected scale shape {scale.shape}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    assert result.shape == (8192, 4096)
    return result.to(torch.bfloat16)


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Dequant fp8 wo_a weights inline: pair (wo_a.scale, wo_a.weight) -> bf16 wo_a.weight.

    2601 checkpoint:
      layers.0.attn.wo_a.weight  torch.bfloat16  [8192, 4096]  64.00MB  min=-0.375 max=0.3125

    2604 checkpoint:
      layers.0.attn.wo_a.scale  torch.float8_e8m0fnu  [64, 32]  0.00MB
      layers.0.attn.wo_a.weight  torch.float8_e4m3fn  [8192, 4096]  32.00MB
    """
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()
