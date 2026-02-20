from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import (
    normal_decode_set_metadata,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

@dataclass
class HpcAttentionMetadata:
    cache_seqlens_int32: torch.Tensor = None  # [bs] int32
    max_seq_len_q: int = 1
    max_seq_len_k: int = 0
    cu_seqlens_q: torch.Tensor = None  # [bs+1] int32
    cu_seqlens_k: torch.Tensor = None  # [bs+1] int32
    page_table: torch.Tensor = None  # [bs, max_blocks] int32


def _pad_to(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

import triton
import triton.language as tl

@triton.jit
def pack_scale_th_kernel(
    scale_ptr, cu_ptr, out_ptr,
    H: tl.constexpr,
    PAD: tl.constexpr,
    stride_s0: tl.constexpr,
    stride_s1: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid = tl.program_id(2)

    start = tl.load(cu_ptr + b).to(tl.int32)
    end = tl.load(cu_ptr + b + 1).to(tl.int32)
    L = end - start

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_pad = offs < PAD
    mask_len = offs < L

    tok = start + offs
    val = tl.load(
        scale_ptr + tok * stride_s0 + h * stride_s1,
        mask=mask_len,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        out_ptr + b * stride_o0 + h * stride_o1 + offs,
        val,
        mask=mask_pad,
    )

@torch.no_grad()
def pack_scale_th_triton(
    scale_th: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seq_len_q: int,
    pad_multiple: int = 128,
    block: int = 256,
) -> torch.Tensor:

    assert scale_th.is_cuda and cu_seqlens_q.is_cuda
    assert scale_th.dim() == 2

    T, H = scale_th.shape
    cu = cu_seqlens_q.to(torch.int32).contiguous()
    bs = cu.numel() - 1

    pad = _pad_to(int(max_seq_len_q), pad_multiple)
    out = torch.empty(bs, H, pad, device=scale_th.device, dtype=torch.float32)

    stride_s0 = scale_th.stride(0)
    stride_s1 = scale_th.stride(1)
    stride_o0 = out.stride(0)
    stride_o1 = out.stride(1)

    grid = (bs, H, triton.cdiv(pad, block))

    pack_scale_th_kernel[grid](
        scale_th, cu, out,
        H=H, PAD=pad,
        stride_s0=stride_s0, stride_s1=stride_s1,
        stride_o0=stride_o0, stride_o1=stride_o1,
        BLOCK=block,
        num_warps=4,
    )

    return out

class HpcAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.page_size = model_runner.server_args.page_size
        assert self.page_size in [32, 64], (
            f"HPC attention backend requires page_size = 32 or 64, "
            f"got {self.page_size}"
        )

        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.forward_metadata: HpcAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

        # For decoder-only models, skip encoder_metadata allocation
        self.encoder_metadata = {}

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
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = HpcAttentionMetadata()

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # TODO
                pass
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                batch_size = len(seq_lens)
                device = seq_lens.device
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                # Precompute page table
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                self.decode_cuda_graph_metadata[bs] = metadata

        self.forward_metadata = metadata

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
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # Draft Decode
                # TODO
                pass
            else:
                # Normal Decode
                metadata: HpcAttentionMetadata = self.decode_cuda_graph_metadata[bs]
                max_len = int(seq_lens_cpu.max().item()) if bs > 0 else 0
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len

                normal_decode_set_metadata(
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table,
                    self.req_to_token,
                    req_pool_indices,
                    self.decode_cuda_graph_metadata["strided_indices"],
                    max_seq_pages,
                    seq_lens,
                    0,
                    self.page_size,
                )

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        return 1

    def _build_page_table(self, forward_batch: ForwardBatch, max_seq_len_k: int):
        """Convert SGLang token-level req_to_token into HPC block_ids.

        SGLang stores flat slot indices in req_to_token. HPC needs a page table
        of shape [bs, max_blocks] containing physical page IDs. Since pages are
        contiguous blocks of page_size slots, slot_id // page_size = page_id.
        """
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        max_num_blocks = (max_seq_len_k + self.page_size - 1) // self.page_size

        # Sample one token position per block: 0, 64, 128, ...
        sample_positions = torch.arange(
            0, max_num_blocks * self.page_size, self.page_size, device=device
        )

        # Get the flat slot indices at those positions
        raw = self.req_to_token[forward_batch.req_pool_indices][
            :, sample_positions
        ]  # [bs, max_num_blocks]

        # Convert flat slot indices to page IDs
        block_ids = (raw // self.page_size).to(torch.int32)
        return block_ids

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if not (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_extend()
        ):
            raise NotImplementedError(
                f"HPC attention backend does not support forward mode: "
                f"{forward_batch.forward_mode}"
            )

        cache_seqlens = forward_batch.seq_lens.to(torch.int32)
        max_seq_len_k = int(forward_batch.seq_lens.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
            max_seq_len_q = 1
        else:  # is_extend()
            # Build cu_seqlens_q from extend_seq_lens
            cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            cu_seqlens_q[1:] = torch.cumsum(forward_batch.extend_seq_lens, dim=0).to(
                torch.int32
            )
            max_seq_len_q = int(forward_batch.extend_seq_lens.max().item())

        page_table = self._build_page_table(forward_batch, max_seq_len_k)

        self.forward_metadata = HpcAttentionMetadata(
            cache_seqlens_int32=cache_seqlens,
            max_seq_len_q=max_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            page_table=page_table,
        )

    def _prepare_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ):
        """Save KV cache, reshape Q, and prepare KV cache views."""
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        tp_q_head_num = layer.tp_q_head_num
        head_dim = layer.head_dim
        v_head_dim = layer.v_head_dim

        q_3d = q.contiguous().view(-1, tp_q_head_num, head_dim)

        # Get KV cache and reshape to [num_pages, page_size, kv_heads, dim]
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        key_cache = key_cache.view(-1, self.page_size, key_cache.shape[-2], head_dim)
        value_cache = value_cache.view(
            -1, self.page_size, value_cache.shape[-2], v_head_dim
        )

        return q_3d, key_cache, value_cache, tp_q_head_num, v_head_dim

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        import hpc

        q_3d, key_cache, value_cache, tp_q_head_num, v_head_dim = self._prepare_forward(
            q, k, v, layer, forward_batch, save_kv_cache
        )
        metadata = self.forward_metadata

        if self.kv_cache_dtype_str == "fp8_e4m3":
            T, H, D = q_3d.shape
            x = q_3d.view(T, H * D)

            x_fp8, scale_th = sglang_per_token_group_quant_fp8(
                x,
                group_size=D,
                column_major_scales=False,
                scale_tma_aligned=True,
            )

            q_fp8 = x_fp8.view(T, H, D)

            qscale = pack_scale_th_triton(
                scale_th=scale_th,                       # [T, H]
                cu_seqlens_q=metadata.cu_seqlens_q,      # [bs+1]
                max_seq_len_q=metadata.max_seq_len_q,    # int
                pad_multiple=128,
                block=256,
            )

            o = hpc.attention_with_kvcache_prefill_fp8(
                q=q_fp8,
                kcache=key_cache,
                vcache=value_cache,
                qkscale=(qscale * layer.k_scale.view(1, 1, 1)).contiguous(),
                vscale=layer.v_scale,
                cu_seqlens_q=metadata.cu_seqlens_q,
                block_ids=metadata.page_table,
                seqlens_kvcache=metadata.cache_seqlens_int32,
                max_seqlens_q=metadata.max_seq_len_q,
            )
        else:

            o = hpc.attention_with_kvcache_prefill_bf16(
                q=q_3d,
                kcache=key_cache,
                vcache=value_cache,
                cu_seqlens_q=metadata.cu_seqlens_q,
                block_ids=metadata.page_table,
                seqlens_kvcache=metadata.cache_seqlens_int32,
                max_seqlens_q=metadata.max_seq_len_q,
            )

        return o.view(-1, tp_q_head_num * v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        import hpc

        q_3d, key_cache, value_cache, tp_q_head_num, v_head_dim = self._prepare_forward(
            q, k, v, layer, forward_batch, save_kv_cache
        )
        metadata = self.forward_metadata

        if self.kv_cache_dtype_str == "fp8_e4m3":
            bs, H, D = q_3d.shape
            x = q_3d.reshape(bs, H * D).contiguous()

            x_fp8, qscale = sglang_per_token_group_quant_fp8(
                x,
                group_size=D,
                column_major_scales=False,   # qscale: [bs, H]
                scale_tma_aligned=True,
            )

            q_fp8 = x_fp8.reshape(bs, H, D)

            o = hpc.attention_decode_fp8(
                q=q_fp8,
                kcache=key_cache,
                vcache=value_cache,
                block_ids=metadata.page_table,
                num_seq_kvcache=metadata.cache_seqlens_int32,
                qscale=qscale,
                kscale=layer.k_scale,
                vscale=layer.v_scale,
                new_kv_included=True,
                splitk=True,
            )
        else:
            o = hpc.attention_decode_bf16(
                q=q_3d,
                kcache=key_cache,
                vcache=value_cache,
                block_ids=metadata.page_table,
                num_seq_kvcache=metadata.cache_seqlens_int32,
                new_kv_included=True,
                splitk=True,
            )

        return o.view(-1, tp_q_head_num * v_head_dim)
