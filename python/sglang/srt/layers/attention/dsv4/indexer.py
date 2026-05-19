from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import (
    fused_rope,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.compressor import (
    Compressor,
    _apply_hadamard,
    _walsh_hadamard_matrix,
)
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.utils import add_prefix, is_hip, is_npu

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
    FP8_MAX = torch.finfo(FP8_DTYPE).max
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "torch reference impl hardcodes DSV4 indexer head_dim=128"
    assert block_size == 64, "torch reference impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    for i in range(batch_size):
        q = q_fp8[i, 0]
        q = q.to(torch.float32)
        q_scale = weight[i]
        seq_len = int(seq_lens[i].item())
        assert seq_len <= max_seq_len
        num_pages = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_pages * block_size
        pages = page_table[i, :num_pages]
        kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
        kvcache = kvcache_fp8[pages]
        SCALE_OFFSET = block_size * head_dim
        kvcache_value = kvcache[..., :SCALE_OFFSET].view(dtype=FP8_DTYPE)
        kvcache_scale = kvcache[..., SCALE_OFFSET:].view(dtype=torch.float32)
        kvcache_value = kvcache_value.to(torch.float32)
        kvcache_scale = kvcache_scale.contiguous()
        kvcache_value = kvcache_value.view(padded_seq_len, head_dim)
        kvcache_scale = kvcache_scale.view(padded_seq_len)
        score = F.linear(kvcache_value, q)
        score = F.relu(score)
        score *= q_scale[None, :]
        score = score.sum(dim=1)
        score *= kvcache_scale
        logits[i, :seq_len] = score[:seq_len]

    return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    )
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    if needs_sequential.any():
        sequential_indices = (
            torch.arange(TOPK, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

        raw_indices = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK),
            torch.where(
                sequential_valid,
                sequential_indices,
                torch.tensor(-1, device=device, dtype=torch.int32),
            ),
            raw_indices,
        )
        valid_topk = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
        )

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)

    page_indices = torch.where(
        valid_topk, page_indices, torch.tensor(-1, device=device, dtype=torch.int32)
    )

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(
            valid_topk, raw_indices, torch.tensor(-1, device=device, dtype=torch.int32)
        )
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackendMixin:
    def __init__(self):
        super().__init__()
        self.debug_use_external_c4_sparse_indices: bool = False

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            stream_weights.wait_event(q_scale_ready)
            weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)

        current_stream.wait_stream(stream_q)
        current_stream.wait_stream(stream_weights)

        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        q_fp8, q_scale = act_quant(q)
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        return q_fp8, weights, c4_indexer_kv_cache

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: this runs from MQALayer._forward_prepare,
        # before attn_backend.forward() would trigger the upgrade.
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackendMixin)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DSV4AttnMetadata,
        )

        assert isinstance(core_metadata, DSV4AttnMetadata)
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132

        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.dsv4.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            fn = fp8_paged_mqa_logits_torch
        else:
            from deep_gemm import fp8_paged_mqa_logits as fn

        _c4sl = indexer_metadata.c4_seq_lens
        if _c4sl.dim() == 1:
            _c4sl = _c4sl.unsqueeze(-1)
        logits = fn(
            q_fp8,
            c4_indexer_kv_cache,
            weights,
            _c4sl,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_c4_seq_len,
            False,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = forward_batch.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : core_metadata.c4_sparse_page_indices.size(0)
            ]

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform_512_v2(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            topk_transform_512(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    )
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        freqs_cis: torch.Tensor,
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
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.n_local_heads = self.n_heads
        # main's V4 c4 indexer hardcodes top-512 elsewhere; mirror that on
        # the module so forward_npu (NPU port of iforgetmyname forward_npu_dsv4)
        # has access without piping a server arg through.
        self.index_topk = getattr(config, "index_topk", 512)
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
            True,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
        )
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        self.alt_streams = alt_streams
        if _is_npu:
            # iforgetmyname/dsv4_release nsa_indexer.Indexer.__init__ L699:
            # `self.register_buffer("hadamard_matrix", get_had_pow2(self.head_dim))`.
            # Mirror that here so forward_npu can do the q rotation via a
            # torch matmul (CUDA path uses triton hadamard via rotate_activation).
            self._npu_hadamard_built = False
            # Tag the inner compressor as int8-li_kv so its epilog runs
            # `torch_npu.npu_dynamic_quant` and writes through the NPU int8 +
            # float16 scale buffer pair (set_compress_buffer NPU branch).
            # Mirrors iforgetmyname compressor_epilog L538-554 behavior.
            self.compressor.li_kv_dtype = "int8"

    def compute_q(self, q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
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
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if _is_npu and not forward_batch.forward_mode.is_idle():
            # NPU path: do the indexer compute inline (compressor write +
            # q/weights + lightning indexer call) and stash topk indices on
            # the backend's forward_metadata for _forward_compressed to pick
            # up. CUDA path delegates to the triton + deep_gemm pipeline in
            # the backend mixin below.
            topk_idxs = self.forward_npu(x, q_lora, forward_batch)
            forward_batch.attn_backend.forward_metadata.c4_topk_indices = topk_idxs
            return None
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DeepseekV4AttnBackend)
        return forward_batch.attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
        )

    # ------------------------------------------------------------------
    # NPU forward path — port of iforgetmyname/dsv4_release nsa_indexer.py
    # Indexer.forward_npu_dsv4 (L1991) and forward_npu_dsv4_fusion (L2149).
    # ------------------------------------------------------------------

    def _ensure_npu_hadamard(self, device: torch.device) -> torch.Tensor:
        H = _walsh_hadamard_matrix(self.head_dim, torch.float32, device)
        if not self._npu_hadamard_built:
            self.register_buffer("hadamard_matrix", H, persistent=False)
            self._npu_hadamard_built = True
        return H

    def _compute_q_npu(
        self, q_lora: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # iforgetmyname forward_npu_dsv4 L2010-2040: wq_b → reshape to
        # (T, n_heads, head_dim) → in-place rope on the rope-slice → hadamard
        # rotation (Walsh-Hadamard matmul). The CUDA `compute_q` path uses
        # tvm_ffi `fused_rope` + triton `rotate_activation`; both are absent
        # on NPU.
        from sglang.srt.models.deepseek_v4 import _v4_rope_inplace_npu

        bs = q_lora.shape[0]
        q, _ = self.wq_b(q_lora)
        q = q.view(bs, self.n_local_heads, self.head_dim)
        _v4_rope_inplace_npu(
            q[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions,
        )
        return _apply_hadamard(q, self.hadamard_matrix)

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Compute c4 top-k sparse indices on NPU; returns a [T, index_topk]
        int32 tensor.

        Steps (mirroring iforgetmyname forward_npu_dsv4):
          1. Materialize q via wq_b + rope + hadamard.
          2. Run the inner self.compressor (which on NPU writes the indexer
             c4 compress kv + state to the pool).
          3. Project x through weights_proj and apply softmax/head scale.
          4. Either `npu_quant_lightning_indexer` (int8 li_kv) or per-request
             einsum + topk (bf16 li_kv) to produce the top-k indices.
        """
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_group,
            get_attention_tp_size,
        )

        ratio = self.compressor.ratio  # = 4 for the c4 indexer
        device = x.device
        self._ensure_npu_hadamard(device)
        bs = x.shape[0]
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        # q path
        q = self._compute_q_npu(q_lora, forward_batch.positions)

        # weights path — keep the bf16 → bf16 projection (iforgetmyname
        # forward_npu_dsv4 L2044) and apply the combined softmax + n_heads
        # scaling here so the int8 indexer kernel receives `weights * scale`.
        weights, _ = self.weights_proj(x)
        weights = weights * (self.softmax_scale * self.n_heads**-0.5)

        # compressor path — writes c4 indexer compress kv + state on NPU.
        self.compressor(x, forward_batch)

        # Step-5d sentinel gate. Until SGLANG_DSV4_NPU_REAL_INDEXER=1, NPU
        # short-circuits to -1 sentinel (kept available so callers can keep
        # using the dense-equivalent path). With the flag on, fall through
        # to `_forward_npu_fused` below which runs the real
        # npu_quant_lightning_indexer.
        if _is_npu and not envs.SGLANG_DSV4_NPU_REAL_INDEXER.get():
            T = bs
            return torch.full(
                (T, self.index_topk),
                -1,
                dtype=torch.int32,
                device=device,
            )

        # Prefer the fused int8 lightning indexer when the indexer KV is
        # quantized (matches iforgetmyname's li_kv_dtype == "int8" branch).
        li_kv_dtype = getattr(self.compressor, "li_kv_dtype", "bf16")
        if li_kv_dtype == "int8":
            # Step-5e: skip the indexer kernel call when this rank's batch
            # is empty (no tokens to score). DP attention can leave some
            # ranks with an empty batch in flight; calling
            # npu_quant_lightning_indexer with T=0 / kv_len=0 deadlocks
            # async (some internal collective never returns) — sync mode
            # masked this. Return the sentinel topk so downstream
            # _forward_compressed gets a well-shaped tensor on the empty
            # rank without ever entering the indexer kernel.
            kv_lens = forward_batch.seq_lens
            if bs == 0 or (kv_lens.numel() > 0 and int(kv_lens.sum().item()) == 0):
                return torch.full(
                    (bs, self.index_topk),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
            li_cmp_kv = forward_batch.token_to_kv_pool.get_compress_buffer(
                self.layer_id, True
            )
            li_kv_scale = (
                forward_batch.token_to_kv_pool.get_compress_dequant_scale_buffer(
                    self.layer_id, True
                )
            )
            return self._forward_npu_fused(
                q, li_cmp_kv, li_kv_scale, weights, forward_batch
            )

        # bf16 li_kv path: per-request einsum + topk against the indexer
        # compress buffer — slow but architecture-faithful fallback.
        seqlens_cpu = forward_batch.seq_lens_cpu
        end_pos = forward_batch.seq_lens.cumsum(dim=0)
        page_table = forward_batch.attn_backend.forward_metadata.c4_page_table
        attn_tp_size = get_attention_tp_size()
        topk_idxs: list[torch.Tensor] = []
        for i, _end_token in enumerate(end_pos):
            seq_i = int(seqlens_cpu[i])
            kv_indices = _get_kv_indices(
                forward_batch, seq_i // ratio, page_table, i, seq_i // ratio
            )
            kv_cache_value = (
                forward_batch.token_to_kv_pool.get_compress_buffer(
                    self.layer_id, True, kv_indices
                )
            )
            if is_prefill:
                start = 0 if i == 0 else int(end_pos[i - 1])
                end = int(end_pos[i])
                index_score = torch.einsum(
                    "shd,td->sht",
                    q[start:end, ...],
                    kv_cache_value.squeeze(1),
                )  # [s, n_heads, seq_i//ratio]
                index_score = (
                    index_score.relu_() * weights.unsqueeze(-1)[start:end, ...]
                ).sum(dim=1)
                if attn_tp_size > 1 and getattr(self, "enable_indexer_tp", False):
                    get_attention_tp_group().all_reduce(index_score)
                # Causal mask in compressed-token coordinates.
                arange_kv = torch.arange(seq_i // ratio, device=device)
                arange_q = torch.arange(1, seq_i + 1, device=device).unsqueeze(1)
                causal = arange_kv.repeat(seq_i, 1) >= (arange_q // ratio)
                index_score += torch.where(
                    causal, float("-inf"), torch.zeros((), device=device)
                )
                topk_idx = index_score.topk(
                    min(self.index_topk, seq_i // ratio), dim=-1
                )[1]
                # Drop the diagonal token (position seq_i % ratio == 0
                # leaves a self-loop after the // ratio division).
                drop = topk_idx >= (
                    torch.arange(1, seq_i + 1, device=device).unsqueeze(1) // ratio
                )
                topk_idx = torch.where(drop, -1, topk_idx)
            else:
                index_score = torch.einsum(
                    "shd,td->sht",
                    q[i : i + 1, ...],
                    kv_cache_value.squeeze(1),
                )
                index_score = (
                    index_score.relu_() * weights.unsqueeze(-1)[i]
                ).sum(dim=1)
                topk_idx = index_score.topk(
                    min(self.index_topk, seq_i // ratio), dim=-1
                )[1]
            topk_idx = F.pad(
                topk_idx,
                (0, self.index_topk - topk_idx.shape[-1]),
                mode="constant",
                value=-1,
            )
            topk_idxs.append(topk_idx)
        return torch.cat(topk_idxs, dim=0).to(dtype=torch.int32)

    def _forward_npu_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # iforgetmyname forward_npu_dsv4_fusion L2149 — single fused
        # `npu_quant_lightning_indexer` call. Reads c4_page_table and
        # li_quant_metadata from the backend's forward_metadata (li_quant_
        # metadata is already populated on main; c4_page_table arrives in
        # roadmap step 3).
        import torch_npu  # local import: NPU only

        q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
        fm = forward_batch.attn_backend.forward_metadata
        li_quant_metadata = fm.kernel_metadata["li_quant_metadata"]
        kwargs = dict(
            query=q_int8,
            key=k,
            key_dequant_scale=k_scale.squeeze(-2),
            actual_seq_lengths_query=fm.actual_seq_lengths_q,
            actual_seq_lengths_key=fm.actual_seq_lengths_kv,
            block_table=fm.c4_page_table,
            layout_query="TND",
            layout_key="PA_BSND",
            weights=weights.to(torch.float16),
            query_dequant_scale=q_scale.to(torch.float16),
            cmp_ratio=4,
            query_quant_mode=0,
            key_quant_mode=0,
            sparse_mode=3,
            sparse_count=self.index_topk,
            metadata=li_quant_metadata,
        )
        if envs.SGLANG_DSV4_NPU_SPARSE_ATTN_DEBUG.get():
            import logging as _logging
            _lg = _logging.getLogger("v4-indexer-dbg")
            asq = kwargs["actual_seq_lengths_query"]
            ask = kwargs["actual_seq_lengths_key"]
            bt = kwargs["block_table"]
            _lg.warning(
                "[V4-NPU-indexer-dbg] layer=%d "
                "q.shape=%s q.dtype=%s "
                "k.shape=%s k.dtype=%s "
                "k_scale.shape=%s k_scale.dtype=%s "
                "weights.shape=%s "
                "q_scale.shape=%s "
                "actual_seq_q=%s asq.dtype=%s "
                "actual_seq_kv=%s ask.dtype=%s "
                "block_table.shape=%s bt.dtype=%s bt.range=[%s,%s] "
                "index_topk=%d",
                self.layer_id,
                tuple(kwargs["query"].shape), kwargs["query"].dtype,
                tuple(kwargs["key"].shape), kwargs["key"].dtype,
                tuple(kwargs["key_dequant_scale"].shape), kwargs["key_dequant_scale"].dtype,
                tuple(kwargs["weights"].shape),
                tuple(kwargs["query_dequant_scale"].shape),
                asq.tolist() if asq.numel() < 32 else "(too long)", asq.dtype,
                ask.tolist() if ask.numel() < 32 else "(too long)", ask.dtype,
                tuple(bt.shape), bt.dtype,
                int(bt.min().item()) if bt.numel() else "(empty)",
                int(bt.max().item()) if bt.numel() else "(empty)",
                self.index_topk,
            )
        topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(**kwargs)
        return topk_idxs.view(-1, self.index_topk)


def _get_kv_indices(
    forward_batch: ForwardBatch,
    kv_len: int,
    page_table: torch.Tensor,
    req_idx: int,
    seqlen: int,
) -> torch.Tensor:
    # Inlined from iforgetmyname/dsv4_release ascend_backend.get_kv_indices
    # (also duplicated in compressor.py — kept private here so indexer doesn't
    # re-import a private symbol from compressor).
    logic_start = max(0, seqlen - kv_len)
    logic_end = seqlen
    page_size = forward_batch.attn_backend.page_size
    if page_size == 1:
        return page_table[req_idx, logic_start:logic_end]
    logic_pos = torch.arange(logic_start, logic_end, device=page_table.device)
    block_id = logic_pos // page_size
    offset_in_block = logic_pos % page_size
    return page_table[req_idx, block_id] * page_size + offset_in_block
