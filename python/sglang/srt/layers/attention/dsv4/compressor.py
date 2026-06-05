from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.jit_kernel.deepseek_v4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
    linear_bf16_fp32,
    triton_create_paged_compress_data,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.dp_attention import get_attention_cp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.utils import add_prefix, is_npu

_is_npu = is_npu()


def _walsh_hadamard_matrix(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    # bf16 Sylvester matrix with the n**-0.5 norm factor baked in by dividing
    # by sqrt(2) at each doubling (log2(n) steps total). `dtype` is accepted
    # for backward-compat with callers; ascend builds in bf16 only.
    cache = _walsh_hadamard_matrix._cache  # type: ignore[attr-defined]
    key = (n, str(device))
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not ((n & (n - 1) == 0) and (n > 0)):
        raise ValueError(f"n must be a positive power of 2, got {n}")
    had = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    while had.shape[0] != n:
        had = torch.cat((torch.cat([had, had], 1), torch.cat([had, -had], 1)), 0)
        had /= math.sqrt(2)
    had = had.contiguous()
    cache[key] = had
    return had


_walsh_hadamard_matrix._cache = {}  # type: ignore[attr-defined]


def _apply_hadamard(inp: torch.Tensor, hadamard_matrix: torch.Tensor) -> torch.Tensor:
    # The n**-0.5 scale is already baked into `hadamard_matrix` (see
    # _walsh_hadamard_matrix above), so this is just `inp @ H` then bf16 cast.
    init_shape = inp.shape
    flat = inp.view(-1, hadamard_matrix.shape[0])
    return flat.matmul(hadamard_matrix).view(init_shape).to(torch.bfloat16)


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class FusedCompressMetadata(NamedTuple):
    write_loc: torch.Tensor
    extra_data: Optional[torch.Tensor]
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan]

    def copy_(self, other: FusedCompressMetadata) -> None:
        from .metadata import maybe_copy_inplace

        self.write_loc.copy_(other.write_loc)
        maybe_copy_inplace(self.extra_data, src=other.extra_data)
        self.plan.copy_(other.plan)


class CompressorBackendMixin:
    def get_paged_compress_metadata(self, compress_ratio: int) -> FusedCompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, FusedCompressMetadata)
        return metadata

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,
        is_paged: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        assert compress_ratio in (
            4,
            128,
        ), f"DSV4 supports CSA(4x) and HCA(128x) only, got {compress_ratio=}"
        if is_paged:
            metadata = self.get_paged_compress_metadata(compress_ratio)
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
            else:
                last_dim = 2 * head_dim * coff
                assert kv_score_buffer.shape[-1] == last_dim
                kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        else:
            plan = make_compressor_plan(compress_ratio, forward_batch)
            metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
        indices, extra_data, plan = metadata

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            indices=indices,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=extra_data,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            norm.variance_epsilon,
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: the concrete backend (DeepseekV4AttnBackend)
        # owns this helper. MQALayer._forward_prepare calls us before
        # attn_backend.forward(), so Raw -> DSV4Metadata must happen here too
        # (e.g. 1.6T layer 0 has compress_ratio=128 and needs cX_compress_metadata).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(new_compressed_kv.bfloat16())
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        # PREP_IN_CG lazy upgrade (see forward_core_compressor for rationale).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            new_compressed_kv_fp8, new_compressed_kv_scale = act_quant(
                new_compressed_kv
            )
            token_to_kv_pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                index_k=new_compressed_kv_fp8,
                index_k_scale=new_compressed_kv_scale,
            )


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def make_compressor_plan(
    compress_ratio: Literal[4, 128],
    forward_batch: ForwardBatch,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if forward_batch.forward_mode.is_decode():
        seq_lens_32 = forward_batch.seq_lens.to(torch.int32)
        return CompressorDecodePlan(compress_ratio, seq_lens_32)
    if forward_batch.forward_mode.is_prefill():
        assert not forward_batch.forward_mode.is_target_verify()
        extend_lens_list = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_list is not None and seq_lens_cpu is not None
        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            num_q_tokens=sum(extend_lens_list),
            seq_lens=seq_lens_cpu,
            extend_lens=torch.tensor(extend_lens_list),
            device=forward_batch.seq_lens.device,
        )
    elif forward_batch.forward_mode.is_target_verify():
        raise NotImplementedError("target verify mode to be implemented")
    else:
        raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> FusedCompressMetadata:
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # assert ring_size % compress_ratio == 0

    def clip_down(positions: torch.Tensor) -> torch.Tensor:
        return positions // compress_ratio * compress_ratio

    def get_raw_loc(positions: torch.Tensor) -> torch.Tensor:
        positions = positions.masked_fill(positions < 0, 0)
        loc = req_to_token[req_pool_indices, positions]
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(loc)
        swa_pages = swa_loc // swa_page_size
        state_loc = swa_pages * ring_size + swa_loc % ring_size
        return (state_loc // compress_ratio).to(torch.int32)

    is_overlap = is_overlap_compress(compress_ratio)

    if is_prefill:
        assert extend_lens is not None
        write_loc, extra_data = triton_create_paged_compress_data(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
        )

        plan_kwargs: dict
        if seq_lens_cpu is None:
            assert num_q_tokens is not None
            plan_kwargs = dict(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
            )
        else:
            assert extend_lens_cpu is not None
            plan_kwargs = dict(
                num_q_tokens=sum(extend_lens_cpu),
                seq_lens=torch.tensor(seq_lens_cpu),
                extend_lens=torch.tensor(extend_lens_cpu),
            )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            device=seq_lens.device,
            use_cuda_graph=use_prefill_cuda_graph,
            **plan_kwargs,
        )
    else:
        write_positions = clip_down(seq_lens - 1)
        write_loc = get_raw_loc(write_positions)
        if is_overlap:
            write_overlap_loc = get_raw_loc(write_positions - compress_ratio)
            extra_data = write_overlap_loc.view(-1, 1)
        else:
            extra_data = None
        plan = CompressorDecodePlan(compress_ratio, seq_lens.to(torch.int32))

    return FusedCompressMetadata(write_loc=write_loc, extra_data=extra_data, plan=plan)


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        freqs_cis: torch.Tensor,
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
        self.rope_head_dim = config.qk_rope_head_dim
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.freqs_cis = freqs_cis

        self.ape_converted = False

        if _is_npu:
            # Walsh-Hadamard matrix the NPU forward_npu path applies to kv
            # after rope via `apply_hadamard(kv, self.hadamard_matrix)`. CUDA
            # uses the triton fast_hadamard_transform in rotate_activation()
            # and never touches this buffer. Built lazily on first NPU forward
            # (init runs on CPU before weights move to NPU); cache key
            # includes device so it's safe.
            self._npu_hadamard_built = False
            # Lazy caches for the fused-compressor op (built on first call,
            # after weight loading + device move). See _ensure_fused_caches.
            self._fused_caches_built = False
            self._fused_wkv_w: Optional[torch.Tensor] = None
            self._fused_wgate_w: Optional[torch.Tensor] = None
            self._fused_norm_weight_fp32: Optional[torch.Tensor] = None

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        if _is_npu:
            # The hotfix below permutes ape for the CUDA triton compress
            # kernel layout. The NPU forward_npu path uses ape in its natural
            # [ratio, coff*head_dim] layout, so skip the permute here.
            return

        if self.overlap:
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            ape = torch.cat([ape[0], ape[1]], dim=0)
            self.ape.data.copy_(ape.view(self.ratio, -1))

    def _get_state_pool(self, forward_batch: ForwardBatch) -> CompressStatePool:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)

        assert isinstance(ret, CompressStatePool)

        return ret

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        if _is_npu:
            # NPU path: run the full compress flow inline. Writes go straight
            # to the kv pool via set_compress_*_buffer, so there's nothing
            # for forward_core_compressor to write afterwards.
            #
            # Fused path (_forward_npu_fused) reads write locations from
            # fm.c{ratio}_loc, populated for decode and for non-chunked prefill
            # (start_pos=0) by _build_npu_compress_metadata_prefill. Both modes
            # are gated by the single SGLANG_DSV4_NPU_FUSED_COMPRESSOR flag.
            # The earlier prefill-fused crash was stale non-tail
            # c{N}_state_page_table entries (req-slot reuse) corrupting other
            # requests' state via the op's WriteToCacheState; fixed by zeroing
            # non-tail page columns in the prefill metadata builder.
            fmode = forward_batch.forward_mode
            if envs.SGLANG_DSV4_NPU_FUSED_COMPRESSOR.get() and (
                fmode.is_decode()
                or (fmode.is_prefill() and not fmode.is_target_verify())
            ):
                return self._forward_npu_fused(x, forward_batch.positions, forward_batch)
            return self.forward_npu(x, forward_batch.positions, forward_batch)

        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
        if nsa_use_prefill_cp(forward_batch):
            kv_score = cp_all_gather_rerange_output(
                kv_score,
                get_attention_cp_size(),
                forward_batch,
                torch.cuda.current_stream(),
            )

        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4AttnBackend)
        kv_score_buffer = self._get_state_pool(forward_batch)
        kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score
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
            is_paged=True,
        )

    # ------------------------------------------------------------------
    # NPU forward path. The CUDA path above delegates to the backend mixin
    # (triton compress_forward + compress_fused_norm_rope_inplace); NPU has
    # no equivalent triton kernels, so the full per-request prefill + decode
    # loop runs here, calling torch_npu fallbacks for rope and a torch matmul
    # for the Walsh-Hadamard rotation.
    # ------------------------------------------------------------------

    def _ensure_npu_hadamard(self, device: torch.device) -> torch.Tensor:
        H = _walsh_hadamard_matrix(self.head_dim, torch.float32, device)
        if not self._npu_hadamard_built:
            # Register as buffer so checkpoint save / move_to_device picks it
            # up; subsequent calls just return the cached matrix.
            self.register_buffer("hadamard_matrix", H, persistent=False)
            self._npu_hadamard_built = True
        return H

    def _ensure_fused_caches(self) -> None:
        """Build the per-instance views the fused compressor op needs.

        Lazy because at __init__ time wkv_gate.weight is uninitialized and
        norm.weight hasn't been loaded yet; first forward fires after
        weight loading + device move, when these slices are stable.

        * ``_fused_wkv_w`` / ``_fused_wgate_w``: views into
          ``wkv_gate.weight`` (which stores ``[kv;gate]`` concatenated
          along output dim, kv first per ``ReplicatedLinear`` convention).
          The op signature wants them separately; views avoid the copy
          cost on every forward.
        * ``_fused_norm_weight_fp32``: fp32 copy of the fp32 RMSNorm
          weight. The op signature requires fp32 for norm_weight; the cast result is
          a constant since the underlying weight is frozen post-load.
        """
        if self._fused_caches_built:
            return
        coff = 1 + int(self.overlap)
        split = coff * self.head_dim
        # wkv_gate.weight shape: [2*coff*head_dim, hidden_size]. Split row-wise.
        w = self.wkv_gate.weight
        assert w.shape[0] == 2 * split, (
            f"wkv_gate.weight rows={w.shape[0]} != 2*coff*head_dim={2*split}"
        )
        self._fused_wkv_w = w[:split]
        self._fused_wgate_w = w[split:]
        self._fused_norm_weight_fp32 = self.norm.weight.to(torch.float32)
        self._fused_caches_built = True

    def _forward_npu_fused(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Fused-op NPU compressor path — single ``torch.ops.custom.compressor``
        call replaces the per-request Python loop in :meth:`forward_npu`.

        The op:
          1. Computes ``kv_state = x @ wkv``, ``score_state = x @ wgate``.
          2. Writes (kv_state, score_state) into ``state_cache`` at offsets
             derived from ``start_pos`` + ``cu_seqlens``.
          3. Per-ratio: adds ``ape``, softmax over the ratio dim, sum-reduces,
             applies RMSNorm + interleave-mode rope (rotary_mode=2), returns
             ``cmp_kv`` of shape ``[min(T, T//ratio + B), head_dim]``.

        State_cache writes happen inside the op, so we do NOT also call
        ``set_compress_state_buffer`` afterwards — only the compressed-kv
        epilog (:meth:`_compressor_epilog_npu`) runs on the returned tensor.

        Requires ``state_dtype=fp32`` (already the case for DSV4), the new
        backend metadata fields from :meth:`_compute_compress_locs` and
        :meth:`_build_npu_compress_metadata_prefill`, and a wheel where
        ``torch.ops.custom.compressor`` is registered.
        """
        from sglang.srt.models.deepseek_v4 import get_fused_compressor_rope_cos_sin

        ratio = self.ratio
        coff = 1 + int(self.overlap)
        device = x.device
        self._ensure_npu_hadamard(device)
        self._ensure_fused_caches()

        fm = forward_batch.attn_backend.forward_metadata
        positions_cmp = getattr(fm, f"positions_cmp_padding_c{ratio}", None)
        page_table = getattr(fm, f"c{ratio}_state_page_table", None)
        start_pos = getattr(fm, "start_pos", None)
        seqused = getattr(fm, "seqused", None)
        # cu_seqlens: prefix-sum query lengths with leading 0, [bs+1] int32.
        cu_seqlens = getattr(fm, "actual_seq_lengths_q_pa", None)
        assert positions_cmp is not None and page_table is not None, (
            "fused compressor needs backend metadata "
            "(positions_cmp_padding / c*_state_page_table) — make sure "
            "_build_npu_compress_metadata ran before this forward."
        )
        assert start_pos is not None, "fused compressor needs start_pos"
        assert cu_seqlens is not None, "fused compressor needs cu_seqlens"

        # state_cache: fp32 [block_num, page_size, 2*coff*D].
        pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            state_cache = pool.get_indexer_compress_state_cache(self.layer_id)
        else:
            state_cache = pool.get_attention_compress_state_cache(self.layer_id)

        cos, sin = get_fused_compressor_rope_cos_sin(
            self.freqs_cis, positions_cmp, dtype=torch.float32
        )

        cmp_kv = torch.ops.custom.compressor(
            x,
            self._fused_wkv_w,
            self._fused_wgate_w,
            state_cache,
            self.ape,
            self._fused_norm_weight_fp32,
            rope_sin=sin,
            rope_cos=cos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=ratio,
            state_block_table=page_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            coff=coff,
            norm_eps=self.norm.variance_epsilon,
            rotary_mode=2,
            cache_mode=1,
        )

        # cmp_kv shape: [min(T, T//ratio + B), head_dim]. For prefill the loc
        # tensor may be shorter than the padded output (trailing slots are
        # never compressed-to); trim to len(loc) before hadamard + epilog.
        loc = getattr(fm, f"c{ratio}_loc", None)
        if loc is not None and loc.numel() < cmp_kv.shape[0]:
            cmp_kv = cmp_kv[: loc.numel()]

        if forward_batch.attn_backend.graph_mode or cmp_kv.shape[0] > 0:
            if self.rotate:
                cmp_kv = _apply_hadamard(cmp_kv, self.hadamard_matrix)
            self._compressor_epilog_npu(cmp_kv, forward_batch)

    def forward_npu(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Inline NPU compress forward — writes compressed kv to the pool and
        returns ``None``.

        Per-request loop:

        * Prefill: split the seq into ``cutoff = seqlen - seqlen % ratio``
          tokens to compress + ``remainder`` to stash as state. For overlap
          (ratio=4), additionally stash the last ``ratio`` of the cutoff as
          overlap state. State writes happen via ``set_compress_state_buffer``;
          the cutoff portion gets ape-weighted softmax over the ratio dim,
          summed, then norm + rope + (optional) hadamard, then written via
          ``set_compress_buffer`` (in ``_compressor_epilog_npu``).
        * Decode: append the new (kv, score) to the state ring; if this token
          completes a ratio-aligned chunk, gather the chunk from the state
          buffer (overlap: 2*ratio, non-overlap: ratio), do ape-weighted
          softmax + sum, and write the compressed token via
          ``set_compress_buffer``.

        Relies on the pool's ``set_compress_state_buffer`` /
        ``get_compress_state_buffer`` / ``set_compress_buffer`` API.
        """
        import torch_npu  # local: NPU-only, used for npu_rotary_mul below

        ratio, overlap, d = self.ratio, self.overlap, self.head_dim
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return None

        device = x.device
        self._ensure_npu_hadamard(device)
        dtype = x.dtype
        x_f32 = x.float()
        # wkv + wgate are fused into a single wkv_gate.weight of shape
        # [2*coff*head_dim, hidden_size]; load_weights concatenates kv first
        # then wgate, so a [coff*hd, coff*hd] split along the output dim
        # recovers separate wkv(x) / wgate(x).
        coff = 1 + int(overlap)
        W = self.wkv_gate.weight.float()
        kv_full = F.linear(x_f32, W[: coff * d])  # [T, coff*d]
        score_full = F.linear(x_f32, W[coff * d :])  # [T, coff*d]

        seq_lens_cpu = forward_batch.seq_lens_cpu
        is_prefill = forward_batch.forward_mode.is_prefill()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        backend_fm = forward_batch.attn_backend.forward_metadata
        if ratio == 4:
            page_table = backend_fm.c4_state_page_table
        else:
            page_table = backend_fm.c128_state_page_table

        kv_out_list: list[torch.Tensor] = []
        kv_state_to_be_cached: list[torch.Tensor] = []
        score_state_to_be_cached: list[torch.Tensor] = []
        state_loc_list: list[torch.Tensor] = []
        kv_out_positions: list[torch.Tensor] = []
        # Step-5c per-token write loc bookkeeping. For each compressed token
        # we will write, record (req_idx_in_batch, compressed_seq_pos_in_req)
        # so we can compute the c{N}_kv_pool slot from the slab allocator
        # rather than from out_cache_loc // ratio (which only happens to be
        # right when the request's raw kv allocation aligns to ratio).
        write_req_indices: list[torch.Tensor] = []
        write_pos_in_req: list[torch.Tensor] = []
        seqlen_offset = 0
        # Running offset into the tail-only state bundle. The bundle's flat
        # layout is ``[req0_alloc_len_slots, req1_alloc_len_slots, ...]``
        # where ``alloc_len_i = seqlen_i - c{ratio}_state_alloc_offset_i`` —
        # NOT the raw seqlen (see ScheduleBatch._compute_dsv4_state_lens_extend).
        state_bundle_offset = 0

        for idx, seqlen in enumerate(seq_lens_cpu):
            seqlen = int(seqlen)
            if seqlen == 0:
                continue
            if is_prefill:
                pos_req = positions[seqlen_offset : seqlen_offset + seqlen]

                # Per-req tail-only state alloc range. Same formula as
                # ScheduleBatch._compute_dsv4_state_lens_extend; recomputed
                # here to avoid threading another tensor through forward_batch.
                tail_128 = seqlen % 128
                if ratio == 4:
                    c_alloc_len = (
                        tail_128 + 128
                        if (tail_128 <= 3 and seqlen >= 128)
                        else tail_128
                    )
                else:  # ratio == 128
                    c_alloc_len = tail_128
                c_alloc_offset = seqlen - c_alloc_len

                # Bundle slice for this req. The NPU paged state pool emits
                # real slot ids in the bundle (no ring-hash); slice by
                # ``state_bundle_offset`` (cumulative alloc_len across reqs),
                # NOT by ``seqlen_offset`` (cumulative raw seqlen).
                bundle = forward_batch.out_cache_loc_dsv4
                assert bundle is not None, (
                    "Compressor.forward_npu prefill on NPU needs the DSV4 "
                    "alloc bundle; expected maybe_write_dsv4_extend to have "
                    "populated batch.out_cache_loc_dsv4 before forward."
                )
                bundle_state_loc = (
                    bundle.out_c4_state_loc
                    if ratio == 4
                    else bundle.out_c128_state_loc
                )
                if c_alloc_len > 0:
                    # Only require a populated state bundle when this req
                    # actually allocates slots. A 128-aligned prefill at
                    # ratio==128 has c_alloc_len == seqlen % 128 == 0: there is
                    # no partial tail to cache (the whole sequence compresses
                    # cleanly into c128 chunks, and the next chunk is opened by
                    # decode), so this req contributes zero state slots. If
                    # EVERY req in the batch is 128-aligned, out_c128_state_loc
                    # is legitimately empty (allocator returns _empty_loc) — not
                    # a misconfiguration. An empty bundle while c_alloc_len > 0
                    # *does* mean c{ratio}_state_attn_allocator was never
                    # initialized.
                    assert (
                        bundle_state_loc is not None
                        and bundle_state_loc.numel() > 0
                    ), (
                        f"Compressor.forward_npu prefill: bundle.out_c{ratio}_state_loc "
                        f"is empty/None — DSV4NPUTokenToKVPoolAllocator's "
                        f"c{ratio}_state_attn_allocator was not initialized (check "
                        f"pool_configurator's NPU branch + npu_state_pool_size)."
                    )
                    out_cache_loc = bundle_state_loc[
                        state_bundle_offset : state_bundle_offset + c_alloc_len
                    ]
                    state_bundle_offset += c_alloc_len
                else:
                    # No tail to cache: empty slot view, never indexed below.
                    # For c128 (overlap=False) the only state-write branch is
                    # `remainder > 0`, and remainder == seqlen % 128 ==
                    # c_alloc_len, so c_alloc_len == 0 skips it. For c4
                    # c_alloc_len is always > 0 (tail + 128 when tail <= 3), so
                    # this branch is reached only for c128.
                    out_cache_loc = torch.empty(
                        (0,), dtype=torch.int64, device=device
                    )
                remainder = seqlen % ratio
                cutoff = seqlen - remainder
                # ``cutoff`` in raw coords; subtract ``c_alloc_offset`` for
                # slice-relative indexing into the per-req bundle slice.
                cutoff_in_slice = cutoff - c_alloc_offset
                should_compress = cutoff >= ratio
                # ratio-strided positions for the cutoff chunks (one rope
                # position per compressed token).
                pos_compressed = pos_req[:cutoff:ratio]
                kv = kv_full[seqlen_offset : seqlen_offset + seqlen]
                score = score_full[seqlen_offset : seqlen_offset + seqlen]

                if overlap and cutoff >= ratio:
                    # Stash the trailing ratio tokens of the cutoff so the
                    # next decode step can do overlap compression across the
                    # boundary. State alloc covers [cutoff - ratio, seqlen)
                    # for ratio=4 by construction (formula picks tail+128 or
                    # tail; in both cases the [cutoff-ratio, cutoff) window
                    # is inside [alloc_offset, seqlen)).
                    kv_state_to_be_cached.append(kv[cutoff - ratio : cutoff])
                    score_state_to_be_cached.append(
                        score[cutoff - ratio : cutoff] + self.ape
                    )
                    state_loc_list.append(
                        out_cache_loc[cutoff_in_slice - ratio : cutoff_in_slice]
                    )
                if remainder > 0:
                    kv_cut, kv_rem = kv.split([cutoff, remainder], dim=0)
                    score_cut, score_rem = score.split([cutoff, remainder], dim=0)
                    kv_state_to_be_cached.append(kv_rem)
                    score_state_to_be_cached.append(score_rem + self.ape[:remainder])
                    state_loc_list.append(out_cache_loc[-remainder:])
                    kv = kv_cut
                    score = score_cut

                if should_compress:
                    kv = kv.unflatten(0, (-1, ratio))  # [n_chunks, ratio, coff*d]
                    score = score.unflatten(0, (-1, ratio)) + self.ape
                    if overlap:
                        kv = self._overlap_transform(kv, value=0.0)
                        score = self._overlap_transform(score, value=float("-inf"))
                    kv_compressed = (kv * score.softmax(dim=1)).sum(
                        dim=1
                    )  # [n_chunks, d]
                    n_compressed_this_req = kv_compressed.shape[0]
                    kv_out_list.append(kv_compressed)
                    kv_out_positions.append(pos_compressed)
                    write_req_indices.append(
                        torch.full(
                            (n_compressed_this_req,),
                            idx,
                            dtype=torch.int64,
                            device=device,
                        )
                    )
                    write_pos_in_req.append(
                        torch.arange(
                            n_compressed_this_req,
                            dtype=torch.int64,
                            device=device,
                        )
                    )
                seqlen_offset += seqlen
            else:
                # Decode: one token per request. Append (kv, score+ape[pos%r])
                # to the state ring at c{4,128}_state_loc[idx]; if this token
                # is the last of a ratio-aligned chunk, gather the chunk and
                # produce one compressed kv via ape-softmax-sum.
                start_pos = seqlen - 1
                should_compress = (start_pos + 1) % ratio == 0
                pos_req = positions[idx : idx + 1] + (1 - ratio)
                kv = kv_full[idx : idx + 1]
                score = score_full[idx : idx + 1] + self.ape[start_pos % ratio]
                if ratio == 4:
                    state_loc_decode = backend_fm.c4_state_loc
                else:
                    state_loc_decode = backend_fm.c128_state_loc
                token_to_kv_pool.set_compress_state_buffer(
                    self.layer_id,
                    state_loc_decode[idx : idx + 1],
                    kv.view(1, 1, -1),
                    score.view(1, 1, -1),
                    None,
                    self.is_in_indexer,
                )
                if should_compress:
                    if overlap:
                        kv_indices = _get_kv_indices(
                            forward_batch, 2 * ratio, page_table, idx, seqlen
                        )
                        kv_state, score_state = (
                            token_to_kv_pool.get_compress_state_buffer(
                                self.layer_id, self.is_in_indexer, kv_indices
                            )
                        )
                        # kv_state / score_state: [2*r, 1, coff*d] → [2*r, d]
                        kv_state = kv_state.squeeze(1)
                        score_state = score_state.squeeze(1)
                        kv_state = torch.cat(
                            [kv_state[:ratio, :d], kv_state[ratio:, d:]], dim=0
                        )
                        score_state = torch.cat(
                            [score_state[:ratio, :d], score_state[ratio:, d:]],
                            dim=0,
                        )
                        kv_compressed = (kv_state * score_state.softmax(dim=0)).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        kv_indices = _get_kv_indices(
                            forward_batch, ratio, page_table, idx, seqlen
                        )
                        kv_state, score_state = (
                            token_to_kv_pool.get_compress_state_buffer(
                                self.layer_id, self.is_in_indexer, kv_indices
                            )
                        )
                        kv_compressed = (
                            kv_state[:, 0] * score_state[:, 0].softmax(dim=0)
                        ).sum(dim=0, keepdim=True)
                    kv_out_list.append(kv_compressed)
                    kv_out_positions.append(pos_req)
                    # Decode: 1 compressed token at compressed_seq_pos = seqlen//ratio - 1
                    decode_pos = seqlen // ratio - 1
                    write_req_indices.append(
                        torch.tensor([idx], dtype=torch.int64, device=device)
                    )
                    write_pos_in_req.append(
                        torch.tensor([decode_pos], dtype=torch.int64, device=device)
                    )

        # Flush the prefill state stash to the pool in one shot.
        if kv_state_to_be_cached:
            kv_state_cat = torch.cat(kv_state_to_be_cached, dim=0).unsqueeze(1)
            score_state_cat = torch.cat(score_state_to_be_cached, dim=0).unsqueeze(1)
            state_loc_cat = torch.cat(state_loc_list, dim=0)
            token_to_kv_pool.set_compress_state_buffer(
                self.layer_id,
                state_loc_cat,
                kv_state_cat,
                score_state_cat,
                None,
                self.is_in_indexer,
            )

        # Norm + rope + optional hadamard on the freshly compressed tokens,
        # then write via _compressor_epilog_npu with explicit slab-derived locs.
        if kv_out_list:
            kv_out = torch.cat(kv_out_list, dim=0).to(dtype)
            pos_out = torch.cat(kv_out_positions, dim=0)
            kv_out = self.norm(kv_out)
            # torch_npu.npu_rotary_mul takes cos/sin in repeat_interleave(2)
            # layout reshaped to (T, 1, 1, rope_dim). freqs_cis here is the
            # complex polar(1, theta) tensor built by precompute_freqs_cis;
            # cos=real, sin=imag at the rope_dim/2 frequency pair resolution.
            rope_dim = self.rope_head_dim
            # Use the same module-level contig cache as the outer rope path;
            # see _get_contig_freqs_real_imag in models/deepseek_v4.py for
            # why (.real / .imag on a complex tensor are strided views and
            # aclnnIndex over them triggers StridedSlice materialization).
            from sglang.srt.models.deepseek_v4 import _get_contig_freqs_real_imag
            freqs_real, freqs_imag = _get_contig_freqs_real_imag(self.freqs_cis)
            cos_half = freqs_real[pos_out].to(kv_out.dtype)
            sin_half = freqs_imag[pos_out].to(kv_out.dtype)
            cos = (
                cos_half.repeat_interleave(2, dim=-1)
                .view(-1, 1, 1, rope_dim)
                .contiguous()
            )
            sin = (
                sin_half.repeat_interleave(2, dim=-1)
                .view(-1, 1, 1, rope_dim)
                .contiguous()
            )
            rope_slice = kv_out[..., -rope_dim:]
            rope_view = rope_slice.unsqueeze(-2).unsqueeze(1)  # (T, 1, 1, rope_dim)
            rope_rot = torch_npu.npu_rotary_mul(
                rope_view, cos, sin, rotary_mode="interleave"
            )
            rope_slice.copy_(rope_rot.view_as(rope_slice))
            if self.rotate:
                kv_out = _apply_hadamard(kv_out, self.hadamard_matrix)
            # c{N}_kv_pool slot for each compressed token. The per-req
            # token-level slot id table on DSV4NPUReqToTokenPool is indexed
            # directly by compressed-seq position — no page table indirection
            # needed since the table elements already are c-pool slot ids.
            req_indices_flat = torch.cat(write_req_indices, dim=0)
            pos_in_req_flat = torch.cat(write_pos_in_req, dim=0)
            req_pool_flat = forward_batch.req_pool_indices[req_indices_flat]
            c_table = (
                forward_batch.req_to_token_pool.req_to_token_c4
                if ratio == 4
                else forward_batch.req_to_token_pool.req_to_token_c128
            )
            write_locs = c_table[
                req_pool_flat.to(torch.int64), pos_in_req_flat.to(torch.int64)
            ].to(torch.int32)
            self._compressor_epilog_npu(kv_out, forward_batch, override_loc=write_locs)
        return None

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        # Overlap layout: given (n_chunks, ratio, coff*d), build
        # (n_chunks, 2*ratio, d) where the first ratio rows hold the current
        # chunk's left half (..., :d) and the last ratio rows hold the
        # previous chunk's right half (..., d:). First chunk's right half is
        # filled with `value`.
        n_chunks, r, _ = tensor.shape
        d = self.head_dim
        out = tensor.new_full((n_chunks, 2 * r, d), value)
        out[:, r:] = tensor[..., d:]
        out[1:, :r] = tensor[:-1, :, :d]
        return out

    def _compressor_epilog_npu(
        self,
        kv: torch.Tensor,
        forward_batch: ForwardBatch,
        override_loc: Optional[torch.Tensor] = None,
    ) -> None:
        # Quant + write — quant only when this is an indexer compressor with
        # int8 li_kv. For the bf16 indexer / attention compressor branches,
        # kv_scale is None.
        kv_scale: Optional[torch.Tensor] = None
        li_kv_dtype = getattr(self, "li_kv_dtype", "bf16")
        if li_kv_dtype == "int8" and self.is_in_indexer:
            import torch_npu  # local import: only available on NPU

            kv, kv_scale = torch_npu.npu_dynamic_quant(kv)
            kv_scale = kv_scale.to(torch.float16)

        if override_loc is not None:
            loc = override_loc
        else:
            backend_fm = forward_batch.attn_backend.forward_metadata
            loc = backend_fm.c4_loc if self.ratio == 4 else backend_fm.c128_loc
        forward_batch.token_to_kv_pool.set_compress_buffer(
            self.layer_id,
            loc,
            kv,
            kv_scale,
            self.is_in_indexer,
        )


def _get_kv_indices(
    forward_batch: "ForwardBatch",
    kv_len: int,
    page_table: torch.Tensor,
    req_idx: int,
    seqlen: int,
) -> torch.Tensor:
    # Return the flat state-buffer indices of the trailing ``kv_len`` slots
    # for request ``req_idx`` (used to gather the overlap window or one
    # ratio-chunk during decode compression).
    logic_start = max(0, seqlen - kv_len)
    logic_end = seqlen
    page_size = forward_batch.attn_backend.page_size
    if page_size == 1:
        return page_table[req_idx, logic_start:logic_end]
    logic_pos = torch.arange(logic_start, logic_end, device=page_table.device)
    block_id = logic_pos // page_size
    offset_in_block = logic_pos % page_size
    return page_table[req_idx, block_id] * page_size + offset_in_block
