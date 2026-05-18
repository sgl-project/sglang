from __future__ import annotations

import math
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
    # iforgetmyname/dsv4_release hardware_backend/npu/utils.py:get_had_pow2.
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
    # iforgetmyname/dsv4_release nsa_indexer.py:apply_hadamard. The n**-0.5
    # scale is already baked into `hadamard_matrix` (see _walsh_hadamard_matrix
    # above), so this is just `inp @ H` followed by bf16 cast.
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
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
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
            # iforgetmyname/dsv4_release nsa_indexer.Compressor.__init__ L231
            # registers a Walsh-Hadamard matrix that the NPU forward_ori uses
            # for `apply_hadamard(kv, self.hadamard_matrix)` after rope. Build
            # the same here; CUDA path uses triton fast_hadamard_transform via
            # rotate_activation() instead and never touches this buffer.
            # Build lazily on first NPU forward (init runs on CPU before
            # weights move to NPU); cache key includes device so it's safe.
            self._npu_hadamard_built = False

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        if _is_npu:
            # The hotfix below permutes ape for the CUDA triton compress kernel
            # layout. The NPU forward_npu path uses ape in its natural
            # [ratio, coff*head_dim] layout (matches iforgetmyname's load
            # convention), so skip the permute here.
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

        if _is_npu and envs.SGLANG_DSV4_NPU_REAL_COMPRESSOR.get():
            # NPU path: do the full compress flow inline (writes go straight
            # to the kv pool via set_compress_*_buffer; nothing to return for
            # forward_core_compressor to write afterwards). Mirrors
            # iforgetmyname/dsv4_release nsa_indexer.Compressor.forward_ori.
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
    # NPU forward path — port of iforgetmyname/dsv4_release nsa_indexer.py
    # Compressor.forward_ori (L241). The CUDA path above delegates to the
    # backend mixin (triton compress_forward + compress_fused_norm_rope_inplace);
    # NPU has no equivalent triton kernels, so we do the full per-request
    # prefill + decode loop here, calling torch_npu fallbacks for rope and
    # a torch matmul for the Walsh-Hadamard rotation.
    # ------------------------------------------------------------------

    def _ensure_npu_hadamard(self, device: torch.device) -> torch.Tensor:
        H = _walsh_hadamard_matrix(self.head_dim, torch.float32, device)
        if not self._npu_hadamard_built:
            # Register as buffer so checkpoint save / move_to_device picks it
            # up; subsequent calls just return the cached matrix.
            self.register_buffer("hadamard_matrix", H, persistent=False)
            self._npu_hadamard_built = True
        return H

    def forward_npu(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Inline NPU compress forward — writes compressed kv to the pool and
        returns ``None``.

        Per-request loop matching iforgetmyname's ``forward_ori`` semantics:

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

        ``set_compress_state_buffer`` / ``get_compress_state_buffer`` /
        ``set_compress_buffer`` are pool API additions that come in roadmap
        step 2 — until they exist this function fails AttributeError, which
        is fine because ``SGLANG_DSV4_NPU_REAL_COMPRESSOR`` is off by default.
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
        # main fuses wkv + wgate into a single wkv_gate.weight
        # ([2*coff*head_dim, hidden_size]); load_weights concats kv first then
        # wgate (deepseek_v4.py L1513), so a [coff*hd, coff*hd] split along the
        # output dim recovers iforgetmyname's separate wkv(x) and wgate(x).
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

        for idx, seqlen in enumerate(seq_lens_cpu):
            seqlen = int(seqlen)
            if seqlen == 0:
                continue
            if is_prefill:
                pos_req = positions[seqlen_offset : seqlen_offset + seqlen]
                # State writes index the compress STATE pool (flat ring buffer),
                # not the full kv pool. Translate via the V4 token pool helper
                # (matches the formula used by the CUDA compressor's get_raw_loc
                # at compressor.py L226-233 and the NPU backend's
                # _build_npu_compress_metadata).
                raw_kv_loc = forward_batch.out_cache_loc[
                    seqlen_offset : seqlen_offset + seqlen
                ]
                out_cache_loc = token_to_kv_pool.translate_kv_loc_to_compress_state_loc(
                    raw_kv_loc, ratio
                )
                remainder = seqlen % ratio
                cutoff = seqlen - remainder
                should_compress = cutoff >= ratio
                # ratio-strided positions for the cutoff chunks (one rope
                # position per compressed token).
                pos_compressed = pos_req[:cutoff:ratio]
                kv = kv_full[seqlen_offset : seqlen_offset + seqlen]
                score = score_full[seqlen_offset : seqlen_offset + seqlen]

                if overlap and cutoff >= ratio:
                    # Stash the trailing ratio tokens of the cutoff so the
                    # next decode step can do overlap compression across the
                    # boundary — matches iforgetmyname forward_ori L281-286.
                    kv_state_to_be_cached.append(kv[cutoff - ratio : cutoff])
                    score_state_to_be_cached.append(
                        score[cutoff - ratio : cutoff] + self.ape
                    )
                    state_loc_list.append(out_cache_loc[cutoff - ratio : cutoff])
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
            # iforgetmyname compressor rope: ComplexExpRotaryEmbedding.forward
            # calls torch_npu.npu_rotary_mul with cos/sin in repeat_interleave(2)
            # layout reshaped to (T, 1, 1, rope_dim). freqs_cis here is the
            # complex polar(1, theta) tensor built by precompute_freqs_cis;
            # cos=real, sin=imag at the rope_dim/2 frequency pair resolution.
            rope_dim = self.rope_head_dim
            cos_half = self.freqs_cis.real[pos_out].to(kv_out.dtype)
            sin_half = self.freqs_cis.imag[pos_out].to(kv_out.dtype)
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
            # Slab → c{N}_kv_pool slot for each compressed token.
            req_indices_flat = torch.cat(write_req_indices, dim=0)
            pos_in_req_flat = torch.cat(write_pos_in_req, dim=0)
            req_pool_flat = forward_batch.req_pool_indices[req_indices_flat]
            kernel_page_size = forward_batch.attn_backend.page_size
            page_seq = pos_in_req_flat // kernel_page_size
            offset = pos_in_req_flat % kernel_page_size
            pages_table = token_to_kv_pool.get_req_to_token_c_pages(ratio)
            kernel_page = pages_table[req_pool_flat.to(torch.int64), page_seq]
            write_locs = (kernel_page.to(torch.int64) * kernel_page_size + offset).to(
                torch.int32
            )
            self._compressor_epilog_npu(kv_out, forward_batch, override_loc=write_locs)
        return None

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        # Overlap layout from iforgetmyname Compressor.overlap_transform:
        # given (n_chunks, ratio, coff*d), build (n_chunks, 2*ratio, d) where
        # the first ratio rows hold the current chunk's left half (..., :d)
        # and the last ratio rows hold the previous chunk's right half
        # (..., d:). First chunk's right half is filled with `value`.
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
        # int8 li_kv (matches iforgetmyname compressor_epilog L538). For the
        # bf16 indexer / attention compressor branches, kv_scale is None.
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
    # Inlined from iforgetmyname/dsv4_release ascend_backend.get_kv_indices:
    # return the flat state-buffer indices of the trailing ``kv_len`` slots
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
