from __future__ import annotations

import os
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.srt.layers.attention.dsv4.compressor import Compressor as _CompressorBase
from sglang.srt.layers.attention.dsv4.fused_compress_triton import (
    fused_ape_pool_norm_rope,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    fused_norm_rope_inplace_triton,
    fused_softmax_pool_triton,
)

try:
    from sglang.srt.layers.deepseek_v4_rope import fused_softmax_pool_triton
except ImportError:
    fused_softmax_pool_triton = None
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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


class DeepseekRefRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return rms_normalize_triton(x, self.eps, self.weight)


class CompressorHip(_CompressorBase):
    """HIP (ROCm) specific Compressor implementation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = DeepseekRefRMSNorm(self.head_dim, eps=self.norm.variance_epsilon)
        self._freqs_cis_real: torch.Tensor | None = None

    @cached_property
    def use_fused_compress(self) -> bool:
        return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()

    @cached_property
    def use_hip_fused_compress(self) -> bool:
        return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()

    @cached_property
    def use_fused_compress_triton(self) -> bool:
        # The fused Triton kernel only benefits non-overlap (HCA, ratio=128)
        # but HCA's K=128 loop is too sequential to outperform batched ops.
        # CSA (overlap=True) has a reshape/overlap-transform semantic mismatch.
        # Disabled until a tiled kernel for CSA overlap is implemented.
        return False

    def _get_states(
        self,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> KVAndScore:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def _get_state_pool(self, attn_backend: AttentionBackend) -> CompressStatePool:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        return seq_len % ratio + (ratio == 4) * ratio

    @staticmethod
    def compute_state_len_indices(seq_len: int, ratio: int):
        state_len = seq_len % ratio + (ratio == 4) * ratio
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
        attn_backend: AttentionBackend,
    ):
        backend = attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4HipRadixBackend)
        token_to_kv_pool = backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        state_pool = self._get_state_pool(backend)
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = backend.req_to_token_pool.req_to_token
        assert not self.forward_mode.is_target_verify()

        assert extend_lens is not None and prefix_lens is not None
        device = kv_and_scores.kv.device

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

            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                pt += extend_lens[i]
                continue

            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

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
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio

            if self.use_hip_fused_compress:
                kv_compressed = fused_softmax_pool_triton(
                    kv_and_score_to_compress.kv_score,
                    kv_and_score_to_compress._item_size,
                )
            else:
                kv_compressed = (
                    kv_and_score_to_compress.kv
                    * kv_and_score_to_compress.score.softmax(dim=1)
                ).sum(dim=1)

            assert kv_compressed.dtype == torch.float32

            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            if self.use_hip_fused_compress:
                fused_norm_rope_inplace_triton(
                    kv_compressed, self.norm.weight, self.norm.eps, freqs_cis
                )
            else:
                kv_compressed = self.norm(kv_compressed)
                apply_rotary_emb_triton(
                    kv_compressed[..., -self.rope_head_dim :], freqs_cis
                )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

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
        attn_backend: AttentionBackend,
    ):
        """Paged and cudagraph compatible version of compress_decode"""
        assert self.ape_converted
        state_pool = self._get_state_pool(attn_backend)
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = attn_backend.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_target_verify():
            draft_tokens = attn_backend.speculative_num_draft_tokens
            offsets = torch.arange(1, draft_tokens + 1, device=seq_lens.device)
            seq_lens_2d = seq_lens[:, None] + offsets[None, :]
            seq_lens = seq_lens_2d.view(-1)
            req_pool_indices = req_pool_indices.repeat_interleave(draft_tokens)

        raw_locs = req_to_token[req_pool_indices, seq_lens - 1]

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
        bs = seq_lens.size(0)

        if self.use_fused_compress_triton and not self.overlap:
            # Fused path for non-overlap (HCA, ratio=128, coff=1):
            # APE + softmax-pool + norm + RoPE in one kernel.
            # Overlap (CSA) is excluded because the overlap_transform_decode
            # rearranges A/B halves across the coff dimension in a way
            # that simple reshape cannot replicate correctly.
            raw = kv_and_score_to_compress.kv_score
            gathered = raw.reshape(bs, self.ratio, raw.shape[-1]).contiguous()

            comp_positions = (seq_lens - 1) // self.ratio * self.ratio
            freqs_real_table = self._get_freqs_cis_real()
            freqs_batch = freqs_real_table[comp_positions]

            kv_compressed = fused_ape_pool_norm_rope(
                kv_score_gathered=gathered,
                ape=self.ape,
                rms_weight=self.norm.weight,
                rms_eps=self.norm.eps,
                freqs_cis_real=freqs_batch,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                ratio=self.ratio,
                overlap=self.overlap,
            )
            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)
            return kv_compressed

        # Unfused reference path
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        if self.overlap:
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        if self.use_hip_fused_compress:
            kv_compressed = fused_softmax_pool_triton(
                kv_and_score_to_compress.kv_score,
                kv_and_score_to_compress._item_size,
            )
        else:
            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)
        if self.use_hip_fused_compress:
            freqs_cis = self._init_freqs_cis_per_decode_step(forward_batch, seq_lens)
            fused_norm_rope_inplace_triton(
                kv_compressed, self.norm.weight, self.norm.eps, freqs_cis
            )
        else:
            kv_compressed = self.norm(kv_compressed)
            freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        return kv_compressed

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        backend = attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4HipRadixBackend)
        kv_score_buffer = self._get_state_pool(backend)
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

    def _get_freqs_cis_real(self) -> torch.Tensor:
        """Cache the float32 view of freqs_cis (complex64 -> real interleaved)."""
        if self._freqs_cis_real is None:
            if self.freqs_cis.is_complex():
                self._freqs_cis_real = (
                    torch.view_as_real(self.freqs_cis).flatten(-2).contiguous()
                )
            else:
                self._freqs_cis_real = self.freqs_cis.contiguous()
        return self._freqs_cis_real

    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        if self.use_fused_compress and (
            envs.SGLANG_OPT_DPSK_V4_RADIX.get()
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_extend_without_speculative()
            )
        ):
            return self.compress_fused(
                kv_score, forward_batch, attn_backend=attn_backend
            )

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
                attn_backend=attn_backend,
            )
        elif forward_batch.forward_mode.is_extend():
            result = self.compress_extend(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        else:
            msg = f"Forward mode {forward_batch.forward_mode} not supported in Compressor."
            raise NotImplementedError(msg)

        return result

    def _init_freqs_cis_per_decode_step(
        self,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        attr = f"freqs_cis_c{self.ratio}"
        cached = getattr(forward_batch, attr, None)
        if cached is not None:
            return cached
        decoded = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        setattr(forward_batch, attr, decoded)
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)
        kv_score = self.compute_kv_score(x, forward_batch)
        self.forward_mode = forward_batch.forward_mode
        return self.compress_dispatch(
            kv_score, forward_batch, attn_backend=attn_backend
        )
