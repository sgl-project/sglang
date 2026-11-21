# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kimi-K3 ROCm MLA decode: write the query and the latent KV cache in one launch.

K3 MLA applies no RoPE. The writer runs AITER's NoPE fused concat/cache kernel
(fused_qk_cat_and_cache_mla): it copies the PE halves through unrotated, so Q and a
BF16 cache equal the split concat/cache chain's bit for bit, including -0.0, inf and
NaN lanes. An FP8 cache matches for -0.0 and in-range values; the kernel's cast
saturates inf, which torch's does not. Where the attention backend reads an FP8 query
padded to its kernel head count (mla_fp8_q_out), the same launch casts Q into that
buffer.

kimi_k3 calls create_q_cache_writer on every platform; this module alone decides
where the fused write runs (fused_q_cache_available) and returns None elsewhere.
"""

from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import print_info_once

_KV_LORA_RANK = 512
_ROPE_HEAD_DIM = 64
# SGLang's MLA pool is [slots, 1, 576]; the kernel takes BF16 or e4m3fn (gfx950's
# FP8) and asserts on anything else.
_CACHE_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)

# With save_kv_cache=False, these backends read only the BF16 Q and the KV cache;
# Triton target-verify runs its extend kernel, which also reads k.
_DECODE_BACKENDS = frozenset({"aiter", "triton"})
_TARGET_VERIFY_BACKENDS = frozenset({"aiter"})


def fused_q_cache_available() -> bool:
    """Whether the fused write runs here: gfx950 with SGLANG_USE_AITER=1, the only
    target it was measured on."""
    if not (
        envs.SGLANG_USE_AITER.get()
        and torch.version.hip is not None
        and torch.cuda.is_available()
    ):
        return False
    hip = current_arch().hip
    return hip is not None and hip.gcn_arch == "gfx950"


def create_q_cache_writer(
    *, kv_lora_rank: int, qk_rope_head_dim: int
) -> Optional["KimiK3RocmMlaQCacheWriter"]:
    """The writer where the fused kernel runs, else None, which keeps the split
    chain."""
    if (kv_lora_rank, qk_rope_head_dim) != (
        _KV_LORA_RANK,
        _ROPE_HEAD_DIM,
    ) or not fused_q_cache_available():
        return None
    # Present in every AITER pinned in docker/rocm.Dockerfile; not on gfx1250, which this
    # writer never targets (fused_q_cache_available).
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_cat_and_cache_mla

    print_info_once(
        "Kimi-K3 MLA: decode and target-verify write Q and the latent KV cache "
        "in one AITER launch (fused_qk_cat_and_cache_mla)"
    )
    return KimiK3RocmMlaQCacheWriter(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        fused_cat_and_cache=fused_qk_cat_and_cache_mla,
    )


def _supports_cache(kv_cache: torch.Tensor) -> bool:
    """Whether the kernel can write this KV cache: the [slots, 1, 576] MLA pool in
    BF16 or e4m3fn. A paged 4D cache would be indexed as one slot per block."""
    return (
        kv_cache.ndim == 3
        and kv_cache.shape[1] == 1
        and kv_cache.shape[2] == _KV_LORA_RANK + _ROPE_HEAD_DIM
        and kv_cache.dtype in _CACHE_DTYPES
    )


def _check_fp8_q_out(
    fp8_q: torch.Tensor, *, q_nope: torch.Tensor, head_dim: int
) -> None:
    """The backend's FP8 query buffer must be a contiguous [tokens, >= q heads,
    head_dim] e4m3fn tensor on the query device; anything else is a backend bug."""
    tokens, heads = q_nope.shape[:2]
    if not (
        fp8_q.dtype == torch.float8_e4m3fn
        and fp8_q.ndim == 3
        and fp8_q.shape[0] == tokens
        and fp8_q.shape[1] >= heads
        and fp8_q.shape[2] == head_dim
        and fp8_q.is_contiguous()
        and fp8_q.device == q_nope.device
    ):
        raise ValueError(
            f"mla_fp8_q_out must be a [{tokens}, >={heads}, {head_dim}] "
            f"contiguous float8_e4m3fn buffer on {q_nope.device}, got "
            f"{tuple(fp8_q.shape)} {fp8_q.dtype} stride {fp8_q.stride()} "
            f"on {fp8_q.device}"
        )


class KimiK3RocmMlaQCacheWriter:
    def __init__(
        self,
        *,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        fused_cat_and_cache: Callable[..., Tuple[torch.Tensor, ...]],
    ) -> None:
        self._fused_cat_and_cache = fused_cat_and_cache
        self._head_dim = kv_lora_rank + qk_rope_head_dim

    def write(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
        k_scale_float: Optional[float],
        attention_backend: str,
        is_target_verify: bool,
        get_fp8_q_out: Optional[Callable[[], Optional[torch.Tensor]]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (q, k placeholder) after writing Q and the KV cache,
        or None to leave both to the split concat/cache chain.

        ``get_fp8_q_out`` is the attention backend's ``mla_fp8_q_out``, called only
        once the launch will run. If it returns a buffer, Q is cast into its first
        heads in FP8 and the padded buffer is returned as q.
        """
        backends = _TARGET_VERIFY_BACKENDS if is_target_verify else _DECODE_BACKENDS
        if (
            # An idle rank has nothing to write.
            q_nope.shape[0] == 0
            or attention_backend not in backends
            # The split chain's AITER write casts without the scale,
            # its Triton decode write divides by it; they agree only at 1.
            or k_scale_float not in (None, 1.0)
            or not _supports_cache(kv_cache)
        ):
            return None
        if slot_mapping.stride(0) != 1:
            # The kernel indexes slot_mapping without a stride; the caller makes it
            # contiguous, so this is a contract violation, not a fallback.
            raise ValueError(
                f"slot_mapping must be contiguous, got stride {slot_mapping.stride()}"
            )
        q = self._write_fused(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            get_fp8_q_out=get_fp8_q_out,
        )
        # The KV cache is already written; attention reads only the shape of k.
        k_placeholder = torch.empty(
            (q_nope.shape[0], 1, self._head_dim),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        return q, k_placeholder

    def _write_fused(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        get_fp8_q_out: Optional[Callable[[], Optional[torch.Tensor]]],
    ) -> torch.Tensor:
        fp8_q = None if get_fp8_q_out is None else get_fp8_q_out()
        if fp8_q is not None:
            _check_fp8_q_out(fp8_q, q_nope=q_nope, head_dim=self._head_dim)
        q, _, _, _ = self._fused_cat_and_cache(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            # K3's KV scale is 1 (write checks it), so the kernel skips the scale.
            k_scale=None,
            apply_scale=False,
            # The kernel casts unscaled and saturating, which is the asm verify's
            # static FP8 quant at its unit q scale; the pad heads stay zero.
            q_out=None if fp8_q is None else fp8_q[:, : q_nope.shape[1]],
            q_out_dtype=q_nope.dtype,
        )
        return q if fp8_q is None else fp8_q
