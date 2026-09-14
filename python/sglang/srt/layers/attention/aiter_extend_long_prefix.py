"""EXTEND attention over a long cached prefix on aiter's CK paged batch-prefill.

The chunk's K/V are already in the KV cache when attention runs, so the whole
causal attention of the chunk over prefix + chunk is one paged call whose page
table is the prefix indices followed by the chunk's cache locations.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.extend_attention import _copy_unified_indices_kernel


def build_paged_kv_indices(
    prefix_kv_indptr: torch.Tensor,
    prefix_kv_indices: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int32 `(kv_indptr, page_indices)` covering prefix + chunk per request."""
    device = prefix_kv_indptr.device
    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
    lens = prefix_lens + extend_seq_lens[:bs]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    total = int(prefix_kv_indices.numel()) + int(out_cache_loc.numel())
    page_indices = torch.empty(total, dtype=torch.int32, device=device)
    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        out_cache_loc,
        kv_indptr,
        page_indices,
        bs,
    )
    return kv_indptr, page_indices


class AiterLongPrefixExtend:
    """Holds the aiter `mha_batch_prefill` entry point and the per-device descale tensors."""

    def __init__(self):
        from aiter.ops.mha import mha_batch_prefill_func

        self._mha_batch_prefill = mha_batch_prefill_func
        self._descales: dict[tuple[float, str], torch.Tensor] = {}

    @classmethod
    def try_create(cls) -> Optional[AiterLongPrefixExtend]:
        """Return an instance, or `None` when the installed aiter lacks the kernel."""
        try:
            return cls()
        except Exception:
            return None

    def _descale(self, value: float, device: torch.device) -> torch.Tensor:
        key = (float(value), str(device))
        descale = self._descales.get(key)
        if descale is None:
            descale = torch.full((1,), float(value), dtype=torch.float32, device=device)
            self._descales[key] = descale
        return descale

    def forward(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        page_indices: torch.Tensor,
        max_extend_len: int,
        max_seq_len: int,
        sm_scale: float,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Causal attention of `q` over prefix + chunk, written into `o` (bf16, page size 1)."""
        kv_dtype = k_buffer.dtype
        is_fp8 = kv_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if is_fp8:
            # the CK kernel takes q in the cache dtype, as the Triton fp8 path does
            q_cast = q.to(kv_dtype)
            q_descale = self._descale(1.0, q.device)
            k_descale = self._descale(1.0 if k_scale is None else k_scale, q.device)
            v_descale = self._descale(1.0 if v_scale is None else v_scale, q.device)
        else:
            q_cast = q if q.dtype == kv_dtype else q.to(kv_dtype)
            q_descale = k_descale = v_descale = None
        if qo_indptr.dtype != torch.int32:
            qo_indptr = qo_indptr.to(torch.int32)
        self._mha_batch_prefill(
            q_cast,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            page_indices,
            int(max_extend_len),
            int(max_seq_len),
            softmax_scale=sm_scale,
            causal=True,
            out=o,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return o
