# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-4 backend for Cosmos3 multiview sparse attention.

FA4 expresses a sparse mask the same way FlexAttention does: a block map that
classifies every (q_tile, kv_tile) as skipped, full, or partial, plus a
``mask_mod`` that resolves individual pairs inside partial tiles. The block map
this module hands over is the one ``build_multiview_block_sparsity`` already
builds; only the ``mask_mod`` is new, because FA4 compiles CuTe DSL rather than
tracing Python.

Rather than re-implement the visibility algebra in CuTe, the kernel reads a
truth table that the host already computed over semantic runs. Every token
carries the id of its run; the answer for a pair is one bit at
``(q_run, k_run)``. The rules stay in Python, and this kernel never learns what
a view or a frame is.

Everything CuTe/CUTLASS is imported lazily: ``flash-attn-4`` is an optional
Hopper/Blackwell-only dependency, and the multiview module must stay importable on
CPU-only hosts.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    MultiviewBlockSparsity,
    fa4_sparse_block_sizes,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_OP_NAMESPACE = "sglang_diffusion"
_OP_NAME = "cosmos3_multiview_fa4"


class _Fa4Entry(NamedTuple):
    flash_attn_func: Any
    block_sparse_cls: Any
    mask_mod: Any


_entry: _Fa4Entry | None = None


def _build_mask_mod(cutlass, cute, fa_utils):
    """Multiview mask_mod over the packed run truth table.

    ``aux_tensors`` are, in order:

    * ``q_word_base`` ``(seqlen_q,)`` int32: ``q_run * words_per_row``.
    * ``k_group_ids`` ``(seqlen_k,)`` int32: the key token's run id.
    * ``allowed_words`` ``(q_runs * words_per_row,)`` int32: the truth table
      over run pairs, bit-packed 32 keys per word.

    FA4 wraps the indices modulo ``seqlen_q``/``seqlen_k`` before calling a
    mask_mod that has aux tensors, so reads on out-of-range padded lanes stay
    in bounds; those lanes are force-masked afterwards. That holds only while
    the aux tensors are exactly ``seqlen_q``/``seqlen_k`` long, which
    ``multiview_fa4_attention`` checks.
    """

    @cute.jit
    def multiview_mask_mod(
        batch: Any,
        head: Any,
        m_idx: Any,
        n_idx: Any,
        seqlen_info: Any,
        aux_tensors: Any,
    ) -> Any:
        q_word_base = aux_tensors[0]
        k_group_ids = aux_tensors[1]
        allowed_words = aux_tensors[2]

        # The query row is shared by every lane of a scalar mask_mod call, so
        # its lookup is hoisted out of the per-key loop.
        base = q_word_base[m_idx[0]]
        result = cute.make_rmem_tensor(n_idx.shape, dtype=cutlass.Boolean)
        for j in cutlass.range_constexpr(cute.size(n_idx.shape)):
            group_k = k_group_ids[n_idx[j]]
            word = allowed_words[base + group_k // cutlass.Int32(32)]
            shift = cutlass.Uint32(group_k % cutlass.Int32(32))
            result[j] = cutlass.Boolean(
                fa_utils.shr_u32(cutlass.Uint32(word), shift) & cutlass.Uint32(1)
            )
        return result.load()

    return multiview_mask_mod


def _load_fa4() -> _Fa4Entry:
    """Import FA4 and build the mask_mod once per process.

    Raises rather than falling back: ``backend='fa4'`` is an explicit request,
    and silently running a different kernel would invalidate any comparison
    against the Triton path.
    """
    global _entry
    if _entry is not None:
        return _entry
    try:
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import flash_attn_func
        from flash_attn.cute import utils as fa_utils
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cosmos3 multiview backend='fa4' requires the FlashAttention-4 CuTe "
            "package (flash-attn-4), which needs CUDA 13 and an SM90 or SM100 GPU. "
            f"Import failed: {exc}"
        ) from exc

    _entry = _Fa4Entry(
        flash_attn_func=flash_attn_func,
        block_sparse_cls=BlockSparseTensorsTorch,
        mask_mod=_build_mask_mod(cutlass, cute, fa_utils),
    )
    logger.info("Cosmos3 multiview attention using the FlashAttention-4 CuTe backend.")
    return _entry


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: MultiviewBlockSparsity,
) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 4:
            raise ValueError(
                f"Cosmos3 multiview FA4 expects [B, S, H, D] {name}, got {tuple(tensor.shape)}."
            )
        if tensor.device.type != "cuda":
            raise ValueError(
                f"Cosmos3 multiview FA4 requires CUDA tensors, {name} is on {tensor.device}."
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Cosmos3 multiview FA4 supports {[str(d) for d in _SUPPORTED_DTYPES]}, "
                f"{name} has dtype {tensor.dtype}."
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"Cosmos3 multiview FA4 requires contiguous [B, S, H, D] {name}."
            )
    if k.shape[:3] != v.shape[:3]:
        raise ValueError(
            "Cosmos3 multiview FA4 key/value geometry mismatch: "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if q.shape[1] != sparsity.q_len:
        raise ValueError(
            "Cosmos3 multiview FA4 padded query length must match the block map: "
            f"q={q.shape[1]}, mask={sparsity.q_len}."
        )
    if k.shape[1] != sparsity.kv_len:
        raise ValueError(
            "Cosmos3 multiview FA4 padded key length must match the block map: "
            f"k={k.shape[1]}, mask={sparsity.kv_len}."
        )
    # FA4 wraps aux reads modulo these lengths; a shorter tensor would alias.
    if sparsity.q_word_base.numel() != sparsity.q_len:
        raise ValueError(
            "Cosmos3 multiview FA4 q_word_base must have one entry per padded query token."
        )
    if sparsity.k_group_ids.numel() != sparsity.kv_len:
        raise ValueError(
            "Cosmos3 multiview FA4 k_group_ids must have one entry per padded key token."
        )


# Wrapping the FA4 launch as a torch.library custom op keeps it opaque to
# torch.compile. FA4's Python entry point is a JIT compile-cache lookup, so a
# raw call lets Dynamo trace the CUTLASS DSL and guard on the contents of FA4's
# own kernel cache; those guards fail as FA4 compiles more kernels and the
# frame is eventually dropped to eager. The custom op gives Dynamo one
# Tensor -> Tensor boundary instead. The hasattr guard keeps this idempotent
# across test re-imports that pop the module from sys.modules.
if not hasattr(getattr(torch.ops, _OP_NAMESPACE), _OP_NAME):

    @torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}", mutates_args=())
    def _cosmos3_multiview_fa4_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        partial_counts: torch.Tensor,
        partial_indices: torch.Tensor,
        full_counts: torch.Tensor,
        full_indices: torch.Tensor,
        q_word_base: torch.Tensor,
        k_group_ids: torch.Tensor,
        allowed_words: torch.Tensor,
        q_block_size: int,
        kv_block_size: int,
    ) -> torch.Tensor:
        """``MultiviewBlockSparsity`` flattened to the tensors/ints a schema allows."""
        entry = _load_fa4()
        # FA4 accepts singleton batch/head dims and broadcasts them; the
        # multiview mask is identical across both.
        block_sparse = entry.block_sparse_cls(
            mask_block_cnt=partial_counts[None, None],
            mask_block_idx=partial_indices[None, None],
            full_block_cnt=full_counts[None, None],
            full_block_idx=full_indices[None, None],
            block_size=(q_block_size, kv_block_size),
        )
        out = entry.flash_attn_func(
            q,
            k,
            v,
            mask_mod=entry.mask_mod,
            # Same order as MultiviewBlockSparsity.aux_tensors(), which is the
            # order _build_mask_mod indexes them in.
            aux_tensors=[q_word_base, k_group_ids, allowed_words],
            block_sparse_tensors=block_sparse,
        )
        if isinstance(out, tuple):
            out = out[0]
        # FA4 may hand back a view of its own workspace; a custom op must not
        # return a tensor aliasing anything it does not own.
        return out.contiguous()

    @_cosmos3_multiview_fa4_op.register_fake
    def _(
        q,
        k,
        v,
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        q_word_base,
        k_group_ids,
        allowed_words,
        q_block_size,
        kv_block_size,
    ):
        # FA4 returns the query layout unchanged: [B, S_q_padded, H_q, D].
        return torch.empty_like(q)


_cosmos3_multiview_fa4_op = getattr(getattr(torch.ops, _OP_NAMESPACE), _OP_NAME)


def multiview_fa4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: MultiviewBlockSparsity,
) -> torch.Tensor:
    """Run FA4 over contiguous ``[B, S, H, D]`` tensors with the multiview mask.

    ``softmax_scale`` is left to FA4's default of ``1/sqrt(head_dim)``, which is
    also the FlexAttention default the Triton path relies on. Validation stays
    outside the custom op so shape errors name this function rather than a
    schema mismatch.
    """
    _validate(q, k, v, sparsity)

    expected = fa4_sparse_block_sizes(q.device)
    if (sparsity.q_block_size, sparsity.kv_block_size) != expected:
        raise ValueError(
            f"Cosmos3 multiview FA4 requires a {expected} sparse block map on compute "
            f"capability {torch.cuda.get_device_capability(q.device)[0]}.x, got "
            f"({sparsity.q_block_size}, {sparsity.kv_block_size})."
        )

    return _cosmos3_multiview_fa4_op(
        q,
        k,
        v,
        sparsity.partial_counts,
        sparsity.partial_indices,
        sparsity.full_counts,
        sparsity.full_indices,
        sparsity.q_word_base,
        sparsity.k_group_ids,
        sparsity.allowed_words,
        sparsity.q_block_size,
        sparsity.kv_block_size,
    )
