# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-4 backend for Cosmos3 block-sparse attention.

FA4 expresses a sparse mask the same way FlexAttention does: a block map that
classifies every (q_tile, kv_tile) as skipped, full, or partial, plus a
``mask_mod`` that resolves individual pairs inside partial tiles. The block maps
this module hands over are the ones the model modules already build (the
Transfer-Auto multiview map of ``build_multiview_block_sparsity`` and the
Sim-Auto replay map of ``build_replay_fa4_mask``); only the ``mask_mod`` is
new, because FA4 compiles CuTe DSL rather than tracing Python.

Rather than re-implement the visibility algebra in CuTe, the kernel reads a
truth table that the host already computed over semantic runs. Every token
carries the id of its run; the answer for a pair is one byte at
``(q_run, key position)``, or one bit at ``(q_run, k_run)`` for a table too
large to expand over the keys. The rules stay in Python, and this kernel never
learns what a view or a frame is.

Everything CuTe/CUTLASS is imported lazily: the vendored FA4 CuTe kernels are
Hopper/Blackwell-only, and the multiview module must stay importable on
CPU-only hosts.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import msgspec
import torch

from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_attention import (
    MultiviewBlockSparsity,
    fa4_sparse_block_sizes,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_OP_NAMESPACE = "sglang_diffusion"
_OP_NAME = "cosmos3_multiview_fa4"


# eq=False: the fields are tensors, so a generated __eq__ would return a tensor
# rather than a bool. Instances are cache values compared by identity.
class Fa4BlockSparsity(msgspec.Struct, frozen=True, eq=False):
    """Block map and pair table in the layout the FA4 custom op consumes.

    ``partial_*``/``full_*`` are the FlexAttention KV-block layout. The other
    tensors resolve pairs inside partial tiles: ``allowed_table`` in either
    form ``fa4_mask_table`` builds, with ``q_table_base`` and ``k_group_ids``
    exactly as long as the padded query/key streams (see ``_build_mask_mods``).
    """

    partial_counts: torch.Tensor
    partial_indices: torch.Tensor
    full_counts: torch.Tensor
    full_indices: torch.Tensor
    q_table_base: torch.Tensor
    k_group_ids: torch.Tensor
    allowed_table: torch.Tensor
    q_block_size: int
    kv_block_size: int


class _Fa4Entry(NamedTuple):
    flash_attn_func: Any
    block_sparse_cls: Any
    # mask_mod per ``allowed_table`` dtype: int8 key-expanded, int32 packed words
    mask_mods: dict[torch.dtype, Any]


_entry: _Fa4Entry | None = None


def _build_mask_mods(cutlass, cute, fa_utils) -> dict[torch.dtype, Any]:
    """The multiview mask_mods, one per form of the truth table.

    ``aux_tensors`` are, in order:

    * ``q_table_base`` ``(seqlen_q,)`` int32: the query row's offset into the
      table, ``q_run * seqlen_k`` (key-expanded) or ``q_run * words_per_row``.
    * ``k_group_ids`` ``(seqlen_k,)`` int32: the key token's run id, read only
      for the packed words.
    * ``allowed_table``: int8 over ``(q_run, key position)``, or int32 words
      over run pairs, bit-packed 32 key runs per word.

    FA4 wraps the indices modulo ``seqlen_q``/``seqlen_k`` before calling a
    mask_mod that has aux tensors, so reads on out-of-range padded lanes stay
    in bounds; those lanes are force-masked afterwards. That holds only while
    ``q_table_base``/``k_group_ids`` are exactly ``seqlen_q``/``seqlen_k`` long
    and a key-expanded row is ``seqlen_k`` wide, which
    ``fa4_block_sparse_attention`` checks.
    """

    @cute.jit
    def key_expanded_mask_mod(
        batch: Any,
        head: Any,
        m_idx: Any,
        n_idx: Any,
        seqlen_info: Any,
        aux_tensors: Any,
    ) -> Any:
        q_table_base = aux_tensors[0]
        allowed_bytes = aux_tensors[2]

        # The query row is shared by every lane of a scalar mask_mod call, so
        # its lookup is hoisted out of the per-key loop.
        base = q_table_base[m_idx[0]]
        result = cute.make_rmem_tensor(n_idx.shape, dtype=cutlass.Boolean)
        for j in cutlass.range_constexpr(cute.size(n_idx.shape)):
            # The table holds exactly 0 or 1, so the byte is the predicate.
            result[j] = cutlass.Boolean(allowed_bytes[base + cutlass.Int32(n_idx[j])])
        return result.load()

    @cute.jit
    def packed_mask_mod(
        batch: Any,
        head: Any,
        m_idx: Any,
        n_idx: Any,
        seqlen_info: Any,
        aux_tensors: Any,
    ) -> Any:
        q_table_base = aux_tensors[0]
        k_group_ids = aux_tensors[1]
        allowed_words = aux_tensors[2]

        base = q_table_base[m_idx[0]]
        result = cute.make_rmem_tensor(n_idx.shape, dtype=cutlass.Boolean)
        for j in cutlass.range_constexpr(cute.size(n_idx.shape)):
            group_k = k_group_ids[n_idx[j]]
            word = allowed_words[base + group_k // cutlass.Int32(32)]
            shift = cutlass.Uint32(group_k % cutlass.Int32(32))
            result[j] = cutlass.Boolean(
                fa_utils.shr_u32(cutlass.Uint32(word), shift) & cutlass.Uint32(1)
            )
        return result.load()

    return {torch.int8: key_expanded_mask_mod, torch.int32: packed_mask_mod}


def _load_fa4() -> _Fa4Entry:
    """Import FA4 and build the mask_mods once per process.

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

        from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_func
        from sglang.kernels.ops.attention.flash_attn.cute import utils as fa_utils
        from sglang.kernels.ops.attention.flash_attn.cute.block_sparsity import (
            BlockSparseTensorsTorch,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cosmos3 multiview backend='fa4' requires the vendored FlashAttention-4 "
            "CuTe kernels, which need CUDA 13 and an SM90 or SM100 GPU. "
            f"Import failed: {exc}"
        ) from exc

    _entry = _Fa4Entry(
        flash_attn_func=flash_attn_func,
        block_sparse_cls=BlockSparseTensorsTorch,
        mask_mods=_build_mask_mods(cutlass, cute, fa_utils),
    )
    logger.info("Cosmos3 multiview attention using the FlashAttention-4 CuTe backend.")
    return _entry


def _validate_tensors(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 4:
            raise ValueError(
                f"Cosmos3 FA4 expects [B, S, H, D] {name}, got {tuple(tensor.shape)}."
            )
        if tensor.device.type != "cuda":
            raise ValueError(
                f"Cosmos3 FA4 requires CUDA tensors, {name} is on {tensor.device}."
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Cosmos3 FA4 supports {[str(d) for d in _SUPPORTED_DTYPES]}, "
                f"{name} has dtype {tensor.dtype}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"Cosmos3 FA4 requires contiguous [B, S, H, D] {name}.")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError(
            "Cosmos3 FA4 key/value geometry mismatch: "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}."
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
        q_table_base: torch.Tensor,
        k_group_ids: torch.Tensor,
        allowed_table: torch.Tensor,
        q_block_size: int,
        kv_block_size: int,
    ) -> torch.Tensor:
        """``Fa4BlockSparsity`` flattened to the tensors/ints a schema allows."""
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
            mask_mod=entry.mask_mods[allowed_table.dtype],
            # The order the mask_mods of _build_mask_mods index them in.
            aux_tensors=[q_table_base, k_group_ids, allowed_table],
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
        q_table_base,
        k_group_ids,
        allowed_table,
        q_block_size,
        kv_block_size,
    ):
        # FA4 returns the query layout unchanged: [B, S_q_padded, H_q, D].
        return torch.empty_like(q)


_cosmos3_multiview_fa4_op = getattr(getattr(torch.ops, _OP_NAMESPACE), _OP_NAME)


def fa4_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparsity: Fa4BlockSparsity,
) -> torch.Tensor:
    """Run FA4 over contiguous ``[B, S, H, D]`` tensors with a block-sparse mask.

    ``softmax_scale`` is left to FA4's default of ``1/sqrt(head_dim)``, which is
    also the FlexAttention default the Triton paths rely on. Validation stays
    outside the custom op so shape errors name this function rather than a
    schema mismatch.
    """
    _validate_tensors(q, k, v)
    # FA4 wraps aux reads modulo these lengths; a shorter tensor would alias.
    if q.shape[1] != sparsity.q_table_base.numel():
        raise ValueError(
            "Cosmos3 FA4 padded query length must match the block map: "
            f"q={q.shape[1]}, q_table_base={sparsity.q_table_base.numel()}."
        )
    if k.shape[1] != sparsity.k_group_ids.numel():
        raise ValueError(
            "Cosmos3 FA4 padded key length must match the block map: "
            f"k={k.shape[1]}, k_group_ids={sparsity.k_group_ids.numel()}."
        )
    table = sparsity.allowed_table
    if table.dtype == torch.int8 and table.numel() % k.shape[1]:
        raise ValueError(
            "Cosmos3 FA4 key-expanded mask table rows must span the padded keys: "
            f"k={k.shape[1]}, allowed_table={table.numel()}."
        )
    expected = fa4_sparse_block_sizes(q.device)
    if (sparsity.q_block_size, sparsity.kv_block_size) != expected:
        raise ValueError(
            f"Cosmos3 FA4 requires a {expected} sparse block map on compute "
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
        sparsity.q_table_base,
        sparsity.k_group_ids,
        sparsity.allowed_table,
        sparsity.q_block_size,
        sparsity.kv_block_size,
    )


def multiview_fa4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: MultiviewBlockSparsity,
) -> torch.Tensor:
    """FA4 over the Transfer-Auto ``[UND | GEN]`` stream with its multiview block map."""
    if q.shape[1] != sparsity.q_len or k.shape[1] != sparsity.kv_len:
        raise ValueError(
            "Cosmos3 multiview FA4 padded geometry must match the block map: "
            f"q={q.shape[1]}, k={k.shape[1]}, mask=({sparsity.q_len}, {sparsity.kv_len})."
        )
    return fa4_block_sparse_attention(
        q,
        k,
        v,
        sparsity=Fa4BlockSparsity(
            partial_counts=sparsity.partial_counts,
            partial_indices=sparsity.partial_indices,
            full_counts=sparsity.full_counts,
            full_indices=sparsity.full_indices,
            q_table_base=sparsity.q_table_base,
            k_group_ids=sparsity.k_group_ids,
            allowed_table=sparsity.allowed_table,
            q_block_size=sparsity.q_block_size,
            kv_block_size=sparsity.kv_block_size,
        ),
    )
