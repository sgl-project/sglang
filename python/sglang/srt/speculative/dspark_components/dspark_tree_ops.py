"""NPU tree-op adapter layer for the DSV4 DSpark speculative path.

This module is the single NPU device boundary for the DSpark tree algorithm: it
owns the per-step node plan, the four tree operators (``compressor_tree_project``,
``compressor_tree``, ``quant_lightning_indexer_tree``,
``sparse_attn_sharedkv_tree``), the proven node-page commit kernels, and the SWA
KV move. A GPU implementation may replace this module with the same public
signatures and the same invariants; the rest of the speculative stack must not
contain any other device-specific tree code.

Projection, compression and indexing use the in-tree Triton package.
``SGLANG_DSPARK_TREE_ATTN_IMPL=ascendc`` selects the sgl-kernel-npu AscendC
attention, and ``triton`` selects Triton. When unset, attention follows
``SGLANG_DSPARK_TREE_BACKEND``:

* ``sgl_kernel_npu`` (default): use the in-tree Triton package.
* ``vendor``: load the vllm-ascend ``_C_ascend`` bindings listed in
  ``SGLANG_DSPARK_TREE_OPS`` and call its AscendC tree attention operator.

The public functions keep the same signatures for every backend. If an
operator is unavailable the selected backend raises ``RuntimeError`` instead of
silently falling through to a different implementation.

Invariants (mirroring the reference ``tree/`` package):

* token-level ``req_to_token``/``block_size = 1`` — the commit and compressor
  kernels index the absolute page table by ``position // rows`` (one row per
  token), never by a >1 block size.
* node id != KV slot — every tree node consumes its *own* hidden row
  (``hidden_ids == arange(n)``); the node id is a tree coordinate, not a cache
  slot.
* ancestors globalized per request + request-root padding — ``build_node_plan``
  adds each request's row offset to every ancestor and pads ``-1`` entries with
  the request root offset (vLLM convention).
* RoPE clamp / full absolute state+RoPE tables — ``compress_nodes`` forwards the
  caller's *full* absolute ``table`` / ``cos`` / ``sin`` unchanged; it never
  re-slices them (the caller is responsible for the clamped full tables).
* trim/pad rows — ``compress_nodes`` trims a padded attention-TP hidden tail to
  ``plan.n``; the attention caller trims ``q`` to ``plan.n`` and pads the output.
* compressor skip ``is_in_indexer`` — the *caller* routes the indexer compressor
  through ``quantize_indexer_nodes`` + ``record_compressed_nodes`` instead of
  ``compress_nodes``'s ``node_kv`` sink; this module exposes both paths.
* all-or-none tree attention fields — ``build_tree_attn_kwargs`` zeroes
  ``tree_ancestors``/``tree_prefix`` whenever ``cmp_tree_kv`` is absent.
* SWA-only layers use the base ``torch.ops.npu.sparse_attn_sharedkv`` op (not
  the tree op); handled by the backend, not this module.
* hidden commit order — ``freeze_node_pages`` resolves a page row by ascending
  request priority and latest accepted depth, then commits staging in one pass.
* per-step ``NodePlan`` cache — ``get_node_plan`` builds the plan once per verify
  step and reuses it across compressor/indexer/attention layers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

_TREE_BACKEND_ENV = "SGLANG_DSPARK_TREE_BACKEND"
_TREE_ATTN_IMPL_ENV = "SGLANG_DSPARK_TREE_ATTN_IMPL"
_TREE_OPS_ENV = "SGLANG_DSPARK_TREE_OPS"
_TREE_BACKENDS = ("vendor", "sgl_kernel_npu")
_TREE_BACKEND_ALIASES = {
    "native": "vendor",
    "vllm": "vendor",
    "triton": "sgl_kernel_npu",
    "sgl": "sgl_kernel_npu",
}
_TREE_OPS_LOADED = False
_BYTE_BLOCK = 512

__all__ = [
    "NodePlan",
    "NodeRecord",
    "build_node_plan",
    "get_node_plan",
    "compress_nodes",
    "tree_indexer",
    "quantize_indexer_nodes",
    "tree_indexer_select_all",
    "build_tree_attn_kwargs",
    "build_tree_attn_metadata",
    "attend_nodes",
    "record_compressed_nodes",
    "prepare_nodes",
    "commit_nodes",
    "move_swa_kv",
]


# ---------------------------------------------------------------------------
# Node plan
# ---------------------------------------------------------------------------


@dataclass
class SwaMovePlan:
    """Cached SWA row mapping and per-layer gather scratch."""

    source: torch.Tensor
    destination: torch.Tensor
    scratch: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class NodePlan:
    """Per-verify-step tree geometry shared by compressor, indexer and attention.

    ``n`` is the flat node/row count across the batch; ``max_q`` is the per-request
    verify width (root + budget). ``ancestors`` is already globalized to flat
    verify-row ids with the request root used for padding.
    """

    n: int
    max_q: int
    start: torch.Tensor  # [n] int32 prefix base per node
    depths: torch.Tensor  # [n] int64 depth per node (root = 0)
    ancestors: torch.Tensor  # [n, A] int64 root-first node ids
    req_ids: torch.Tensor  # [n] int64 request index per node
    query_cu: torch.Tensor  # [n + 1] int32
    hidden_ids: torch.Tensor  # [n] int64
    token_index: torch.Tensor  # [n] int64
    max_prefix: int | None = None
    window_pages_cache: dict[tuple[int, int, int, int], tuple[int, ...]] = field(
        default_factory=dict
    )
    cu_seqlens_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = (
        field(default_factory=dict)
    )
    max_seqlen_cache: dict[tuple[int, int, int], int] = field(default_factory=dict)
    attn_metadata_cache: dict[tuple, torch.Tensor] = field(default_factory=dict)
    indexed_table_cache: dict[
        tuple, tuple[torch.Tensor | None, torch.Tensor | None]
    ] = field(default_factory=dict)
    swa_move_plan: SwaMovePlan | None = None
    indexer_lengths_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    max_kv_cache: int | None = None
    indexer_metadata_cache: dict[
        tuple[tuple[int, ...], int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = field(default_factory=dict)
    select_all_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    residue_cache: dict[int, torch.Tensor] = field(default_factory=dict)


def build_node_plan(pre) -> NodePlan:
    """Derive the node plan from a ``TreePrePlan`` (no device readback).

    ``pre.metadata`` must expose ``verify_ids``, ``tree_ancestors``,
    ``node_req_ids``, ``query_width``, ``tree_prefix``, ``positions`` and
    ``node_cu_seqlens_q`` (the ``TreeMetadata`` contract of the reference
    ``tree/metadata.py``).
    """
    meta = pre.metadata
    n = meta.verify_ids.numel()
    device = meta.verify_ids.device
    ancestors = meta.tree_ancestors.to(torch.int64)
    offsets = meta.node_req_ids.to(torch.int64) * meta.query_width
    # vLLM's build_ancestors uses the request's root node for padding, not
    # -1; vendor tree indexer/attention gather this table directly.
    ancestors = torch.where(
        ancestors >= 0,
        ancestors + offsets[:, None],
        offsets[:, None],
    )
    return NodePlan(
        n=n,
        max_q=meta.query_width,
        start=meta.tree_prefix.to(torch.int32),
        depths=(meta.positions - meta.tree_prefix.to(meta.positions.dtype)).to(
            torch.int64
        ),
        ancestors=ancestors,
        req_ids=meta.node_req_ids.to(torch.int64),
        query_cu=meta.node_cu_seqlens_q.to(torch.int32),
        hidden_ids=torch.arange(n, dtype=torch.int64, device=device),
        token_index=torch.arange(n, dtype=torch.int64, device=device),
        max_prefix=pre.max_prefix,
    )


def get_node_plan(pre) -> NodePlan:
    """Return the per-step node plan, building it at most once.

    The reference builds the tree plan once per verify step and reuses it for the
    compressor, indexer and attention; rebuilding the same tensors several times
    per layer adds host work and device synchronization to every decode step.
    """
    plan = getattr(pre, "node_plan", None)
    if plan is None:
        plan = build_node_plan(pre)
        pre.node_plan = plan
    return plan


# ---------------------------------------------------------------------------
# Backend selection and operator resolution
# ---------------------------------------------------------------------------


def _tree_backend() -> str:
    """Return the selected tree backend (vendor / sgl_kernel_npu)."""
    raw = os.environ.get(_TREE_BACKEND_ENV, "sgl_kernel_npu").strip().lower()
    backend = _TREE_BACKEND_ALIASES.get(raw, raw)
    if backend not in _TREE_BACKENDS:
        raise ValueError(
            f"{_TREE_BACKEND_ENV} must be one of {_TREE_BACKENDS}, got {raw!r}"
        )
    return backend


def _tree_attn_impl() -> str:
    impl = os.environ.get(_TREE_ATTN_IMPL_ENV, "").strip().lower()
    if not impl:
        return "vendor" if _tree_backend() == "vendor" else "triton"
    if impl not in ("ascendc", "triton"):
        raise ValueError(f"{_TREE_ATTN_IMPL_ENV} must be ascendc or triton, got {impl!r}")
    return impl


def _ensure_vendor_ops() -> None:
    """Load the vllm-ascend tree bindings from SGLANG_DSPARK_TREE_OPS once."""
    global _TREE_OPS_LOADED
    if _TREE_OPS_LOADED:
        return
    raw_paths = os.environ.get(_TREE_OPS_ENV, "")
    loaded: list[str] = []
    for raw in raw_paths.split(os.pathsep):
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(path):
            continue
        try:
            torch.ops.load_library(path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load DSpark tree op binding {path!r}: {exc}"
            ) from exc
        loaded.append(path)
    _TREE_OPS_LOADED = True
    if not loaded and _tree_backend() == "vendor":
        raise RuntimeError(
            f"{_TREE_BACKEND_ENV}=vendor requires at least one tree binding in "
            f"{_TREE_OPS_ENV}; no existing .so was found."
        )


def _vendor_namespace():
    namespace = getattr(torch.ops, "_C_ascend", None)
    if namespace is None:
        raise RuntimeError(
            "torch.ops._C_ascend is unavailable; set SGLANG_DSPARK_TREE_OPS "
            "to the vllm-ascend tree binding .so files."
        )
    return namespace


def _vendor_compressor_is_fused(project) -> bool:
    """Return whether the vendor binding exposes the fused aggregation inputs."""
    try:
        schema = str(project.default._schema)
    except AttributeError:
        try:
            schema = str(project._schema)
        except AttributeError:
            return False
    return all(name in schema for name in ("state", "ancestors", "norm_weight"))


def _load_compressor_ops(*, backend: str | None = None):
    if (backend or _tree_backend()) == "vendor":
        _ensure_vendor_ops()
        namespace = _vendor_namespace()
        try:
            project = namespace.compressor_tree_project
        except AttributeError as exc:
            raise RuntimeError(
                "the loaded vllm-ascend binding does not expose "
                "compressor_tree_project"
            ) from exc
        fused = _vendor_compressor_is_fused(project)
        tree = getattr(namespace, "compressor_tree", None)
        if tree is None and not fused:
            raise RuntimeError(
                "the loaded vllm-ascend binding exposes neither a fused "
                "compressor_tree_project nor compressor_tree"
            )
        return project, tree, fused
    try:
        from sgl_kernel_npu.tree import compressor_tree
        from sgl_kernel_npu.tree import compressor_tree_project
    except Exception as exc:
        raise RuntimeError(
            "DSpark tree compression requires the sgl_kernel_npu.tree compressor "
            "package (compressor_tree / compressor_tree_project); it is "
            "unavailable in this process."
        ) from exc
    return compressor_tree_project, compressor_tree, False


def _load_sgl_kernel_indexer_op():
    try:
        from sgl_kernel_npu.tree import quant_lightning_indexer_tree
    except Exception as exc:
        raise RuntimeError(
            "DSpark tree indexing requires the sgl_kernel_npu.tree indexer "
            "package (quant_lightning_indexer_tree); it is unavailable in this "
            "process."
        ) from exc
    return quant_lightning_indexer_tree


def _load_vendor_indexer_op():
    _ensure_vendor_ops()
    namespace = _vendor_namespace()
    try:
        return namespace.quant_lightning_indexer_tree
    except AttributeError as exc:
        raise RuntimeError(
            "the loaded vllm-ascend binding does not expose "
            "quant_lightning_indexer_tree"
        ) from exc


def _load_sgl_ascendc_op(name: str):
    try:
        import sgl_kernel_npu  # noqa: F401 - registers torch.ops.npu

        return getattr(torch.ops.npu, name)
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError(
            f"DSpark AscendC tree attention requires torch.ops.npu.{name}; "
            "install the A3 sgl-kernel-npu wheel with native tree attention."
        ) from exc


def _load_attention_op():
    impl = _tree_attn_impl()
    if impl == "ascendc":
        return _load_sgl_ascendc_op("sparse_attn_sharedkv_tree")
    if impl == "vendor":
        _ensure_vendor_ops()
        namespace = _vendor_namespace()
        try:
            return namespace.npu_sparse_attn_sharedkv_tree
        except AttributeError as exc:
            raise RuntimeError(
                "the loaded vllm-ascend binding does not expose "
                "npu_sparse_attn_sharedkv_tree"
            ) from exc
    try:
        from sgl_kernel_npu.tree import sparse_attn_sharedkv_tree
    except Exception as exc:
        raise RuntimeError(
            "DSpark tree attention requires the sgl_kernel_npu.tree attention "
            "package (sparse_attn_sharedkv_tree); it is unavailable in this "
            "process."
        ) from exc
    return sparse_attn_sharedkv_tree


# ---------------------------------------------------------------------------
# Node compression (compressor_tree_project -> compressor_tree)
# ---------------------------------------------------------------------------


def compress_nodes(
    module,
    hidden: torch.Tensor,
    state: torch.Tensor,
    table: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: NodePlan,
    *,
    record_sink: list | None = None,
    kv_buffer: torch.Tensor | None = None,
    kv_table: torch.Tensor | None = None,
):
    """Project and aggregate per-node compressed KV.

    ``module`` must expose the sglang DSV4 compressor attributes: ``ratio``,
    ``overlap``, ``head_dim``, ``ape``, ``rope_head_dim``, ``norm``,
    ``_fused_wkv_w``, ``_fused_wgate_w``, ``_fused_norm_weight_fp32``.
    Returns ``(node_kv [n, head] bf16, nodes [(n+1), 1, 1, 2*coff*head] fp32)``.

    ``state``/``table``/``cos``/``sin`` are the caller's *full absolute* state and
    RoPE tables (token-level, ``block_size = 1``); they are forwarded to the
    compressor unchanged.
    """
    for name in ("_fused_wkv_w", "_fused_wgate_w", "_fused_norm_weight_fp32"):
        if getattr(module, name, None) is None:
            raise RuntimeError(
                f"compressor is missing {name}; call the backend's "
                "_ensure_fused_caches() before tree compression."
            )
    if hidden.shape[0] < plan.n:
        raise ValueError(
            f"tree node hidden rows {hidden.shape[0]} < plan.n {plan.n}; "
            "the target verify did not produce every tree node."
        )
    if hidden.shape[0] > plan.n:
        # Target verify pads the attention-TP tail with unused rows.
        hidden = hidden[: plan.n]
    if plan.ancestors.shape[0] != plan.n:
        raise ValueError("ancestors row count must equal plan.n")
    coff = 1 + int(module.overlap)
    head = int(module.head_dim)
    device = hidden.device
    nodes = torch.zeros(
        (plan.n + 1, 1, 1, 2 * coff * head), dtype=torch.float32, device=device
    )
    ratio = int(module.ratio)
    residue = plan.residue_cache.get(ratio)
    if residue is None:
        residue = ((plan.start.long() + plan.depths) % ratio).to(torch.int32)
        plan.residue_cache[ratio] = residue
    # build_node_plan keeps hidden_ids as arange(n); avoid a redundant gather.
    hidden_nodes = hidden[: plan.n]
    compressor_tree_project, compressor_tree, fused = _load_compressor_ops(
        backend="sgl_kernel_npu"
    )
    if fused:
        node_kv = compressor_tree_project(
            hidden_nodes,
            module._fused_wkv_w,
            module._fused_wgate_w,
            nodes,
            module.ape,
            residue,
            plan.query_cu,
            state=state,
            prefix=plan.start,
            ancestors=plan.ancestors,
            depths=plan.depths,
            requests=plan.req_ids,
            norm_weight=module._fused_norm_weight_fp32,
            cos=cos,
            sin=sin,
            table=table,
        )
    else:
        compressor_tree_project(
            hidden_nodes,
            module._fused_wkv_w,
            module._fused_wgate_w,
            nodes,
            module.ape,
            residue,
            plan.query_cu,
        )
        node_kv = compressor_tree(
            state,
            nodes,
            module._fused_norm_weight_fp32,
            cos,
            sin,
            table,
            plan.start,
            plan.ancestors,
            plan.depths,
            plan.req_ids,
            int(module.ratio),
            int(module.rope_head_dim),
            float(module.norm.variance_epsilon),
        )
    if record_sink is not None:
        # ratio=1 records the projected state row (one per node); the commit
        # kernel treats ratio=1 as state and writes it back to the state cache.
        record_sink.append(
            record_compressed_nodes(
                plan,
                original=state,
                node_data=nodes[1:],
                table=table,
                ratio=1,
            )
        )
        if kv_buffer is not None and kv_table is not None:
            record_sink.append(
                record_compressed_nodes(
                    plan,
                    original=kv_buffer,
                    node_data=node_kv,
                    table=kv_table,
                    ratio=int(module.ratio),
                )
            )
    return node_kv, nodes


# ---------------------------------------------------------------------------
# Tree attention (sparse_attn_sharedkv_tree)
# ---------------------------------------------------------------------------


def _cached_cu_seqlens(
    plan: NodePlan, seq: torch.Tensor, ratio: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(ratio), int(seq.data_ptr()), int(seq.numel()))
    cached = plan.cu_seqlens_cache.get(key)
    if cached is not None:
        return cached
    cu_ori = torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32)))
    cu_cmp = torch.cat(
        (seq.new_zeros(1), (seq // int(ratio)).cumsum(0, dtype=torch.int32))
    )
    cached = (cu_ori, cu_cmp)
    plan.cu_seqlens_cache[key] = cached
    return cached


def _cached_max_seqlen(plan: NodePlan, seq: torch.Tensor, ratio: int) -> int:
    key = (int(ratio), int(seq.data_ptr()), int(seq.numel()))
    cached = plan.max_seqlen_cache.get(key)
    if cached is not None:
        return cached
    value = max(1, int(seq.max().item()) if seq.numel() else 0)
    plan.max_seqlen_cache[key] = value
    return value


def build_tree_attn_metadata(
    *, query, metadata, plan, ratio, window, cmp_sparse_indices, compressed
):
    """Build the tiling metadata the tree attention op requires.

    Triton ignores metadata contents. Both AscendC paths use a schedule
    computed from the actual query geometry and compressed index width.
    """
    seq = metadata.node_seqused_kv.to(torch.int32)
    cu_ori, cu_cmp = _cached_cu_seqlens(plan, seq, ratio)
    impl = _tree_attn_impl()
    if impl == "triton":
        metadata_tensor = torch.zeros(1024, dtype=torch.int32, device=query.device)
        return metadata_tensor, cu_ori, cu_cmp

    ratio = int(ratio)
    cmp_topk = (
        int(cmp_sparse_indices.shape[-1])
        if cmp_sparse_indices is not None and ratio == 4
        else 0
    )
    has_cmp_kv = compressed is not None
    if (
        impl == "ascendc"
        and has_cmp_kv
        and ratio == 4
        and cmp_topk not in (512, 1024)
    ):
        raise ValueError(
            "sgl-kernel-npu AscendC tree attention requires a C4 index width "
            f"of 512 or 1024 for host metadata, got {cmp_topk}."
        )
    cache_key = (
        impl,
        int(query.shape[1]),
        int(query.shape[2]),
        ratio,
        int(window),
        cmp_topk,
        has_cmp_kv,
        int(seq.data_ptr()),
        int(seq.numel()),
    )
    cached = plan.attn_metadata_cache.get(cache_key)
    if cached is not None:
        return cached, cu_ori, cu_cmp

    kwargs = dict(
        num_heads_q=query.shape[1],
        num_heads_kv=1,
        head_dim=query.shape[2],
        batch_size=plan.n,
        cmp_topk=cmp_topk,
        cmp_ratio=ratio,
        has_ori_kv=True,
        has_cmp_kv=has_cmp_kv,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(window) - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    if impl == "ascendc":
        op = _load_sgl_ascendc_op("sparse_attn_sharedkv_metadata_host")
        # The host scheduler reads CPU int32 lengths and returns an NPU tensor.
        metadata_tensor = op(
            cu_seqlens_q=plan.query_cu.to(device="cpu", dtype=torch.int32).contiguous(),
            seqused_kv=seq.to(device="cpu").contiguous(),
            **kwargs,
        )
    else:
        _ensure_vendor_ops()
        custom = getattr(torch.ops, "custom", None)
        op = getattr(custom, "npu_sparse_attn_sharedkv_metadata", None)
        if op is None:
            op = getattr(
                _vendor_namespace(), "npu_sparse_attn_sharedkv_metadata", None
            )
        if op is None:
            raise RuntimeError(
                "DSpark tree attention requires "
                "torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata or "
                "torch.ops.custom.npu_sparse_attn_sharedkv_metadata"
            )
        metadata_tensor = op(
            seqused_q=seq.new_empty(0),
            max_seqlen_q=1,
            max_seqlen_kv=_cached_max_seqlen(plan, seq, ratio),
            device=str(query.device),
            cu_seqlens_q=plan.query_cu,
            cu_seqlens_ori_kv=cu_ori,
            cu_seqlens_cmp_kv=cu_cmp,
            seqused_kv=seq,
            **kwargs,
        )
    plan.attn_metadata_cache[cache_key] = metadata_tensor
    return metadata_tensor, cu_ori, cu_cmp


def build_tree_attn_kwargs(
    *,
    q,
    ori_kv,
    ori_sparse_indices,
    cu_seqlens_q,
    seqused_kv,
    tree_ancestors,
    tree_prefix,
    cmp_ratio,
    softmax_scale,
    **kwargs,
):
    """Assemble the native op kwargs, validating the required contract.

    Tree pool inputs (``cmp_tree_kv``/``tree_ancestors``/``tree_prefix``) are
    all-or-none: absent ``cmp_tree_kv`` zeroes the other two, matching the op's
    CheckFeature requirement.
    """
    required = {
        "q": q,
        "ori_kv": ori_kv,
        "ori_sparse_indices": ori_sparse_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing tree attention inputs: {missing}")
    cmp_tree_kv = kwargs.pop("cmp_tree_kv", None)
    if cmp_tree_kv is not None:
        if tree_ancestors is None or tree_prefix is None:
            raise ValueError(
                "tree attention requires tree_ancestors and tree_prefix when "
                "cmp_tree_kv is supplied"
            )
    else:
        # SWA-only layers have no compressed tree KV. The op's CheckFeature
        # requires cmp_tree_kv / ancestors / prefixes to be all-or-none.
        tree_ancestors = None
        tree_prefix = None
    out = dict(required)
    out.update(
        cmp_kv=kwargs.pop("cmp_kv", None),
        cmp_sparse_indices=kwargs.pop("cmp_sparse_indices", None),
        ori_block_table=kwargs.pop("ori_block_table", None),
        cmp_block_table=kwargs.pop("cmp_block_table", None),
        sinks=kwargs.pop("sinks", None),
        metadata=kwargs.pop("metadata", None),
        cmp_tree_kv=cmp_tree_kv,
        tree_ancestors=tree_ancestors,
        tree_prefix=tree_prefix,
        cu_seqlens_ori_kv=kwargs.pop("cu_seqlens_ori_kv", None),
        cu_seqlens_cmp_kv=kwargs.pop("cu_seqlens_cmp_kv", None),
        cmp_ratio=int(cmp_ratio),
        softmax_scale=float(softmax_scale),
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(kwargs.pop("ori_win_left", 127)),
        ori_win_right=int(kwargs.pop("ori_win_right", 0)),
        layout_q="TND",
        layout_kv="PA_ND",
    )
    out.update(kwargs)
    return out


def _tree_attention(**kwargs) -> torch.Tensor:
    """Call the selected compressed tree attention operator."""
    op = _load_attention_op()
    op_kwargs = build_tree_attn_kwargs(**kwargs)
    q = op_kwargs.pop("q")
    return op(q, **op_kwargs)[0]


def _table_cache_token(table: torch.Tensor | None):
    if table is None:
        return (0, (), ())
    return (int(table.data_ptr()), tuple(table.shape), tuple(table.stride()))


def _cached_indexed_tables(
    plan: NodePlan,
    ori_block_table: torch.Tensor | None,
    cmp_block_table: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    key = (
        _table_cache_token(ori_block_table),
        _table_cache_token(cmp_block_table),
        int(plan.req_ids.data_ptr()),
        tuple(plan.req_ids.shape),
    )
    cached = plan.indexed_table_cache.get(key)
    if cached is not None:
        return cached
    ori = (
        ori_block_table.index_select(0, plan.req_ids)
        if ori_block_table is not None
        else None
    )
    cmp = (
        cmp_block_table.index_select(0, plan.req_ids)
        if cmp_block_table is not None
        else None
    )
    cached = (ori, cmp)
    plan.indexed_table_cache[key] = cached
    return cached


def attend_nodes(
    *,
    query: torch.Tensor,
    original: torch.Tensor,
    compressed: torch.Tensor,
    node_kv: torch.Tensor,
    metadata,
    plan: NodePlan,
    ori_sparse_indices: torch.Tensor,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    sinks,
    ratio: int,
    window: int,
    scale: float,
):
    """Tree attention over shared history + per-node compressed KV."""
    metadata_tensor, cu_ori, cu_cmp = build_tree_attn_metadata(
        query=query,
        metadata=metadata,
        plan=plan,
        ratio=ratio,
        window=window,
        cmp_sparse_indices=cmp_sparse_indices,
        compressed=compressed,
    )
    ori_table, cmp_table = _cached_indexed_tables(
        plan, ori_block_table, cmp_block_table
    )
    return _tree_attention(
        q=query,
        ori_kv=original,
        cmp_kv=compressed,
        cmp_tree_kv=node_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_table,
        cmp_block_table=cmp_table,
        cu_seqlens_q=plan.query_cu,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        seqused_kv=metadata.node_seqused_kv,
        tree_ancestors=plan.ancestors,
        tree_prefix=plan.start,
        sinks=sinks,
        metadata=metadata_tensor,
        cmp_ratio=ratio,
        softmax_scale=scale,
        ori_win_left=window - 1,
        ori_win_right=0,
    )


# ---------------------------------------------------------------------------
# Tree indexer (quant_lightning_indexer_tree)
# ---------------------------------------------------------------------------


def tree_indexer_select_all(plan: NodePlan, topk: int, device) -> torch.Tensor:
    """Host-capacity shortcut: when history/4 <= topk every block is selected."""
    cached = plan.select_all_cache.get(int(topk))
    if cached is not None:
        return cached
    lengths = _cached_indexer_lengths(plan)[0]
    positions = torch.arange(topk, dtype=torch.int32, device=device).expand(
        plan.n, -1
    )
    selected = torch.where(
        positions < lengths.unsqueeze(1).to(torch.int32),
        positions,
        torch.full_like(positions, -1),
    ).unsqueeze(1)
    plan.select_all_cache[int(topk)] = selected
    return selected


def quantize_indexer_nodes(node_kv: torch.Tensor):
    """Quantize bf16 per-node indexer K to INT8 + FP16 per-node scale.

    Mirrors the indexer's own per-token dynamic quant. The tree op requires
    ``nodes`` INT8 [n, 128] and ``node_scale`` FP16 [n].
    """
    import torch_npu

    flat = node_kv.reshape(-1, node_kv.shape[-1])
    nodes, scale = torch_npu.npu_dynamic_quant(flat)
    return nodes.view(node_kv.shape), scale.reshape(-1).to(torch.float16)


def _tree_indexer_lengths(plan: NodePlan):
    seq = (plan.start.long() + plan.depths + 1).int()
    return seq // 4, seq % 4


def _cached_indexer_lengths(plan: NodePlan) -> tuple[torch.Tensor, torch.Tensor]:
    cached = plan.indexer_lengths_cache
    if cached is None:
        cached = _tree_indexer_lengths(plan)
        plan.indexer_lengths_cache = cached
    return cached


def _cached_max_kv(plan: NodePlan) -> int:
    cached = plan.max_kv_cache
    if cached is None:
        if plan.max_prefix is not None:
            # ancestors width is max_depth + 1, so max_prefix + width is
            # max(start) + max(depth) + 1 without a device readback.
            cached = plan.max_prefix + int(plan.ancestors.shape[1])
        else:
            cached = int(plan.start.max().item() + plan.depths.max().item() + 1)
        plan.max_kv_cache = cached
    return cached


def _tree_indexer_metadata(
    *, query: torch.Tensor, plan: NodePlan, topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the vendor candidate-block schedule for the tree indexer."""
    cache_key = (tuple(query.shape), int(topk))
    cached = plan.indexer_metadata_cache.get(cache_key)
    if cached is not None:
        return cached
    lengths, residual = _cached_indexer_lengths(plan)
    max_kv = _cached_max_kv(plan)
    _ensure_vendor_ops()
    namespace = _vendor_namespace()
    op = getattr(namespace, "npu_quant_lightning_indexer_v2_metadata", None)
    if op is None:
        custom = getattr(torch.ops, "custom", None)
        op = getattr(custom, "npu_quant_lightning_indexer_v2_metadata", None)
    if op is None:
        raise RuntimeError(
            "DSpark tree indexer requires "
            "npu_quant_lightning_indexer_v2_metadata from the vllm-ascend "
            "binding"
        )
    schedule = op(
        num_heads_q=query.shape[1],
        num_heads_k=1,
        head_dim=query.shape[2],
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=plan.query_cu,
        seqused_k=lengths,
        cmp_residual_k=residual,
        batch_size=plan.n,
        max_seqlen_q=1,
        max_seqlen_k=max(1, max_kv // 4),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device=str(query.device),
    )
    cached = (lengths, residual, schedule)
    plan.indexer_metadata_cache[cache_key] = cached
    return cached


def _invoke_tree_indexer(
    op,
    *,
    query,
    weights,
    query_scale,
    history,
    history_scale,
    nodes,
    node_scale,
    table,
    plan,
    topk,
    lengths,
    residual,
    schedule,
):
    return op(
        query,
        history,
        weights.reshape(query.shape[:2]).half(),
        query_scale.reshape(query.shape[:2]).half(),
        history_scale,
        nodes.reshape(query.shape[0], query.shape[2]),
        node_scale,
        plan.ancestors.to(torch.int64),
        plan.start.to(torch.int32),
        plan.req_ids.to(torch.int64),
        table.to(torch.int32),
        plan.query_cu.to(torch.int32),
        lengths,
        residual,
        schedule,
        topk,
    )


def _sgl_kernel_tree_indexer(**kwargs) -> torch.Tensor:
    op = _load_sgl_kernel_indexer_op()
    plan = kwargs["plan"]
    query = kwargs["query"]
    if kwargs["lengths"] is None or kwargs["residual"] is None:
        kwargs["lengths"], kwargs["residual"] = _cached_indexer_lengths(plan)
    if kwargs["schedule"] is None:
        # The Triton op validates the metadata ABI and ignores its contents.
        kwargs["schedule"] = torch.zeros(
            1024, dtype=torch.int32, device=query.device
        )
    return _invoke_tree_indexer(op, **kwargs)


def _vendor_tree_indexer(**kwargs) -> torch.Tensor:
    op = _load_vendor_indexer_op()
    plan = kwargs["plan"]
    query = kwargs["query"]
    if (
        kwargs["lengths"] is None
        or kwargs["residual"] is None
        or kwargs["schedule"] is None
    ):
        lengths, residual, schedule = _tree_indexer_metadata(
            query=query, plan=plan, topk=kwargs["topk"]
        )
        kwargs["lengths"] = lengths
        kwargs["residual"] = residual
        kwargs["schedule"] = schedule
    return _invoke_tree_indexer(op, **kwargs)


def tree_indexer(
    *,
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    history: torch.Tensor,
    history_scale: torch.Tensor,
    nodes: torch.Tensor,
    node_scale: torch.Tensor,
    table: torch.Tensor,
    plan: NodePlan,
    topk: int,
    lengths: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the TopK compressed-KV indices for every tree node, ``[n, 1, topk]``."""
    return _sgl_kernel_tree_indexer(
        query=query,
        weights=weights,
        query_scale=query_scale,
        history=history,
        history_scale=history_scale,
        nodes=nodes,
        node_scale=node_scale,
        table=table,
        plan=plan,
        topk=topk,
        lengths=lengths,
        residual=residual,
        schedule=schedule,
    )


# ---------------------------------------------------------------------------
# Node-page commit (record -> freeze -> commit)
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    """One layer's compressed KV buffer plus its per-node rows."""

    original: torch.Tensor  # [pages, rows, ..., dim]
    node_data: torch.Tensor  # [n, ..., dim] contiguous
    table: torch.Tensor  # [reqs, width] int32 page table
    pages: tuple[int, ...]
    ratio: int


def _window_pages(plan: NodePlan, table: torch.Tensor, ratio: int, rows: int):
    cache = getattr(plan, "window_pages_cache", None)
    if cache is None:
        cache = {}
        plan.window_pages_cache = cache
    cache_key = (
        int(table.data_ptr()),
        int(table.shape[1]),
        int(ratio),
        int(rows),
    )
    if cache_key in cache:
        return cache[cache_key]

    if plan.req_ids.numel() == 0:
        cache[cache_key] = ()
        return ()
    # NodePlan.req_ids is built as request-major contiguous rows, so the first
    # row of request r is r * max_q.  Keep the table on-device and only move
    # the (small) unique page set back to the host.
    depth_limit = int(plan.max_q)
    if depth_limit <= 0:
        cache[cache_key] = ()
        return ()
    n_req = plan.req_ids.numel() // depth_limit
    device = plan.start.device
    first = torch.arange(n_req, dtype=torch.long, device=device) * depth_limit
    starts = plan.start[first].long()
    positions = starts.unsqueeze(1) + torch.arange(
        depth_limit, dtype=torch.long, device=device
    )
    cols = (positions // ratio) // rows
    table_width = int(table.shape[1])
    valid = cols < table_width
    rows_idx = torch.arange(n_req, dtype=torch.long, device=device).unsqueeze(1)
    if table_width == 0:
        cache[cache_key] = ()
        return ()
    page = table.long()[rows_idx, cols.clamp(max=table_width - 1)]
    pages = page[valid & (page > 0)]
    if pages.numel() == 0:
        cache[cache_key] = ()
        return ()
    result = tuple(int(x) for x in torch.unique(pages).tolist())
    cache[cache_key] = result
    return result


def record_compressed_nodes(
    plan: NodePlan,
    *,
    original: torch.Tensor,
    node_data: torch.Tensor,
    table: torch.Tensor,
    ratio: int,
) -> NodeRecord:
    """Record the pages and node rows for the post-accept commit kernels."""
    if ratio not in (1, 4, 128):
        raise ValueError(f"tree node commit ratio must be 1/4/128, got {ratio}")
    if original.dim() < 3:
        raise ValueError("original compressed buffer must be [pages, rows, ...]")
    if node_data.shape[0] != plan.n:
        raise ValueError(f"node rows {node_data.shape[0]} != plan.n {plan.n}")
    if not node_data.is_contiguous():
        raise ValueError("node_data must be contiguous for the commit kernel")
    rows = int(original.shape[1])
    row_bytes = 1
    for dim in original.shape[2:]:
        row_bytes *= int(dim)
    row_bytes *= original.element_size()
    node_row_bytes = 1
    for dim in node_data.shape[1:]:
        node_row_bytes *= int(dim)
    node_row_bytes *= node_data.element_size()
    if node_row_bytes != row_bytes:
        raise ValueError(
            f"node row bytes {node_row_bytes} != cache row bytes {row_bytes}"
        )
    return NodeRecord(
        original=original,
        node_data=node_data,
        table=table,
        pages=_window_pages(plan, table, ratio, rows),
        ratio=int(ratio),
    )


def build_node_descriptors(records: list[NodeRecord]):
    """Pure descriptor list for freeze_node_pages; also returns staging size."""
    descriptors: list[tuple[int, ...]] = []
    total = 0
    max_rows = 1
    for record in records:
        original = record.original
        element = original.element_size()
        rows = int(original.shape[1])
        row_bytes = 1
        for dim in original.shape[2:]:
            row_bytes *= int(dim)
        row_bytes *= element
        page_stride = int(original.stride(0)) * element
        row_stride = int(original.stride(1)) * element
        for page in record.pages:
            descriptors.append(
                (
                    original.data_ptr() + page * page_stride,
                    row_stride,
                    row_bytes,
                    rows,
                    record.node_data.data_ptr(),
                    record.table.data_ptr(),
                    int(record.table.shape[1]),
                    int(page),
                    total,
                    int(record.ratio),
                )
            )
            total += rows * row_bytes
            max_rows = max(max_rows, rows)
    return descriptors, total, max_rows


def globalize_winners(plan: NodePlan, winners: torch.Tensor) -> torch.Tensor:
    """Map per-request endpoint node ids to flat verify-row ids.

    ``acceptance.endpoint_node`` is request-local (0..query_width-1). The commit
    kernel indexes ``Depths`` / ``Ancestors`` by flat row, so every request after
    the first needs its row offset added.
    """
    offsets = torch.arange(
        winners.numel(), dtype=winners.dtype, device=winners.device
    ) * int(plan.max_q)
    return winners + offsets


_NODE_COMMIT_KERNELS = None


def _node_commit_kernels():
    """Lazily JIT the two Triton node-page kernels (never import Triton eagerly)."""
    global _NODE_COMMIT_KERNELS, tl
    if _NODE_COMMIT_KERNELS is not None:
        return _NODE_COMMIT_KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise NotImplementedError(
            "DSpark tree node commit requires Triton (freeze_node_pages / "
            "commit_node_pages); install triton to commit node pages."
        ) from exc

    @triton.jit
    def freeze_node_pages(
        Desc,
        Ancestors,
        Depths,
        Starts,
        Req,
        Winners,
        Active,
        Staging,
        PageActive,
        N: tl.constexpr,
        MAX_Q: tl.constexpr,
        BN: tl.constexpr,
        CBLOCK: tl.constexpr,
    ):
        job = tl.program_id(0)
        d = Desc + job * 10
        original = tl.load(d).to(tl.pointer_type(tl.uint8))
        row_stride = tl.load(d + 1)
        row_bytes = tl.load(d + 2)
        rows = tl.load(d + 3)
        node_data = tl.load(d + 4).to(tl.pointer_type(tl.uint8))
        table = tl.load(d + 5).to(tl.pointer_type(tl.int32))
        width = tl.load(d + 6)
        page_id = tl.load(d + 7)
        stage_offset = tl.load(d + 8)
        ratio = tl.load(d + 9)
        node = tl.arange(0, BN)
        req = tl.load(Req + node, node < N, other=0)
        winner = tl.load(Winners + req)
        depth = tl.load(Depths + node, node < N, other=0)
        win_depth = tl.load(Depths + winner)
        ancestor = tl.load(Ancestors + winner * MAX_Q + depth)
        selected = (node < N) & (depth <= win_depth) & (ancestor == node) & tl.load(
            Active + req
        )
        pos = tl.load(Starts + node, node < N, other=0) + depth
        # ratio=1 denotes state; compressed KV is emitted at the window boundary.
        selected = selected & ((pos + 1) % ratio == 0)
        position = (pos + 1) // ratio - 1
        row_of = position % rows
        col_of = position // rows
        page = tl.load(
            table + req * width + col_of,
            (node < N) & (col_of < width),
            other=0,
        )
        on_page = selected & (page == page_id)
        enabled = tl.sum(on_page.to(tl.int32), 0) > 0
        tl.store(PageActive + job, enabled)
        if enabled:
            # Ascending request priority, latest accepted depth within a request.
            priority = (N - req) * (MAX_Q + 1) + depth
            lanes = tl.arange(0, CBLOCK)
            for r in tl.range(0, rows):
                match = on_page & (row_of == r)
                best = tl.max(tl.where(match, priority, -1), 0)
                chosen = tl.max(tl.where(match & (priority == best), node, -1), 0)
                # Select the row by value; a pointer phi across different
                # buffers is rejected by the Ascend Triton backend.
                chosen_safe = tl.maximum(chosen, 0)
                node_src = node_data + chosen_safe * row_bytes
                orig_src = original + r * row_stride
                for begin in tl.range(0, row_bytes, CBLOCK):
                    offset = begin + lanes
                    in_range = offset < row_bytes
                    node_value = tl.load(
                        node_src + offset, in_range & (chosen >= 0), other=0
                    )
                    orig_value = tl.load(
                        orig_src + offset, in_range & (chosen < 0), other=0
                    )
                    value = tl.where(chosen >= 0, node_value, orig_value)
                    tl.store(
                        Staging + stage_offset + r * row_bytes + offset,
                        value,
                        in_range,
                    )

    @triton.jit
    def commit_node_pages(Desc, Staging, Active, CBLOCK: tl.constexpr):
        job = tl.program_id(0)
        d = Desc + job * 10
        if tl.load(Active + job):
            rows = tl.load(d + 3)
            target = tl.load(d).to(tl.pointer_type(tl.uint8))
            stride = tl.load(d + 1)
            size = tl.load(d + 2)
            stage = tl.load(d + 8)
            lanes = tl.arange(0, CBLOCK)
            for r in tl.range(0, rows):
                for begin in tl.range(0, size, CBLOCK):
                    offset = begin + lanes
                    value = tl.load(
                        Staging + stage + r * size + offset,
                        offset < size,
                        other=0,
                    )
                    tl.store(
                        target + r * stride + offset,
                        value,
                        offset < size,
                    )

    _NODE_COMMIT_KERNELS = (freeze_node_pages, commit_node_pages)
    return _NODE_COMMIT_KERNELS


def prepare_nodes(plan: NodePlan, records: list[NodeRecord], winners, active, device):
    """Freeze accepted node rows into staging before touching the live cache."""
    winners = globalize_winners(plan, winners)
    freeze_node_pages, _ = _node_commit_kernels()
    descriptors, total, max_rows = build_node_descriptors(records)
    if not descriptors:
        return (
            torch.empty((0, 10), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.uint8, device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
            1,
        )
    desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
    staging = torch.empty(total, dtype=torch.uint8, device=device)
    enabled = torch.empty(len(descriptors), dtype=torch.bool, device=device)
    bn = 1 << max(int(plan.n) - 1, 0).bit_length()
    freeze_node_pages[(len(descriptors),)](
        desc,
        plan.ancestors,
        plan.depths,
        plan.start,
        plan.req_ids,
        winners,
        active,
        staging,
        enabled,
        plan.n,
        # MAX_Q is the Ancestors row stride and the depth-priority base. It is
        # max_depth + 1, not the target query width (which is budget + 1 and
        # can be larger when budget > depth).
        int(plan.ancestors.shape[1]),
        bn,
        _BYTE_BLOCK,
    )
    return desc, staging, enabled, max_rows


def commit_nodes(prepared) -> None:
    """Scatter the frozen staging rows into the live compressed cache pages."""
    _, commit_node_pages = _node_commit_kernels()
    desc, staging, enabled, max_rows = prepared
    if desc.shape[0]:
        commit_node_pages[(len(desc),)](desc, staging, enabled, _BYTE_BLOCK)


# ---------------------------------------------------------------------------
# SWA KV move (accepted full/SWA rows -> contiguous history slots)
# ---------------------------------------------------------------------------






def _build_swa_move_plan(plan: NodePlan, pool) -> SwaMovePlan:
    cached = getattr(plan, "swa_move_plan", None)
    if cached is not None:
        return cached
    mask = plan.source_rows >= 0
    source = plan.source_rows[mask].to(torch.int64)
    dest = plan.dest_slots[mask].to(torch.int64)
    src_swa = pool.translate_loc_from_full_to_swa(source)
    dst_swa = pool.translate_loc_from_full_to_swa(dest)
    valid = (src_swa >= 0) & (dst_swa >= 0)
    cached = SwaMovePlan(src_swa[valid], dst_swa[valid])
    plan.swa_move_plan = cached
    return cached


def move_swa_kv(pool, plan) -> None:
    """Move accepted full/SWA KV rows to contiguous history slots.

    This covers the SWA/ori half for every layer (including compressed layers,
    whose recent window lives in the same SWA buffer). Compressed KV is committed
    separately by the node page kernels.
    """
    move = _build_swa_move_plan(plan, pool)
    if move.source.numel() == 0:
        return
    for layer_id in range(int(pool.layer_num)):
        buffer = pool.get_swa_buffer(layer_id)
        flat = buffer.flatten(0, 1)
        scratch = move.scratch.get(layer_id)
        scratch_shape = (move.source.numel(), *flat.shape[1:])
        if (
            scratch is None
            or scratch.shape != scratch_shape
            or scratch.dtype != flat.dtype
            or scratch.device != flat.device
        ):
            scratch = torch.empty(
                scratch_shape, dtype=flat.dtype, device=flat.device
            )
            move.scratch[layer_id] = scratch
        # Gather before writing so src/dst overlap cannot corrupt a source row.
        torch.index_select(flat, 0, move.source, out=scratch)
        flat.index_copy_(0, move.destination, scratch)
