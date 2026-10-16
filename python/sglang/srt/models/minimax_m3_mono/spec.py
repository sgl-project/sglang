"""MiniMax-M3 mono spec (gfx950): one whole-layer launch for each sparse MoE layer.

The kernels are built for one rank of a TP group whose size is fixed at import
(``config.TP``); ``supported_tp`` reports it and ``Packed`` refuses a runtime TP
that disagrees.

A launch (``sglang.kernels.ops.mono.gfx950.minimax_m3``, from ROCm/aiter PR6047) is K1 fused in front of K4::

    K1 part        (ar, res) -> h, q; K / V / index cache insert; indexer scores
    K4 part        top-k, attention .. FFN, both all-reduces in-kernel -> (ar, h_mid)

with ``(ar, res)`` the (all-reduced MLP output, residual) pair the decoder layer hands its input norm, so a layer
computes the same values either way. A launch reads the previous layer's (ar, h_mid) while it writes its own, so both
alternate between two buffers by layer parity, which is baked into each layer's arguments at pack: these buffers are
never reallocated. The embedding, the leading dense-MLP layers and the final norm run the model's modules.

Per step, inside the graph where there is one, the first routed layer derives the row tables the kernels read from
SGLang's decode buffers (``metadata``): a [tokens, blocks] int32 block table (page-128 block ids from
``req_to_token``) and int32 sequence lengths; positions and the slot mapping are read as the int64 buffers they are.
Sparse K / V are page16 SHUFFLE planes (the AITER sparse attention layout prefill writes too), index-K is NHD. The
caches and the request table are bound at attach, which runs in the first forward.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.mono.gfx950.minimax_m3.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    LAYER_SLOTS,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    MAX_WIDTH,
    N_ROUTED,
    ONE_INDEX_HEAD,
    SPARSE_BLOCK,
    TOP_K,
    TOTAL_KV_HEADS,
    TOTAL_Q_HEADS,
    TP,
)
from sglang.srt.environ import envs
from sglang.srt.layers.mono.guards import MonoRefused
from sglang.srt.layers.mono.spec import (
    Feature,
    LayerMode,
    LayerStateBinding,
    MonoContext,
    MonoSpec,
    MonoStep,
)

logger = logging.getLogger(__name__)

SWIGLU_ALPHA = 1.702  # aiter's swiglu_mul_batch bakes this in
# The fused shared expert's routing weight. SGLang's TopK gives it
# sum(routed) / routed_scaling_factor and the MoE output is scaled back by
# routed_scaling_factor, so with renormalized routed weights it contributes 1.0.
SHARED_WEIGHT = 1.0
FP8_CACHE = (torch.float8_e4m3fn, torch.uint8)


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoRefused(what)


def layers(hf_config) -> dict[int, LayerMode]:
    """The MoE layers (``moe_layer_freq``, else every layer): one whole-layer launch each."""
    t = _text_config(hf_config)
    freq = getattr(t, "moe_layer_freq", None)
    return {
        i: LayerMode.WHOLE
        for i in range(t.num_hidden_layers)
        if freq is None or freq[i] != 0
    }


def refuse_config(cfg, hf_config) -> list[str]:
    t = _text_config(hf_config)
    want = dict(
        hidden_size=HIDDEN,
        num_local_experts=N_ROUTED,
        num_experts_per_tok=TOP_K,
        # The model's own totals, not this rank's share: these are compared
        # against the checkpoint's config, which knows nothing about TP.
        # ``num_key_value_heads`` was ``TP``, which is the model's 4 only at
        # TP4 -- at TP2 it refused the very model it supports, and at TP8 the
        # rank's single replicated head is not a total at all.
        num_attention_heads=TOTAL_Q_HEADS,
        num_key_value_heads=TOTAL_KV_HEADS,
        head_dim=HEAD_DIM,
        scoring_func="sigmoid",
        swiglu_alpha=SWIGLU_ALPHA,
    )
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in want.items()
        if getattr(t, k, None) != v
    ]
    if t.num_hidden_layers > LAYER_SLOTS:
        reasons.append(
            f"{t.num_hidden_layers} layers, the mailbox epochs hold {LAYER_SLOTS}"
        )
    if cfg.page_size != SPARSE_BLOCK:
        reasons.append(
            f"--page-size {cfg.page_size}: the kernels index {SPARSE_BLOCK}-token pages"
        )
    if cfg.kv_cache_dtype != "fp8_e4m3":
        reasons.append(
            f"--kv-cache-dtype {cfg.kv_cache_dtype}: the kernels write fp8_e4m3 K / V"
        )
    if cfg.disable_shared_experts_fusion:
        reasons.append(
            "--disable-shared-experts-fusion: the kernels run the shared expert as "
            "the fused expert of the MXFP4 MoE"
        )
    if cfg.speculative_algorithm is not None and (
        cfg.speculative_eagle_topk != 1
        or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        reasons.append(
            "speculative decoding: the kernels verify one linear draft chain "
            "(--speculative-eagle-topk 1, draft tokens = steps + 1)"
        )
    if envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.get() != 1:
        reasons.append(
            "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1 is required: the kernels select "
            "blocks in every layer"
        )
    if not envs.SGLANG_USE_AITER.get():
        reasons.append(
            "SGLANG_USE_AITER=1 is required (the kernels read aiter's MXFP4 MoE layout)"
        )
    if not (
        envs.SGLANG_M3_USE_AITER_SPARSE_PA.get()
        and envs.SGLANG_M3_USE_AITER_PREFILL_INDEXER.get()
    ):
        reasons.append(
            "SGLANG_M3_USE_AITER_SPARSE_PA and SGLANG_M3_USE_AITER_PREFILL_INDEXER "
            "must stay on: the kernels read the page16 SHUFFLE K / V they select"
        )
    if cfg.enable_hierarchical_cache and (
        cfg.hicache_io_backend != "kernel" or cfg.hicache_mem_layout != "layer_first"
    ):
        reasons.append(
            "HiCache keeps the page16 SHUFFLE K / V only with --hicache-io-backend "
            "kernel and --hicache-mem-layout layer_first"
        )
    return reasons


def refuse_model(model) -> list[str]:
    reasons = []
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        reasons.append(
            f"pipelined: layers {model.start_layer}..{model.end_layer} of "
            f"{len(model.layers)}"
        )
    sparse = _sparse_ids(model.layers)
    if not sparse or sparse != list(range(sparse[0], len(model.layers))):
        reasons.append(f"MoE layers {sparse} are not a trailing run of the stack")
    elif sparse[0] > 0:
        # the first launch takes the previous layer's MLP output with its residual
        # add still pending: the kernel applies the add with its input norm
        update = model.layers[sparse[0] - 1].ffn_boundary.declaration.update
        if not update.is_plain_add:
            reasons.append(
                f"layer {sparse[0] - 1} hands on a {type(update).__name__} residual "
                "update; the kernels apply a plain add"
            )
    return reasons


class Packed:
    """One rank's state the launches read: per-layer weight views, the bound caches,
    scratch, the peer buffer, the (ar, h_mid) buffer pairs and the step's row tables."""

    def __init__(self, model, ctx: MonoContext) -> None:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        _need(ctx.tp_size == TP, f"TP {ctx.tp_size}, kernels are built for {TP}")
        _need(
            par.attn_tp_size == TP and par.moe_tp_size == TP and par.moe_ep_size == 1,
            f"attention TP {par.attn_tp_size} / MoE TP {par.moe_tp_size} / EP "
            f"{par.moe_ep_size}: the kernels shard attention and experts over TP",
        )
        cus = torch.cuda.get_device_properties(ctx.device).multi_processor_count
        _need(
            cus == BLOCKS, f"{cus} CUs: the kernels need all {BLOCKS} CTAs co-resident"
        )
        self.rank = ctx.tp_rank
        sparse_ids = _sparse_ids(model.layers)
        self.first_layer = sparse_ids[0]
        self.modes = {i: LayerMode.WHOLE for i in sparse_ids}
        self._pack_layers(model.layers, sparse_ids, _text_config(ctx.hf_config))
        self._bind_caches(sparse_ids)
        self._alloc_buffers(ctx)
        logger.info(
            "[mono][rank %d] minimax_m3 packed: %d sparse layers (%d..%d), %d "
            "block-table columns, %d KV slots",
            self.rank, len(self.sparse), self.first_layer, len(model.layers) - 1,
            self.bt_width, self.num_slots,
        )  # fmt: skip

    def _pack_layers(self, layers, sparse_ids, config) -> None:
        """Each sparse layer's weight views (``SparseMoeLayer``) and the build
        constants every launch width shares."""
        from sglang.srt.models.minimax_m3_mono.weights import (
            SparseMoeLayer,
            sparse_selection,
        )

        cos_sin_tables: dict = {}
        self.sparse = [
            SparseMoeLayer.from_layer(layers[i], i, ONE_INDEX_HEAD, cos_sin_tables)
            for i in sparse_ids
        ]
        _, init_blocks, local_blocks = sparse_selection(config)
        sm_scale = float(layers[self.first_layer].self_attn.scaling)
        _need(abs(sm_scale - HEAD_DIM**-0.5) < 1e-12, f"softmax scale {sm_scale}")
        self.k4_build = dict(
            npes=TP,
            sm_scale=sm_scale,
            eps=float(config.rms_norm_eps),
            route_scale=float(layers[self.first_layer].mlp.routed_scaling_factor),
            shared_weight=SHARED_WEIGHT,
            swiglu_limit=float(config.swiglu_limit),
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            heads=ONE_INDEX_HEAD,
        )

    def _alloc_buffers(self, ctx: MonoContext) -> None:
        """Scratch, the peer buffer, the step counter, the (ar, h_mid) pairs, each
        layer's fixed K1 arguments and the per-width row tables."""
        from sglang.kernels.ops.mono.gfx950.minimax_m3.layout import (
            SCRATCH_BYTES as K4_SCRATCH_BYTES,
        )
        from sglang.kernels.ops.mono.gfx950.minimax_m3.layout import (
            sym_layout,
        )
        from sglang.kernels.ops.mono.gfx950.minimax_m3.pre_attn import (
            K1_ARGS,
        )
        from sglang.kernels.ops.mono.gfx950.minimax_m3.pre_attn import (
            SCRATCH_BYTES as K1_SCRATCH_BYTES,
        )
        from sglang.srt.layers.mono.runtime.peer_memory import PeerBuffer

        dev = ctx.device
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.peers = PeerBuffer(
            sym_layout(TP)["_bytes"], ctx.cpu_group, self.rank, TP, dev
        )
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        bf16 = torch.bfloat16
        # row k = token k of the step; sparse layer i reads ars[i % 2] (the
        # previous layer's output) and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h_mids = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        self.q = torch.empty(
            MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in),
                "w_qkv": _ptr(lw.w_qkv), "s_qkv": _ptr(lw.s_qkv),
                "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik),
                "cos_sin": _ptr(lw.cos_sin), "iq_out": _ptr(self.iq),
                "index_cache": self.index_ptrs[i],
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )
        # per token count: the step's block table and sequence lengths
        self.tables = {
            n: (
                torch.zeros(n, self.bt_width, dtype=torch.int32, device=dev),
                torch.ones(n, dtype=torch.int32, device=dev),
            )
            for n in range(1, MAX_TOKENS + 1)
        }

    def _bind_caches(self, sparse_ids) -> None:
        """Bind FP8 K / V / index-K, LOCAL_KV_HEADS heads a rank, page-128 allocation.

        Logical views stay NHD. Sparse K / V bytes use page-16 SHUFFLE storage;
        index-K and HiCache transfer rows remain NHD.
        """
        # The caches and the request-to-token table hang off the attention
        # backend of the current forward (TboAttnBackend-style aliases).
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxHybridAttnBackend,
        )
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if isinstance(backend, MiniMaxHybridAttnBackend):
            backend = backend.sparse  # the sparse backend owns the MiniMax pool
        _need(
            backend.index_topk_freq == 1,
            "mono selects blocks per layer; set SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1",
        )
        pool = backend.kv_pool
        _need(
            isinstance(pool, MiniMaxSparseKVPool),
            f"{type(pool).__name__}: not the MiniMax sparse KV pool",
        )
        _need(pool.use_aiter_sparse_pa, "the kernels need page-16 SHUFFLE KV storage")
        main = pool.main_pool
        _need(
            pool.page_size == SPARSE_BLOCK,
            f"page size {pool.page_size}, kernels index {SPARSE_BLOCK}-token pages",
        )
        _need(
            main.kv_cache_layout == "nhd",
            f"KV cache logical layout {main.kv_cache_layout!r}, expected NHD views",
        )
        _need(
            main.head_num == LOCAL_KV_HEADS,
            f"{main.head_num} kv heads a rank, kernels write {LOCAL_KV_HEADS}",
        )
        k0 = pool.get_key_buffer(sparse_ids[0])
        v0 = pool.get_value_buffer(sparse_ids[0])
        i0 = pool.get_index_k_buffer(sparse_ids[0])
        _need(
            k0.dtype in FP8_CACHE and v0.dtype in FP8_CACHE,
            f"KV cache {k0.dtype} / {v0.dtype}, kernels write e4m3",
        )
        _need(i0.dtype in FP8_CACHE, f"index cache {i0.dtype}, kernels write e4m3")
        _need(
            k0.shape[0] == i0.shape[0]
            and k0.shape[-1] == HEAD_DIM
            and i0.shape[-1] == HEAD_DIM,
            f"KV cache {tuple(k0.shape)} / index cache {tuple(i0.shape)}",
        )
        self.num_slots = k0.shape[0]
        self.k_ptrs, self.v_ptrs, self.index_ptrs = [], [], []
        self.state_bindings = {}
        for lid in sparse_ids:
            k, v, ik = (
                pool.get_key_buffer(lid),
                pool.get_value_buffer(lid),
                pool.get_index_k_buffer(lid),
            )
            _need(
                k.is_contiguous() and v.is_contiguous() and ik.is_contiguous()
                and k.shape == k0.shape and v.shape == v0.shape and ik.shape == i0.shape,
                f"layer {lid}: cache views differ from layer {sparse_ids[0]}'s",
            )  # fmt: skip
            self.k_ptrs.append(_ptr(k))
            self.v_ptrs.append(_ptr(v))
            self.index_ptrs.append(_ptr(ik))
            # cached pointers must honor HiCache's per-layer H2D loads, as the
            # pool's own get_key_buffer does on the module path
            self.state_bindings[lid] = LayerStateBinding(
                lid, (k, v, ik), pool._wait_for_layer
            )
        req_to_token = backend.req_to_token
        _need(
            req_to_token.dtype == torch.int32 and req_to_token.dim() == 2,
            f"req_to_token {req_to_token.dtype} {tuple(req_to_token.shape)}",
        )
        self.req_to_token = req_to_token
        # EAGLE reserves an extra allocator page for draft lookahead. It is
        # not an increase to the configured model context or score capacity.
        self.max_context_len = int(backend.max_context_len)
        self.bt_width = (self.max_context_len + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        _need(
            0 < self.bt_width <= MAX_INDEX_BLOCKS,
            f"{self.bt_width} blocks a request, scores hold {MAX_INDEX_BLOCKS}",
        )
        self.cols = torch.arange(
            0, self.bt_width * SPARSE_BLOCK, SPARSE_BLOCK, dtype=torch.int64,
            device=req_to_token.device,
        )  # fmt: skip

    def prepare_step(self, forward_batch, step: MonoStep):
        """This step's causal rows from the replay-fresh decode / verify buffers."""
        from sglang.srt.models.minimax_m3_mono.metadata import prepare_token_rows

        bt, seq_lens = self.tables[step.rows]
        return prepare_token_rows(
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.cols,
            bt,
            seq_lens,
            forward_batch.out_cache_loc[: step.rows],
            step.query_len,
            step.is_verify,
            self.max_context_len,
        )


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _sparse_ids(layers) -> list[int]:
    return [i for i, layer in enumerate(layers) if layer.is_layer_sparse]


class M3Kernel:
    """The ``rows``-token launch every sparse layer shares (its weights come from ``packed``)."""

    def __init__(self, packed: Packed, rows: int) -> None:
        from sglang.kernels.ops.mono.gfx950.minimax_m3.post_attn import (
            build_post_attn_kernel,
        )

        self.packed = packed
        self.rows = rows
        # FlyDSL compiles on the first launch: the eager warm-up forward that
        # reaches this width before its graph capture
        self.launcher = build_post_attn_kernel(tokens=rows, **packed.k4_build)

    def launch(
        self,
        step: MonoStep,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch,
        memo: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse layer ``layer_id`` in a decoder layer's own terms: ``(hidden_states,
        residual)`` in, ``(hidden_states, residual)`` out. ``ar`` has to be in
        ``ars[i % 2]``, which it already is when the preceding launch left it there;
        the copy is for the first sparse layer, which the modules hand it. ``memo``:
        the step's row tables, derived once by its first routed layer."""
        p = self.packed
        i = layer_id - p.first_layer
        n = step.rows
        rows = memo.get("rows")
        if rows is None:
            rows = memo["rows"] = p.prepare_step(forward_batch, step)
        block_table, seq_lens = rows
        ar = p.ars[i % 2]
        if hidden_states.data_ptr() != ar.data_ptr():
            ar[:n].copy_(hidden_states.view(n, HIDDEN))
        lw = p.sparse[i]
        self.launcher(
            _ptr(p.h), _ptr(p.q), _ptr(block_table), _ptr(seq_lens),
            p.k_ptrs[i], p.v_ptrs[i], _ptr(lw.k_scale), _ptr(lw.v_scale),
            _ptr(lw.w_o), _ptr(lw.s_o), _ptr(lw.g_post), _ptr(lw.gate),
            _ptr(lw.bias), _ptr(lw.w13), _ptr(lw.s13), _ptr(lw.w2), _ptr(lw.s2),
            _ptr(p.h_mids[(i + 1) % 2]), _ptr(p.ars[(i + 1) % 2]),
            _ptr(p.scratch4), p.peers.local, _ptr(p.peers.addresses),
            _ptr(p.step), p.rank, lw.layer_id,
            block_table.shape[1], step.query_len,
            _ptr(p.k1_args[i]), _ptr(positions),
            _ptr(forward_batch.out_cache_loc), _ptr(residual.view(n, HIDDEN)),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        if i == len(p.sparse) - 1:
            # Opens the next step's epoch, which is what keeps a layer's mailbox
            # tags distinct from the same layer's tags one step ago. Enqueued
            # after the last launch of this step, so no kernel can see it early.
            p.step.add_(1)
        return p.ars[(i + 1) % 2][:n], p.h_mids[(i + 1) % 2][:n]

    def capture_input(self) -> torch.Tensor:
        """K1's BF16 residual after the input add, before the norm: the hidden state
        an EAGLE3 draft captures at this layer's attention boundary (a reused buffer:
        the collector copies it)."""
        return self.packed.h[: self.rows]

    def health(self) -> Optional[torch.Tensor]:
        return None  # no bounded waits


def pack(model, ctx: MonoContext) -> Packed:
    return Packed(model, ctx)


def build(ctx: MonoContext, packed: Packed, rows: int) -> M3Kernel:
    return M3Kernel(packed, rows)


def eligible(ctx: MonoContext, step: MonoStep) -> Optional[str]:
    args = ctx.server_args
    if not step.is_verify:
        # a speculative target verifies; a plain decode step there is not served
        return None if args.speculative_algorithm is None else "decode under spec"
    if step.query_len != args.speculative_num_draft_tokens:
        return "verify rows a request"
    return None


def state_bindings(ctx: MonoContext, packed: Packed) -> dict[int, LayerStateBinding]:
    return packed.state_bindings


MONO_SPEC = MonoSpec(
    name="minimax_m3",
    architectures=(
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ),
    layers=layers,
    widths=tuple(range(1, MAX_WIDTH + 1)),
    supports=frozenset(
        {Feature.SPEC_VERIFY, Feature.HIERARCHICAL_CACHE, Feature.EAGER}
    ),
    supported_tp=(TP,),
    refuse_config=refuse_config,
    refuse_model=refuse_model,
    pack=pack,
    eligible=eligible,
    state_bindings=state_bindings,
    build=build,
)
