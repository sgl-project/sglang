"""Double Sparsity algorithm — K-channel sparse decode for long-context LLMs.

Production decode runs through ``try_native_sparse_decode`` (the native
Triton pipeline in ``triton_ops/double_sparsity_native_decode.py``); the
``retrieve_topk`` / FA3-adaptor path below is preserved as a fallback for
``bs > scratch_max_bs`` callers but is not on the production hot path at
the benchmark operating point.

See ``benchmark/double_sparsity/DESIGN.md`` for the design rationale, the
gate definitions, and the production recipe.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
    ds_native_sparse_decode,
)
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityCalibration,
    DoubleSparsityRuntimeConfig,
    channel_indices_for_runtime,
    gqa_reduction_id,
    parse_calibration_file,
    torch_dtype_for_klabel,
    validate_against_model,
)
from sglang.srt.mem_cache.sparsity.triton_ops.k_label_kernels import (
    ds_compute_k_label_torch_ref,
    ds_compute_k_label_write,
)
from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
    _compute_q_label,
    ds_select_tokens_torch_ref,
    ds_select_tokens_triton,
)

logger = logging.getLogger(__name__)


class DoubleSparsityAlgorithm(BaseSparseAlgorithm):
    """K-channel Double Sparsity for decode-heavy long-context inference.

    Properties:
      - K-channels only (V uses the full head_dim at attention time).
      - Per-KV-head scoring + GQA reduction so all Q heads in a GQA group
        share one selected set per request.
      - ``page_size = 1`` (one KV slot per token).
      - Configurable GQA reduction (``max_abs`` / ``mean`` / ``soq``),
        default ``max_abs``.
      - TP-aware: each rank carries a slice of the global-indexed
        calibration JSON.

    Production decode uses the native sparse-decode pipeline
    (``try_native_sparse_decode``); the ``retrieve_topk`` path remains
    available as a fallback for ``bs > scratch_max_bs``.
    """

    def __init__(
        self,
        config,
        device: torch.device,
        *,
        runtime_config: DoubleSparsityRuntimeConfig,
        calibration: DoubleSparsityCalibration,
        tp_size: int = 1,
        tp_rank: int = 0,
        num_kv_heads_local: Optional[int] = None,
        num_q_heads_local: Optional[int] = None,
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(config, device, **kwargs)
        runtime_config.validate()

        if calibration.heavy_channels != runtime_config.heavy_channels:
            raise ValueError(
                f"runtime heavy_channels ({runtime_config.heavy_channels}) does not match "
                f"calibration heavy_channels ({calibration.heavy_channels}); regenerate "
                f"calibration or pass --double-sparsity-heavy-channels="
                f"{calibration.heavy_channels}."
            )

        self.runtime_config = runtime_config
        self.calibration = calibration
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.num_kv_heads_local = (
            num_kv_heads_local
            if num_kv_heads_local is not None
            else calibration.num_kv_heads_global // tp_size
        )
        self.num_q_heads_local = (
            num_q_heads_local
            if num_q_heads_local is not None
            else calibration.num_heads // tp_size
        )
        self.head_dim = head_dim if head_dim is not None else calibration.head_dim
        self.klabel_dtype = torch_dtype_for_klabel(runtime_config.klabel_dtype)

        # Per-layer int32 channel indices, shape [num_kv_heads_local, S].
        self.channel_indices: Dict[int, torch.Tensor] = channel_indices_for_runtime(
            calibration,
            tp_size=tp_size,
            tp_rank=tp_rank,
            device=device,
        )

        # Per-layer K_label side cache. Allocated in
        # initialize_representation_pool once the KV pool is available.
        self.k_label: Dict[int, torch.Tensor] = {}
        self.start_layer: int = 0
        self.end_layer: int = 0

        # Legacy selection scratch — used by retrieve_topk + FA3 adaptor.
        # Allocated once in _allocate_selection_scratch and narrowed
        # per-call to bs.
        self._block_topk_logical: Optional[torch.Tensor] = None
        self._block_topk_scores: Optional[torch.Tensor] = None
        self._merged_logical: Optional[torch.Tensor] = None
        self._merged_scores: Optional[torch.Tensor] = None
        self._selected_logical: Optional[torch.Tensor] = None
        self._valid_lengths: Optional[torch.Tensor] = None

        # Native sparse-decode scratch. Production decode reads/writes
        # these tensors via .narrow(0, 0, bs) on each call; the captured
        # CUDA graph references stable device pointers.
        self._native_att_out: Optional[torch.Tensor] = None
        self._native_selected_physical: Optional[torch.Tensor] = None
        self._native_mid_out: Optional[torch.Tensor] = None
        self._native_mid_o_logexpsum: Optional[torch.Tensor] = None
        self._native_output: Optional[torch.Tensor] = None
        self._native_req_to_token_indexed: Optional[torch.Tensor] = None
        # BLOCK_SEQ for the split-K sparse attention kernel; sized to
        # match the per-decode-step total_selected.
        self._native_attn_block_seq: int = 128

        logger.info(
            "DoubleSparsity init: layers=%d S=%d kv_heads_local=%d q_heads_local=%d "
            "tp_size=%d tp_rank=%d klabel_dtype=%s gqa_reduction=%s",
            calibration.num_layers,
            runtime_config.heavy_channels,
            self.num_kv_heads_local,
            self.num_q_heads_local,
            tp_size,
            tp_rank,
            runtime_config.klabel_dtype,
            runtime_config.gqa_reduction,
        )

    @classmethod
    def from_server_config(
        cls,
        config,
        device: torch.device,
        *,
        calibration_path: str,
        runtime_config: DoubleSparsityRuntimeConfig,
        head_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads_global: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        **kwargs,
    ) -> "DoubleSparsityAlgorithm":
        """Convenience constructor that loads + validates calibration vs the model."""
        calibration = parse_calibration_file(calibration_path)
        validate_against_model(
            calibration,
            head_dim=head_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads_global=num_kv_heads_global,
        )
        return cls(
            config,
            device,
            runtime_config=runtime_config,
            calibration=calibration,
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_kv_heads_local=num_kv_heads_global // tp_size,
            num_q_heads_local=num_heads // tp_size,
            head_dim=head_dim,
            **kwargs,
        )

    def effective_sparse_mask(
        self,
        forward_batch,
        req_pool_indices: torch.Tensor,
        default_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gate on current `seq_lens`, not admission-time `prompt_lens`.

        The base coordinator computes `default_mask` from
        `prompt_lens >= min_sparse_prompt_len`. For DS that's wrong:
        a request admitted with prompt_len=100 but generating to
        seq_len=5000 would never become sparse, even though by then
        decode is squarely above `min_seq_len`. We recompute from
        current seq_lens so late-crossing rows flip to sparse, and
        the coordinator threads this same mask into both retrieve_topk
        and the FA3 adaptor.
        """
        seq_lens = forward_batch.seq_lens.to(default_mask.device)
        return seq_lens >= self.runtime_config.min_seq_len

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ) -> None:
        """Allocate per-layer K_label side cache.

        Shape per layer: `[num_tokens_in_pool, num_kv_heads_local, S]`. Memory is
        ~12% on top of the KV pool when `S/D ≈ 32/128` and is owned by this
        algorithm (we deliberately do not touch `MHATokenToKVPool`).
        """
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.states = states
        self.start_layer = start_layer
        self.end_layer = end_layer

        num_tokens_in_pool = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
        S = self.runtime_config.heavy_channels
        for layer_id in range(start_layer, end_layer):
            self.k_label[layer_id] = torch.zeros(
                (num_tokens_in_pool, self.num_kv_heads_local, S),
                dtype=self.klabel_dtype,
                device=self.device,
            )

        logger.info(
            "DoubleSparsity K_label allocated: layers=%d num_tokens_in_pool=%d "
            "kv_heads_local=%d S=%d dtype=%s mem_per_layer=%.2f MiB",
            end_layer - start_layer,
            num_tokens_in_pool,
            self.num_kv_heads_local,
            S,
            self.klabel_dtype,
            (
                num_tokens_in_pool
                * self.num_kv_heads_local
                * S
                * self.k_label[start_layer].element_size()
            )
            / (1024 * 1024),
        )

        self._allocate_selection_scratch()
        self._allocate_native_scratch()

    def _allocate_native_scratch(self) -> None:
        """Pre-allocate scratch for the native sparse-decode path.

        Sized for the static worst case: `scratch_max_bs`, `H_kv_local`,
        `H_q_local`, `head_dim`, `max_ctx`, `total_selected`. Buffers are
        threaded into `ds_native_sparse_decode` via `.narrow(0, 0, bs)`
        on each call; the kernel writes in-place.

        At 70B/TP=8/128K/concurrency=16: ~9 MiB total — dominated by
        `att_out_approx[H_kv=1, bs=16, max_ctx=131072]` (8 MiB).
        """
        rt = self.runtime_config
        max_ctx = self.req_to_token_pool.req_to_token.shape[1]
        bs = rt.scratch_max_bs
        h_kv = self.num_kv_heads_local
        h_q = self.num_q_heads_local
        d = self.head_dim
        total_selected = rt.token_budget + rt.sink_tokens + rt.recent_tokens
        attn_block_seq = self._native_attn_block_seq
        max_num_blocks = (total_selected + attn_block_seq - 1) // attn_block_seq

        device = self.device
        # Layout [bs, H_kv, max_ctx] so torch.topk yields [bs, H_kv, top_k]
        # directly (no transpose round-trip into _build_selected_physical).
        self._native_att_out = torch.full(
            (bs, h_kv, max_ctx),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        self._native_selected_physical = torch.zeros(
            (bs, h_kv, total_selected), dtype=torch.int32, device=device
        )
        self._native_mid_out = torch.zeros(
            (bs, h_q, max_num_blocks, d), dtype=torch.float32, device=device
        )
        self._native_mid_o_logexpsum = torch.full(
            (bs, h_q, max_num_blocks),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        # Output tensor. Allocated in bf16 (the Llama family's compute dtype
        # and the K_label dtype default). If a future caller passes fp16 q,
        # `try_native_sparse_decode` reallocates — but pre-warming here means
        # the *normal* case sees zero allocation under capture.
        self._native_output = torch.zeros(
            (bs, h_q, d), dtype=self.klabel_dtype, device=device
        )

        # req_to_token indexed per-batch. Re-gathered on first call per step
        # in try_native_sparse_decode (cached only if shape matches).
        self._native_req_to_token_indexed = torch.zeros(
            (bs, max_ctx), dtype=torch.int32, device=device
        )

        bytes_total = (
            self._native_att_out.numel() * 4
            + self._native_selected_physical.numel() * 4
            + self._native_mid_out.numel() * 4
            + self._native_mid_o_logexpsum.numel() * 4
            + self._native_req_to_token_indexed.numel() * 4
        )
        logger.info(
            "DoubleSparsity native scratch allocated: bs=%d h_kv=%d h_q=%d d=%d "
            "max_ctx=%d total_selected=%d max_blocks=%d total=%.2f MiB",
            bs,
            h_kv,
            h_q,
            d,
            max_ctx,
            total_selected,
            max_num_blocks,
            bytes_total / (1024 * 1024),
        )

        # Construct + warm up the selector backend now (outside the
        # CUDA-graph capture region that ModelRunner will start later).
        # FlashInfer / SGL kernels JIT-compile on first call; doing that
        # under capture is what triggered "Triton Error [CUDA]: illegal
        # memory access" in early bench runs. The first .select() call
        # below pulls those compiles + lazy aux-tensor allocations to
        # init time, where they're allowed.
        if rt.selector_backend == "torch":
            self._selector = None
        else:
            from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
                make_selector,
            )

            # max_top_k / max_ctx are only consumed by ftka_raft_topk;
            # other backends ignore them. Sizing here mirrors the
            # token_budget / req_to_token width that the algorithm
            # already enforces.
            self._selector = make_selector(
                rt.selector_backend,
                max_bs=bs,
                h_kv=h_kv,
                device=device,
                max_top_k=rt.token_budget,
                max_ctx=self._native_req_to_token_indexed.shape[1],
            )
            self._warmup_selector(bs, h_kv)

    def _warmup_selector(self, max_bs: int, h_kv: int) -> None:
        """Issue selector calls at every bs the CUDA graph might capture so
        any JIT-compile / lazy-alloc happens outside the capture region.

        FlashInfer's ``top_k_page_table_transform`` JIT-compiles a fresh
        Triton kernel for each (num_rows, k) shape it sees. The default
        SGLang capture sweeps bs in {1, 2, 4, 8, 12, 16, 24, 32, ...},
        so warming only the worst case left smaller bs's first call
        landing inside capture — which surfaces as a Triton
        ``load_binary`` failure with ``illegal memory access``."""
        if self._selector is None:
            return
        rt = self.runtime_config
        att_out = self._native_att_out
        r2t = self._native_req_to_token_indexed

        # Seed att_out with a finite region so topk has valid winners
        # (sink/recent/oob masked out as in the production score kernel).
        att_out.fill_(float("-inf"))
        att_out[..., rt.sink_tokens : rt.token_budget + rt.sink_tokens].zero_()
        r2t.zero_()
        device = att_out.device
        seq_lens_full = torch.full(
            (max_bs,),
            max(
                rt.min_seq_len, rt.token_budget + rt.sink_tokens + rt.recent_tokens + 1
            ),
            dtype=torch.int64,
            device=device,
        )

        # bs sweep mirrors SGLang's `_get_cuda_graph_bs` default ladder.
        # Includes 1 and every power-of-two up to max_bs (cap), plus a
        # couple of in-between values that the runner uses.
        bs_ladder = sorted({1, 2, 4, 8, 12, 16, 24, 32}) + [max_bs]
        bs_ladder = sorted({b for b in bs_ladder if 1 <= b <= max_bs})

        for bs in bs_ladder:
            # Failures propagate: warmup is the only place this selector
            # is invoked outside the CUDA-graph capture region, so a
            # silent log-and-continue means the *real* first call lands
            # inside capture and crashes the server mid-init. Refusing
            # to start is the honest outcome.
            self._selector.select(
                att_out_approx=att_out[:bs, :h_kv],
                req_to_token_indexed=r2t[:bs],
                seq_lens=seq_lens_full[:bs],
                top_k=rt.token_budget,
                sink_tokens=rt.sink_tokens,
                recent_tokens=rt.recent_tokens,
                out=self._native_selected_physical[:bs, :h_kv],
            )
        torch.cuda.synchronize()
        logger.info(
            "DoubleSparsity selector warmup OK: backend=%s bs_sweep=%s h_kv=%d top_k=%d",
            rt.selector_backend,
            bs_ladder,
            h_kv,
            rt.token_budget,
        )

    def forward_begin(self, forward_batch) -> None:
        """Per-decode-step gather of ``req_to_token[req_pool_indices]``.

        Writes into ``self._native_req_to_token_indexed`` so all layers
        in the step read from the same buffer (saves 79 redundant
        ``torch.index_select`` launches per step at 80 layers).

        Capture-safe: the kernel reads a device tensor pointer that is
        captured once. At replay-time the surrounding pre-replay write
        from ``ModelRunner.forward`` (and at capture-time the write
        from ``CUDAGraphRunner.capture_one_batch_size``) refreshes the
        contents through the stable pointer. ``try_native_sparse_decode``
        no longer issues its own ``index_select`` — it just
        ``.narrow(0, 0, bs)``-views this buffer.

        No-op when:
          * native scratch is not allocated (test paths or DS legacy-
            only build),
          * not a decode/idle batch (extend stays on the legacy /
            dense path),
          * ``bs > scratch_max_bs`` (caller falls through to the FA3
            legacy adaptor, which gathers ``req_to_token`` itself).
        """
        if self._native_req_to_token_indexed is None:
            return
        if not forward_batch.forward_mode.is_decode_or_idle():
            return
        req_pool_indices = forward_batch.req_pool_indices
        bs = req_pool_indices.shape[0]
        if bs > self._native_req_to_token_indexed.shape[0]:
            return
        req_to_token = self.req_to_token_pool.req_to_token
        r2t_indexed = self._native_req_to_token_indexed.narrow(0, 0, bs)
        torch.index_select(req_to_token, 0, req_pool_indices, out=r2t_indexed)

    def try_native_sparse_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        *,
        save_kv_cache: bool = True,
    ) -> Optional[torch.Tensor]:
        """Native sparse-decode forward, or None to fall back to the FA3 path.

        Returns `None` (signaling fallback) when:
          * scratch is not allocated yet (test/CPU paths);
          * any row has `seq_len < total_selected` (sparse-decode kernel
            assumes a full top-k history exists);
          * bs > scratch_max_bs.

        Returns a `[bs, H_q*D]` output tensor when the native path runs.
        Writes K/V into the KV pool internally (matching FA3 backend
        behavior).
        """
        if self._native_att_out is None:
            return None

        rt = self.runtime_config
        seq_lens = forward_batch.seq_lens
        if not seq_lens.is_cuda:
            return None

        # Eligibility gate is intentionally Python-static — NO host sync on
        # `seq_lens`. The capture path traces the kernel launches we make
        # here, and any host-sync-based Python branch during capture would
        # freeze the captured path to that branch's choice forever (the
        # captured graph cannot re-evaluate at replay time).
        #
        # We trust the caller (the bench, or production server admission)
        # to only invoke DS-on when seq_lens are >= min_seq_len for every
        # active row. Outside that contract, the score kernel masks the
        # out-of-history tail to -inf and torch.topk picks junk indices
        # — output is computed against garbage K/V (no crash). A
        # capture-safe per-request fallback is post-v0 work.
        bs = q.shape[0]
        if bs > self._native_att_out.shape[0]:  # [bs, H_kv, max_ctx] layout
            return None

        # Reshape q: callers hand us [bs, H_q*D] flat.
        if q.dim() == 2:
            q_3d = q.view(bs, self.num_q_heads_local, self.head_dim)
        else:
            q_3d = q

        # Write K/V to the KV pool now (FA3 backend would have done this
        # inside its `forward_decode`; we're bypassing it).
        if save_kv_cache and k is not None and v is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # Q_label: gather + GQA-reduce per-(bs, kv_head). Small alloc per
        # call (~bs*H_kv*S floats); preallocation is a v0.1 optimization.
        from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
            gqa_reduction_id,
        )

        q_label = _compute_q_label(
            q_3d,
            self.channel_indices[layer.layer_id],
            num_kv_heads=self.num_kv_heads_local,
            gqa_reduction_id=gqa_reduction_id(rt.gqa_reduction),
        )  # [bs, h_kv, S]

        # Read the pre-indexed req_to_token table written once per step
        # by `forward_begin`. Callers that skip `forward_begin` (direct
        # unit-test invocation, etc.) must populate the scratch
        # themselves before calling this.
        r2t_indexed = self._native_req_to_token_indexed.narrow(0, 0, bs)

        # Output dtype must match the preallocated `_native_output`
        # (klabel_dtype, bf16 by default). Fail-loud rather than
        # reallocate — under CUDA-graph capture a lazy reallocation
        # would land inside the captured region and either silently
        # break replay or trip `cudaErrorStreamCaptureInvalidated`.
        # Callers configure the dtype via `--double-sparsity-klabel-dtype`.
        if self._native_output.dtype != q_3d.dtype:
            raise RuntimeError(
                f"q dtype {q_3d.dtype} != preallocated _native_output dtype "
                f"{self._native_output.dtype} ({self.runtime_config.klabel_dtype}). "
                f"DS configuration must match the model's q dtype."
            )

        # Slice scratch to bs. Layout for att_out is [bs, H_kv, max_ctx]
        # (bs is dim 0); the others have bs at dim 0 too. The score
        # kernel covers all of [0, max_ctx) via `num_blocks =
        # cdiv(max_ctx, block_t)` programs, so no per-step reset is
        # needed.
        output = self._native_output.narrow(0, 0, bs)
        att_out = self._native_att_out.narrow(0, 0, bs)
        sel_phys = self._native_selected_physical.narrow(0, 0, bs)
        mid_out = self._native_mid_out.narrow(0, 0, bs)
        mid_log = self._native_mid_o_logexpsum.narrow(0, 0, bs)

        # Selector is constructed + warmed up in `_allocate_native_scratch`
        # so JIT / lazy-alloc happens before CUDA-graph capture starts.
        ds_native_sparse_decode(
            q=q_3d,
            k_buffer=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_buffer=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_label_layer=self.k_label[layer.layer_id],
            q_label=q_label,
            req_to_token_indexed=r2t_indexed,
            seq_lens=seq_lens,
            top_k=rt.token_budget,
            sink_tokens=rt.sink_tokens,
            recent_tokens=rt.recent_tokens,
            sm_scale=layer.scaling,
            att_out_approx=att_out,
            selected_physical=sel_phys,
            mid_out=mid_out,
            mid_o_logexpsum=mid_log,
            output=output,
            attn_block_seq=self._native_attn_block_seq,
            selector=self._selector,
        )

        # Flatten to the 2D shape downstream code expects.
        return output.view(bs, self.num_q_heads_local * self.head_dim)

    def _allocate_selection_scratch(self) -> None:
        """Pre-allocate selection-pipeline scratch (stage-1/2 and union outputs).

        Sized to the static worst case for the captured graph:
          * max_ctx       = req_to_token.shape[1]
          * num_blocks    = ceil(max_ctx / block_t)
          * effective_bud = min(token_budget, num_blocks * k_block)
          * scratch_bs    = runtime_config.scratch_max_bs

        Total size is bounded (<1 MiB at 70B/TP=8/128K). Buffers are
        threaded through ``ds_select_tokens_triton`` via
        ``.narrow(0, 0, bs)`` row-slices on each call; the helpers write
        in-place. Used by the legacy retrieve_topk + FA3-adaptor fallback
        only; the native sparse-decode path has its own scratch.
        """
        rt = self.runtime_config
        max_ctx = self.req_to_token_pool.req_to_token.shape[1]
        num_blocks = (max_ctx + rt.block_t - 1) // rt.block_t
        effective_budget = min(rt.token_budget, num_blocks * rt.k_block)
        bs = rt.scratch_max_bs
        h_kv = self.num_kv_heads_local
        device = self.device
        NEG_INF = float("-inf")

        self._block_topk_logical = torch.full(
            (bs, h_kv, num_blocks, rt.k_block),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._block_topk_scores = torch.full(
            (bs, h_kv, num_blocks, rt.k_block),
            NEG_INF,
            dtype=torch.float32,
            device=device,
        )
        self._merged_logical = torch.full(
            (bs, h_kv, effective_budget),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._merged_scores = torch.full(
            (bs, h_kv, effective_budget),
            NEG_INF,
            dtype=torch.float32,
            device=device,
        )
        self._selected_logical = torch.full(
            (bs, rt.max_selected_per_request),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._valid_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

        bytes_total = (
            self._block_topk_logical.numel() * 4
            + self._block_topk_scores.numel() * 4
            + self._merged_logical.numel() * 4
            + self._merged_scores.numel() * 4
            + self._selected_logical.numel() * 4
            + self._valid_lengths.numel() * 4
        )
        logger.info(
            "DoubleSparsity selection scratch allocated: bs=%d h_kv=%d "
            "num_blocks=%d k_block=%d effective_budget=%d "
            "max_selected=%d total=%.2f KiB",
            bs,
            h_kv,
            num_blocks,
            rt.k_block,
            effective_budget,
            rt.max_selected_per_request,
            bytes_total / 1024,
        )

    def _write_k_label(
        self,
        layer_id: int,
        k: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        """Single entry point for K_label writes (extend or decode).

        Uses the Triton kernel on CUDA; falls back to the torch reference on
        CPU. Both paths are byte-equivalent for the test fixtures.
        """
        if k.numel() == 0:
            return
        chan = self.channel_indices[layer_id]
        kl = self.k_label[layer_id]
        if k.is_cuda:
            ds_compute_k_label_write(k, chan, out_cache_loc, kl)
        else:
            ds_compute_k_label_torch_ref(k, chan, out_cache_loc, kl)

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Write K_label for new tokens during prefill (extend phase).

        SGLang's coordinator funnels both prefill and decode through
        `attention_end -> construct_representations + update_representations`,
        but only one of the two should fire per call. We dispatch on
        `forward_mode` and use `forward_batch.out_cache_loc` (the same source
        the dense backend uses) so K_label writes target exactly the physical
        token ids that just received K writes.
        """
        if forward_batch.forward_mode.is_extend():
            self._write_k_label_for_new_tokens(layer_id, k_buffer, forward_batch)

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Append K_label for the freshly decoded tokens (decode phase)."""
        if forward_batch.forward_mode.is_decode_or_idle():
            self._write_k_label_for_new_tokens(layer_id, k_buffer, forward_batch)

    def _write_k_label_for_new_tokens(
        self,
        layer_id: int,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Gather K rows written this step from the KV pool and update K_label.

        Skipped when `save_kv_cache=False` — otherwise the side cache desyncs
        from the KV pool. Used by both extend and decode (the only thing that
        differs between phases is which `forward_mode` predicate gates the call).

        In the piecewise-extend path (`unified_attention_with_output`),
        `forward_batch.out_cache_loc` is full-padded by the time
        `attention_end` runs (the function restores it before returning).
        We narrow to `num_token_non_padded_cpu`, which is set on every
        ForwardBatch and matches the actual token count.
        """
        if not getattr(forward_batch, "save_kv_cache", True):
            return
        out_loc = forward_batch.out_cache_loc
        real = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if real is not None and out_loc.numel() > real:
            out_loc = out_loc[:real]
        if out_loc.numel() == 0:
            return
        k_new = k_buffer[out_loc]
        self._write_k_label(layer_id, k_new, out_loc)

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy selection path: select per-request logical token positions
        for the FA3 backend to attend to.

        Returns ``(selected_logical[bs, max_selected], valid_lengths[bs])``.
        ``DSFlashAttentionAdaptor`` (``backend/ds_flash_attention_adaptor.py``)
        maps logical → physical via ``req_to_token``, preserves logical
        order (no physical sort), and writes the FA3 page-table.

        CUDA path: two-stage block-topk Triton + score-aware torch union
        (``ds_select_tokens_triton`` dispatches to ``select_triton.py``).
        CPU path: per-request torch reference (oracle / parity check only,
        never on the production path).

        Production decode flows through ``try_native_sparse_decode``
        instead; this is the fallback for ``bs > scratch_max_bs``.
        """
        forward_batch = kwargs.get("forward_batch")
        if forward_batch is None:
            raise ValueError(
                "DoubleSparsity retrieve_topk requires forward_batch kwarg"
            )
        seq_lens = forward_batch.seq_lens.to(queries.device)
        # Attention layers commonly hand us `q` as `[bs, H_q*D]` (post-projection
        # flat view). Selection kernels need `[bs, H_q, D]`. Reshape here using
        # the algorithm-known geometry.
        if queries.dim() == 2:
            queries = queries.view(
                queries.shape[0], self.num_q_heads_local, self.head_dim
            )

        if queries.is_cuda:
            bs = queries.shape[0]
            # Narrow the preallocated scratch to the current batch size.
            # All helpers write in-place into these views; the returned
            # tensors share storage with the algorithm's permanent buffers
            # so the production decode path performs zero output-tensor
            # allocation per step.
            if self._selected_logical is None:
                # initialize_representation_pool was never called — only
                # happens in tests that exercise retrieve_topk directly
                # without setting up the pool. Fall back to the unthreaded
                # path so existing tests stay green.
                return ds_select_tokens_triton(
                    queries=queries,
                    channel_idx=self.channel_indices[layer_id],
                    k_label_layer=self.k_label[layer_id],
                    req_to_token=self.req_to_token_pool.req_to_token,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    num_kv_heads=self.num_kv_heads_local,
                    token_budget=self.runtime_config.token_budget,
                    recent_tokens=self.runtime_config.recent_tokens,
                    sink_tokens=self.runtime_config.sink_tokens,
                    min_seq_len=self.runtime_config.min_seq_len,
                    max_selected=self.runtime_config.max_selected_per_request,
                    gqa_reduction_id=gqa_reduction_id(
                        self.runtime_config.gqa_reduction
                    ),
                    block_t=self.runtime_config.block_t,
                    k_block=self.runtime_config.k_block,
                )
            if bs > self._selected_logical.shape[0]:
                raise RuntimeError(
                    f"DoubleSparsity decode batch size {bs} exceeds "
                    f"scratch_max_bs={self._selected_logical.shape[0]}; "
                    f"raise --max-running-requests or --cuda-graph-max-bs."
                )
            return ds_select_tokens_triton(
                queries=queries,
                channel_idx=self.channel_indices[layer_id],
                k_label_layer=self.k_label[layer_id],
                req_to_token=self.req_to_token_pool.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                num_kv_heads=self.num_kv_heads_local,
                token_budget=self.runtime_config.token_budget,
                recent_tokens=self.runtime_config.recent_tokens,
                sink_tokens=self.runtime_config.sink_tokens,
                min_seq_len=self.runtime_config.min_seq_len,
                max_selected=self.runtime_config.max_selected_per_request,
                gqa_reduction_id=gqa_reduction_id(self.runtime_config.gqa_reduction),
                block_t=self.runtime_config.block_t,
                k_block=self.runtime_config.k_block,
                block_topk_logical=self._block_topk_logical.narrow(0, 0, bs),
                block_topk_scores=self._block_topk_scores.narrow(0, 0, bs),
                merged_logical=self._merged_logical.narrow(0, 0, bs),
                merged_scores=self._merged_scores.narrow(0, 0, bs),
                selected_logical=self._selected_logical.narrow(0, 0, bs),
                valid_lengths=self._valid_lengths.narrow(0, 0, bs),
            )

        return ds_select_tokens_torch_ref(
            queries=queries,
            channel_idx=self.channel_indices[layer_id],
            k_label_layer=self.k_label[layer_id],
            req_to_token=self.req_to_token_pool.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_kv_heads=self.num_kv_heads_local,
            token_budget=self.runtime_config.token_budget,
            recent_tokens=self.runtime_config.recent_tokens,
            sink_tokens=self.runtime_config.sink_tokens,
            min_seq_len=self.runtime_config.min_seq_len,
            max_selected=self.runtime_config.max_selected_per_request,
            gqa_reduction_id=gqa_reduction_id(self.runtime_config.gqa_reduction),
        )


def parse_double_sparsity_calibration(server_args) -> DoubleSparsityCalibration:
    """Helper used by tests and the factory."""
    if not server_args.double_sparsity_config:
        raise ValueError(
            "--double-sparsity-config is required when --enable-double-sparsity is set"
        )
    return parse_calibration_file(server_args.double_sparsity_config)


__all__ = [
    "DoubleSparsityAlgorithm",
    "parse_double_sparsity_calibration",
]
