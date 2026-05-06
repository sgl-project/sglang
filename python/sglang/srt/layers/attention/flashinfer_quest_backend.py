"""FlashInfer + Quest sparse decode WITHOUT hisparse offloading.

Profiling: set ``QUEST_PROFILE_STAGES=1`` in the environment to wrap each
forward_decode stage with cuda events.  After ``QUEST_PROFILE_STAGES_N``
calls (default 200), the backend logs per-stage GPU-time (ms): mean,
median, p99.  Use this to find which step dominates Mode 2's per-token
ITL vs dense.


This is "Mode 2": Quest's bounding-box top-k page selection drives sparse
decode attention against a fully on-GPU MHA KV cache.  No HiSparseCoordinator,
no host pool, no swap-in kernel.

Comparison with the hisparse path:

  flashinfer_hisparse  : sparse + offload   (Quest selects top-k, then a
                          swap-in kernel materializes those tokens in a small
                          per-request device buffer; everything else lives on
                          host pinned memory)
  flashinfer_quest     : sparse, no offload (Quest selects top-k token
                          positions, FlashInfer reads them directly from the
                          fully on-GPU KV cache via per-request kv_indices)

Mode 2 is the "is sparsity itself the win, or is it the offloading?"
attribution baseline.  Same accuracy as Mode 3 (same algorithm).  Should win
throughput at small/medium batch (no offloading overhead) and lose at large
batch (full KV in GPU memory caps concurrency).

Plan/run separation (so cuda graph capture works):
  * ``init_forward_metadata`` (eager per-step) and
    ``init_forward_metadata_capture_cuda_graph`` (capture-time) call
    ``_plan_for_step`` to invoke FlashInfer's plan().  plan() does CPU work
    + a host roundtrip that is incompatible with cuda graph capture, so it
    must happen OUTSIDE the captured forward window.
  * ``init_forward_metadata_replay_cuda_graph`` does NOT re-plan — the
    captured kernels reuse the capture-time plan_info (which was computed
    for the worst-case workload, so any smaller actual workload at replay
    is safe).
  * ``forward_decode`` (called per layer, inside the captured graph) only
    does graph-safe ops: Quest scoring + scatter-pack into the kv_indices
    buffer + ``wrapper.run()``.

Hardcoded for v1:
  * MHA pool only (regular MHATokenToKVPool, not the hisparse variant).
  * Inherits dense extend (forward_extend) from FlashInferAttnBackend; only
    forward_decode is overridden for sparse selection.
  * No chunked-prefill: server_args validation forces chunked_prefill_size
    such that bounds bookkeeping stays one-shot per request.
  * Skip-first-decode tracked on the backend (mirrors what HiSparseCoordinator
    does for the hisparse path).
  * Invalidation on request finish is best-effort (no scheduler hook in
    Mode 2); page_valid bits for freed slots may carry over until reused.
  * Capture-time plan assumes uniform sm_scale + logit_cap across layers
    (true for standard transformers).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

logger = logging.getLogger(__name__)

# Per-stage profiling (env-gated; off by default).
_PROFILE_STAGES = os.environ.get("QUEST_PROFILE_STAGES", "0") == "1"
_PROFILE_STAGES_N = int(os.environ.get("QUEST_PROFILE_STAGES_N", "200"))


class _StageProfiler:
    """Records cuda-event time for tagged stages, dumps stats every N calls."""

    def __init__(self, dump_every: int):
        self.dump_every = dump_every
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.calls = 0  # full forward_decode calls (not stages)
        self.dumped = False

    def time(self, tag: str):
        return _StageContext(self, tag)

    def record_call(self):
        self.calls += 1
        if not self.dumped and self.calls >= self.dump_every:
            self.dump()
            self.dumped = True

    def dump(self):
        import sys
        torch.cuda.synchronize()
        lines = [f"### Quest stage profile (over {self.calls} forward_decode calls):"]
        # Order tags alphabetically for stable output.
        for tag in sorted(self.events):
            pairs = self.events[tag]
            ms = [s.elapsed_time(e) for s, e in pairs]
            ms_sorted = sorted(ms)
            n = len(ms)
            mean = sum(ms) / n
            median = ms_sorted[n // 2]
            p99 = ms_sorted[min(n - 1, int(0.99 * n))]
            total = sum(ms)
            lines.append(
                f"  {tag:<30s}  n={n:5d}  mean={mean*1000:6.1f}us  med={median*1000:6.1f}us  p99={p99*1000:6.1f}us  total={total:7.1f}ms"
            )
        # Use print to stderr directly so sglang's log-level filtering
        # doesn't swallow the dump.
        print("\n".join(lines), file=sys.stderr, flush=True)


class _StageContext:
    def __init__(self, profiler: _StageProfiler, tag: str):
        self.profiler = profiler
        self.tag = tag
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_evt.record()
        return self

    def __exit__(self, *args):
        self.end_evt.record()
        self.profiler.events.setdefault(self.tag, []).append(
            (self.start_evt, self.end_evt)
        )


_PROFILER: Optional[_StageProfiler] = (
    _StageProfiler(_PROFILE_STAGES_N) if _PROFILE_STAGES else None
)

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_DEFAULT_WORKSPACE_BYTES = 384 * 1024 * 1024


class FlashInferQuestDecodeBackend(FlashInferAttnBackend):
    """Quest sparse decode against a full on-GPU MHA KV pool."""

    def __init__(self, runner: "ModelRunner"):
        # Inherit the regular FlashInfer backend so prefill (forward_extend)
        # is dense and the parent's wrapper machinery stays usable.  We only
        # override forward_decode for sparse selection.
        super().__init__(runner, init_new_workspace=False)

        from sglang.srt.layers.dp_attention import get_attention_tp_size
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        cfg = parse_hisparse_config(runner.server_args)
        if cfg.algorithm != "quest":
            raise ValueError(
                "flashinfer_quest backend requires --hisparse-config algorithm='quest'."
            )

        self.runner = runner
        self.device = torch.device(runner.device)

        self.quest: "QuestAlgorithm" = QuestAlgorithm(
            top_k=cfg.top_k,
            page_size=cfg.quest_page_size,
            device=self.device,
        )
        self.quest.init_storage(
            start_layer=runner.start_layer,
            end_layer=runner.end_layer,
            max_reqs=runner.req_to_token_pool.req_to_token.shape[0],
            max_context_len=runner.req_to_token_pool.max_context_len,
            kv_heads=runner.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=runner.model_config.head_dim,
        )
        self.top_k = cfg.top_k
        self.num_qo_heads = (
            runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.head_dim = runner.model_config.head_dim
        self.req_to_token = runner.req_to_token_pool.req_to_token

        max_reqs = runner.req_to_token_pool.req_to_token.shape[0]

        # Per-bs sparse decode wrapper cache.  kv_indices is sized for
        # the worst case (all reqs at full top_k) plus 1 scratch slot at
        # the tail used by the graph-safe scatter-pack to absorb writes
        # for invalid positions (j >= actual_lens[i]) without conditional
        # control flow.  Per-call packing fills only the prefix
        # [0, sum(actual_lens)) and kv_indptr reflects actual per-request
        # lengths so FlashInfer doesn't see padding duplicates (which
        # would over-weight the tail token in softmax).
        self._kv_indptr_buf = torch.zeros(
            max_reqs + 1, dtype=torch.int32, device=self.device,
        )
        self._kv_indices_buf = torch.zeros(
            max_reqs * self.top_k + 1, dtype=torch.int32, device=self.device,
        )
        self._scatter_scratch_idx = max_reqs * self.top_k  # last slot
        self._kv_last_page_len_buf = torch.ones(
            max_reqs, dtype=torch.int32, device=self.device,
        )
        self._kv_indptr_template = torch.arange(
            0,
            (max_reqs + 1) * self.top_k,
            step=self.top_k,
            dtype=torch.int32,
            device=self.device,
        )
        # Per-step actual_lens, written by init_forward_metadata (eager) or
        # by the captured cumsum (graph). Read by every layer's
        # forward_decode for the scatter mask.  Stable buffer address so
        # cuda graph sees the same tensor across capture/replay.
        self._actual_lens_buf = torch.zeros(
            max_reqs, dtype=torch.int32, device=self.device,
        )
        # Pre-built [0, top_k) range used in the scatter mask; allocated
        # once to keep forward_decode allocation-free (graph-friendly).
        self._range_top_k = torch.arange(
            self.top_k, dtype=torch.int32, device=self.device,
        )

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, fast_decode_plan
        self._WrapperCls = BatchDecodeWithPagedKVCacheWrapper
        self._fast_decode_plan = fast_decode_plan

        # Workspace shared across all per-bs wrappers.  We can reuse the
        # parent's workspace_buffer to avoid double allocation.
        self._sparse_workspace = (
            self.workspace_buffer
            if hasattr(self, "workspace_buffer") and self.workspace_buffer is not None
            else torch.empty(_DEFAULT_WORKSPACE_BYTES, dtype=torch.uint8,
                             device=self.device)
        )

        self._sparse_wrappers: dict = {}
        self._sparse_wrapper: Optional[object] = None
        self._current_bs: Optional[int] = None
        # Capture-time sm_scale + logits_soft_cap (set on first plan).  The
        # graph-replay path reuses the capture-time plan_info; eager mode
        # plans fresh each step but with the same scaling values.
        self._cached_sm_scale: Optional[float] = None
        self._cached_logits_soft_cap: float = 0.0

        # Per-request "skip first decode bounds update" flag.  Set when prefill
        # bounds are computed for a request; cleared on the first decode step
        # for that request.  Mirrors HiSparseCoordinator._skip_first_backup.
        self._skip_first_decode_update = torch.zeros(
            max_reqs, dtype=torch.bool, device=self.device,
        )

        self.kv_data_type = runner.kv_cache_dtype
        self.q_data_type = runner.dtype

    # ------------------------------------------------------ wrapper cache

    def _get_or_create_sparse_wrapper(self, bs: int):
        if bs in self._sparse_wrappers:
            return self._sparse_wrappers[bs]

        kv_indptr = self._kv_indptr_buf[: bs + 1]
        kv_indices = self._kv_indices_buf[: bs * self.top_k]
        kv_last_page_len = self._kv_last_page_len_buf[:bs]
        # Seed the buffer with the worst-case uniform top_k stride so the
        # constructor's first plan() sees a sensible state.
        kv_indptr.copy_(self._kv_indptr_template[: bs + 1])

        wrapper = self._WrapperCls(
            self._sparse_workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=kv_indptr,
            paged_kv_indices_buffer=kv_indices,
            paged_kv_last_page_len_buffer=kv_last_page_len,
        )
        # Initial regular plan() — populates the wrapper's _cached_module
        # which fast_decode_plan needs.  Uses worst-case uniform top_k
        # so the cached scheduling is for max workload.
        wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            sm_scale=1.0 / (self.head_dim ** 0.5),
            data_type=self.kv_data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
        )
        self._sparse_wrappers[bs] = wrapper
        return wrapper

    # ------------------------------------------------------ plan helper

    def _plan_for_step(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        sm_scale: float,
        logits_soft_cap: float,
        worst_case: bool = False,
    ) -> None:
        """Stamp the wrapper's stored buffers and call FlashInfer's plan().

        Used at three call sites:
          * eager per-step (init_forward_metadata): plan with the actual
            indptr derived from this step's seq_lens (optimal scheduling).
          * cuda-graph capture (init_forward_metadata_capture_cuda_graph):
            plan with the WORST-CASE indptr (uniform top_k stride) so the
            captured kernels can handle any actual indptr at replay time.
          * (replay path does NOT re-plan — it reuses capture-time plan_info.)

        Uses ``fast_decode_plan`` with ``global_override_indptr_cpu`` to
        skip FlashInfer's internal device→host copy of indptr.
        """
        if worst_case:
            actual_lens_cpu = torch.full(
                (bs,), self.top_k, dtype=torch.int32,
            )
        else:
            if seq_lens_cpu is None:
                seq_lens_cpu = seq_lens.cpu()
            actual_lens_cpu = seq_lens_cpu.to(torch.int32).clamp_max(self.top_k)

        # CPU-side indptr (cumulative actual_lens) — passed to plan via
        # global_override_indptr_cpu to skip the device→host sync.
        indptr_cpu = torch.zeros(bs + 1, dtype=torch.int32)
        indptr_cpu[1:] = torch.cumsum(actual_lens_cpu, dim=0).to(torch.int32)

        # Mirror to the device-side actual_lens + indptr buffers.  These
        # writes are blocking (called outside any captured window) so the
        # buffers are in a known state when capture begins.
        self._actual_lens_buf[:bs].copy_(actual_lens_cpu, non_blocking=False)
        self._kv_indptr_buf[: bs + 1].copy_(indptr_cpu, non_blocking=False)

        self._cached_sm_scale = sm_scale
        self._cached_logits_soft_cap = logits_soft_cap

        self._fast_decode_plan(
            self._sparse_wrapper,
            indptr=self._kv_indptr_buf[: bs + 1],
            indices=self._kv_indices_buf[: bs * self.top_k],
            last_page_len=self._kv_last_page_len_buf[:bs],
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            sm_scale=sm_scale,
            logits_soft_cap=logits_soft_cap,
            data_type=self.kv_data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
            global_override_indptr_cpu=indptr_cpu,
        )

    # --------------------------------------------------- AttentionBackend

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Per-step prep (eager / non-graph path).

        Parent does the dense flashinfer planning; we add Quest's decode
        bounds update (for the previous step's just-decoded token) and
        plan the sparse wrapper with this step's actual indptr."""
        super().init_forward_metadata(forward_batch)

        if forward_batch.forward_mode.is_decode():
            bs = int(forward_batch.batch_size)
            self._sparse_wrapper = self._get_or_create_sparse_wrapper(bs)
            self._current_bs = bs
            self._do_quest_decode_step_update(
                forward_batch.req_pool_indices, forward_batch.seq_lens,
            )
            # Cache step-level state once per step (used by every layer's
            # retrieve_topk).  Saves ~6 redundant ops per layer in the
            # captured graph.
            self.quest.prepare_step(forward_batch.seq_lens)
            # Plan with this step's actual indptr.  sm_scale + logits_soft_cap
            # are layer-independent for standard transformers; we lazily
            # capture them on the first plan from a representative layer
            # (forward_decode populates the cache on its first invocation).
            sm_scale = (
                self._cached_sm_scale
                if self._cached_sm_scale is not None
                else 1.0 / (self.head_dim ** 0.5)
            )
            self._plan_for_step(
                bs=bs,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=getattr(forward_batch, "seq_lens_cpu", None),
                sm_scale=sm_scale,
                logits_soft_cap=self._cached_logits_soft_cap,
                worst_case=False,
            )

    def _do_quest_decode_step_update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Update Quest running bounds with the previous step's decoded K.

        Used by both the non-graph path (via ``init_forward_metadata``) and
        the cuda-graph replay path (via ``init_forward_metadata_replay_cuda_graph``).
        Skip during capture: capture uses synthetic batches and we don't want
        their K to pollute the running bounds.

        Skips requests flagged "first decode after prefill" — for those,
        bounds were already seeded by ``update_prefill_representations``
        and no decode token has been generated yet."""
        skip_mask = self._skip_first_decode_update[req_pool_indices]
        active_mask = ~skip_mask
        # Always clear the skip flag for the requests we just saw so the
        # NEXT decode step does update.
        self._skip_first_decode_update[req_pool_indices] = False

        # No host-device sync here: if all reqs are skipped, ``active_idx``
        # is empty and the fused kernel + finalize below are no-ops.  The
        # previous ``bool(active_mask.any().item())`` guard added a per-step
        # sync that visibly bumped ITL on the cuda-graph replay path.
        active_idx = req_pool_indices[active_mask]
        active_seq_lens = seq_lens[active_mask]
        # The just-decoded token sits at position seq_len - 2 (the post-
        # increment in prepare_for_decode bumped seq_lens by 1 for the
        # upcoming step's about-to-be-generated token).
        prev_token_pos = (active_seq_lens - 2).clamp(min=0).long()
        device_locs = self.req_to_token[active_idx, prev_token_pos]

        # Fused all-layer kernel: one launch instead of num_layers Python
        # iterations (each previously did gather + minimum/maximum on
        # running bounds).  Then finalize counters across all layers.
        pool = self.runner.token_to_kv_pool
        self.quest.update_decode_representations_fused(
            k_data_ptrs=pool.k_data_ptrs,
            req_indices=active_idx,
            device_locs=device_locs,
        )
        self.quest.maybe_finalize_decode_representations(active_idx)

    # ------------------------------------------------------ forward_extend

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Dense extend (parent).  Quest bounds compute is deferred to the
        last layer's call (single fused kernel processes all layers).

        The previous per-layer Python loop (with .tolist() syncs and a
        separate update_prefill_representations call per req per layer) was
        a major source of prefill-time overhead — each .tolist() is a D2H
        sync, multiplied by 48 layers × N reqs, and each per-(layer, req)
        op chain serialised through the staging path.  Deferring keeps the
        same correctness (bounds are read only by retrieve_topk at decode
        time) while collapsing the work into a single per-req fused kernel
        launch."""
        out = super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        # Bounds depend on K for all Quest layers; run after the last Quest
        # layer's K has been written to its buffer.
        if save_kv_cache and layer.layer_id == self.quest.end_layer - 1:
            self._update_quest_for_extend(forward_batch)
        return out

    def _update_quest_for_extend(
        self, forward_batch: "ForwardBatch"
    ) -> None:
        """Compute Quest prefill bounds for each request in this extend call.

        Called ONCE per extend (at the last Quest layer's forward_extend),
        not per layer.  For each request we recompute bounds from positions
        ``0 .. seq_len - 1`` (pulled via ``req_to_token``).  This is correct
        under chunked prefill (each chunk recomputes from scratch — wasteful
        but accurate); the backend forces chunked_prefill off so each request
        sees exactly one extend call.
        """
        # One .tolist() per extend (vs. 2 × num_layers previously).  Lists
        # are needed for Python-level iteration over the per-req fused kernel.
        req_pool_indices = forward_batch.req_pool_indices.tolist()
        seq_lens = forward_batch.seq_lens.tolist()
        k_data_ptrs = forward_batch.token_to_kv_pool.k_data_ptrs

        for req_idx, seq_len in zip(req_pool_indices, seq_lens):
            if seq_len <= 0:
                continue
            prefill_indices = self.req_to_token[req_idx, :seq_len]
            # Fused all-layer kernel: one launch instead of num_layers
            # per-layer Python ops.
            self.quest.update_prefill_representations_fused(
                req_pool_idx=req_idx,
                k_data_ptrs=k_data_ptrs,
                prefill_indices=prefill_indices,
            )
            self._skip_first_decode_update[req_idx] = True

    # ------------------------------------------------------ forward_decode

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Sparse decode: save K/V → Quest top-k → FlashInfer sparse attention.

        Per-request kv_indices length = ``min(seq_len, top_k)``; positions
        are scatter-packed into the wrapper's pre-allocated kv_indices buffer
        (graph-safe, no boolean indexing).  Plan was done at step entry
        (init_forward_metadata or capture-time hook), not here.

        First-call side effect: caches ``layer.scaling`` + ``layer.logit_cap``
        on ``self`` so subsequent eager-mode plans use the right values.
        """
        if self._current_bs is None:
            raise RuntimeError(
                "init_forward_metadata must run before forward_decode"
            )
        bs = self._current_bs

        # Lazy-cache layer scaling on first call so eager-mode init can plan
        # without seeing a layer reference.
        if self._cached_sm_scale is None:
            self._cached_sm_scale = layer.scaling
            self._cached_logits_soft_cap = getattr(layer, "logit_cap", 0.0) or 0.0

        # Local timer no-op when profiling disabled.
        prof = _PROFILER

        # 1. Save the just-generated K/V to the regular MHA pool.
        if prof is not None:
            with prof.time("1_set_kv_buffer"):
                if save_kv_cache and k is not None:
                    assert v is not None
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, k, v,
                        getattr(layer, "k_scale", None),
                        getattr(layer, "v_scale", None),
                    )
        else:
            if save_kv_cache and k is not None:
                assert v is not None
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v,
                    getattr(layer, "k_scale", None),
                    getattr(layer, "v_scale", None),
                )

        # 2. Quest top-k logical positions + per-request actual lengths.
        if prof is not None:
            with prof.time("2_retrieve_topk"):
                topk_token_positions, _ = self.quest.retrieve_topk(
                    queries=q, layer_id=layer.layer_id,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                )
        else:
            topk_token_positions, _ = self.quest.retrieve_topk(
                queries=q, layer_id=layer.layer_id,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
            )

        # 3+4. Fused gather + scatter-pack into kv_indices_buf in one Triton
        # launch.  Replaces a separate advanced-indexing gather (with .long()
        # cast + .to(int32) cast) plus the 6-op scatter-pack chain.
        from sglang.srt.mem_cache.sparsity.algorithms.quest_scatter_pack_kernel import (
            quest_only_gather_scatter,
        )
        if prof is not None:
            with prof.time("3_4_gather_scatter"):
                quest_only_gather_scatter(
                    topk_positions=topk_token_positions,
                    req_pool_indices=forward_batch.req_pool_indices,
                    req_to_token=self.req_to_token,
                    actual_lens=self._actual_lens_buf[:bs],
                    kv_indptr=self._kv_indptr_buf[:bs],
                    kv_indices_buf=self._kv_indices_buf,
                    scratch_idx=self._scatter_scratch_idx,
                    top_k=self.top_k,
                )
        else:
            quest_only_gather_scatter(
                topk_positions=topk_token_positions,
                req_pool_indices=forward_batch.req_pool_indices,
                req_to_token=self.req_to_token,
                actual_lens=self._actual_lens_buf[:bs],
                kv_indptr=self._kv_indptr_buf[:bs],
                kv_indices_buf=self._kv_indices_buf,
                scratch_idx=self._scatter_scratch_idx,
                top_k=self.top_k,
            )

        # 5. FlashInfer sparse attention against the full MHA KV pool.
        kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if isinstance(kv_buffer, tuple):
            raw_k, raw_v = kv_buffer
            kv_paged = (
                raw_k.view(-1, 1, self.num_kv_heads, self.head_dim),
                raw_v.view(-1, 1, self.num_kv_heads, self.head_dim),
            )
        else:
            kv_paged = kv_buffer

        if prof is not None:
            with prof.time("5_wrapper_run"):
                o = self._sparse_wrapper.run(
                    q.contiguous().view(-1, self.num_qo_heads, self.head_dim),
                    kv_paged,
                    k_scale=getattr(layer, "k_scale_float", None),
                    v_scale=getattr(layer, "v_scale_float", None),
                )
            prof.record_call()
        else:
            o = self._sparse_wrapper.run(
                q.contiguous().view(-1, self.num_qo_heads, self.head_dim),
                kv_paged,
                k_scale=getattr(layer, "k_scale_float", None),
                v_scale=getattr(layer, "v_scale_float", None),
            )
        return o.view(-1, self.num_qo_heads * self.head_dim)

    # -------------------------------------------------- cuda graph hooks

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        if max_bs > self._kv_indices_buf.shape[0] // self.top_k:
            raise RuntimeError(
                f"cuda graph max_bs={max_bs} exceeds quest backend capacity"
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ) -> None:
        super().init_forward_metadata_capture_cuda_graph(
            bs, num_tokens, req_pool_indices, seq_lens,
            encoder_lens, forward_mode, spec_info,
        )
        if forward_mode.is_decode():
            bs_int = int(bs)
            self._sparse_wrapper = self._get_or_create_sparse_wrapper(bs_int)
            self._current_bs = bs_int
            # Cache step-level state with placeholder seq_lens for capture.
            self.quest.prepare_step(seq_lens[:bs_int])
            # Plan with WORST-CASE indptr (uniform top_k per request) so the
            # captured plan_info handles any actual indptr at replay time.
            sm_scale = (
                self._cached_sm_scale
                if self._cached_sm_scale is not None
                else 1.0 / (self.head_dim ** 0.5)
            )
            self._plan_for_step(
                bs=bs_int,
                seq_lens=seq_lens,
                seq_lens_cpu=None,
                sm_scale=sm_scale,
                logits_soft_cap=self._cached_logits_soft_cap,
                worst_case=True,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        super().init_forward_metadata_replay_cuda_graph(
            bs, req_pool_indices, seq_lens, seq_lens_sum,
            encoder_lens, forward_mode, spec_info, seq_lens_cpu=seq_lens_cpu,
        )
        if forward_mode.is_decode():
            bs_int = int(bs)
            if bs_int not in self._sparse_wrappers:
                raise RuntimeError(
                    f"replay called for bs={bs_int} but no sparse wrapper "
                    f"was captured; init_forward_metadata_capture_cuda_graph first"
                )
            self._sparse_wrapper = self._sparse_wrappers[bs_int]
            self._current_bs = bs_int
            # Real per-step decode: update Quest bounds for the previous
            # token before the captured graph replays.
            self._do_quest_decode_step_update(
                req_pool_indices[:bs_int], seq_lens[:bs_int],
            )
            # Cache step-level state once (instead of per-layer in retrieve_topk).
            self.quest.prepare_step(seq_lens[:bs_int])
            # Re-plan with this step's actual indptr.  This runs OUTSIDE
            # the captured graph window (sglang invokes it before replay)
            # so the host roundtrip inside plan() is allowed.
            sm_scale = (
                self._cached_sm_scale
                if self._cached_sm_scale is not None
                else 1.0 / (self.head_dim ** 0.5)
            )
            self._plan_for_step(
                bs=bs_int,
                seq_lens=seq_lens[:bs_int],
                seq_lens_cpu=(
                    seq_lens_cpu[:bs_int] if seq_lens_cpu is not None else None
                ),
                sm_scale=sm_scale,
                logits_soft_cap=self._cached_logits_soft_cap,
                worst_case=False,
            )
