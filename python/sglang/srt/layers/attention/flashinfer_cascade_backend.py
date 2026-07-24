"""FlashInfer Cascade attention backend.

Subclasses ``FlashInferAttnBackend`` and adds a
``MultiLevelCascadeAttentionWrapper`` decode path that batches the leading
shared-prefix portion of the running batch into a single matmul. The stock
``flashinfer`` backend runs ``BatchDecodeWithPagedKVCacheWrapper`` per-request
even when prefixes are deduped at the storage layer (RadixAttention shares
pages, but Q.K is still computed per-request). FlashInfer ships
``MultiLevelCascadeAttentionWrapper`` which can batch the shared portion
across requests; this backend wires it into SGLang's decode path.

Scope:
    * Decode-only. Extend (prefill) falls through to the parent class.
    * Eager mode: cascade fires when (1) the detected shared-prefix length
      passes ``--cascade-min-prefix-tokens`` and (2) batch size passes
      ``--cascade-min-batch-size``. Otherwise the parent per-request path
      runs.
    * CUDA-graph mode: every captured ``cuda_graph_bs`` gets its own
      wrapper plus pre-allocated indptr/indices buffers; ``plan()`` writes
      into those buffers per replay step (host-side, before the graph
      fires); the captured ``run()`` reads from the same addresses.
      Cascade is always armed in CG-mode regardless of threshold:
      ``common_prefix=0`` is mathematically equivalent to per-request
      decode plus a no-op level-0 launch, so correctness holds at every
      bs in the captured list while keeping the capture graph singular.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    from flashinfer import MultiLevelCascadeAttentionWrapper


class _CascadePlanState(msgspec.Struct):
    """Per-step cascade plan state. Set by ``init_forward_metadata`` (eager)
    or ``init_forward_metadata_out_graph`` (CG) when the cascade arms;
    consumed by ``forward_decode``. ``None`` means cascade did not fire for
    this step (parent's per-request path runs instead).
    """

    common_prefix_tokens: int
    bs: int


class FlashInferCascadeAttnBackend(FlashInferAttnBackend):
    """FlashInfer backend with a cross-request shared-prefix decode path.

    Detection (eager): walks each request's leading slot indices in
    ``req_to_token``, finds the longest run where every request's slot
    matches request 0's. Cascade fires when that run is at least
    ``cascade_min_prefix_tokens`` long AND batch size is at least
    ``cascade_min_batch_size``.

    Detection (CG-mode): same algorithm, but driven from ``req_pool_indices``
    + ``seq_lens`` (forward_batch is not available at replay-time).

    Under CUDA graphs every captured ``cuda_graph_bs`` gets its own wrapper
    plus pre-allocated indptr/indices buffers; ``plan()`` writes into those
    buffers per replay step (host-side, before the graph fires); the captured
    ``run()`` reads from the same addresses. Cascade is always armed in
    CG-mode regardless of threshold: cascade with ``common_prefix=0`` is
    mathematically equivalent to per-request decode plus a no-op level-0
    launch, so correctness holds at every bs in the captured list while
    keeping the capture graph singular (instead of capturing two arms per
    bs).
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)

        self.cascade_min_prefix_tokens: int = int(
            getattr(model_runner.server_args, "cascade_min_prefix_tokens", 128)
        )
        self.cascade_min_batch_size: int = int(
            getattr(model_runner.server_args, "cascade_min_batch_size", 4)
        )
        if self.cascade_min_prefix_tokens < 1:
            self.cascade_min_prefix_tokens = 1
        if self.cascade_min_batch_size < 2:
            self.cascade_min_batch_size = 2

        # Tracks whether the current step is inside CG capture/replay (set by
        # the CG hooks below; reset by eager init).
        self._in_cuda_graph: bool = False

        # Parent's KV pool: SGLang treats each slot as a 1-page row in the
        # flashinfer wrapper (page_size=1). We mirror that here for the
        # cascade wrapper so kv_indices are slot ids the same way.
        self.cascade_page_size: int = 1
        self.num_qo_heads: int = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads_local: int = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim_local: int = model_runner.model_config.head_dim
        self.q_dtype = model_runner.dtype
        self.kv_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_token_stride = self.req_to_token.shape[1]
        self._device = model_runner.device

        # Cap detection cost: scan at most this many leading slot positions.
        # Most realistic prefixes are <= 16K tokens; a hard cap keeps the
        # CPU-side scan O(min(seq_len, cap)).
        self._cascade_scan_cap: int = 32768

        # Eager-mode cascade wrapper. Use_cuda_graph=False; plan allocates
        # scheduler state per call (acceptable for eager since plan runs
        # per step anyway).
        self._cascade_decode_wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None
        if is_flashinfer_available():
            self._cascade_decode_wrapper = MultiLevelCascadeAttentionWrapper(
                num_levels=2,
                float_workspace_buffer=self.workspace_buffer,
                kv_layout="NHD",
                use_cuda_graph=False,
            )

        # Eager plan stash. Set by init_forward_metadata; read by
        # forward_decode. ``None`` means eager cascade did not arm.
        self._cascade_plan: Optional[_CascadePlanState] = None

        # ---- CG-mode state ----
        # Per-bs wrappers + pre-allocated buffers. Filled lazily during
        # init_forward_metadata_out_graph(in_capture=True) (called once per
        # cuda_graph_bs by the runner). Each entry holds a wrapper sized
        # to one specific captured batch size; FlashInfer's CG-mode
        # contract enforces fixed batch_size across plan() calls.
        self._cg_cascade_wrappers: dict = {}  # bs -> MultiLevelCascadeAttentionWrapper
        self._cg_cascade_buffers: dict = {}  # bs -> dict of pre-alloc tensors
        # Per-replay plan stash for CG path (separate from eager path so
        # the two arms cannot interfere). Set by replay hook; read by
        # forward_decode when ``_in_cuda_graph`` is true.
        self._cg_cascade_plan: Optional[_CascadePlanState] = None
        # CG buffer sizing (set in init_cuda_graph_state).
        self._cg_max_bs: int = 0
        # Worst-case shared prefix length: cap by (max_context_len, scan_cap).
        # Using max_context_len ensures level-0 indices buffer can hold any
        # plausible prefix within the model's context window.
        self._cg_max_shared_pages: int = 0
        # Worst-case unique slots per request: max_context_len.
        self._cg_max_pages_per_req: int = 0

        # Toggle: SGLANG_CASCADE_DISABLE_CUDA_GRAPH=1 forces eager-only cascade
        # even when CG is enabled (useful for debugging and bisection).
        self._cg_disabled: bool = (
            os.environ.get("SGLANG_CASCADE_DISABLE_CUDA_GRAPH", "0") == "1"
        )

        # Debug counters. Set ``SGLANG_CASCADE_DEBUG=1`` to log per-step
        # decisions. Used by tests to confirm fire/skip behavior.
        self._dbg_enabled: bool = os.environ.get("SGLANG_CASCADE_DEBUG", "0") == "1"
        self._dbg_total_decode_steps: int = 0
        self._dbg_cascade_fired: int = 0  # eager fires (>= threshold)
        self._dbg_cascade_fired_cg: int = 0  # CG-mode fires (>= threshold)
        self._dbg_cascade_run_cg: int = (
            0  # CG-mode actually ran (incl. below-threshold)
        )
        self._dbg_skip_below_bs: int = 0
        self._dbg_skip_below_prefix: int = 0
        self._dbg_skip_below_prefix_cg: int = 0  # CG ran but threshold not met
        self._dbg_skip_in_cg: int = 0
        self._dbg_skip_not_decode: int = 0

        logger.info(
            "FlashInferCascadeAttnBackend initialized "
            "(min_prefix_tokens=%d, min_batch_size=%d, num_qo_heads=%d, "
            "num_kv_heads=%d, head_dim=%d, cg_disabled=%s)",
            self.cascade_min_prefix_tokens,
            self.cascade_min_batch_size,
            self.num_qo_heads,
            self.num_kv_heads_local,
            self.head_dim_local,
            self._cg_disabled,
        )

    def cascade_debug_counters(self) -> dict:
        """Snapshot of debug counters; used by tests to assert fire/skip
        behavior. Always available regardless of ``SGLANG_CASCADE_DEBUG``.
        """
        return {
            "total_decode_steps": self._dbg_total_decode_steps,
            "cascade_fired": self._dbg_cascade_fired,
            "cascade_fired_cg": self._dbg_cascade_fired_cg,
            "cascade_run_cg": self._dbg_cascade_run_cg,
            "skip_below_bs": self._dbg_skip_below_bs,
            "skip_below_prefix": self._dbg_skip_below_prefix,
            "skip_below_prefix_cg": self._dbg_skip_below_prefix_cg,
            "skip_in_cg": self._dbg_skip_in_cg,
            "skip_not_decode": self._dbg_skip_not_decode,
        }

    # ------------------------------------------------------------------
    # Detection helpers (host-side; used in both eager and CG paths)
    # ------------------------------------------------------------------

    def _detect_common_prefix_from_rpi(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> int:
        """CG-friendly detection: walks the leading slot indices using only
        ``req_pool_indices`` and ``seq_lens_cpu`` (no forward_batch).

        Returns the count of leading positions where all requests share the
        same slot index, capped at ``min_seq - 1`` so each request keeps at
        least 1 unique slot for its current decode token.
        """
        if bs < 2:
            return 0
        if seq_lens_cpu is None:
            seq_lens_cpu = req_pool_indices.new_empty(0)  # placeholder
        if seq_lens_cpu.numel() == 0:
            return 0
        min_seq = int(seq_lens_cpu[:bs].min().item())
        if min_seq <= 1:
            return 0
        scan_n = min(min_seq, self._cascade_scan_cap)
        # Compare/reduce entirely on-device and sync only a single scalar
        # back to host (instead of copying the whole [bs, scan_n] slice).
        leading = self.req_to_token[req_pool_indices[:bs].long(), :scan_n]
        # mismatch[j] is True if any request's slot j differs from request 0's.
        mismatch = (leading != leading[0:1]).any(dim=0)
        # Append a True sentinel at index scan_n so argmax always resolves to
        # the first divergence -- or to scan_n itself when fully shared. This
        # keeps the whole reduction on-device with one scalar sync.
        sentinel = torch.ones(1, dtype=torch.bool, device=mismatch.device)
        common = int(torch.cat([mismatch, sentinel]).to(torch.uint8).argmax().item())
        common = min(common, min_seq - 1)
        return max(0, common)

    def _detect_common_prefix_tokens(self, forward_batch: ForwardBatch, bs: int) -> int:
        """Eager-mode detection wrapper -- forwards to the CG-friendly impl
        using the forward_batch's seq_lens_cpu / req_pool_indices.
        """
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()
        return self._detect_common_prefix_from_rpi(
            bs, forward_batch.req_pool_indices, seq_lens_cpu
        )

    # ------------------------------------------------------------------
    # Eager-mode cascade plan
    # ------------------------------------------------------------------

    def _build_cascade_plan_args(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        common_prefix_tokens: int,
    ):
        """Eager-mode plan builder. Allocates fresh tensors and calls
        ``self._cascade_decode_wrapper.plan(...)``. Returns a stash state
        on success, ``None`` on failure (caller falls through to parent).
        """
        device = forward_batch.input_ids.device

        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()
        seq_lens_list = seq_lens_cpu[:bs].tolist()

        qo_indptr_l0 = torch.tensor([0, bs], dtype=torch.int32, device=device)
        kv_indptr_l0 = torch.tensor(
            [0, common_prefix_tokens], dtype=torch.int32, device=device
        )
        rpi = forward_batch.req_pool_indices
        # Materialize the pool indices to host once; the per-request loop below
        # then indexes a Python list instead of issuing bs separate .item()
        # device syncs per decode step.
        rpi_list = rpi[:bs].cpu().tolist()
        first_rpi = int(rpi_list[0])
        kv_indices_l0 = self.req_to_token[first_rpi, :common_prefix_tokens].to(
            torch.int32
        )
        last_page_len_l0 = torch.tensor([1], dtype=torch.int32, device=device)

        qo_indptr_l1 = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        qo_indptr_l1[1:] = torch.arange(1, bs + 1, dtype=torch.int32, device=device)

        unique_lens = [s - common_prefix_tokens for s in seq_lens_list]
        total_unique = sum(unique_lens)
        kv_indptr_l1 = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr_l1_cpu = [0]
        cum = 0
        for ul in unique_lens:
            cum += ul
            kv_indptr_l1_cpu.append(cum)
        kv_indptr_l1.copy_(
            torch.tensor(kv_indptr_l1_cpu, dtype=torch.int32),
            non_blocking=True,
        )

        if total_unique > 0:
            kv_indices_l1 = torch.empty(total_unique, dtype=torch.int32, device=device)
            offset = 0
            for i in range(bs):
                rpi_i = int(rpi_list[i])
                ul = unique_lens[i]
                if ul > 0:
                    kv_indices_l1[offset : offset + ul] = self.req_to_token[
                        rpi_i,
                        common_prefix_tokens : common_prefix_tokens + ul,
                    ].to(torch.int32)
                    offset += ul
        else:
            return None

        last_page_len_l1 = torch.ones(bs, dtype=torch.int32, device=device)

        try:
            self._cascade_decode_wrapper.plan(
                qo_indptr_arr=[qo_indptr_l0, qo_indptr_l1],
                paged_kv_indptr_arr=[kv_indptr_l0, kv_indptr_l1],
                paged_kv_indices_arr=[kv_indices_l0, kv_indices_l1],
                paged_kv_last_page_len=[last_page_len_l0, last_page_len_l1],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads_local,
                head_dim=self.head_dim_local,
                page_size=self.cascade_page_size,
                causal=False,
                pos_encoding_mode="NONE",
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )
        except Exception as e:
            if self._dbg_enabled:
                logger.warning("Cascade plan failed, falling through: %s", e)
            return None

        return _CascadePlanState(common_prefix_tokens=common_prefix_tokens, bs=bs)

    # ------------------------------------------------------------------
    # CG-mode cascade plan
    # ------------------------------------------------------------------

    def _allocate_cg_cascade_for_bs(self, bs: int) -> None:
        """Allocate per-bs cascade wrapper + indptr/indices buffers. Called
        lazily on first capture for each cuda_graph_bs entry.

        Buffer sizing per level (FlashInfer 0.6.8.post1 contract):
            Level 0 (shared, 1 merged group):
                qo_indptr [2], paged_kv_indptr [2],
                paged_kv_indices [_cg_max_shared_pages],
                paged_kv_last_page_len [1]
            Level 1 (bs unique tails):
                qo_indptr [bs + 1], paged_kv_indptr [bs + 1],
                paged_kv_indices [bs * _cg_max_pages_per_req],
                paged_kv_last_page_len [bs]
        """
        if bs in self._cg_cascade_wrappers:
            return
        d = self._device

        # qo_indptr_l0 = [0, bs] and qo_indptr_l1 = [0, 1, ..., bs] are static
        # for a given captured bs (one query per request, every replay). Set
        # them once here -- the buffer addresses stay stable for the captured
        # graph and the values never change, so the per-replay copies in
        # _fill_cg_cascade_plan are unnecessary.
        qo_indptr_l0 = torch.tensor([0, bs], dtype=torch.int32, device=d)
        qo_indptr_l1 = torch.arange(bs + 1, dtype=torch.int32, device=d)
        kv_indptr_l0 = torch.zeros(2, dtype=torch.int32, device=d)
        kv_indptr_l1 = torch.zeros(bs + 1, dtype=torch.int32, device=d)
        kv_indices_l0 = torch.zeros(
            self._cg_max_shared_pages, dtype=torch.int32, device=d
        )
        # Level-1 indices: bs requests, each up to max_pages_per_req slots.
        kv_indices_l1 = torch.zeros(
            bs * self._cg_max_pages_per_req, dtype=torch.int32, device=d
        )
        last_page_l0 = torch.ones(1, dtype=torch.int32, device=d)
        last_page_l1 = torch.ones(bs, dtype=torch.int32, device=d)

        wrapper = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=self.workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf_arr=[qo_indptr_l0, qo_indptr_l1],
            paged_kv_indptr_buf_arr=[kv_indptr_l0, kv_indptr_l1],
            paged_kv_indices_buf_arr=[kv_indices_l0, kv_indices_l1],
            paged_kv_last_page_len_buf_arr=[last_page_l0, last_page_l1],
        )

        self._cg_cascade_wrappers[bs] = wrapper
        self._cg_cascade_buffers[bs] = {
            "qo_indptr_l0": qo_indptr_l0,
            "qo_indptr_l1": qo_indptr_l1,
            "kv_indptr_l0": kv_indptr_l0,
            "kv_indptr_l1": kv_indptr_l1,
            "kv_indices_l0": kv_indices_l0,
            "kv_indices_l1": kv_indices_l1,
            "last_page_l0": last_page_l0,
            "last_page_l1": last_page_l1,
        }

    def _fill_cg_cascade_plan(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        common_prefix_tokens: int,
    ) -> bool:
        """Fill the per-bs cascade buffers in-place with the current step's
        metadata and call plan(). Returns False on failure (caller logs and
        falls through to parent's CG decode path).

        Buffers are written via ``.copy_()`` / index assignment so device
        addresses remain stable across replays (the captured graph reads
        from these same addresses).
        """
        wrapper = self._cg_cascade_wrappers.get(bs)
        bufs = self._cg_cascade_buffers.get(bs)
        if wrapper is None or bufs is None:
            return False

        d = self._device

        # Read seq_lens host-side once (small sync, bs values).
        if seq_lens_cpu is None:
            # Build the placeholder on CPU directly -- a device tensor here
            # would force an extra GPU->CPU sync via the .tolist() below.
            seq_lens_cpu = torch.zeros(bs, dtype=torch.int32, device="cpu")
        seq_lens_list = seq_lens_cpu[:bs].tolist()
        rpi_list = req_pool_indices[:bs].cpu().tolist()

        # Level 0: 1 merged group of bs queries reading common_prefix_tokens
        # shared slots. With page_size=1 the "last page" is always full
        # (single token per slot), and last_page_len = 1 if there is at
        # least one shared slot, else 0 to indicate level 0 is empty.
        # qo_indptr_l0 ([0, bs]) is static and pre-initialized at allocation.
        bufs["kv_indptr_l0"].copy_(
            torch.tensor([0, common_prefix_tokens], dtype=torch.int32),
            non_blocking=True,
        )
        if common_prefix_tokens > 0:
            # Slot ids: leading common_prefix_tokens from request 0. All
            # requests share these slots (RadixAttention dedupes matched
            # prefixes -> equal leading slot ids imply same physical KV).
            shared_slots = self.req_to_token[
                int(rpi_list[0]), :common_prefix_tokens
            ].to(torch.int32)
            bufs["kv_indices_l0"][:common_prefix_tokens].copy_(
                shared_slots, non_blocking=True
            )
            bufs["last_page_l0"].fill_(1)
        else:
            # No shared slots -> level 0 is a no-op. last_page_len=0 and
            # kv_indices is unread (kv_indptr_l0[1] = 0).
            bufs["last_page_l0"].fill_(0)

        # Level 1: bs unique tails of length (seq_len - common_prefix).
        # qo_indptr_l1 = [0, 1, 2, ..., bs] (one query per request) is static and
        # pre-initialized at allocation.
        unique_lens = [max(0, int(s) - common_prefix_tokens) for s in seq_lens_list]
        kv_indptr_l1_cpu = [0]
        cum = 0
        for ul in unique_lens:
            cum += ul
            kv_indptr_l1_cpu.append(cum)
        total_unique = cum
        bufs["kv_indptr_l1"].copy_(
            torch.tensor(kv_indptr_l1_cpu, dtype=torch.int32),
            non_blocking=True,
        )
        # Sanity: total_unique must fit in pre-alloc buffer.
        if total_unique > bufs["kv_indices_l1"].numel():
            return False

        if total_unique > 0:
            # Build the level-1 indices CPU-side then bulk-copy to the
            # pre-allocated buffer. A Python loop is fine here (bs <= 80).
            offset = 0
            for i in range(bs):
                ul = unique_lens[i]
                if ul == 0:
                    continue
                slots = self.req_to_token[
                    int(rpi_list[i]),
                    common_prefix_tokens : common_prefix_tokens + ul,
                ].to(torch.int32)
                bufs["kv_indices_l1"][offset : offset + ul].copy_(
                    slots, non_blocking=True
                )
                offset += ul

        bufs["last_page_l1"].fill_(1)

        try:
            # plan() runs host-side (it does .cpu() syncs internally to read
            # qo_indptr's last value etc.). It writes into the wrapper's
            # internal scheduler buffers, which are also fixed-shape since
            # the wrapper was built with use_cuda_graph=True.
            wrapper.plan(
                qo_indptr_arr=[bufs["qo_indptr_l0"], bufs["qo_indptr_l1"]],
                paged_kv_indptr_arr=[bufs["kv_indptr_l0"], bufs["kv_indptr_l1"]],
                paged_kv_indices_arr=[
                    bufs["kv_indices_l0"][: max(1, common_prefix_tokens)],
                    bufs["kv_indices_l1"][: max(1, total_unique)],
                ],
                paged_kv_last_page_len=[bufs["last_page_l0"], bufs["last_page_l1"]],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads_local,
                head_dim=self.head_dim_local,
                page_size=self.cascade_page_size,
                causal=False,
                pos_encoding_mode="NONE",
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )
        except Exception as e:
            if self._dbg_enabled:
                logger.warning("CG cascade plan failed at bs=%d: %s", bs, e)
            return False
        return True

    # ------------------------------------------------------------------
    # Forward-metadata hooks
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        # Always run the parent so non-cascade paths (extend, target verify,
        # draft extend, fallback decode) keep working.
        super().init_forward_metadata(forward_batch)

        # Reset both arms; CG hooks set their respective state below.
        self._in_cuda_graph = False
        self._cascade_plan = None
        self._cg_cascade_plan = None

        if not forward_batch.forward_mode.is_decode_or_idle():
            if self._dbg_enabled:
                self._dbg_skip_not_decode += 1
            return

        bs = int(forward_batch.req_pool_indices.shape[0])
        self._dbg_total_decode_steps += 1

        if bs < self.cascade_min_batch_size:
            self._dbg_skip_below_bs += 1
            return

        common = self._detect_common_prefix_tokens(forward_batch, bs)
        if common < self.cascade_min_prefix_tokens:
            self._dbg_skip_below_prefix += 1
            return

        plan = self._build_cascade_plan_args(forward_batch, bs, common)
        if plan is None:
            return
        self._cascade_plan = plan
        self._dbg_cascade_fired += 1
        if self._dbg_enabled:
            logger.info("Cascade fires: bs=%d, common_prefix_tokens=%d", bs, common)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        # Parent allocates cuda_graph_kv_indices etc. for its decode wrappers.
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

        # Size cascade buffers for the worst case in the captured set. We
        # don't know the full cuda_graph_bs list here -- only max_bs. Each
        # per-bs wrapper is allocated lazily in the capture hook.
        self._cg_max_bs = max_bs
        # Worst-case shared prefix length: cap at the model's context window
        # (cascade slot ids must lie within req_to_token's row width).
        self._cg_max_shared_pages = int(self.max_context_len)
        # Worst-case per-request tail length: same context window.
        self._cg_max_pages_per_req = int(self.max_context_len)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """CUDA-graph metadata hook (current out-of-graph API).

        Replaces the deprecated ``init_forward_metadata_{capture,replay}_cuda_graph``
        pair. The decode CUDA-graph runner calls this once per captured batch
        size with ``in_capture=True`` (before recording the graph), and again
        before every ``graph.replay()`` with ``in_capture=False``. Eager decode
        does not reach here -- it uses ``init_forward_metadata`` above.

          * Capture (``in_capture=True``): allocate the per-bs cascade wrapper,
            prime it with a synthetic plan, and arm ``_cg_cascade_plan`` so the
            ``forward_decode`` recorded into the graph invokes the cascade
            wrapper's ``run()`` for this bs.
          * Replay (``in_capture=False``): detect the actual shared prefix and
            refill the pre-allocated buffers in place so the recorded ``run()``
            reads the current step's metadata.
        """
        # Parent sets up its per-request decode wrappers for this bs -- our
        # fallback path, and required for non-decode capture modes.
        super().init_forward_metadata_out_graph(forward_batch, in_capture)

        self._in_cuda_graph = True
        self._cascade_plan = None
        self._cg_cascade_plan = None

        forward_mode = forward_batch.forward_mode
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            return
        if self._cg_disabled:
            self._dbg_skip_in_cg += 1
            return
        if not is_flashinfer_available():
            return

        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens_cpu = forward_batch.seq_lens_cpu

        if bs < self.cascade_min_batch_size:
            # bs too small to ever fire cascade; the parent's per-request
            # decode handles this captured graph.
            self._dbg_skip_in_cg += 1
            return

        if in_capture:
            # Allocate the per-bs cascade wrapper + buffers lazily, then prime
            # it with a synthetic plan (common_prefix=1 + bs unique tails) so
            # the captured run() has valid scheduler state. Capture-time slot
            # ids may point at zero-filled req_to_token rows; that is fine --
            # the captured kernel only records the launch, and replay overwrites
            # the buffers in place.
            self._allocate_cg_cascade_for_bs(bs)
            synth_seq_lens_cpu = (
                seq_lens_cpu
                if seq_lens_cpu is not None
                else forward_batch.seq_lens.cpu()
            )
            ok = self._fill_cg_cascade_plan(bs, req_pool_indices, synth_seq_lens_cpu, 1)
            if ok:
                # Arm so forward_decode takes the cascade path during the
                # capture run; the captured graph then permanently invokes
                # wrapper.run(...) for this bs.
                self._cg_cascade_plan = _CascadePlanState(common_prefix_tokens=1, bs=bs)
            else:
                # Capture-time plan failure: drop cascade for this bs and let
                # the captured graph use the parent's per-request decode.
                self._cg_cascade_wrappers.pop(bs, None)
                self._cg_cascade_buffers.pop(bs, None)
                if self._dbg_enabled:
                    logger.warning(
                        "CG cascade capture-plan failed at bs=%d; falling back "
                        "to parent's per-request decode for this bs.",
                        bs,
                    )
            return

        # ---- Replay: detect actual common prefix + refill buffers in place ----
        if bs not in self._cg_cascade_wrappers:
            # Cascade not captured for this bs (capture-plan failed, or
            # bs < min_batch_size at capture); the parent's path runs.
            self._dbg_skip_in_cg += 1
            return

        common = self._detect_common_prefix_from_rpi(bs, req_pool_indices, seq_lens_cpu)
        # Always plan (even below threshold): the captured graph for this bs has
        # cascade run() recorded, with no mid-graph fallback. Cascade with
        # common=0 is mathematically equivalent to per-request decode plus a
        # no-op level-0 launch, so always arm under CG.
        ok = self._fill_cg_cascade_plan(bs, req_pool_indices, seq_lens_cpu, common)
        if not ok:
            if self._dbg_enabled:
                logger.warning(
                    "CG cascade replay-plan failed at bs=%d, common=%d "
                    "(captured graph may produce incorrect output for this "
                    "step).",
                    bs,
                    common,
                )
            return

        self._cg_cascade_plan = _CascadePlanState(common_prefix_tokens=common, bs=bs)
        self._dbg_cascade_run_cg += 1
        if common >= self.cascade_min_prefix_tokens:
            self._dbg_cascade_fired_cg += 1
            if self._dbg_enabled:
                logger.info(
                    "Cascade fires (CG): bs=%d, common_prefix_tokens=%d",
                    bs,
                    common,
                )
        else:
            self._dbg_skip_below_prefix_cg += 1
            if self._dbg_enabled:
                logger.info(
                    "Cascade ran (CG, below threshold): bs=%d, "
                    "common_prefix_tokens=%d",
                    bs,
                    common,
                )

    # ------------------------------------------------------------------
    # forward_decode (eager + CG dispatch)
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        # Resolve which cascade wrapper (if any) to invoke. Eager path uses
        # ``_cascade_decode_wrapper``; CG path uses the per-bs entry from
        # ``_cg_cascade_wrappers``. The selection is purely based on Python
        # state at call time; under CG capture/replay this resolves once at
        # capture (when ``_in_cuda_graph=True`` and ``_cg_cascade_plan`` is
        # set) and the captured graph then invokes wrapper.run() for that
        # specific wrapper instance every replay.
        if self._in_cuda_graph and self._cg_cascade_plan is not None:
            wrapper = self._cg_cascade_wrappers.get(self._cg_cascade_plan.bs)
        elif (not self._in_cuda_graph) and self._cascade_plan is not None:
            wrapper = self._cascade_decode_wrapper
        else:
            wrapper = None

        if wrapper is None:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        # Cascade arm: write KV-cache for current decode tokens, then run
        # the cascade wrapper.
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # KV pool returns (K, V), each [size+page_size, num_kv_heads, head_dim].
        # Cascade wrapper with page_size=1 expects [num_pages, 1,
        # num_kv_heads, head_dim] per K/V -- add a singleton page dim.
        k_buf, v_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        kv_for_run = (k_buf.unsqueeze(1), v_buf.unsqueeze(1))

        q_3d = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        out = wrapper.run(q_3d, kv_for_run)
        return out.view(-1, layer.tp_q_head_num * layer.head_dim)

    # forward_extend is unchanged from the parent: cascade is decode-only
    # in this initial scope.
