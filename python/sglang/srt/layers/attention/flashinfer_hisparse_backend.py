"""FlashInfer decode backend for Quest + HiSparse.

Decode-only.  For Quest mode the contract is:

  per-step:
    - HiSparseCoordinator.map_last_loc_to_buffer ran in prepare_for_decode
      and updated Quest's running bounds + reserved a hot-buffer slot for
      the upcoming token at each request's out_cache_loc.
    - actual_lens = min(seq_lens, top_k); kv_indptr = cumsum(actual_lens)
      so FlashInfer doesn't see padding duplicates (over-weighting).
    - kv_last_page_len is all-ones (page_size = 1, one token per page).

  per-layer:
    1. set_kv_buffer writes the new K/V to the hot buffer (the pool's
       _resolve_write_loc routes the write to the reserved slot).
    2. quest.retrieve_topk -> (positions, actual_lens) tuple.
    3. coord.swap_in_selected_pages -> top-k device-buffer indices.
    4. Scatter-pack the device indices into the wrapper's kv_indices buffer
       densely (per-request prefix [0, actual_lens[i])).
    5. wrapper.run reads from the buffer and runs sparse attention.

Plan/run separation (so cuda graph capture works):
  * ``init_forward_metadata`` (eager) plans with this step's actual indptr
    via ``fast_decode_plan`` + ``global_override_indptr_cpu`` (no D2H sync).
  * ``init_forward_metadata_capture_cuda_graph`` plans with WORST-CASE
    uniform top_k indptr so the captured plan_info covers any actual
    workload at replay.  Also pre-inits the hisparse swap-in state for
    captured slots (sets req_to_host_pool entries to a safe sentinel so
    the swap-in kernel doesn't OOB on uninitialised host-pool indices).
  * ``init_forward_metadata_replay_cuda_graph`` re-plans with the actual
    indptr for this step (host roundtrip is OK — it runs OUTSIDE the
    captured forward window).

Hardcoded for v1:
  * Single decode wrapper (no SWA / cross-attention dispatch).
  * page_size = 1 (token-level addressing into the device hot buffer).
  * MHA / GQA via :class:`HiSparseMHATokenToKVPool`.
  * Decode only; prefill is the caller's problem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.sparsity.algorithms.quest_scatter_pack_kernel import (
    quest_scatter_pack,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


# Default workspace size for FlashInfer's internal scratch (matches the
# upstream FlashInferAttnBackend default for safety).
_DEFAULT_WORKSPACE_BYTES = 384 * 1024 * 1024


class FlashInferHiSparseDecodeBackend(AttentionBackend):
    """Quest + HiSparse + FlashInfer decode.

    Decode-only backend.  For prefill, pair with a regular FlashInfer prefill
    backend via :class:`HybridAttnBackend` (e.g. ``--prefill-attention-backend
    flashinfer --decode-attention-backend flashinfer_hisparse``).  Calling
    :meth:`forward_extend` directly raises NotImplementedError to make the
    misuse loud.
    """

    def __init__(
        self,
        quest_algorithm: "QuestAlgorithm",
        coordinator: "HiSparseCoordinator",
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_bs: int,
        kv_data_type: torch.dtype,
        q_data_type: torch.dtype,
        device: torch.device,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        self.quest = quest_algorithm
        self.coord = coordinator
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_bs = max_bs
        self.kv_data_type = kv_data_type
        self.q_data_type = q_data_type
        self.device = device
        self.top_k = quest_algorithm.top_k

        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                _DEFAULT_WORKSPACE_BYTES, dtype=torch.uint8, device=device
            )
        self.workspace_buffer = workspace_buffer

        # Pre-allocated buffers for CUDA graph: their pointers are passed
        # to the wrapper at construction.  Per step / per layer we update
        # the contents in place; the wrapper reads them at forward time.
        # kv_indices_buf has +1 slot at the end used as a scatter scratch
        # by the graph-safe scatter-pack (writes for invalid positions
        # land here harmlessly).
        self._kv_indptr_buf = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self._kv_indices_buf = torch.zeros(
            max_bs * self.top_k + 1, dtype=torch.int32, device=device
        )
        self._scatter_scratch_idx = max_bs * self.top_k  # last slot
        # page_size = 1 ⇒ every "last page" has exactly one token.  Filled
        # once at construction; never mutated.
        self._kv_last_page_len_buf = torch.ones(
            max_bs, dtype=torch.int32, device=device
        )

        # Cumulative kv_indptr template — [0, top_k, 2*top_k, ..., max_bs*top_k].
        # Used to seed wrapper construction (worst-case stride) so the
        # initial plan covers max workload.
        self._kv_indptr_template = torch.arange(
            0,
            (max_bs + 1) * self.top_k,
            step=self.top_k,
            dtype=torch.int32,
            device=device,
        )

        # Per-step actual_lens (= min(seq_lens, top_k)), written by init
        # hooks and read per layer for the scatter mask.  Stable buffer
        # so cuda graph sees the same tensor across capture/replay.
        self._actual_lens_buf = torch.zeros(
            max_bs, dtype=torch.int32, device=device,
        )
        # Pre-built [0, top_k) range used in the scatter mask; allocated
        # once to keep forward_decode allocation-free.
        self._range_top_k = torch.arange(
            self.top_k, dtype=torch.int32, device=device,
        )

        # FlashInfer wrapper class kept as an instance attribute so that
        # _get_or_create_wrapper can construct lazily without re-importing.
        self._WrapperCls = BatchDecodeWithPagedKVCacheWrapper

        from flashinfer import fast_decode_plan
        self._fast_decode_plan = fast_decode_plan
        # Capture-time sm_scale + logits_soft_cap (set on first plan).
        self._cached_sm_scale: Optional[float] = None
        self._cached_logits_soft_cap: float = 0.0

        # Per-bs wrapper cache.  Each wrapper is bound to the SAME underlying
        # buffers (sliced to its own bs); ``use_cuda_graph=True`` so the
        # wrapper reads ``_kv_indices_buf`` at forward time (per layer) rather
        # than caching its contents at begin_forward time.  This is what makes
        # per-layer Quest selection compatible with a single pre-planned
        # wrapper, in both eager and captured-graph contexts.
        self._wrappers: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}

        self._wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None
        self._current_bs: Optional[int] = None

    # ----------------------------------------------------- production ctor

    @classmethod
    def from_runner(cls, runner: "ModelRunner") -> "FlashInferHiSparseDecodeBackend":
        """Build the backend from a ModelRunner.

        Pulls Quest + coordinator + dimensions off the runner.  The runner
        must have ``hisparse_coordinator`` populated in quest mode (set up
        in ``ModelRunner.__init__`` when ``--enable-hisparse`` and
        ``hisparse_config algorithm='quest'`` are configured).
        """
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        from sglang.srt.managers.hisparse_coordinator import MODE_QUEST

        coord = runner.hisparse_coordinator
        if coord is None or coord.mode != MODE_QUEST:
            raise RuntimeError(
                "flashinfer_hisparse attention backend requires the coordinator "
                "to be in 'quest' mode; check --enable-hisparse + "
                "--hisparse-config algorithm='quest'."
            )
        quest = coord.quest_algorithm

        num_qo_heads = (
            runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        num_kv_heads = runner.model_config.get_num_kv_heads(get_attention_tp_size())
        head_dim = runner.model_config.head_dim
        # Use the runner's max running batch size as our upper bound.  Lazy
        # wrapper construction in _get_or_create_wrapper handles smaller bs
        # values on demand.
        max_bs = runner.req_to_token_pool.req_to_token.shape[0]

        return cls(
            quest_algorithm=quest,
            coordinator=coord,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_bs=max_bs,
            kv_data_type=runner.kv_cache_dtype,
            q_data_type=runner.dtype,
            device=torch.device(runner.device),
        )

    # ------------------------------------------------------ wrapper cache

    def _get_or_create_wrapper(self, bs: int):
        """Return (constructing if needed) the wrapper bound to ``bs`` slices.

        Wrappers pre-plan with worst-case uniform top_k stride at first
        creation (sets up the wrapper's _cached_module so subsequent
        fast_decode_plan calls work).
        """
        if bs > self.max_bs:
            raise RuntimeError(
                f"FlashInferHiSparseDecodeBackend received bs={bs} > max_bs={self.max_bs}"
            )
        if bs in self._wrappers:
            return self._wrappers[bs]

        kv_indptr_slice = self._kv_indptr_buf[: bs + 1]
        kv_indices_slice = self._kv_indices_buf[: bs * self.top_k]
        kv_last_page_len_slice = self._kv_last_page_len_buf[:bs]
        # Stamp the cumulative-top_k pattern so the constructor's plan is
        # for max workload (any actual workload at runtime is ≤ this).
        kv_indptr_slice.copy_(self._kv_indptr_template[: bs + 1])

        wrapper = self._WrapperCls(
            self.workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            # use_tensor_cores enables the prefill-style kernel which
            # supports more (group_size, head_dim) combinations than the
            # default decode kernel — important for GQA models with odd
            # group sizes (e.g., Qwen2.5: 14 q-heads / 2 kv-heads = 7).
            use_tensor_cores=True,
            paged_kv_indptr_buffer=kv_indptr_slice,
            paged_kv_indices_buffer=kv_indices_slice,
            paged_kv_last_page_len_buffer=kv_last_page_len_slice,
        )
        wrapper.plan(
            indptr=kv_indptr_slice,
            indices=kv_indices_slice,
            last_page_len=kv_last_page_len_slice,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            sm_scale=1.0 / (self.head_dim ** 0.5),
            data_type=self.kv_data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
        )
        self._wrappers[bs] = wrapper
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
        """Stamp wrapper buffers + call FlashInfer's plan().

        Eager step / capture / replay all use this; ``worst_case=True`` plans
        for uniform top_k stride (used at capture so the captured plan_info
        is valid for any actual workload at replay).
        """
        if worst_case:
            actual_lens_cpu = torch.full(
                (bs,), self.top_k, dtype=torch.int32,
            )
        else:
            if seq_lens_cpu is None:
                seq_lens_cpu = seq_lens.cpu()
            actual_lens_cpu = seq_lens_cpu.to(torch.int32).clamp_max(self.top_k)

        indptr_cpu = torch.zeros(bs + 1, dtype=torch.int32)
        indptr_cpu[1:] = torch.cumsum(actual_lens_cpu, dim=0).to(torch.int32)

        # Mirror to device buffers (blocking writes — outside any captured window).
        self._actual_lens_buf[:bs].copy_(actual_lens_cpu, non_blocking=False)
        self._kv_indptr_buf[: bs + 1].copy_(indptr_cpu, non_blocking=False)

        self._cached_sm_scale = sm_scale
        self._cached_logits_soft_cap = logits_soft_cap

        self._fast_decode_plan(
            self._wrapper,
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

    def prepare_wrappers_for_bs(self, bs_list) -> None:
        """Pre-create wrappers for the given bs values.

        Use before CUDA graph capture to keep allocations out of the captured
        region (FlashInfer planning may allocate small device tensors).
        """
        for bs in bs_list:
            self._get_or_create_wrapper(int(bs))

    # --------------------------------------------------------- per-step API

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Eager (non-graph) per-step prep.

        Picks the wrapper for current bs and re-plans with this step's
        actual indptr (optimal scheduling for the actual workload).
        """
        bs = int(forward_batch.batch_size)
        self._wrapper = self._get_or_create_wrapper(bs)
        self._current_bs = bs
        # Quest prefill bounds may still be in flight on the coordinator's
        # bounds_stream (decoupled from staging admission).  Wait on the
        # pending event before retrieve_topk reads page_k_bounds.  No-op
        # after the first decode step for each request.
        self.coord.wait_for_pending_bounds(forward_batch.req_pool_indices)
        # Cache step-level state once (used by every layer's retrieve_topk).
        self.quest.prepare_step(forward_batch.seq_lens)
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

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """AttentionBackend hook: pre-capture state init.

        Per-bs wrappers are created lazily by ``_get_or_create_wrapper``,
        so this is a no-op except for an upper-bound check.
        """
        if max_bs > self.max_bs:
            raise RuntimeError(
                f"cuda graph max_bs={max_bs} exceeds backend max_bs={self.max_bs}; "
                f"increase max_bs at construction"
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
        """CUDA-graph capture: pre-init swap-in state + plan with worst case.

        The swap-in kernel reads ``coord.req_to_host_pool[req_pool_idx]`` —
        if those slots haven't been admitted yet (default -1), the kernel
        will OOB into host memory.  Pre-init touched slots to 0 so the
        kernel reads a valid (zeroed) host position.  Real admits later
        overwrite these entries.
        """
        bs_int = int(bs)
        self._wrapper = self._get_or_create_wrapper(bs_int)
        self._current_bs = bs_int

        # Pre-init swap-in dependencies for the slots this capture touches.
        # req_pool_indices values are typically zeros at capture time; set
        # those slots' host-pool pointer to 0 (a valid, in-bounds index).
        # The lookups go to host_cache[0..top_k) which is allocated and
        # zero-filled — semantically meaningless K/V, but graph-safe.
        #
        # Also set num_real_reqs[0] = bs so the swap-in kernel actually
        # processes the captured reqs (default is 0, which would skip them
        # all and leave top_k_indices at the kernel's pre-fill sentinel
        # value -1; those -1s would then flow into kv_indices_buf and
        # OOB FlashInfer's pool read).  ModelRunner overwrites num_real_reqs
        # per step at runtime, so this is just for capture-time correctness.
        with torch.no_grad():
            unique_slots = req_pool_indices[:bs_int].long().unique()
            self.coord.req_to_host_pool[unique_slots] = 0
            self.coord.req_to_device_buffer[unique_slots] = 0
            self.coord.num_real_reqs[0] = bs_int

        # Cache step-level state with placeholder seq_lens for capture.
        self.quest.prepare_step(seq_lens[:bs_int])

        # Plan with WORST-CASE indptr (uniform top_k per request) so the
        # captured plan_info is valid for any actual indptr at replay.
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
        """CUDA-graph replay: switch to the captured wrapper + re-plan
        with this step's actual indptr (host roundtrip is fine — runs
        OUTSIDE the captured forward window)."""
        bs_int = int(bs)
        if bs_int not in self._wrappers:
            raise RuntimeError(
                f"replay called for bs={bs_int} but no wrapper was captured; call "
                f"init_forward_metadata_capture_cuda_graph first"
            )
        self._wrapper = self._wrappers[bs_int]
        self._current_bs = bs_int
        # Wait on any pending Quest prefill bounds before retrieve_topk runs
        # (bounds compute is decoupled from staging admission and may still
        # be in flight on coord.quest_bounds_stream).  Same as eager path.
        self.coord.wait_for_pending_bounds(req_pool_indices[:bs_int])
        # Cache step-level state once for this replay.
        self.quest.prepare_step(seq_lens[:bs_int])
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

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """AttentionBackend hook: padded seq_lens get this value.

        Padded reqs (bs > num_real_reqs) won't be selected by the swap-in
        kernel anyway (gated on ``num_real_reqs``).  Use 1 to keep all
        downstream arithmetic well-defined (e.g. seq_lens // page_size).
        """
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Quest + HiSparse + FlashInfer per-layer decode.

        Steps (in order):
          1. set_kv_buffer: write new K/V (pool redirects to reserved hot slot).
          2. quest.retrieve_topk: top-k token positions for this layer.
          3. coord.swap_in_selected_pages: map positions → device buffer indices.
          4. Copy result into _kv_indices_buf for this batch.
          5. wrapper.forward: run sparse attention and return.
        """
        if self._current_bs is None:
            raise RuntimeError(
                "init_forward_metadata must be called before forward_decode"
            )
        bs = self._current_bs

        # 1. Save new K/V to the hot buffer.  HiSparseMHATokenToKVPool's
        #    _resolve_write_loc reroutes loc → reserved hot-buffer slot.
        if save_kv_cache and k is not None:
            assert v is not None
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                getattr(layer, "k_scale", None),
                getattr(layer, "v_scale", None),
            )

        # 2. Quest retrieval (positions + actual_lens).  actual_lens is
        #    cached on _actual_lens_buf by init_forward_metadata; we reuse
        #    that for the scatter mask.
        topk_token_positions, _ = self.quest.retrieve_topk(
            queries=q,
            layer_id=layer.layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
        )

        # Lazy-cache layer scaling on first call so eager-mode init can
        # plan with the right values.
        if self._cached_sm_scale is None:
            self._cached_sm_scale = layer.scaling
            self._cached_logits_soft_cap = getattr(layer, "logit_cap", 0.0) or 0.0

        # 3. Materialize host pages into the device buffer; returns
        #    [bs, top_k] device-buffer indices.  Some indices may be
        #    "wasted" beyond actual_lens; we don't read those slots.
        page_table = self.coord.swap_in_selected_pages(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            top_k_result=topk_token_positions,
            layer_id=layer.layer_id,
        )

        # 4. Graph-safe scatter-pack page_table into _kv_indices_buf based
        #    on actual_lens.  Fused Triton kernel replaces the previous
        #    PyTorch op chain (clamp + arange + cmp + add + where +
        #    scatter_); same correctness (clamp, scratch slot for invalid
        #    positions, packed prefix per request) but one launch instead
        #    of ~7 small ops per layer × num_layers per step.
        quest_scatter_pack(
            page_table=page_table,
            actual_lens=self._actual_lens_buf[:bs],
            kv_indptr=self._kv_indptr_buf[:bs],
            kv_indices_buf=self._kv_indices_buf,
            scratch_idx=self._scatter_scratch_idx,
            top_k=self.top_k,
        )

        # 5. FlashInfer forward.
        kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # MHA: get_kv_buffer returns (k_buf, v_buf).  page_size=1 ⇒ each
        # token is its own page; reshape to (num_pages, page_size=1, heads, dim).
        if isinstance(kv_buffer, tuple):
            raw_k, raw_v = kv_buffer
            kv_paged = (
                raw_k.view(-1, 1, self.num_kv_heads, self.head_dim),
                raw_v.view(-1, 1, self.num_kv_heads, self.head_dim),
            )
        else:
            # Combined-KV pool: leave as-is.  Currently unused (Quest mode
            # is hardcoded MHA), included for forward-compatibility.
            kv_paged = kv_buffer

        o = self._wrapper.run(
            q.contiguous().view(-1, self.num_qo_heads, self.head_dim),
            kv_paged,
            k_scale=getattr(layer, "k_scale_float", None),
            v_scale=getattr(layer, "v_scale_float", None),
        )
        return o.view(-1, self.num_qo_heads * self.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Decode-only backend: prefill must be served by a separate backend."""
        raise NotImplementedError(
            "flashinfer_hisparse is decode-only.  Pair with a regular flashinfer "
            "prefill backend via HybridAttnBackend, e.g.:\n"
            "  --prefill-attention-backend flashinfer "
            "--decode-attention-backend flashinfer_hisparse"
        )
