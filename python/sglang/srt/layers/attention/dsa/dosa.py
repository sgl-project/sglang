"""Exact Few prefill over one request's paged FP8 index cache.

Few owns one serial workspace per device/head geometry, including private graphs,
stock-to-Few priors and per-layer carry. Caller tensors are borrowed only until
queued work completes. Capacities affect work and exact recovery, never selection.
"""

from collections import OrderedDict
from types import SimpleNamespace
import math
import os
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode

SEED = 12288
TOPK = 2048
RECENT = 1536
FEW_RECORD_CAPACITY = 114688
FEW_PREFILL_MIN_KEYS = 131072
MIN_FEW_QUERY_CAPACITY = 4096
MAX_FEW_QUERY_CAPACITY = 32768


def causal_id_page_bound(rows, keys, seed=SEED):
    # One common copy plus at most three disjoint solo copies beyond the
    # smallest endpoint. Five streams add at most four partial pages per Q4.
    return (rows // 4) * ((keys - seed + 2 * (rows - 1) + 127) // 128 + 4)


def few_id_pool_pages(rows, keys):
    return max(256 * rows, causal_id_page_bound(rows, keys))


def service_query_storage(logical_rows, keys, score_heads=32):
    rows = min(logical_rows, 16384)
    page_bytes = (128 + 2 + 2) * 4
    budget = rows * (65536 * 5 + 256 * page_bytes)
    while rows * FEW_RECORD_CAPACITY * 5 + few_id_pool_pages(rows, keys) * page_bytes > budget:
        rows = (rows // 16) * 8
    return rows, few_id_pool_pages(rows, keys)


def query_slabs(rows, capacity, recent_rows=RECENT):
    carry_begin = max(0, rows - recent_rows)
    for begin in range(0, rows, capacity):
        end = min(begin + capacity, rows)
        yield begin, end, (end - begin + 7) // 8 * 8, max(0, end - max(begin, carry_begin))


def valid_paged_inputs(device, q, w, cache, table64, table1, starts, ends, result, *, score_heads=32):
    if q.ndim != 3 or w.ndim != 2:
        return False
    rows = len(q)
    if any(value.device != device for value in (q, w, cache, table64, table1, starts, ends, result)):
        return False
    if (q.dtype != torch.float8_e4m3fn or tuple(q.shape) != (rows, score_heads, 128)
            or w.dtype != torch.float32 or tuple(w.shape) != (rows, score_heads)
            or not q.is_contiguous() or not w.is_contiguous()
            or cache.dtype != torch.uint8 or not cache.is_contiguous()
            or cache.data_ptr() % 16 != 0
            or cache.ndim not in (2, 3) or cache.shape[0] == 0
            or tuple(cache.shape[1:]) not in ((8448,), (64, 132))
            or any(value.dtype != torch.int32 for value in (table64, table1, starts, ends, result))):
        return False
    return (table64.ndim == table1.ndim == 2 and len(table64) == len(table1) == 1
            and table64.is_contiguous() and table1.is_contiguous()
            and starts.ndim == ends.ndim == 1 and starts.numel() == ends.numel() == rows
            and result.ndim == 2 and result.is_contiguous()
            and result.shape[0] >= rows and result.shape[1] == TOPK)



def topk_into(scores, offsets, lengths, output):
    from flashinfer.topk import get_topk_module
    from flashinfer.utils import _get_cache_buf

    row_states = _get_cache_buf(f"radix_topk_row_states_{scores.device}", 1024 * 1024,
                                scores.device, zero_init=True)
    get_topk_module().radix_topk_ragged_transform(
        scores, output, offsets, lengths, row_states, TOPK, False, 0, True, row_starts=None)
    return output


class _SeedOutput(TorchDispatchMode):
    """Bind DG's padded output to explicitly reserved storage in this thread."""

    def __init__(self, output):
        super().__init__()
        self.output = output
        self.matches = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.empty.memory_format:
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda" and kwargs.get("dtype") == torch.float32:
                if tuple(args[0]) != tuple(self.output.shape) or self.matches:
                    raise RuntimeError("DeepGEMM padded-output allocation contract changed")
                self.matches += 1
                return self.output
        return func(*args, **kwargs)


class _LeafSelection:
    """The original exact leaf hierarchy and unordered FI, with owned outputs."""

    def __init__(self, rows, keys, device):
        import triton

        self.chunks = triton.cdiv(keys, 32768)
        for name, shape, dtype in (
            ("rows", (8,), torch.int32), ("state", (1,), torch.int32),
            ("finite", (8,), torch.int32), ("lengths", (8 * self.chunks,), torch.int32),
            ("merge_lengths", (8,), torch.int32), ("fallback_lengths", (rows,), torch.int32),
            ("leaf", (8 * self.chunks, 32768), torch.float32),
            ("merge", (8, self.chunks * TOPK), torch.float32),
            ("ids", (8, self.chunks * TOPK), torch.int32),
            ("leaf_selected", (8 * self.chunks, TOPK), torch.int32),
            ("merge_selected", (8, TOPK), torch.int32),
            ("full_selected", (rows, TOPK), torch.int32),
        ):
            setattr(self, name, torch.empty(shape, device=device, dtype=dtype))
        self.leaf_zero = torch.zeros_like(self.lengths)
        self.merge_zero = torch.zeros_like(self.merge_lengths)
        self.full_zero = torch.zeros(rows, device=device, dtype=torch.int32)

    def __call__(self, owner, begin, end):
        n = end - begin
        dense, counts = owner.replay_scores[:n], owner.replay_counts[:n]
        status = owner.error[begin:end]
        owner.modules["leaf"].replay_and_leaf_capacity(
            owner.q[begin:end].view(torch.float8_e4m3fn), owner.pk, owner.ps,
            owner.w[begin:end], owner.starts[begin:end], owner.ends[begin:end],
            status, dense, counts, self.leaf, self.rows, self.state, self.finite,
            self.lengths, owner.live_nk)
        leaf_ids = topk_into(self.leaf, self.leaf_zero, self.lengths, self.leaf_selected)
        core = owner.modules["hierarchy"]
        core.gather(self.leaf, leaf_ids, self.lengths, self.rows, self.state,
                    self.finite, self.merge, self.ids)
        core.final_meta(counts, self.rows, self.state, self.finite,
                        self.merge_lengths, self.fallback_lengths[:n], self.chunks)
        selected = topk_into(self.merge, self.merge_zero, self.merge_lengths, self.merge_selected)
        output = topk_into(dense, self.full_zero[:n], self.fallback_lengths[:n], self.full_selected[:n])
        core.publish(self.ids, selected, self.rows, self.state, self.finite, output)
        owner.modules["publisher"].recovered(output, owner.order[begin:end], status, owner.output)
        owner.replay_work[begin:end].copy_(counts)


class FewBackend:
    """One explicit owner for workspace, graph, request and carry lifetimes.

    H32/H64, D128, FP8 Q/K, F32 weights and top2048 use the same Few algorithm.
    Logical Q is split into bounded scratch blocks; the service supplies a causal
    interval to let completed blocks improve the next block's prior. Other valid
    intervals share one frozen prior and gather across their complete call.
    """

    supports_causal_query_interval = True

    def __init__(self, device, *, max_queries, max_keys, score_heads=32, modules=None):
        if (score_heads not in (32, 64) or not MIN_FEW_QUERY_CAPACITY <= max_queries <= MAX_FEW_QUERY_CAPACITY
                or max_queries % 8 or not 196608 <= max_keys <= 1048576):
            raise ValueError("Few requires H32/H64, Q8 capacity4096..32768 and K196608..1048576")
        import triton
        from . import dosa_ops
        from .dosa_build import load_modules
        from .dosa_graph import GraphPool

        self.triton, self.ops = triton, dosa_ops
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.score_heads = score_heads
        self.logical_query_capacity = max_queries
        self.max_q, self.max_pages = service_query_storage(max_queries, max_keys, score_heads)
        self.max_k, self.record_capacity = max_keys, FEW_RECORD_CAPACITY
        self.high_paid = 28 if score_heads == 64 else 20
        self.modules = load_modules(score_heads) if modules is None else modules
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.lock = threading.RLock()
        self.last_done = self.startup_ready = None
        self.shape = None
        self.hot_graph = self.cold_graph = self.cold_dispatch = None
        self.h22_graphs = None
        self.shape_graphs = OrderedDict()
        self.graph_pool = GraphPool(self.device)
        self.replay_calls = 0
        self.layers = {}
        self.side_stream = torch.cuda.Stream(device=self.device)
        self.bootstrap = SimpleNamespace(B=4, calls=0)
        self._allocate()

    def _empty(self, *shape, dtype=torch.float32):
        return torch.empty(shape, device=self.device, dtype=dtype)

    def _allocate(self):
        q, k, h, pages, cap = self.max_q, self.max_k, self.score_heads, self.max_pages, self.record_capacity
        self.backing = {}
        for name, shape, dtype in (
            ("q", (q, h, 128), torch.uint8), ("w", (q, h), torch.float32),
            ("starts", (q,), torch.int32), ("ends", (q,), torch.int32),
            ("budget8", (q // 8,), torch.int32), ("budget4", (q // 4,), torch.int32),
            ("capture", (q, 129), torch.int64 if h == 64 else torch.int32),
            ("sample_scores", (q, 128), torch.float32),
            ("sample_masks", (q, 128), torch.int64 if h == 64 else torch.int32),
            ("safe_tau", (q,), torch.float32), ("seed_tau", (q,), torch.float32),
            ("error", (q,), torch.int32), ("lengths", (q,), torch.int32),
            ("seed_diag", (q, 4), torch.int32), ("values", (q, cap), torch.uint8),
            ("ids", (q, cap), torch.int32), ("counts", (q,), torch.int32),
            ("solo_counts", (q,), torch.int32), ("common_counts", (q // 4,), torch.int32),
            ("order", (q,), torch.int64), ("ordered_output", (q, TOPK), torch.int32),
            ("output", (q, TOPK), torch.int32), ("counts_original", (q,), torch.int32),
            ("error_original", (q,), torch.int32), ("error_before_completion", (q,), torch.int32),
            ("replay_work", (q,), torch.int32),
            ("select_diag", (q, 2), torch.int32),
            ("boundary_ids", (q, 4096), torch.int32), ("boundary_counts", (q,), torch.int32),
            ("temporary", (q, 8192), torch.float32), ("temporary_ids", (q, 8192), torch.int32),
        ):
            self.backing[name] = self._empty(*shape, dtype=dtype)
        self.zero = torch.zeros(q, device=self.device, dtype=torch.int32)
        self.backing["map_status"] = self.zero  # Read-only mapping status after exact recovery.
        self.seed_end = torch.full((q,), SEED, device=self.device, dtype=torch.int32)
        self.epsilon = torch.zeros(q, device=self.device, dtype=torch.float32)
        self.q_input = self._empty(q, h, 128, dtype=torch.float8_e4m3fn)
        self.w_input = self._empty(q, h)
        self.start_input = self._empty(q, dtype=torch.int32)
        self.end_input = self._empty(q, dtype=torch.int32)
        self.pk = self._empty(k, 128, dtype=torch.float8_e4m3fn)
        self.ps_bytes = self._empty(k, 4, dtype=torch.uint8)
        self.ps = self.ps_bytes.view(torch.float32).view(-1)
        self.rho = self._empty(k, dtype=torch.int32)
        self.live_nk = self._empty(1, dtype=torch.int32)
        self.sample_ids = 32 + torch.arange(128, device=self.device, dtype=torch.int64) * 64
        self.sample_k = self._empty(128, 128, dtype=torch.uint8)
        self.sample_scale = self._empty(128)
        self.plans_backing = {}
        for paid in (14, 16, self.high_paid):
            self.plans_backing[paid] = [
                self._empty(q, paid + 2, 128, dtype=torch.uint8), self._empty(q, paid + 2),
                self._empty(q, 64 if paid <= 16 else 96), self._empty(q), self._empty(q),
                self._empty(q, dtype=torch.int32),
            ]
        self.keep_backing = self._empty(q // 4, self.triton.cdiv(k - SEED, 32), 4, dtype=torch.int32)
        self.page_backing = self._empty(q + q // 4, self.triton.cdiv(k, 128), dtype=torch.int32)
        self.id_pool = self._empty(pages, 128, dtype=torch.int32)
        self.solo_tasks = self._empty(pages, 2, dtype=torch.int32)
        self.common_tasks = self._empty(pages, 2, dtype=torch.int32)
        for name in ("pool_head", "pool_status", "solo_n", "common_n", "cold_runs"):
            setattr(self, name, torch.zeros(1, device=self.device, dtype=torch.int32))
        self.schedule = {
            paid: (self._empty(q // 8, dtype=torch.int32), self._empty(q // 8, dtype=torch.int32),
                   self._empty(2, dtype=torch.int32)) for paid in (14, 16)
        }
        words = self.modules["audit"].packed_record_words()
        self.packed_record = self._empty(q // 4, words, dtype=torch.int32)
        self.fast_budget = self._empty(q // 4, dtype=torch.int32)
        self.slow_budget = self._empty(q // 4, dtype=torch.int32)
        workers = min(self.sms, self.triton.cdiv(k, 1024))
        self.key_max = self._empty(workers)
        self.key_error = self._empty(workers, dtype=torch.int32)
        self.scale_max = self._empty(1)
        self.scale_error = self._empty(1, dtype=torch.int32)
        self.seed_logits = self.seed_selected = None
        self.replay_scores = self._empty(min(q, 512), k)
        self.replay_counts = self._empty(min(q, 512), dtype=torch.int32)
        self.leaf = _LeafSelection(min(q, 512), k, self.device)
        self.plan_epoch = 0
        self.hot_epoch = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.permutation = torch.arange(k, device=self.device, dtype=torch.int32)
        self.swap_a = self._empty(SEED, dtype=torch.int32)
        self.swap_b = self._empty(SEED, dtype=torch.int32)
        self.plan_counts = torch.zeros(4, device=self.device, dtype=torch.int32)
        self.cold_hot = self._empty(SEED, dtype=torch.int64)
        self.cold_k = self._empty(k, 128, dtype=torch.uint8)
        self.cold_scale = self._empty(k)
        self.cold_scores = self._empty(4, k)
        self.cold_masks = self._empty(4, k, dtype=torch.int64 if h == 64 else torch.int32)
        self.cold_q = self._empty(4, h, 128, dtype=torch.uint8)
        self.cold_w = self._empty(4, h)
        self.cold_zero = torch.zeros(4, device=self.device, dtype=torch.int32)
        self.cold_ends = self._empty(4, dtype=torch.int32)
        self.cold_votes = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.cold_partial = self._empty(128, 5, dtype=torch.int16)
        self.cold_state = torch.zeros(136, device=self.device, dtype=torch.int32)
        self.carry_partial = self._empty(128, RECENT + 1, dtype=torch.int16)
        self.carry_state = torch.zeros(136, device=self.device, dtype=torch.int32)
        self.saved_hot = self._empty(SEED, dtype=torch.int64)
        self.restore_event = torch.cuda.Event()
        self.carry_rows = self._empty(RECENT, TOPK, dtype=torch.int32)
        self.identity = torch.arange(k, device=self.device, dtype=torch.int32)
        self.repair_status = torch.zeros(RECENT, device=self.device, dtype=torch.int32)
        self.startup_ready = torch.cuda.Event()
        self.startup_ready.record(torch.cuda.current_stream(self.device))

    def reserve_for_model_load(self, layer_id):
        from flashinfer.topk import get_topk_module
        from flashinfer.utils import _get_cache_buf

        with self.lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.startup_ready)
            if self.seed_logits is None:
                get_topk_module()
                _get_cache_buf(f"radix_topk_row_states_{self.device}", 1024 * 1024,
                               self.device, zero_init=True)
                self.seed_logits = self._empty(self.max_q, SEED + 256)
                self.seed_selected = self._empty(self.max_q, TOPK, dtype=torch.int32)
            if layer_id not in self.layers:
                self.layers[layer_id] = dict(
                    request=None, prior=None, pending=None,
                    votes=torch.zeros(self.max_k, device=self.device, dtype=torch.int32),
                    hot=self._empty(SEED, dtype=torch.int64), ready=torch.cuda.Event(),
                    free=None, gpu_status=self._empty(1, dtype=torch.int32),
                    host_status=torch.empty(1, dtype=torch.int32, device="cpu", pin_memory=True))
            self.startup_ready.record(stream)

    def release_request(self, hot_key):
        """Retire a request's hints while retaining its layer's reserved workspace."""
        with self.lock:
            entry = self.layers.get(hot_key[0])
            if entry is not None and entry["request"] == hot_key[1]:
                entry["request"] = entry["prior"] = entry["pending"] = None

    def _request(self, hot_key):
        entry = self.layers[hot_key[0]]
        if entry["request"] != hot_key[1]:
            entry["request"] = hot_key[1]
            entry["prior"] = entry["pending"] = None
        return entry

    def _votes(self, entry, keys, stream):
        if entry["free"] is not None:
            stream.wait_event(entry["free"])
        return entry["votes"][:keys]

    def _publish_carry(self, entry, votes, keys, recent):
        self.side_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.side_stream):
            votes.record_stream(self.side_stream)
            self.modules["caller"].carry_votes_topk_reset_(
                votes, entry["hot"], self.carry_partial, self.carry_state, SEED, recent, 0)
            entry["ready"].record(self.side_stream)
        entry["free"] = entry["ready"]
        entry["prior"] = (entry["hot"], keys, entry["ready"], 0)

    def _cold_prior(self, q, w, cache, table64, nk, common_end):
        """Compute the original four evenly spread anchors of the current call."""
        import flashinfer

        self.ops._spread_queries[(4,)](q.view(torch.uint8), w, self.cold_q, self.cold_w,
                                  len(q), self.score_heads, num_warps=4)
        self.ops._identity_gather[(self.triton.cdiv(nk, 64),)](
            cache, cache.view(torch.float32), table64, self.cold_k, self.cold_scale,
            nk, NPAGES=cache.shape[0], num_warps=4)
        scores = self.cold_scores.view(-1)[:4 * nk].view(4, nk)
        masks = self.cold_masks.view(-1)[:4 * nk].view(4, nk)
        self.modules["sample"].seed_score(
            self.cold_q.view(torch.float8_e4m3fn), self.cold_k[:nk].view(torch.float8_e4m3fn),
            self.cold_scale[:nk], self.cold_w, scores, masks, self.sms)
        self.cold_ends.fill_(common_end)
        ids = flashinfer.top_k_ragged_transform(scores, self.cold_zero, self.cold_ends, TOPK,
                                                deterministic=False, dsa_graph_safe=True)
        votes = self.cold_votes[:common_end]
        votes.zero_()
        self.ops._add_votes[(self.triton.cdiv(4 * TOPK, 256),)](ids, votes, 4 * TOPK, common_end, 256)
        self.modules["caller"].carry_votes_topk_reset_(
            votes, self.cold_hot, self.cold_partial, self.cold_state, SEED, 4, 0)
        self.bootstrap.calls += 1
        return self.cold_hot, common_end, None, 0

    def _gather(self, entry, q, w, cache, table64, nk, common_end, window_start):
        pending = entry["pending"]
        if pending is not None:
            hot, visible, ready, lower = pending
            ready.synchronize()
            if int(entry["host_status"][0]) == 0 and visible <= common_end and lower >= window_start:
                entry["prior"] = pending
            entry["pending"] = None
        prior = entry["prior"]
        if prior is None or prior[1] > common_end or prior[3] < window_start:
            prior = self._cold_prior(q, w, cache, table64, nk, common_end)
            entry["prior"] = prior
        stream = torch.cuda.current_stream(self.device)
        if prior[2] is not None:
            stream.wait_event(prior[2])
        prior[0].record_stream(stream)
        if self.plan_epoch + 1 >= 1 << 30:
            self.hot_epoch.zero_()
            self.plan_epoch = 0
        self.plan_epoch += 1
        self.modules["caller"].plan_and_permuted_paged_gather_rho_out(
            prior[0], self.hot_epoch[:nk], self.permutation[:nk], self.swap_a, self.swap_b,
            self.plan_counts, window_start, common_end, self.plan_epoch,
            cache.view(cache.shape[0], 64, 132), self.pk[:nk].view(torch.uint8),
            self.ps_bytes[:nk], table64, self.rho[:nk])

    def bind_shape(self, rows, nk):
        if self.shape == rows:
            return
        if self.last_done is not None:
            self.last_done.synchronize()
        self.hot_graph = self.cold_graph = self.cold_dispatch = None
        self.h22_graphs = None
        for name, tensor in self.backing.items():
            size = rows // 8 if name == "budget8" else rows // 4 if name in ("budget4", "common_counts") else rows
            setattr(self, name, tensor[:size])
        self.rows = rows
        self.keep_words = self.keep_backing.view(-1)[:rows // 4 * self.triton.cdiv(self.max_k - SEED, 32) * 4].view(
            rows // 4, self.triton.cdiv(self.max_k - SEED, 32), 4)
        self.page_table = self.page_backing.view(-1)[:(rows + rows // 4) * self.triton.cdiv(self.max_k, 128)].view(
            rows + rows // 4, self.triton.cdiv(self.max_k, 128))
        self.plans = {paid: [value[:rows] for value in fields] for paid, fields in self.plans_backing.items()}
        self.shape = rows
        cached = self.shape_graphs.pop(rows, None)
        if cached is not None:
            self.hot_graph, self.cold_graph, self.cold_dispatch, self.h22_graphs = cached
            self.shape_graphs[rows] = cached
        elif len(self.shape_graphs) >= 2:
            self.shape_graphs.popitem(last=False)

    def _seed(self, q):
        import deep_gemm

        padded = self.seed_logits[:self.rows]
        mode = _SeedOutput(padded)
        with mode:
            scores = deep_gemm.fp8_fp4_mqa_logits(
                (q, None), (self.pk[:SEED], self.ps[:SEED]), self.w,
                self.zero[:self.rows], self.seed_end[:self.rows], clean_logits=False)
        if mode.matches != 1 or scores.data_ptr() != padded.data_ptr() or scores.stride() != (SEED + 256, 1):
            raise RuntimeError("DeepGEMM must return the reserved padded seed output")
        seed = self.modules["seed"]
        seed.prepare5(scores, self.values, self.ids, self.counts, self.seed_tau,
                      self.error, self.lengths, self.seed_diag)
        selected = topk_into(padded, self.zero[:self.rows], self.lengths, self.seed_selected[:self.rows])
        seed.finish5(scores, selected, self.values, self.ids, self.counts, self.seed_tau,
                     self.error, self.lengths, self.seed_diag)

    def _metadata(self, q):
        workers = len(self.key_max)
        self.ops.key_parts_live[(workers,)](
            self.ps, self.rho, self.key_max, self.key_error, self.live_nk,
            self.max_k, SEED, workers, 1024, num_warps=4)
        self.ops.key_finish[(1,)](self.key_max, self.key_error, self.scale_max, self.scale_error,
                             workers, self.triton.next_power_of_2(workers), num_warps=4)
        dot = math.nextafter(128 * 2.0**-23 / (1 - 128 * 2.0**-23), math.inf)
        steps = self.score_heads // 4 + 3
        g = math.nextafter(steps * 2.0**-24 / (1 - steps * 2.0**-24), math.inf)
        self.ops.query_check[(self.rows,)](
            self.w, self.plans[14][5], self.plans[16][5], self.plans[self.high_paid][5],
            self.budget8, self.scale_max, self.scale_error, self.seed_diag, self.seed_tau,
            self.error, self.safe_tau, 8, dot, g, True, self.score_heads, self.high_paid,
            num_warps=1, enable_fp_fusion=False)

    def _emit(self, paid, x, streams):
        m = self.modules
        args = (paid, x, streams, self.id_pool, self.page_table, self.pool_head,
                self.pool_status, self.error, SEED, self.keep_words)
        if paid in (14, 16):
            ids, inverse, config = self.schedule[paid]
            groups = self.rows // 8
            grid = (2 if paid == 14 else 1) * max(groups, 8 * self.sms)
            self.ops.prepare_live[(1,)](
                self.budget4, self.live_nk, ids[:groups], inverse[:groups], config,
                groups, grid, self.sms, self.max_k, SEED, 256 if paid == 14 else 128,
                paid, self.triton.next_power_of_2(groups), num_warps=4)
            method = m["h16"].emit_h16_indexed_trusted_live_k if paid == 14 else m["h18"].emit_h18_indexed_trusted_live_k
            method(*args, ids[:groups], inverse[:groups], config, grid, self.live_nk)
            return
        record, fast, slow = self.packed_record[:self.rows // 4], self.fast_budget[:self.rows // 4], self.slow_budget[:self.rows // 4]
        m["audit"].prepare_fp16_h22_live_k(x, self.error, record, fast, slow, SEED, self.live_nk)
        slow_x, fast_x = list(x), list(x)
        slow_x[13], fast_x[13] = slow, fast
        common = (streams, self.id_pool, self.page_table, self.pool_head, self.pool_status,
                  self.error, SEED, self.keep_words)
        self.h22_args = ((paid, slow_x, *common, self.live_nk),
                         (paid, fast_x, *common, record, self.live_nk))
        if self.h22_graphs is not None and torch.cuda.is_current_stream_capturing():
            m["post"].attach_if_any_not_in(slow, 14, 16, self.h22_graphs[0].raw_cuda_graph())
            m["post"].attach_if_any_not_in(fast, 14, 14, self.h22_graphs[1].raw_cuda_graph())
        else:
            m["audit"].emit_trusted_candidates_live_k(*self.h22_args[0])
            m["audit"].emit_fp16_candidates_live_k(*self.h22_args[1])

    def _recover(self):
        self.cold_runs.add_(1)
        for begin in range(0, self.rows, 512):
            self.leaf(self, begin, min(self.rows, begin + 512))

    def _body(self):
        for value in (self.pool_head, self.pool_status, self.solo_counts, self.common_counts,
                      self.solo_n, self.common_n, self.error, self.replay_work):
            value.zero_()
        self.ops.clear_live_table[(self.rows + self.rows // 4,)](
            self.page_table, self.error, self.pool_status, self.live_nk, self.rows,
            self.page_table.stride(0), self.max_k, SEED, 256, num_warps=4)
        positive = (self.w_input[:self.rows] > 0).sum(1)
        tags = (positive > 14).long() + (positive > 16).long() + (positive > self.high_paid).long()
        self.order.copy_(tags.argsort(stable=True))
        for source, target in ((self.q_input.view(torch.uint8), self.q), (self.w_input, self.w),
                               (self.start_input, self.starts), (self.end_input, self.ends)):
            torch.index_select(source[:self.rows], 0, self.order, out=target)
        maximum = positive.index_select(0, self.order).reshape(-1, 8).max(1).values
        self.budget8.copy_(torch.where(maximum <= 14, 14, torch.where(maximum <= 16, 16, self.high_paid)))
        self.budget4.copy_(self.budget8.repeat_interleave(2))
        q = self.q.view(torch.float8_e4m3fn)
        self._seed(q)
        torch.index_select(self.pk.view(torch.uint8), 0, self.sample_ids, out=self.sample_k)
        torch.index_select(self.ps, 0, self.sample_ids, out=self.sample_scale)
        self.modules["sample"].seed_score(q, self.sample_k.view(torch.float8_e4m3fn),
                                          self.sample_scale, self.w, self.sample_scores, self.sample_masks, self.sms)
        self.capture[:, :128].copy_(self.sample_masks)
        self.capture[:, 128].copy_(self.error)
        for paid in (14, 16, self.high_paid):
            planner = self.modules["direct_abs"] if paid in (14, 16) else self.modules["planner"]
            planner.plan128(paid, q, self.w, self.capture, self.plans[paid], self.budget8)
        self._metadata(q)
        streams = [self.solo_counts, self.solo_tasks, self.solo_n,
                   self.common_counts, self.common_tasks, self.common_n]
        for paid in (14, 16, self.high_paid):
            fields = self.plans[paid]
            x = [self.q, self.pk.view(torch.uint8), self.ps, self.rho, self.w,
                 *fields[:6], self.starts, self.ends, self.budget4, self.safe_tau]
            self._emit(paid, x, streams)
        self.modules["common"].compact_candidates_live_nk(
            streams, self.id_pool, self.page_table, self.pool_head, self.pool_status,
            self.error, self.keep_words, self.budget4, SEED, self.max_k,
            self.starts, self.ends, self.live_nk)
        self.error_before_completion.copy_(self.error)
        exact = self.modules["exact"]
        exact.validate_seed5_live_nk(self.ids, self.counts, self.seed_tau, self.safe_tau,
                                    self.starts, self.ends, self.error, self.max_k, self.live_nk, SEED)
        for method, counts, tasks, task_count, table in (
            (exact.exact_paged_solo_live_nk, self.solo_counts, self.solo_tasks, self.solo_n, self.page_table[:self.rows]),
            (exact.exact_paged_common_live_nk, self.common_counts, self.common_tasks, self.common_n, self.page_table[self.rows:]),
        ):
            method(q, self.pk, self.ps, self.w, self.safe_tau, self.id_pool, counts,
                   tasks, task_count, self.values, self.ids, self.counts, 0, self.sms,
                   table, self.max_pages, self.starts, self.ends, self.error, self.live_nk, SEED)
        self.modules["integer"].select_dense5(
            self.values, self.ids, self.counts, self.starts, self.ends, self.epsilon[:self.rows],
            self.pool_status, self.error, self.ordered_output, self.select_diag,
            self.boundary_ids, self.boundary_counts, self.temporary, self.temporary_ids,
            self.max_k)
        self.modules["post"].complete_h32_fused_large(
            q, self.pk, self.ps, self.w, self.boundary_ids, self.boundary_counts,
            self.select_diag, self.starts, self.ends, self.error, self.ordered_output,
            self.temporary, self.temporary_ids)
        self.output.index_copy_(0, self.order, self.ordered_output)
        if self.cold_graph is None:
            self._recover()
        elif torch.cuda.is_current_stream_capturing():
            self.modules["post"].attach_if_any(self.error, False, self.cold_graph.raw_cuda_graph())
        else:
            self.cold_dispatch.replay()
        self.counts_original.index_copy_(0, self.order, self.counts)
        self.error_original.index_copy_(0, self.order, self.error)

    def _execute(self):
        if self.hot_graph is not None:
            self.hot_graph.replay()
            self.replay_calls += 1
            return
        from .dosa_graph import capture

        self._body()  # Warm all native/FI variants with this real request's fields.
        with capture(self.device) as self.cold_graph:
            self._recover()
        with capture(self.device, keep_graph=False) as self.cold_dispatch:
            self.modules["post"].attach_if_any(self.error, False, self.cold_graph.raw_cuda_graph())
        self.cold_dispatch.children = (self.cold_graph, self.leaf)
        self.cold_runs.zero_()
        children = []
        for method, args in zip((self.modules["audit"].emit_trusted_candidates_live_k,
                                 self.modules["audit"].emit_fp16_candidates_live_k), self.h22_args):
            with capture(self.device) as child:
                method(*args)
            child.inputs = args
            children.append(child)
        self.h22_graphs = tuple(children)
        with capture(self.device, pool=self.graph_pool.acquire(), keep_graph=False) as self.hot_graph:
            self._body()
        self.hot_graph.children = (self.cold_graph, self.cold_dispatch, self.h22_graphs,
                                   self.seed_logits, self.seed_selected)
        self.shape_graphs[self.rows] = (self.hot_graph, self.cold_graph, self.cold_dispatch, self.h22_graphs)

    def _map_vote(self, rows, entry, keys, recent):
        votes = self._votes(entry, keys, torch.cuda.current_stream(self.device))
        self.modules["caller"].map_topk_vote_stats_litetopk_(
            self.output[:rows], self.permutation[:keys], self.map_status[:rows], votes,
            recent)
        return votes

    def _stage(self, q, w, starts, ends, physical):
        logical = len(q)
        for target, source in ((self.q_input, q), (self.w_input, w),
                               (self.start_input, starts), (self.end_input, ends)):
            target[:logical].copy_(source)
            if physical != logical:
                target[logical:physical].copy_(target[logical - 1:logical].expand(
                    physical - logical, *target.shape[1:]))

    def _block(self, q, w, cache, table64, table1, starts, ends, result, *,
               nk, common_end, last_visible, hot_key, ready_event=None):
        stream = torch.cuda.current_stream(self.device)
        physical = (len(q) + 7) // 8 * 8
        self.bind_shape(physical, nk)
        for event in (self.last_done, self.startup_ready, ready_event):
            if event is not None:
                stream.wait_event(event)
        self.live_nk.fill_(nk)
        self._stage(q, w, starts, ends, physical)
        entry = self._request(hot_key)
        self._gather(entry, q, w, cache, table64, nk, common_end, 0)
        self._execute()
        recent = min(len(q), RECENT)
        votes = self._map_vote(len(q), entry, last_visible, recent)
        self._publish_carry(entry, votes, last_visible, recent)
        stream.wait_event(entry["prior"][2])
        self._map_pages(table1, self.output[:len(q)], result[:len(q)])

    @staticmethod
    def _map_pages(table1, ids, result):
        from sglang.kernels.ops.attention.dsa.transform_index import transform_index_page_table_decode_fast

        transform_index_page_table_decode_fast(table1.expand(len(ids), -1), ids, result=result, page_size=1)

    def run_paged(self, q, w, cache, table64, table1, starts, ends, result, *,
                  nk, common_end, last_visible, hot_key, window_start=0,
                  ready_event=None, causal_query_start=None):
        """Select a request using caller-owned page mappings.

        Active table64 entries must name physical pages in cache; table1 maps
        each active logical key to the caller's PAGED output ID. SGLang's pool
        supplies these mappings. Cache byte storage requires16-byte alignment.
        Result storage must be disjoint from inputs; read-only inputs may alias.
        Caller keeps borrowed inputs alive and active cache/table values stable
        until queued work completes on the calling stream.
        """
        rows = q.shape[0] if q.ndim else 0
        inputs = (q, w, cache, table64, table1, starts, ends, result)
        if (torch.cuda.is_current_stream_capturing() or not 0 < rows <= self.logical_query_capacity
                or not SEED <= nk <= self.max_k or window_start != 0
                or not SEED <= common_end <= last_visible <= nk
                or not valid_paged_inputs(self.device, *inputs, score_heads=self.score_heads)
                or table64.shape[1] < self.triton.cdiv(nk, 64) or table1.shape[1] < nk):
            return False
        with self.lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            for event in (self.last_done, self.startup_ready, ready_event):
                if event is not None:
                    stream.wait_event(event)
            for value in inputs:
                value.record_stream(stream)
            if self.seed_logits is None or hot_key[0] not in self.layers:
                self.reserve_for_model_load(hot_key[0])
            try:
                if rows <= self.max_q:
                    self._block(*inputs, nk=nk, common_end=common_end, last_visible=last_visible,
                                hot_key=hot_key, ready_event=ready_event)
                else:
                    causal = (causal_query_start is not None and common_end == causal_query_start + 1
                              and last_visible == causal_query_start + rows)
                    self._slabs(*inputs, nk=nk, common_end=common_end, last_visible=last_visible,
                                hot_key=hot_key, causal=causal)
                return True
            finally:
                if self.last_done is None:
                    self.last_done = torch.cuda.Event()
                self.last_done.record(stream)

    def _slabs(self, q, w, cache, table64, table1, starts, ends, result, *,
               nk, common_end, last_visible, hot_key, causal):
        stream = torch.cuda.current_stream(self.device)
        slabs = tuple(query_slabs(len(q), self.max_q))
        entry = self._request(hot_key)
        previous = entry["prior"]
        saved = previous is not None
        if saved:
            if previous[2] is not None:
                stream.wait_event(previous[2])
            self.saved_hot.copy_(previous[0])
        committed = False
        recent = min(len(q), RECENT)
        repair = causal and slabs[-1][1] - slabs[-1][0] < recent
        try:
            if not causal:
                self.bind_shape(slabs[0][2], nk)
                self.live_nk.fill_(nk)
                self._gather(entry, q, w, cache, table64, nk, common_end, 0)
                votes = self._votes(entry, last_visible, stream)
            for begin, end, physical, overlap in slabs:
                logical = end - begin
                if causal:
                    visible = common_end + end - 1
                    scan_k = max(SEED, visible)
                    self._block(q[begin:end], w[begin:end], cache, table64, table1,
                                starts[begin:end], ends[begin:end], result[begin:end],
                                nk=scan_k, common_end=common_end + begin,
                                last_visible=visible, hot_key=hot_key)
                else:
                    self.bind_shape(physical, nk)
                    self.live_nk.fill_(nk)
                    self._stage(q[begin:end], w[begin:end], starts[begin:end], ends[begin:end], physical)
                    self._execute()
                    self._map_vote(logical, entry, last_visible, overlap)
                    self._map_pages(table1, self.output[:logical], result[begin:end])
                if repair and overlap:
                    offset = end - overlap - (len(q) - recent)
                    self.carry_rows[offset:offset + overlap].copy_(self.output[logical - overlap:logical])
                if self.last_done is None:
                    self.last_done = torch.cuda.Event()
                self.last_done.record(stream)
            if repair:
                votes = self._votes(entry, last_visible, stream)
                self.modules["caller"].map_topk_vote_stats_litetopk_(
                    self.carry_rows[:recent], self.identity[:last_visible], self.repair_status[:recent],
                    votes, recent)
            if repair or not causal:
                self._publish_carry(entry, votes, last_visible, recent)
                stream.wait_event(entry["prior"][2])
            committed = True
        finally:
            if not committed:
                stream.wait_stream(self.side_stream)
                if saved:
                    previous[0].copy_(self.saved_hot)
                    self.restore_event.record(stream)
                    entry["prior"] = (previous[0], previous[1], self.restore_event, previous[3])
                else:
                    entry["prior"] = None

    def publish_stock_paged(self, active_paged, table64, *, hot_key, visible_end, window_start=0):
        lower = max(SEED, FEW_PREFILL_MIN_KEYS - self.logical_query_capacity)
        if (torch.cuda.is_current_stream_capturing() or window_start != 0
                or not lower <= visible_end < min(FEW_PREFILL_MIN_KEYS, self.max_k + 1)
                or active_paged.device != self.device or table64.device != self.device
                or active_paged.dtype != torch.int32 or table64.dtype != torch.int32
                or active_paged.ndim != 2 or not 0 < len(active_paged) <= self.logical_query_capacity
                or active_paged.shape[1] != TOPK or active_paged.stride(1) != 1
                or active_paged.stride(0) < TOPK or table64.ndim != 2 or len(table64) != 1
                or table64.shape[1] < self.triton.cdiv(visible_end, 64) or not table64.is_contiguous()):
            return False
        with self.lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            for event in (self.last_done, self.startup_ready):
                if event is not None:
                    stream.wait_event(event)
            entry = self._request(hot_key)
            entry["prior"] = entry["pending"] = None
            recent, pages = min(len(active_paged), RECENT), self.triton.cdiv(visible_end, 64)
            snapshot = self._empty(recent * TOPK + pages, dtype=torch.int32)
            snapshot[:recent * TOPK].view(recent, TOPK).copy_(active_paged[-recent:])
            snapshot[recent * TOPK:].copy_(table64[0, :pages])
            self.side_stream.wait_stream(stream)
            with torch.cuda.stream(self.side_stream):
                votes = self._votes(entry, visible_end, self.side_stream)
                hash_storage = self.carry_partial.view(-1).view(torch.int32)
                snapshot.record_stream(self.side_stream)
                self.modules["caller"].stock_paged_to_votes_(snapshot, hash_storage, votes, entry["gpu_status"], recent)
                self.modules["caller"].carry_votes_topk_reset_(
                    votes, entry["hot"], self.carry_partial, self.carry_state, SEED, recent, 0)
                entry["host_status"].copy_(entry["gpu_status"], non_blocking=True)
                entry["ready"].record(self.side_stream)
            entry["free"] = entry["ready"]
            entry["pending"] = (entry["hot"], visible_end, entry["ready"], 0)
            return True


_SLOTS = {}
_SLOT_LOCK = threading.Lock()


def _enabled():
    mode = os.environ.get("SGLANG_DOSA_MODE", "off").lower()
    if mode not in ("off", "few"):
        raise ValueError("SGLANG_DOSA_MODE must be off or few")
    return mode == "few"


def configured_query_capacity(server_args=None):
    from sglang.srt.arg_groups.overrides import max_prefill_buffer_tokens

    if server_args is None:
        from sglang.srt.runtime_context import get_server_args
        server_args = get_server_args()
    rows = int(max_prefill_buffer_tokens(server_args))
    if rows <= 0:
        rows = int(server_args.max_prefill_tokens or 0)
    capacity = (rows + 7) // 8 * 8
    return capacity if MIN_FEW_QUERY_CAPACITY <= capacity <= MAX_FEW_QUERY_CAPACITY else 0


def _supported(indexer, device):
    return (device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0)
            and indexer.n_heads in (32, 64) and indexer.head_dim == 128
            and indexer.index_topk == TOPK and indexer.num_init_tokens == 0
            and indexer.num_local_tokens == 0 and not indexer.dsa_enable_prefill_cp
            and not indexer.logits_with_pp_recv)


def prepare_indexer_backend(indexer):
    """Reserve Few and its layer's carry before the model sizes the KV pool."""
    if not _enabled() or not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = configured_query_capacity()
    if not capacity or not _supported(indexer, device):
        return False
    key = (str(device), indexer.n_heads)
    with _SLOT_LOCK:
        backend = _SLOTS.get(key)
        if backend is None:
            backend = FewBackend(device, max_queries=capacity, max_keys=1048576,
                                 score_heads=indexer.n_heads)
        if backend.logical_query_capacity != capacity:
            raise RuntimeError("DoSA prefill capacity changed after workspace reservation")
        backend.reserve_for_model_load(indexer.layer_id)
        _SLOTS[key] = backend
    return True


def _selection_supported(metadata, *, stock=False):
    from sglang.srt.environ import envs

    return (metadata.topk_transform_method.name == "PAGED"
            and metadata.topk_backend.name in (("SGL_KERNEL",) if stock else ("SGL_KERNEL", "FLASHINFER"))
            and not metadata.force_unfused_topk and envs.SGLANG_DSA_FUSE_TOPK.get()
            and not envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get())


def _prepared(device, heads):
    with _SLOT_LOCK:
        backend = _SLOTS.get((str(device), heads))
    if backend is None:
        raise RuntimeError("DoSA workspace must be prepared before model KV sizing")
    if backend.logical_query_capacity != configured_query_capacity():
        raise RuntimeError("DoSA prefill capacity changed after workspace reservation")
    return backend


def try_paged_prefill(indexer, forward_batch, layer_id, q, weights, metadata,
                     starts, ends, table64, result, pool):
    if (not _enabled() or not indexer._dosa_prefill_prepared
            or torch.cuda.is_current_stream_capturing()
            or not _supported(indexer, q.device) or not _selection_supported(metadata)
            or q.ndim != 3 or weights.ndim != 2):
        return False
    rows = starts.numel()
    capacity = configured_query_capacity()
    lengths = metadata.get_indexer_seq_len_cpu()
    extend, rids = forward_batch.extend_seq_lens_cpu, forward_batch.rids
    if (not 0 < rows <= capacity or lengths is None or lengths.device.type != "cpu"
            or lengths.numel() != 1 or extend is None or len(extend) != 1
            or int(extend[0]) != rows or rids is None or len(rids) != 1
            or table64.ndim != 2 or len(table64) != 1):
        return False
    nk = int(lengths[0])
    common_end = nk - rows + 1
    if not FEW_PREFILL_MIN_KEYS <= nk <= 1048576 or common_end < SEED:
        return False
    backend = _prepared(q.device, indexer.n_heads)
    return backend.run_paged(
        q[:rows], weights[:rows], indexer._get_index_k_read_buffer(pool, layer_id),
        table64, metadata.get_page_table_1(), starts, ends, result, nk=nk,
        common_end=common_end, last_visible=nk, hot_key=(layer_id, rids[0]),
        causal_query_start=nk - rows)


def publish_stock_paged_prior(indexer, forward_batch, layer_id, metadata,
                              table64, paged_result, *, rows):
    if (not _enabled() or not indexer._dosa_prefill_prepared
            or torch.cuda.is_current_stream_capturing()
            or not _supported(indexer, paged_result.device)
            or forward_batch.attn_cp_metadata is not None
            or not forward_batch.forward_mode.is_extend_without_speculative()
            or not _selection_supported(metadata, stock=True)
            or metadata.attn_metadata.topk_indices_offset is not None):
        return False
    lengths, batch_lengths = metadata.get_indexer_seq_len_cpu(), forward_batch.seq_lens_cpu
    extend, rids = forward_batch.extend_seq_lens_cpu, forward_batch.rids
    if (lengths is None or lengths.device.type != "cpu" or lengths.numel() != 1
            or batch_lengths is None or batch_lengths.device.type != "cpu" or batch_lengths.numel() != 1
            or extend is None or len(extend) != 1 or int(extend[0]) != rows
            or rids is None or len(rids) != 1 or not isinstance(rids[0], str) or not rids[0]
            or rows <= 0 or paged_result.ndim != 2 or len(paged_result) < rows):
        return False
    visible = int(lengths[0])
    if visible != int(batch_lengths[0]) or not max(rows, SEED) <= visible <= 1048576:
        return False
    return _prepared(paged_result.device, indexer.n_heads).publish_stock_paged(
        paged_result[:rows], table64, hot_key=(layer_id, rids[0]), visible_end=visible)
