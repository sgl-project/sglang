"""Early rejection of requests whose predicted prefill-side TTFT is too long.

A request is rejected on arrival when draining the prefill queue ahead of it,
itself included, is predicted to exceed SGLANG_EARLY_REJECT_THRESHOLD seconds
(<= 0 disables). The queue (waiting queue, chunked remainder and, on a PD
prefill server, the bootstrap queue) is drained pass by pass under the chunk
budget through a per-step cost model fitted online:

    step_us = c0 + c1 * uncached + c2 * kv_reads

Queued requests are credited their prefix match plus known storage hits.
Two ratios learned from completed requests correct the credit: survival (the
share of credited match still cached at run time) and gain (the share of the
rest found cached anyway). Without enough history nothing is rejected.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Sequence, Tuple

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    is_health_check_generate_req,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Prefill steps kept for the fit, split evenly between small and large steps.
_STEP_WINDOW = 256
# Before the fit is usable, longer steps (warmup, compiles) are skipped. After,
# a step is clipped to the larger of this and _STEP_CLIP times its prediction:
# steady slow steps are real cost, while rare stalls only get bounded weight.
_STEP_MAX_US = 2_000_000
_STEP_CLIP = 3.0
# Below this many tokens of evidence a match ratio is noise.
_MATCH_MIN_TOKENS = 50_000
# Half-life of the match ratios, in completed requests.
_MATCH_HALFLIFE = 40
_RIDGE = 1e-6
# Simulated passes before a queue counts as unbounded.
_MAX_SIM_STEPS = 20_000
# Admission snapshots and pending verdicts kept per request. A verdict is
# consumed on enqueue, normally the same round; only arrivals stuck in grammar
# compilation linger, far fewer than this.
_SNAPSHOT_CAP = 4096


def kv_reads(prefix: float, new: float, chunk_size: int) -> float:
    """KV tokens attention reads for ``new`` tokens after ``prefix``, per chunk.

    Additive over chunks, so one call prices a prefill however it is split.
    """
    if new <= 0 or chunk_size <= 0:
        return 0.0
    return new * (prefix + new / 2.0) / chunk_size


class StepRegression:
    """Online least squares for ``step_us = c0 + c1*U + c2*R`` (U, R in ktokens).

    Small and large (U >= chunk/2) steps live in separate FIFO buckets so that
    neither regime evicts the other: small steps pin c0, large ones the slopes.
    The normal equations are kept as running sums; the fit is redone lazily.
    """

    def __init__(self, window: int = _STEP_WINDOW) -> None:
        # (step_us, U, R, wall_us or 0)
        self.small: Deque[Tuple[float, float, float, float]] = deque(maxlen=window // 2)
        self.large: Deque[Tuple[float, float, float, float]] = deque(maxlen=window // 2)
        # Upper triangle of X^T X: (00, 01, 02, 11, 12, 22).
        self._xtx = [0.0] * 6
        self._xty = [0.0] * 3
        self._yty = 0.0
        self._coeffs: Optional[Tuple[float, float, float]] = None
        self._dirty = True
        # Largest step ever seen; bounds extrapolation and never ages out.
        self._max_ktokens = 0.0
        # Mean wall minus GPU time per step (CPU work GPU timing misses).
        self._gap_us = 0.0

    @property
    def n_samples(self) -> int:
        return len(self.small) + len(self.large)

    def push(
        self,
        step_us: float,
        uncached: int,
        reads: float,
        large: bool = False,
        wall_us: float = 0,
    ) -> None:
        """``wall_us`` > 0 only for GPU-timed contiguous steps (feeds the gap)."""
        if step_us <= 0 or uncached <= 0:
            return
        bucket = self.large if large else self.small
        if len(bucket) == bucket.maxlen:
            self._accumulate(bucket[0], -1.0)
        sample = (
            float(step_us),
            uncached / 1000.0,
            max(0.0, reads) / 1000.0,
            float(max(0, wall_us)),
        )
        bucket.append(sample)
        self._accumulate(sample, 1.0)
        self._max_ktokens = max(self._max_ktokens, sample[1])
        self._dirty = True

    def _accumulate(self, sample, sign: float) -> None:
        y, x1, x2, _ = sample
        t = self._xtx
        t[0] += sign
        t[1] += sign * x1
        t[2] += sign * x2
        t[3] += sign * x1 * x1
        t[4] += sign * x1 * x2
        t[5] += sign * x2 * x2
        b = self._xty
        b[0] += sign * y
        b[1] += sign * x1 * y
        b[2] += sign * x2 * y
        self._yty += sign * y * y

    def coefficients(self) -> Optional[Tuple[float, float, float]]:
        """``(c0, c1, c2)`` in us and us per ktoken, or None before a usable fit."""
        if not self._dirty:
            return self._coeffs
        self._dirty = False
        gaps = [s[3] - s[0] for s in self.large if s[3] > 0]
        self._gap_us = max(0.0, sum(gaps) / len(gaps)) if gaps else 0.0
        solved = self._solve()
        if solved is not None:
            self._coeffs = solved
        return self._coeffs

    def _solve(self) -> Optional[Tuple[float, float, float]]:
        # Non-negative slopes by enumerating the four active sets. At full load
        # every step sits at the chunk budget and an unconstrained c1 can go
        # negative.
        if self._xtx[0] < 3.0:
            return None
        fits = {}
        for mask in range(4):
            fit = self._solve_active([i for i in (0, 1) if mask & (1 << i)])
            if fit is None or fit[1] < 0.0 or fit[2] < 0.0:
                continue
            sse = self._sse(fit)
            if math.isfinite(sse):
                fits[mask] = (sse, fit)
        # Keep c2 only when significant (partial F-test, t >= 2); with little
        # spread in KV reads it would fit noise.
        for mask in (2, 3):
            if mask not in fits:
                continue
            sse = fits[mask][0]
            base = fits.get(mask & 1) or fits.get(0)
            dof = self._xtx[0] - 1 - bin(mask).count("1")
            if base is None or dof <= 0 or base[0] - sse < 4.0 * sse / dof:
                del fits[mask]
        best = min(fits.values(), key=lambda f: f[0], default=None)
        # An intercept-only fit does not respond to load at all.
        if best is None or (best[1][1] == 0.0 and best[1][2] == 0.0):
            return None
        return best[1]

    def _full_xtx(self):
        t = self._xtx
        return ((t[0], t[1], t[2]), (t[1], t[3], t[4]), (t[2], t[4], t[5]))

    def _solve_active(self, active: List[int]) -> Optional[Tuple[float, float, float]]:
        full = self._full_xtx()
        cols = [0] + [i + 1 for i in active]
        size = len(cols)
        t = self._xtx
        ridge = _RIDGE * max(1.0, t[0] + t[3] + t[5])
        matrix = []
        for r, cr in enumerate(cols):
            row = [full[cr][cc] for cc in cols]
            if cr != 0:
                row[r] += ridge
            row.append(self._xty[cr])
            matrix.append(row)
        for col in range(size):
            pivot = max(range(col, size), key=lambda r: abs(matrix[r][col]))
            if abs(matrix[pivot][col]) < 1e-12:
                return None
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            divisor = matrix[col][col]
            for j in range(col, size + 1):
                matrix[col][j] /= divisor
            for row in range(size):
                factor = matrix[row][col]
                if row == col or factor == 0.0:
                    continue
                for j in range(col, size + 1):
                    matrix[row][j] -= factor * matrix[col][j]
        out = [0.0, 0.0, 0.0]
        for r, cr in enumerate(cols):
            out[cr] = matrix[r][size]
        if not all(math.isfinite(v) for v in out):
            return None
        return (out[0], out[1], out[2])

    def _sse(self, coeffs: Tuple[float, float, float]) -> float:
        # y'y - 2 b'X'y + b'X'X b, from the running sums.
        full = self._full_xtx()
        quad = sum(
            coeffs[i] * full[i][j] * coeffs[j] for i in range(3) for j in range(3)
        )
        lin = sum(coeffs[i] * self._xty[i] for i in range(3))
        return self._yty - 2.0 * lin + quad

    def predict_step_us(
        self, uncached: float, reads: float, cached: bool = False
    ) -> Optional[float]:
        """``cached`` reuses the last fit, refitted only when a prediction for
        an arrival needs it."""
        coeffs = self._coeffs if cached else self.coefficients()
        if coeffs is None:
            return None
        c0, c1, c2 = coeffs
        k = uncached / 1000.0
        base = c0 + c1 * k
        ref = self._max_ktokens
        if 0.0 < ref < k:
            # Past the largest step seen, scale that step: the per-token cost
            # does not fall with step size.
            base = max(base, (c0 + c1 * ref) * k / ref)
        return max(1.0, base + c2 * max(0.0, reads) / 1000.0 + self._gap_us)

    def simulate_queue_s(
        self,
        pairs: Sequence[Tuple[int, int]],
        my_pair: Tuple[int, int],
        chunk_size: int,
        max_reqs: Optional[int] = None,
    ) -> Optional[float]:
        """Seconds to drain ``(seqlen, uncached)`` pairs in order.

        Each pass takes uncached tokens from the head until the chunk budget or
        ``max_reqs`` is spent, as the prefill adder does, so c0 is paid per
        pass, not per request. Decode steps interleaved with prefill are not
        modeled.
        """
        if self.coefficients() is None:
            return None
        # [uncached left, context resident]
        queue = [[u, max(0, n - u)] for n, u in [*pairs, my_pair] if u > 0]
        total_us = 0.0
        head = 0
        for _ in range(_MAX_SIM_STEPS):
            while head < len(queue) and queue[head][0] <= 0:
                head += 1
            if head == len(queue):
                return total_us / 1e6
            budget = chunk_size
            step_u, step_r = 0, 0.0
            idx = head
            end = len(queue) if max_reqs is None else min(len(queue), head + max_reqs)
            while idx < end and budget > 0:
                take = min(queue[idx][0], budget)
                step_u += take
                step_r += kv_reads(queue[idx][1], take, chunk_size)
                queue[idx][0] -= take
                queue[idx][1] += take
                budget -= take
                idx += 1
            total_us += self.predict_step_us(step_u, step_r)
        # The deepest backlog; None would admit.
        return math.inf


class AdmissionEstimator:
    """Step cost model and match correction, kept on the deciding rank only."""

    def __init__(
        self, chunk_size: int, min_samples: int, max_reqs: Optional[int] = None
    ) -> None:
        self.chunk_size = chunk_size
        self.min_samples = min_samples
        # Requests per prefill batch (--prefill-max-requests); None = unbounded.
        self.max_reqs = max_reqs
        self.regression = StepRegression()
        # Decayed token totals behind the survival and gain ratios.
        self.credited = 0.0
        self.survived = 0.0
        self.uncredited = 0.0
        self.gained = 0.0

    def push_step(
        self, wall_us: int, gpu_us: int, contiguous: bool, prefix_lens, extend_lens
    ) -> None:
        """Fit GPU time when timed: under overlap, wall deltas move time
        between neighbouring steps. Wall time is the fallback."""
        uncached = sum(extend_lens)
        reads = sum(
            kv_reads(p, e, self.chunk_size) for p, e in zip(prefix_lens, extend_lens)
        )
        # Priced by the last fit, before this sample joins it.
        pred = self.regression.predict_step_us(uncached, reads, cached=True)
        if pred is None:
            if wall_us >= _STEP_MAX_US:
                return
            cap = _STEP_MAX_US
        else:
            cap = max(_STEP_MAX_US, _STEP_CLIP * pred)
        wall_us = min(wall_us, cap)
        gpu_us = min(gpu_us, cap)
        self.regression.push(
            gpu_us if gpu_us > 0 else wall_us,
            uncached,
            reads,
            large=2 * uncached >= self.chunk_size,
            wall_us=wall_us if gpu_us > 0 and contiguous else 0,
        )

    def push_match(self, n_tokens: int, credited: int, cached: int) -> None:
        """One finished request: what it was credited vs. what it found cached."""
        decay = 0.5 ** (1.0 / _MATCH_HALFLIFE)
        cached = max(0, cached)
        credited = max(0, min(credited, n_tokens))
        if credited > 0:
            self.credited = self.credited * decay + credited
            self.survived = self.survived * decay + min(cached, credited)
        if n_tokens > credited:
            self.uncredited = self.uncredited * decay + n_tokens - credited
            self.gained = self.gained * decay + max(0, cached - credited)

    def survival(self) -> float:
        if self.credited < _MATCH_MIN_TOKENS:
            return 1.0
        return self.survived / self.credited

    def gain(self) -> float:
        if self.uncredited < _MATCH_MIN_TOKENS:
            return 0.0
        return self.gained / self.uncredited

    def price(self, pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Charge ``u * (1 - gain) + (1 - survival) * matched`` per request."""
        gain, lost = self.gain(), 1.0 - self.survival()
        if gain <= 0.0 and lost <= 0.0:
            return list(pairs)
        return [(n, min(n, int(u * (1.0 - gain) + lost * (n - u)))) for n, u in pairs]

    def estimate_s(self, pairs, my_pair) -> Optional[float]:
        """Predicted drain through ``my_pair``, or None without enough history."""
        if self.regression.n_samples < self.min_samples:
            return None
        priced = self.price([*pairs, my_pair])
        est = self.regression.simulate_queue_s(
            priced[:-1], priced[-1], self.chunk_size, self.max_reqs
        )
        return est if est is not None and est > 0 else None


def _check_supported(scheduler: Scheduler) -> None:
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        raise ValueError(
            "SGLANG_EARLY_REJECT_THRESHOLD is not supported on a PD decode server."
        )
    # Step times are per stage under PP, so the estimate would run optimistic.
    if get_parallel().pp_size > 1:
        raise ValueError(
            "SGLANG_EARLY_REJECT_THRESHOLD does not support --pp-size > 1."
        )
    # The controller is built once for the startup role.
    if get_disagg().enable_pd_role_switch:
        raise ValueError(
            "SGLANG_EARLY_REJECT_THRESHOLD does not support --enable-pd-role-switch."
        )
    if not scheduler.chunked_prefill_size or scheduler.chunked_prefill_size <= 0:
        raise ValueError(
            "SGLANG_EARLY_REJECT_THRESHOLD requires --chunked-prefill-size > 0."
        )
    if scheduler.schedule_policy != "fcfs" or scheduler.enable_priority_scheduling:
        logger.warning(
            "Early reject prices the whole queue drain; off FCFS a request that "
            "is overtaken can still exceed the threshold."
        )


@dataclass
class _Snapshot:
    matched: int
    est_s: float = math.nan


class AdmissionController:
    """Scheduler-side early rejection.

    Every rank runs the admission prefix match, since ``match_prefix`` updates
    the radix tree and the trees must stay identical. One rank per attention
    group (pp0, attn_tp0, attn_cp0) decides and broadcasts its verdict.
    """

    @classmethod
    def maybe_create(cls, scheduler: Scheduler) -> Optional[AdmissionController]:
        threshold_s = envs.SGLANG_EARLY_REJECT_THRESHOLD.get()
        if threshold_s <= 0:
            return None
        _check_supported(scheduler)
        logger.info("Early reject (prefill) on: threshold=%.1fs", threshold_s)
        return cls(scheduler, threshold_s)

    def __init__(self, scheduler: Scheduler, threshold_s: float) -> None:
        self.scheduler = scheduler
        self.threshold_s = threshold_s
        ps = get_parallel()
        self.estimator = (
            AdmissionEstimator(
                scheduler.chunked_prefill_size,
                envs.SGLANG_EARLY_REJECT_REGRESSION_MIN_SAMPLES.get(),
                get_schedule().prefill_max_requests,
            )
            if ps.pp_rank == 0 and ps.attn_tp_rank == 0 and ps.attn_cp_rank == 0
            else None
        )
        self._snapshots: OrderedDict[str, _Snapshot] = OrderedDict()
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self._resident_prefetch = isinstance(scheduler.tree_cache, UnifiedRadixCache)
        # Verdicts wait here until the request is enqueued (insertion ordered).
        self._reject_rids: Dict[str, None] = {}

    # ---- arrival -------------------------------------------------------

    def on_recv(self, recv_reqs: List) -> None:
        """Judge this round's requests; runs on every rank, in the same order."""
        reject: List[str] = []
        # Earlier admits of this round queue ahead of later ones.
        admitted_ahead: List[Tuple[int, int]] = []
        judged = False
        for recv_req in recv_reqs:
            if isinstance(recv_req, BatchTokenizedGenerateReqInput):
                reqs = list(recv_req)
            elif isinstance(recv_req, TokenizedGenerateReqInput):
                reqs = [recv_req]
            else:
                continue
            for r in reqs:
                # Session requests are matched against session state, not the
                # tree; they are admitted and priced as uncached.
                if (
                    is_health_check_generate_req(r)
                    or r.session_params is not None
                    or r.session_id is not None
                    or not r.input_ids
                ):
                    continue
                judged = True
                n = len(r.input_ids)
                matched = self._match_prefix(r)
                if self.estimator is None:
                    continue
                snap = _Snapshot(matched)
                self._remember(r.rid, snap)
                est = self.estimator.estimate_s(
                    self._pending_pairs() + admitted_ahead, (n, n - matched)
                )
                if est is None:
                    continue
                snap.est_s = est
                if est > self.threshold_s:
                    reject.append(r.rid)
                    logger.info(
                        "Early reject rid=%s est_s=%.2f threshold_s=%.2f",
                        r.rid,
                        est,
                        self.threshold_s,
                    )
                else:
                    admitted_ahead.append((n, n - matched))
        if judged:
            # Accumulated: a paused scheduler may hold inputs past this round.
            for rid in self._broadcast(reject) or ():
                if len(self._reject_rids) >= _SNAPSHOT_CAP:
                    del self._reject_rids[next(iter(self._reject_rids))]
                self._reject_rids[rid] = None

    def _match_prefix(self, r) -> int:
        tree_cache = self.scheduler.tree_cache
        # Multimodal prompts are matched on padded ids built later; count them
        # uncached.
        if tree_cache is None or r.mm_inputs is not None:
            return 0
        token_ids = r.input_ids
        # Mirrors match_kv_cache's key; Req appends the LoRA id to extra_key.
        extra_key = r.extra_key
        if r.lora_id is not None:
            extra_key = (extra_key or "") + r.lora_id
        tail = tree_cache.window_reprefill_tail_tokens()
        try:
            result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=token_ids,
                        extra_key=extra_key,
                        limit=max(0, len(token_ids) - tail) if tail else None,
                        cache_salt=r.cache_salt,
                    )
                )
            )
        except Exception:
            # Uncached only errs pessimistic.
            logger.warning(
                "Early reject prefix match failed rid=%s", r.rid, exc_info=True
            )
            return 0
        # Capped as Req._compute_max_prefix_len: at least one token is always
        # prefilled, and input logprobs recompute from their start. Embed
        # overrides and SGLANG_RADIX_FORCE_MISS, which disable the match, are
        # not handled.
        max_len = len(token_ids) - 1
        if r.return_logprob and r.logprob_start_len >= 0:
            max_len = min(max_len, r.logprob_start_len)
        matched = result.device_prefix_len + result.host_hit_length
        return max(0, min(matched, max_len))

    def _remember(self, rid: str, snap: _Snapshot) -> None:
        if len(self._snapshots) >= _SNAPSHOT_CAP:
            self._snapshots.popitem(last=False)
        self._snapshots[rid] = snap

    def _pending_pairs(self) -> List[Tuple[int, int]]:
        """(seqlen, uncached) of every request ahead of a new arrival."""
        s = self.scheduler
        queued = list(s.waiting_queue)
        if s.disaggregation_mode == DisaggregationMode.PREFILL:
            queued += list(s.disagg_prefill_bootstrap_queue.queue)
        pairs = [(req.seqlen, self._queued_uncached(req)) for req in queued]
        chunked = s.chunked_req
        if chunked is not None and chunked.seqlen > chunked.prefix_len:
            pairs.append((chunked.seqlen, chunked.seqlen - chunked.prefix_len))
        return pairs

    def _queued_uncached(self, req: Req) -> int:
        n = req.seqlen
        matched = req.num_matched_prefix_tokens
        snap = self._snapshots.get(req.rid)
        if req.last_node is None and snap is not None:
            # Never matched since arrival (FCFS matches only at schedule time);
            # once matched, even a zero hit is current.
            matched = snap.matched
        uncached = max(0, n - matched)
        # A pending storage prefetch is outside the radix match until it lands.
        span = self._storage_span(req.cache_request_handle)
        if span is not None:
            uncached -= max(0, min(span[1], n) - max(span[0], matched))
        return max(0, uncached)

    def _storage_span(self, handle) -> Optional[Tuple[int, int]]:
        tree_cache = self.scheduler.tree_cache
        span = tree_cache.waiting_storage_span(handle)
        if span is not None or not self._resident_prefetch:
            return span
        # waiting_storage_span covers buffer_only host memory only; with a
        # resident host cache read the accepted prefetch directly.
        info = tree_cache.ongoing_prefetch.get(handle)
        if info is None or info.host_indices is None:
            return None
        op = info.operation
        return op.storage_start, op.storage_start + op.storage_hit_count

    def _broadcast(self, reject: List[str]) -> Optional[List[str]]:
        # Same groups as the request broadcast, so the two agree.
        ps = get_parallel()
        if ps.attn_dp_enabled:
            return attn_cp_tp_broadcast_pyobj(reject)
        if ps.tp_size > 1:
            g = ps.tp_group
            return broadcast_pyobj(reject, g.rank, g.cpu_group, src=g.ranks[0])
        return reject

    def reject_on_enqueue(self, req: Req) -> bool:
        """Finish a request rejected on arrival before it joins any queue."""
        if req.is_retracted or req.rid not in self._reject_rids:
            return False
        del self._reject_rids[req.rid]
        self._snapshots.pop(req.rid, None)
        # Already refused for its own reason (too long, bad input): keep it.
        if req.to_finish is not None:
            return False
        prepare_abort(
            req,
            f"Predicted TTFT exceeds the early-reject threshold {self.threshold_s:.2f}s.",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
        s = self.scheduler
        s.request_retirer.retire_unadmitted(req)
        if (
            s.disaggregation_mode == DisaggregationMode.PREFILL
            and req.bootstrap_room is not None
        ):
            # Decode registers the room anyway; fail it rather than leak it.
            kv_manager = s.disagg_prefill_bootstrap_queue.kv_manager
            if isinstance(kv_manager, CommonKVManager):
                kv_manager.abandon_room(req.bootstrap_room)
        # Every rank retires it; count once. Rejects bypass the tokenizer's
        # num_aborted_requests_total.
        reporter = s.metrics_reporter
        if self.estimator is not None and reporter.enable_metrics:
            reporter.metrics_collector.increment_early_rejected_reqs()
        return True

    # ---- feedback ------------------------------------------------------

    def start_gpu_timer(self, batch: ScheduleBatch, stream):
        """Events bracketing a prefill forward on the forward stream.

        They time the GPU work alone. Under overlap the wall time between
        results also holds CPU scheduling, and can shift between adjacent steps
        when the CPU lags the GPU.
        """
        if (
            self.estimator is None
            or not batch.forward_mode.is_extend_without_speculative()
        ):
            return None
        module = self.scheduler.device_module
        events = (module.Event(enable_timing=True), module.Event(enable_timing=True))
        events[0].record(stream=stream)
        return events

    def on_prefill_step(
        self, batch: ScheduleBatch, result, step_us: int, contiguous: bool
    ) -> None:
        if self.estimator is None:
            return
        gpu_us = -1
        events = (
            result.prefill_gpu_events
            if isinstance(result, GenerationBatchResult)
            else None
        )
        # Read after copy_done has synced, so this never blocks.
        if events is not None and events[1].query():
            gpu_us = int(events[0].elapsed_time(events[1]) * 1000)
        self.estimator.push_step(
            step_us,
            gpu_us,
            contiguous,
            batch.prefix_lens or (),
            batch.extend_lens or (),
        )

    def on_extend_processed(self, batch: ScheduleBatch) -> None:
        """Feed the match ratios with requests that finished prefill."""
        if self.estimator is None:
            return
        for req in batch.reqs:
            if req.time_stats.prefill_finished_time <= 0:
                continue
            snap = self._snapshots.pop(req.rid, None)
            if snap is not None:
                self._on_prefill_finished(req, snap)

    def _on_prefill_finished(self, req: Req, snap: _Snapshot) -> None:
        n = len(req.origin_input_ids)
        cached = (
            req.cached_tokens_device
            + req.cached_tokens_host
            + req.cached_tokens_storage
        )
        # Storage hits are credited as for queued requests, so gain keeps only
        # what neither foresaw.
        self.estimator.push_match(n, snap.matched + req.storage_hit_length, cached)
        ts = req.time_stats
        # PD admits at bootstrap-queue entry, so the actual includes the
        # bootstrap wait. nan: admitted without a prediction (cold start).
        start = ts.prefill_bootstrap_queue_entry_time or ts.wait_queue_entry_time
        if start <= 0:
            return
        actual_s = ts.prefill_finished_time - start
        logger.info(
            "Admission check (prefill) rid=%s predicted_drain_s=%.2f actual_s=%.2f "
            "diff_s=%.2f n=%d adm_matched=%d exec_device=%d exec_host=%d "
            "exec_storage=%d bootstrap_s=%.2f queue_s=%.2f forward_s=%.2f",
            req.rid,
            snap.est_s,
            actual_s,
            snap.est_s - actual_s,
            n,
            snap.matched,
            req.cached_tokens_device,
            req.cached_tokens_host,
            req.cached_tokens_storage,
            ts.wait_queue_entry_time - start if ts.wait_queue_entry_time > 0 else -1.0,
            ts.forward_entry_time - ts.wait_queue_entry_time
            if ts.forward_entry_time > 0 and ts.wait_queue_entry_time > 0
            else -1.0,
            ts.prefill_finished_time - ts.forward_entry_time
            if ts.forward_entry_time > 0
            else -1.0,
        )
