"""PHANTOM — HSA zero-copy ghost-draft speculative decoding.

AMD gfx103x-optimized speculative worker that uses ROCm's HSA shared memory
for zero-copy CPU→GPU draft token transfer.

Architecture:
  1. At prefill, freezes an n-gram corpus snapshot (constant cache — read-only)
  2. A CPU ghost thread continuously pre-builds draft trees from the frozen
     corpus into pinned (HSA-accessible) memory buffers
  3. On each decode step, GPU reads draft tokens directly from pinned memory
     via PCIe — zero explicit copy on ROCm HSA systems
  4. GPU runs standard NgramVerifyInput tree verification
  5. Meanwhile, CPU ghost thread is already building the NEXT tree

Benefits on AMD gfx103x:
  - Draft state lives in system RAM (pinned), not VRAM → more KV cache budget
  - HSA shared memory = zero-copy reads from GPU
  - N-gram corpus scan is pure CPU work → truly overlapped with GPU verify
  - Frozen corpus = no locks, no synchronization on reads

Usage: --speculative-algorithm PHANTOM --disable-overlap-schedule
"""

import logging
import os
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_prefilter import AdaptiveThresholdController
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.sok import KernelCache, KernelFingerprint, SOKConfig, ShapeProfile, DispatchTelemetry
from sglang.srt.speculative.sok.fingerprint import detect_fingerprint

try:
    from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
except ImportError:
    reconstruct_indices_from_tree_mask = None

logger = logging.getLogger(__name__)

# Track one-time warnings to avoid log spam
_sgl_kernel_warned = False


class _GhostBuffer:
    """N-buffered pinned memory for zero-copy CPU→GPU draft transfer.

    Ring buffer of N slots: CPU writes to the next free slot while GPU reads
    from the oldest ready slot. On ROCm/HSA, pinned memory is directly
    accessible by the GPU via PCIe without explicit copy.

    N=1: sync-only (debug mode, no CPU/GPU overlap)
    N=2: classic double-buffering (default, matches original behavior)
    N=3: one extra lookahead buffer for slow ghost threads
    N=4: maximum decoupling
    """

    def __init__(self, max_batch_size: int, draft_token_num: int, num_buffers: int = 2):
        assert 1 <= num_buffers <= 4, f"num_buffers must be 1-4, got {num_buffers}"
        K = draft_token_num
        self.num_buffers = num_buffers

        # Allocate N pinned buffer pairs
        self._tokens = [
            torch.zeros(max_batch_size * K, dtype=torch.int64, pin_memory=True)
            for _ in range(num_buffers)
        ]
        self._masks = [
            torch.zeros(max_batch_size * K * K, dtype=torch.bool, pin_memory=True)
            for _ in range(num_buffers)
        ]

        # Ring buffer indices
        self._write_idx = 0  # next slot for CPU to write
        self._read_idx = 0   # next slot for GPU to read
        self._ready_count = 0  # how many slots have valid data
        self._lock = threading.Lock()
        self._ready = threading.Event()

        # Metadata per slot
        self._slot_bs = [0] * num_buffers
        self.active_bs = 0
        self.active_k = K

    # ── Backward-compatible properties (for 2-buffer callers) ──

    @property
    def gpu_tokens(self) -> torch.Tensor:
        return self._tokens[self._read_idx]

    @property
    def gpu_mask(self) -> torch.Tensor:
        return self._masks[self._read_idx]

    @property
    def cpu_tokens(self) -> torch.Tensor:
        return self._tokens[self._write_idx]

    @property
    def cpu_mask(self) -> torch.Tensor:
        return self._masks[self._write_idx]

    def swap(self, bs: int, k: int = 0):
        """CPU finished writing — mark slot ready, advance write pointer.

        If the ring is full (all slots have unread data), overwrites the
        oldest slot rather than blocking — the GPU always wants the freshest
        data, and a stale slot is worse than a lost slot.
        """
        with self._lock:
            if self._ready_count >= self.num_buffers - 1:
                # Ring full — advance read pointer to drop oldest stale slot
                self._read_idx = (self._read_idx + 1) % self.num_buffers
                self._ready_count -= 1
            self._slot_bs[self._write_idx] = bs
            self._write_idx = (self._write_idx + 1) % self.num_buffers
            self._ready_count += 1
            self.active_bs = bs
            if k > 0:
                self.active_k = k
            self._ready.set()

    def wait_ready(self, timeout: float = 0.001) -> bool:
        """Wait for at least one ready slot."""
        return self._ready.wait(timeout=timeout)

    def consume(self):
        """GPU done reading — release slot, advance read pointer."""
        with self._lock:
            self._read_idx = (self._read_idx + 1) % self.num_buffers
            self._ready_count = max(self._ready_count - 1, 0)
            if self._ready_count == 0:
                self._ready.clear()
            # Update active_bs to next ready slot if available
            if self._ready_count > 0:
                self.active_bs = self._slot_bs[self._read_idx]


class _NegativeFilter:
    """Bloom filter tracking historically-rejected n-gram draft patterns.

    After each verify step, rejected draft token bigrams are inserted.
    The ghost thread scores candidate sequences against the filter to
    prefer drafts with fewer known-bad patterns.

    Uses a compact bytearray-based bloom filter with 2 hash functions.
    At 64K bits (8KB) and ~1000 patterns, false positive rate is ~1%.

    Aging: resets every `age_limit` insertions to prevent saturation.
    """

    def __init__(self, num_bits: int = 65536, age_limit: int = 2000):
        self._bits = bytearray(num_bits // 8)
        self._num_bits = num_bits
        self._count = 0
        self._age_limit = age_limit

    def _hash1(self, val: int) -> int:
        h = (val ^ 0x811C9DC5) * 0x01000193
        return h % self._num_bits

    def _hash2(self, val: int) -> int:
        h = ((val >> 16) ^ val) * 0x45D9F3B
        h = ((h >> 16) ^ h) * 0x45D9F3B
        return ((h >> 16) ^ h) % self._num_bits

    def insert_bigram(self, tok_a: int, tok_b: int):
        """Insert a rejected bigram pattern. Auto-ages when saturated."""
        if self._count >= self._age_limit:
            self._bits = bytearray(self._num_bits // 8)
            self._count = 0
            logger.debug("PHANTOM: negative filter aged (reset after %d)", self._age_limit)
        val = tok_a * 131071 + tok_b
        idx1 = self._hash1(val)
        idx2 = self._hash2(val)
        self._bits[idx1 // 8] |= (1 << (idx1 % 8))
        self._bits[idx2 // 8] |= (1 << (idx2 % 8))
        self._count += 1

    def query_bigram(self, tok_a: int, tok_b: int) -> bool:
        """Check if a bigram was previously rejected (may have false positives)."""
        val = tok_a * 131071 + tok_b
        idx1 = self._hash1(val)
        idx2 = self._hash2(val)
        return (
            (self._bits[idx1 // 8] & (1 << (idx1 % 8))) != 0
            and (self._bits[idx2 // 8] & (1 << (idx2 % 8))) != 0
        )

    def score_sequence(self, tokens: np.ndarray) -> float:
        """Score a draft sequence: fraction of bigrams NOT in the filter (higher=better)."""
        if len(tokens) < 2:
            return 1.0
        good = 0
        total = len(tokens) - 1
        for i in range(total):
            if not self.query_bigram(int(tokens[i]), int(tokens[i + 1])):
                good += 1
        return good / total

    def reset(self):
        """Clear the filter."""
        self._bits = bytearray(self._num_bits // 8)
        self._count = 0

    @property
    def size(self) -> int:
        return self._count


class _QuantTelemetry:
    """Lightweight inference telemetry for quantization-aware calibration.

    Accumulates signals during PHANTOM inference that inform which parts
    of the model are producing bad logits and can be quantized aggressively.

    All signals are collected from data already computed during the verify
    step — zero additional GPU forward passes required.

    Collected signals:
      1. Channel importance (running mean of |hidden_states| per dim)
         → hot channels need precision, cold channels can be crushed
      2. Logit margin at accept/reject boundaries
         → small margin = quant noise would flip the decision
      3. Token confusion pairs (draft vs actual when rejected)
         → embedding rows that confuse each other need precision
      4. Position-wise acceptance rate
         → which tree depths the model struggles with

    Periodic snapshots are written to disk for offline analysis by
    a quantization calibration tool.
    """

    def __init__(self, hidden_dim: int = 0, max_confusion_pairs: int = 8192,
                 snapshot_interval: int = 200, snapshot_dir: Optional[str] = None):
        self.hidden_dim = hidden_dim
        self._snapshot_interval = snapshot_interval
        self._snapshot_dir = snapshot_dir
        self._step = 0

        # Signal 1: channel importance — EMA of |activation| per hidden dim
        # Initialized lazily on first call (hidden_dim may not be known yet)
        self._channel_sum: Optional[np.ndarray] = None
        self._channel_count: int = 0

        # Signal 2: logit margin at decision boundaries
        # margin = logit[top1] - logit[top2] for accepted/rejected positions
        self._margin_accepted = deque(maxlen=5000)   # margins where draft was right
        self._margin_rejected = deque(maxlen=5000)   # margins where draft was wrong

        # Signal 3: confusion pairs — (draft_tok, actual_tok) frequency
        self._confusion: dict = {}  # {(draft, actual): count}
        self._max_confusion = max_confusion_pairs

        # Signal 4: position-wise acceptance histogram
        self._pos_accepted = np.zeros(32, dtype=np.int64)  # up to 32 draft positions
        self._pos_total = np.zeros(32, dtype=np.int64)

    def record(self, logits: torch.Tensor, hidden_states: Optional[torch.Tensor],
               draft_tokens: torch.Tensor, verified_ids: torch.Tensor,
               accept_lens: torch.Tensor, bs: int, K: int):
        """Record one round of telemetry from verify outputs.

        Args:
            logits: [num_tokens, vocab_size] — final logits from verify pass
            hidden_states: [num_tokens, hidden_dim] or None — last-layer activations
            draft_tokens: [bs * K] — what PHANTOM proposed
            verified_ids: [bs] — what the model actually picked
            accept_lens: [bs] — how many draft tokens were accepted per request
            bs: batch size
            K: draft length
        """
        self._step += 1

        try:
            self._record_channel_importance(hidden_states)
            self._record_logit_margins(logits, accept_lens, bs, K)
            self._record_confusion(draft_tokens, verified_ids, accept_lens, bs, K)
            self._record_position_acceptance(accept_lens, K)
        except Exception:
            pass  # telemetry is best-effort

        if (self._snapshot_dir and self._snapshot_interval > 0
                and self._step % self._snapshot_interval == 0):
            self._write_snapshot()

    def _record_channel_importance(self, hidden_states: Optional[torch.Tensor]):
        """Signal 1: accumulate |activation| per hidden dimension."""
        if hidden_states is None:
            return
        # Move to CPU, take abs mean across token dimension
        h = hidden_states.float().abs().mean(dim=0).cpu().numpy()
        if self._channel_sum is None:
            self._channel_sum = np.zeros_like(h)
            self.hidden_dim = len(h)
        self._channel_sum += h
        self._channel_count += 1

    def _record_logit_margins(self, logits: torch.Tensor,
                              accept_lens: torch.Tensor, bs: int, K: int):
        """Signal 2: top1-top2 logit gap at accept/reject boundaries."""
        if logits is None or logits.dim() != 2:
            return
        # Only sample a few positions per round to keep cost near-zero
        top2 = logits.topk(2, dim=-1).values  # [num_tokens, 2]
        margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()

        acc_cpu = accept_lens.cpu().numpy() if accept_lens.is_cuda else accept_lens.numpy()
        idx = 0
        for r in range(bs):
            acc = int(acc_cpu[r])
            for pos in range(min(K, len(margins) - idx)):
                m = float(margins[idx + pos])
                if pos < acc:
                    self._margin_accepted.append(m)
                else:
                    self._margin_rejected.append(m)
            idx += K

    def _record_confusion(self, draft_tokens: torch.Tensor,
                          verified_ids: torch.Tensor,
                          accept_lens: torch.Tensor, bs: int, K: int):
        """Signal 3: which draft tokens the model overrides (confusion pairs)."""
        draft_cpu = draft_tokens.cpu()
        verified_cpu = verified_ids.cpu()
        acc_cpu = accept_lens.cpu()

        for r in range(bs):
            acc = int(acc_cpu[r])
            # The token at position acc is the first rejected draft token;
            # verified_ids[r] is what the model chose instead
            reject_pos = r * K + acc
            if reject_pos < len(draft_cpu) and acc < K:
                draft_tok = int(draft_cpu[reject_pos])
                actual_tok = int(verified_cpu[r])
                if draft_tok != 0 and actual_tok != 0 and draft_tok != actual_tok:
                    pair = (draft_tok, actual_tok)
                    self._confusion[pair] = self._confusion.get(pair, 0) + 1
                    # Prune if too large — keep top-frequency pairs
                    if len(self._confusion) > self._max_confusion:
                        threshold = sorted(self._confusion.values())[-self._max_confusion // 2]
                        self._confusion = {
                            k: v for k, v in self._confusion.items() if v >= threshold
                        }

    def _record_position_acceptance(self, accept_lens: torch.Tensor, K: int):
        """Signal 4: which draft positions succeed/fail."""
        acc_cpu = accept_lens.cpu().numpy() if accept_lens.is_cuda else accept_lens.numpy()
        for acc in acc_cpu:
            acc = int(acc)
            for pos in range(min(K, len(self._pos_total))):
                self._pos_total[pos] += 1
                if pos < acc:
                    self._pos_accepted[pos] += 1

    def _write_snapshot(self):
        """Periodic dump to disk for offline quant calibration."""
        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            path = os.path.join(self._snapshot_dir, f"quant_telem_{self._step}.npz")

            data = {"step": self._step}

            if self._channel_sum is not None and self._channel_count > 0:
                data["channel_importance"] = self._channel_sum / self._channel_count

            if self._margin_accepted:
                data["margin_accepted"] = np.array(list(self._margin_accepted))
            if self._margin_rejected:
                data["margin_rejected"] = np.array(list(self._margin_rejected))

            if self._confusion:
                pairs = sorted(self._confusion.items(), key=lambda x: -x[1])[:1000]
                data["confusion_pairs"] = np.array(
                    [(d, a, c) for (d, a), c in pairs], dtype=np.int64
                )

            data["pos_accepted"] = self._pos_accepted.copy()
            data["pos_total"] = self._pos_total.copy()

            np.savez_compressed(path, **data)
            logger.info("PHANTOM quant telemetry snapshot: %s (%d steps)", path, self._step)
        except Exception as e:
            logger.debug("PHANTOM quant telemetry write failed: %s", e)

    def get_summary(self) -> dict:
        """Summary for periodic log."""
        avg_margin_acc = (
            sum(self._margin_accepted) / len(self._margin_accepted)
            if self._margin_accepted else 0.0
        )
        avg_margin_rej = (
            sum(self._margin_rejected) / len(self._margin_rejected)
            if self._margin_rejected else 0.0
        )
        return {
            "steps": self._step,
            "channel_samples": self._channel_count,
            "margin_acc": round(avg_margin_acc, 3),
            "margin_rej": round(avg_margin_rej, 3),
            "confusion_pairs": len(self._confusion),
            "pos_rate": [
                round(float(self._pos_accepted[i]) / max(float(self._pos_total[i]), 1), 2)
                for i in range(min(8, len(self._pos_total)))
                if self._pos_total[i] > 0
            ],
        }

    def reset(self):
        """Clear all accumulated telemetry."""
        self._step = 0
        self._channel_sum = None
        self._channel_count = 0
        self._margin_accepted.clear()
        self._margin_rejected.clear()
        self._confusion.clear()
        self._pos_accepted[:] = 0
        self._pos_total[:] = 0


class _GhostJob:
    """Immutable job descriptor for one ghost-thread round.

    Freezes batch_tokens, K, bs, and a monotonic job_id so the ghost thread
    and consumer can detect stale results even when batch size stays the same.
    """
    __slots__ = ("job_id", "batch_tokens", "K", "bs", "num_variants")

    def __init__(self, job_id: int, batch_tokens: List[List[int]],
                 K: int, bs: int, num_variants: int = 1):
        self.job_id = job_id
        self.batch_tokens = batch_tokens
        self.K = K
        self.bs = bs
        self.num_variants = num_variants


class _GhostPool:
    """Pool of ghost CPU workers for parallel diverse corpus lookups.

    Each worker runs in its own thread, performing corpus lookups with
    different context windows for diversity. Workers share the C++ trie
    (GIL-free via pybind11 call_guard) and write to independent pinned
    HSA buffers for zero-copy GPU reads.

    With max_workers=1 (or active_count=1), behavior is identical to the
    original single-thread ghost system — fully backward compatible.

    Diversity strategy:
      Worker 0: full context window (standard, highest quality)
      Worker 1: half context window (more recent patterns, wider matches)
      Worker 2: double context window (more history, more specific matches)
      Worker 3+: offset context by worker_id * stride
    """

    def __init__(self, max_workers: int, max_batch_size: int, K_alloc: int,
                 max_match_window: int, num_ghost_variants: int):
        self.max_workers = max(1, min(max_workers, 8))
        self._active_count = 0
        self._max_match_window = max_match_window
        self._num_variants = num_ghost_variants
        self._round_id = 0
        self._job_counter = 0
        self.patched_positions: list = []

        # Per-worker state — each worker gets a single-slot pinned buffer
        self._bufs = [
            _GhostBuffer(max_batch_size, K_alloc, num_buffers=1)
            for _ in range(self.max_workers)
        ]
        self._threads: list = [None] * self.max_workers
        self._stops = [threading.Event() for _ in range(self.max_workers)]
        self._requests = [threading.Event() for _ in range(self.max_workers)]
        self._jobs: list = [None] * self.max_workers
        self._job_locks = [threading.Lock() for _ in range(self.max_workers)]
        # Worker state: 0=IDLE, 1=BUILDING, 2=READY
        self._states = [0] * self.max_workers
        self._state_locks = [threading.Lock() for _ in range(self.max_workers)]
        self._active_job_ids = [-1] * self.max_workers

        # References set after construction via set_corpus()
        self._corpus = None
        self._neg_filter = None
        self._neg_controller = None

    def set_corpus(self, corpus, neg_filter, neg_controller):
        """Set references to corpus and negative filter (called after corpus creation)."""
        self._corpus = corpus
        self._neg_filter = neg_filter
        self._neg_controller = neg_controller

    @property
    def active_count(self) -> int:
        return self._active_count

    def start(self, n: int = 1):
        """Start n worker threads (capped at max_workers)."""
        self._active_count = max(1, min(n, self.max_workers))
        try:
            avail = sorted(os.sched_getaffinity(0))
        except OSError:
            avail = []
        for i in range(self._active_count):
            if self._threads[i] is not None and self._threads[i].is_alive():
                continue
            self._stops[i].clear()
            self._threads[i] = threading.Thread(
                target=self._worker_loop, args=(i,),
                name=f"phantom-ghost-{i}", daemon=True,
            )
            self._threads[i].start()
            # Pin to distinct CPU cores (last N available cores)
            if len(avail) > 1:
                core = avail[-(i + 1)] if i + 1 <= len(avail) else avail[-1]
                try:
                    os.sched_setaffinity(self._threads[i].native_id, {core})
                    logger.info("PHANTOM: ghost-%d → core %d", i, core)
                except (OSError, AttributeError):
                    pass
        logger.info(
            "PHANTOM: ghost pool started (%d/%d workers)", self._active_count, self.max_workers
        )

    def stop(self):
        """Stop all worker threads."""
        for i in range(self.max_workers):
            self._stops[i].set()
            self._requests[i].set()
        for i in range(self.max_workers):
            if self._threads[i] is not None:
                self._threads[i].join(timeout=2.0)
                self._threads[i] = None
        self._active_count = 0

    def submit(self, batch_tokens: list, bs: int, K: int):
        """Fan out a ghost job to all active workers with diversity."""
        self._round_id += 1
        W = self._max_match_window
        for i in range(self._active_count):
            if i == 0:
                job_tokens = batch_tokens
            elif i == 1:
                half = max(W // 2, 4)
                job_tokens = [t[-half:] if len(t) > half else t for t in batch_tokens]
            elif i == 2:
                wide = min(W * 2, 128)
                job_tokens = [t[-wide:] if len(t) > wide else t for t in batch_tokens]
            else:
                offset = (i - 2) * max(W // 4, 2)
                job_tokens = [t[offset:] if len(t) > offset else t for t in batch_tokens]

            self._job_counter += 1
            job = _GhostJob(
                job_id=self._job_counter,
                batch_tokens=job_tokens,
                K=K, bs=bs,
                num_variants=self._num_variants if i == 0 else 1,
            )
            with self._job_locks[i]:
                self._jobs[i] = job
            self._requests[i].set()

    def get_first_ready(self, bs: int, timeout: float = 0.002):
        """Return (worker_id, draft_tokens_gpu, tree_mask_gpu) or None."""
        deadline = time.monotonic() + timeout
        while True:
            for i in range(self._active_count):
                with self._state_locks[i]:
                    if self._states[i] == 2 and self._bufs[i].active_bs == bs:
                        buf = self._bufs[i]
                        K = buf.active_k
                        draft = buf.gpu_tokens[:bs * K].cuda(non_blocking=True)
                        mask = buf.gpu_mask[:bs * K * K].cuda(non_blocking=True)
                        # DMA must complete before worker can reuse the buffer
                        torch.cuda.synchronize()
                        buf.consume()
                        self._states[i] = 0
                        return i, draft, mask
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0001)

    def scale_to(self, n: int):
        """Adjust active worker count. Starts/stops threads as needed."""
        n = max(1, min(n, self.max_workers))
        if n > self._active_count:
            old = self._active_count
            self._active_count = n
            try:
                avail = sorted(os.sched_getaffinity(0))
            except OSError:
                avail = []
            for i in range(old, n):
                if self._threads[i] is not None and self._threads[i].is_alive():
                    continue
                self._stops[i].clear()
                self._threads[i] = threading.Thread(
                    target=self._worker_loop, args=(i,),
                    name=f"phantom-ghost-{i}", daemon=True,
                )
                self._threads[i].start()
                if len(avail) > 1:
                    core = avail[-(i + 1)] if i + 1 <= len(avail) else avail[-1]
                    try:
                        os.sched_setaffinity(self._threads[i].native_id, {core})
                    except (OSError, AttributeError):
                        pass
            logger.info("PHANTOM: scaled UP to %d ghost workers", n)
        elif n < self._active_count:
            for i in range(n, self._active_count):
                self._stops[i].set()
                self._requests[i].set()
            for i in range(n, self._active_count):
                if self._threads[i] is not None:
                    self._threads[i].join(timeout=2.0)
                    self._threads[i] = None
                self._stops[i].clear()
            self._active_count = n
            logger.info("PHANTOM: scaled DOWN to %d ghost workers", n)

    def reset(self):
        """Reset all pool state (called from clear_cache_pool)."""
        self._round_id = 0
        self._job_counter = 0
        self._active_job_ids = [-1] * self.max_workers
        for i in range(self.max_workers):
            with self._state_locks[i]:
                self._states[i] = 0
        self.patched_positions = []

    # ── Worker loop (runs in each ghost thread) ──

    def _worker_loop(self, wid: int):
        """Per-worker ghost loop: wait → build → mark ready."""
        while not self._stops[wid].is_set():
            self._requests[wid].wait(timeout=0.1)
            if self._stops[wid].is_set():
                break
            self._requests[wid].clear()

            with self._job_locks[wid]:
                job = self._jobs[wid]
                self._jobs[wid] = None
            if job is None:
                continue

            with self._state_locks[wid]:
                self._states[wid] = 1  # BUILDING

            try:
                req_drafts, mask = self._corpus.batch_get(job.batch_tokens)
                bs, K = job.bs, job.K
                if len(req_drafts) != bs * K:
                    with self._state_locks[wid]:
                        self._states[wid] = 0
                    continue

                # Scan+patch only on worker 0 (primary context)
                patched = []
                if (wid == 0 and job.num_variants > 1
                        and self._neg_filter is not None
                        and self._neg_filter.size > 0
                        and self._neg_controller is not None
                        and self._neg_controller.is_active):
                    bad = self._scan_bad(req_drafts, bs, K)
                    if bad:
                        req_drafts, mask, patched = self._patch_bad(
                            job, req_drafts, mask, bad, bs, K
                        )
                if wid == 0:
                    self.patched_positions = patched

                # Copy to pinned buffer and mark ready
                buf = self._bufs[wid]
                buf.cpu_tokens[:bs * K].copy_(torch.from_numpy(req_drafts))
                buf.cpu_mask[:bs * K * K].copy_(torch.from_numpy(mask))
                self._active_job_ids[wid] = job.job_id
                buf.swap(bs, k=K)

                with self._state_locks[wid]:
                    self._states[wid] = 2  # READY

            except Exception as e:
                logger.debug("PHANTOM ghost-%d error: %s", wid, e)
                with self._state_locks[wid]:
                    self._states[wid] = 0

    def _scan_bad(self, drafts: np.ndarray, bs: int, K: int) -> list:
        """Phase 2: identify draft positions containing known-bad bigrams."""
        bad = []
        for r in range(bs):
            seq = drafts[r * K: (r + 1) * K]
            for i in range(len(seq) - 1):
                tok_a, tok_b = int(seq[i]), int(seq[i + 1])
                if tok_a != 0 and tok_b != 0 and self._neg_filter.query_bigram(tok_a, tok_b):
                    bad.append((r, i + 1))
        return bad

    def _patch_bad(self, job: _GhostJob, drafts: np.ndarray,
                   mask: np.ndarray, bad_positions: list,
                   bs: int, K: int):
        """Phase 3: cherry-pick replacement tokens for flagged positions."""
        W = self._max_match_window
        alt_windows = [min(W * 2, 128), max(W // 2, 4)]
        patched = []
        for win in alt_windows:
            alt_tokens = [t[-win:] if len(t) > win else t for t in job.batch_tokens]
            try:
                alt_drafts, alt_mask = self._corpus.batch_get(alt_tokens)
            except Exception:
                continue
            if len(alt_drafts) != bs * K:
                continue
            remaining = []
            for r, pos in bad_positions:
                alt_tok = int(alt_drafts[r * K + pos])
                orig_tok = int(drafts[r * K + pos])
                if alt_tok == 0 or alt_tok == orig_tok:
                    remaining.append((r, pos))
                    continue
                prev_tok = int(drafts[r * K + pos - 1]) if pos > 0 else 0
                if prev_tok != 0 and self._neg_filter.query_bigram(prev_tok, alt_tok):
                    remaining.append((r, pos))
                    continue
                drafts[r * K + pos] = alt_tok
                m_off = r * K * K + pos * K
                mask[m_off:m_off + K] = alt_mask[m_off:m_off + K]
                patched.append((r, pos, orig_tok, alt_tok))
            bad_positions = remaining
            if not bad_positions:
                break
        return drafts, mask, patched


class _GhostScaler:
    """Throughput-based hill-climbing scaler for ghost worker count.

    Measures accepted tokens/second over a sliding window. Periodically
    tentatively adds a worker, measures the throughput delta, and keeps
    the change only if positive. Hysteresis cooldown prevents oscillation.
    """

    def __init__(self, pool: _GhostPool, measure_window: int = 100,
                 cooldown: int = 200):
        self._pool = pool
        self._measure_window = measure_window
        self._cooldown = cooldown
        self._round = 0
        self._cooldown_remaining = 0
        self._baseline_tps = 0.0
        self._trial_active = False
        self._trial_direction = 0
        self._tps_window: deque = deque(maxlen=measure_window)
        self._last_time = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._pool.max_workers > 1

    def record(self, accepted_tokens: int):
        """Called each round with number of accepted tokens."""
        if not self.enabled:
            return
        now = time.monotonic()
        dt = now - self._last_time
        if dt > 0:
            self._tps_window.append(accepted_tokens / dt)
        self._last_time = now
        self._round += 1

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return
        if self._round % self._measure_window != 0:
            return

        current_tps = sum(self._tps_window) / max(len(self._tps_window), 1)

        if self._trial_active:
            if current_tps > self._baseline_tps * 1.02:
                self._baseline_tps = current_tps
                logger.info(
                    "PHANTOM scaler: kept change (%.1f t/s, %d workers)",
                    current_tps, self._pool.active_count,
                )
            else:
                self._pool.scale_to(self._pool.active_count - self._trial_direction)
                self._cooldown_remaining = self._cooldown
                logger.info(
                    "PHANTOM scaler: reverted (%.1f ≤ %.1f t/s), cooldown %d",
                    current_tps, self._baseline_tps, self._cooldown,
                )
            self._trial_active = False
        else:
            self._baseline_tps = current_tps
            if self._pool.active_count < self._pool.max_workers:
                self._trial_direction = 1
                self._pool.scale_to(self._pool.active_count + 1)
                self._trial_active = True
                logger.info(
                    "PHANTOM scaler: trial +1 worker (now %d)", self._pool.active_count
                )

    def reset(self):
        self._round = 0
        self._cooldown_remaining = 0
        self._baseline_tps = 0.0
        self._trial_active = False
        self._tps_window.clear()
        self._last_time = time.monotonic()


class _DynamicGammaController:
    """EMA-based adaptive speculative draft length (γ) controller.

    Calibrates to the model's characteristic acceptance rate during a warmup
    period, then adjusts γ using relative thresholds. This avoids the problem
    of fixed thresholds (e.g. 0.7 grow / 0.3 shrink) which would pin γ to
    minimum on low-acceptance models like Bonsai-4B (~15% n-gram acceptance).

    Lifecycle:
      - Created in PhantomWorker.__init__
      - update() called each verification round with per-token acceptance
      - freeze() when auto-fallback activates (no signal available)
      - unfreeze() when auto-fallback deactivates (decay EMA toward baseline)
      - reset() on clear_cache_pool (new context, new model)
    """

    def __init__(self, initial_gamma: int, min_gamma: int, max_gamma: int,
                 alpha: float = 0.15, calibration_rounds: int = 20,
                 warmup_rounds: int = 5, cooldown_rounds: int = 3):
        self.gamma = initial_gamma
        self.initial_gamma = initial_gamma
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self._alpha = alpha
        self._calibration_rounds = calibration_rounds
        self._warmup = warmup_rounds
        self._cooldown_max = cooldown_rounds

        # Internal state
        self._accept_ema: float = -1.0  # sentinel: uninitialized
        self._baseline: float = -1.0  # learned after calibration
        self._rounds: int = 0
        self._cooldown: int = 0
        self._frozen: bool = False

    def update(self, accept_rate: float) -> int:
        """Feed acceptance rate (num_accepted / (bs * K)), returns new γ."""
        if self._frozen:
            return self.gamma

        self._rounds += 1

        # Initialize EMA on first sample
        if self._accept_ema < 0:
            self._accept_ema = accept_rate
        else:
            self._accept_ema = ((1 - self._alpha) * self._accept_ema
                                + self._alpha * accept_rate)

        # Learn baseline after calibration window
        if self._rounds == self._calibration_rounds and self._baseline < 0:
            self._baseline = max(self._accept_ema, 0.01)
            logger.info("PHANTOM γ-ctrl: baseline acceptance=%.3f (from %d rounds)",
                        self._baseline, self._calibration_rounds)

        # No adaptation during warmup or cooldown
        if self._rounds < self._warmup:
            return self.gamma
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.gamma

        a = self._accept_ema
        base = self._baseline if self._baseline >= 0 else max(a, 0.01)

        # Relative thresholds calibrated to model's acceptance regime
        grow_fast = base * 2.0    # 2× baseline → aggressive grow
        grow_mild = base * 1.3    # 1.3× baseline → mild grow
        shrink_mild = base * 0.6  # 0.6× baseline → mild shrink
        shrink_fast = base * 0.3  # 0.3× baseline → aggressive shrink

        if a >= grow_fast:
            delta = 2
        elif a >= grow_mild:
            delta = 1
        elif a <= shrink_fast:
            delta = -2
        elif a <= shrink_mild:
            delta = -1
        else:
            delta = 0

        if delta != 0:
            old = self.gamma
            self.gamma = max(self.min_gamma, min(self.max_gamma, self.gamma + delta))
            if self.gamma != old:
                self._cooldown = self._cooldown_max
                logger.debug("PHANTOM γ-ctrl: %d→%d (ema=%.3f, base=%.3f)",
                             old, self.gamma, a, base)

        return self.gamma

    def freeze(self):
        """Freeze updates (called when auto-fallback activates)."""
        self._frozen = True

    def unfreeze(self):
        """Unfreeze updates (called when auto-fallback deactivates).

        Decays EMA toward baseline to avoid stale signal after a long
        fallback period where no acceptance data was collected.
        """
        self._frozen = False
        if self._baseline >= 0 and self._accept_ema >= 0:
            self._accept_ema = 0.5 * self._accept_ema + 0.5 * self._baseline

    def reset(self, initial_gamma: Optional[int] = None):
        """Full reset (called on clear_cache_pool / context switch)."""
        self.gamma = initial_gamma if initial_gamma is not None else self.initial_gamma
        self._accept_ema = -1.0
        self._baseline = -1.0
        self._rounds = 0
        self._cooldown = 0
        self._frozen = False

    @property
    def accept_ema(self) -> float:
        """Current acceptance EMA (for logging/diagnostics)."""
        return self._accept_ema if self._accept_ema >= 0 else 0.0

    @property
    def baseline(self) -> float:
        """Learned baseline acceptance (for logging/diagnostics)."""
        return self._baseline if self._baseline >= 0 else 0.0


class _FrozenCorpus:
    """Read-only snapshot of an n-gram corpus for lock-free CPU access.

    Created once at prefill from the live NgramCorpus, then frozen.
    The ghost thread reads from this without any synchronization.
    """

    def __init__(self, corpus, draft_token_num: int, max_match_window: int):
        self._corpus = corpus
        self.draft_token_num = draft_token_num
        self.max_match_window = max_match_window
        self._frozen = False

    def freeze(self):
        """Freeze the corpus (sync any pending inserts, then mark read-only)."""
        self._corpus.synchronize()
        self._frozen = True
        logger.info("PHANTOM: corpus frozen (%d draft tokens)", self.draft_token_num)

    def batch_get(self, batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Read-only lookup — safe to call from any thread after freeze()."""
        token_arr, mask_arr = self._corpus.batch_get(batch_tokens)
        return token_arr, mask_arr

    def insert(self, batch_tokens: List[List[int]]):
        """Insert tokens into corpus (only before freeze)."""
        if not self._frozen:
            self._corpus.batch_put(batch_tokens)


class PhantomWorker:
    """PHANTOM — HSA zero-copy ghost-draft speculative worker.

    Combines:
    - Frozen n-gram corpus (constant cache, no locks)
    - CPU ghost thread (builds trees in pinned HSA memory)
    - Double-buffered zero-copy GPU reads
    - Standard NgramVerifyInput tree verification

    The "X" adapts to system characteristics:
    - PCIe bandwidth determines copy latency (zero on HSA, ~μs on non-HSA)
    - CPU core count determines ghost thread throughput
    - System RAM determines corpus capacity

    Args:
        server_args: Server configuration.
        gpu_id: GPU device index.
        tp_rank: Tensor-parallel rank.
        dp_rank: Data-parallel rank.
        moe_ep_rank: MoE expert-parallel rank.
        attn_cp_rank: Attention context-parallel rank.
        moe_dp_rank: MoE data-parallel rank.
        nccl_port: NCCL communication port.
        target_worker: The main model worker.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.page_size = server_args.page_size
        self.draft_token_num = server_args.speculative_num_draft_tokens or 8
        self.max_batch_size = target_worker.max_running_requests
        # max_match_window controls how far back we look for n-gram matches
        self.max_match_window_size = getattr(
            server_args, "speculative_ngram_max_trie_depth", 18
        )

        # Create the n-gram corpus (will be frozen after first prefill)
        from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

        raw_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=self.draft_token_num,
        )
        self.corpus = _FrozenCorpus(
            raw_corpus, self.draft_token_num, self.max_match_window_size
        )

        # N-buffered pinned memory (HSA zero-copy on AMD)
        # Allocate for _max_draft (not draft_token_num) so dynamic γ growth
        # doesn't exceed buffer bounds at runtime.
        self._max_draft_alloc = min(self.draft_token_num * 2, 16)

        # GPU-side tensors for tree reconstruction (these must be in VRAM)
        # Also sized for _max_draft_alloc to support dynamic γ.
        K_alloc = self._max_draft_alloc
        max_bs = self.max_batch_size
        self.positions = torch.empty(max_bs * K_alloc, dtype=torch.int64, device=self.device)
        self.retrive_index = torch.empty((max_bs, K_alloc), dtype=torch.int64, device=self.device)
        self.retrive_next_token = torch.empty((max_bs, K_alloc), dtype=torch.int64, device=self.device)
        self.retrive_next_sibling = torch.empty((max_bs, K_alloc), dtype=torch.int64, device=self.device)

        self._corpus_frozen = False

        # Multi-variant ghost lookups (scan+patch pipeline variant count)
        self._num_ghost_variants = getattr(server_args, 'phantom_num_ghosts', 1)

        # Negative filter (bloom filter for rejected bigrams)
        self._neg_filter = _NegativeFilter()

        # Adaptive controller for negative filter pipeline (phases 2+3)
        self._neg_controller = AdaptiveThresholdController(
            initial_threshold=2000.0,
            min_threshold=500.0,
            max_threshold=5000.0,
            ema_alpha=0.1,
            precision_target=0.70,
            precision_floor=0.30,
            warmup_steps=20,
            step_size=200.0,
            backoff_cooldown=40,
        )

        # Ghost pool — manages 1-N ghost CPU workers with diversity
        max_ghosts = max(1, getattr(server_args, 'phantom_num_ghosts', 1))
        self._ghost_pool = _GhostPool(
            max_workers=max_ghosts,
            max_batch_size=self.max_batch_size,
            K_alloc=K_alloc,
            max_match_window=self.max_match_window_size,
            num_ghost_variants=self._num_ghost_variants,
        )
        self._ghost_pool.set_corpus(self.corpus, self._neg_filter, self._neg_controller)

        # Ghost scaler — throughput-based adaptive worker count
        self._ghost_scaler = _GhostScaler(self._ghost_pool)

        # Quantization telemetry — collects signals for offline calibration
        telem_dir = os.environ.get(
            "PHANTOM_QUANT_TELEM_DIR", "/tmp/phantom_quant_telem"
        )
        self._quant_telem = _QuantTelemetry(
            snapshot_interval=500,
            snapshot_dir=telem_dir,
        )

        # Stats
        self.total_rounds = 0
        self.ghost_hits = 0  # ghost tree was ready when GPU needed it
        self.ghost_misses = 0  # ghost tree wasn't ready, built synchronously
        self._log_interval = 50

        # Adaptive metrics (sliding window)
        self._accept_window = deque(maxlen=20)  # recent acceptance rates
        self._corpus_hit_count = 0  # non-trivial drafts from corpus
        self._corpus_miss_count = 0  # all-zero / failed drafts
        self._corpus_insert_count = 0  # total corpus inserts

        # Auto-fallback state
        self._fallback_active = False
        self._fallback_streak = 0  # consecutive low-acceptance rounds
        self._fallback_probe_counter = 0
        _FALLBACK_THRESHOLD = 0.4
        _FALLBACK_STREAK_LIMIT = 10
        _REENABLE_THRESHOLD = 0.5
        _PROBE_INTERVAL = 50
        self._fb_threshold = _FALLBACK_THRESHOLD
        self._fb_streak_limit = _FALLBACK_STREAK_LIMIT
        self._fb_reenable = _REENABLE_THRESHOLD
        self._fb_probe_interval = _PROBE_INTERVAL

        # Dynamic γ state — cap at buffer allocation size
        self._initial_draft_num = self.draft_token_num
        self._min_draft = 2
        self._max_draft = self._max_draft_alloc

        # Dynamic γ controller — EMA-based with baseline calibration
        self._gamma_ctrl = _DynamicGammaController(
            initial_gamma=self.draft_token_num,
            min_gamma=self._min_draft,
            max_gamma=self._max_draft,
        )

        # Detect HSA capability
        self._is_hsa = self._detect_hsa()

        # SOK — Self-Optimizing Kernel subsystem (Phase F1: cache + prewarm, F2: telemetry)
        self._sok_config = SOKConfig()
        self._sok_profile = None
        self._sok_telemetry = None
        try:
            self._sok_fingerprint = detect_fingerprint()
            self._sok_cache = KernelCache(self._sok_config, self._sok_fingerprint)
            self._sok_cache.load_manifest()
            self._sok_cache.load_hot_shapes()
            # Bootstrap from existing Triton cache on first ever run
            if self._sok_cache.get_stats()["entries"] == 0:
                scanned = self._sok_cache.scan_triton_cache()
                if scanned > 0:
                    self._sok_cache.save_manifest()
                    logger.info("PHANTOM-SOK: bootstrapped %d entries from Triton cache", scanned)
            # F2: Shape profile + telemetry
            self._sok_profile = ShapeProfile()
            profile_path = self._sok_cache.cache_dir / self._sok_fingerprint.hex_digest / "shape_profiles.json"
            self._sok_profile.load(profile_path)
            self._sok_telemetry = DispatchTelemetry(
                self._sok_config, self._sok_cache, self._sok_profile,
            )
            # Background prewarm (non-blocking)
            self._sok_prewarm_thread = threading.Thread(
                target=self._sok_cache.prewarm, daemon=True, name="sok-prewarm"
            )
            self._sok_prewarm_thread.start()
            sok_stats = self._sok_cache.get_stats()
            prof_stats = self._sok_profile.get_stats()
            logger.info(
                "PHANTOM-SOK: loaded manifest (%d entries, %d hot shapes, %d shape profiles, fp=%s)",
                sok_stats["entries"], sok_stats["hot_shapes"],
                prof_stats["shapes"], self._sok_fingerprint.hex_digest[:12],
            )
        except Exception as e:
            logger.warning("PHANTOM-SOK: init failed, running without cache: %s", e)
            self._sok_cache = None
            self._sok_fingerprint = None
            self._sok_profile = None
            self._sok_telemetry = None

        logger.info(
            "PHANTOM worker: draft_tokens=%d (max=%d), max_bs=%d, HSA=%s, "
            "ghost_pool=%d workers, pinned_buf=%.1f KB per worker",
            self.draft_token_num, self._max_draft_alloc, max_bs, self._is_hsa,
            self._ghost_pool.max_workers,
            self._ghost_pool._bufs[0]._tokens[0].nelement() * 8 / 1024,
        )

    @staticmethod
    def _detect_hsa() -> bool:
        """Detect if we're running on AMD ROCm with HSA support."""
        try:
            return hasattr(torch.version, "hip") and torch.version.hip is not None
        except Exception:
            return False

    # ---- Proxy attributes ----

    @property
    def max_running_requests(self):
        return self.target_worker.max_running_requests

    @property
    def model_config(self):
        return self.target_worker.model_runner.model_config

    def get_memory_pool(self):
        return self.target_worker.get_memory_pool()

    def clear_cache_pool(self):
        # Persist SOK manifest + shape profiles before clearing
        if getattr(self, "_sok_cache", None) is not None:
            self._sok_cache.save_manifest()
            self._sok_cache.save_hot_shapes()
        if getattr(self, "_sok_profile", None) is not None and getattr(self, "_sok_fingerprint", None) is not None:
            profile_path = self._sok_cache.cache_dir / self._sok_fingerprint.hex_digest / "shape_profiles.json"
            self._sok_profile.save(profile_path)
        self._ghost_pool.stop()
        self.corpus._corpus.reset()
        self._corpus_frozen = False
        self.total_rounds = 0
        self.ghost_hits = 0
        self.ghost_misses = 0
        self._accept_window.clear()
        self._corpus_hit_count = 0
        self._corpus_miss_count = 0
        self._corpus_insert_count = 0
        self._fallback_active = False
        self._fallback_streak = 0
        self._fallback_probe_counter = 0
        self._gamma_ctrl.reset(self._initial_draft_num)
        self.draft_token_num = self._initial_draft_num
        self._neg_filter.reset()
        self._neg_controller = AdaptiveThresholdController(
            initial_threshold=2000.0,
            min_threshold=500.0,
            max_threshold=5000.0,
            ema_alpha=0.1,
            precision_target=0.70,
            precision_floor=0.30,
            warmup_steps=20,
            step_size=200.0,
            backoff_cooldown=40,
        )
        self._ghost_pool.set_corpus(self.corpus, self._neg_filter, self._neg_controller)
        self._ghost_pool.reset()
        self._ghost_scaler.reset()
        self._quant_telem.reset()

    # ---- Ghost pool management ----

    def _start_ghost_pool(self):
        """Start ghost worker pool (delegates to _GhostPool)."""
        self._ghost_pool.start(n=1)

    def _stop_ghost_pool(self):
        """Stop ghost worker pool."""
        self._ghost_pool.stop()

    def _submit_ghost_request(self, batch: ScheduleBatch):
        """Submit a ghost job to the pool (fans out to all active workers)."""
        batch_tokens = []
        for req in batch.reqs:
            tokens = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(tokens)
        self._ghost_pool.submit(batch_tokens, len(batch_tokens), self.draft_token_num)

    # ---- Helpers ----

    @staticmethod
    def _efficient_concat_last_n(seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]
        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _update_negative_filter(self, draft_tokens: torch.Tensor,
                                accept_lens: list, bs: int, K: int):
        """Insert rejected draft bigrams into the negative bloom filter.

        Only inserts the bigram AT the rejection boundary (last accepted →
        first rejected), not the entire rejected suffix. Tokens beyond the
        boundary weren't verified by the target model, so treating them as
        "bad" would poison the filter with unverified patterns.
        """
        try:
            draft_cpu = draft_tokens.cpu().numpy() if draft_tokens.is_cuda else draft_tokens.numpy()
            for r in range(bs):
                acc_len = int(accept_lens[r]) if r < len(accept_lens) else 0
                if acc_len <= 0 or acc_len >= K:
                    continue
                # The bigram at the rejection point: token[acc_len-1] → token[acc_len]
                idx_accepted = r * K + acc_len - 1
                idx_rejected = r * K + acc_len
                if idx_rejected < len(draft_cpu):
                    tok_a, tok_b = int(draft_cpu[idx_accepted]), int(draft_cpu[idx_rejected])
                    if tok_a != 0 and tok_b != 0:
                        self._neg_filter.insert_bigram(tok_a, tok_b)
        except Exception:
            pass  # non-critical — filter is best-effort

    def _record_patch_outcomes(self, accept_lens: list, K: int):
        """Compare patched positions against verify accept_lengths.

        A patch at position P in request R is "correct" if P < accept_length[R],
        meaning the replacement token was accepted by the target model.

        Feeds outcomes into the adaptive controller which adjusts:
          - Whether scan+patch phases should keep running
          - The bloom filter's age_limit (via threshold → age_limit sync)

        Also stores preference pairs for future contrastive training:
          (original_token, replacement_token, was_accepted)
        """
        patched = self._ghost_pool.patched_positions
        if not patched:
            # No patches this round — still count the step so warmup advances
            self._neg_controller.record_outcome(0, 0)
            return

        n_patched = len(patched)
        n_correct = 0
        for r, pos, orig_tok, new_tok in patched:
            acc_len = int(accept_lens[r]) if r < len(accept_lens) else 0
            if pos < acc_len:
                n_correct += 1

        self._neg_controller.record_outcome(n_patched, n_correct)

        # Sync controller threshold → bloom filter age_limit
        # Lower threshold = keep more history = more aggressive filtering
        self._neg_filter._age_limit = max(int(self._neg_controller.threshold), 200)

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:

        # On extend/prefill: seed the corpus and freeze it
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._handle_extend(batch)

        bs = batch.batch_size()
        if bs == 0 or batch.forward_mode.is_idle():
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        self.total_rounds += 1
        K = self.draft_token_num

        # ── Auto-fallback: skip spec decode when acceptance is consistently low ──
        if self._fallback_active:
            self._fallback_probe_counter += 1
            if self._fallback_probe_counter < self._fb_probe_interval:
                return self._fallback_target_only(batch)
            # Probe round: try spec decode, check if acceptance recovered
            self._fallback_probe_counter = 0

        # Try to use the ghost pool's pre-built tree (first-ready selection)
        ready = self._ghost_pool.get_first_ready(bs, timeout=0.002)

        if ready is not None:
            wid, draft_tokens_gpu, tree_mask_gpu = ready
            self.ghost_hits += 1
        else:
            # No worker ready — build synchronously (fallback)
            self.ghost_misses += 1
            draft_tokens_gpu, tree_mask_gpu = self._build_tree_sync(batch)

        # Reconstruct tree indices (must be in VRAM and contiguous)
        positions = self.positions[:bs * K]
        retrive_index = self.retrive_index[:bs, :K].contiguous()
        retrive_next_token = self.retrive_next_token[:bs, :K].contiguous()
        retrive_next_sibling = self.retrive_next_sibling[:bs, :K].contiguous()

        try:
            if reconstruct_indices_from_tree_mask is None:
                global _sgl_kernel_warned
                if not _sgl_kernel_warned:
                    logger.error(
                        "PHANTOM: sgl_kernel.speculative not installed — "
                        "tree reconstruction unavailable. PHANTOM speculative "
                        "decoding is effectively disabled (falling back to "
                        "target-only decode). Install sgl_kernel to enable."
                    )
                    _sgl_kernel_warned = True
                return self._fallback_target_only(batch)

            reconstruct_indices_from_tree_mask(
                tree_mask_gpu,
                batch.seq_lens,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                bs,
                K,
            )
        except Exception as e:
            logger.warning("PHANTOM reconstruct_indices failed: %s", e)
            return self._fallback_target_only(batch)

        # Build full attention mask
        USE_FULL_MASK = True
        if USE_FULL_MASK:
            mask_np = tree_mask_gpu.cpu().numpy().reshape(bs, K, K)
            tree_mask_parts = []
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix_mask = torch.ones((K, seq_len - 1), device=self.device)
                tree_part = torch.from_numpy(mask_np[i]).to(self.device)
                full_mask = torch.cat((prefix_mask, tree_part), dim=1).to(torch.bool)
                tree_mask_parts.append(full_mask.flatten())
            tree_mask_final = torch.cat(tree_mask_parts, dim=0)
        else:
            tree_mask_final = tree_mask_gpu

        # Set up verification — use None for tree_mask to get causal attention
        # instead of custom mask (avoids RDNA2 masked-lane OOB in mask pointer arithmetic)
        original_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_tokens_gpu,
            None,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            K,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

        # Run target verification on GPU
        model_worker_batch = batch.get_model_worker_batch()

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )

            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph

            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted = verify_input.verify(
                batch, logits_output, self.page_size, None
            )
            accept_lens = verify_input.accept_length
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_algorithm = original_algo

            # CRITICAL: verify() updates batch.seq_lens/seq_lens_cpu but NOT
            # seq_lens_sum or orig_seq_lens.  In PHANTOM's reused-batch pattern,
            # stale seq_lens_sum causes kv_indices underallocation in
            # triton_backend → OOB write → HIP 700.
            batch.seq_lens_sum = batch.seq_lens.sum().item()
            if batch.orig_seq_lens is not None:
                batch.orig_seq_lens = batch.seq_lens.clone()

            # ── Track acceptance rate ──
            accept_rate = num_accepted / max(bs * K, 1)
            self._accept_window.append(accept_rate)

            # ── Ghost scaler: record throughput sample ──
            self._ghost_scaler.record(num_accepted)

            # Save local ref before clearing batch state — telemetry needs draft tokens
            verify_draft_tokens = verify_input.draft_token

            # ── Feed rejected patterns into negative filter ──
            self._update_negative_filter(
                verify_draft_tokens, accept_lens, bs, K
            )

            # ── Adaptive controller: measure patch outcomes ──
            self._record_patch_outcomes(accept_lens, K)

            # ── Quantization telemetry: capture free signals ──
            self._quant_telem.record(
                logits=logits_output.next_token_logits,
                hidden_states=logits_output.hidden_states,
                draft_tokens=verify_draft_tokens,
                verified_ids=next_token_ids,
                accept_lens=accept_lens,
                bs=bs, K=K,
            )

            # CRITICAL: Clear stale batch state from verify round.
            # prepare_for_decode() returns early for spec algorithms, so
            # batch.spec_info (with stale positions), batch.input_ids (K-wide
            # draft tensor), and batch.output_ids persist into the next round
            # causing position corruption and garbled output.
            batch.spec_info = None
            batch.input_ids = None
            batch.output_ids = None

            # ── Auto-fallback logic ──
            if accept_rate < self._fb_threshold:
                self._fallback_streak += 1
            else:
                self._fallback_streak = 0

            if self._fallback_streak >= self._fb_streak_limit and not self._fallback_active:
                self._fallback_active = True
                self._fallback_probe_counter = 0
                self._gamma_ctrl.freeze()
                logger.info("PHANTOM: auto-fallback ENABLED (acceptance=%.2f for %d rounds)",
                            accept_rate, self._fallback_streak)
            elif self._fallback_active and accept_rate >= self._fb_reenable:
                self._fallback_active = False
                self._fallback_streak = 0
                self._gamma_ctrl.unfreeze()
                logger.info("PHANTOM: auto-fallback DISABLED (acceptance recovered to %.2f)",
                            accept_rate)

            # ── Dynamic γ: adaptive draft length via EMA controller ──
            self.draft_token_num = self._gamma_ctrl.update(accept_rate)

            # Update corpus with accepted tokens (keeps it growing)
            self._update_corpus(batch)

            # Submit next ghost request (CPU starts building while we return)
            self._submit_ghost_request(batch)

            self._periodic_log()

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=num_accepted,
                can_run_cuda_graph=can_run_cuda_graph,
                accept_lens=accept_lens,
            )

        batch.spec_algorithm = original_algo
        return self._fallback_target_only(batch)

    # ---- Extend / prefill handling ----

    def _handle_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """On prefill: seed corpus with prompt tokens, freeze, start ghost thread."""
        # Seed corpus with prompt tokens
        batch_tokens = []
        for req in batch.reqs:
            put_ids = list(req.origin_input_ids) + list(req.output_ids)
            batch_tokens.append(put_ids)
        self.corpus.insert(batch_tokens)

        # Freeze corpus after first extend (makes it read-only for ghost thread)
        if not self._corpus_frozen:
            self.corpus.freeze()
            self._corpus_frozen = True
            self._start_ghost_pool()

        # Run normal extend on target
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )

        # Pre-submit first ghost request
        self._submit_ghost_request(batch)

        return result

    def _update_corpus(self, batch: ScheduleBatch):
        """Update corpus with newly generated tokens.

        NOTE: After freeze, we still allow inserts into the underlying corpus.
        The frozen flag was for the initial snapshot; ongoing inserts keep the
        corpus relevant as generation continues.  The ghost thread's reads
        and main thread's inserts go through the C++ corpus which handles
        its own internal synchronization via asyncInsert + synchronize.

        Token window is bounded to `max_match_window_size` to limit corpus
        memory growth. Only the most recent tokens per request are inserted.
        """
        max_window = min(
            self.max_match_window_size,
            self.corpus._corpus.draft_token_num * 3,
        )
        batch_tokens = []
        for req in batch.reqs:
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids,
                req.output_ids,
                max_window,
            )
            batch_tokens.append(put_ids)
        self.corpus._corpus.batch_put(batch_tokens)
        self._corpus_insert_count += 1

    def _build_tree_sync(self, batch: ScheduleBatch):
        """Synchronous fallback: build tree on CPU, copy to GPU."""
        bs = batch.batch_size()
        K = self.draft_token_num
        corpus_K = self._initial_draft_num  # corpus always returns this many

        self.corpus._corpus.synchronize()
        batch_tokens = []
        for req in batch.reqs:
            tokens = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(tokens)

        req_drafts, mask = self.corpus.batch_get(batch_tokens)
        # Track corpus hit rate (non-trivial = at least one non-zero draft token)
        if np.any(req_drafts != 0):
            self._corpus_hit_count += 1
        else:
            self._corpus_miss_count += 1

        # Corpus always returns corpus_K tokens per request; slice to current K
        if K < corpus_K:
            req_drafts_reshaped = req_drafts.reshape(bs, corpus_K)[:, :K].reshape(-1)
            mask_reshaped = mask.reshape(bs, corpus_K, corpus_K)[:, :K, :K].reshape(-1)
            req_drafts = np.ascontiguousarray(req_drafts_reshaped)
            mask = np.ascontiguousarray(mask_reshaped)
        elif K > corpus_K:
            # Dynamic γ grew beyond corpus K — pad with zeros
            padded = np.zeros(bs * K, dtype=req_drafts.dtype)
            padded_mask = np.zeros(bs * K * K, dtype=mask.dtype)
            for i in range(bs):
                padded[i * K:i * K + corpus_K] = req_drafts[i * corpus_K:(i + 1) * corpus_K]
            req_drafts = padded
            mask = padded_mask

        draft_tokens_gpu = torch.from_numpy(req_drafts).to(self.device, non_blocking=True)
        tree_mask_gpu = torch.from_numpy(mask).to(self.device, non_blocking=True)
        return draft_tokens_gpu, tree_mask_gpu

    def _fallback_target_only(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Fall back to a single target-model decode step (no speculation).

        This is called when ghost drafts are unavailable, blocked, or all filtered.
        Must fully prepare the batch for decode since prepare_for_decode() returns
        early when spec_algorithm is active — leaving input_ids, out_cache_loc,
        seq_lens, and kv tracking un-updated.
        """
        from sglang.srt.mem_cache.common import alloc_for_decode

        bs = batch.batch_size()
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = None
        batch.input_embeds = None
        batch.output_ids = None  # prevent stale speculative output_ids leaking

        # Clear prefill-only metadata (mirrors prepare_for_decode)
        if hasattr(batch, "attn_cp_metadata") and batch.attn_cp_metadata is not None:
            batch.attn_cp_metadata = None
        if hasattr(batch, "nsa_cp_metadata") and batch.nsa_cp_metadata is not None:
            batch.nsa_cp_metadata = None

        # input_ids may still be K-wide from prepare_for_verify; resize to bs
        batch.input_ids = torch.tensor(
            [req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
             for req in batch.reqs],
            dtype=torch.int64, device=self.device,
        )

        # Accumulate penalizer state for the decode token (mirrors prepare_for_decode)
        if batch.sampling_info and batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                batch.input_ids.to(torch.int64)
            )

        # Allocate exactly 1 KV cache slot per request (what prepare_for_decode does)
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )

        # Scheduler skips output_ids.append for spec v1 — do it here
        next_ids = result.next_token_ids
        if isinstance(next_ids, torch.Tensor):
            next_ids = next_ids.tolist()
        for req, tok in zip(batch.reqs, next_ids):
            req.output_ids.append(tok)

        # Advance seq_lens and per-request KV bookkeeping
        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens.add_(1)
        batch.seq_lens_sum += bs
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        return result

    def _periodic_log(self):
        if self.total_rounds % self._log_interval == 0 and self.total_rounds > 0:
            total = self.ghost_hits + self.ghost_misses
            ghost_pct = self.ghost_hits / max(total, 1) * 100
            avg_accept = (sum(self._accept_window) / len(self._accept_window)
                          if self._accept_window else 0.0)
            corpus_total = self._corpus_hit_count + self._corpus_miss_count
            corpus_pct = self._corpus_hit_count / max(corpus_total, 1) * 100
            neg_size = self._neg_filter.size if self._neg_filter else 0
            neg_age = self._neg_filter._count if self._neg_filter else 0
            ctrl = self._neg_controller.get_state()
            pool_active = self._ghost_pool.active_count
            logger.info(
                "PHANTOM stats: rounds=%d, ghost=%.1f%% (%d/%d), "
                "accept=%.2f (ema=%.3f base=%.3f), γ=%d, corpus_hit=%.1f%%, "
                "fallback=%s, buf=%s, neg_filter=%d (age_lim=%d), "
                "pool=%d/%d workers, patch_ema=%.2f τ=%.0f %s",
                self.total_rounds, ghost_pct, self.ghost_hits, total,
                avg_accept, self._gamma_ctrl.accept_ema, self._gamma_ctrl.baseline,
                self.draft_token_num, corpus_pct,
                "ON" if self._fallback_active else "off",
                "HSA" if self._is_hsa else "pinned",
                neg_size, self._neg_filter._age_limit,
                pool_active, self._ghost_pool.max_workers,
                ctrl["precision_ema"], ctrl["threshold"],
                "(warmup)" if ctrl["in_warmup"] else
                "(OFF)" if not ctrl["enabled"] else "",
            )
            # Quant telemetry summary (every 5th log interval = ~250 rounds)
            if self.total_rounds % (self._log_interval * 5) == 0:
                qt = self._quant_telem.get_summary()
                logger.info(
                    "PHANTOM quant telem: margin_acc=%.3f margin_rej=%.3f "
                    "confusion=%d ch_samples=%d pos_rate=%s",
                    qt["margin_acc"], qt["margin_rej"],
                    qt["confusion_pairs"], qt["channel_samples"],
                    qt["pos_rate"],
                )
            # SOK manifest save (every 500 rounds) + stats
            if self._sok_cache is not None and self.total_rounds % 500 == 0:
                self._sok_cache.save_manifest()
                self._sok_cache.save_hot_shapes()
                ss = self._sok_cache.get_stats()
                logger.info(
                    "PHANTOM-SOK: saved manifest (%d entries, hits=%d, misses=%d, hot=%d)",
                    ss["entries"], ss["hits"],
                    ss["misses"], ss["hot_shapes"],
                )
                # F2: persist shape profiles + telemetry summary
                if self._sok_profile is not None:
                    profile_path = self._sok_cache.cache_dir / self._sok_fingerprint.hex_digest / "shape_profiles.json"
                    self._sok_profile.save(profile_path)
                if self._sok_telemetry is not None:
                    ts = self._sok_telemetry.get_summary()
                    logger.info(
                        "PHANTOM-SOK telemetry: dispatches=%d, sampled=%d, "
                        "avg_us=%.1f, hit_rate=%.2f, shapes=%d",
                        ts["dispatches"], ts["sampled"],
                        ts["avg_latency_us"], ts["cache_hit_rate"],
                        ts["unique_shapes"],
                    )

    def __del__(self):
        try:
            # Persist SOK manifest + shape profiles on shutdown
            if getattr(self, "_sok_cache", None) is not None:
                self._sok_cache.save_manifest()
                self._sok_cache.save_hot_shapes()
            if getattr(self, "_sok_profile", None) is not None and getattr(self, "_sok_fingerprint", None) is not None:
                profile_path = self._sok_cache.cache_dir / self._sok_fingerprint.hex_digest / "shape_profiles.json"
                self._sok_profile.save(profile_path)
            self._stop_ghost_pool()
        except Exception:
            pass
