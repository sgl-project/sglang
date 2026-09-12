"""Graph cache and live draft-probability binding for compact verification."""

import atexit
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.speculative.compact_verify.peer_graph import ConditionalPeerVerify

logger = logging.getLogger(__name__)
_cache = OrderedDict()
_runtime_ok = None


@triton.jit
def _q_selected(
    Ptr,
    Candidates,
    Out,
    K: tl.constexpr,
    S: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ptr = tl.load(Ptr).to(tl.pointer_type(tl.float32))
    token = tl.load(Candidates + (i // K) * S + i % K + 1, i < R, other=0)
    value = tl.load(ptr + i * V + token, i < R, other=0)
    tl.store(Out + i, value, i < R)


@triton.jit
def _accept_pointer(P, Ptr, Candidates, Coins, Count, S: tl.constexpr, V: tl.constexpr):
    b = tl.program_id(0)
    qptr = tl.load(Ptr).to(tl.pointer_type(tl.float32))
    n, active = 0, 1
    for j in range(S - 1):
        token = tl.load(Candidates + b * S + j + 1)
        p = tl.load(P + b * S + j).to(tl.float32)
        q = tl.load(qptr + (b * (S - 1) + j) * V + token)
        u = tl.load(Coins + b * S + j)
        active = active & (u * q < p)
        n += active
    tl.store(Count + b, n)


@triton.jit
def _q_recovery(
    Ptr, Counts, Out, K: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr
):
    b, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    row = tl.minimum(tl.load(Counts + b), K - 1)
    ptr = tl.load(Ptr).to(tl.pointer_type(tl.float32))
    q = tl.load(ptr + (b * K + row) * V + col, col < V, other=0)
    tl.store(Out + b * V + col, q, col < V)


def runtime_supported():
    global _runtime_ok
    if _runtime_ok is None:
        _runtime_ok = (
            torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (10, 0)
            and dist.get_world_size() == 4
            and dist.get_backend() == "nccl"
        )
        if _runtime_ok:
            library = Path(torch.__file__).parent / "lib/libtorch_cuda.so"
            with library.open("rb") as source:
                _runtime_ok = (
                    hashlib.file_digest(source, "sha256").hexdigest()
                    == "3db19ad428ce41d67d5267cae04872e101a4cf72449eee27cfc4c424f6eb50b7"
                )
        if not _runtime_ok:
            logger.warning("COMPACT_SPEC_VERIFY fallback=unqualified_runtime")
    return _runtime_ok


class ServingVerifier(ConditionalPeerVerify):
    def __init__(self, local, q, candidates, coins, final_coins, indices):
        super().__init__(
            local,
            q,
            candidates.clone(),
            coins.clone(),
            final_coins.clone(),
            capacity=min(8, local.shape[0] * local.shape[1]),
        )
        # The shared buffer is also the stable graph input. No second local-logit
        # buffer or full draft-q staging buffer is required.
        self.local = self.shared
        self.qptr = torch.empty(1, dtype=torch.int64, device=local.device)
        self.dummy_q = torch.zeros((1, 1, 1), device=local.device)
        self.bind(local, q, candidates, coins, final_coins, indices)
        self.graph = None

    def bind(self, local, q, candidates, coins, final_coins, indices):
        self.local.copy_(local)
        self.candidates.copy_(candidates)
        self.coins.copy_(coins)
        self.final_coins.copy_(final_coins)
        self.idx.copy_(indices)
        # Keep the tensor alive and record its use on the graph replay stream;
        # only its pointer, not B*K*V probability data, is staged.
        self.q = q
        pointer = torch.tensor([q.data_ptr()], dtype=torch.int64, pin_memory=True)
        self.qptr.copy_(pointer, non_blocking=True)
        q.record_stream(torch.cuda.current_stream())

    def uncertainty_flags(self, p):
        k = self.s - 1
        q = torch.empty((self.b, k), device=p.device, dtype=torch.float32)
        _q_selected[(triton.cdiv(self.b * k, 256),)](
            self.qptr, self.candidates, q, k, self.s, self.v, self.b * k, 256
        )
        product = (self.coins[:, :k] * q).double()
        lo = self.lower.view(self.b, self.s)[:, :k]
        hi = self.upper.view(self.b, self.s)[:, :k]
        flags = ~((product < lo) | (product >= hi))
        return torch.cat(
            (flags, torch.zeros((self.b, 1), device=p.device, dtype=torch.bool)), 1
        ).flatten()

    def finish(self, p):
        count = torch.empty(self.b, device=p.device, dtype=torch.int32)
        _accept_pointer[(self.b,)](
            p, self.qptr, self.candidates, self.coins, count, self.s, self.v
        )
        dist.broadcast(count, 0)
        recovery_p = torch.softmax(
            self.gather(self.local[self.batch_idx, count.long()]).float(), -1
        )
        recovery_q = torch.empty_like(recovery_p)
        _q_recovery[(self.b, triton.cdiv(self.v, 2048))](
            self.qptr, count, recovery_q, self.s - 1, self.v, 2048
        )
        recovery_q = torch.where(torch.isnan(recovery_q), 0, recovery_q)
        difference = recovery_p - recovery_q
        residual = torch.where(
            (count == self.s - 1)[:, None],
            recovery_p,
            torch.where(difference > 0, difference, 0),
        )
        final_idx = torch.arange(self.b, device=p.device, dtype=torch.int32)[:, None]
        final, _, _ = self.sample(
            residual[:, None],
            self.dummy_q,
            self.candidates[:, :1].contiguous(),
            final_idx,
            self.coins[:, :1].contiguous(),
        )
        positions = torch.arange(self.s, device=p.device)[None, :]
        shifted = torch.cat((self.candidates[:, 1:], self.candidates[:, :1]), 1).int()
        local_predict = torch.where(positions < count[:, None], shifted, 0)
        local_predict.scatter_(1, count.long()[:, None], final)
        predict = torch.zeros_like(local_predict).flatten()
        predict.scatter_(0, self.idx.flatten().long(), local_predict.flatten())
        accepted = torch.where(positions <= count[:, None], self.idx, -1)
        dist.broadcast(predict, 0)
        dist.broadcast(accepted, 0)
        return predict, accepted, count, p

    def run(self):
        if self.graph is None:
            self.graph, self.output = self.capture()
        self.graph.replay()
        return self.output

    def close(self):
        if self.graph is not None:
            torch.cuda.synchronize()
            self.graph.reset()
            self.graph = None


def close_all():
    for runner in _cache.values():
        runner.close()
    _cache.clear()


atexit.register(close_all)


def sample(local, q, candidates, coins, final_coins, indices):
    if not runtime_supported():
        return None
    b, s = candidates.shape
    if not 1024 <= b * s <= 4096 or (b * s) % 16:
        return None
    if local.dtype != torch.bfloat16 or local.shape != (b * s, 38720):
        return None
    if (
        q is None
        or q.dtype != torch.float32
        or q.shape != (b, s - 1, 154880)
        or not q.is_contiguous()
    ):
        return None
    key = (b, s)
    local = local.view(b, s, 38720)
    if key not in _cache:
        if len(_cache) >= 2:
            _, old = _cache.popitem(last=False)
            old.close()
        _cache[key] = ServingVerifier(local, q, candidates, coins, final_coins, indices)
        logger.info(
            "COMPACT_SPEC_VERIFY route=conditional_peer rows=%d full_target_gather=False",
            b * s,
        )
    else:
        _cache[key].bind(local, q, candidates, coins, final_coins, indices)
        _cache.move_to_end(key)
    output = _cache[key].run()
    if envs.SGLANG_COMPACT_SPEC_VERIFY_SHADOW.get():
        runner = _cache[key]
        reference = runner.baseline()
        for actual, expected in zip(output[:3], reference[:3]):
            torch.testing.assert_close(
                actual.flatten(), expected.flatten(), rtol=0, atol=0
            )
        logger.info(
            "COMPACT_SPEC_VERIFY_SHADOW exact=True rows=%d repaired_rows=%d range_fallback_rows=%d",
            b * s,
            int(output[4]),
            int(runner.domain_invalid_rows.sum()),
        )
    # The graph follows qptr; keep no full draft distribution in an idle cache.
    # record_stream() in bind protects the input until replay has consumed it.
    _cache[key].q = None
    return output[0], output[2] + 1, output[1]
