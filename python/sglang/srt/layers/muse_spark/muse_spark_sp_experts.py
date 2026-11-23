"""SP experts: the W4A16 kernel pushes each route's output row to its token's owner (Hopper; None: NCCL)."""

import ctypes
import functools
import logging
from typing import Optional

import torch

from sglang.kernels.jit import muse_spark_cuda as cu
from sglang.kernels.ops.moe.muse_spark_nvfp4_w4a16_moe import (
    _compile,
    nvfp4_w4a16_moe_sp_finalize,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 16384
TOP_K = 8


def max_tokens() -> int:
    """Tokens of the largest SP forward."""
    from sglang.srt.runtime_context import get_schedule

    return max(MAX_TOKENS, get_schedule().chunked_prefill_size or 0)


def _symmetric(group, shape, dtype, device):
    """A symmetric memory buffer, its handle and every rank's view of it."""
    import torch.distributed._symmetric_memory as symm_mem

    buf = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group.device_group.group_name)
    return buf, hdl, [hdl.get_buffer(p, shape, dtype) for p in range(group.world_size)]


_state = {}


class SPExpertBuffers:
    """One TP group's symmetric memory buffers, shared by all layers."""

    def __init__(self, group, device: torch.device, hidden: int, dtype=torch.bfloat16):
        self.tp, self.rank = group.world_size, group.rank_in_group
        self.rows = -(-max_tokens() // self.tp)
        self.shard, _, shards = _symmetric(group, (self.rows, hidden), dtype, device)
        self.recv, self.recv_hdl, self.peers = _symmetric(
            group, (TOP_K, self.rows, hidden), dtype, device
        )
        self.shard_ptrs = [t.data_ptr() for t in shards]
        self.recv_ptrs = [t.data_ptr() for t in self.peers]
        self._recv_ptr_table = torch.tensor(self.recv_ptrs, device=device)
        self.routes, _, routes = _symmetric(
            group, (2, self.tp * self.rows + 1, TOP_K), torch.int32, device
        )
        self.route_ns, self._attn_done = (
            self.routes[0, -1].zero_(),
            torch.zeros(1, dtype=torch.int64, device=device),
        )
        self._route_ptr_table = torch.tensor(
            [t.data_ptr() for t in routes], device=device
        )
        try:
            _push_function(device.index)
            self._push_ok = True
        except Exception as e:
            logger.warning("SP push all-to-all unavailable (%s)", e)
            self._push_ok = False
        ok = torch.tensor([int(self._push_ok)], device=device)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=group.device_group
        )
        self._push_ok = bool(ok.item())

    def _fits(self, x: torch.Tensor, n: int, cap: int) -> bool:
        return (
            self._push_ok
            and x.dtype == self.recv.dtype
            and n <= cap
            and x.shape[1] % 8 == 0
        )

    def _push(self, name, grid, *args):
        _launch(name, grid, *args)
        self.recv_hdl.barrier(channel=0)

    def _push_ag(self, x, n, base=0, src_stride=0):
        self._push(
            "sp_push_ag",
            (_blocks(x, self.tp), self.tp),
            x,
            self._recv_ptr_table,
            *(ctypes.c_longlong(v // 8) for v in (n, base + self.rank * n, src_stride)),
            self.rank,
        )

    def push_all_to_all(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        rows, k = x.shape[0] // self.tp, x.shape[1]
        kf = k * self.tp
        if rows > self.rows or not self._fits(x, rows * kf, self.recv.numel()):
            return None
        self._push(
            "sp_push_a2a",
            (_blocks(x, self.tp), self.tp),
            x,
            self._recv_ptr_table,
            rows,
            k // 8,
            kf // 8,
            self.rank,
        )
        return self.recv.view(-1)[: rows * kf].view(rows, kf)

    def o_proj_push_plan(self, num_tokens: int, k: int) -> Optional[tuple]:
        """Where pfa writes o_proj input so recv holds push_all_to_all's result."""
        rows, kf = -(-num_tokens // self.tp), k * self.tp
        if not self._push_ok or rows * kf > self.recv.numel() or k % 8:
            return None
        return self._recv_ptr_table, rows, kf, self.rank * k, 0, rows * self.tp

    def pushed_input(self, plan: tuple) -> torch.Tensor:
        rows, kf = plan[1:3]
        self.recv_hdl.barrier(channel=0)
        _launch("sp_mark", 1, self._attn_done)
        return self.recv.view(-1)[: rows * kf].view(rows, kf)

    def push_rows(self, plan: tuple, base: int, n: int) -> None:
        """Zero tokens base .. base + n - 1 per ``plan`` (no attention rows here)."""
        rows, kf, col = plan[1:4]
        while n > 0:
            o, r = divmod(base, rows)
            m = min(n, rows - r)
            self.peers[o].view(-1)[: rows * kf].view(rows, kf)[
                r : r + m, col : col + kf // self.tp
            ].zero_()
            base, n = base + m, n - m

    def push_all_gather(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """View of the recv upper half (lower half: route rows), or None."""
        rows, k = x.shape
        half = self.recv.numel() // 2
        n = self.tp * rows * k
        if TOP_K * rows * self.recv.shape[2] > half or not self._fits(x, n, half):
            return None
        self._push_ag(x, rows * k, half)
        return self.recv.view(-1)[half : half + n].view(self.tp * rows, k)

    def gathered_experts(
        self, run_experts, shard: torch.Tensor, num_tokens: int
    ) -> Optional[torch.Tensor]:
        """This rank's rows of ``run_experts`` summed over ranks in FP32, fixed order (one rounding), or None."""
        rows, k = shard.shape
        n = self.tp * rows * k
        if n * shard.element_size() <= 16 << 20 or not self._fits(
            shard, n, self.recv.numel()
        ):
            return None
        flat = self.recv.view(-1)
        self._push_ag(shard, rows * k)
        partial = run_experts(
            flat[:n].view(self.tp * rows, k)[:num_tokens]
        ).contiguous()
        if partial.shape[0] != self.tp * rows:
            partial = torch.nn.functional.pad(
                partial, (0, 0, 0, self.tp * rows - partial.shape[0])
            )
        self.recv_hdl.barrier(channel=0)
        self._push_ag(partial, rows * k, src_stride=rows * k)
        out = torch.empty((rows, k), dtype=shard.dtype, device=shard.device)
        _launch(
            "sp_slot_sum",
            _blocks(out, 1),
            flat[:n],
            out,
            ctypes.c_longlong(out.numel() // 8),
            self.tp,
        )
        return out

    def push_embed(self, ids, weight, lo, hi):
        """This rank's SP rows of a vocab-parallel embedding (``weight``: ids [lo, hi)), or None."""
        rows, h = -(-ids.shape[0] // self.tp), weight.shape[1]
        if not self._fits(weight, rows * h, self.recv.numel() // 2):
            return None
        self._push(
            "sp_push_embed",
            _blocks(weight, 1),
            ids,
            weight,
            self._recv_ptr_table,
            ids.shape[0],
            rows,
            h // 8,
            ctypes.c_longlong(lo),
            ctypes.c_longlong(hi),
            self.rank,
        )
        return self.recv.view(-1)[: rows * h].view(rows, h)

    def gather_routing(
        self, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> Optional[tuple]:
        """All ranks' routes (= NCCL all-gather) or None."""
        rows, n = topk_ids.shape[0], topk_ids.numel() // 4
        if (
            not self._push_ok
            or topk_ids.shape[1] != TOP_K
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        ):
            return None
        ints = (
            n,
            self.rank * n,
            self.routes[0].numel() // 4,
            self.tp * self.rows * TOP_K + self.rank,
        )
        self._push(
            "sp_push_route",
            (-(-n // 512), self.tp),
            topk_ids,
            topk_weights,
            self._route_ptr_table,
            self._attn_done,
            *ints,
        )
        return self.routes[1, : self.tp * rows].view(torch.float32), self.routes[
            0, : self.tp * rows
        ]

    def sum_received(self, rows: int, topk_weights: torch.Tensor) -> torch.Tensor:
        """``topk_weights``: this rank's rows."""
        return nvfp4_w4a16_moe_sp_finalize(self.recv, topk_weights, 0, rows)


_PUSH_SRC = r"""
#include "ptx.cuh"
extern "C" __global__ void __launch_bounds__(512) sp_push_a2a(const uint4* __restrict__ x,
    const u64* __restrict__ dst, int rows, int k16, int kf16, int me) {
  const int p = (blockIdx.y + me) % gridDim.y;  // rotated by sender: the senders spread over the receivers
  uint4* d = AS(uint4, dst[p]) + me * k16;
  const uint4* s = x + (i64)p * rows * k16;
  const i64 n = (i64)rows * k16;
  for (i64 i = blockIdx.x * 512LL + threadIdx.x; i < n; i += gridDim.x * 512LL) {
    const i64 r = i / k16;
    d[r * kf16 + (i - r * k16)] = s[i];
  }
}

extern "C" __global__ void __launch_bounds__(512) sp_push_ag(const uint4* __restrict__ x,
    const u64* __restrict__ dst, i64 n16, i64 off16, i64 src16, int me) {
  const int p = (blockIdx.y + me) % gridDim.y;
  uint4* d = AS(uint4, dst[p]) + off16;
  const uint4* s = x + p * src16;
  for (i64 i = blockIdx.x * 512LL + threadIdx.x; i < n16; i += gridDim.x * 512LL) d[i] = s[i];
}

DEVI u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
extern "C" __global__ void sp_mark(u64* t) { if (threadIdx.x == 0) *t = gtime(); }

extern "C" __global__ void __launch_bounds__(512) sp_push_route(const uint4* __restrict__ ids, const uint4* __restrict__ w,
    const u64* __restrict__ dst, const u64* __restrict__ t0, int n16, int off16, int half16, int ns_at) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    const int d = (int)min(gtime() - *t0, 2000000000ULL);
    for (int p = 0; p < gridDim.y; ++p) AS(int, dst[p])[ns_at] = d;
  }
  uint4* d = AS(uint4, dst[blockIdx.y]) + off16;
  for (int i = blockIdx.x * 512 + threadIdx.x; i < n16; i += gridDim.x * 512) { d[i] = ids[i]; d[half16 + i] = w[i]; }
}

extern "C" __global__ void __launch_bounds__(512) sp_push_embed(const i64* __restrict__ ids, const uint4* __restrict__ w,
    const u64* __restrict__ dst, int n, int rows, int h16, i64 lo, i64 hi, int me) {
  for (int t = (blockIdx.x * 512 + threadIdx.x) >> 5; t < n; t += gridDim.x * 16) {
    const i64 id = ids[t];
    const bool mine = id >= lo && id < hi;
    if (!mine && (id >= 0 || t / rows != me)) continue;
    uint4* d = AS(uint4, dst[t / rows]) + (i64)(t % rows) * h16;
    const uint4* s = w + (id - lo) * h16;
    for (int i = threadIdx.x & 31; i < h16; i += 32) d[i] = mine ? s[i] : uint4{};
  }
}

extern "C" __global__ void __launch_bounds__(512) sp_slot_sum(const uint4* __restrict__ slots, uint4* __restrict__ out, i64 n8, int tp) {
  for (i64 i = blockIdx.x * 512LL + threadIdx.x; i < n8; i += gridDim.x * 512LL) {
    float a[8] = {};
    for (int s = 0; s < tp; ++s) {
      const uint4 v = slots[s * n8 + i];
      const u32 w[4] = {v.x, v.y, v.z, v.w};
      for (int h = 0; h < 4; ++h) { a[2 * h] += bf_lo(w[h]); a[2 * h + 1] += bf_hi(w[h]); }
    }
    u32 o[4];
    for (int h = 0; h < 4; ++h) o[h] = bf2(a[2 * h], a[2 * h + 1]);
    out[i] = make_uint4(o[0], o[1], o[2], o[3]);
  }
} """


@functools.cache
def _push_function(device_index: int):
    with torch.cuda.device(device_index):
        module = _compile("sp_push_a2a.cu", _PUSH_SRC)
        return {
            n: cu.function(module, n)
            for n in (
                "sp_push_a2a",
                "sp_push_ag",
                "sp_push_route",
                "sp_slot_sum",
                "sp_mark",
                "sp_push_embed",
            )
        }


def _launch(name, grid, x, *args):
    vals = [
        ctypes.c_void_p(a.data_ptr())
        if isinstance(a, torch.Tensor)
        else ctypes.c_int(a)
        if isinstance(a, int)
        else a
        for a in (x, *args)
    ]
    cu.launch(
        _push_function(x.device.index)[name],
        grid,
        512,
        0,
        torch.cuda.current_stream(x.device).cuda_stream,
        vals,
    )


def _blocks(x, tp):
    return -(
        -2 * torch.cuda.get_device_properties(x.device).multi_processor_count // tp
    )


class SPAllGatherMatmul:
    """qkv's SP all-gather fused with its matmul."""

    def __init__(self, group, device: torch.device, k: int):
        self.tp, self.me, self.rows = (
            group.world_size,
            group.rank_in_group,
            -(-max_tokens() // group.world_size),
        )
        self.buf, self.hdl, self.peer = _symmetric(
            group, (2, self.rows, k), torch.bfloat16, device
        )
        self.recv = torch.empty(
            (self.rows * self.tp, k), dtype=torch.bfloat16, device=device
        )
        self.stream = torch.cuda.Stream(device)

    def stage(self, x: torch.Tensor, parity: int) -> Optional[torch.Tensor]:
        """Where x's producer can write it so that the call needs no staging copy, or None."""
        ok = (
            x.shape[0] <= self.rows
            and x.shape[1:] == self.buf.shape[2:]
            and x.dtype == self.buf.dtype
        )
        return self.buf[parity, : x.shape[0]] if ok else None

    def __call__(self, linear, x: torch.Tensor) -> Optional[torch.Tensor]:
        rows, w, parity, me = (
            x.shape[0],
            linear.weight.t(),
            linear._sp_ag_parity,
            self.me,
        )
        if self.stage(x, parity) is None or w.dtype != x.dtype:
            return None
        if x.data_ptr() != self.buf[parity].data_ptr():
            self.buf[parity, :rows].copy_(x)
        self.hdl.barrier(channel=0)
        main, out, arrivals = (
            torch.cuda.current_stream(),
            x.new_empty((rows * self.tp, w.shape[1])),
            [],
        )
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            for p in [(me + s) % self.tp for s in range(1, self.tp)]:
                self.recv[p * rows : (p + 1) * rows].copy_(self.peer[p][parity, :rows])
                arrivals.append((p, self.stream.record_event()))
        torch.mm(x, w, out=out[me * rows : (me + 1) * rows])
        for p, ev in arrivals:
            main.wait_event(ev)
            torch.mm(
                self.recv[p * rows : (p + 1) * rows],
                w,
                out=out[p * rows : (p + 1) * rows],
            )
        return out


def prepare(group, device: torch.device, hidden: int) -> Optional[SPExpertBuffers]:
    key = (group.device_group.group_name, device.index)
    if key not in _state:
        try:
            _state[key] = SPExpertBuffers(group, device, hidden)
        except Exception as e:
            logger.warning(
                "SP experts without NVLink buffers (%s): all-gather + reduce-scatter", e
            )
            _state[key] = None
    return _state[key]
