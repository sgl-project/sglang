"""DFlash tree verify: all-greedy steps verify an 8-node token tree instead of a chain.

Grown best-first over the selector lattice (CandidateSelector.build_lattice) by summed log-softmax:
node 0 is the bonus token, each pop adds its top 4 children. Node j verifies at prefix + depth[j]
seeing anc[j] (bit i: node i is j or an ancestor). Accept keeps the longest root path the target's
argmax agrees with (lowest node on a tie), copying its K / V rows into the committed slots rather
than rewriting req_to_token (under overlap a finished request's slot may hold a new one). Needs
selector top_k in [4, 32], block size 8, page size 1, and a target whose supports_tree_verify() is true.
"""

import ctypes
import functools
import logging

import torch

from sglang.kernels.jit import muse_spark_cuda as cu

TREE_NODES = 8

_SRC = r"""
#include "ptx.cuh"
#define N 8
#define KIDS 4

DEVI int argmax_lane(float v) {  // the lowest lane holding the max: one redux on an order-preserving key (no NaN here; + 0.f folds -0 into +0)
  u32 k = __float_as_uint(v + 0.f);
  k = (k & 0x80000000u) ? ~k : (k | 0x80000000u);
  return __ffs(ballot(k == __reduce_max_sync(~0u, k))) - 1;
}
DEVI void warp_argmax(float& v, int& i) {
  i = argmax_lane(v);
  v = shfl(v, i);
}
DEVI float lane_log_softmax(float x, bool valid) {  // non-finite drop out: a NaN would split the warp
  valid = valid && isfinite(x);
  x = valid ? x : NEG_INF;
  const float m = shfl(x, argmax_lane(x));  // the row max (a zero max's sign may differ; no pick sees it)
  if (m == NEG_INF) return NEG_INF;
  float e = valid ? expf(x - m) : 0.f;
  for (int o = 16; o; o >>= 1) e += shfl_xor(e, o);
  return valid ? x - m - logf(e) : NEG_INF;
}

extern "C" __global__ void dflash_tree_build(const i64* cand, const float* scores, const i64* anchor, int B, int S, int K,
                                             i64* tok, int* parent, u8* depth, u8* anc) {
  const int lane = threadIdx.x & 31, b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (b >= B) return;
  const i64* cd = cand + (i64)b * S * K;
  const float* sc = scores + (i64)b * S * K * K;
  float f_lp = NEG_INF;
  int f_slot = 0, f_c = 0, f_par = 0;
  float lp = lane_log_softmax(lane < K ? sc[lane] : NEG_INF, lane < K);
  for (int k = 0; k < KIDS; ++k) {
    float v = lp; int i = lane;
    warp_argmax(v, i);
    if (lane == k) { f_lp = v; f_c = i; }
    if (lane == i) lp = NEG_INF;
  }
  i64 my_tok = anchor[b];
  int my_par = -1, my_depth = 0, n = 1;
  for (int it = 1; it < N; ++it) {
    float v = f_lp; int i = lane;
    warp_argmax(v, i);
    if (v == NEG_INF) break;
    const int ps = shfl(f_slot, i), pc = shfl(f_c, i), pp = shfl(f_par, i);
    if (lane == i) f_lp = NEG_INF;
    const int node = n++;
    if (lane == node) { my_tok = cd[ps * K + pc]; my_par = pp; my_depth = ps + 1; }
    if (ps + 1 < S) {
      const float* row = sc + ((i64)(ps + 1) * K + pc) * K;
      float l2 = lane_log_softmax(lane < K ? row[lane] : NEG_INF, lane < K);
      for (int k = 0; k < KIDS; ++k) {
        float v2 = l2; int i2 = lane;
        warp_argmax(v2, i2);
        if (lane == i2) l2 = NEG_INF;
        const u32 free_lanes = ballot(f_lp == NEG_INF);
        if (free_lanes && lane == __ffs(free_lanes) - 1) { f_lp = v + v2; f_slot = ps + 1; f_c = i2; f_par = node; }
      }
    }
  }
  if (lane >= n) { my_tok = cd[0]; my_par = 0; my_depth = 1; }  // frontier ran dry
  u32 bits = 1u << lane;
  for (int j = 1; j < N; ++j) {
    const int pj = shfl(my_par, j);
    const u32 bp = shfl(bits, pj < 0 ? 0 : pj);
    if (lane == j) bits |= bp;
  }
  const int o = b * N + lane;
  if (lane < N) { tok[o] = my_tok; parent[o] = my_par; depth[o] = my_depth; anc[o] = bits; }
}

// Grid (B, layers): K / V sources are all read before any write (a slot can be both).
extern "C" __global__ void __launch_bounds__(128, 16)
    dflash_tree_accept(const i64* tok, const int* parent, const i64* pred, const i64* loc, const i64* swa_map,
                       const u64* k_ptrs, const u64* v_ptrs, const u8* is_swa, int row_bytes, i64* out, int* commit,
                       i64* bonus, i64* perm) {
  __shared__ int4 rows[2][N][32];
  __shared__ u32 walk[2];
  const int b = blockIdx.x, l = blockIdx.y, chunks = row_bytes / 16;
  if (threadIdx.x < 32) {
    const int j = threadIdx.x;
    const bool own = j < N;
    const i64 t = own ? tok[b * N + j] : 0, pr = own ? pred[b * N + j] : 0;
    const int p = own ? parent[b * N + j] : -1;
    u32 used = 1u, path = 0u;
    int node = 0, acc = 0;
    i64 want = shfl(pr, 0);
    for (int d = 1; d < N; ++d) {
      const u32 kids = ballot(j > 0 && p == node && t == want) & ~used;
      if (!kids) break;
      node = __ffs(kids) - 1; path |= (u32)node << (4 * d); used |= 1u << node; acc = d;
      want = shfl(pr, node);
    }
    const i64 next_tok = shfl(t, (path >> (4 * ((j + 1) % N))) & 15u);
    for (int k = 1, m = acc + 1; k < N; ++k)
      if (!((used >> k) & 1u)) path |= (u32)k << (4 * m++);
    if (l == 0 && own) {
      out[b * N + j] = j < acc ? next_tok : (j == acc ? want : 0);
      perm[b * N + j] = b * N + ((path >> (4 * j)) & 15u);
      if (j == 0) { commit[b] = acc + 1; bonus[b] = want; }
    }
    if (j == 0) { walk[0] = path; walk[1] = acc; }
  }
  __syncthreads();
  const u32 path = walk[0];
  const int acc = walk[1];
  const i64* lo = loc + b * N;
  char* base[2] = {(char*)k_ptrs[l], (char*)v_ptrs[l]};
  for (int pass = 0; pass < 2; ++pass) {
    if (pass) __syncthreads();
    for (int i = threadIdx.x; i < 2 * N * chunks; i += blockDim.x) {
      const int kv = i / (N * chunks), d = i / chunks % N, c = i % chunks, pd = (path >> (4 * d)) & 15u;
      if (d < 1 || d > acc || pd == d) continue;
      i64 s = lo[pass ? d : pd];
      if (is_swa[l]) s = swa_map[s];
      int4* row = (int4*)(base[kv] + s * row_bytes) + c;
      if (pass) *row = rows[kv][d][c]; else rows[kv][d][c] = *row;
    }
  }
}
"""


@functools.cache
def _kernels():
    mod = cu.build_module(
        "dflash_tree.cu",
        _SRC,
        [
            b"--gpu-architecture=sm_%d%d" % torch.cuda.get_device_capability(),
            b"-std=c++17",
        ],
    )
    return {n: cu.function(mod, n) for n in ("dflash_tree_build", "dflash_tree_accept")}


def _launch(name, grid, block, *args):
    if grid:
        args = [
            ctypes.c_void_p(a.data_ptr() if a is not None else 0)
            if not isinstance(a, int)
            else ctypes.c_int(a)
            for a in args
        ]
        cu.launch(
            _kernels()[name],
            grid,
            block,
            0,
            torch.cuda.current_stream().cuda_stream,
            args,
        )


class TreeBuffers:
    """Static [max_bs, TREE_NODES] tree arrays."""

    def __init__(self, max_bs: int, device):
        rows = torch.arange(TREE_NODES, device=device)
        self.tok, self.perm = (
            torch.zeros((max_bs, TREE_NODES), dtype=torch.int64, device=device)
            for _ in range(2)
        )
        self.chain = (
            (rows - 1).int(),
            rows.to(torch.uint8),
            ((2 << rows) - 1).to(torch.uint8),
        )
        self.parent, self.depth, self.anc = (c.repeat(max_bs, 1) for c in self.chain)
        self.max_bs = None

    def set_chains(self, bs: int) -> None:
        for buf, c in zip((self.parent, self.depth, self.anc), self.chain):
            buf[:bs].copy_(c)


_BUFFERS = {}


def tree_buffers(device, max_bs=None):
    """The device's TreeBuffers, created by the first call that gives ``max_bs``."""
    device = torch.device(device)
    if device.type != "cuda":  # NVRTC kernels: other devices verify chains
        return None
    index = device.index
    index = torch.cuda.current_device() if index is None else index
    if index not in _BUFFERS and max_bs is not None:
        _BUFFERS[index] = TreeBuffers(max_bs, torch.device("cuda", index))
        _kernels()
    return _BUFFERS.get(index)


def tree_max_batch(model_runner) -> int:
    """Largest verify batch (also: verify blocks per extend forward) taking a tree (0: none); needs the backends."""
    tree = tree_buffers(model_runner.device)
    if tree is None:
        return 0
    if tree.max_bs is None:
        backends = [
            b
            for b in (
                model_runner.attn_backend,
                getattr(model_runner, "decode_attn_backend", None),
            )
            if b is not None
        ]
        from sglang.srt.runtime_context import get_exec, get_parallel

        m = (
            0
            if get_parallel().attn_dp_enabled
            else min(
                tree.tok.shape[0], model_runner.model.tree_verify_max_batch(backends)
            )
        )
        tree.max_bs = max(
            (b for b in get_exec().graph.cuda_graph_config.decode.bs if b <= m),
            default=0,
        )  # pads to a tree graph
        logging.getLogger(__name__).info(
            "DFLASH tree verify: %d-node trees up to batch %d", TREE_NODES, tree.max_bs
        )
    return tree.max_bs


def build_tree(*, candidate_ids, scores, anchor, out: TreeBuffers) -> None:
    """Fill ``out`` rows [:B] from the lattice; only all-greedy steps read them (other steps rewrite the chains)."""
    B, S, K = candidate_ids.shape
    assert B <= out.tok.shape[0]
    i64 = torch.int64
    _launch(
        "dflash_tree_build",
        (B + 3) // 4,
        128,
        candidate_ids.to(i64).contiguous(),
        scores.float().contiguous(),
        anchor.to(i64).contiguous(),
        B,
        S,
        K,
        out.tok,
        out.parent,
        out.depth,
        out.anc,
    )


def accept_tree(
    *, tree: TreeBuffers, target_predict, verify_cache_loc, kv_move: "TreeKVMove"
):
    """Greedy accept over ``tree`` rows [:B] and the accepted paths' K / V copy, one launch; ``tree.perm``: committed node order as hidden-row ids."""
    B, dev, m = verify_cache_loc.shape[0], target_predict.device, kv_move
    out = torch.empty((B, TREE_NODES), dtype=torch.int64, device=dev)
    commit, bonus = (
        torch.empty((B,), dtype=torch.int32, device=dev),
        torch.empty((B,), dtype=torch.int64, device=dev),
    )
    _launch(
        "dflash_tree_accept",
        (B, m.k_ptrs.numel()),
        128,
        tree.tok,
        tree.parent,
        target_predict.to(torch.int64).contiguous(),
        verify_cache_loc.to(torch.int64).contiguous(),
        m.swa_map,
        m.k_ptrs,
        m.v_ptrs,
        m.is_swa,
        m.row_bytes,
        out,
        commit,
        bonus,
        tree.perm,
    )
    return out, commit, bonus


class TreeKVMove:
    """Per-layer K / V buffer pointers of a (hybrid) KV pool."""

    def __init__(self, kv_pool, device):
        hybrid = hasattr(kv_pool, "swa_kv_pool")
        pools = (
            [(kv_pool.full_kv_pool, 0), (kv_pool.swa_kv_pool, 1)]
            if hybrid
            else [(kv_pool, 0)]
        )
        ptrs, row_bytes = [], set()
        for pool, swa in pools:
            for k, v in zip(pool.k_buffer, pool.v_buffer):
                assert k.is_contiguous() and v.is_contiguous()
                row_bytes |= {
                    k[0].numel() * k.element_size(),
                    v[0].numel() * v.element_size(),
                }
                ptrs.append((k.data_ptr(), v.data_ptr(), swa))
        (self.row_bytes,) = row_bytes
        assert self.row_bytes % 16 == 0 and self.row_bytes <= 512
        self.k_ptrs, self.v_ptrs, self.is_swa = (
            torch.tensor(c, device=device) for c in zip(*ptrs)
        )
        self.is_swa = self.is_swa.to(torch.uint8)
        self.swa_map = kv_pool.full_to_swa_index_mapping if hybrid else None
        assert self.swa_map is None or self.swa_map.dtype == torch.int64
