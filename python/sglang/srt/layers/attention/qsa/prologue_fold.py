"""One-launch QSA indexer prologue for CUDA-graph decode / target verify / draft extend (h12).

rev4 runs the indexer prologue as two launches on the side stream:
``qsa_index_q_prep_kernel`` (one 4-warp CTA per token: index-Q gemma norm + MRoPE, raw-K ring
store, RoPE-position store) and then ``qsa_index_k_compress_kernel`` (one warp per graph row:
fp32 group mean -> gemma norm -> MRoPE -> compressed-cache store), whose ring reads depend on
q_prep's ring writes. ``qsa_index_prologue_fold_kernel`` does both in one launch of
``max(tokens, groups)`` CTAs x 5 warps:

* warps 0-3 of CTA b run q_prep's per-token body for token b unchanged;
* warp 4 of CTA b compresses group b. Each group member is resolved against this step's
  ring writes: a member slot written by an in-step token t (``cache_loc[t]``) reads token t's
  raw K straight from the projection output and its RoPE coordinates from ``positions``,
  exactly what q_prep stores there; any other member reads the ring / RoPE-position buffer,
  which this launch does not write. That is the value rev4's k_compress reads after q_prep.

Everything that does not depend on the projection output (positions, slots, group slots,
older ring rows, their RoPE coordinates, cos/sin rows) is issued before ``griddepcontrol.wait``.
Per-row math is the header's: ``qsa_gemma_norm_row`` / ``qsa_mrope_apply`` from
``qsa_indexer.cuh`` and k_compress's mean / RMS partial layout copied verbatim, compiled with
the same flags, so q, ring rows, RoPE positions and compressed rows are bit-identical to rev4.

``SGLANG_QSA_QPF=0`` restores the two-launch path. ``SGLANG_QSA_QPF_SHADOW=1`` also runs rev4's
two kernels on snapshots in the same graph and counts mismatches on the device
(q, ring rows, RoPE positions, compressed rows, top-512 rows); ``qpf_shadow_log`` prints them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

logger = logging.getLogger(__name__)

QPF_ENABLED = os.environ.get("SGLANG_QSA_QPF", "0") != "0"
QPF_SHADOW = os.environ.get("SGLANG_QSA_QPF_SHADOW", "0") == "1"
_SHADOW_FIELDS = (
    "checks",
    "rows",
    "q_mismatch",
    "ring_mismatch",
    "rope_mismatch",
    "ck_rows",
    "ck_mismatch",
    "topk_mismatch",
    "topk_rows",
    "logit_mismatch",
    "topk_value_mismatch",
    "topk_aa_set_mismatch",
)

_SOURCE = r"""
#include "attention/qsa_indexer.cuh"

namespace sglang {

struct QsaIndexPrologueFoldParams {
  // q_prep (QsaIndexQPrepParams)
  const void* qk;                 // [tokens, (num_q_heads + 1) * kHeadDim]
  void* q_out;                    // [tokens, q_heads_padded, kHeadDim]
  const void* q_weight;           // [kHeadDim]
  const float* cos_sin_cache;     // [positions_capacity, rotary_dim]
  const int32_t* axis_map;        // [rotary_dim / 2]
  const int64_t* positions;       // [num_axes, tokens] (row stride may exceed tokens)
  const int64_t* cache_loc;       // [tokens]
  void* key_state_buffer;         // [slots, kHeadDim]
  int64_t* rope_position_buffer;  // [slots, 3]
  int64_t positions_stride;
  int32_t num_axes;
  int32_t num_q_heads;
  int32_t q_heads_padded;
  int32_t rotary_dim;
  float q_eps;
  // k_compress (QsaIndexKCompressParams)
  const int32_t* group_locs;      // [groups, compress_ratio]
  const void* k_weight;           // [kHeadDim]
  const int32_t* write_locs;      // [groups]
  void* compressed_k_buffer;      // [compressed_slots, kHeadDim]
  int32_t compress_ratio;
  int32_t num_groups;
  int32_t num_tokens;
  float k_eps;
};

/// Pull a cos/sin cache row towards the SM before griddepcontrol.wait; the values are read
/// later by qsa_mrope_apply unchanged.
SGL_DEVICE void qpf_prefetch_row(const float* row, int32_t rotary_dim, uint32_t lane) {
  const int32_t off = static_cast<int32_t>(lane) * 32;
  if (off < rotary_dim) {
    asm volatile("prefetch.global.L1 [%0];" ::"l"(row + off));
  }
}

template <typename T, typename TOut, int kHeadDim, bool kIsNeox, bool kUsePDL>
__global__ __launch_bounds__(160) void qsa_index_prologue_fold_kernel(
    const QsaIndexPrologueFoldParams __grid_constant__ params) {
  using namespace device;
  constexpr int kPerLane = kHeadDim / kWarpThreads;
  using vec_t = AlignedVector<T, kPerLane>;
  using out_vec_t = AlignedVector<TOut, kPerLane>;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  __shared__ T smem_rows[5][kHeadDim];

  if (warp < 4) {
    // ---------------- q_prep: token b (qsa_index_q_prep_kernel body) ----------------
    const uint32_t token = blockIdx.x;
    const bool active = token < static_cast<uint32_t>(params.num_tokens);
    int64_t loc = 0;
    int64_t pos[3] = {0, 0, 0};
    if (active) {
      loc = params.cache_loc[token];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        const int64_t ax = a < params.num_axes ? a : 0;
        pos[a] = params.positions[ax * params.positions_stride + token];
      }
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        if (a < params.num_axes) {
          qpf_prefetch_row(params.cos_sin_cache + pos[a] * params.rotary_dim, params.rotary_dim, lane);
        }
      }
    }

    device::PDLWaitPrimary<kUsePDL>();

    if (active) {
      const int64_t qk_row = static_cast<int64_t>(token) * (params.num_q_heads + 1) * kHeadDim;
      for (int32_t h = static_cast<int32_t>(warp); h < params.q_heads_padded; h += 4) {
        TOut* out_row =
            static_cast<TOut*>(params.q_out) + (static_cast<int64_t>(token) * params.q_heads_padded + h) * kHeadDim;
        if (h < params.num_q_heads) {
          const T* x_row = static_cast<const T*>(params.qk) + qk_row + h * kHeadDim;
          qsa_gemma_norm_row<T, kHeadDim>(x_row, static_cast<const T*>(params.q_weight), params.q_eps, smem_rows[warp]);
          qsa_mrope_apply<T, TOut, kHeadDim, kIsNeox>(
              smem_rows[warp], out_row, params.cos_sin_cache, params.axis_map, pos, params.rotary_dim);
        } else {
          out_vec_t zv;
          zv.fill(DTypeTrait<TOut>::from(0.0f));
          zv.store(out_row, lane);  // offset is in vector units
        }
      }

      // Raw token-K and RoPE coordinates are stored for every token, whether or
      // not the token completes a compression group.
      if (warp == 0) {
        vec_t kv;
        kv.load(
            static_cast<const T*>(params.qk) + qk_row + params.num_q_heads * kHeadDim,
            lane);  // offset is in vector units
        kv.store(static_cast<T*>(params.key_state_buffer) + loc * kHeadDim, lane);
      }
      if (warp == 1 && lane < 3) {
        params.rope_position_buffer[loc * 3 + lane] = pos[lane];
      }
    }
  } else {
    // ---------------- k_compress: group b (qsa_index_k_compress_kernel body) ----------------
    const uint32_t group = blockIdx.x;
    const bool active = group < static_cast<uint32_t>(params.num_groups);
    constexpr int kMaxRatio = 16;
    const int32_t* locs = params.group_locs + static_cast<int64_t>(group) * params.compress_ratio;
    int32_t loc0 = 0;
    int32_t write_loc = 0;
    // match[r]: in-step token whose ring store lands on member r's slot (the last such token), or -1.
    int32_t match[kMaxRatio];
    vec_t ring[kMaxRatio];
    int64_t pos[3] = {0, 0, 0};
    float kw[kPerLane];  // the k gemma weight is static: load it with the other early loads
    if (active) {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        kw[i] = static_cast<float>(static_cast<const T*>(params.k_weight)[lane * kPerLane + i]);
      }
      loc0 = locs[0];
      write_loc = params.write_locs[group];
#pragma unroll
      for (int r = 0; r < kMaxRatio; ++r) {
        match[r] = -1;
      }
      // Each lane loads its token's slot and RoPE coordinates together, so the group-start
      // member's coordinates come from a shuffle instead of a dependent load after the match.
      int64_t pos0[3] = {0, 0, 0};
      for (int32_t base = 0; base < params.num_tokens; base += kWarpThreads) {
        const int32_t t = base + static_cast<int32_t>(lane);
        const bool live = t < params.num_tokens;
        const int64_t slot = live ? params.cache_loc[t] : int64_t{-1};
        int64_t tpos[3] = {0, 0, 0};
        if (live) {
#pragma unroll
          for (int a = 0; a < 3; ++a) {
            const int64_t ax = a < params.num_axes ? a : 0;
            tpos[a] = params.positions[ax * params.positions_stride + t];
          }
        }
#pragma unroll
        for (int r = 0; r < kMaxRatio; ++r) {
          if (r < params.compress_ratio) {
            const uint32_t hit = __ballot_sync(0xffffffffu, slot == static_cast<int64_t>(locs[r]));
            if (hit != 0u) {
              const int src = 31 - __clz(static_cast<int>(hit));
              match[r] = base + src;
              if (r == 0) {
#pragma unroll
                for (int a = 0; a < 3; ++a) {
                  pos0[a] = __shfl_sync(0xffffffffu, tpos[a], src);
                }
              }
            }
          }
        }
      }
      // Members no in-step token writes: the ring row and RoPE coordinates rev4 reads are
      // untouched by this launch, so they load before the wait.
#pragma unroll
      for (int r = 0; r < kMaxRatio; ++r) {
        if (r < params.compress_ratio && match[r] < 0) {
          ring[r].load(
              static_cast<const T*>(params.key_state_buffer) + static_cast<int64_t>(locs[r]) * kHeadDim,
              lane);  // offset is in vector units
        }
      }
      if (match[0] >= 0) {
#pragma unroll
        for (int a = 0; a < 3; ++a) {
          pos[a] = pos0[a];  // = positions[axis][match[0]], what q_prep stores for that slot
        }
      } else {
#pragma unroll
        for (int a = 0; a < 3; ++a) {
          pos[a] = params.rope_position_buffer[static_cast<int64_t>(loc0) * 3 + a];
        }
      }
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        if (a < params.num_axes || match[0] < 0) {
          qpf_prefetch_row(params.cos_sin_cache + pos[a] * params.rotary_dim, params.rotary_dim, lane);
        }
      }
    }

    device::PDLWaitPrimary<kUsePDL>();

    if (active) {
      // In-step members: token t's raw K is the bf16 row q_prep copies into the ring.
#pragma unroll
      for (int r = 0; r < kMaxRatio; ++r) {
        if (r < params.compress_ratio && match[r] >= 0) {
          ring[r].load(
              static_cast<const T*>(params.qk) +
                  static_cast<int64_t>(match[r]) * (params.num_q_heads + 1) * kHeadDim +
                  params.num_q_heads * kHeadDim,
              lane);  // offset is in vector units
        }
      }

      // fp32 mean over the group, rounded to the storage dtype exactly like
      // average_pool_qsa_keys (float().mean(dim=1).to(dtype)).
      T* smem_row = smem_rows[4];
      float mf[kPerLane];
      {
        float acc[kPerLane];
        // Same member order as k_compress (r = 0, 1, ...); fully unrolled so `ring` stays in
        // registers.
#pragma unroll
        for (int r = 0; r < kMaxRatio; ++r) {
          if (r < params.compress_ratio) {
            const vec_t& v = ring[r];
#pragma unroll
            for (int i = 0; i < kPerLane; ++i) {
              const float f = static_cast<float>(v[i]);
              acc[i] = (r == 0) ? f : acc[i] + f;
            }
          }
        }
        const float inv_ratio = 1.0f / static_cast<float>(params.compress_ratio);
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const T m = DTypeTrait<T>::from(acc[i] * inv_ratio);
          mf[i] = static_cast<float>(m);
          smem_row[lane * kPerLane + i] = m;
        }
        __syncwarp();
        // Sum of squares in flashinfer RMSNormKernel's exact partial layout
        // (thread t sums elements [8t, 8t+8), inactive lanes contribute 0), so
        // the result stays bit-equal to the eager k_layernorm this replaces.
        static_assert(kHeadDim % 8 == 0);
        float ss = 0.0f;
        if (lane < kHeadDim / 8) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float f = static_cast<float>(smem_row[lane * 8 + j]);
            ss += f * f;
          }
        }
        ss = warp::reduce_sum(ss);
        const float nf = math::rsqrt(ss / kHeadDim + params.k_eps);
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const float wf = kw[i];
          smem_row[lane * kPerLane + i] = DTypeTrait<T>::from(mf[i] * nf * (1.0f + wf));
        }
        __syncwarp();
      }

      TOut* out_row = static_cast<TOut*>(params.compressed_k_buffer) + static_cast<int64_t>(write_loc) * kHeadDim;
      qsa_mrope_apply<T, TOut, kHeadDim, kIsNeox>(
          smem_row, out_row, params.cos_sin_cache, params.axis_map, pos, params.rotary_dim);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, typename TOut, int kHeadDim, bool kIsNeox, bool kUsePDL>
void qsa_index_prologue_fold(
    tvm::ffi::TensorView qk,
    tvm::ffi::TensorView q_out,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView cos_sin_cache,
    tvm::ffi::TensorView axis_map,
    tvm::ffi::TensorView positions,
    int64_t num_axes,
    tvm::ffi::TensorView cache_loc,
    tvm::ffi::TensorView key_state_buffer,
    tvm::ffi::TensorView rope_position_buffer,
    int64_t num_q_heads,
    int64_t rotary_dim,
    float q_eps,
    tvm::ffi::TensorView group_locs,
    tvm::ffi::TensorView k_weight,
    tvm::ffi::TensorView write_locs,
    tvm::ffi::TensorView compressed_k_buffer,
    int64_t compress_ratio,
    float k_eps) {
  using namespace host;
  auto tokens = SymbolicSize{"tokens"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  constexpr int64_t D = kHeadDim;

  TensorMatcher({tokens, (num_q_heads + 1) * D}).with_dtype<T>().with_device(device).verify(qk);
  auto heads_padded = SymbolicSize{"heads_padded"};
  TensorMatcher({tokens, heads_padded, D}).with_dtype<TOut>().with_device(device).verify(q_out);
  TensorMatcher({D}).with_dtype<T>().with_device(device).verify(q_weight);
  auto cache_rows = SymbolicSize{"cos_sin_cache_rows"};
  TensorMatcher({cache_rows, rotary_dim}).with_dtype<fp32_t>().with_device(device).verify(cos_sin_cache);
  TensorMatcher({rotary_dim / 2}).with_dtype<int32_t>().with_device(device).verify(axis_map);
  TensorMatcher({num_axes, tokens}).with_dtype<int64_t>().with_device(device).with_strides({-1, 1}).verify(positions);
  TensorMatcher({tokens}).with_dtype<int64_t>().with_device(device).verify(cache_loc);
  auto slots = SymbolicSize{"state_slots"};
  TensorMatcher({slots, D}).with_dtype<T>().with_device(device).verify(key_state_buffer);
  TensorMatcher({slots, 3}).with_dtype<int64_t>().with_device(device).verify(rope_position_buffer);
  auto groups = SymbolicSize{"groups"};
  TensorMatcher({groups, compress_ratio}).with_dtype<int32_t>().with_device(device).verify(group_locs);
  TensorMatcher({D}).with_dtype<T>().with_device(device).verify(k_weight);
  TensorMatcher({groups}).with_dtype<int32_t>().with_device(device).verify(write_locs);
  auto compressed_slots = SymbolicSize{"compressed_slots"};
  TensorMatcher({compressed_slots, D}).with_dtype<TOut>().with_device(device).verify(compressed_k_buffer);

  const int64_t num_tokens = tokens.unwrap();
  const int64_t q_heads_padded = heads_padded.unwrap();
  const int64_t num_groups = groups.unwrap();
  CHECK_HOST(num_tokens > 0) << "qsa_index_prologue_fold: no tokens";
  CHECK_HOST(num_groups > 0) << "qsa_index_prologue_fold: no groups";
  CHECK_HOST(num_axes == 1 || num_axes == 3) << "qsa_index_prologue_fold: positions must have 1 or 3 axes";
  CHECK_HOST(q_heads_padded >= num_q_heads) << "qsa_index_prologue_fold: padded heads < num_q_heads";
  CHECK_HOST(rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= D) << "qsa_index_prologue_fold: invalid rotary_dim";
  CHECK_HOST(compress_ratio > 0 && compress_ratio <= 16) << "qsa_index_prologue_fold: invalid compress_ratio";

  const auto params = QsaIndexPrologueFoldParams{
      .qk = qk.data_ptr(),
      .q_out = q_out.data_ptr(),
      .q_weight = q_weight.data_ptr(),
      .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
      .axis_map = static_cast<const int32_t*>(axis_map.data_ptr()),
      .positions = static_cast<const int64_t*>(positions.data_ptr()),
      .cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr()),
      .key_state_buffer = key_state_buffer.data_ptr(),
      .rope_position_buffer = static_cast<int64_t*>(rope_position_buffer.data_ptr()),
      .positions_stride = positions.stride(0),
      .num_axes = static_cast<int32_t>(num_axes),
      .num_q_heads = static_cast<int32_t>(num_q_heads),
      .q_heads_padded = static_cast<int32_t>(q_heads_padded),
      .rotary_dim = static_cast<int32_t>(rotary_dim),
      .q_eps = q_eps,
      .group_locs = static_cast<const int32_t*>(group_locs.data_ptr()),
      .k_weight = k_weight.data_ptr(),
      .write_locs = static_cast<const int32_t*>(write_locs.data_ptr()),
      .compressed_k_buffer = compressed_k_buffer.data_ptr(),
      .compress_ratio = static_cast<int32_t>(compress_ratio),
      .num_groups = static_cast<int32_t>(num_groups),
      .num_tokens = static_cast<int32_t>(num_tokens),
      .k_eps = k_eps,
  };
  const int64_t ctas = num_tokens > num_groups ? num_tokens : num_groups;
  LaunchKernel(static_cast<uint32_t>(ctas), 160, device.unwrap())
      .enable_pdl(kUsePDL)(qsa_index_prologue_fold_kernel<T, TOut, kHeadDim, kIsNeox, kUsePDL>, params);
}

}  // namespace sglang
"""

_SOURCE_DIGEST = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]


def _source_path() -> str:
    configured = os.environ.get("SGLANG_QSA_QPF_SRC_DIR")
    if not configured or not pathlib.Path(configured).is_absolute():
        raise RuntimeError("QPF requires an absolute, node-local SGLANG_QSA_QPF_SRC_DIR")
    root = pathlib.Path(configured).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"qsa_prologue_fold_{_SOURCE_DIGEST}.cuh"
    if not path.exists():
        tmp = root / f".{path.name}.{os.getpid()}"
        tmp.write_text(_SOURCE)
        os.replace(tmp, path)
    logger.info("Q38_QPF_SOURCE_PATH path=%s source_digest=%s", path, _SOURCE_DIGEST)
    return str(path)


@cache_once
def _module(dtype: torch.dtype, out_dtype: torch.dtype, head_dim: int, is_neox_style: bool):
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    args = make_cpp_args(dtype, out_dtype, head_dim, is_neox_style, is_arch_support_pdl())
    return load_jit(
        "qsa_prologue_fold",
        *args,
        "s" + _SOURCE_DIGEST,
        cuda_files=[_source_path()],
        cuda_wrappers=[("fold", f"qsa_index_prologue_fold<{args}>")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )


def precompile(dtype: torch.dtype, out_dtype: torch.dtype, head_dim: int, is_neox_style: bool) -> None:
    """Build (or load) the module now, at init: a first build at serving time stalls the scheduler."""
    _module(dtype, out_dtype, head_dim, is_neox_style)


def qsa_index_prologue_fold(
    qk: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    q_weight: torch.Tensor,
    cache_loc: torch.Tensor,
    key_state_buffer: torch.Tensor,
    rope_position_buffer: torch.Tensor,
    num_q_heads: int,
    rotary_dim: int,
    q_eps: float,
    is_neox_style: bool,
    q_heads_padded: int,
    out_dtype: torch.dtype,
    group_locs: torch.Tensor,
    k_weight: torch.Tensor,
    write_locs: torch.Tensor,
    compressed_k_buffer: torch.Tensor,
    compress_ratio: int,
    k_eps: float,
) -> torch.Tensor:
    """q_prep + k_compress in one launch; arguments as ``qsa_index_q_norm_rope_store`` followed by
    ``qsa_index_k_compress_store`` on the ring this step writes. Returns the index Q."""
    head_dim = q_weight.shape[0]
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    q_out = torch.empty(
        (qk.shape[0], q_heads_padded, head_dim), dtype=out_dtype, device=qk.device
    )
    _module(qk.dtype, out_dtype, head_dim, is_neox_style).fold(
        qk,
        q_out,
        q_weight,
        cos_sin_cache,
        axis_map,
        positions,
        positions.shape[0],
        cache_loc,
        key_state_buffer,
        rope_position_buffer,
        num_q_heads,
        rotary_dim,
        q_eps,
        group_locs,
        k_weight,
        write_locs,
        compressed_k_buffer,
        compress_ratio,
        k_eps,
    )
    return q_out


# ---------------------------------------------------------------------------------------------
# Shadow (SGLANG_QSA_QPF_SHADOW=1): rev4's two kernels on snapshots, compared on the device.
# ---------------------------------------------------------------------------------------------
_shadow_counts: dict = {}
_shadow_calls = 0
_shadow_last = None


def shadow_counts(device) -> Optional[torch.Tensor]:
    """The device's persistent counters. They are created outside stream capture (the graph
    runner's warmup passes): a tensor created during capture would be zero-filled by a captured
    node on every replay. None while capturing before they exist (that capture is not shadowed)."""
    key = str(device)
    counts = _shadow_counts.get(key)
    if counts is None:
        if torch.cuda.is_current_stream_capturing():
            logger.warning("qpf shadow: counters missing during capture; this graph is not shadowed")
            return None
        counts = torch.zeros(len(_SHADOW_FIELDS), dtype=torch.int64, device=device)
        _shadow_counts[key] = counts
    return counts


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() == 2:
        return t.view(torch.int16)
    if t.element_size() == 1:
        return t.view(torch.uint8)
    return t


def shadow_snapshot(key_state_buffer: torch.Tensor, rope_position_buffer: torch.Tensor):
    """Copies of the ring and RoPE-position buffers taken before the fused launch."""
    return key_state_buffer.clone(), rope_position_buffer.clone()


def shadow_reference(
    snap,
    *,
    qk,
    positions,
    cos_sin_cache,
    axis_map,
    q_weight,
    cache_loc,
    num_q_heads,
    rotary_dim,
    q_eps,
    is_neox_style,
    q_heads_padded,
    out_dtype,
    group_locs,
    k_weight,
    write_locs,
    compress_ratio,
    k_eps,
):
    """rev4's q_prep then k_compress on the snapshots; compressed rows go to a [groups, D]
    scratch (row g = group g). Returns (q_ref, ring_ref, rope_ref, ck_ref)."""
    from sglang.kernels.ops.attention.qsa_indexer import (
        qsa_index_k_compress_store,
        qsa_index_q_norm_rope_store,
    )

    ring_ref, rope_ref = snap
    q_ref = qsa_index_q_norm_rope_store(
        qk,
        positions,
        cos_sin_cache,
        axis_map,
        q_weight,
        cache_loc,
        ring_ref,
        rope_ref,
        num_q_heads,
        rotary_dim,
        q_eps,
        is_neox_style,
        q_heads_padded=q_heads_padded,
        out_dtype=out_dtype,
    )
    groups = group_locs.shape[0]
    ck_ref = torch.empty((groups, q_weight.shape[0]), dtype=out_dtype, device=qk.device)
    qsa_index_k_compress_store(
        ring_ref,
        group_locs,
        rope_ref,
        cos_sin_cache,
        axis_map,
        k_weight,
        torch.arange(groups, dtype=torch.int32, device=qk.device),
        ck_ref,
        compress_ratio,
        rotary_dim,
        k_eps,
        is_neox_style,
    )
    return q_ref, ring_ref, rope_ref, ck_ref


def shadow_compare(
    ref,
    *,
    q,
    key_state_buffer,
    rope_position_buffer,
    compressed_k_buffer,
    write_locs,
    compress_ratio,
) -> None:
    """Accumulate mismatch counts. Ring rows [0, ratio) belong to request slot 0 (graph padding
    rows; their writers race in rev4 too) and compressed slot 0 is the padding slot."""
    q_ref, ring_ref, rope_ref, ck_ref = ref
    counts = shadow_counts(q.device)
    if counts is None:
        return
    r0 = compress_ratio
    q_mm = (_bits(q) != _bits(q_ref)).sum()
    ring_mm = (_bits(key_state_buffer[r0:]) != _bits(ring_ref[r0:])).sum()
    rope_mm = (rope_position_buffer[r0:] != rope_ref[r0:]).sum()
    live = (write_locs != 0)
    got = compressed_k_buffer.index_select(0, write_locs.long())
    ck_mm = ((_bits(got) != _bits(ck_ref)).sum(dim=1) * live).sum()
    delta = torch.stack(
        [
            torch.ones((), dtype=torch.int64, device=q.device),
            torch.full((), q.shape[0], dtype=torch.int64, device=q.device),
            q_mm.to(torch.int64),
            ring_mm.to(torch.int64),
            rope_mm.to(torch.int64),
            live.sum().to(torch.int64),
            ck_mm.to(torch.int64),
        ]
    )
    counts[: delta.shape[0]].add_(delta)


def _sorted_selected_scores(indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per row, the sorted scores of the selected block indices (-1 / out-of-range = -inf)."""
    idx = indices.reshape(indices.shape[0], -1).long()
    valid = (idx >= 0) & (idx < logits.shape[1])
    vals = torch.gather(logits.float(), 1, idx.clamp(0, logits.shape[1] - 1))
    vals = torch.where(valid, vals, torch.full_like(vals, float("-inf")))
    return torch.sort(vals, dim=-1).values


def _bits32(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.int32) if t.element_size() == 4 else _bits(t).to(torch.int32)


def shadow_compare_topk(
    got, ref, aa=None, aa_logits=None, ref_logits=None, lengths=None, min_scored=512
) -> None:
    """Top-512 of the fused Q (``got``, the selection attention uses) vs rev4's Q (``ref``).
    topk_mismatch: rows whose index sets differ. With logits: logit_mismatch counts scorer
    outputs that differ bitwise between the two Qs, and topk_value_mismatch counts rows whose
    sorted selected-score multisets differ (a tie-aware exact check: the top-k may pick any of
    several blocks with the same score). topk_aa_set_mismatch: rows where rev4's top-k rerun on
    the same Q picks a different set (its own run-to-run choice among ties)."""
    counts = shadow_counts(got.device)
    if counts is None:
        return
    a = torch.sort(got.reshape(got.shape[0], -1), dim=-1).values
    b = torch.sort(ref.reshape(ref.shape[0], -1), dim=-1).values
    i_tk = _SHADOW_FIELDS.index("topk_mismatch")
    counts[i_tk : i_tk + 1].add_((a != b).any(dim=-1).sum().to(torch.int64).reshape(1))
    counts[i_tk + 1 : i_tk + 2].add_(torch.full((1,), got.shape[0], dtype=torch.int64, device=got.device))
    if aa is not None:
        c = torch.sort(aa.reshape(aa.shape[0], -1), dim=-1).values
        i = _SHADOW_FIELDS.index("topk_aa_set_mismatch")
        counts[i : i + 1].add_((a != c).any(dim=-1).sum().to(torch.int64).reshape(1))
    if (
        aa_logits is not None
        and ref_logits is not None
        and aa_logits.shape == ref_logits.shape
        and lengths is not None
        and lengths.shape[0] == aa_logits.shape[0]
    ):
        # Only scored entries: rows longer than the top-k (shorter rows select 0..len-1 without
        # scores and the scorer may skip them) and columns below each row's length.
        lens = lengths.reshape(-1, 1).long()
        cols = torch.arange(aa_logits.shape[1], device=aa_logits.device).reshape(1, -1)
        scored_row = lens > min_scored
        live = (cols < lens) & scored_row
        i = _SHADOW_FIELDS.index("logit_mismatch")
        lm = ((_bits32(aa_logits) != _bits32(ref_logits)) & live).sum()
        counts[i : i + 1].add_(lm.to(torch.int64).reshape(1))
        if got.shape[-1] == ref.shape[-1] and got.ndim == 2 and got.shape[0] == aa_logits.shape[0]:
            va = _sorted_selected_scores(got, aa_logits)
            vb = _sorted_selected_scores(ref, ref_logits)
            vm = (((_bits32(va) != _bits32(vb)).any(dim=-1, keepdim=True)) & scored_row).sum()
            i = _SHADOW_FIELDS.index("topk_value_mismatch")
            counts[i : i + 1].add_(vm.to(torch.int64).reshape(1))


def qpf_shadow_log(every: int = 64) -> None:
    """Host-side, once per replay: print the device counters every ``every`` calls."""
    global _shadow_calls, _shadow_last
    if not QPF_SHADOW or not _shadow_counts:
        return
    _shadow_calls += 1
    if _shadow_calls % every:
        return
    total = None
    for counts in _shadow_counts.values():
        values = counts.tolist()
        total = values if total is None else [a + b for a, b in zip(total, values)]
    record = dict(zip(_SHADOW_FIELDS, total))
    if record != _shadow_last:
        _shadow_last = record
        print("[qpf-shadow] " + json.dumps(record), flush=True)
