"""Token-strip GDN prefill causal conv (width 4) with the q / k / v split
output: bit-identical to the Triton ``_causal_conv1d_fwd_kernel`` split launch
(``causal_conv1d_fn_split_qkv``), which tiles 8 tokens per program and loads
every input row four times.

Here one thread owns 8 channels of a sequence's token strip and carries the
previous three rows in registers, so each input row is read once. Per output
it replays the Triton op sequence: bf16 products (mul.rn.bf16), an fp32 sum
in tap order from 0 (+ bias), SiLU as acc / (1 + ex2.approx(-acc * log2 e))
with div.full, and one round to bf16. Used only when
SGLANG_GDN_PREFILL_CONV_RETILE is set.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

CUDA_SOURCE = r'''#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

namespace sglang {

namespace gdn_strip {

// mul.bf16 of the Triton kernel, two lanes at once (each lane one RN product).
SGL_DEVICE uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

SGL_DEVICE fp32x2_t widen(uint32_t v) {
  // bf16 -> fp32 is exact: the halves shifted into the high 16 bits.
  return fp32x2_t{__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)};
}

SGL_DEVICE fp32x2_t add2(fp32x2_t a, fp32x2_t b) {
  fp32x2_t d;
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "add.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}

// Triton's acc / (1 + tl.exp(-acc)) on NVIDIA: exp(z) = ex2.approx(z * log2 e),
// then an fp32 add and div.full.
SGL_DEVICE float silu(float acc) {
  float e;
  asm("{\n\t.reg .f32 z;\n\tneg.f32 z, %1;\n\tmul.f32 z, z, 0f3FB8AA3B;\n\tex2.approx.f32 %0, z;\n\t}"
      : "=f"(e)
      : "f"(acc));
  float denom;
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(denom) : "f"(e));
  float out;
  asm("div.full.f32 %0, %1, %2;" : "=f"(out) : "f"(acc), "f"(denom));
  return out;
}

SGL_DEVICE uint32_t round2(float x, float y) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<const uint32_t*>(&v);
}

}  // namespace gdn_strip

struct GdnConvStripParams {
  const uint16_t* x;                // element (c, t) at x[c + t * x_stride_tok]
  const uint16_t* weight;           // element (c, k) at weight[c * w_stride_dim + k * w_stride_k]
  const uint16_t* bias;             // [dim] (kHasBias)
  uint16_t* conv_states;            // element (line, c, k) at [line * s_seq + c * s_dim + k * s_tok]
  const int32_t* cache_indices;     // [nseq]; nullptr: sequence i uses line i
  const bool* has_initial_state;    // [nseq]; nullptr: no initial state
  const int32_t* query_start_loc;   // [nseq + 1]
  uint16_t* out;                    // flat q [T, d0] | k [T, d1 - d0] | v [T, dim - d1]
  int64_t x_stride_tok, w_stride_dim, w_stride_k, s_seq, s_dim, s_tok, total_tokens;
  int32_t dim, d0, d1, num_cache_lines, pad_slot_id;
};

/**
 * \brief Width-4 causal conv over a sequence's token strip, split q / k / v output.
 *
 * Grid (channel groups, strips, sequences); thread t of a group owns channels
 * c0 .. c0 + 7 of strip y of sequence z. Strip 0 also owns the sequence's
 * conv-state update, after it has read the initial state, as the Triton chunk-0
 * program does; no other strip reads the state (kTokens >= 3).
 *
 * \tparam kThreads  threads per CTA (8 channels each)
 * \tparam kTokens   tokens per strip
 * \tparam kUnroll   rows loaded ahead per step
 */
template <int kThreads, int kTokens, int kUnroll, bool kSilu, bool kHasBias>
__global__ __launch_bounds__(kThreads) void gdn_prefill_conv_strip_kernel(const GdnConvStripParams __grid_constant__ p) {
  using Vec = device::AlignedVector<uint32_t, 4>;  // 8 bf16
  constexpr int kWidth = 4;
  static_assert(kTokens >= kWidth - 1 && kTokens % kUnroll == 0);

  const int c0 = (static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x)) * 8;
  if (c0 >= p.dim) return;
  const int seq = blockIdx.z;
  const int64_t start = p.query_start_loc[seq];
  const int64_t len = p.query_start_loc[seq + 1] - start;
  const int64_t t_begin = static_cast<int64_t>(blockIdx.y) * kTokens;
  if (t_begin >= len) return;
  const int64_t line = p.cache_indices != nullptr ? static_cast<int64_t>(p.cache_indices[seq]) : seq;
  if (line == p.pad_slot_id) return;

  const uint16_t* x_seq = p.x + c0 + start * p.x_stride_tok;
  auto load_row = [&](int64_t t) {
    Vec v;
    v.load(x_seq + t * p.x_stride_tok);
    return v;
  };

  // Weights as bf16 pairs per tap: w2[k][q] = (w[c0 + 2q][k], w[c0 + 2q + 1][k]).
  uint32_t w2[kWidth][4];
#pragma unroll
  for (int k = 0; k < kWidth; ++k) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t lo = p.weight[(c0 + 2 * q) * p.w_stride_dim + k * p.w_stride_k];
      const uint32_t hi = p.weight[(c0 + 2 * q + 1) * p.w_stride_dim + k * p.w_stride_k];
      w2[k][q] = lo | (hi << 16);
    }
  }
  fp32x2_t acc0[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    if constexpr (kHasBias) {
      // Triton: tl.zeros(...) + bias.to(fp32).
      const float b0 = __uint_as_float(static_cast<uint32_t>(p.bias[c0 + 2 * q]) << 16);
      const float b1 = __uint_as_float(static_cast<uint32_t>(p.bias[c0 + 2 * q + 1]) << 16);
      acc0[q] = gdn_strip::add2(fp32x2_t{0.0f, 0.0f}, fp32x2_t{b0, b1});
    } else {
      acc0[q] = fp32x2_t{0.0f, 0.0f};
    }
  }

  // Rows t - 3, t - 2, t - 1 of the first token of the strip.
  Vec win[kWidth - 1];
  if (t_begin == 0) {
    uint16_t* state = p.conv_states + line * p.s_seq + static_cast<int64_t>(c0) * p.s_dim;
    const bool load_init = p.has_initial_state != nullptr && p.has_initial_state[seq];
#pragma unroll
    for (int k = 0; k < kWidth - 1; ++k) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t v = 0;
        if (load_init) {
          v = static_cast<uint32_t>(state[(2 * q) * p.s_dim + k * p.s_tok]) |
              (static_cast<uint32_t>(state[(2 * q + 1) * p.s_dim + k * p.s_tok]) << 16);
        }
        win[k][q] = v;
      }
    }
    // The new state: the last three inputs, shifted in after the old state for
    // a sequence shorter than three (zeros without an initial state).
    uint16_t next_state[8][kWidth - 1];
#pragma unroll
    for (int k = 0; k < kWidth - 1; ++k) {
      const int64_t src = len - (kWidth - 1) + k;  // input row feeding state column k
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint16_t v = 0;
        if (src >= 0) {
          v = x_seq[src * p.x_stride_tok + i];
        } else if (load_init && line < p.num_cache_lines) {
          v = state[i * p.s_dim + (k + len) * p.s_tok];
        }
        next_state[i][k] = v;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int k = 0; k < kWidth - 1; ++k) {
        state[i * p.s_dim + k * p.s_tok] = next_state[i][k];
      }
    }
  } else {
#pragma unroll
    for (int k = 0; k < kWidth - 1; ++k) {
      win[k] = load_row(t_begin - (kWidth - 1) + k);
    }
  }

  int64_t col_base, width;
  if (c0 < p.d0) {
    col_base = 0, width = p.d0;
  } else if (c0 < p.d1) {
    col_base = p.d0, width = p.d1 - p.d0;
  } else {
    col_base = p.d1, width = p.dim - p.d1;
  }
  uint16_t* o_seq = p.out + col_base * p.total_tokens + (c0 - col_base) + start * width;

  const int64_t t_end = (t_begin + kTokens < len) ? t_begin + kTokens : len;
  Vec cur[kUnroll];
#pragma unroll
  for (int u = 0; u < kUnroll; ++u) {
    if (t_begin + u < t_end) cur[u] = load_row(t_begin + u);
  }
  for (int64_t t = t_begin; t < t_end; t += kUnroll) {
    Vec nxt[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (t + kUnroll + u < t_end) nxt[u] = load_row(t + kUnroll + u);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (t + u >= t_end) break;
      Vec res;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        fp32x2_t acc = acc0[q];
        acc = gdn_strip::add2(acc, gdn_strip::widen(gdn_strip::mul_bf16x2(win[0][q], w2[0][q])));
        acc = gdn_strip::add2(acc, gdn_strip::widen(gdn_strip::mul_bf16x2(win[1][q], w2[1][q])));
        acc = gdn_strip::add2(acc, gdn_strip::widen(gdn_strip::mul_bf16x2(win[2][q], w2[2][q])));
        acc = gdn_strip::add2(acc, gdn_strip::widen(gdn_strip::mul_bf16x2(cur[u][q], w2[3][q])));
        if constexpr (kSilu) {
          acc = fp32x2_t{gdn_strip::silu(acc.x), gdn_strip::silu(acc.y)};
        }
        res[q] = gdn_strip::round2(acc.x, acc.y);
      }
      res.store(o_seq + (t + u) * width);
      win[0] = win[1];
      win[1] = win[2];
      win[2] = cur[u];
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      cur[u] = nxt[u];
    }
  }
}

template <int kThreads, int kTokens, int kUnroll, bool kSilu, bool kHasBias>
struct GdnPrefillConvStrip {
  static constexpr auto kernel = gdn_prefill_conv_strip_kernel<kThreads, kTokens, kUnroll, kSilu, kHasBias>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView conv_states,
      const tvm::ffi::TensorView cache_indices,
      const tvm::ffi::TensorView has_initial_state,
      const tvm::ffi::TensorView query_start_loc,
      const tvm::ffi::TensorView out,
      int64_t max_seq_len,
      int64_t d0,
      int64_t d1,
      int64_t use_cache_indices,
      int64_t use_initial_state,
      int64_t pad_slot_id) {
    using namespace host;
    RuntimeCheck(x.ndim() == 2 && x.stride(0) == 1 && x.stride(1) % 8 == 0, "x must be a [dim, T] channel-last view");
    const int64_t dim = x.size(0);
    const int64_t total = x.size(1);
    const int64_t nseq = query_start_loc.size(0) - 1;
    RuntimeCheck(dim % 8 == 0 && d0 % 8 == 0 && d1 % 8 == 0 && 0 < d0 && d0 < d1 && d1 < dim, "bad split dims");
    RuntimeCheck(out.numel() == dim * total, "out must hold dim * T elements");
    RuntimeCheck(conv_states.ndim() == 3 && conv_states.size(1) == dim && conv_states.size(2) >= 3, "bad conv_states");
    RuntimeCheck(weight.ndim() == 2 && weight.size(0) == dim && weight.size(1) == 4, "weight must be [dim, 4]");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                 "x and out must be 16-byte aligned");
    if (nseq <= 0 || total == 0 || max_seq_len <= 0) return;
    const auto params = GdnConvStripParams{
        .x = static_cast<const uint16_t*>(x.data_ptr()),
        .weight = static_cast<const uint16_t*>(weight.data_ptr()),
        .bias = kHasBias ? static_cast<const uint16_t*>(bias.data_ptr()) : nullptr,
        .conv_states = static_cast<uint16_t*>(conv_states.data_ptr()),
        .cache_indices = use_cache_indices ? static_cast<const int32_t*>(cache_indices.data_ptr()) : nullptr,
        .has_initial_state = use_initial_state ? static_cast<const bool*>(has_initial_state.data_ptr()) : nullptr,
        .query_start_loc = static_cast<const int32_t*>(query_start_loc.data_ptr()),
        .out = static_cast<uint16_t*>(out.data_ptr()),
        .x_stride_tok = x.stride(1),
        .w_stride_dim = weight.stride(0),
        .w_stride_k = weight.stride(1),
        .s_seq = conv_states.stride(0),
        .s_dim = conv_states.stride(1),
        .s_tok = conv_states.stride(2),
        .total_tokens = total,
        .dim = static_cast<int32_t>(dim),
        .d0 = static_cast<int32_t>(d0),
        .d1 = static_cast<int32_t>(d1),
        .num_cache_lines = static_cast<int32_t>(conv_states.size(0)),
        .pad_slot_id = static_cast<int32_t>(pad_slot_id),
    };
    const dim3 grid(
        static_cast<uint32_t>((dim / 8 + kThreads - 1) / kThreads),
        static_cast<uint32_t>((max_seq_len + kTokens - 1) / kTokens),
        static_cast<uint32_t>(nseq));
    LaunchKernel(grid, kThreads, x.device())(kernel, params);
  }
};

}  // namespace sglang
'''


def _source_path() -> str:
    from sglang.kernels.jit.utils.compile.cache import cache_root

    data = CUDA_SOURCE.encode()
    digest = hashlib.sha256(data).hexdigest()[:16]
    directory = cache_root() / "inline-src"
    path = directory / f"gdn_prefill_conv_strip-{digest}.cuh"
    if not path.is_file() or path.read_bytes() != data:
        directory.mkdir(parents=True, exist_ok=True)
        staged = directory / f".{path.name}.{os.getpid()}.tmp"
        staged.write_bytes(data)
        os.replace(staged, path)
    return str(path)


def _b(x: bool) -> str:
    return "true" if x else "false"


@cache_once
def _jit_module(threads: int, tokens: int, unroll: int) -> Module:
    args = make_cpp_args(threads, tokens, unroll)
    return load_jit(
        "gdn_prefill_conv_strip",
        *args,
        cuda_files=[_source_path()],
        cuda_wrappers=[
            (f"run_{int(silu)}{int(bias)}", f"GdnPrefillConvStrip<{args}, {_b(silu)}, {_b(bias)}>::run")
            for silu in (False, True)
            for bias in (False, True)
        ],
    )


def supported(x: torch.Tensor, weight: torch.Tensor, conv_states: Optional[torch.Tensor], q_dim: int, k_dim: int) -> bool:
    """Shapes and layouts the strip kernel serves; others keep the Triton launch."""
    dim = x.shape[0]
    return (
        x.dim() == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.shape == (dim, 4)
        and conv_states is not None
        and conv_states.dtype == torch.bfloat16
        and conv_states.dim() == 3
        and conv_states.shape[1] == dim
        and conv_states.shape[2] >= 3
        and x.stride(0) == 1
        and x.stride(1) % 8 == 0
        and x.data_ptr() % 16 == 0
        and dim % 8 == 0
        and q_dim % 8 == 0
        and k_dim % 8 == 0
        and 0 < q_dim < q_dim + k_dim < dim
    )


def warmup(threads: int, tokens: int, unroll: int) -> None:
    _jit_module(threads, tokens, unroll)


def conv_split_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_seq_len: int,
    q_dim: int,
    k_dim: int,
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu: bool,
    pad_slot_id: int,
    threads: int,
    tokens: int,
    unroll: int,
) -> torch.Tensor:
    """The flat [q | k | v] token-major output of causal_conv1d_fn(split_dims=(q_dim, q_dim + k_dim))."""
    out = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    module = _jit_module(threads, tokens, unroll)
    fn = getattr(module, f"run_{int(silu)}{int(bias is not None)}")
    fn(
        x,
        weight,
        bias if bias is not None else weight,
        conv_states,
        cache_indices if cache_indices is not None else query_start_loc,
        has_initial_state if has_initial_state is not None else query_start_loc,
        query_start_loc,
        out,
        int(max_seq_len),
        int(q_dim),
        int(q_dim + k_dim),
        int(cache_indices is not None),
        int(has_initial_state is not None),
        int(pad_slot_id),
    )
    return out
