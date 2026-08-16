"""MXFP4 KV cache quantize/dequantize ops (CUDA, sm86).

Kernels live in ``cuda_kernels/mxfp4_kv.cu`` and are JIT-compiled on first
use via ``torch.utils.cpp_extension.load``. Layout (block_size = 32):

    data  [S, H, D/2]  uint8   <- packed E2M1 (2 per byte, lo = even idx)
    scale [S, H, D/32] uint8   <- E8M0 exponent-only scale

Only head_dim = 128 is supported (Qwen3).
"""

import os

import torch
from torch.utils import cpp_extension

_HERE = os.path.dirname(os.path.abspath(__file__))

_ext = None


_CU_SRC = os.path.join(_HERE, "cuda_kernels", "mxfp4_kv.cu")
_FUSED_SRC = os.path.join(_HERE, "cuda_kernels", "mxfp4_decode_fused.cu")

# Declarations only; definitions live in cuda_sources (kernel launch syntax).
_CPP_SRC = r"""
#include <torch/extension.h>
void mxfp4_quantize_store(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, int64_t, int64_t, int64_t);
void mxfp4_dequantize(torch::Tensor, torch::Tensor, torch::Tensor,
                      int64_t, int64_t, int64_t);
void mxfp4_dequantize_indices(torch::Tensor, torch::Tensor, torch::Tensor,
                              torch::Tensor, int64_t, int64_t, int64_t);
void mxfp4_decode_fused(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, int64_t, int64_t, double, int64_t);
"""

# Launch wrappers: 256-thread CTA, 16 (token, head) rows per CTA.
_CUDA_WRAPPER_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

// All kernels must launch on the caller's current stream: sglang runs its
// forward on a non-default CUDA stream, while a bare <<<>>> launch goes to the
// NULL (legacy default) stream with no ordering vs. torch ops (e.g. the
// staging copy_ above) — that silently corrupts the input data.
static cudaStream_t cur_stream(int64_t s) { return (cudaStream_t)s; }

void mxfp4_quantize_store(torch::Tensor cache_kv, torch::Tensor loc,
                          torch::Tensor data, torch::Tensor scale,
                          int64_t t, int64_t h, int64_t stream) {
  const int rows = (int)(t * h);
  const int grid = (rows + 15) / 16;
  mxfp4_quantize_store_kernel<<<grid, 256, 0, cur_stream(stream)>>>(
      (const __nv_bfloat16*)cache_kv.data_ptr(),
      (int64_t*)loc.data_ptr(), (uint8_t*)data.data_ptr(),
      (uint8_t*)scale.data_ptr(), (int)t, (int)h);
}

void mxfp4_dequantize(torch::Tensor data, torch::Tensor scale,
                      torch::Tensor out, int64_t t, int64_t h,
                      int64_t stream) {
  const int rows = (int)(t * h);
  const int grid = (rows + 15) / 16;
  mxfp4_dequantize_kernel<<<grid, 256, 0, cur_stream(stream)>>>(
      (const uint8_t*)data.data_ptr(), (const uint8_t*)scale.data_ptr(),
      (__nv_bfloat16*)out.data_ptr(), (int)t, (int)h);
}

void mxfp4_dequantize_indices(torch::Tensor data, torch::Tensor scale,
                              torch::Tensor indices, torch::Tensor out,
                              int64_t i, int64_t h, int64_t stream) {
  const int rows = (int)(i * h);
  const int grid = (rows + 15) / 16;
  mxfp4_dequantize_indices_kernel<<<grid, 256, 0, cur_stream(stream)>>>(
      (const uint8_t*)data.data_ptr(), (const uint8_t*)scale.data_ptr(),
      (const int*)indices.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
      (int)i, (int)h);
}

void mxfp4_decode_fused(torch::Tensor q, torch::Tensor k_data, torch::Tensor k_scale,
                        torch::Tensor v_data, torch::Tensor v_scale,
                        torch::Tensor kv_indices, torch::Tensor kv_indptr,
                        torch::Tensor o, torch::Tensor lse, int64_t num_qo_heads,
                        int64_t num_kv_heads, double sm_scale, int64_t stream) {
  flashinfer::Mxfp4DecodeParams params;
  params.q = (const __nv_bfloat16*)q.data_ptr();
  // DEBUG passthrough mode: k_data/v_data are bf16 tensors
  params.k_data = (const uint8_t*)k_data.data_ptr();
  params.k_scale = (const uint8_t*)k_scale.data_ptr();
  params.v_data = (const uint8_t*)v_data.data_ptr();
  params.v_scale = (const uint8_t*)v_scale.data_ptr();
  params.kv_indices = (const int*)kv_indices.data_ptr();
  params.kv_indptr = (const int*)kv_indptr.data_ptr();
  params.o = (__nv_bfloat16*)o.data_ptr();
  params.lse = lse.numel() ? (float*)lse.data_ptr() : nullptr;
  params.n = (int)kv_indices.numel();
  params.num_qo_heads = (int)num_qo_heads;
  params.num_kv_heads = (int)num_kv_heads;
  params.sm_scale = (float)sm_scale;
  const int batch = (int)kv_indptr.numel() - 1;
  dim3 grid(batch, num_kv_heads);
  dim3 block(flashinfer::kBdx, flashinfer::kBdy, flashinfer::kBdz);
  // K/V fp16 tiles (2 stages x K+V) + float merge area for sync_states.
  const size_t smem = 2 * flashinfer::kStages * flashinfer::kBdz * flashinfer::kBdy *
                      flashinfer::kTile * flashinfer::kHeadDim * sizeof(__half) +
                      2 * (flashinfer::kBdz * flashinfer::kBdy * flashinfer::kHeadDim *
                               sizeof(float) +
                           flashinfer::kBdz * flashinfer::kBdy * 2 * sizeof(float));
  mxfp4_decode_fused_kernel<<<grid, block, smem, cur_stream(stream)>>>(
      *reinterpret_cast<flashinfer::Mxfp4DecodeParams*>(&params));
}
"""


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    with open(_CU_SRC) as f:
        cuda_src = f.read()
    with open(_FUSED_SRC) as f:
        fused_src = f.read()
    _ext = cpp_extension.load_inline(
        name="mxfp4_kv",
        cpp_sources=_CPP_SRC,
        cuda_sources=[cuda_src, fused_src, _CUDA_WRAPPER_SRC],
        functions=[
            "mxfp4_quantize_store",
            "mxfp4_dequantize",
            "mxfp4_dequantize_indices",
            "mxfp4_decode_fused",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_86,code=sm_86",
            "-std=c++17",
        ],
        extra_include_paths=[os.path.join(_HERE, "cuda_kernels")],
        verbose=False,
    )
    return _ext


# The kernel reads input tensors from GPU memory that must be plain cudaMalloc
# allocations. torch's small-object pool (cudaMemMap-backed, used for small
# tensors) faults on device reads under this driver (CUDA 12.6 / 570), so
# stage inputs through preallocated large buffers. The staged copies also stay
# alive across the async kernel launch.
_KV_BUF_CAP = 32 << 20  # 32M bf16 elements = 64MB, plain cudaMalloc
_LOC_BUF_CAP = 1 << 20  # 1M slots = 8MB, plain cudaMalloc
_IDX_BUF_CAP = 1 << 20  # 1M indices = 4MB, plain cudaMalloc
_Q_BUF_CAP = 8 << 20  # 8M bf16 elements = 16MB (>= 2048 x 32 heads x 128 dim), plain cudaMalloc
_buffers = {}  # name -> base tensor; slices keep the base alive here
# copy_ is async; the source (possibly a temporary from .contiguous()/.to())
# must stay referenced until the copy kernel has run. A forward stages K/V for
# every layer (36 layers x K+V = 72 calls per forward), and with the
# per-call synchronize removed (CUDA-graph capture forbids it) the copies stay
# in flight behind the GPU; the ring must never recycle a still-referenced
# source. 256 slots cover several concurrent forwards (overlap schedule).
# References cost no memory; only _buffers holds real allocations.
_keepalive = {}
_keepalive_ring = {}


def _ring_next(name: str) -> int:
    slot = _keepalive_ring.get(name, 0)
    _keepalive_ring[name] = (slot + 1) % 256
    return slot


def _stage(name: str, tensor: torch.Tensor, cap: int, dtype=None) -> torch.Tensor:
    """Copy into a large plain-cudaMalloc buffer; return the slice view."""
    n = tensor.numel()
    assert n <= cap, f"{name} too large: {n} > {cap}"
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    buf = _buffers.get(name)
    if buf is None:
        buf = torch.empty(cap, dtype=tensor.dtype, device="cuda")
        _buffers[name] = buf
    slot = _ring_next(name)
    _keepalive[(name, slot)] = tensor
    buf[:n].view(tensor.shape).copy_(tensor)
    return buf[:n]


_out_ring = []  # ring of large plain-cudaMalloc buffers for kernel outputs


def alloc_output(shape, dtype) -> torch.Tensor:
    """Fresh slice of a large plain-cudaMalloc buffer for a kernel output.

    Same rationale as _stage: outputs allocated during CUDA graph capture land
    in the graph memory pool (cudaMemMap-backed), which custom kernels cannot
    safely read/write under this driver (CUDA 12.6 / 570). A small ring keeps
    concurrent forwards (overlap schedule) from reusing a live buffer.
    """
    n = 1
    for s in shape:
        n *= s
    if not _out_ring:
        for _ in range(8):
            _out_ring.append(torch.empty(_Q_BUF_CAP, dtype=dtype, device="cuda"))
    buf = _out_ring[_ring_next("out") % 8]
    assert n <= buf.numel(), f"output too large: {n} > {buf.numel()}"
    return buf[:n].view(shape)


def quantize_and_store(
    cache_kv: torch.Tensor,  # [T, H, 128] bf16
    loc: torch.Tensor,       # [T] int64 slot per token
    data: torch.Tensor,      # [S, H, 64] uint8 out
    scale: torch.Tensor,     # [S, H, 4] uint8 out
) -> None:
    """Quantize bf16 KV rows and scatter them into pool slots."""
    ext = _load_ext()
    t, h, d = cache_kv.shape
    assert d == 128, "MXFP4 KV kernel only supports head_dim=128"
    cache_kv = _stage("kv", cache_kv.contiguous(), _KV_BUF_CAP)
    loc = _stage("loc", loc, _LOC_BUF_CAP, dtype=torch.int64)
    ext.mxfp4_quantize_store(
        cache_kv, loc, data, scale, t, h, torch.cuda.current_stream().cuda_stream
    )


def dequantize(
    data: torch.Tensor,  # [T, H, 64] uint8
    scale: torch.Tensor, # [T, H, 4] uint8
    out: torch.Tensor,   # [T, H, 128] bf16
) -> None:
    """Dequantize contiguous rows."""
    ext = _load_ext()
    t, h, _ = data.shape
    ext.mxfp4_dequantize(
        data, scale, out, t, h, torch.cuda.current_stream().cuda_stream
    )


def dequantize_indices(
    data: torch.Tensor,   # [S, H, 64] uint8
    scale: torch.Tensor,  # [S, H, 4] uint8
    indices: torch.Tensor,  # [I] int32 slot list
    out: torch.Tensor,    # [I, H, 128] bf16
) -> None:
    """Dequantize rows gathered in ``indices`` order (flashinfer kv_indices)."""
    ext = _load_ext()
    i = indices.numel()
    h = data.shape[1]
    indices = _stage("idx", indices, _IDX_BUF_CAP, dtype=torch.int32)
    ext.mxfp4_dequantize_indices(
        data, scale, indices, out, i, h, torch.cuda.current_stream().cuda_stream
    )


def decode_fused(
    q: torch.Tensor,       # [batch, qo_heads, 128] bf16
    k_data: torch.Tensor,  # [S, H, 64] u8
    k_scale: torch.Tensor, # [S, H, 4] u8
    v_data: torch.Tensor,
    v_scale: torch.Tensor,
    kv_indices: torch.Tensor,  # [n] int32 pool slots
    kv_indptr: torch.Tensor,   # [batch+1] int32
    o: torch.Tensor,           # [batch, qo_heads, 128] bf16 out
    lse: torch.Tensor,         # [batch, qo_heads] float out
    sm_scale: float,
) -> None:
    """Fused MXFP4 decode attention (reads packed fp4 KV directly)."""
    ext = _load_ext()
    batch = kv_indptr.numel() - 1
    # q arrives as flat [bs, qh*head_dim] from the attention backend (or
    # [bs, qh, head_dim] in unit tests); derive qh so per-request q offsets
    # (bx * qh + head) * head_dim stay in range for batch > 1.
    qh = q.shape[1] // 128 if q.dim() == 2 else q.shape[1]
    kh = k_data.shape[1]
    # Stage q too: under CUDA graphs the query comes from a small graph-input
    # buffer (cudaMemMap small-object pool) which custom kernels cannot read.
    q = _stage("fused_q", q.contiguous(), _Q_BUF_CAP)
    kv_indices = _stage("fused_idx", kv_indices, _IDX_BUF_CAP, dtype=torch.int32)
    kv_indptr = _stage("fused_indptr", kv_indptr, 1 << 16, dtype=torch.int32)
    ext.mxfp4_decode_fused(
        q, k_data, k_scale, v_data, v_scale, kv_indices, kv_indptr,
        o, lse, qh, kh, sm_scale, torch.cuda.current_stream().cuda_stream,
    )


def reference_quantize(x: torch.Tensor, block: int = 32) -> tuple:
    """Torch reference (MXFP4, block 16 or 32), for kernel validation."""
    b, m, n = x.shape
    reshaped = x.view(b, m * n // block, block).float()  # full-precision max, like the kernel
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    # exp = ceil(log2(block_max / 6)); block_max == 0 -> -127
    safe = torch.clamp(block_max / 6.0, min=1e-10)
    scale_exp = torch.ceil(torch.log2(safe))
    scale_exp = torch.where(block_max == 0, torch.full_like(scale_exp, -127.0), scale_exp)
    scale_exp = torch.clamp(scale_exp, -127, 127)
    scale_bits = (scale_exp + 127).to(torch.uint8)
    scaled = reshaped / torch.exp2(scale_exp)
    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) > bounds, dim=-1).to(torch.uint8)
    # Round-to-nearest-even at exact midpoints (odd index rounds up).
    for bound in bounds:
        mid = (abs_vals == bound) & (magnitude_bits % 2 == 1)
        magnitude_bits = magnitude_bits + mid.to(torch.uint8)
    fp4 = (sign_bits + magnitude_bits).view(b, m, n)
    packed = (fp4[..., 1::2] << 4) + fp4[..., 0::2]
    return packed, scale_bits.view(b, m, n // block)
