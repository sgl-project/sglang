"""Hand-written gfx950 (MI35x) assembly attention for EAGLE/MTP verify, draft
extend (ragged q_len 1..4) and q_len-1 decode: head_dim 256, fp8 page-16 NHD
KV cache, GQA ratios 16 and 8.

vattn3_core.s (split-KV main kernel, one code object per GQA ratio) and vred.s
(segment reduce) ship as source and are assembled at first use with ROCm clang
into a per-process temp dir; launches go through ctypes hipModuleLaunchKernel
on the current torch stream. Kernarg ABI is guarded three ways:
  1. single ctypes.Structure(_pack_=1) definition, fields filled by name;
  2. sizeof() asserted against the expected constant at import;
  3. sizeof() cross-checked against the .amdhsa_kernarg_size the kernel itself
     declares in its .s, at build time.
Descale factors are read by the kernel from device pointers, so launches are
HIP-graph-capture safe (no host sync).
"""

import ctypes
import os
import re
import subprocess
import tempfile

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG2E = 1.4426950408889634

_hip = None
_kernels = {}
_build_dir = None


class VattnKernelArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("k_ptr", ctypes.c_int64),
        ("v_ptr", ctypes.c_int64),
        ("q_ptr", ctypes.c_int64),
        ("block_table_ptr", ctypes.c_int64),
        ("seq_lens_ptr", ctypes.c_int64),
        ("cu_seqlens_q_ptr", ctypes.c_int64),
        ("segm_out_ptr", ctypes.c_int64),
        ("segm_max_ptr", ctypes.c_int64),
        ("segm_expsum_ptr", ctypes.c_int64),
        ("num_segments", ctypes.c_int32),
        ("block_table_stride", ctypes.c_int32),
        ("q_stride", ctypes.c_int32),
        ("num_kv_heads", ctypes.c_int32),
        ("num_q_heads", ctypes.c_int32),
        ("scale_log2e", ctypes.c_float),
        ("p_bias", ctypes.c_float),
        ("magic_m", ctypes.c_uint32),
        ("magic_sh", ctypes.c_int32),
        ("_pad", ctypes.c_uint32),
        ("k_descale_ptr", ctypes.c_int64),
        ("v_descale_ptr", ctypes.c_int64),
    ]


class VredKernelArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("out_ptr", ctypes.c_int64),
        ("segm_out_ptr", ctypes.c_int64),
        ("segm_max_ptr", ctypes.c_int64),
        ("segm_expsum_ptr", ctypes.c_int64),
        ("num_q_heads", ctypes.c_uint32),
        ("num_segments", ctypes.c_uint32),
        ("out_stride0", ctypes.c_uint32),
        ("out_stride1", ctypes.c_uint32),
        ("magic_m", ctypes.c_uint32),
        ("magic_sh", ctypes.c_uint32),
    ]


assert ctypes.sizeof(VattnKernelArgs) == 128
assert ctypes.sizeof(VredKernelArgs) == 56


def _declared_kernarg_size(source_file):
    with open(os.path.join(_DIR, source_file)) as f:
        text = f.read()
    m = re.search(r"^\s*\.amdhsa_kernarg_size\s+(\d+)", text, re.MULTILINE)
    assert m, f"{source_file}: no .amdhsa_kernarg_size declaration"
    return int(m.group(1))


def _hip_lib():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipModuleLoad.restype = ctypes.c_int
        _hip.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _hip.hipModuleGetFunction.restype = ctypes.c_int
        _hip.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _hip.hipModuleLaunchKernel.restype = ctypes.c_int
        _hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(
            f"vattn_asm: {what} failed: {rc} "
            f"({_hip_lib().hipGetErrorString(rc).decode()})"
        )


class _Kernel:
    def __init__(self, co_path, name, kernarg_size):
        h = _hip_lib()
        self.mod = ctypes.c_void_p()
        _check(
            h.hipModuleLoad(ctypes.byref(self.mod), co_path.encode()),
            f"hipModuleLoad {co_path}",
        )
        self.fn = ctypes.c_void_p()
        _check(
            h.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, name.encode()),
            f"hipModuleGetFunction {name}",
        )
        self.kernarg_size = kernarg_size

    def launch(self, grid, block, args, stream):
        assert ctypes.sizeof(args) == self.kernarg_size
        size = ctypes.c_size_t(self.kernarg_size)
        extra = (ctypes.c_void_p * 5)(
            1,
            ctypes.cast(ctypes.byref(args), ctypes.c_void_p).value,
            2,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value,
            3,
        )
        _check(
            _hip_lib().hipModuleLaunchKernel(
                self.fn,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                ctypes.c_void_p(stream),
                None,
                ctypes.cast(extra, ctypes.c_void_p),
            ),
            "hipModuleLaunchKernel",
        )


def _clang():
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    return os.path.join(rocm, "llvm", "bin", "clang")


_TOOLCHAIN_AVAILABLE = None


def asm_toolchain_available() -> bool:
    global _TOOLCHAIN_AVAILABLE
    if _TOOLCHAIN_AVAILABLE is None:
        _TOOLCHAIN_AVAILABLE = os.path.exists(_clang())
    return _TOOLCHAIN_AVAILABLE


def _build(name, text):
    global _build_dir
    if _build_dir is None:
        _build_dir = tempfile.mkdtemp(prefix="sglang_vattn_asm_")
    src = os.path.join(_build_dir, name + ".s")
    with open(src, "w") as f:
        f.write(text)
    co = os.path.join(_build_dir, name + ".co")
    subprocess.run(
        [
            _clang(),
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "-I",
            _DIR,
            "-o",
            co,
            src,
        ],
        check=True,
        cwd=_DIR,
    )
    return co


def _get_kernel(hq):
    kern = _kernels.get(hq)
    if kern is None:
        declared = _declared_kernarg_size("vattn3_core.s")
        assert declared == ctypes.sizeof(VattnKernelArgs), (
            f"kernarg ABI drift: host struct is {ctypes.sizeof(VattnKernelArgs)} B, "
            f"vattn3_core.s declares {declared} B"
        )
        co = _build(
            f"hq{hq}_v3",
            f".set HQ, {hq}\n.set NBUF, 3\n"
            f".set NTDMA, 1\n.set ABL, 0\n"
            f'.include "vattn3_core.s"\n',
        )
        kern = _Kernel(co, "vattn_asm", declared)
        _kernels[hq] = kern
        print(
            f"[vattn_asm] gfx950 asm verify-attention kernel active "
            f"(in-tree loader, GQA ratio {hq}, built {co})",
            flush=True,
        )
    return kern


def _get_reduce():
    kern = _kernels.get("reduce")
    if kern is None:
        declared = _declared_kernarg_size("vred.s")
        assert declared == ctypes.sizeof(VredKernelArgs), (
            f"kernarg ABI drift: host struct is {ctypes.sizeof(VredKernelArgs)} B, "
            f"vred.s declares {declared} B"
        )
        co = _build("vred", '.set DBG, 0\n.include "vred.s"\n')
        kern = _Kernel(co, "vred_asm", declared)
        _kernels["reduce"] = kern
    return kern


def _magic_u32(d):
    """(M, sh): ((n*M) >> 32) >> sh == n//d for all n < 2**31."""
    if d & (d - 1) == 0:
        return (1 << 31, d.bit_length() - 2)
    s = d.bit_length() - 1
    M = ((1 << (32 + s)) + d - 1) // d
    assert M < (1 << 32)
    return (M, s)


def mtp_verify_attn_num_segments(num_seqs: int, num_kv_heads: int) -> int:
    num_cus = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    segs = max(1, num_cus // max(1, num_seqs * num_kv_heads))
    segs = 1 << (segs.bit_length() - 1)
    return max(1, min(64, segs))


def mtp_verify_attn_fwd_asm(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    cu_seqlens_q,
    k_descale,
    v_descale,
    softmax_scale,
    num_segments=None,
    out=None,
):
    """Same contract as aiter.mtp_verify_attn_fwd_asm (see that docstring)."""
    num_tokens, num_q_heads, head_size = q.shape
    num_seqs = seq_lens.shape[0]
    num_kv_heads = k_cache.shape[2]
    if num_segments is None:
        num_segments = mtp_verify_attn_num_segments(num_seqs, num_kv_heads)
    segm_out = torch.empty(
        num_tokens,
        num_q_heads,
        num_segments,
        head_size,
        dtype=torch.float32,
        device=q.device,
    )
    segm_max = torch.empty(
        num_tokens, num_q_heads, num_segments, dtype=torch.float32, device=q.device
    )
    segm_expsum = torch.empty_like(segm_max)
    if out is None:
        out = torch.empty(
            num_tokens, num_q_heads, head_size, dtype=q.dtype, device=q.device
        )
    stream = torch.cuda.current_stream().cuda_stream

    kern = _get_kernel(num_q_heads // num_kv_heads)
    m, sh = _magic_u32(num_segments * 16)
    args = VattnKernelArgs(
        k_ptr=k_cache.data_ptr(),
        v_ptr=v_cache.data_ptr(),
        q_ptr=q.data_ptr(),
        block_table_ptr=block_tables.data_ptr(),
        seq_lens_ptr=seq_lens.data_ptr(),
        cu_seqlens_q_ptr=cu_seqlens_q.data_ptr(),
        segm_out_ptr=segm_out.data_ptr(),
        segm_max_ptr=segm_max.data_ptr(),
        segm_expsum_ptr=segm_expsum.data_ptr(),
        num_segments=num_segments,
        block_table_stride=block_tables.stride(0),
        q_stride=q.stride(0),
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        scale_log2e=float(softmax_scale) * _LOG2E,
        p_bias=1.0,
        magic_m=m,
        magic_sh=sh,
        k_descale_ptr=k_descale.data_ptr(),
        v_descale_ptr=v_descale.data_ptr(),
    )
    kern.launch((num_segments, num_seqs, num_kv_heads), (512, 1, 1), args, stream)

    assert out.stride(2) == 1 and out.dtype == torch.bfloat16
    rm, rsh = _magic_u32(num_q_heads)
    rargs = VredKernelArgs(
        out_ptr=out.data_ptr(),
        segm_out_ptr=segm_out.data_ptr(),
        segm_max_ptr=segm_max.data_ptr(),
        segm_expsum_ptr=segm_expsum.data_ptr(),
        num_q_heads=num_q_heads,
        num_segments=num_segments,
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        magic_m=rm,
        magic_sh=rsh,
    )
    _get_reduce().launch((num_tokens * num_q_heads, 1, 1), (256, 1, 1), rargs, stream)
    return out
