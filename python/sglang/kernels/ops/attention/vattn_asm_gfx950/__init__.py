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
_available = None
_disabled_reason = None


class AsmKernelUnavailable(RuntimeError):
    """Raised once when the kernel cannot be assembled or loaded; the caller
    falls back to its Triton path and asm_kernel_available() stays False."""


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
        ("seg_plan_ptr", ctypes.c_int64),  # 0 = legacy fixed-SEGS split
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
        ("tok_nseg_ptr", ctypes.c_int64),  # 0 = legacy: reduce all num_segments
    ]


assert ctypes.sizeof(VattnKernelArgs) == 136
assert ctypes.sizeof(VredKernelArgs) == 64


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


def asm_kernel_available() -> bool:
    """True on a gfx950 device with ROCm clang present, until a build or load
    failure disables the kernel for the rest of the process."""
    global _available
    if _available is None:
        arch = ""
        if torch.version.hip and torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(0).gcnArchName
        _available = arch.startswith("gfx950") and os.path.exists(_clang())
    return _available and _disabled_reason is None


def _disable(reason):
    global _disabled_reason
    _disabled_reason = reason
    print(
        f"[vattn_asm] disabling the gfx950 asm attention kernel, falling back to "
        f"the Triton path: {reason}",
        flush=True,
    )
    raise AsmKernelUnavailable(reason)


def _build(name, text):
    global _build_dir
    if _build_dir is None:
        _build_dir = tempfile.mkdtemp(prefix="sglang_vattn_asm_")
    src = os.path.join(_build_dir, name + ".s")
    with open(src, "w") as f:
        f.write(text)
    co = os.path.join(_build_dir, name + ".co")
    try:
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
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", None) or str(e)
        _disable(f"assembling {name}.s failed: {detail.strip()[:400]}")
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
        try:
            kern = _Kernel(co, "vattn_asm", declared)
        except RuntimeError as e:
            _disable(f"loading {co} failed: {e}")
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
        try:
            kern = _Kernel(co, "vred_asm", declared)
        except RuntimeError as e:
            _disable(f"loading {co} failed: {e}")
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


_SEG_PLAN_TARGET_WGS = None


def _seg_plan_target_wgs() -> int:
    global _SEG_PLAN_TARGET_WGS
    if _SEG_PLAN_TARGET_WGS is None:
        _SEG_PLAN_TARGET_WGS = max(
            1,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
        )
    return _SEG_PLAN_TARGET_WGS


def mtp_verify_attn_seg_max(num_seqs: int, num_kv_heads: int) -> int:
    """Static grid.x for the planned split: 2x the legacy per-seq count, clamped to 16..64."""
    return max(16, min(64, 2 * mtp_verify_attn_num_segments(num_seqs, num_kv_heads)))


def _get_plan_kernel():
    kern = _kernels.get("plan")
    if kern is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _vattn_seg_plan_kernel(
            seq_lens_ptr,
            cu_q_ptr,
            plan_ptr,
            tok_nseg_ptr,
            num_seqs,
            num_work,
            target_wgs,
            seg_max,
            BLOCK_B: tl.constexpr,
            BLOCK_W: tl.constexpr,
            BLOCK_Q: tl.constexpr,
        ):
            # one program per sequence; every program recomputes the (cheap) batch-wide plan
            pid = tl.program_id(0)
            b = tl.arange(0, BLOCK_B)
            bm = b < num_seqs
            slen = tl.load(seq_lens_ptr + b, mask=bm, other=0).to(tl.int32)
            nt = (slen + 15) // 16
            total = tl.sum(nt, axis=0)
            mx = tl.max(nt, axis=0)
            # smallest T (tiles per segment) with sum_b ceil(nt_b / T) <= target_wgs and max_b ceil(nt_b / T) <= seg_max
            lo = tl.maximum(
                tl.maximum(
                    (total + target_wgs - 1) // target_wgs,
                    (mx + seg_max - 1) // seg_max,
                ),
                1,
            )
            slack = target_wgs - num_seqs
            hi = tl.where(
                slack > 0,
                (total + tl.maximum(slack, 1) - 1) // tl.maximum(slack, 1),
                lo,
            )
            hi = tl.maximum(hi, lo)
            for _ in range(16):
                mid = (lo + hi) // 2
                fits = tl.sum((nt + mid - 1) // mid, axis=0) <= target_wgs
                hi = tl.where(fits, mid, hi)
                lo = tl.where(fits, lo, mid + 1)
            T = hi
            nseg = (nt + T - 1) // T
            ends = tl.cumsum(nseg, axis=0)
            tot = tl.sum(nseg, axis=0)
            my_n = tl.sum(tl.where(b == pid, nseg, 0), axis=0)
            my_start = tl.sum(tl.where(b == pid, ends, 0), axis=0) - my_n
            if pid == 0:
                tl.store(plan_ptr, T)
            w = tl.arange(0, BLOCK_W)
            tl.store(plan_ptr + 1 + my_start + w, (pid << 16) | w, mask=w < my_n)
            idle = tot + pid + w * num_seqs  # idle tail, strided over programs
            tl.store(
                plan_ptr + 1 + idle,
                tl.full((BLOCK_W,), -1, tl.int32),
                mask=idle < num_work,
            )
            q0 = tl.load(cu_q_ptr + pid).to(tl.int32)
            q1 = tl.load(cu_q_ptr + pid + 1).to(tl.int32)
            for t0 in range(q0, q1, BLOCK_Q):
                t = t0 + tl.arange(0, BLOCK_Q)
                tl.store(
                    tok_nseg_ptr + t,
                    tl.full((BLOCK_Q,), 0, tl.int32) + my_n,
                    mask=t < q1,
                )

        kern = _vattn_seg_plan_kernel
        _kernels["plan"] = kern
    return kern


def seg_plan_target_wgs(num_kv_heads: int) -> int:
    """Working WGs to aim for: one per CU, shared over the kv-head grid dim."""
    return max(1, _seg_plan_target_wgs() // max(1, num_kv_heads))


def build_seg_plan(seq_lens, cu_seqlens_q, num_tokens, seg_max, num_kv_heads=1):
    """plan int32[1 + seg_max*num_seqs] = (T tiles/segment, work list seq<<16|seg, -1 past the end),
    tok_nseg int32[num_tokens] = segment count of the sequence owning each query token. One Triton launch,
    static shapes, graph-capture safe. T is the smallest segment length whose total WG count fits the CU
    budget, so uniform batches reproduce the legacy split exactly and skewed batches get per-length counts.
    """
    import triton

    num_seqs = seq_lens.shape[0]
    num_work = seg_max * num_seqs
    plan = torch.empty(1 + num_work, dtype=torch.int32, device=seq_lens.device)
    tok_nseg = torch.empty(
        max(num_tokens, 1), dtype=torch.int32, device=seq_lens.device
    )
    _get_plan_kernel()[(num_seqs,)](
        seq_lens,
        cu_seqlens_q,
        plan,
        tok_nseg,
        num_seqs,
        num_work,
        seg_plan_target_wgs(num_kv_heads),
        seg_max,
        BLOCK_B=max(16, triton.next_power_of_2(num_seqs)),
        BLOCK_W=64,
        BLOCK_Q=16,
        num_warps=4,
    )
    return plan, tok_nseg


_PLAN_CACHE = {}


def reset_seg_plan_cache():
    """Called by the attention backend at the start of every forward (eager and graph capture)."""
    _PLAN_CACHE.clear()


def _cached_seg_plan(seq_lens, cu_seqlens_q, num_tokens, seg_max, num_kv_heads):
    # torch.cuda.is_current_stream_capturing() is part of the key: graph capture warms
    # up and then records on the same tensors, and a plan built during warmup must not
    # be reused while recording (its kernel would be missing from the graph).
    key = (
        seq_lens.data_ptr(),
        cu_seqlens_q.data_ptr(),
        seq_lens._version,
        cu_seqlens_q._version,
        num_tokens,
        seq_lens.shape[0],
        seg_max,
        num_kv_heads,
        torch.cuda.is_current_stream_capturing(),
    )
    hit = _PLAN_CACHE.get(key)
    if hit is None:
        plan, tok_nseg = build_seg_plan(
            seq_lens, cu_seqlens_q, num_tokens, seg_max, num_kv_heads
        )
        # keep the key tensors alive so their storage cannot be reused under the same address while cached
        hit = _PLAN_CACHE[key] = (plan, tok_nseg, seq_lens, cu_seqlens_q)
    return hit[0], hit[1]


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
    use_seg_plan=True,
):
    """Same contract as aiter.mtp_verify_attn_fwd_asm (see that docstring).

    use_seg_plan=False forces the fixed per-sequence split of #37465 (used by the tests as the
    reference split); production callers leave it on."""
    num_tokens, num_q_heads, head_size = q.shape
    num_seqs = seq_lens.shape[0]
    num_kv_heads = k_cache.shape[2]
    plan = tok_nseg = None
    if num_segments is None:
        if (
            use_seg_plan and num_seqs > 1
        ):  # bs=1: nothing to balance, the fixed split already uses 64 segments
            num_segments = mtp_verify_attn_seg_max(num_seqs, num_kv_heads)
            plan, tok_nseg = _cached_seg_plan(
                seq_lens, cu_seqlens_q, num_tokens, num_segments, num_kv_heads
            )
        else:
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
        seg_plan_ptr=plan.data_ptr() if plan is not None else 0,
    )
    if plan is not None:
        # 1-D work list: working WGs first, idle tail exits in the prologue
        kern.launch(
            (num_segments * num_seqs, 1, num_kv_heads), (512, 1, 1), args, stream
        )
    else:
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
        tok_nseg_ptr=tok_nseg.data_ptr() if tok_nseg is not None else 0,
    )
    _get_reduce().launch((num_tokens * num_q_heads, 1, 1), (256, 1, 1), rargs, stream)
    return out
