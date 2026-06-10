"""Path B fusion for SwitchGLU: gate gather_qmv with silu(gate) * x_up epilogue.

Why this exists
---------------
The existing `FusedSwitchUpGate` (fused_switch_glu.py) concatenates up_proj
and gate_proj weights along the output dim and runs one gather_qmm. That saves
one kernel launch per layer but doubles the matmul's output dim, which pushes
MLX's quantized GEMV into a worse tile/occupancy config. At bs >= 4 on
Qwen3-30B-A3B-4bit this is a net regression (~2% slower at bs=32).

Path B keeps up_proj and gate_proj separate (matmul kernels see their natural
N — no tile regression) and instead fuses the *activation* into the gate
matmul. Concretely:

    Baseline (3 kernels per MoE layer in the swiglu front-half):
        x_up   = gather_qmm(x, W_up)
        x_gate = gather_qmm(x, W_gate)
        out    = silu(x_gate) * x_up         # 1 compiled kernel via mlx_lm.swiglu

    Path B (2 kernels):
        x_up = gather_qmm(x, W_up)
        out  = fused_gate_qmv_silu_mul(x, W_gate, ..., x_up)
                                              # one custom Metal kernel

This removes one kernel launch per MoE layer with no change in matmul kernel
shapes. The measured end to end decode impact is within run to run noise (bs=1,
K=12 interleaved trials on Qwen3-30B-A3B-4bit: on minus off 0.4%, a quarter of
the noise band), so v1 lands off by default as a correct fusion substrate, not
a measured speedup.

Scope of v1
-----------
Targets the configuration shared by Qwen3-30B-A3B-4bit and Qwen1.5-MoE-A2.7B-4bit:
- bits=4, mode='affine', group_size=64
- K (input_dim) divisible by 512  (Qwen3: 2048, Qwen1.5: 2048)
- N (output_dim) divisible by 8   (Qwen3: 768, Qwen1.5: 1408)
- Scales/biases dtype matches the input dtype (bf16 or fp16)

Anything outside that falls back to the unfused mlx_lm path.

When the fast `gather_qmv` from #22283 lands, the qmv inner loop here can be
replaced by a call into it; the epilogue (silu * x_up) doesn't change.
"""

from __future__ import annotations

import logging
import weakref

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Constants matching MLX's affine_qmv_fast for bits=4, group_size=64.
# Lifted directly from mlx/include/mlx/backend/metal/kernels/quantized.h
# (qmv_fast_impl), so the inner-loop layout matches MLX's own gather_qmm in
# that regime.
_BITS = 4
_GROUP_SIZE = 64
_SIMD_SIZE = 32
_PACK_FACTOR = 8  # 32 / bits
_BYTES_PER_PACK = 4  # sizeof(uint32_t)
_PACKS_PER_THREAD = 2
_NUM_SIMDGROUPS = 2
_RESULTS_PER_SIMDGROUP = 4
_VALUES_PER_THREAD = _PACK_FACTOR * _PACKS_PER_THREAD  # 16
_BLOCK_SIZE = _VALUES_PER_THREAD * _SIMD_SIZE  # 512
_ROWS_PER_TG = _NUM_SIMDGROUPS * _RESULTS_PER_SIMDGROUP  # 8


# Metal source for the fused kernel.
# Body only — mx.fast.metal_kernel auto-generates the kernel signature
# based on input_names / output_names and the template params below.
_KERNEL_SOURCE = r"""
    // Mirrors qmv_fast_impl<T, group_size=64, bits=4> from MLX's quantized.h
    // with a silu(result) * x_up write epilogue.
    //
    // Inputs:
    //   x       [M_tok, K]                — pre-gather activations (T)
    //   w       [E, N, K * 4 / 32]        — packed 4-bit weights (uint32)
    //   s       [E, N, K / GROUP_SIZE]    — affine scales (T)
    //   b       [E, N, K / GROUP_SIZE]    — affine biases (T)
    //   idx     [M_tok * TOPK]            — expert per (token, topk) pair (uint32)
    //   x_up    [M_tok * TOPK, N]         — precomputed up output (T)
    // Output:
    //   y       [M_tok * TOPK, N]         — silu(gate_qmv(x)) * x_up

    constexpr int BITS = 4;
    constexpr int GROUP_SIZE = 64;
    constexpr int SIMD_SIZE = 32;
    constexpr int PACK_FACTOR = 8;
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACKS_PER_THREAD = 2;
    constexpr int NUM_SIMDGROUPS = 2;
    constexpr int RESULTS_PER_SIMDGROUP = 4;
    constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;   // 16
    constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;            // 512
    constexpr int SCALE_STEP_PER_THREAD = GROUP_SIZE / VALUES_PER_THREAD; // 4

    // Compile-time problem dims (template params)
    constexpr int K = IN_VEC_SIZE;
    constexpr int N = OUT_VEC_SIZE;
    constexpr int TOPK = TOP_K;

    constexpr int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;    // K/2 bytes per row
    constexpr int in_vec_size_g = K / GROUP_SIZE;                       // groups per row

    // tid.x = (token, topk) pair index in M_tok*TOPK
    // tid.y = output row block index (each block writes 8 rows)
    uint mt = threadgroup_position_in_grid.x;
    uint out_row_block = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    uint m = mt / TOPK;                      // index into pre-gather x
    uint e = idx[mt];                        // expert id

    // Base pointers for this (expert) row block.
    // Weights / scales / biases live in [E, N, ...] tensors.
    const device uint8_t* ws = (const device uint8_t*)(w
        + uint64_t(e) * uint64_t(N) * uint64_t(in_vec_size_w / BYTES_PER_PACK));
    const device T* scales_p = s
        + uint64_t(e) * uint64_t(N) * uint64_t(in_vec_size_g);
    const device T* biases_p = b
        + uint64_t(e) * uint64_t(N) * uint64_t(in_vec_size_g);

    // out_row indexes the first of RESULTS_PER_SIMDGROUP rows this simdgroup handles.
    int out_row = out_row_block * NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP
                + simd_gid * RESULTS_PER_SIMDGROUP;

    ws += out_row * in_vec_size_w + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
    scales_p += out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;
    biases_p += out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;

    // Pre-gather x: shape [M_tok, K]
    const device T* x_p = x + uint64_t(m) * uint64_t(K)
                            + uint64_t(simd_lid) * uint64_t(VALUES_PER_THREAD);

    float result[RESULTS_PER_SIMDGROUP] = {0, 0, 0, 0};
    thread float x_thread[VALUES_PER_THREAD];

    // Outer loop over K in BLOCK_SIZE-wide chunks.
    // K % BLOCK_SIZE == 0 is required (checked in Python wrapper).
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // --- load_vector for bits=4 ---
        float sum = 0;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            float a0 = float(x_p[i]);
            float a1 = float(x_p[i + 1]);
            float a2 = float(x_p[i + 2]);
            float a3 = float(x_p[i + 3]);
            sum += a0 + a1 + a2 + a3;
            x_thread[i]     = a0;
            x_thread[i + 1] = a1 / 16.0f;
            x_thread[i + 2] = a2 / 256.0f;
            x_thread[i + 3] = a3 / 4096.0f;
        }

        // For each of the 4 output rows this simdgroup is responsible for...
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
            const device uint16_t* ws_u16 =
                (const device uint16_t*)(ws + row * in_vec_size_w);
            float scale_v = float(scales_p[row * in_vec_size_g]);
            float bias_v  = float(biases_p[row * in_vec_size_g]);

            // --- qdot for bits=4, values_per_thread=16 ---
            float accum = 0;
            for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
                uint16_t packed = ws_u16[i];
                accum += (x_thread[4 * i]     * float(packed & 0x000f) +
                          x_thread[4 * i + 1] * float(packed & 0x00f0) +
                          x_thread[4 * i + 2] * float(packed & 0x0f00) +
                          x_thread[4 * i + 3] * float(packed & 0xf000));
            }
            result[row] += scale_v * accum + sum * bias_v;
        }

        ws        += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;  // += 256 bytes
        scales_p  += BLOCK_SIZE / GROUP_SIZE;                    // += 8 groups
        biases_p  += BLOCK_SIZE / GROUP_SIZE;
        x_p       += BLOCK_SIZE;
    }

    // Write epilogue: simd-sum across lanes, then silu(gate) * x_up.
    device T* y_p = y + uint64_t(mt) * uint64_t(N) + uint64_t(out_row);
    const device T* x_up_p = x_up + uint64_t(mt) * uint64_t(N) + uint64_t(out_row);

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        float gate_v = simd_sum(result[row]);
        if (simd_lid == 0) {
            // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float silu_v = gate_v / (1.0f + metal::precise::exp(-gate_v));
            y_p[row] = T(silu_v * float(x_up_p[row]));
        }
    }
"""


# Build kernel lazily so import-time on non-MLX systems doesn't fail.
_kernel_cache: dict = {}

# AOT pre-compile cache: tracks which (dtype, K, N, T) tuples have been warmed.
# MLX specializes the Metal kernel on template args at first dispatch, so the
# first call per unique tuple pays a ~3ms compile cost. Pre-warming at patch
# time moves that cost out of the first forward pass.
_aot_warmed: set = set()


def _get_kernel(dtype: mx.Dtype):
    """Return a compiled mx.fast.metal_kernel for the given input dtype.

    Kept per-dtype because mx.fast.metal_kernel specializes on template args
    at first call, and we want clean separation between fp16 / bf16 variants.
    """
    if dtype not in _kernel_cache:
        # Strip the `mlx.core.` prefix and any dots from the dtype repr so the
        # kernel name is a valid C identifier (Metal's host_name attribute and
        # function-name slot don't accept '.').
        dtype_tag = str(dtype).replace("mlx.core.", "").replace(".", "_")
        _kernel_cache[dtype] = mx.fast.metal_kernel(
            name=f"affine_gather_qmv_silu_mul_4bit_gs64_{dtype_tag}",
            input_names=["x", "w", "s", "b", "idx", "x_up"],
            output_names=["y"],
            source=_KERNEL_SOURCE,
        )
    return _kernel_cache[dtype]


def fused_gate_qmv_silu_mul(
    x: mx.array,
    gate_w: mx.array,
    gate_s: mx.array,
    gate_b: mx.array,
    indices: mx.array,
    x_up: mx.array,
) -> mx.array:
    """Compute ``silu(gather_qmm(x, W_gate)) * x_up`` in one kernel.

    Shapes:
        x       : (..., 1, 1, K)               input activations (pre-gather)
        gate_w  : (E, N, K // PACK_FACTOR)     packed 4-bit weights
        gate_s  : (E, N, K // GROUP_SIZE)      affine scales
        gate_b  : (E, N, K // GROUP_SIZE)      affine biases
        indices : (..., T)                     expert per (token, topk) pair
        x_up    : (..., T, 1, N)               pre-computed up output
        y       : (..., T, 1, N)               returned

    Numerical contract: equivalent within floating point tolerance to::

        x_gate = mx.gather_qmm(x, gate_w, gate_s, gate_b, rhs_indices=indices,
                               transpose=True, group_size=GROUP_SIZE, bits=BITS,
                               mode='affine')
        y = nn.silu(x_gate) * x_up

    up to floating-point ordering of accumulations (which matches MLX's own
    qmv_fast_impl exactly).
    """
    # Validate the regime this kernel supports. Outside it, the caller should
    # fall back to the unfused MLX path.
    K = gate_s.shape[-1] * _GROUP_SIZE
    N = gate_w.shape[-2]
    if K % _BLOCK_SIZE != 0:
        raise ValueError(
            f"fused_gate_qmv_silu_mul: K={K} not divisible by {_BLOCK_SIZE}. "
            f"Use the unfused path."
        )
    if N % _ROWS_PER_TG != 0:
        raise ValueError(
            f"fused_gate_qmv_silu_mul: N={N} not divisible by {_ROWS_PER_TG}."
        )
    # Sanity: scales/biases dtype must match x dtype for in-kernel float() conversion.
    if gate_s.dtype != x.dtype or gate_b.dtype != x.dtype:
        raise ValueError(
            f"fused_gate_qmv_silu_mul: dtype mismatch x={x.dtype} "
            f"s={gate_s.dtype} b={gate_b.dtype}"
        )

    # Shape handling: x always has K as its last axis. M_tok is the number of
    # distinct pre-gather tokens (= x.size // K). T is the top_k axis carried
    # by `indices`. In the unsorted SwitchGLU path x has shape (B, 1, 1, K) and
    # indices is (B, T); in the sorted path x has shape (B*T, 1, K) and
    # indices is (B*T, 1) after our reshape. Either way: M_tok * T == idx.size
    # and M_tok * K == x.size.
    assert x.shape[-1] == K, f"x last dim {x.shape[-1]} != K={K}"
    M_tok = x.size // K
    T = indices.shape[-1]
    assert (
        M_tok * T == indices.size
    ), f"M_tok({M_tok}) * T({T}) != indices.size({indices.size})"
    x_flat = x.reshape(M_tok, K)
    idx_flat = indices.reshape(M_tok * T)
    if idx_flat.dtype != mx.uint32:
        idx_flat = idx_flat.astype(mx.uint32)

    # x_up has N as its last axis and total size M_tok * T * N. The singleton
    # rank dims (1 or 2 of them) get folded away by reshape.
    assert (
        x_up.shape[-1] == N and x_up.size == M_tok * T * N
    ), f"x_up shape {x_up.shape} does not match M_tok({M_tok})*T({T})*N({N})"
    x_up_flat = x_up.reshape(M_tok * T, N)

    kernel = _get_kernel(x.dtype)
    (y_flat,) = kernel(
        inputs=[x_flat, gate_w, gate_s, gate_b, idx_flat, x_up_flat],
        template=[
            ("T", x.dtype),
            ("IN_VEC_SIZE", K),
            ("OUT_VEC_SIZE", N),
            ("TOP_K", T),
        ],
        # grid is in *threads*, not threadgroups: total threads = product of
        # (grid_x, grid_y, grid_z). One threadgroup processes one (mt, row_block).
        # Threadgroup is 64 = 2 simdgroups × 32 lanes.
        grid=(M_tok * T * 64, N // _ROWS_PER_TG, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(M_tok * T, N)],
        output_dtypes=[x.dtype],
    )

    # Reshape to x_up's shape, which is exactly what self.activation(x_up,
    # x_gate) would have returned (silu*mul is shape-preserving).
    return y_flat.reshape(x_up.shape)


def _aot_warm_kernel(switch_mlp, top_k: int) -> None:
    """Pre-compile the fused kernel for the (dtype, K, N, T) tuples this
    layer will dispatch at runtime.

    MLX's mx.fast.metal_kernel specializes Metal source on template args
    at first dispatch (one Metal compile per unique tuple, ~3ms each).
    Issuing one dummy dispatch per shape moves that compile out of the
    first forward pass and into model init. The module-level _aot_warmed
    set means only the first layer per shape actually compiles; the
    remaining 47 hit the cache and no-op.

    Warms both the unsorted decode shape (T=top_k) and the sorted
    large-batch shape (T=1) used by the gather-sort path.
    """
    gate = switch_mlp.gate_proj
    K = gate.scales.shape[-1] * _GROUP_SIZE
    N = gate.weight.shape[-2]
    dtype = gate.scales.dtype
    for T in (top_k, 1):
        key = (dtype, K, N, T)
        if key in _aot_warmed:
            continue
        M_tok = 1
        x_dummy = mx.zeros((M_tok, 1, 1, K), dtype=dtype)
        idx_dummy = mx.zeros((M_tok, T), dtype=mx.uint32)
        x_up_dummy = mx.zeros((M_tok, T, 1, N), dtype=dtype)
        out = fused_gate_qmv_silu_mul(
            x_dummy,
            gate["weight"],
            gate["scales"],
            gate.get("biases"),
            idx_dummy,
            x_up_dummy,
        )
        mx.eval(out)
        _aot_warmed.add(key)
        logger.info(
            "Path B AOT: warmed fused kernel dtype=%s K=%d N=%d T=%d",
            dtype,
            K,
            N,
            T,
        )


def can_fuse(switch_mlp) -> bool:
    """Cheap structural check: does this SwitchGLU match the Path B v1 regime?"""
    try:
        from mlx_lm.models.switch_layers import (
            QuantizedSwitchLinear,
            SwiGLU,
            SwitchGLU,
        )
    except ImportError:
        return False
    up = switch_mlp.up_proj
    gate = switch_mlp.gate_proj
    if not isinstance(up, QuantizedSwitchLinear) or not isinstance(
        gate, QuantizedSwitchLinear
    ):
        return False
    # A model that overrides SwitchGLU.__call__ runs a custom forward, but the
    # patch installs a subclass __call__ that imposes the stock semantics
    # fused_forward reimplements, silently bypassing the override. Decline when
    # the forward is not the stock SwitchGLU.__call__ (evaluated at patch time,
    # before the class swap, so this sees the model's real class).
    if type(switch_mlp).__call__ is not SwitchGLU.__call__:
        return False
    # fused_forward bakes silu into both the kernel and its fallback, and a
    # swapped activation= leaves __call__ stock, so the check above cannot see
    # it. Exact type, fail closed: a SwiGLU subclass may change the math.
    if type(getattr(switch_mlp, "activation", None)) is not SwiGLU:
        return False
    # Learned per-expert bias, added after the matmul in
    # QuantizedSwitchLinear.__call__ as ``x + bias[indices]`` whenever
    # ``"bias" in self``. This is the affine learned bias, distinct from the
    # quant ``biases`` (zero-points) the kernel already consumes. The fused
    # kernel recomputes the gate matmul in-register and has no slot for the
    # learned bias, so a gate carrying one would silently drop it. up_proj runs
    # its normal path (its bias, if any, is already in x_up), so only the gate
    # is at risk: fall back to the unfused path when the gate has a learned bias.
    if "bias" in gate:
        return False
    if up.bits != 4 or up.group_size != 64 or up.mode != "affine":
        return False
    if gate.bits != 4 or gate.group_size != 64 or gate.mode != "affine":
        return False
    if up.biases is None or gate.biases is None:
        return False
    # The kernel reads gate scales/biases as the activation dtype (in-kernel
    # float() then writes back as the activation dtype), so the runtime
    # precondition in fused_gate_qmv_silu_mul requires a single shared float
    # dtype across the quant params. Check it here so a mismatch falls through
    # to the unfused path instead of raising at forward. The packed weight is
    # uint32 and intentionally excluded from the dtype contract.
    param_dtype = gate.scales.dtype
    if (
        gate.biases.dtype != param_dtype
        or up.scales.dtype != param_dtype
        or up.biases.dtype != param_dtype
    ):
        return False
    # Validate from the gate projection: fused_gate_qmv_silu_mul recomputes the
    # gate matmul and keys K/N off gate dims, so the gate is the operand that
    # must satisfy the tiling constraints (equivalent to up only while
    # up.shape == gate.shape).
    K = gate.scales.shape[-1] * _GROUP_SIZE
    N = gate.weight.shape[-2]
    if K % _BLOCK_SIZE != 0 or N % _ROWS_PER_TG != 0:
        return False
    return True


_fallback_warned = False


def _fused_gate_or_fallback(gate_proj, x, idx, x_up, sorted_indices=False):
    """silu(gate_qmv(x)) * x_up via the fused kernel; on ValueError fall back to
    the unfused gate projection. gather_qmm tolerates the activation dtype the
    fused kernel rejects, which can_fuse cannot pre-check at patch time. Warns
    once.
    """
    # Kernel layout only: the Metal kernel reads T from indices.shape[-1], and
    # _gather_sort folded top_k into M_tok, so the sorted path needs an explicit
    # T=1 axis. The fallback must not see it: gather_qmm broadcasts an (M_tok, 1)
    # index against sorted x's (M_tok,) batch dim into an M_tok x M_tok cross
    # product.
    gate_idx = idx.reshape(-1, 1) if sorted_indices else idx
    gw = gate_proj["weight"]
    gs = gate_proj["scales"]
    gb = gate_proj.get("biases")
    try:
        return fused_gate_qmv_silu_mul(x, gw, gs, gb, gate_idx, x_up)
    except ValueError as e:
        global _fallback_warned
        if not _fallback_warned:
            logger.warning(
                "Path B: fused gate kernel declined inputs (%s); using the "
                "unfused gate path for this and matching calls.",
                e,
            )
            _fallback_warned = True
        # Reference expression by construction: the same projection call with
        # the same flat idx up_proj/down_proj receive.
        return nn.silu(gate_proj(x, idx, sorted_indices=sorted_indices)) * x_up


class FusedSwitchSwiGLU(nn.Module):
    """SwitchGLU forward with Path B fusion installed.

    Wraps an existing SwitchGLU instance. Reads up_proj / gate_proj weights
    directly (no concatenation), runs up_proj as usual, then calls the fused
    kernel for ``silu(gate_qmv(x)) * x_up`` in one shot.

    Replaces ``switch_mlp.__call__`` via patch_switch_glu_with_fused_swiglu;
    SwitchGLU.up_proj / gate_proj / activation are *not* replaced and remain
    available for fallback paths (e.g. sorted-indices large-batch case).
    """

    def __init__(self, switch_mlp):
        super().__init__()
        # Weak ref: sw stores the bound fused_forward, so a strong ref here would
        # cycle (sw -> method -> self -> sw). sw outlives every call into it.
        self._switch_mlp = weakref.proxy(switch_mlp)

    def fused_forward(self, x, indices):
        """Same contract as SwitchGLU.__call__ but with fused activation."""
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        sw = self._switch_mlp
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x_up = sw.up_proj(x, idx, sorted_indices=do_sort)

        swiglu = _fused_gate_or_fallback(
            sw.gate_proj, x, idx, x_up, sorted_indices=do_sort
        )

        out = sw.down_proj(swiglu, idx, sorted_indices=do_sort)

        if do_sort:
            out = _scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)


def patch_switch_glu_with_fused_swiglu(model) -> int:
    """Install Path B on every eligible SwitchGLU in the model.

    Replaces ``switch_mlp.__call__`` with FusedSwitchSwiGLU.fused_forward.
    Leaves ``up_proj``, ``gate_proj``, ``activation`` in place so the original
    code path is still reachable for sorted-large-batch (handled internally) and
    for any callers that bypass the patched __call__.

    Returns number of layers patched.
    """
    from mlx_lm.models.switch_layers import SwitchGLU

    patched = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        sw = getattr(mlp, "switch_mlp", None)
        if not isinstance(sw, SwitchGLU):
            continue
        if not can_fuse(sw):
            continue
        # Idempotent: skip if already patched.
        if getattr(sw, "_path_b_installed", False):
            continue
        # AOT: pre-compile the fused kernel for this layer's shapes. top_k
        # lives on the parent MoE block (e.g. Qwen3MoeSparseMoeBlock.top_k).
        # If absent, skip warming and fall back to lazy JIT on first dispatch.
        top_k = getattr(mlp, "top_k", None)
        if top_k is not None:
            _aot_warm_kernel(sw, int(top_k))
        # One-off SwitchGLU subclass rather than rewriting up_proj/gate_proj: the
        # activation fusion folds silu(gate)*x_up into the gate matmul, which has
        # to intercept the forward (the projection level can't express it).
        # can_fuse declines a non-stock __call__, so a customized forward falls
        # back unpatched. Python resolves __call__ on the type, so swap
        # sw.__class__ to a subclass; cache it on the exact class (cls.__dict__,
        # not hasattr which walks the MRO) so a SwitchGLU subclass gets its own
        # entry instead of being downcast to the base.
        sw._path_b_call = FusedSwitchSwiGLU(sw).fused_forward
        cls = type(sw)
        if "_PathBSubclass" not in cls.__dict__:
            cls._PathBSubclass = type(
                f"{cls.__name__}_PathB",
                (cls,),
                {"__call__": lambda self, *a, **kw: self._path_b_call(*a, **kw)},
            )
        sw.__class__ = cls._PathBSubclass
        sw._path_b_installed = True
        patched += 1

    if patched == 0:
        logger.warning(
            "patch_switch_glu_with_fused_swiglu: no eligible SwitchGLU found"
        )
    else:
        logger.info(f"patch_switch_glu_with_fused_swiglu: patched {patched} layers")
    return patched
