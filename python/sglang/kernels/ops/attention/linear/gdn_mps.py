"""Gated DeltaNet (GDN) ops for Apple MPS backed by Metal compute shaders.

``gdn_mps.metal`` is compiled once per dtype/geometry variant with
``torch.mps.compile_shader`` and dispatched on torch MPS tensors. Shared
conventions:

- Cache indices and query start locations are int32; initial-state flags are
  bool. A negative cache index marks a padding row: its output is zero and no
  state slot is read or written.
- State pools are indexed by slot along dim 0 and may be per-layer views of a
  larger pool, so slot/head/row strides are passed to the kernels.
- Conv state is ``[S, C, K-1]`` (oldest input first); SSM state is
  ``[S, Hv, Dv, Dk]`` in fp32.
"""

import functools
from pathlib import Path
from typing import Optional

import torch

_METAL_PATH = Path(__file__).with_name("gdn_mps.metal")
_MSL_DTYPES = {
    torch.bfloat16: "bfloat",
    torch.float16: "half",
    torch.float32: "float",
}
_SIMD_LANES = 32
_SG_PER_TG = 4
_CONV_GROUP_SIZE = 256
# The recurrence keeps 32 state columns per lane, so Dk must be 32 * 2^m.
_SUPPORTED_DK = (64, 128, 256)


@functools.cache
def _metal_src() -> str:
    return _METAL_PATH.read_text(encoding="utf-8")


def _msl(dtype: torch.dtype) -> str:
    if dtype not in _MSL_DTYPES:
        raise ValueError(f"GDN MPS kernels do not support dtype {dtype}")
    return _MSL_DTYPES[dtype]


@functools.cache
def _conv_lib(
    in_dtype: torch.dtype,
    state_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    width: int,
    has_bias: bool,
):
    defs = (
        f"#define KERNEL_CONV 1\n#define IN_T {_msl(in_dtype)}\n"
        f"#define ST_T {_msl(state_dtype)}\n#define W_T {_msl(weight_dtype)}\n"
        f"#define K {width}\n#define HAS_BIAS {int(has_bias)}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


@functools.cache
def _recurrent_lib(
    in_dtype: torch.dtype,
    gate_dtype: torch.dtype,
    a_log_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    head_k_dim: int,
):
    defs = (
        f"#define KERNEL_RECURRENT 1\n#define IN_T {_msl(in_dtype)}\n"
        f"#define GATE_T {_msl(gate_dtype)}\n#define AL_T {_msl(a_log_dtype)}\n"
        f"#define DT_T {_msl(dt_bias_dtype)}\n#define DK {head_k_dim}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


def _i32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int32).contiguous()


def _flags(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.bool:
        raise ValueError(f"has_initial_state must be bool, got {t.dtype}")
    return t.contiguous().view(torch.uint8)


def _conv_group(threads: int) -> int:
    return min(_CONV_GROUP_SIZE, threads)


def _conv_args(x, conv_states, weight, bias, width):
    if not 2 <= width <= 4:
        raise ValueError(f"conv width {width} outside the supported 2..4")
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is None:
        return x, weight, weight, False
    if bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return x, weight, bias.contiguous(), True


@torch.no_grad()
def causal_conv1d_update_mps(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """One decode step of the causal conv with SiLU; shifts slot windows in place.

    Coverage: x, states and weight/bias each bf16, fp16 or fp32 (independent),
    K in 2..4, strided x rows and pool views. Current x is not rounded to the
    state dtype before the conv; only the stored history is.
    """
    batch, channels = x.shape
    width = weight.shape[-1]
    x, weight, bias, has_bias = _conv_args(x, conv_states, weight, bias, width)
    lib = _conv_lib(x.dtype, conv_states.dtype, weight.dtype, width, has_bias)
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if batch * channels == 0:
        return y
    lib.gdn_conv1d_update(
        x,
        conv_states,
        weight,
        bias,
        _i32(cache_indices),
        y,
        x.stride(0),
        *conv_states.stride(),
        channels,
        threads=batch * channels,
        group_size=_conv_group(batch * channels),
    )
    return y


@torch.no_grad()
def causal_conv1d_varlen_mps(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Varlen prefill conv with SiLU; saves each sequence's last K-1 inputs.

    Same coverage as ``causal_conv1d_update_mps``; the final window is written
    by a second dispatch so no thread reads a window another one updates.
    """
    num_tokens, channels = x.shape
    width = weight.shape[-1]
    x, weight, bias, has_bias = _conv_args(x, conv_states, weight, bias, width)
    lib = _conv_lib(x.dtype, conv_states.dtype, weight.dtype, width, has_bias)
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if num_tokens == 0 or cache_indices.numel() == 0:
        return y.zero_()
    slots = _i32(cache_indices)
    qsl = _i32(query_start_loc)
    initial = _flags(has_initial_state)
    strides = (x.stride(0), *conv_states.stride())
    lib.gdn_conv1d_varlen(
        x,
        conv_states,
        weight,
        bias,
        slots,
        qsl,
        initial,
        y,
        *strides,
        channels,
        slots.numel(),
        threads=num_tokens * channels,
        group_size=_conv_group(num_tokens * channels),
    )
    lib.gdn_conv1d_varlen_state(
        x,
        conv_states,
        slots,
        qsl,
        initial,
        *strides,
        channels,
        threads=slots.numel() * channels,
        group_size=_conv_group(slots.numel() * channels),
    )
    return y


def _check_recurrent(q, k, v, a, b, A_log, dt_bias, ssm_states):
    head_k_dim = q.shape[-1]
    if head_k_dim not in _SUPPORTED_DK:
        raise ValueError(f"head_k_dim {head_k_dim} not in {_SUPPORTED_DK}")
    if q.shape[1:] != k.shape[1:]:
        raise ValueError("q and k must have the same head count and head dim")
    if ssm_states.shape[1:3] != v.shape[1:] or ssm_states.shape[-1] != head_k_dim:
        raise ValueError("ssm_states must be [slots, Hv, Dv, Dk] matching v and k")
    if v.shape[1] % k.shape[1]:
        raise ValueError("num_v_heads must be a multiple of num_k_heads")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k and v must share a dtype")
    if a.dtype != b.dtype:
        raise ValueError("a and b must share a dtype")
    if ssm_states.dtype != torch.float32:
        raise ValueError("ssm_states must be fp32")
    # The kernels load q/k heads and state rows as 4-element vectors.
    for t in (q, k, ssm_states):
        if t.stride(-1) != 1 or any(s % 4 for s in t.stride()[:-1]):
            raise ValueError("q/k/ssm_states need 4-aligned strides")
        if t.storage_offset() % 4:
            raise ValueError("q/k/ssm_states need 4-aligned storage offsets")


@torch.no_grad()
def gated_delta_rule_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gated delta rule over varlen sequences with fused gating and q/k L2 norm.

    Decode is the case ``query_start_loc == arange(B + 1)``; the final fp32
    state of each sequence is written back to its slot in place. Coverage:
    q/k/v bf16, fp16 or fp32, Dk in {64, 128, 256}, any Dv (rows past Dv are
    masked), Hv a multiple of Hk, fp32 state, strided inputs and pool views.
    """
    _check_recurrent(q, k, v, a, b, A_log, dt_bias, ssm_states)
    head_k_dim = q.shape[-1]
    num_v_heads, head_v_dim = v.shape[1:]
    o = torch.empty(v.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0 or cache_indices.numel() == 0:
        return o.zero_()
    lib = _recurrent_lib(q.dtype, a.dtype, A_log.dtype, dt_bias.dtype, head_k_dim)
    rows_per_group = 1024 // head_k_dim
    blocks = -(-head_v_dim // rows_per_group)
    groups = cache_indices.numel() * num_v_heads * blocks
    group_count = -(-groups // _SG_PER_TG)
    v = v.contiguous() if v.stride(-1) != 1 else v
    a = a.contiguous()
    b = b.contiguous()
    num_tokens = q.shape[0]
    num_k_heads = k.shape[1]
    gate_shape = (num_tokens, num_v_heads)
    g = torch.empty(gate_shape, dtype=torch.float32, device=q.device)
    beta = torch.empty(gate_shape, dtype=torch.float32, device=q.device)
    norms = torch.empty(
        (num_tokens, num_k_heads, 2), dtype=torch.float32, device=q.device
    )
    prep_threads = num_tokens * (num_k_heads + num_v_heads)
    lib.gdn_prep(
        q,
        k,
        a,
        b,
        A_log.contiguous(),
        dt_bias.contiguous(),
        g,
        beta,
        norms,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        a.stride(0),
        b.stride(0),
        num_k_heads,
        num_v_heads,
        num_tokens * num_k_heads,
        threads=prep_threads,
        group_size=_conv_group(prep_threads),
    )
    lib.gdn_recurrent(
        q,
        k,
        v,
        g,
        beta,
        norms,
        ssm_states,
        _i32(cache_indices),
        _i32(query_start_loc),
        _flags(has_initial_state),
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        g.stride(0),
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        num_v_heads,
        num_k_heads,
        head_v_dim,
        cache_indices.numel(),
        threads=group_count * _SG_PER_TG * _SIMD_LANES,
        group_size=_SG_PER_TG * _SIMD_LANES,
    )
    return o
