"""MUSA TileLang RMSNorm + gate kernels."""

import functools

import tilelang
import tilelang.language as T
import torch

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang.utils import (
    MUSA_COMMON_PASS_CONFIGS,
    MUSA_COMPILE_FLAGS,
    tilelang_dtype,
)
from sglang.srt.utils.custom_op import register_custom_op

__all__ = ["RMSNorm"]

_LOG2E = 1.4426950408889634

_RMSNORM_PASS_CONFIGS = dict(MUSA_COMMON_PASS_CONFIGS)
for _key, _value in (
    ("TL_DISABLE_DATA_RACE_CHECK", True),
    ("TL_DISABLE_SAFE_COPY_PREDICATION", True),
    ("TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION", True),
    ("TL_CONFIG_INDEX_BITWIDTH", 32),
):
    if hasattr(tilelang.PassConfigKey, _key):
        _RMSNORM_PASS_CONFIGS[getattr(tilelang.PassConfigKey, _key)] = _value


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_RMSNORM_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _rms_norm_gated_kernel(
    dtype: str,
    hidden_size: int,
    rows_per_block: int,
    lanes_per_row: int,
):
    m = T.dynamic("m")
    x_stride_row = T.dynamic("x_stride_row")
    y_stride_row = T.dynamic("y_stride_row")
    z_stride_row = T.dynamic("z_stride_row")
    vec_size = tilelang.cdiv(hidden_size, lanes_per_row)
    num_shuffles = lanes_per_row.bit_length() - 1
    threads = rows_per_block * lanes_per_row

    @T.prim_func
    def sglang_musa_rms_norm_gated(
        x: T.StridedTensor((m, hidden_size), (x_stride_row, 1), dtype),
        y: T.StridedTensor((m, hidden_size), (y_stride_row, 1), dtype),
        w: T.Tensor((hidden_size,), dtype),
        z: T.StridedTensor((m, hidden_size), (z_stride_row, 1), dtype),
        rstd: T.Tensor((m,), "float32"),
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(m, rows_per_block), threads=threads) as (bid,):
            tid = T.get_thread_binding()
            lane = tid % lanes_per_row
            row_in_block = tid // lanes_per_row
            row = bid * rows_per_block + row_in_block
            offset = T.alloc_var("int32")
            x_local = T.alloc_local((vec_size,), "float32")
            z_local = T.alloc_local((vec_size,), "float32")
            sum_local = T.alloc_local((1,), "float32")
            inv_rms = T.alloc_local((1,), "float32")

            if row < m:
                sum_local[0] = 0.0
                for i in T.vectorized(vec_size):
                    offset = lane * vec_size + i
                    if offset < hidden_size:
                        x_local[i] = T.cast(x[row, offset], "float32")
                        sum_local[0] += x_local[i] * x_local[i]

                for i in T.unroll(num_shuffles):
                    sum_local[0] += T.shfl_xor(sum_local[0], (lanes_per_row // 2) >> i)

                inv_rms[0] = T.rsqrt(sum_local[0] / hidden_size + eps)
                if lane == 0:
                    rstd[row] = inv_rms[0]

                for i in T.vectorized(vec_size):
                    offset = lane * vec_size + i
                    if offset < hidden_size:
                        z_local[i] = T.cast(z[row, offset], "float32")
                        y[row, offset] = T.cast(
                            x_local[i]
                            * inv_rms[0]
                            * T.cast(w[offset], "float32")
                            * z_local[i]
                            / (1.0 + T.exp2(-z_local[i] * _LOG2E)),
                            dtype,
                        )

    return sglang_musa_rms_norm_gated


_rms_norm_gated_kernel.mode = "lazy"


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    target="musa",
    pass_configs=_RMSNORM_PASS_CONFIGS,
    compile_flags=MUSA_COMPILE_FLAGS,
)
def _rms_norm_gated_kernel_cta(
    dtype: str,
    hidden_size: int,
    threads: int,
):
    m = T.dynamic("m")
    x_stride_row = T.dynamic("x_stride_row")
    y_stride_row = T.dynamic("y_stride_row")
    z_stride_row = T.dynamic("z_stride_row")
    warps_per_cta = threads // 32

    @T.prim_func
    def sglang_musa_rms_norm_gated_cta(
        x: T.StridedTensor((m, hidden_size), (x_stride_row, 1), dtype),
        y: T.StridedTensor((m, hidden_size), (y_stride_row, 1), dtype),
        w: T.Tensor((hidden_size,), dtype),
        z: T.StridedTensor((m, hidden_size), (z_stride_row, 1), dtype),
        rstd: T.Tensor((m,), "float32"),
        eps: T.float32,
    ):
        with T.Kernel(m, threads=threads) as (row,):
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            sum_local = T.alloc_var("float32")
            inv_rms = T.alloc_var("float32")
            warp_sum = T.alloc_shared((warps_per_cta,), "float32")
            inv_rms_shared = T.alloc_shared((1,), "float32")

            sum_local = 0.0
            for offset_base in T.serial(0, hidden_size, threads):
                offset = offset_base + tid
                if offset < hidden_size:
                    x_val = T.cast(x[row, offset], "float32")
                    sum_local += x_val * x_val

            sum_local = T.warp_reduce_sum(sum_local)
            if lane == 0:
                warp_sum[warp] = sum_local
            T.sync_threads()

            sum_local = T.if_then_else(tid < warps_per_cta, warp_sum[tid], 0.0)
            if warp == 0:
                sum_local = T.warp_reduce_sum(sum_local)
                if lane == 0:
                    inv_rms = T.rsqrt(sum_local / hidden_size + eps)
                    inv_rms_shared[0] = inv_rms
                    rstd[row] = inv_rms
            T.sync_threads()

            for offset_base in T.serial(0, hidden_size, threads):
                offset = offset_base + tid
                if offset < hidden_size:
                    x_val = T.cast(x[row, offset], "float32")
                    z_val = T.cast(z[row, offset], "float32")
                    y[row, offset] = T.cast(
                        x_val
                        * inv_rms_shared[0]
                        * T.cast(w[offset], "float32")
                        * z_val
                        / (1.0 + T.exp2(-z_val * _LOG2E)),
                        dtype,
                    )

    return sglang_musa_rms_norm_gated_cta


_rms_norm_gated_kernel_cta.mode = "lazy"


def _launch_rms_norm_gated(
    x: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    rows_per_block: int = 8,
    lanes_per_row: int = 16,
    cta_threads: int | None = None,
) -> None:
    if cta_threads is not None:
        kernel = _rms_norm_gated_kernel_cta(
            tilelang_dtype(x.dtype),
            x.shape[-1],
            cta_threads,
        )
        kernel(x, out, weight, z, rstd, float(eps))
        return

    kernel = _rms_norm_gated_kernel(
        tilelang_dtype(x.dtype),
        x.shape[-1],
        rows_per_block,
        lanes_per_row,
    )
    kernel(x, out, weight, z, rstd, float(eps))


@register_custom_op(
    op_name="musa_rms_norm_gated",
    mutates_args=["out", "rstd"],
)
def _rms_norm_gated_custom(
    x: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    rows_per_block: int = 8,
    lanes_per_row: int = 16,
    cta_threads: int | None = None,
) -> None:
    # Keep the TileLang executable launch opaque to Dynamo.
    _launch_rms_norm_gated(
        x,
        out,
        weight,
        z,
        rstd,
        eps,
        rows_per_block,
        lanes_per_row,
        cta_threads,
    )


def _rms_norm_gated_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    rows_per_block: int | None = None,
    lanes_per_row: int | None = None,
    cta_threads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2 or z.dim() != 2:
        raise RuntimeError("rms_norm_gated expects x/z with shape [M, N].")
    if x.shape != z.shape:
        raise RuntimeError("rms_norm_gated expects x and z to have the same shape.")
    if weight.shape != (x.shape[-1],):
        raise RuntimeError("rms_norm_gated weight shape mismatch.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("rms_norm_gated expects fp16 or bf16 input.")
    if z.dtype != x.dtype or weight.dtype != x.dtype:
        raise RuntimeError("rms_norm_gated expects x/z/weight to have the same dtype.")
    if x.stride(-1) != 1 or z.stride(-1) != 1:
        raise RuntimeError("rms_norm_gated requires x/z contiguous in the last dim.")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.stride(-1) != 1:
        raise RuntimeError(
            "rms_norm_gated output must match x and be last-dim contiguous."
        )

    hidden_size = x.shape[-1]
    if cta_threads is None and hidden_size >= 4096:
        cta_threads = 512
    if cta_threads is not None:
        if cta_threads not in (128, 256, 512):
            raise RuntimeError("cta_threads must be one of 128, 256, 512.")
        if cta_threads > 1024:
            raise RuntimeError("invalid cta_threads.")

    if lanes_per_row is None:
        lanes_per_row = 32 if hidden_size % 32 == 0 else 16
    if rows_per_block is None:
        rows_per_block = 4 if hidden_size >= 8192 and lanes_per_row == 32 else 8
    if hidden_size % lanes_per_row != 0:
        raise RuntimeError("hidden_size must be divisible by lanes_per_row.")
    if lanes_per_row not in (4, 8, 16, 32):
        raise RuntimeError("lanes_per_row must be one of 4, 8, 16, 32.")
    if rows_per_block < 1 or rows_per_block * lanes_per_row > 1024:
        raise RuntimeError("invalid rows_per_block/lanes_per_row combination.")

    rstd = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    _rms_norm_gated_custom(
        x,
        out,
        weight,
        z,
        rstd,
        eps,
        rows_per_block,
        lanes_per_row,
        cta_threads,
    )
    return out, rstd


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    if (
        z is not None
        and bias is None
        and group_size in (None, x.shape[-1])
        and norm_before_gate
        and is_rms_norm
        and activation in ("swish", "silu")
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dim() == 2
        and z.dim() == 2
        and x.stride(-1) == 1
        and z.stride(-1) == 1
        and weight.stride(-1) == 1
    ):
        out, rstd = _rms_norm_gated_impl(x, weight, z, eps, out=out)
        return out, None, rstd

    from sglang.srt.layers.attention.fla.layernorm_gated import (
        _layer_norm_fwd as _triton_layer_norm_fwd,
    )

    return _triton_layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )


def _fallback_layernorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    from sglang.srt.layers.attention.fla.layernorm_gated import layernorm_fn

    return layernorm_fn(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )


def layernorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    x_shape_og = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    z_2d = None
    if z is not None:
        if z.shape != x_shape_og:
            raise RuntimeError("z shape must match x shape")
        z_2d = z.reshape(-1, z.shape[-1])

    if (
        z_2d is not None
        and bias is None
        and group_size in (None, x_2d.shape[-1])
        and norm_before_gate
        and is_rms_norm
        and activation in ("swish", "silu")
        and x_2d.dtype in (torch.float16, torch.bfloat16)
        and x_2d.stride(-1) == 1
        and z_2d.stride(-1) == 1
        and weight.stride(-1) == 1
    ):
        y, _, _ = _layer_norm_fwd(
            x_2d,
            weight,
            bias,
            eps,
            z=z_2d,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
            activation=activation,
        )
        return y.reshape(x_shape_og)

    return _fallback_layernorm_fn(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )


def rms_norm_gated(
    *,
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        if z.shape != x_shape_og:
            raise RuntimeError("z shape must match x shape")
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y, _, _ = _layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )
    return y.reshape(x_shape_og)


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
        activation: str = "swish",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        if (
            z is not None
            and self.bias is None
            and self.group_size in (None, x.shape[-1])
            and self.norm_before_gate
            and self.activation in ("swish", "silu")
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            x_shape_og = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            z_2d = z.reshape(-1, z.shape[-1])
            if x_2d.stride(-1) == 1 and z_2d.stride(-1) == 1:
                y, _ = _rms_norm_gated_impl(
                    x_2d,
                    self.weight,
                    z_2d,
                    self.eps,
                )
                return y.reshape(x_shape_og)
        return layernorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True,
            activation=self.activation,
        )
