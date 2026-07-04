# Benchmarks RMSNorm / fused_add_rmsnorm / LayerNorm implementations across
# providers (PyTorch, sgl_kernel, FlashInfer, SGLang JIT/Triton, and the
# optional external FlagGems / quack repos) on the captured diffusion
# workload shapes.
#
# FlagGems and quack are optional third-party providers; when they are not
# importable on the current box those cells are reported as N/A via
# marker.skip rather than crashing the sweep.
import os

import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.diffusion.triton.norm import norm_infer, rms_norm_fn
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="self-skips in CI, standalone tool",
)

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

EPS = 1e-6
DTYPE = torch.bfloat16

# (shape_id, op, input_shape) captured from real diffusion workloads.
ACTUAL_DIFFUSION_SHAPES = [
    ("qwen_ln_4096x3072", "layernorm", (1, 4096, 3072)),
    ("qwen_ln_26x3072", "layernorm", (1, 26, 3072)),
    ("qwen_rms_26x3584", "rmsnorm", (1, 26, 3584)),
    ("qwen_edit_ln_8308x3072", "layernorm", (1, 8308, 3072)),
    ("qwen_edit_rms_200x3584", "rmsnorm", (1, 200, 3584)),
    ("flux_ln_4096x3072", "layernorm", (1, 4096, 3072)),
    ("flux_rms_512x4096", "rmsnorm", (1, 512, 4096)),
    ("flux2_ln_4608x6144", "layernorm", (1, 4608, 6144)),
    ("flux2_rms_4608x48x128", "rmsnorm", (1, 4608, 48, 128)),
    ("zimage_ln_4128x3840", "layernorm", (1, 4128, 3840)),
    ("zimage_rms_4096x3840", "rmsnorm", (1, 4096, 3840)),
    ("wan_ti2v_rms_512x4096", "rmsnorm", (1, 512, 4096)),
    ("hunyuan_rms_140x4096", "rmsnorm", (1, 140, 4096)),
    ("hunyuan_fused_add_140x4096", "fused_add_rmsnorm", (140, 4096)),
    ("mova_ln_44100x5120", "layernorm", (1, 44100, 5120)),
    ("mova_rms_44100x5120", "rmsnorm", (1, 44100, 5120)),
]

CI_SHAPES = [("qwen_rms_26x3584", "rmsnorm", (1, 26, 3584))]


def _rows_hidden(input_shape):
    rows = 1
    for dim in input_shape[:-1]:
        rows *= dim
    return rows, input_shape[-1]


def _load_flaggems_rmsnorm():
    try:
        from flag_gems.ops.rms_norm import rms_norm
    except ModuleNotFoundError:
        marker.skip("flag_gems not installed")
    return rms_norm


def _load_flaggems_layernorm():
    try:
        from flag_gems.ops.layernorm import layer_norm
    except ModuleNotFoundError:
        marker.skip("flag_gems not installed")
    return layer_norm


def _load_flaggems_fused_add():
    try:
        from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm
    except ModuleNotFoundError:
        marker.skip("flag_gems not installed")
    return fused_add_rms_norm


def _load_quack_rmsnorm():
    import importlib

    try:
        quack_rmsnorm = importlib.import_module("quack.rmsnorm")
        return quack_rmsnorm.rmsnorm_fwd
    except (ModuleNotFoundError, AttributeError):
        marker.skip("quack not available")


def _load_quack_layernorm():
    import importlib

    try:
        quack_rmsnorm = importlib.import_module("quack.rmsnorm")
        return quack_rmsnorm.layernorm_fwd
    except (ModuleNotFoundError, AttributeError):
        marker.skip("quack not available")


# ============================================================================
# rmsnorm
# ============================================================================
RMS_SHAPES = [s for s in ACTUAL_DIFFUSION_SHAPES if s[1] == "rmsnorm"]
RMS_CI = [s for s in CI_SHAPES if s[1] == "rmsnorm"] or [RMS_SHAPES[0]]
RMS_PROVIDERS = [
    "pytorch",
    "sgl_kernel",
    "flashinfer",
    "jit_rmsnorm",
    "triton_rms_norm_fn",
    "quack",
    "flaggems",
]


@marker.parametrize("shape", RMS_SHAPES, RMS_CI)
@marker.benchmark("provider", RMS_PROVIDERS)
def bench_rmsnorm(shape, provider: str):
    import flashinfer.norm as flashinfer_norm
    import sgl_kernel

    _, _, input_shape = shape
    batch_size, hidden_size = _rows_hidden(input_shape)
    x = torch.randn((batch_size, hidden_size), device=DEFAULT_DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=DTYPE)
    out = torch.empty_like(x)

    if provider == "pytorch":

        def fn(x):
            return F.rms_norm(x, (hidden_size,), weight, EPS)

    elif provider == "sgl_kernel":

        def fn(x):
            return sgl_kernel.rmsnorm(x, weight, eps=EPS, out=out)

    elif provider == "flashinfer":

        def fn(x):
            return flashinfer_norm.rmsnorm(x, weight, eps=EPS, out=out)

    elif provider == "jit_rmsnorm":

        def fn(x):
            return jit_rmsnorm(x, weight, out, EPS)

    elif provider == "triton_rms_norm_fn":

        def fn(x):
            return rms_norm_fn(x, weight, bias=None, residual=None, eps=EPS)

    elif provider == "quack":
        quack_rmsnorm_fwd = _load_quack_rmsnorm()

        def fn(x):
            return quack_rmsnorm_fwd(x, weight, eps=EPS)

    else:  # flaggems
        flaggems_rms_norm = _load_flaggems_rmsnorm()

        def fn(x):
            return flaggems_rms_norm(x, (hidden_size,), weight, EPS)

    # Pass x as input_args so do_bench rotates (clones) it per iteration to
    # avoid the L2-hot effect of reusing one buffer. Count both read inputs and
    # the output buffer in the reported HBM bandwidth.
    return marker.do_bench(
        fn,
        input_args=(x,),
        memory_args=(x, weight),
        memory_output=(out,),
    )


# ============================================================================
# fused_add_rmsnorm
# ============================================================================
FUSED_SHAPES = [s for s in ACTUAL_DIFFUSION_SHAPES if s[1] == "fused_add_rmsnorm"]
FUSED_PROVIDERS = [
    "pytorch",
    "sgl_kernel",
    "flashinfer",
    "jit_fused_add_rmsnorm",
    "quack",
    "flaggems",
]


@marker.parametrize("shape", FUSED_SHAPES, FUSED_SHAPES[:1])
@marker.benchmark("provider", FUSED_PROVIDERS)
def bench_fused_add_rmsnorm(shape, provider: str):
    import flashinfer.norm as flashinfer_norm
    import sgl_kernel

    _, _, input_shape = shape
    batch_size, hidden_size = _rows_hidden(input_shape)
    x = torch.randn((batch_size, hidden_size), device=DEFAULT_DEVICE, dtype=DTYPE)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=DTYPE)

    if provider == "pytorch":

        def fn(x, residual):
            return F.rms_norm(x + residual, (hidden_size,), weight, EPS)

    elif provider == "sgl_kernel":

        def fn(x, residual):
            return sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps=EPS)

    elif provider == "flashinfer":

        def fn(x, residual):
            return flashinfer_norm.fused_add_rmsnorm(x, residual, weight, eps=EPS)

    elif provider == "jit_fused_add_rmsnorm":

        def fn(x, residual):
            return jit_fused_add_rmsnorm(x, residual, weight, EPS)

    elif provider == "quack":
        quack_rmsnorm_fwd = _load_quack_rmsnorm()

        def fn(x, residual):
            return quack_rmsnorm_fwd(x, weight, residual=residual, eps=EPS)

    else:  # flaggems
        flaggems_fused_add = _load_flaggems_fused_add()

        def fn(x, residual):
            return flaggems_fused_add(x, residual, (hidden_size,), weight, EPS)

    # sgl_kernel / flashinfer / jit / flaggems fused_add write x + residual in
    # place; passing them as input_args makes do_bench clone both per iter so
    # timing is not L2-hot (the prior input_args=() silently skipped this).
    # Count both read inputs and in-place outputs in reported HBM bandwidth.
    return marker.do_bench(
        fn,
        input_args=(x, residual),
        memory_args=(x, residual, weight),
        memory_output=(x, residual),
    )


# ============================================================================
# layernorm
# ============================================================================
LN_SHAPES = [s for s in ACTUAL_DIFFUSION_SHAPES if s[1] == "layernorm"]
LN_CI = [s for s in CI_SHAPES if s[1] == "layernorm"] or [LN_SHAPES[0]]
LN_PROVIDERS = ["pytorch", "triton_norm_infer", "flashinfer", "quack", "flaggems"]


@marker.parametrize("shape", LN_SHAPES, LN_CI)
@marker.benchmark("provider", LN_PROVIDERS)
def bench_layernorm(shape, provider: str):
    import flashinfer.norm as flashinfer_norm

    _, _, input_shape = shape
    batch_size, hidden_size = _rows_hidden(input_shape)
    x = torch.randn((batch_size, hidden_size), device=DEFAULT_DEVICE, dtype=DTYPE)
    weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=DTYPE)
    bias = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=DTYPE)
    fp32_weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=torch.float32)
    fp32_bias = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    if provider == "pytorch":

        def fn(x):
            return F.layer_norm(x, (hidden_size,), weight, bias, EPS)

    elif provider == "triton_norm_infer":

        def fn(x):
            return norm_infer(x, weight, bias, eps=EPS, is_rms_norm=False, out=out)

    elif provider == "flashinfer":

        def fn(x):
            return flashinfer_norm.layernorm(x, fp32_weight, fp32_bias, EPS)

    elif provider == "quack":
        quack_layernorm_fwd = _load_quack_layernorm()

        def fn(x):
            return quack_layernorm_fwd(x, fp32_weight, fp32_bias, EPS)

    else:  # flaggems
        flaggems_layer_norm = _load_flaggems_layernorm()

        def fn(x):
            return flaggems_layer_norm(x, (hidden_size,), weight, bias)[0]

    # Pass x as input_args so do_bench rotates (clones) it per iteration to
    # avoid the L2-hot effect. Count both read inputs and output buffer.
    return marker.do_bench(
        fn,
        input_args=(x,),
        memory_args=(x, weight, bias),
        memory_output=(out,),
    )


SEP = "=" * 80


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("Benchmark: rmsnorm")
    print(f"{SEP}\n")
    bench_rmsnorm.run()

    print(f"\n{SEP}")
    print("Benchmark: fused_add_rmsnorm")
    print(f"{SEP}\n")
    bench_fused_add_rmsnorm.run()

    print(f"\n{SEP}")
    print("Benchmark: layernorm")
    print(f"{SEP}\n")
    bench_layernorm.run()
