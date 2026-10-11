"""Fresh-process eager attention latency, through SGLang's startup hook.

Run once with --prepare to seed a dedicated cache, then in a NEW process with
the same cache and --prepare (and without, for a control). No CUDA graphs or
model downloads. Two real RadixLinearAttention modules provide model metadata;
the timed operation is one actual SGLang kernel wrapper, not a server forward.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.prepare import (
    LinearAttentionPrepareSpec,
    make_prepare_inputs,
    prepare_model_linear_attention,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention


def reference(x, kind, mode):
    q, k, v = (x[name].double() for name in ("q", "k", "v"))
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    if mode == "prefill":
        q, k = q.bfloat16().double(), k.bfloat16().double()
    if kind == "kda":
        gate = (
            -5
            * torch.sigmoid(
                x["A_log"].double().exp()[None, None, :, None]
                * (x["g"].double() + x["dt_bias"].double())
            )
        ).exp()
        beta = x["beta"].double().sigmoid()
    elif mode == "decode":
        gate = (
            -x["A_log"].double().exp()
            * torch.nn.functional.softplus(x["a"].double() + x["dt_bias"].double())
        ).exp()
        beta = x["b"].double().sigmoid()
    else:
        gate, beta = x["g"].double().exp(), x["beta"].double()
    state = x["ssm_states"].double().clone()
    outputs = []
    for i in range(q.shape[1]):
        slot = i if mode == "decode" else 0
        s = state[slot]
        decay = gate[0, i]
        s = s * (decay[:, None, :] if kind == "kda" else decay[:, None, None])
        residual = (v[0, i] - torch.einsum("hvk,hk->hv", s, k[0, i])) * beta[
            0, i, :, None
        ]
        s = s + residual[:, :, None] * k[0, i, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", s, q[0, i]) / math.sqrt(q.shape[-1]))
        state[slot] = s
    return torch.stack(outputs)[None], state


def compare(actual, expected, atol, rtol):
    diff = (actual.cpu().double() - expected).abs()
    allowed = atol + rtol * expected.abs()
    return {
        "passed": bool(torch.isfinite(diff).all() and (diff <= allowed).all()),
        "max_abs": diff.max().item(),
        "max_normalized": (diff / allowed).max().item(),
        "elements": diff.numel(),
        "atol": atol,
        "rtol": rtol,
    }


@contextlib.contextmanager
def poison_allocations(enabled):
    original = {}
    if enabled:
        for name in ("empty", "empty_like", "empty_strided"):
            fn = getattr(torch, name)
            original[name] = fn

            def allocate(*args, _fn=fn, **kwargs):
                tensor = _fn(*args, **kwargs)
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))
                return tensor

            setattr(torch, name, allocate)
    try:
        yield
    finally:
        for name, fn in original.items():
            setattr(torch, name, fn)


def forbid_compilation():
    """Test-only guards: a disk hit may load, but must not compile or tune."""
    from cutlass import cute
    from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner

    def fail(*args, **kwargs):
        raise AssertionError("Compilation/autotuning forbidden in restart test")

    class NoCompile:
        __call__ = staticmethod(fail)

        def __getitem__(self, options):
            return self

    cute.compile = NoCompile()
    JitSpecCuteDsl.build = fail
    ASTSource.make_ir = fail
    Autotuner._bench = fail


def main(args):
    if args.forbid_compilation:
        forbid_compilation()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    kind, mode = args.op.split("-")
    size = args.size
    heads, dim = 8, 128
    state = torch.full(
        (max(size if mode == "decode" else 1, 2), heads, dim, dim),
        0.02,
        device=device,
        dtype=torch.float32,
    )
    saved_pool = state.clone()
    model = torch.nn.ModuleList(
        [
            RadixLinearAttention(
                i,
                heads,
                heads,
                heads,
                dim,
                dim,
                dim,
                A_log=torch.zeros(heads, device=device),
                dt_bias=torch.zeros(
                    (heads, dim) if kind == "kda" else (heads,), device=device
                ),
                lower_bound=-5.0 if kind == "kda" else None,
            )
            for i in range(2)
        ]
    )
    kernel = TritonKDAKernel() if kind == "kda" else FlashInferGDNKernel()
    dispatcher = SimpleNamespace(extend_kernel=kernel, decode_kernel=kernel)
    backend = SimpleNamespace(kernel_dispatcher=dispatcher, mamba_chunk_size=64)
    runner = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        model=model,
        attn_backend=backend,
        decode_attn_backend=None,
        decode_attn_backend_group=[],
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda _: SimpleNamespace(temporal=state)
        ),
    )
    os.environ["SGLANG_LINEAR_ATTN_PREPARE_PREFILL_TOKENS"] = (
        str(size) if mode == "prefill" else ""
    )
    os.environ["SGLANG_LINEAR_ATTN_PREPARE_DECODE_BATCH_SIZES"] = (
        str(size) if mode == "decode" else ""
    )

    phase = ["init"]
    counts = {}

    def count(name):
        bucket = counts.setdefault(phase[0], {})
        bucket[name] = bucket.get(name, 0) + 1

    if args.diagnose:
        from triton.runtime.autotuner import Autotuner

        bench = Autotuner._bench

        def counted_bench(*a, **kw):
            count("triton_benchmarks")
            return bench(*a, **kw)

        Autotuner._bench = counted_bench
        triton.knobs.runtime.jit_post_compile_hook = lambda **kw: count(
            "triton_memory_misses"
        )
        from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl

        build = JitSpecCuteDsl.build

        def counted_build(*a, **kw):
            count("cute_builds")
            return build(*a, **kw)

        JitSpecCuteDsl.build = counted_build
        from cutlass import cute

        class CompileProbe:
            def __init__(self, compiler):
                self.compiler = compiler

            def __call__(self, *a, **kw):
                count("cute_compiler_calls")
                return self.compiler(*a, **kw)

            def __getitem__(self, options):
                return CompileProbe(self.compiler[options])

            def __getattr__(self, name):
                return getattr(self.compiler, name)

        cute.compile = CompileProbe(cute.compile)

    report = {
        "args": vars(args),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache": {
            name: os.getenv(name)
            for name in (
                "SGLANG_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TRITON_CACHE_AUTOTUNING",
                "FLA_CACHE_RESULTS",
            )
        },
    }
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.cuda.nvtx.range("INIT"):
        report["preparation"] = (
            prepare_model_linear_attention(runner) if args.prepare else []
        )
    torch.cuda.synchronize()
    report["init_seconds"] = time.perf_counter() - start
    assert torch.equal(state, saved_pool), "init modified serving state"
    report["serving_state_unchanged"] = True
    layer = model[0]
    spec = LinearAttentionPrepareSpec(
        heads,
        heads,
        heads,
        dim,
        dim,
        torch.bfloat16,
        state,
        layer.A_log,
        layer.dt_bias,
        layer.lower_bound,
    )
    inputs = make_prepare_inputs(spec, mode, size, kind=kind)
    # Random data, guards preserving strides, and saved inputs outside timing.
    guards, saved = {}, {}
    for name, tensor in list(inputs.items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        if name in ("q", "k", "v", "g", "beta", "a", "b"):
            tensor.copy_(
                torch.randn(tensor.shape, dtype=tensor.dtype, device=device) * 0.2
            )
            if kind == "gdn" and mode == "prefill":
                if name == "g":
                    tensor.copy_(-tensor.abs() - 0.2)
                elif name == "beta":
                    tensor.sigmoid_()
        if name not in ("A_log", "dt_bias"):
            length = 1 + sum(
                (s - 1) * st for s, st in zip(tensor.shape, tensor.stride())
            )
            raw = torch.full((length + 256,), 17, dtype=tensor.dtype, device=device)
            view = raw.as_strided(tensor.shape, tensor.stride(), 128)
            view.copy_(tensor)
            inputs[name] = view
            guards[name] = raw
        saved[name] = inputs[name].clone()
    expected = reference({name: t.cpu() for name, t in saved.items()}, kind, mode)
    call = kernel.decode if mode == "decode" else kernel.extend
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start_event.record()
    end_event.record()
    end_event.synchronize()
    samples = []
    for index in range(1 if args.qualify else args.samples + 1):
        for name, value in saved.items():
            inputs[name].copy_(value)
        torch.cuda.synchronize()
        phase[0] = "first" if index == 0 else "steady"
        with torch.cuda.nvtx.range(phase[0].upper()), poison_allocations(args.qualify):
            start = time.perf_counter_ns()
            start_event.record()
            output = call(**inputs)
            end_event.record()
            enqueued = time.perf_counter_ns()
            end_event.synchronize()
            stop = time.perf_counter_ns()
        samples.append(
            {
                "wall_us": (stop - start) / 1000,
                "enqueue_us": (enqueued - start) / 1000,
                "stream_span_us": start_event.elapsed_time(end_event) * 1000,
            }
        )
        if index == 0:
            # Snapshot only; CPU reference comparison is deferred until after timing.
            first_output = (output[0] if isinstance(output, tuple) else output).clone()
            first_state = inputs["ssm_states"].clone()
    validation = {
        "output": compare(first_output, expected[0], 0.002, 0.02),
        "state": compare(first_state, expected[1], 0.006, 0.02),
        "guards": all(
            bool((raw[:128] == 17).all() and (raw[-128:] == 17).all())
            for raw in guards.values()
        ),
    }
    report.update(
        samples=samples,
        validation=validation,
        diagnostic_counts=counts,
        first_wall_us=samples[0]["wall_us"],
        steady_median_wall_us=statistics.median(s["wall_us"] for s in samples[1:])
        if len(samples) > 1
        else None,
    )
    report["passed"] = validation["guards"] and all(
        validation[k]["passed"] for k in ("output", "state")
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "init_seconds",
                    "first_wall_us",
                    "steady_median_wall_us",
                    "diagnostic_counts",
                )
            }
        )
    )
    assert report["passed"], validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op", choices=("kda-prefill", "gdn-prefill", "gdn-decode"), required=True
    )
    parser.add_argument(
        "--size", type=int, default=256, help="prefill tokens or decode batch size"
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument(
        "--forbid-compilation",
        action="store_true",
        help="Fail on CuTe compilation/build or Triton lowering/autotuning",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.size <= 0 or args.samples <= 0:
        parser.error("size and samples must be positive")
    if Path(args.report).exists():
        parser.error("refusing to overwrite report")
    main(args)
