"""Diagnose clamp dispatch against independent clamped/unclamped MoE math."""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import torch

os.environ["SGLANG_USE_AITER_MOE_GU_ITLV"] = "0"
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy", device="cuda"), role="test")
import aiter
from aiter.ops.shuffle import shuffle_weight

from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def emit(value):
    print(json.dumps(value), flush=True)


def quantize(w):
    e, n, k = w.shape
    blocks = w.float().reshape(e, n // 128, 128, k // 128, 128)
    maximum = torch.finfo(aiter.dtypes.fp8).max
    scale = blocks.abs().amax((2, 4)).clamp_min(1e-8) / maximum
    q = (blocks / scale[:, :, None, :, None]).clamp(-maximum, maximum)
    q = q.to(aiter.dtypes.fp8).reshape_as(w)
    ref = (q.float().reshape_as(blocks) * scale[:, :, None, :, None]).reshape_as(w)
    return q, scale.contiguous(), ref


def oracle(x, w1, w2, ids, weights, clamp):
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        gate, up = (x.float() @ w1[expert].float().T).chunk(2, -1)
        if clamp:
            gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        y = (torch.nn.functional.silu(gate) * up) @ w2[expert].float().T
        result += y * ((ids == expert) * weights).sum(-1, keepdim=True)
    return result


def error(actual, gold):
    return (
        (actual.float() - gold).square().mean().sqrt()
        / gold.square().mean().sqrt().clamp_min(1e-8)
    ).item()


module = importlib.import_module("aiter.fused_moe")
emit(
    {
        "kind": "source",
        "path": module.__file__,
        "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "fp8_dtype": str(aiter.dtypes.fp8),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "ck_stage1_signature": str(inspect.signature(module.ck_moe_stage1)),
    }
)
original_metadata = module.get_2stage_cfgs


def report_metadata(*args, **kwargs):
    value = original_metadata(*args, **kwargs)
    emit({"kind": "dispatch", "metadata": str(value)})
    return value


module.get_2stage_cfgs = report_metadata
torch.manual_seed(20260913)
rows = []
with torch.inference_mode():
    experts, hidden, inter = 9, 512, 256
    original = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05
        for shape in ((experts, inter * 2, hidden), (experts, hidden, inter))
    ]
    for precision in ("bf16", "native_fp8"):
        if precision == "bf16":
            w1, w2 = original
            ref1, ref2, s1, s2 = w1, w2, None, None
            quant_type = AiterQuantType.NONE
        else:
            w1, s1, ref1 = quantize(original[0])
            w2, s2, ref2 = quantize(original[1])
            quant_type = AiterQuantType.PER_128X128
        p1, p2 = shuffle_weight(w1, (16, 16)), shuffle_weight(w2, (16, 16))
        p1.is_shuffled = p2.is_shuffled = True
        for n in (1, 8, 16):
            ids = torch.stack(
                (
                    torch.arange(n, device="cuda") % 8,
                    (torch.arange(n, device="cuda") + 3) % 8,
                    torch.full((n,), 8, device="cuda"),
                ),
                -1,
            ).int()
            weights = (
                torch.tensor([0.4, 0.6, 1.0], device="cuda").expand(n, -1).contiguous()
            )
            for clamp in (0.0, 10.0):
                x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
                if clamp:
                    x *= 12
                gold = oracle(x, ref1, ref2, ids, weights, clamp)
                unclamped = oracle(x, ref1, ref2, ids, weights, 0)
                config = MoeRunnerConfig(
                    activation="silu",
                    top_k=3,
                    num_experts=experts,
                    num_local_experts=experts,
                    hidden_size=hidden,
                    intermediate_size_per_partition=inter,
                    num_fused_shared_experts=1,
                    swiglu_limit=clamp or None,
                    gate_up_interleaved=False,
                )
                for backend in os.environ.get("CLAMP_BACKENDS", "aiter").split(","):
                    assert backend in ("aiter", "triton")
                    if backend == "aiter":
                        core = AiterRunnerCore(config)
                        info = AiterMoeQuantInfo(
                            p1,
                            p2,
                            quant_type=quant_type,
                            w13_scale=s1,
                            w2_scale=s2,
                            swiglu_limit=clamp,
                        )

                        def run():
                            return core.run(
                                AiterRunnerInput(x, ids, weights, quant_type), info, {}
                            ).hidden_states

                    else:
                        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                            fused_experts,
                        )

                        def run():
                            return fused_experts(
                                x.clone(),
                                w1,
                                w2,
                                (weights, ids, None),
                                config,
                                use_fp8_w8a8=precision != "bf16",
                                w1_scale=s1,
                                w2_scale=s2,
                                block_shape=[128, 128] if precision != "bf16" else None,
                            )

                    try:
                        actual = run()
                        torch.cuda.synchronize()
                        finite = bool(torch.isfinite(actual).all())
                        err = error(actual, gold)
                        passed = finite and err < (
                            0.06 if precision != "bf16" else 0.02
                        )
                        row = dict(
                            backend=backend,
                            precision=precision,
                            n=n,
                            clamp=clamp,
                            nrmse=err,
                            unclamped_nrmse=error(actual, unclamped),
                            passed=passed,
                        )
                        if passed and n == 8:
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(3):
                                    run()
                            torch.cuda.current_stream().wait_stream(stream)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                captured = run()
                            for _ in range(5):
                                graph.replay()
                            torch.cuda.synchronize()
                            row["graph_nrmse"] = error(captured, gold)
                            row["graph_passed"] = row["graph_nrmse"] < (
                                0.06 if precision != "bf16" else 0.02
                            )
                    except (
                        ImportError,
                        AttributeError,
                        NotImplementedError,
                        AssertionError,
                    ) as exc:
                        row = dict(
                            backend=backend,
                            precision=precision,
                            n=n,
                            clamp=clamp,
                            passed=False,
                            error=repr(exc),
                        )
                    # Runtime GPU faults are intentionally not swallowed.
                    rows.append(row)
                    emit(row)
emit(
    {
        "kind": "summary",
        "cases": len(rows),
        "passed": sum(r.get("passed", False) for r in rows),
        "scope": "primitive diagnosis only, not shared-expert loader or serving qualification",
    }
)
