"""Current SGLang AITER routed+shared math; not loader/endpoint qualification."""

import json
import os

import torch

os.environ["SGLANG_USE_AITER_MOE_GU_ITLV"] = "0"
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy", device="cuda"), role="test")
from aiter.ops.shuffle import shuffle_weight

from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def quantize(w):
    e, n, k = w.shape
    blocks = w.float().reshape(e, n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax((2, 4)).clamp_min(1e-8) / 448
    q = (
        (blocks / scale[:, :, None, :, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape_as(w)
    )
    dequant = (q.float().reshape_as(blocks) * scale[:, :, None, :, None]).reshape_as(w)
    return q, scale.contiguous(), dequant


def oracle(x, w1, w2, ids, scores, limit):
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(w1.shape[0]):
        gu = x.float() @ w1[e].float().T
        gate, up = gu.chunk(2, -1)
        if limit:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        yy = (torch.nn.functional.silu(gate) * up) @ w2[e].float().T
        weight = ((ids == e) * scores).sum(-1, keepdim=True)
        out += yy * weight
    return out


torch.manual_seed(20260913)
with torch.inference_mode():
    e, hidden, inter = 9, 512, 256
    originals = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05
        for shape in ((e, 2 * inter, hidden), (e, hidden, inter))
    ]
    for precision in ("bf16", "fp8"):
        if precision == "fp8":
            w1, s1, ref1 = quantize(originals[0])
            w2, s2, ref2 = quantize(originals[1])
            quant_type = AiterQuantType.PER_128X128
        else:
            w1, w2 = originals
            ref1, ref2, s1, s2 = w1, w2, None, None
            quant_type = AiterQuantType.NONE
        packed1, packed2 = shuffle_weight(w1, (16, 16)), shuffle_weight(w2, (16, 16))
        packed1.is_shuffled = packed2.is_shuffled = True
        for n in (1, 8, 16, 65):
            for limit in (0.0, 10.0):
                x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
                if limit:
                    x *= 12
                ids = torch.stack(
                    (
                        torch.arange(n, device="cuda") % 8,
                        (torch.arange(n, device="cuda") + 3) % 8,
                        torch.full((n,), 8, device="cuda"),
                    ),
                    -1,
                ).int()
                scores = (
                    torch.tensor([0.4, 0.6, 1.0], device="cuda", dtype=torch.float32)
                    .expand(n, -1)
                    .contiguous()
                )
                core = AiterRunnerCore(
                    MoeRunnerConfig(
                        activation="silu",
                        top_k=3,
                        num_experts=e,
                        hidden_size=hidden,
                        intermediate_size_per_partition=inter,
                        num_fused_shared_experts=1,
                    )
                )
                info = AiterMoeQuantInfo(
                    packed1,
                    packed2,
                    quant_type=quant_type,
                    w13_scale=s1,
                    w2_scale=s2,
                    swiglu_limit=limit,
                )
                inputs = AiterRunnerInput(x, ids, scores, quant_type)
                actual = core.run(inputs, info, {}).hidden_states
                gold = oracle(x, ref1, ref2, ids, scores, limit)
                err = (
                    (actual.float() - gold).square().mean().sqrt()
                    / gold.square().mean().sqrt().clamp_min(1e-8)
                ).item()
                print(
                    json.dumps(
                        {
                            "precision": precision,
                            "tokens": n,
                            "clamp": limit,
                            "nrmse": err,
                        }
                    ),
                    flush=True,
                )
                assert torch.isfinite(actual).all()
                assert err < (0.06 if precision == "fp8" else 0.02), err
                if n in (8, 16):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            core.run(inputs, info, {})
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = core.run(inputs, info, {}).hidden_states
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, captured, atol=0.02, rtol=0.02)
print("SHARED_MOE_RUNNER_PASS_NO_CHECKPOINT_OR_ENDPOINT", flush=True)
