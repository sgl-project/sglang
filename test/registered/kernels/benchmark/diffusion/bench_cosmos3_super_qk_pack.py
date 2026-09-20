"""Native split versus rounded QK-norm/RoPE/KV-pack at Super T2I TP2 shapes."""

import statistics

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    _apply_qwen3_qk_norm_rope_pack_kv,
    _apply_qwen3_qk_norm_rope_split,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def measure(fn, reset, graph=False):
    for _ in range(10):
        reset()
        fn()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            _output = fn()  # Retain captured outputs through timing.
        fn = capture.replay
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(40):
        # Restore the request-owned input before the timer; fusion mutates Q.
        reset()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    for tokens in (1024, 4096):
        source = torch.randn(1, tokens, 40, 128, device="cuda", dtype=torch.bfloat16)
        qkv = source.clone()
        prefix = torch.randn(1, 23, 8, 128, device="cuda", dtype=torch.bfloat16)
        k_prefix, v_prefix = prefix[:, :, :4], prefix[:, :, 4:]
        norms = [RMSNorm(128, eps=1e-6).cuda().bfloat16() for _ in range(2)]
        for norm in norms:
            norm.weight.normal_()
        positions = torch.arange(tokens, device="cuda")
        angles = torch.randn(tokens, 64, device="cuda")
        rounded_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).bfloat16()
        native_cache = rounded_cache.float()

        def split():
            q, k = _apply_qwen3_qk_norm_rope_split(
                qkv[:, :, :32], qkv[:, :, 32:36], *norms, 128, native_cache
            )
            return (
                q,
                torch.cat([k_prefix, k], dim=1),
                torch.cat([v_prefix, qkv[:, :, 36:]], dim=1),
            )

        def fused():
            return _apply_qwen3_qk_norm_rope_pack_kv(
                qkv[:, :, :32],
                qkv[:, :, 32:36],
                qkv[:, :, 36:],
                k_prefix,
                v_prefix,
                *norms,
                128,
                rounded_cache,
                positions,
                round_norm_before_rope=True,
            )

        def reset():
            qkv.copy_(source)

        reference = split()
        actual = fused()
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(actual, reference, strict=True)
        )
        for graph in (False, True):
            native_us = measure(split, reset, graph)
            fused_us = measure(fused, reset, graph)
            print(
                f"tokens={tokens}, prefix=23, q_heads=32, kv_heads=4, head_dim=128, graph={graph}: native={native_us:.3f} us, fused={fused_us:.3f} us, speedup={native_us / fused_us:.3f}x"
            )


if __name__ == "__main__":
    main()
