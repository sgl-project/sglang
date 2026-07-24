"""Benchmark TP16 Kimi-K3 fused decode against the production fallback chain."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.kda_fused_decode import kda_fused_decode
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

_HEADS = 6
_D = 128
_SEG = _HEADS * _D
_DECODE = TritonKDAKernel()


def _make_inputs(batch):
    torch.manual_seed(20260724 + batch)
    slots = batch + 3
    device = "cuda"
    weight = torch.randn(3 * _SEG, 4, device=device, dtype=torch.float32) * 0.01
    weight_t = weight.t().contiguous()
    return {
        "mixed_qkv": torch.randn(batch, 3 * _SEG, device=device, dtype=torch.bfloat16),
        "a": torch.randn(batch, _SEG, device=device, dtype=torch.bfloat16),
        "beta": torch.randn(batch, _HEADS, device=device, dtype=torch.bfloat16),
        "conv": torch.randn(slots, 3, 3 * _SEG, device=device, dtype=torch.bfloat16),
        "weight": weight,
        "weight_t": (
            weight_t[:, :_SEG].contiguous(),
            weight_t[:, _SEG : 2 * _SEG].contiguous(),
            weight_t[:, 2 * _SEG :].contiguous(),
        ),
        "bias": torch.randn(3 * _SEG, device=device, dtype=torch.float32) * 0.01,
        "A_log": torch.randn(1, 1, _HEADS, 1, device=device, dtype=torch.float32),
        "dt_bias": torch.randn(_SEG, device=device, dtype=torch.float32),
        "onorm_g": torch.randn(batch, _SEG, device=device, dtype=torch.bfloat16),
        "onorm_weight": torch.randn(_D, device=device, dtype=torch.float32),
        "state": torch.randn(slots, _HEADS, _D, _D, device=device, dtype=torch.float32)
        * 0.01,
        "indices": torch.arange(batch, device=device, dtype=torch.int32),
    }


def _fused(inputs):
    return kda_fused_decode(
        inputs["mixed_qkv"],
        inputs["a"],
        inputs["beta"],
        inputs["conv"],
        *inputs["weight_t"],
        inputs["bias"],
        inputs["A_log"].reshape(-1),
        inputs["dt_bias"],
        inputs["onorm_g"],
        inputs["onorm_weight"],
        inputs["state"],
        inputs["indices"],
        _D**-0.5,
        1e-6,
        -5.0,
    )


def _fallback(inputs):
    batch = inputs["mixed_qkv"].shape[0]
    qkv = causal_conv1d_update(
        inputs["mixed_qkv"],
        inputs["conv"].transpose(-1, -2),
        inputs["weight"],
        inputs["bias"],
        activation="silu",
        conv_state_indices=inputs["indices"],
    )
    core = _DECODE.packed_decode(
        qkv,
        inputs["a"],
        inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=_D**-0.5,
        ssm_states=inputs["state"],
        cache_indices=inputs["indices"],
        num_v_heads=_HEADS,
        head_v_dim=_D,
        lower_bound=-5.0,
    )
    return rms_norm_gated(
        core,
        inputs["onorm_g"].view(1, batch, _HEADS, _D),
        inputs["onorm_weight"],
        None,
        activation="sigmoid",
        eps=1e-6,
    )


@marker.parametrize("batch", [1, 2, 8, 16, 32, 64, 128], [1, 64])
@marker.benchmark("impl", ["fallback", "fused"])
def benchmark(batch: int, impl: str):
    if torch.cuda.get_device_capability()[0] < 9:
        marker.skip("KDA fused decode requires SM90+")
    inputs = _make_inputs(batch)
    fn = _fallback if impl == "fallback" else _fused
    return marker.do_bench(
        lambda: fn(inputs),
        graph_clone_args=(),
        memory_args=(
            inputs["mixed_qkv"],
            inputs["a"],
            inputs["beta"],
            inputs["conv"],
            inputs["state"],
        ),
        memory_output=(inputs["conv"], inputs["state"]),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
