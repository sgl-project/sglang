import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.activation.activation import relu2
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    _apply_qwen3_qk_norm_rope_pack_kv,
    _apply_qwen3_qk_norm_rope_split,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def eager_relu2(x):
    x = F.relu(x)
    return x * x


@marker.parametrize("case", ["qk_400", "qk_1024", "relu2_400", "relu2_8190"])
@marker.benchmark("provider", ["eager", "fused"])
def benchmark(case, provider):
    operation, tokens = case.split("_")
    tokens = int(tokens)
    if operation == "relu2":
        x = create_random(1, tokens, 9216)
        fn = eager_relu2 if provider == "eager" else lambda x: relu2(x, fast_math=False)
        return marker.do_bench(fn, input_args=(x,))

    qkv = create_random(1, tokens, 32, 128)
    k_und, v_und = create_random(1, 32, 8, 128), create_random(1, 32, 8, 128)
    angles = torch.randn(tokens, 64, device=qkv.device)
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(qkv.dtype)
    if provider == "eager":
        cache = cache.float()
    positions = torch.arange(tokens, device=qkv.device)
    q_norm = RMSNorm(128, eps=1e-6).to(device=qkv.device, dtype=qkv.dtype)
    k_norm = RMSNorm(128, eps=1e-6).to(device=qkv.device, dtype=qkv.dtype)

    def fn(qkv, k_und, v_und, cache, positions):
        q, k, v = qkv[:, :, :16], qkv[:, :, 16:24], qkv[:, :, 24:]
        if provider == "eager":
            q, k = _apply_qwen3_qk_norm_rope_split(q, k, q_norm, k_norm, 128, cache)
            return q, torch.cat((k_und, k), dim=1), torch.cat((v_und, v), dim=1)
        return _apply_qwen3_qk_norm_rope_pack_kv(
            q,
            k,
            v,
            k_und,
            v_und,
            q_norm,
            k_norm,
            128,
            cache,
            positions,
            round_norm_before_rope=True,
        )

    return marker.do_bench(fn, input_args=(qkv, k_und, v_und, cache, positions))


if __name__ == "__main__":
    benchmark.run()
