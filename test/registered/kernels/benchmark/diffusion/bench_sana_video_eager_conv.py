import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.multimodal_gen.runtime.models.dits.sana import (
    sana_conv_bias_glu,
    sana_conv_bias_silu,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("operation", ["bias_silu", "bias_glu"])
@marker.benchmark("provider", ["eager", "fused"])
def benchmark(operation, provider):
    channels = 2240 if operation == "bias_silu" else 13440
    x = create_random(21, channels, 30, 52).to(memory_format=torch.channels_last)
    conv = torch.nn.Conv2d(
        channels,
        13440,
        1 if operation == "bias_silu" else 3,
        padding=0 if operation == "bias_silu" else 1,
        groups=1 if operation == "bias_silu" else channels,
    ).to(device=x.device, dtype=x.dtype)

    @torch.inference_mode()
    def fn(x):
        if provider == "fused":
            fused = (
                sana_conv_bias_silu if operation == "bias_silu" else sana_conv_bias_glu
            )
            return fused(conv, x, allow_eager=True)
        y = conv(x)
        if operation == "bias_silu":
            return F.silu(y)
        y, gate = y.chunk(2, dim=1)
        return y * F.silu(gate)

    return marker.do_bench(fn, input_args=(x,))


if __name__ == "__main__":
    benchmark.run()
