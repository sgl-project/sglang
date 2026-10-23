# SPDX-License-Identifier: Apache-2.0
"""SM89 Qwen image norm + dynamic FP8 quantization; run on an idle GPU."""

import json

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_layernorm_modulate
from sglang.kernels.ops.layernorm.layernorm_modulate_fp8 import layernorm_modulate_fp8
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8


def main():
    torch.manual_seed(42)
    x = torch.randn((1, 4096, 4096), device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)

    def baseline(x, modulation):
        y = fused_layernorm_modulate(x, modulation, None, 1e-6)
        return sglang_per_token_quant_fp8(y.view(-1, 4096))

    results = []
    for order in ("baseline", "fused", "fused", "baseline"):
        fn = baseline if order == "baseline" else layernorm_modulate_fp8
        measured = marker.do_bench(
            fn,
            input_args=(x, modulation),
            warmup_iters=5,
            replay_iters=50,
            disable_log_bandwidth=True,
        )
        results.append({"case": order, "ms": measured.times[0] * 1000})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
