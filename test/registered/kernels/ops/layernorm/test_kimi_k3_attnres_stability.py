# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Synthetic AttnRes cancellation and branch preservation on one SM103 GPU.

The ready one-slot Oproj mailbox exercises consumer arithmetic and barriers only;
this does not qualify TP8 producers, TwoShot, full-model quality or performance.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-b", runner_config="2-gpu-gb300")


def _gpu_ready():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM103 GPU and CuTe")


@pytest.mark.parametrize(
    "consumer,batch,nb,final",
    [
        ("cluster", 1, 1, False),
        ("cluster", 64, 8, False),
        ("cluster", 1, 0, False),
        ("cluster", 1, 1, True),
        ("oproj", 1, 1, False),
        ("oproj", 64, 8, False),
        ("oproj", 1, 0, False),
    ],
)
def test_attnres_cancellation_is_finite_in_eager_and_graph(consumer, batch, nb, final):
    import cutlass
    from cutlass import cute

    from sglang.kernels.jit.kimi_k3.host import stream
    from sglang.kernels.jit.kimi_k3.tensor_adapter import to_cute
    from sglang.kernels.ops.communication.kimi_k3_specialized.oproj_consumer import (
        OprojConsumer,
    )
    from sglang.kernels.ops.layernorm.kimi_k3_specialized.attn_res import ClusterAttnRes

    saved = torch.full((7168,), 140.0, dtype=torch.bfloat16, device="cuda")
    seed = -saved.expand(batch, -1).contiguous()
    seed[:, 0] = -142.0
    prefix = seed.clone()
    bank = torch.zeros(batch, max(1, nb), 7168, dtype=seed.dtype, device="cuda")
    if nb:
        bank[:, 0].copy_(saved)
    bank_before = bank.clone()
    weight, query = torch.ones_like(saved), torch.zeros_like(saved)
    output, delta = torch.empty_like(seed), torch.zeros_like(seed)
    if consumer == "cluster":
        kernel = ClusterAttnRes(nb, slots=0, final=final)
        tensors = (delta, prefix, bank, weight, query, weight, output)
    else:
        delta = delta.unsqueeze(0)
        kernel = OprojConsumer(nb, groups=4 if batch <= 32 else 1)
        tensors = (delta, prefix, bank, weight, query, weight, output, None, None)
    args = tuple(None if t is None else to_cute(t) for t in tensors)
    args += (cutlass.Float32(1.0e-5), cutlass.Float32(1.0e-5))
    compiled = cute.compile(kernel, *args, stream(torch.cuda.current_device()))
    mixture = (seed.double() + bank[:, :nb].double().sum(1)) / (nb + 1)
    if final:
        mixture = mixture.bfloat16().double()
    expected = (
        mixture * torch.rsqrt(mixture.square().mean(-1, keepdim=True) + 1.0e-5)
    ).bfloat16()

    def launch():
        prefix.copy_(seed)
        delta.zero_()  # The consumer re-arms its mailbox after each launch.
        compiled(*args, stream(torch.cuda.current_device()))

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for mode in ("eager", "graph"):
        if mode == "eager":
            launch()
        else:
            for _ in range(3):
                graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all(), (consumer, nb, mode)
        torch.testing.assert_close(output, expected, rtol=1 / 128, atol=1 / 64)
        assert torch.equal(prefix, seed), "zero-delta prefix changed"
        assert torch.equal(bank, bank_before), "read-only bank changed"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
