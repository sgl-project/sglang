import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Mxfp4MegaMoE,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100",
)
def test_split_mxfp4_experts_match_bf16_reference(monkeypatch):
    """Packing preserves gate/up order and the expert-specific E8M0 scales."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.mega_moe import run_mega_routed_experts

    monkeypatch.setenv(
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.name, "64"
    )
    torch.manual_seed(7)
    experts, hidden, intermediate = 8, 1024, 512
    layer = torch.nn.Module()
    layer.num_experts = experts
    layer.moe_runner_config = MoeRunnerConfig()
    layer._mega_moe_nvfp4 = False
    layer._mega_moe_w4a4 = True
    scheme = CompressedTensorsW4A4Mxfp4MegaMoE()
    with torch.device("cuda"):
        scheme.create_weights(layer, experts, hidden, intermediate, torch.bfloat16)
    scheme.create_moe_runner(layer, layer.moe_runner_config)

    fp4_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )

    def initialize(packed, scales):
        packed.random_(0, 256)
        scales.random_(121, 125)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        exponents = scales.int().repeat_interleave(32, dim=-1) - 127
        return torch.ldexp(fp4_values[nibbles.long()], exponents)

    gate_up = initialize(layer.w13_weight_packed, layer.w13_weight_scale)
    down = initialize(layer.w2_weight_packed, layer.w2_weight_scale)
    scheme.process_weights_after_loading(layer)

    expert_ids = torch.tensor([[0, 3], [1, 6], [7, 4]], device="cuda")
    routing_weights = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]], device="cuda")
    inputs = fp4_values[torch.randint(16, (3, hidden), device="cuda")].bfloat16()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    try:
        with get_parallel().override(
            moe_ep_group=SimpleNamespace(device_group=dist.group.WORLD)
        ):
            output = run_mega_routed_experts(
                layer,
                inputs,
                expert_ids,
                routing_weights,
                hidden_size=hidden,
                intermediate_size=intermediate,
                top_k=2,
                num_tokens=inputs.shape[0],
            )
        gate, up = torch.einsum(
            "th,tenh->ten", inputs.float(), gate_up[expert_ids]
        ).chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        expected = torch.einsum(
            "tei,tehi,te->th", activated, down[expert_ids], routing_weights
        )
        relative_l2 = (output.float() - expected).norm() / expected.norm()
        assert relative_l2.item() < 0.20
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
