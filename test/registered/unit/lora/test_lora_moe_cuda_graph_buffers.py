from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_moe_cuda_graph_buffers_use_logical_marlin_dimensions():
    config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=8,
        hidden_size=32,
        intermediate_size_per_partition=16,
        top_k=4,
        is_gated=True,
    )
    quant_info = MarlinMoeQuantInfo(
        w13_qweight=torch.empty((8, 2, 64), dtype=torch.int32),
        w2_qweight=torch.empty((8, 1, 64), dtype=torch.int32),
        w13_scales=torch.empty((8, 32)),
        w2_scales=torch.empty((8, 16)),
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
    )
    moe_layer = SimpleNamespace(
        base_layer=SimpleNamespace(moe_runner_config=config),
        _quant_info=quant_info,
    )
    backend = BaseLoRABackend.__new__(BaseLoRABackend)
    backend.device = torch.device("cpu")

    backend.init_cuda_graph_moe_buffers(
        max_bs=3,
        max_loras=2,
        compute_dtype=torch.bfloat16,
        moe_layer=moe_layer,
    )

    buffers = backend.moe_cg_buffers
    assert buffers["intermediate_cache1"].shape == (3, 4, 32)
    assert buffers["intermediate_cache2"].shape == (12, 16)
    assert buffers["intermediate_cache3"].shape == (3, 4, 32)
    assert buffers["out_hidden_states"].shape == (3, 32)
    assert buffers["intermediate_cache1"].dtype == torch.bfloat16
    assert buffers["intermediate_cache1"].device.type == "cpu"
