"""Real CUDA quantizer parity and graph-storage tests for online refresh.

Run explicitly on CUDA; this does not replace an end-to-end Olmo publication test.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_layer(kind):
    from sglang.srt.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
        Fp8MoEMethod,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(model_path="unused", moe_runner_backend="triton")
    )
    config = Fp8Config(is_checkpoint_fp8_serialized=False)
    with torch.device("cuda"):
        layer = torch.nn.Module()
        if kind == "moe":
            method = Fp8MoEMethod(config)
            layer.moe_runner_config = SimpleNamespace(is_gated=True)
            layer.num_local_experts = 4
            layer.intermediate_size_per_partition = 256
            layer.hidden_size = 128
            layer.use_presharded_weights = False
            layer.moe_tp_size = 1
            method.create_weights(layer, 4, 128, 256, torch.bfloat16)
            weights = ("w13_weight", "w2_weight")
        else:
            method = Fp8LinearMethod(config)
            method.create_weights(
                layer,
                128,
                [256, 256],
                128,
                512,
                torch.bfloat16,
                weight_loader=lambda *args: None,
            )
            weights = ("weight",)
    return layer, method, weights


@pytest.mark.parametrize("kind", ["moe", "linear"])
def test_native_quantizer_matches_cold_after_split_bucket_updates(kind):
    torch.manual_seed(17)
    layer, method, names = make_layer(kind)
    sources = {name: torch.randn_like(getattr(layer, name)) for name in names}
    for name, value in sources.items():
        getattr(layer, name).data.copy_(value)
    method.process_weights_after_loading(layer)
    serving = dict(layer.named_parameters())
    pointers = {name: value.data_ptr() for name, value in serving.items()}
    # Capture reads of serving tensors. Replaying must see newly published data.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_sum = sum(value.float().sum() for value in serving.values())
    for factor in (1, -3, 1):
        method.restore_weights_before_loading(layer)
        for name, value in sources.items():
            target = getattr(layer, name)
            # Simulate distinct gate/up buckets (and expert slices for MoE).
            for index in reversed(range(value.shape[0])):
                target.data[index].copy_(value[index] * factor)
        method.process_weights_after_loading(layer)
        cold, cold_method, _ = make_layer(kind)
        for name, value in sources.items():
            getattr(cold, name).data.copy_(value * factor)
        cold_method.process_weights_after_loading(cold)
        for name, value in layer.named_parameters():
            assert value is serving[name]
            assert value.data_ptr() == pointers[name]
            torch.testing.assert_close(
                value.float(), getattr(cold, name).float(), atol=0, rtol=0
            )
        graph.replay()
        torch.cuda.synchronize()
        expected_sum = sum(value.float().sum() for value in layer.parameters())
        torch.testing.assert_close(graph_sum, expected_sum, rtol=0, atol=0)
