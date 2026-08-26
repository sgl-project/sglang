"""ROCm regression tests for explicit AITER MoE backend selection."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(not is_hip(), reason="AITER routing is ROCm-only")
def test_explicit_aiter_runner_selects_aiter_topk(monkeypatch):
    """The explicit MoE backend must not fall back to the non-AITER router."""
    from sglang.srt.layers.moe import topk as moe_topk

    expected_weights = torch.full((2, 10), 0.1, device="cuda")
    expected_ids = torch.arange(10, dtype=torch.int32, device="cuda").repeat(2, 1)
    calls = []

    def fake_aiter_fused_topk(*args, **kwargs):
        calls.append((args, kwargs))
        return expected_weights, expected_ids

    monkeypatch.setattr(moe_topk, "will_use_aiter_moe", lambda: True)
    monkeypatch.setattr(
        moe_topk,
        "_get_aiter_topk_ops",
        lambda: (None, fake_aiter_fused_topk),
    )

    hidden_states = torch.randn((2, 32), dtype=torch.bfloat16, device="cuda")
    gating_output = torch.randn((2, 512), dtype=torch.bfloat16, device="cuda")
    weights, ids = moe_topk.fused_topk(
        hidden_states,
        gating_output,
        topk=10,
        renormalize=True,
    )

    assert len(calls) == 1
    assert weights is expected_weights
    assert ids is expected_ids


@pytest.mark.skipif(not is_hip(), reason="AITER routing is ROCm-only")
def test_explicit_aiter_runner_selects_unquantized_aiter_moe(monkeypatch):
    """BF16 MoE must honor the explicit runner without the global AITER flag."""
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.quantization import unquant

    created = []

    class FakeRunner:
        def __init__(self, backend, config):
            self.runner_backend = backend
            created.append((backend, config))

    method = object.__new__(unquant.UnquantizedFusedMoEMethod)
    method.use_flashinfer_trtllm_moe = False
    method.use_flashinfer_cutlass = False
    method.use_deep_gemm = False
    method.use_triton_kernels = False

    monkeypatch.setattr(unquant, "MoeRunner", FakeRunner)
    monkeypatch.setattr(unquant, "will_use_aiter_moe", lambda: True)
    monkeypatch.setattr(unquant, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE)

    config = SimpleNamespace()
    layer = SimpleNamespace(intermediate_size_per_partition=640)
    method.create_moe_runner(layer, config)

    assert method.runner.runner_backend is MoeRunnerBackend.TRITON
    assert method._aiter_runner.runner_backend is MoeRunnerBackend.AITER
    assert [backend for backend, _ in created] == [
        MoeRunnerBackend.TRITON,
        MoeRunnerBackend.AITER,
    ]


@pytest.mark.skipif(not is_hip(), reason="AITER routing is ROCm-only")
def test_unquantized_aiter_moe_pads_and_loads_tp_shards(monkeypatch):
    """AITER padding must not change the logical TP checkpoint offsets."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization import unquant

    monkeypatch.setattr(unquant, "will_use_aiter_moe", lambda: True)
    monkeypatch.setattr(
        "sglang.srt.layers.moe.fused_moe_triton.layer.will_use_aiter_moe",
        lambda: True,
    )

    method = unquant.UnquantizedFusedMoEMethod()
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.quant_method = method
    layer.moe_tp_size = 2
    layer.moe_tp_rank = 1
    layer.intermediate_size_per_partition = 320
    layer.use_presharded_weights = False
    layer.use_triton_kernels = False
    layer.use_flashinfer_trtllm_moe = False
    layer.quant_config = None

    method.create_weights(
        layer=layer,
        num_experts=1,
        hidden_size=16,
        intermediate_size_per_partition=320,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight.shape == (1, 768, 16)
    assert layer.w2_weight.shape == (1, 16, 384)
    assert layer.w13_weight.weight_padded
    assert layer.w2_weight.weight_padded
    assert layer.intermediate_pad == 64

    w1 = torch.arange(640 * 16, dtype=torch.float32).reshape(640, 16)
    w3 = w1 + 100_000
    w2 = torch.arange(16 * 640, dtype=torch.float32).reshape(16, 640)
    layer._load_w13(layer.w13_weight.data[0], 0, "w1", w1, tp_rank=1)
    layer._load_w13(layer.w13_weight.data[0], 0, "w3", w3, tp_rank=1)
    layer._load_w2(layer.w2_weight.data[0], 1, "w2", w2, tp_rank=1)

    torch.testing.assert_close(
        layer.w13_weight.data[0, :320], w1[320:].to(torch.bfloat16)
    )
    torch.testing.assert_close(
        layer.w13_weight.data[0, 384:704], w3[320:].to(torch.bfloat16)
    )
    torch.testing.assert_close(
        layer.w2_weight.data[0, :, :320], w2[:, 320:].to(torch.bfloat16)
    )
    assert torch.count_nonzero(layer.w13_weight.data[0, 320:384]) == 0
    assert torch.count_nonzero(layer.w13_weight.data[0, 704:768]) == 0
    assert torch.count_nonzero(layer.w2_weight.data[0, :, 320:384]) == 0


@pytest.mark.skipif(not is_hip(), reason="AITER routing is ROCm-only")
def test_unquantized_aiter_quant_info_carries_padding(monkeypatch):
    """The AITER runner must receive the physical padding metadata."""
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization import unquant

    captured = []

    class FakeAiterRunner:
        def run(self, dispatch_output, quant_info):
            captured.append(quant_info)
            return quant_info

    method = object.__new__(unquant.UnquantizedFusedMoEMethod)
    method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
    method._aiter_runner = FakeAiterRunner()
    method.use_flashinfer_cutlass = False
    method.use_flashinfer_trtllm_moe = False
    method.use_deep_gemm = False
    method.use_triton_kernels = False

    layer = SimpleNamespace(
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        dispatcher=SimpleNamespace(),
        hidden_pad=0,
        intermediate_pad=64,
    )
    dispatch_output = SimpleNamespace(hidden_states=torch.empty(1))

    method.forward_cuda(layer, dispatch_output)

    assert len(captured) == 1
    assert captured[0].expert_mask is None
    assert captured[0].hidden_pad == 0
    assert captured[0].intermediate_pad == 64
