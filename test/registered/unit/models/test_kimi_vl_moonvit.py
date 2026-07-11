import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

from sglang.srt.models.kimi_vl_moonvit import Learnable2DInterpPosEmb


def test_learnable_2d_pos_emb_caches_inference_interpolation(monkeypatch):
    module = Learnable2DInterpPosEmb(height=2, width=2, dim=4).eval()
    inputs = torch.zeros(6, 4)
    grid_hw = torch.tensor([[2, 3]])
    calls = 0
    original_interpolate = torch.nn.functional.interpolate

    def counting_interpolate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_interpolate(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", counting_interpolate)
    first = module(inputs, grid_hw)
    second = module(inputs, grid_hw)
    torch.testing.assert_close(first, second)
    assert calls == 1


def test_learnable_2d_pos_emb_does_not_cache_training_interpolation(monkeypatch):
    module = Learnable2DInterpPosEmb(height=2, width=2, dim=4).train()
    inputs = torch.zeros(6, 4)
    grid_hw = torch.tensor([[2, 3]])
    calls = 0
    original_interpolate = torch.nn.functional.interpolate

    def counting_interpolate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_interpolate(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", counting_interpolate)
    module(inputs, grid_hw)
    module(inputs, grid_hw)
    assert calls == 2
