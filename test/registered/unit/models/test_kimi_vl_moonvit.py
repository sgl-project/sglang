import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

from sglang.srt.models import kimi_vl_moonvit
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


def test_learnable_2d_pos_emb_evicts_oldest_inference_cache_entry(monkeypatch):
    monkeypatch.setattr(kimi_vl_moonvit, "_MAX_INFERENCE_POS_EMB_CACHE_ENTRIES", 1)
    module = Learnable2DInterpPosEmb(height=2, width=2, dim=4).eval()
    inputs = torch.zeros(6, 4)

    module(inputs, torch.tensor([[2, 3]]))
    module(inputs, torch.tensor([[3, 2]]))

    assert len(module._interpolated_pos_emb_cache) == 1
    assert ((3, 2), module.weight.dtype, module.weight.device) in (
        module._interpolated_pos_emb_cache
    )


@pytest.mark.parametrize("use_host_grid", [True, False])
def test_temporal_pooling_preserves_spatial_blocks_in_mixed_batch(use_host_grid):
    grids = [(1, 4, 6), (3, 6, 9)]
    generator = torch.Generator().manual_seed(13)
    clips = [torch.randn(t, h, w, 5, generator=generator) for t, h, w in grids]
    outputs = kimi_vl_moonvit.tpool_patch_merger(
        torch.cat([clip.flatten(0, 2) for clip in clips]),
        torch.tensor(grids),
        merge_kernel_size=(2, 3),
        grid_thw_list=grids if use_host_grid else None,
    )
    assert len(outputs) == len(clips)
    for clip, output in zip(clips, outputs):
        _, height, width, channels = clip.shape
        frame = clip.mean(dim=0)
        expected = torch.stack(
            [
                frame[h : h + 2, w : w + 3].reshape(6, channels)
                for h in range(0, height, 2)
                for w in range(0, width, 3)
            ]
        )
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
