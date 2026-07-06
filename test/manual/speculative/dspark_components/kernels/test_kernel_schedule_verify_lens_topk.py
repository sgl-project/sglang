import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

GAMMA = 5


def _make_confidence(mode, bs, device):
    g = torch.Generator(device=device).manual_seed(1234 + bs)
    base = torch.rand(bs, GAMMA, device=device, generator=g)
    if mode == "random":
        return base
    if mode == "ties":
        return torch.full((bs, GAMMA), 0.5, device=device)
    if mode == "coarse":
        return (base * 4).floor() / 4
    if mode == "some_invalid":
        return torch.where(base < 0.3, torch.zeros_like(base), base)
    raise ValueError(mode)


@pytest.mark.parametrize("bs", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("budget", [0, 1, 3, 7, 10, 1000])
@pytest.mark.parametrize("mode", ["random", "ties", "coarse", "some_invalid"])
def test_triton_matches_torch_selection(bs, budget, mode):
    device = torch.device("cuda")
    cfg = DSparkScheduleConfig(gamma=GAMMA)
    confidence = _make_confidence(mode, bs, device)
    ref = schedule_verify_lens_topk(confidence=confidence, budget=budget, cfg=cfg)
    got = schedule_verify_lens_topk_triton(
        confidence=confidence, budget=budget, cfg=cfg
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
