"""K3 GPU preprocess: shared batched pipeline hook contract (CPU parts)."""

import torch

from sglang.srt.multimodal.processors.kimi_k3 import _fill_transparent_bg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_fill_transparent_bg_batch_matches_per_image():
    """Compositing a batch must be bitwise identical to per-image calls
    (the batched pipeline applies it to whole resize groups)."""
    torch.manual_seed(0)
    batch = torch.rand(3, 4, 8, 6) * 255.0
    cfg = {"pattern": "chessboard", "chessboard_square_size": 2}

    batched = _fill_transparent_bg(batch, cfg)
    per_image = torch.cat(
        [_fill_transparent_bg(batch[i : i + 1], cfg) for i in range(batch.shape[0])]
    )
    assert torch.equal(batched, per_image)


def test_fill_transparent_bg_rgb_passthrough_batch():
    batch = torch.rand(2, 3, 4, 4) * 255.0
    assert _fill_transparent_bg(batch, {"pattern": "white"}) is batch


def test_fill_transparent_bg_no_config_drops_alpha():
    batch = torch.rand(2, 4, 4, 4) * 255.0
    out = _fill_transparent_bg(batch, None)
    assert out.shape == (2, 3, 4, 4)
    assert torch.equal(out, batch[:, :3])
