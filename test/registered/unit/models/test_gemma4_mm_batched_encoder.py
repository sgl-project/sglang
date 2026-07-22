"""Unit tests for the batched vision-encoder path in ``gemma4_mm.py``.

The vision tower and embedder are stubbed so the tests run on CPU without the
real Gemma-4 checkpoint; they assert encoder/embedder call counts, output
ordering, budget-bound chunking, and device-agnostic budget computation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from sglang.srt.models import gemma4_mm as gemma4_mm_module
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


def _make_fake_model(
    hidden_size: int = 16,
    *,
    encoder_max_batch: int | None = None,
    fail_pad: bool = False,
):
    """Lightweight stand-in exposing only the attributes the encoder helpers
    touch. The fake tower records call shapes and embeds each patch as a
    constant vector keyed on its batch row, so per-item ordering is verifiable.
    """

    class _FakeTower:
        device = torch.device("cpu")

        def __init__(self):
            self.calls: List[tuple[torch.Tensor, torch.Tensor]] = []

        def __call__(self, pv: torch.Tensor, pp: torch.Tensor):
            self.calls.append((pv.clone(), pp.clone()))
            b, n, _ = pv.shape
            # pp == -1 marks padding (the real Gemma4 convention).
            pooler_mask = (pp != -1).all(dim=-1)
            hidden = (
                torch.arange(b, dtype=torch.float32)
                .view(b, 1, 1)
                .repeat(1, n, hidden_size)
            )
            return hidden, pooler_mask

    class _FakeEmbedVision(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden
            self.calls: List[torch.Tensor] = []

        def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
            self.calls.append(inputs_embeds.clone())
            return inputs_embeds  # identity projection

    class _LM:
        def __init__(self, hidden):
            self.config = SimpleNamespace(hidden_size=hidden)
            self.device = torch.device("cpu")

        def dtype(self):
            return torch.float32

    text_config = SimpleNamespace(hidden_size=hidden_size)
    config = SimpleNamespace(text_config=text_config)

    # Default to an effectively unbounded budget so batching runs; the
    # `encoder_max_batch` kwarg overrides it to exercise chunking.
    if encoder_max_batch is None:
        budget = 1 << 40
        per_patch = 1
    else:
        budget = encoder_max_batch
        per_patch = 1

    fake = SimpleNamespace(
        config=config,
        vision_tower=_FakeTower(),
        embed_vision=_FakeEmbedVision(hidden_size),
        language_model=_LM(hidden_size),
        _encoder_budget_bytes=budget,
        _encoder_bytes_per_patch=per_patch,
    )
    # Bind the real (unbound) methods onto the fake instance.
    cls = gemma4_mm_module.Gemma4ForConditionalGeneration
    for name in [
        "_flatten_pixel_lists",
        "_batched_encode",
        "_gather_mm_features",
        "_encoder_max_batch",
        "get_image_feature",
        "get_video_feature",
    ]:
        fn = getattr(cls, name)
        setattr(fake, name, fn.__get__(fake, type(fake)))

    fake._fail_pad = fail_pad
    # parameters() is used by the empty-input path; return one tensor.
    fake.parameters = lambda: iter([torch.zeros(1)])
    return fake


def _make_item(num_images: int, num_patches: int):
    """Construct a minimal MultimodalDataItem-like object with `num_images`
    images each shaped (num_patches, 4)."""
    pv_list = [torch.full((num_patches, 4), float(i)) for i in range(num_images)]
    pp_list = [
        torch.arange(num_patches).unsqueeze(-1).repeat(1, 2).float()
        for _ in range(num_images)
    ]
    return SimpleNamespace(feature=pv_list, image_position_ids=pp_list)


def test_single_resolution_single_call():
    fake = _make_fake_model()
    item = _make_item(num_images=6, num_patches=10)
    out = fake.get_image_feature([item])

    # 1 encoder forward over [6, 10, 4]
    assert len(fake.vision_tower.calls) == 1, fake.vision_tower.calls
    pv, _ = fake.vision_tower.calls[0]
    assert pv.shape == (6, 10, 4)

    # 1 batched embedder call over (1, 60, 16)
    assert len(fake.embed_vision.calls) == 1
    assert fake.embed_vision.calls[0].shape == (1, 60, 16)

    # Output is (60, 16): 6 images × 10 valid patches × hidden 16
    assert out.shape == (60, 16)


def test_mixed_resolution_bucketing():
    fake = _make_fake_model()
    # 2 small images (5 patches each) and 1 big image (12 patches)
    small = _make_item(num_images=2, num_patches=5)
    big = _make_item(num_images=1, num_patches=12)
    fake.get_image_feature([small, big])

    # Two buckets: one for 5 patches (batch=2), one for 12 patches (batch=1).
    assert len(fake.vision_tower.calls) == 2
    shapes = sorted(call[0].shape for call in fake.vision_tower.calls)
    assert shapes == [(1, 12, 4), (2, 5, 4)]

    # Still a single embedder call over all valid tokens.
    assert len(fake.embed_vision.calls) == 1
    total_tokens = 2 * 5 + 1 * 12
    assert fake.embed_vision.calls[0].shape == (1, total_tokens, 16)


def test_chunking_when_max_batch_set():
    # With per_patch=1 and patches=2, cost-per-item = 2.
    # budget=4 -> 4//2 = 2 items per chunk; 6 items -> 3 encoder calls.
    fake = _make_fake_model(encoder_max_batch=4)
    item = _make_item(num_images=6, num_patches=2)
    fake.get_image_feature([item])
    assert len(fake.vision_tower.calls) == 3
    # Still 1 embedder call.
    assert len(fake.embed_vision.calls) == 1


def test_empty_returns_empty_tensor():
    fake = _make_fake_model()
    out = fake.get_image_feature([])
    assert out.shape == (0, 16)


def test_prepass_real_interleave_preserves_order():
    """A prepass (already-embedded) entry between two raw-pixel entries must
    stay in walk order, not be hoisted to the front. The prepass rows carry a
    sentinel value (99) the tower never produces.
    """
    fake = _make_fake_model(hidden_size=16)

    real0 = torch.zeros(4, 4)  # raw pixels, image 0
    prepass = torch.full((3, 16), 99.0)  # already at hidden_size -> prepass
    real1 = torch.ones(4, 4)  # raw pixels, image 1

    pp = torch.arange(4).unsqueeze(-1).repeat(1, 2).float()
    item = SimpleNamespace(
        feature=[real0, prepass, real1],
        image_position_ids=[pp, None, pp],
    )

    out = fake.get_image_feature([item])

    # 4 (real0) + 3 (prepass) + 4 (real1) = 11 tokens, in that order.
    assert out.shape == (11, 16)
    # The 3 prepass rows must sit in the middle (rows 4..7), not at the front.
    assert torch.all(out[4:7] == 99.0), out
    # Surrounding rows are real-image outputs (never the 99 sentinel).
    assert not torch.any(out[:4] == 99.0)
    assert not torch.any(out[7:] == 99.0)


def test_lazy_budget_is_device_agnostic():
    """The budget is computed lazily from a device-agnostic memory query, so
    batching stays active on non-CUDA devices (here: CPU) instead of falling
    back to single-item batches.
    """
    fake = _make_fake_model(hidden_size=16)
    # Force the lazy-init path: zero budget, real per-patch cost.
    fake._encoder_budget_bytes = 0
    fake._encoder_bytes_per_patch = 1

    max_batch = fake._encoder_max_batch(patches_per_item=4)
    assert fake._encoder_budget_bytes > 0, "expected a device-agnostic budget"
    assert max_batch > 1, "budget should permit batching, not fall back to 1"

    item = _make_item(num_images=6, num_patches=4)
    fake.get_image_feature([item])
    assert len(fake.vision_tower.calls) == 1, fake.vision_tower.calls


if __name__ == "__main__":
    test_single_resolution_single_call()
    test_mixed_resolution_bucketing()
    test_chunking_when_max_batch_set()
    test_empty_returns_empty_tensor()
    test_prepass_real_interleave_preserves_order()
    test_lazy_budget_is_device_agnostic()
    print("ALL TESTS PASSED")
