"""Unit test for the weight-update post_load_weights hook — CPU-only.

`update_weights_from_tensor` writes parameters via the "direct", custom-loader,
or `model.load_weights` (None) formats, none of which run the loader's post-load
step. Models that rebuild derived weight caches in `post_load_weights`
(fp32-transposed router gates, repacked quant operands, …) would then serve
stale derived weights after an in-place update. The fix calls `post_load_weights`
once after all three load paths when the model exposes it.

The WeightUpdater method is device-agnostic here, so the test runs on CPU with a
tiny hand-built model — no GPU or checkpoint required.
"""

import unittest

import torch
import torch.nn as nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DerivedCacheModel(nn.Module):
    """Model whose derived buffer must track the base weight (w_derived = 2*w)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 4))
        self.register_buffer("w_derived", torch.empty(4, 4))
        self.post_load_calls = 0
        self._rebuild()

    def _rebuild(self):
        self.w_derived.copy_(self.w.detach() * 2)

    def load_weights(self, named_tensors):
        # Mirrors real models: load_weights copies params but does NOT itself
        # run post_load_weights (the loader wrapper normally does that).
        params = dict(self.named_parameters())
        for name, tensor in named_tensors:
            params[name].data.copy_(tensor)

    def post_load_weights(self):
        self.post_load_calls += 1
        self._rebuild()


class _PlainModel(nn.Module):
    """No post_load_weights hook — the fix must be a no-op, not a crash."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 4))


def _make_updater(model):
    # WeightUpdater is a frozen dataclass; build it via the constructor and
    # stub the fields the direct-load path never touches.
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: None,
    )


class TestWeightUpdaterPostLoadWeightsHook(CustomTestCase):
    def test_direct_load_refreshes_derived_cache(self):
        model = _DerivedCacheModel()
        updater = _make_updater(model)
        new_w = torch.full((4, 4), 1.5)

        ok, _ = updater.update_weights_from_tensor([("w", new_w)], load_format="direct")

        self.assertTrue(ok)
        # base parameter updated by the direct load
        torch.testing.assert_close(model.w.detach(), new_w)
        # hook ran, so the derived cache tracks the new weights (was stale before)
        self.assertEqual(model.post_load_calls, 1)
        torch.testing.assert_close(model.w_derived, new_w * 2)

    def test_none_format_refreshes_derived_cache(self):
        # load_format=None calls model.load_weights directly, which also skips
        # post_load_weights — the fix must refresh the derived cache here too.
        model = _DerivedCacheModel()
        updater = _make_updater(model)
        new_w = torch.full((4, 4), 2.5)

        ok, _ = updater.update_weights_from_tensor([("w", new_w)], load_format=None)

        self.assertTrue(ok)
        torch.testing.assert_close(model.w.detach(), new_w)
        self.assertEqual(model.post_load_calls, 1)
        torch.testing.assert_close(model.w_derived, new_w * 2)

    def test_model_without_hook_is_noop(self):
        model = _PlainModel()
        updater = _make_updater(model)
        new_w = torch.full((4, 4), 2.0)

        ok, _ = updater.update_weights_from_tensor([("w", new_w)], load_format="direct")

        self.assertTrue(ok)
        torch.testing.assert_close(model.w.detach(), new_w)


if __name__ == "__main__":
    unittest.main()
