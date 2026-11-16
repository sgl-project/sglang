"""WeightChecker checksum on a non-CUDA device (CPU stands in for TPU)."""

import hashlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_executor.weight_checker import ChecksumInfo, WeightChecker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChecksumWithoutCuda(CustomTestCase):
    def test_checksum_does_not_require_cuda(self):
        """The checksum action must work on hosts without CUDA, e.g. TPU engines
        checked by an RL trainer after every weight update."""
        model = nn.Linear(4, 4)
        enter_scope(self, published_topology())
        checker = WeightChecker(get_model=lambda: model)
        with patch.object(
            torch.cuda, "synchronize", side_effect=RuntimeError("no CUDA device")
        ):
            out = checker.handle("checksum", role="target")
        info = ChecksumInfo.model_validate(out)
        self.assertEqual(set(info.checksums), {"weight", "bias"})


class TestRawChecksumWithoutCuda(CustomTestCase):
    def test_raw_checksum_includes_loaded_buffers(self) -> None:
        """Hash persistent buffers while excluding derived rotary caches."""
        for names in (None, ["layer_scalar"]):
            with self.subTest(names=names):
                model = nn.Linear(2, 2)
                scalar = torch.tensor([1.5])
                model.register_buffer("layer_scalar", scalar)
                model.register_buffer("cos_sin_cache", torch.ones(2), persistent=False)
                with published_topology():
                    checker = WeightChecker(get_model=lambda: model)
                    result = checker.handle(
                        action="raw_checksum", names=names, role="target"
                    )

                expected_names = (
                    {"weight", "bias", "layer_scalar"} if names is None else set(names)
                )
                self.assertEqual(set(result["checksums"]), expected_names)
                self.assertEqual(
                    result["checksums"]["layer_scalar"],
                    hashlib.sha256(
                        scalar.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                )

    def test_missing_names_report_runner_and_all_missing_tensors(self) -> None:
        """Report unavailable names explicitly instead of leaking a bare KeyError."""
        with published_topology():
            checker = WeightChecker(get_model=lambda: nn.Linear(2, 2))
            with self.assertRaisesRegex(
                ValueError, "draft runner.*missing_a.*missing_b"
            ):
                checker.handle(
                    action="raw_checksum",
                    names=["weight", "missing_b", "missing_a"],
                    role="draft",
                )


class _ModelWithNonPersistentBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.rotary_emb = nn.Module()
        # Same shape of buffer as MRotaryEmbedding.axis_map: an index map that
        # is built at init and never sent by a weight update.
        self.rotary_emb.register_buffer(
            "axis_map", torch.tensor([0, 1, 2, 0, 1, 2]), persistent=False
        )
        self.register_buffer("token_map", torch.arange(8), persistent=False)
        self.register_buffer("running_mean", torch.zeros(4))
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)


class TestResetTensorsKeepsNonPersistentBuffers(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _ModelWithNonPersistentBuffers()
        enter_scope(self, published_topology())
        self.checker = WeightChecker(get_model=lambda: self.model)

    def test_reset_keeps_non_persistent_buffers(self):
        """reset_tensors must not write random values into buffers that are not
        in the state dict, because a weight update cannot restore them."""
        axis_map = self.model.rotary_emb.axis_map.clone()
        token_map = self.model.token_map.clone()
        self.checker.handle("reset_tensors", role="target")
        self.assertTrue(torch.equal(self.model.rotary_emb.axis_map, axis_map))
        self.assertTrue(torch.equal(self.model.token_map, token_map))

    def test_reset_still_changes_params_and_persistent_buffers(self):
        weight = self.model.linear.weight.clone()
        running_mean = self.model.running_mean.clone()
        self.checker.handle("reset_tensors", role="target")
        self.assertFalse(torch.equal(self.model.linear.weight, weight))
        self.assertFalse(torch.equal(self.model.running_mean, running_mean))

    def test_compare_passes_after_restoring_state_dict(self):
        """Simulate a weight update: it restores only the state dict."""
        self.checker.handle("snapshot", role="target")
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.checker.handle("reset_tensors", role="target")
        self.model.load_state_dict(state)
        self.checker.handle("compare", role="target")


if __name__ == "__main__":
    unittest.main()
