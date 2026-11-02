"""WeightChecker checksum on a non-CUDA device (CPU stands in for TPU)."""

import hashlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.utils.weight_checker import ChecksumInfo, WeightChecker
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


if __name__ == "__main__":
    unittest.main()
