import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import (
    _validate_dsv41_encoder_only_ratio_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _official_config(**overrides):
    values = dict(
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDsv41EncoderOnlyRatioLayout(unittest.TestCase):
    def test_accepts_official_target_plus_bundled_mtp_layout(self):
        _validate_dsv41_encoder_only_ratio_layout(_official_config())

    def test_rejects_missing_bundled_mtp_ratios(self):
        with self.assertRaisesRegex(ValueError, "expected 40\\+3"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=[0, 0] + [2] * 18 + [1] * 20)
            )

    def test_rejects_nonzero_bundled_mtp_ratio(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "MTP layers to have ratio 0"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=ratios)
            )

    def test_rejects_invalid_target_ratio(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        ratios[10] = 4
        with self.assertRaisesRegex(ValueError, "target-layer ratio-0/1/2"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=ratios)
            )

    def test_rejects_nonproducer_boundary(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        ratios[20] = 0
        with self.assertRaisesRegex(ValueError, "layer 20"):
            _validate_dsv41_encoder_only_ratio_layout(
                _official_config(compress_ratios=ratios)
            )


if __name__ == "__main__":
    unittest.main()
