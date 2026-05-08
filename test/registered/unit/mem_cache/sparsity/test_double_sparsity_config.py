"""Unit tests for Double Sparsity calibration config parsing & validation."""

import json
import unittest
from pathlib import Path

from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityRuntimeConfig,
    parse_calibration_dict,
    parse_calibration_file,
    validate_against_model,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


FIXTURE_PATH = Path(__file__).parent / "_fixtures" / "tiny_ds_calibration.json"


def _load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestParseCalibration(CustomTestCase):
    def test_parse_fixture(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        self.assertEqual(calib.schema_version, 1)
        self.assertEqual(calib.head_dim, 16)
        self.assertEqual(calib.num_layers, 4)
        self.assertEqual(calib.num_heads, 8)
        self.assertEqual(calib.num_kv_heads_global, 4)
        self.assertEqual(calib.heavy_channels, 8)
        self.assertEqual(calib.channel_type, "k")
        self.assertEqual(calib.indexing, "global_kv_head_id")
        self.assertEqual(set(calib.channels.keys()), {0, 1, 2, 3})
        for layer_id, t in calib.channels.items():
            self.assertEqual(t.shape, (4, 8))
            self.assertEqual(str(t.dtype), "torch.int32")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_calibration_file("/tmp/__definitely_missing_ds_config.json")

    def test_schema_version_mismatch(self):
        raw = _load_fixture()
        raw["schema_version"] = 99
        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse_calibration_dict(raw)

    def test_unsupported_channel_type(self):
        raw = _load_fixture()
        raw["channel_type"] = "v"
        with self.assertRaisesRegex(ValueError, "channel_type"):
            parse_calibration_dict(raw)

    def test_unsupported_indexing(self):
        raw = _load_fixture()
        raw["indexing"] = "local_kv_head_id"
        with self.assertRaisesRegex(ValueError, "indexing"):
            parse_calibration_dict(raw)

    def test_layer_count_mismatch(self):
        raw = _load_fixture()
        raw["channels"].pop("3")
        with self.assertRaisesRegex(ValueError, "channels has"):
            parse_calibration_dict(raw)

    def test_heads_per_layer_mismatch(self):
        raw = _load_fixture()
        raw["channels"]["0"] = raw["channels"]["0"][:2]  # 2 rows instead of 4
        with self.assertRaisesRegex(ValueError, "expected 4"):
            parse_calibration_dict(raw)

    def test_heavy_channels_row_length_mismatch(self):
        raw = _load_fixture()
        raw["channels"]["0"][0] = list(range(7))  # length 7 vs heavy_channels=8
        with self.assertRaisesRegex(ValueError, "row length"):
            parse_calibration_dict(raw)

    def test_index_out_of_range(self):
        raw = _load_fixture()
        raw["channels"]["0"][0] = [0, 1, 2, 3, 4, 5, 6, 99]  # 99 >= head_dim=16
        with self.assertRaisesRegex(ValueError, "out of range"):
            parse_calibration_dict(raw)

    def test_duplicate_channel_indices(self):
        raw = _load_fixture()
        raw["channels"]["0"][0] = [0, 0, 1, 2, 3, 4, 5, 6]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_calibration_dict(raw)

    def test_heavy_channels_exceeds_head_dim(self):
        raw = _load_fixture()
        raw["heavy_channels"] = 17  # > head_dim=16
        with self.assertRaisesRegex(ValueError, "heavy_channels"):
            parse_calibration_dict(raw)

    def test_num_heads_not_divisible_by_num_kv_heads(self):
        raw = _load_fixture()
        raw["num_heads"] = 7  # 7 not divisible by 4
        with self.assertRaisesRegex(ValueError, "divisible"):
            parse_calibration_dict(raw)

    def test_num_heads_less_than_num_kv_heads(self):
        raw = _load_fixture()
        raw["num_heads"] = 2
        raw["num_kv_heads"] = 4
        with self.assertRaisesRegex(ValueError, "must be >= num_kv_heads"):
            parse_calibration_dict(raw)


class TestValidateAgainstModel(CustomTestCase):
    def setUp(self):
        self.calib = parse_calibration_file(FIXTURE_PATH)

    def test_match(self):
        validate_against_model(
            self.calib,
            head_dim=16,
            num_layers=4,
            num_heads=8,
            num_kv_heads_global=4,
        )

    def test_head_dim_mismatch(self):
        with self.assertRaisesRegex(ValueError, "head_dim mismatch"):
            validate_against_model(
                self.calib,
                head_dim=128,
                num_layers=4,
                num_heads=8,
                num_kv_heads_global=4,
            )

    def test_num_layers_mismatch(self):
        with self.assertRaisesRegex(ValueError, "num_layers mismatch"):
            validate_against_model(
                self.calib,
                head_dim=16,
                num_layers=80,
                num_heads=8,
                num_kv_heads_global=4,
            )

    def test_num_kv_heads_mismatch(self):
        with self.assertRaisesRegex(ValueError, "num_kv_heads mismatch"):
            validate_against_model(
                self.calib,
                head_dim=16,
                num_layers=4,
                num_heads=8,
                num_kv_heads_global=8,
            )


class TestRuntimeConfig(CustomTestCase):
    def _make(self, **overrides) -> DoubleSparsityRuntimeConfig:
        defaults = dict(
            heavy_channels=8,
            token_budget=64,
            recent_tokens=4,
            sink_tokens=4,
            min_seq_len=128,
            max_selected_per_request=256,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
        )
        defaults.update(overrides)
        return DoubleSparsityRuntimeConfig(**defaults)

    def test_defaults_validate(self):
        self._make().validate()

    def test_recent_tokens_must_be_at_least_one(self):
        with self.assertRaisesRegex(ValueError, "recent_tokens"):
            self._make(recent_tokens=0).validate()

    def test_min_seq_len_must_fit_max_selected(self):
        with self.assertRaisesRegex(ValueError, "max_selected_per_request"):
            self._make(min_seq_len=300, max_selected_per_request=256).validate()

    def test_unknown_gqa_reduction(self):
        with self.assertRaisesRegex(ValueError, "gqa_reduction"):
            self._make(gqa_reduction="median").validate()

    def test_unknown_klabel_dtype(self):
        with self.assertRaisesRegex(ValueError, "klabel_dtype"):
            self._make(klabel_dtype="fp8").validate()

    def test_zero_token_budget(self):
        with self.assertRaisesRegex(ValueError, "token_budget"):
            self._make(token_budget=0).validate()


if __name__ == "__main__":
    unittest.main()
