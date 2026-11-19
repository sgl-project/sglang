"""CPU-only tests for --weight-cache-max-bytes enforcement."""

import argparse
import json
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.weight_cache import daemon as daemon_mod
from sglang.srt.weight_cache.daemon import WeightCacheDaemon, estimate_checkpoint_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _bare_daemon(**overrides) -> WeightCacheDaemon:
    daemon = object.__new__(WeightCacheDaemon)
    daemon.gpu_id = 0
    daemon.model = None
    daemon.model_path = "/models/demo"
    daemon.load_format = "auto"
    daemon.revision = None
    daemon.preloaded_weights_bytes = 0
    daemon.weight_cache_max_bytes = None
    daemon.cpu_offload_gb = 0
    daemon.offload_group_size = 0
    daemon.ple_offload_embedding = False
    for name, value in overrides.items():
        setattr(daemon, name, value)
    return daemon


class TestLoadBudget(CustomTestCase):
    def _validate(self, daemon, *, total_bytes=1000, fraction=0.8):
        with (
            mock.patch.object(
                daemon_mod,
                "get_schedule",
                return_value=SimpleNamespace(mem_fraction_static=fraction),
            ),
            mock.patch.object(
                daemon_mod.current_platform,
                "get_device_total_memory",
                return_value=total_bytes,
            ),
        ):
            daemon._validate_loaded_weights()

    def test_unset_cap_keeps_previous_behavior(self):
        daemon = _bare_daemon(preloaded_weights_bytes=10_000)
        daemon._validate_loaded_weights()

    def test_cap_and_budget_accept_the_exact_boundary(self):
        daemon = _bare_daemon(preloaded_weights_bytes=800, weight_cache_max_bytes=800)
        self._validate(daemon)

    def test_cap_rejects_oversized_weights(self):
        daemon = _bare_daemon(preloaded_weights_bytes=801, weight_cache_max_bytes=800)
        with self.assertRaisesRegex(
            RuntimeError, r"801 bytes.*weight-cache-max-bytes.*800 bytes"
        ):
            daemon._validate_loaded_weights()

    def test_rejected_load_releases_model_before_export(self):
        daemon = _bare_daemon(preloaded_weights_bytes=801, weight_cache_max_bytes=800)
        daemon.model = object()
        with mock.patch.object(
            daemon_mod.current_platform, "empty_cache"
        ) as empty_cache:
            with self.assertRaisesRegex(RuntimeError, "weight-cache-max-bytes"):
                daemon._admit_loaded_weights()
        self.assertIsNone(daemon.model)
        empty_cache.assert_called_once()

    def test_static_upper_bound_rejects(self):
        daemon = _bare_daemon(preloaded_weights_bytes=801, weight_cache_max_bytes=10000)
        with self.assertRaisesRegex(RuntimeError, "static upper bound"):
            self._validate(daemon)

    def test_unresolved_fraction_rejects(self):
        daemon = _bare_daemon(preloaded_weights_bytes=1, weight_cache_max_bytes=10000)
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            self._validate(daemon, fraction=None)


def _write_safetensors(path: str, tensors: dict) -> None:
    header = {}
    offset = 0
    for name, nbytes in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": [nbytes // 4],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw_header = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw_header)))
        f.write(raw_header)
        f.write(b"\0" * offset)


class TestCheckpointEstimate(CustomTestCase):
    def test_safetensors_header_sums_tensor_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_safetensors(
                os.path.join(tmp, "model.safetensors"), {"a": 100, "b": 200}
            )
            self.assertEqual(estimate_checkpoint_bytes(tmp), 300)

    def test_index_selects_listed_shards_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_safetensors(
                os.path.join(tmp, "model-00001-of-00002.safetensors"), {"a": 100}
            )
            _write_safetensors(
                os.path.join(tmp, "model-00002-of-00002.safetensors"), {"b": 200}
            )
            _write_safetensors(os.path.join(tmp, "extra.safetensors"), {"unused": 999})
            index = {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
            with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)

            self.assertEqual(estimate_checkpoint_bytes(tmp), 300)

    def test_unrelated_index_cannot_hide_loaded_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_safetensors(os.path.join(tmp, "consolidated.safetensors"), {"a": 4})
            _write_safetensors(os.path.join(tmp, "model-00001.safetensors"), {"b": 200})
            _write_safetensors(os.path.join(tmp, "model-00002.safetensors"), {"c": 200})
            with open(
                os.path.join(tmp, "consolidated.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"a": "consolidated.safetensors"}}, f)
            estimate = estimate_checkpoint_bytes(tmp)
            # AUTO ignores the consolidated index and loads all three files.
            self.assertTrue(estimate is None or estimate >= 404)

    def test_unrecognized_load_format_defers_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_safetensors(os.path.join(tmp, "model.safetensors"), {"a": 4})
            self.assertIsNone(estimate_checkpoint_bytes(tmp, load_format="mistral"))

    def test_missing_index_shard_defers_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_safetensors(os.path.join(tmp, "one.safetensors"), {"a": 4})
            with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
                json.dump(
                    {
                        "weight_map": {
                            "a": "one.safetensors",
                            "b": "missing.safetensors",
                        }
                    },
                    f,
                )
            self.assertIsNone(estimate_checkpoint_bytes(tmp))

    def test_invalid_offsets_fall_back_to_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            header = json.dumps({"w": {"data_offsets": [10, 2]}}).encode()
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", len(header)) + header)
            self.assertEqual(estimate_checkpoint_bytes(tmp), os.path.getsize(path))

    def test_unreadable_header_falls_back_to_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", 10**9))
            self.assertEqual(estimate_checkpoint_bytes(tmp), os.path.getsize(path))

    def test_non_safetensors_file_uses_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pytorch_model.bin")
            with open(path, "wb") as f:
                f.write(b"x" * 123)
            self.assertEqual(estimate_checkpoint_bytes(tmp), 123)

    def test_missing_checkpoint_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing")
            with mock.patch(
                "sglang.srt.utils.common.find_local_repo_dir", return_value=None
            ):
                self.assertIsNone(estimate_checkpoint_bytes(missing))

    def test_unset_cap_does_not_scan_checkpoint(self):
        daemon = _bare_daemon(weight_cache_max_bytes=None)
        with mock.patch.object(daemon_mod, "estimate_checkpoint_bytes") as estimate:
            daemon._enforce_estimated_cap()
        estimate.assert_not_called()

    def test_capped_offloading_rejected_before_checkpoint_scan(self):
        for option, value in (
            ("cpu_offload_gb", 1),
            ("offload_group_size", 1),
            ("ple_offload_embedding", True),
        ):
            with self.subTest(option=option):
                daemon = _bare_daemon(weight_cache_max_bytes=100, **{option: value})
                with mock.patch.object(
                    daemon_mod, "estimate_checkpoint_bytes"
                ) as estimate:
                    with self.assertRaisesRegex(ValueError, "offloading"):
                        daemon._enforce_estimated_cap()
                estimate.assert_not_called()

    def test_estimated_size_at_cap_is_accepted(self):
        daemon = _bare_daemon(weight_cache_max_bytes=100)
        with mock.patch.object(
            daemon_mod, "estimate_checkpoint_bytes", return_value=100
        ) as estimate:
            daemon._enforce_estimated_cap()
        estimate.assert_called_once_with("/models/demo", None, "auto")

    def test_unestimable_checkpoint_defers_to_post_load(self):
        daemon = _bare_daemon(weight_cache_max_bytes=100)
        with (
            mock.patch.object(
                daemon_mod, "estimate_checkpoint_bytes", return_value=None
            ),
            self.assertLogs(daemon_mod.logger, level="WARNING") as logs,
        ):
            daemon._enforce_estimated_cap()

        self.assertIn("enforced after loading", "\n".join(logs.output))

    def test_real_checkpoint_estimate_rejects_over_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_safetensors(os.path.join(tmp, "model.safetensors"), {"w": 101})
            daemon = _bare_daemon(model_path=tmp, weight_cache_max_bytes=100)

            with self.assertRaisesRegex(
                RuntimeError, r"estimated checkpoint size.*101.*100"
            ):
                daemon._enforce_estimated_cap()

    def test_estimated_size_over_cap_rejects_before_gpu_load(self):
        daemon = _bare_daemon(model_path="/models/too-big", weight_cache_max_bytes=100)
        with (
            mock.patch.object(
                daemon_mod.current_platform, "is_cuda_alike", return_value=True
            ),
            mock.patch.object(
                daemon_mod.WeightCacheDaemon, "_assert_ipc_compatible_allocator"
            ),
            mock.patch.object(
                daemon_mod, "estimate_checkpoint_bytes", return_value=101
            ) as estimate,
            mock.patch.object(daemon_mod.current_platform, "set_device") as set_device,
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"estimated checkpoint size.*101.*100"
            ):
                daemon.load()

        estimate.assert_called_once_with("/models/too-big", None, "auto")
        set_device.assert_not_called()
        self.assertIsNone(daemon.model)


class TestCliFlag(CustomTestCase):
    def test_unset_defaults_to_none_and_parses_bytes(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        default = ServerArgs.from_cli_args(parser.parse_args(["--model-path", "dummy"]))
        configured = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model-path",
                    "dummy",
                    "--weight-cache-max-bytes",
                    "1048576",
                ]
            )
        )

        self.assertIsNone(default.weight_cache_max_bytes)
        self.assertEqual(configured.weight_cache_max_bytes, 1048576)


if __name__ == "__main__":
    unittest.main()
