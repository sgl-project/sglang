"""Unit tests for the MLX remote-code gate.

mlx-lm executes ``config.json``'s ``model_file`` unconditionally at load
time, so SGLang refuses such checkpoints before any checkpoint Python runs
unless the server was started with ``--trust-remote-code``. The refusal
tests prove non-execution with a sentinel ``model_file`` whose import would
leave an observable marker.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)

from sglang.srt.hardware_backend.mlx.remote_code_gate import (  # noqa: E402
    RemoteCodeGateError,
    ensure_remote_code_allowed,
)

_SENTINEL = "GATE FAILED: checkpoint python executed"


def _make_checkpoint(tmp: Path, config: dict, *, with_sentinel: bool = True) -> Path:
    (tmp / "config.json").write_text(json.dumps(config))
    if with_sentinel:
        # Importing this file would create marker.txt — the refusal tests
        # assert it never appears.
        (tmp / "evil.py").write_text(
            "from pathlib import Path\n"
            f"Path(__file__).parent.joinpath('marker.txt').write_text({_SENTINEL!r})\n"
        )
    return tmp


class TestEnsureRemoteCodeAllowed(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_sentinel_not_executed(self):
        self.assertFalse(
            (self.dir / "marker.txt").exists(),
            "checkpoint python executed despite gate refusal",
        )

    def test_refuses_model_file_without_trust(self):
        _make_checkpoint(
            self.dir, {"model_type": "muse_glimmer", "model_file": "evil.py"}
        )
        with self.assertRaisesRegex(RemoteCodeGateError, "--trust-remote-code"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=False)
        self._assert_sentinel_not_executed()

    def test_allows_model_file_with_trust(self):
        _make_checkpoint(
            self.dir, {"model_type": "muse_glimmer", "model_file": "evil.py"}
        )
        ensure_remote_code_allowed(self.dir, trust_remote_code=True)
        # The gate itself never imports the file either way.
        self._assert_sentinel_not_executed()

    def test_builtin_checkpoint_passes_without_trust(self):
        _make_checkpoint(self.dir, {"model_type": "qwen3"}, with_sentinel=False)
        ensure_remote_code_allowed(self.dir, trust_remote_code=False)

    def test_missing_config_rejected(self):
        with self.assertRaisesRegex(RemoteCodeGateError, "no config.json"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)

    def test_malformed_config_rejected(self):
        (self.dir / "config.json").write_text("{not json")
        with self.assertRaisesRegex(RemoteCodeGateError, "not valid JSON"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)

    def test_non_object_config_rejected(self):
        (self.dir / "config.json").write_text('["a", "b"]')
        with self.assertRaisesRegex(RemoteCodeGateError, "JSON object"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)

    def test_missing_model_file_target_rejected(self):
        _make_checkpoint(
            self.dir,
            {"model_type": "muse_glimmer", "model_file": "nope.py"},
            with_sentinel=False,
        )
        with self.assertRaisesRegex(RemoteCodeGateError, "does not exist"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)

    def test_absolute_model_file_rejected(self):
        _make_checkpoint(
            self.dir,
            {"model_type": "muse_glimmer", "model_file": "/etc/anything.py"},
            with_sentinel=False,
        )
        with self.assertRaisesRegex(RemoteCodeGateError, "relative path"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)

    def test_traversal_model_file_rejected(self):
        _make_checkpoint(
            self.dir,
            {"model_type": "muse_glimmer", "model_file": "../outside.py"},
            with_sentinel=False,
        )
        with self.assertRaisesRegex(RemoteCodeGateError, "relative path"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)

    def test_non_string_model_file_rejected(self):
        _make_checkpoint(
            self.dir,
            {"model_type": "muse_glimmer", "model_file": 42},
            with_sentinel=False,
        )
        with self.assertRaisesRegex(RemoteCodeGateError, "non-string"):
            ensure_remote_code_allowed(self.dir, trust_remote_code=True)


@unittest.skipUnless(_HAS_MLX, "requires mlx + mlx_lm")
class TestModelRunnerGateWiring(CustomTestCase):
    """The runner must gate BEFORE calling mlx_lm's loader, on the same
    resolved directory it then loads from."""

    class _StopInit(Exception):
        pass

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_refusal_precedes_loader_call(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        _make_checkpoint(
            self.dir, {"model_type": "muse_glimmer", "model_file": "evil.py"}
        )
        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.mlx_lm_load"
        ) as loader:
            with self.assertRaisesRegex(RemoteCodeGateError, "--trust-remote-code"):
                MlxModelRunner(model_path=str(self.dir), trust_remote_code=False)
            loader.assert_not_called()
        self.assertFalse((self.dir / "marker.txt").exists())

    def test_trusted_load_uses_resolved_directory(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        _make_checkpoint(
            self.dir, {"model_type": "muse_glimmer", "model_file": "evil.py"}
        )
        with patch(
            "sglang.srt.hardware_backend.mlx.model_runner.mlx_lm_load",
            side_effect=self._StopInit,
        ) as loader:
            with self.assertRaises(self._StopInit):
                MlxModelRunner(model_path=str(self.dir), trust_remote_code=True)
            loader.assert_called_once()
            called_path = loader.call_args.args[0]
            self.assertEqual(
                Path(called_path).resolve(),
                self.dir.resolve(),
                "loader must receive the same directory the gate inspected",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
