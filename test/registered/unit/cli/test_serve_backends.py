# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import shlex
import unittest
from unittest.mock import MagicMock, patch

from sglang.cli.serve import serve
from sglang.cli.serve_backends import (
    SERVE_BACKEND_API_VERSION,
    RegisteredServeBackend,
    ServeBackend,
    ServeBackendDetection,
    ServeBackendRegistry,
    ServeRequest,
)
from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_entry_point(name, factory, distribution=None):
    entry_point = MagicMock()
    entry_point.name = name
    entry_point.value = f"fake_{name}:create_backend"
    entry_point.load.return_value = factory
    if distribution is None:
        entry_point.dist = None
    else:
        entry_point.dist = MagicMock()
        entry_point.dist.name = distribution
    return entry_point


def _backend(*, detector=None, requires_model_path=True, api_version=None):
    if api_version is None:
        api_version = SERVE_BACKEND_API_VERSION
    return ServeBackend(
        api_version=api_version,
        run=MagicMock(),
        detect=detector,
        requires_model_path=requires_model_path,
    )


class TestServeBackendRegistry(unittest.TestCase):
    @patch("sglang.cli.serve_backends.entry_points")
    def test_listing_does_not_import_out_of_tree_backends(self, mock_entry_points):
        factory = MagicMock(return_value=_backend())
        entry_point = _make_entry_point("omni", factory)
        mock_entry_points.return_value = [entry_point]

        registry = ServeBackendRegistry({"llm": _backend()})

        self.assertEqual(registry.available_names, ("llm", "omni"))
        entry_point.load.assert_not_called()
        factory.assert_not_called()

    @patch("sglang.cli.serve_backends.entry_points")
    def test_explicit_selection_only_loads_selected_backend(self, mock_entry_points):
        omni_entry_point = _make_entry_point(
            "omni", MagicMock(return_value=_backend()), "sglang-omni"
        )
        other_entry_point = _make_entry_point(
            "other", MagicMock(return_value=_backend()), "other-project"
        )
        mock_entry_points.return_value = [omni_entry_point, other_entry_point]

        registry = ServeBackendRegistry({"llm": _backend()})
        registered = registry.get("omni")

        self.assertEqual(registered.name, "omni")
        self.assertEqual(registered.distribution, "sglang-omni")
        omni_entry_point.load.assert_called_once_with()
        other_entry_point.load.assert_not_called()

    @patch("sglang.cli.serve_backends.entry_points")
    def test_auto_detection_selects_one_unique_match(self, mock_entry_points):
        matching = _backend(
            detector=MagicMock(return_value=ServeBackendDetection.MATCH)
        )
        non_matching = _backend(
            detector=MagicMock(return_value=ServeBackendDetection.NO_MATCH)
        )
        mock_entry_points.return_value = [
            _make_entry_point("omni", lambda: matching),
            _make_entry_point("speech", lambda: non_matching),
        ]
        registry = ServeBackendRegistry({"llm": _backend()})
        request = ServeRequest(argv=("--model-path", "model"), model_path="model")

        self.assertEqual(registry.auto_detect(request).name, "omni")

    @patch("sglang.cli.serve_backends.entry_points")
    def test_auto_detection_rejects_ambiguous_matches(self, mock_entry_points):
        mock_entry_points.return_value = [
            _make_entry_point(
                "omni",
                lambda: _backend(
                    detector=MagicMock(return_value=ServeBackendDetection.MATCH)
                ),
            ),
            _make_entry_point(
                "speech",
                lambda: _backend(
                    detector=MagicMock(return_value=ServeBackendDetection.MATCH)
                ),
            ),
        ]
        registry = ServeBackendRegistry({"llm": _backend()})
        request = ServeRequest(argv=("--model-path", "model"), model_path="model")

        with self.assertRaisesRegex(RuntimeError, "Multiple serve backends matched"):
            registry.auto_detect(request)

    @patch("sglang.cli.serve_backends.entry_points")
    def test_broken_optional_detector_does_not_block_llm_fallback(
        self, mock_entry_points
    ):
        broken_entry_point = _make_entry_point("broken", MagicMock())
        broken_entry_point.load.side_effect = ImportError("optional dependency missing")
        mock_entry_points.return_value = [broken_entry_point]
        llm = _backend()
        registry = ServeBackendRegistry({"llm": llm})
        request = ServeRequest(argv=("--model-path", "model"), model_path="model")

        with self.assertLogs("sglang.cli.serve_backends", level="WARNING"):
            selected = registry.auto_detect(request)

        self.assertIs(selected.backend, llm)
        with self.assertRaisesRegex(RuntimeError, "Failed to load serve backend"):
            registry.get("broken")

    @patch("sglang.cli.serve_backends.entry_points")
    def test_duplicate_provider_names_are_rejected_when_selected(
        self, mock_entry_points
    ):
        mock_entry_points.return_value = [
            _make_entry_point("omni", lambda: _backend(), "provider-a"),
            _make_entry_point("omni", lambda: _backend(), "provider-b"),
        ]
        registry = ServeBackendRegistry({"llm": _backend()})

        with self.assertRaisesRegex(RuntimeError, "provider-a, provider-b"):
            registry.get("omni")

    @patch("sglang.cli.serve_backends.entry_points")
    def test_out_of_tree_backend_cannot_replace_builtin(self, mock_entry_points):
        mock_entry_points.return_value = [
            _make_entry_point("llm", lambda: _backend(), "bad-provider")
        ]

        with self.assertRaisesRegex(RuntimeError, "cannot replace"):
            ServeBackendRegistry({"llm": _backend()})

    @patch("sglang.cli.serve_backends.entry_points")
    def test_backend_api_version_is_validated(self, mock_entry_points):
        mock_entry_points.return_value = [
            _make_entry_point(
                "future",
                lambda: _backend(api_version=SERVE_BACKEND_API_VERSION + 1),
            )
        ]
        registry = ServeBackendRegistry({"llm": _backend()})

        with self.assertRaisesRegex(RuntimeError, "uses API version"):
            registry.get("future")


class TestServeBackendDispatch(unittest.TestCase):
    @patch("sglang.cli.serve.kill_process_tree")
    @patch("sglang.srt.plugins.load_plugins")
    @patch("sglang.cli.serve._create_backend_registry")
    def test_explicit_backend_receives_normalized_model_path(
        self, mock_registry_factory, mock_load_plugins, mock_kill
    ):
        backend = _backend()
        registry = MagicMock()
        registry.get.return_value = RegisteredServeBackend("omni", backend)
        mock_registry_factory.return_value = registry

        serve(
            None,
            ["Example/Model", "--model-type", "omni", "--pipeline-parallel", "2"],
        )

        request = backend.run.call_args.args[0]
        self.assertEqual(
            request.argv,
            ("--model-path", "Example/Model", "--pipeline-parallel", "2"),
        )
        self.assertEqual(request.model_path, "Example/Model")
        self.assertTrue(request.model_path_is_positional)
        mock_load_plugins.assert_called_once_with()
        mock_kill.assert_called_once()

    @patch("sglang.cli.serve.kill_process_tree")
    @patch("sglang.srt.plugins.load_plugins")
    @patch("sglang.cli.serve._create_backend_registry")
    def test_explicit_backend_can_support_config_only_requests(
        self, mock_registry_factory, _mock_load_plugins, _mock_kill
    ):
        backend = _backend(requires_model_path=False)
        registry = MagicMock()
        registry.get.return_value = RegisteredServeBackend("pipeline", backend)
        mock_registry_factory.return_value = registry

        serve(None, ["--model-type=pipeline", "--config", "pipeline.yaml"])

        request = backend.run.call_args.args[0]
        self.assertIsNone(request.model_path)
        self.assertEqual(request.argv, ("--config", "pipeline.yaml"))

    @patch("sglang.cli.serve.kill_process_tree")
    @patch("sglang.srt.plugins.load_plugins")
    @patch("sglang.cli.serve._create_backend_registry")
    def test_auto_detection_uses_registry(
        self, mock_registry_factory, _mock_load_plugins, _mock_kill
    ):
        backend = _backend()
        registry = MagicMock()
        registry.auto_detect.return_value = RegisteredServeBackend("omni", backend)
        mock_registry_factory.return_value = registry

        serve(None, ["Example/Model"])

        registry.auto_detect.assert_called_once()
        backend.run.assert_called_once()

    @patch("sglang.cli.serve.kill_process_tree")
    @patch("sglang.srt.plugins.load_plugins")
    @patch("sglang.cli.serve._create_backend_registry")
    def test_targeted_help_is_forwarded_without_startup_or_model_path(
        self, mock_registry_factory, mock_load_plugins, mock_kill
    ):
        backend = _backend()
        registry = MagicMock()
        registry.get.return_value = RegisteredServeBackend("omni", backend)
        mock_registry_factory.return_value = registry

        serve(None, ["--model-type", "omni", "--help"])

        request = backend.run.call_args.args[0]
        self.assertEqual(request.argv, ("--help",))
        self.assertIsNone(request.model_path)
        mock_load_plugins.assert_not_called()
        mock_kill.assert_not_called()


DIFFUSION_MODELS = {"black-forest-labs/FLUX.1-dev"}


def _cookbook_argv(command):
    lines = [line.rstrip(" \\") for line in command.strip().splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    start = next(i for i, line in enumerate(lines) if line.startswith("sglang serve"))
    return shlex.split(" ".join(lines[start:]))[2:]


DEEPSEEK_V4_H200 = r"""
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --mem-fraction-static 0.90 \
  --host 0.0.0.0 \
  --port 30000
"""

QWEN38_GB300_MULTI_NODE = r"""
SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
EP_DISABLE_GIN=1 \
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --tp-size 16 \
  --attn-dp-size 4 \
  --nnodes 4 \
  --node-rank <node-rank> \
  --dist-init-addr <node0-ip>:20000 \
  --ep-size 16 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 30000
"""

KIMI_K3_B200 = r"""
sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --host 0.0.0.0 \
  --port 30000
"""

DIRECT_LAUNCH_CASES = {
    "docs quickstart, positional model": (
        ["Qwen/Qwen3-0.6B", "--host", "127.0.0.1", "--port", "30000"],
        "llm",
        ["--model-path", "Qwen/Qwen3-0.6B", "--host", "127.0.0.1", "--port", "30000"],
    ),
    "CI popen_launch_server": (
        ["--model-path", "meta-llama/Llama-3.1-8B-Instruct"]
        + ["--host", "127.0.0.1", "--port", "21000"],
        "llm",
        None,
    ),
    "--model-path=value": (
        ["--model-path=Qwen/Qwen3-8B", "--tp-size", "2"],
        "llm",
        None,
    ),
    "--model alias": (["--model", "Qwen/Qwen3-8B", "--tp", "2"], "llm", None),
    "explicit llm backend": (
        ["Qwen/Qwen3-8B", "--model-type", "llm", "--tp-size", "2"],
        "llm",
        ["--model-path", "Qwen/Qwen3-8B", "--tp-size", "2"],
    ),
    "explicit backend, = form": (
        ["--model-type=llm", "--model-path", "Qwen/Qwen3-8B"],
        "llm",
        ["--model-path", "Qwen/Qwen3-8B"],
    ),
    "diffusion model detected": (
        ["black-forest-labs/FLUX.1-dev", "--port", "30000"],
        "diffusion",
        ["--model-path", "black-forest-labs/FLUX.1-dev", "--port", "30000"],
    ),
    "explicit diffusion backend": (
        ["black-forest-labs/flux-3-action-droid", "--model-type", "diffusion"],
        "diffusion",
        ["--model-path", "black-forest-labs/flux-3-action-droid"],
    ),
    "cookbook DeepSeek-V4 on H200": (_cookbook_argv(DEEPSEEK_V4_H200), "llm", None),
    "cookbook Qwen3.8 on GB300, multi-node": (
        _cookbook_argv(QWEN38_GB300_MULTI_NODE),
        "llm",
        None,
    ),
    "cookbook Kimi-K3 on B200, JSON flag": (_cookbook_argv(KIMI_K3_B200), "llm", None),
}

# TODO(Yichi): expect the CLI Guide for these once it lands.
GUIDE_CASES = {
    "bare model": (["Qwen/Qwen3-8B"], "llm", ["--model-path", "Qwen/Qwen3-8B"]),
    "bare model with --mode": (
        ["Qwen/Qwen3-8B", "--mode", "balanced"],
        "llm",
        ["--model-path", "Qwen/Qwen3-8B", "--mode", "balanced"],
    ),
}


class TestServeDirectLaunch(CustomTestCase):
    def _serve(self, argv):
        ran = []

        def capture(name):
            return lambda request: ran.append((name, request.argv))

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("sglang.cli.serve._run_llm", capture("llm")))
            stack.enter_context(
                patch("sglang.cli.serve._run_diffusion", capture("diffusion"))
            )
            stack.enter_context(
                patch(
                    "sglang.cli.serve.get_is_diffusion_model",
                    lambda path: path in DIFFUSION_MODELS,
                )
            )
            stack.enter_context(
                patch("sglang.cli.serve_backends.entry_points", return_value=[])
            )
            stack.enter_context(patch("sglang.srt.plugins.load_plugins"))
            stack.enter_context(patch("sglang.cli.serve.kill_process_tree"))
            serve(None, list(argv))

        self.assertEqual(len(ran), 1)
        return ran[0]

    def _assert_cases(self, cases):
        for name, (argv, backend, received) in cases.items():
            with self.subTest(name):
                expected = tuple(argv if received is None else received)
                self.assertEqual(self._serve(argv), (backend, expected))

    def test_direct_launch_reaches_backend_unchanged(self):
        self._assert_cases(DIRECT_LAUNCH_CASES)

    def test_guide_candidates_launch_directly_today(self):
        self._assert_cases(GUIDE_CASES)

    def test_mode_is_not_a_server_flag(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_server_args(["--model-path", "Qwen/Qwen3-8B", "--mode", "balanced"])

    # TODO(Yichi): expect the CLI Guide's model prompt once it lands.
    def test_missing_model_is_rejected(self):
        with self.assertRaisesRegex(Exception, "--model-path is required"):
            self._serve([])


if __name__ == "__main__":
    unittest.main()
