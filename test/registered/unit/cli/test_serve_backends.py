# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import sys
import unittest
from importlib import resources
from unittest.mock import MagicMock, patch

from sglang.cli import interactive
from sglang.cli.interactive import (
    NO_MODE_MESSAGE,
    NO_RECIPE_MESSAGE,
    command_argv,
    edit_flags,
    format_command,
    recipes_for_this_machine,
    run_guide,
)
from sglang.cli.serve import serve
from sglang.cli.serve_backends import (
    SERVE_BACKEND_API_VERSION,
    RegisteredServeBackend,
    ServeBackend,
    ServeBackendDetection,
    ServeBackendRegistry,
    ServeRequest,
)
from sglang.srt.server_args import compute_world_size, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
    "bare model without a Cookbook command": (
        ["Qwen/Qwen3-8B"],
        "llm",
        ["--model-path", "Qwen/Qwen3-8B"],
    ),
    "bare diffusion model": (
        ["black-forest-labs/FLUX.1-dev"],
        "diffusion",
        ["--model-path", "black-forest-labs/FLUX.1-dev"],
    ),
    "explicit diffusion backend": (
        ["black-forest-labs/flux-3-action-droid", "--model-type", "diffusion"],
        "diffusion",
        ["--model-path", "black-forest-labs/flux-3-action-droid"],
    ),
    "cookbook DeepSeek-V4 on H200": (command_argv(DEEPSEEK_V4_H200), "llm", None),
    "cookbook Qwen3.8 on GB300, multi-node": (
        command_argv(QWEN38_GB300_MULTI_NODE),
        "llm",
        None,
    ),
    "cookbook Kimi-K3 on B200, JSON flag": (command_argv(KIMI_K3_B200), "llm", None),
}

QWEN = "Qwen/Qwen3.8-Flash-Next"
QWEN_NVFP4 = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
H200 = "NVIDIA H200"

GUIDE_CASES = {
    "bare model": ([QWEN], (QWEN, False)),
    "--dry-run after the model": ([QWEN, "--dry-run"], (QWEN, True)),
    "--dry-run before the model": (["--dry-run", QWEN], (QWEN, True)),
}


class TestServeDirectLaunch(CustomTestCase):
    def _run(self, argv, gpus=(H200, 4)):
        ran = []

        def capture(name):
            return lambda request: ran.append((name, request.argv))

        def guide(model, recipes, dry_run):
            ran.append(("guide", (model, dry_run)))

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(interactive, "_detect_gpus", return_value=gpus)
            )
            stack.enter_context(patch("sglang.cli.serve.run_guide", guide))
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
        return ran

    def _serve(self, argv):
        ran = self._run(argv)
        self.assertEqual(len(ran), 1)
        return ran[0]

    def _assert_cases(self, cases):
        for name, (argv, backend, received) in cases.items():
            with self.subTest(name):
                expected = tuple(argv if received is None else received)
                self.assertEqual(self._serve(argv), (backend, expected))

    def test_direct_launch_reaches_backend_unchanged(self):
        self._assert_cases(DIRECT_LAUNCH_CASES)

    def test_bare_model_opens_guide(self):
        for name, (argv, expected) in GUIDE_CASES.items():
            with self.subTest(name):
                self.assertEqual(self._serve(argv), ("guide", expected))

    def test_dry_run_with_explicit_args_prints_without_launching(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ran = self._run(["--model-path", "Qwen/Qwen3-8B", "--tp", "2", "--dry-run"])
        self.assertEqual(ran, [])
        self.assertEqual(
            stdout.getvalue(), "sglang serve --model-path Qwen/Qwen3-8B --tp 2\n"
        )

    def test_model_without_a_cookbook_command_says_so(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ran = self._run(["Qwen/Qwen3-8B", "--dry-run"])
        self.assertEqual(ran, [])
        self.assertEqual(stdout.getvalue(), "sglang serve Qwen/Qwen3-8B\n")
        self.assertIn(
            NO_RECIPE_MESSAGE.format(model="Qwen/Qwen3-8B"), stderr.getvalue()
        )

    def test_model_without_a_command_for_this_machine_launches_directly(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            ran = self._run([QWEN], gpus=(H200, 8))
        self.assertEqual(ran, [("llm", ("--model-path", QWEN))])
        self.assertIn("on 8x NVIDIA H200 (it has", stderr.getvalue())
        self.assertIn("launching it with the default flags", stderr.getvalue())

    def test_missing_model_is_rejected(self):
        for argv in ([], ["--dry-run"]):
            with self.subTest(argv):
                with self.assertRaisesRegex(Exception, "--model-path is required"):
                    self._run(argv)

    def test_server_args_has_no_dry_run(self):
        # serve.py strips --dry-run; a ServerArgs flag of the same name would be shadowed.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                prepare_server_args(["--model-path", "Qwen/Qwen3-8B", "--dry-run"])


def _exported():
    return {
        path.name.removesuffix(".json").replace("--", "/"): json.loads(path.read_text())
        for path in (resources.files("sglang.cli") / "_commands").iterdir()
        if path.name.endswith(".json")
    }


def _recipe(model, hw, mode):
    return next(e for e in _exported()[model] if e["hw"] == hw and e["mode"] == mode)


def _launch_argv(argv):
    return [sys.executable, "-m", "sglang.cli.main", "serve", *argv]


class TestExportedCookbookCommands(CustomTestCase):
    def test_every_command_parses_as_server_args(self):
        exported = _exported()
        self.assertTrue(exported)
        for model, entries in exported.items():
            for entry in entries:
                with self.subTest(entry["url"]):
                    server_args = prepare_server_args(command_argv(entry["command"]))
                    self.assertEqual(server_args.model_path, model)
                    gpus = compute_world_size(
                        dp_size=server_args.dp_size,
                        tp_size=server_args.tp_size,
                        pp_size=server_args.pp_size,
                    )
                    self.assertEqual(entry["gpus"], gpus)
                    prefix = entry["command"].split("sglang serve")[0]
                    env_lines = [
                        line.rstrip(" \\")
                        for line in prefix.splitlines()
                        if line.strip() and not line.startswith("#")
                    ]
                    self.assertEqual(
                        env_lines,
                        [f"{key}={value}" for key, value in entry["env"].items()],
                    )


class _PipedStdout(io.StringIO):
    """A piped stdout across exec: only what was flushed before it survives."""

    def __init__(self):
        super().__init__()
        self.flushed = ""

    def flush(self):
        super().flush()
        self.flushed = self.getvalue()


class TestCliGuide(CustomTestCase):
    def _guide(self, model, *, device, gpus, answers=None, dry_run=False):
        """Run the Guide on a fake machine; passing answers makes it a terminal."""
        stdout, stderr = _PipedStdout(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(interactive, "_detect_gpus", return_value=(device, gpus))
            )
            stack.enter_context(
                patch.object(
                    interactive, "_is_terminal", return_value=answers is not None
                )
            )
            stack.enter_context(patch("builtins.input", side_effect=answers or []))
            execvpe = stack.enter_context(patch.object(interactive.os, "execvpe"))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            try:
                recipes = recipes_for_this_machine(model)
                if recipes:
                    run_guide(model=model, recipes=recipes, dry_run=dry_run)
            except SystemExit as e:
                return execvpe, stdout.getvalue(), stderr.getvalue(), e
        printed = stdout.flushed if execvpe.called else stdout.getvalue()
        return execvpe, printed, stderr.getvalue(), None

    def test_launches_the_printed_cookbook_command(self):
        """A piped stdout (an agent) gets the command; exec drops unflushed output."""
        recipe = _recipe(QWEN, "h200", "low-latency")
        execvpe, stdout, stderr, exit = self._guide(QWEN, device=H200, gpus=4)

        self.assertIsNone(exit)
        self.assertEqual(stdout, recipe["command"] + "\n")
        self.assertIn("Mode: low-latency (default", stderr)
        self.assertIn(f'run "sglang serve {QWEN} --dry-run"', stderr)
        execvpe.assert_called_once()
        file, argv, _env = execvpe.call_args.args
        self.assertEqual(file, sys.executable)
        self.assertEqual(argv, _launch_argv(command_argv(recipe["command"])))

    def test_default_mode_order(self):
        base = _recipe(QWEN, "h200", "low-latency")
        for modes, expected in (
            (("high-throughput", "low-latency", "balanced"), "balanced"),
            (("high-throughput", "low-latency"), "low-latency"),
        ):
            recipes = [
                dict(base, mode=mode, command=f"sglang serve --mode {mode}")
                for mode in modes
            ]
            with (
                self.subTest(modes),
                patch.object(interactive, "_load_recipes", return_value=recipes),
            ):
                execvpe, _, _, _ = self._guide(QWEN, device=H200, gpus=4)
                self.assertEqual(
                    execvpe.call_args.args[1], _launch_argv(["--mode", expected])
                )

    def test_recipe_follows_visible_gpu_count(self):
        recipe = _recipe(QWEN_NVFP4, "b200", "low-latency")
        execvpe, _, _, _ = self._guide(QWEN_NVFP4, device="NVIDIA B200", gpus=1)
        self.assertEqual(
            execvpe.call_args.args[1], _launch_argv(command_argv(recipe["command"]))
        )

        execvpe, _, stderr, exit = self._guide(QWEN, device=H200, gpus=8)
        self.assertIsNone(exit)
        execvpe.assert_not_called()
        self.assertIn("on 8x NVIDIA H200", stderr)
        self.assertIn("4x h200", stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES", stderr)

    def test_recipe_env_reaches_the_server_unless_exported(self):
        recipe = _recipe(QWEN_NVFP4, "rtx6000", "low-latency")
        with patch.dict(interactive.os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "mine"}):
            execvpe, stdout, stderr, _ = self._guide(
                QWEN_NVFP4,
                device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                gpus=1,
            )
        self.assertTrue(stdout.startswith("PYTORCH_CUDA_ALLOC_CONF="))
        self.assertIn("Using your exported PYTORCH_CUDA_ALLOC_CONF", stderr)
        env = execvpe.call_args.args[2]
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "mine")
        self.assertEqual(
            env["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"],
            recipe["env"]["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"],
        )

    def test_no_command_for_this_machine_falls_back(self):
        for name, device, gpus, message in (
            ("unknown GPU", "NVIDIA A100-SXM4-80GB", 4, "4x NVIDIA A100"),
            ("no GPU", None, 0, "this machine (no GPU)"),
        ):
            with self.subTest(name):
                execvpe, _, stderr, exit = self._guide(QWEN, device=device, gpus=gpus)
                self.assertIsNone(exit)
                execvpe.assert_not_called()
                self.assertIn(message, stderr)
                self.assertIn("launching it with the default flags", stderr)

    def test_unverified_command_warns(self):
        recipe = _recipe(QWEN, "h200", "low-latency")
        for verified, warns in (
            ("verified", False),
            ("in-progress", True),
            ("unverified", True),
        ):
            with (
                self.subTest(verified),
                patch.object(
                    interactive,
                    "_load_recipes",
                    return_value=[dict(recipe, verified=verified)],
                ),
            ):
                _, _, stderr, _ = self._guide(QWEN, device=H200, gpus=4)
                self.assertEqual("Warning:" in stderr, warns)

    def test_dry_run_prints_without_launching(self):
        recipe = _recipe(QWEN, "h200", "low-latency")
        for answers in (None, [""]):
            with self.subTest(terminal=answers is not None):
                execvpe, stdout, stderr, exit = self._guide(
                    QWEN, device=H200, gpus=4, answers=answers, dry_run=True
                )
                self.assertIsNone(exit)
                execvpe.assert_not_called()
                self.assertEqual(stdout, recipe["command"] + "\n")
                self.assertNotIn("--dry-run", stderr)

    def test_terminal_asks_mode_then_flag_edits(self):
        recipe = _recipe(QWEN, "h200", "high-throughput")
        expected = command_argv(recipe["command"])
        expected[expected.index("--port") + 1] = "8000"
        i = expected.index("--reasoning-parser")
        del expected[i : i + 2]

        execvpe, stdout, _, _ = self._guide(
            QWEN,
            device=H200,
            gpus=4,
            answers=["2", "--port 8000", "!--reasoning-parser", ""],
        )

        self.assertEqual(execvpe.call_args.args[1], _launch_argv(expected))
        self.assertTrue(stdout.startswith(recipe["command"] + "\n"))
        self.assertTrue(
            stdout.endswith(format_command(env=recipe["env"], argv=expected) + "\n")
        )

    def test_terminal_asks_again_after_bad_input(self):
        recipe = _recipe(QWEN, "h200", "low-latency")
        execvpe, _, stderr, _ = self._guide(
            QWEN, device=H200, gpus=4, answers=["9", "", "8000", ""]
        )
        self.assertIn("Pick 1-2 or a mode name.", stderr)
        self.assertIn("Expected a flag", stderr)
        self.assertEqual(
            execvpe.call_args.args[1], _launch_argv(command_argv(recipe["command"]))
        )

    def test_eof_or_ctrl_c_cancels_without_launching(self):
        for error in (EOFError, KeyboardInterrupt):
            for answers in ([error()], ["", error()]):
                with self.subTest(error=error.__name__, turn=len(answers)):
                    execvpe, _, stderr, exit = self._guide(
                        QWEN, device=H200, gpus=4, answers=answers
                    )
                    execvpe.assert_not_called()
                    self.assertEqual(exit.code, 130)
                    self.assertIn("Cancelled", stderr)

    def test_model_without_modes_uses_the_default_command(self):
        recipe = dict(_recipe(QWEN, "h200", "low-latency"), mode=None)
        with patch.object(interactive, "_load_recipes", return_value=[recipe]):
            execvpe, _, stderr, _ = self._guide(
                "Example/Model", device=H200, gpus=4, answers=[""]
            )
        self.assertIn(NO_MODE_MESSAGE, stderr)
        self.assertNotIn("Mode [", stderr)
        execvpe.assert_called_once()

    def test_hardware_id_from_device_name(self):
        for device, hw in (
            ("NVIDIA GB300", "gb300"),
            ("NVIDIA B300 SXM6 AC", "b300"),
            ("NVIDIA GB200", "gb200"),
            ("NVIDIA B200", "b200"),
            ("NVIDIA GB10", "dgx-spark"),
            ("NVIDIA H200", "h200"),
            ("NVIDIA GH200 480GB", None),
            ("AMD Instinct MI355X", "mi355x"),
            ("NVIDIA A100-SXM4-80GB", None),
            (None, None),
        ):
            with self.subTest(device):
                self.assertEqual(interactive._hardware_id(device), hw)


class TestFlagEdits(unittest.TestCase):
    ARGV = ["--model-path", "Qwen/Qwen3-8B", "--tp", "4", "--enable-metrics"]

    def test_set_add_and_remove(self):
        self.assertEqual(
            edit_flags(argv=self.ARGV, edits="--tp=2 --port 8000 !--enable-metrics"),
            ["--model-path", "Qwen/Qwen3-8B", "--tp", "2", "--port", "8000"],
        )

    def test_bad_edits_are_rejected(self):
        for edits in ("8000", "!--port", "!--tp 2", "--port '8000"):
            with self.subTest(edits):
                with self.assertRaises(ValueError):
                    edit_flags(argv=self.ARGV, edits=edits)

    def test_format_matches_every_cookbook_command(self):
        for entries in _exported().values():
            for entry in entries:
                with self.subTest(entry["url"]):
                    lines = entry["command"].splitlines()
                    command = "\n".join(l for l in lines if not l.startswith("#"))
                    formatted = format_command(
                        env=entry["env"], argv=command_argv(entry["command"])
                    )
                    self.assertEqual(formatted, command)


if __name__ == "__main__":
    unittest.main()
