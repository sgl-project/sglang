# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sglang.cli.utils import (
    _is_diffusion_model_from_hub_metadata,
    get_is_diffusion_model,
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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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


class TestDiffusionImportIsolation(unittest.TestCase):
    def test_local_model_card_detects_diffusion_without_model_names(self):
        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "README.md"), "w") as model_card:
                model_card.write("---\npipeline_tag: text-to-image\n---\n")
            self.assertTrue(get_is_diffusion_model(model_path))

    @patch("sglang.cli.utils.HfApi")
    def test_hub_metadata_detects_diffusion_without_model_names(self, mock_hf_api):
        info = mock_hf_api.return_value.model_info.return_value
        info.library_name = None
        info.pipeline_tag = "text-to-video"
        info.tags = []
        self.assertTrue(_is_diffusion_model_from_hub_metadata("org/model"))

        info.pipeline_tag = "text-generation"
        self.assertFalse(_is_diffusion_model_from_hub_metadata("org/model"))

    def test_llm_detection_does_not_import_multimodal_gen(self):
        script = """
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
from sglang.cli.utils import get_is_diffusion_model

model_info = SimpleNamespace(
    library_name="transformers",
    pipeline_tag="text-generation",
    tags=["transformers"],
)
with patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError), patch(
    "sglang.cli.utils.HfApi"
) as hf_api:
    hf_api.return_value.model_info.return_value = model_info
    assert not get_is_diffusion_model("org/llm")

assert not any(
    name == "sglang.multimodal_gen" or name.startswith("sglang.multimodal_gen.")
    for name in sys.modules
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
