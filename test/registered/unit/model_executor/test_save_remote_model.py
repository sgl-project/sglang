import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from sys import modules
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import mock

os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
modules.setdefault("deep_gemm", ModuleType("deep_gemm"))

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestSaveRemoteModel(unittest.TestCase):
    def test_run_save_remote_passes_rpc_params_without_extra_nesting(self):
        from sglang.save_model_remote_loader import SaveRemoteArgs, run_save_remote

        calls = []

        class FakeEngine:
            def __init__(self, server_args):
                self.server_args = server_args

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def save_remote_model(self, **kwargs):
                calls.append(kwargs)

        server_args = SimpleNamespace(model_path="model", log_level="warning")
        save_args = SaveRemoteArgs(
            url="mooncake:///model", draft_url="mooncake:///draft"
        )

        with mock.patch("sglang.save_model_remote_loader.Engine", FakeEngine):
            run_save_remote(server_args, save_args)

        self.assertEqual(
            calls,
            [{"url": "mooncake:///model", "draft_url": "mooncake:///draft"}],
        )

    def test_hf_repo_save_downloads_tokenizer_metadata(self):
        from sglang.srt.connector.utils import COMMON_REMOTE_MODEL_FILES
        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = object.__new__(ModelRunner)
        runner.model = object()
        runner.model_config = SimpleNamespace(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            revision="main",
        )
        runner.load_config = SimpleNamespace(download_dir="/tmp/download")

        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.os.path.isdir",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.model_executor.model_runner.snapshot_download",
                return_value="/tmp/hf",
            ) as snapshot_download,
            mock.patch(
                "sglang.srt.model_executor.model_runner.get_lock",
                return_value=nullcontext(),
            ),
            mock.patch(
                "sglang.srt.model_loader.loader.RemoteModelLoader.save_model"
            ) as save_model,
        ):
            ModelRunner.save_remote_model(runner, "mooncake:///model")

        snapshot_download.assert_called_once()
        allow_patterns = snapshot_download.call_args.kwargs["allow_patterns"]
        self.assertIn("*.json", allow_patterns)
        for file_name in COMMON_REMOTE_MODEL_FILES:
            self.assertIn(file_name, allow_patterns)
        save_model.assert_called_once_with(runner.model, "/tmp/hf", "mooncake:///model")

    def test_remote_model_save_writes_tokenizer_metadata_files(self):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        class FakeConnector:
            def __init__(self):
                self.strings = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def setstr(self, key, obj):
                self.strings[key] = obj

            def batch_put_from(self, keys, tensors):
                pass

        fake_connector = FakeConnector()
        fake_model = mock.Mock()
        fake_model.state_dict.return_value = {}

        with TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "config.json").write_text("{}", encoding="utf-8")
            Path(tmp_dir, "tokenizer.model").write_text("spm", encoding="utf-8")
            Path(tmp_dir, "chat_template.jinja").write_text(
                "template", encoding="utf-8"
            )
            Path(tmp_dir, "README.md").write_text("docs", encoding="utf-8")

            with (
                mock.patch(
                    "sglang.srt.model_loader.loader.create_remote_connector",
                    return_value=fake_connector,
                ),
                mock.patch(
                    "sglang.srt.model_loader.loader.get_connector_type",
                    return_value="KV",
                ),
                mock.patch(
                    "sglang.srt.model_loader.loader.get_tensor_model_parallel_rank",
                    return_value=0,
                ),
            ):
                RemoteModelLoader.save_model(fake_model, tmp_dir, "mooncake:///model")

        self.assertEqual(fake_connector.strings["model/files/config.json"], "{}")
        self.assertEqual(fake_connector.strings["model/files/tokenizer.model"], "spm")
        self.assertEqual(
            fake_connector.strings["model/files/chat_template.jinja"], "template"
        )
        self.assertNotIn("model/files/README.md", fake_connector.strings)


if __name__ == "__main__":
    unittest.main()
