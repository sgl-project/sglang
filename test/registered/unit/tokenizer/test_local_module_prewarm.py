import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from transformers import (
    BertTokenizer,
    CLIPImageProcessor,
    LlamaConfig,
    LlavaConfig,
    LlavaProcessor,
    dynamic_module_utils,
)

from sglang.srt.environ import envs
from sglang.srt.hf_transformers import processor
from sglang.srt.hf_transformers.processor import (
    load_scheduler_tokenizer,
    maybe_prewarm_scheduler_tokenizer,
)
from sglang.srt.runtime_context import process_model_config, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@contextmanager
def isolated_hf_modules_cache(cache):
    prefix = dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME

    def pop_dynamic_modules():
        return {
            name: sys.modules.pop(name)
            for name in tuple(sys.modules)
            if name == prefix or name.startswith(prefix + ".")
        }

    with (
        patch.object(dynamic_module_utils, "HF_MODULES_CACHE", str(cache)),
        patch.object(sys, "path", sys.path.copy()),
    ):
        # A package from another cache can still resolve relative imports via
        # its old __path__. Restore that package after this fixture is done.
        previous = pop_dynamic_modules()
        try:
            yield
        finally:
            pop_dynamic_modules()
            sys.modules.update(previous)


def write_custom_tokenizer(model):
    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoTokenizer": [None, "custom.LocalTokenizer"]}
    config_path.write_text(json.dumps(config))
    (model / "custom.py").write_text(
        "from transformers import BertTokenizer\n"
        "from .helper import MARKER\n"
        "class LocalTokenizer(BertTokenizer):\n    marker = MARKER\n"
    )
    (model / "helper.py").write_text("MARKER = 'warm'\n")


class TestLocalModulePrewarm(CustomTestCase):
    def test_prewarm_failure_preserves_scheduler_launch(self):
        from sglang.srt.entrypoints.engine import Engine

        self.addCleanup(reset_context)
        with (
            tempfile.TemporaryDirectory() as model,
            patch.object(
                dynamic_module_utils, "HF_MODULES_CACHE", str(Path(model) / "modules")
            ),
        ):
            args = ServerArgs(
                model_path="dummy",
                tokenizer_path=model,
                device="cpu",
                tp_size=2,
                trust_remote_code=True,
            )
            publish(args, role="tokenizer")
            for failing_step in ("get_serving", "tempfile.TemporaryDirectory"):
                for launch_error in (None, OSError("scheduler startup failed")):
                    with (
                        self.subTest(step=failing_step, launch_error=launch_error),
                        patch(
                            f"{processor.__name__}.{failing_step}",
                            side_effect=ValueError("prewarm failed"),
                        ) as step,
                        self.assertLogs(processor.logger, level="WARNING") as logs,
                        patch(
                            "multiprocessing.Process.start", side_effect=launch_error
                        ) as start,
                    ):
                        if launch_error is None:
                            _, processes = Engine._launch_scheduler_processes(
                                args, None, None
                            )
                            self.assertEqual(len(processes), 2)
                            self.assertEqual(start.call_count, 2)
                        else:
                            with self.assertRaises(OSError) as raised:
                                Engine._launch_scheduler_processes(args, None, None)
                            self.assertIs(raised.exception, launch_error)
                        step.assert_called_once()
                        self.assertIn("prewarm failed", "\n".join(logs.output))

    def test_prewarm_only_for_concurrent_local_custom_code(self):
        self.addCleanup(reset_context)
        with (
            tempfile.TemporaryDirectory() as model,
            patch.object(
                dynamic_module_utils, "HF_MODULES_CACHE", str(Path(model) / "modules")
            ),
        ):
            for overrides, rust_server in (
                ({"trust_remote_code": False}, False),
                ({"skip_tokenizer_init": True}, False),
                ({"nnodes": 2, "node_rank": 1}, False),  # One reader on a follower.
                ({"tokenizer_path": "org/remote-tokenizer"}, False),
                ({"tp_size": 1}, True),  # No Python frontend on the Rust head node.
            ):
                with (
                    self.subTest(overrides=overrides, rust_server=rust_server),
                    envs.SGLANG_RUST_SERVER.override(rust_server),
                ):
                    options = dict(
                        model_path="dummy",
                        tokenizer_path=model,
                        device="cpu",
                        tp_size=2,
                        trust_remote_code=True,
                    )
                    publish(ServerArgs(**(options | overrides)), role="tokenizer")
                    maybe_prewarm_scheduler_tokenizer()
                    self.assertFalse((Path(model) / "modules").exists())

    def test_tp1_worker_loads_custom_tokenizer_without_recopying(self):
        from sglang.srt.entrypoints.engine import Engine

        self.addCleanup(reset_context)
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "llava"
            cache = Path(root) / "modules"
            LlavaConfig(
                architectures=["LlavaForConditionalGeneration"]
            ).save_pretrained(model)
            LlavaProcessor(
                image_processor=CLIPImageProcessor(), tokenizer=BertTokenizer()
            ).save_pretrained(model)
            # A resolved ServerArgs can retain an HF config class from the real cache.
            config_path = model / "config.json"
            config = json.loads(config_path.read_text())
            config["auto_map"] = {"AutoConfig": "configuration.LocalConfig"}
            config_path.write_text(json.dumps(config))
            (model / "configuration.py").write_text(
                "from transformers import LlavaConfig\n"
                "class LocalConfig(LlavaConfig):\n    pass\n"
            )
            processor_config = model / "processor_config.json"
            config = json.loads(processor_config.read_text())
            config["auto_map"] = {"AutoProcessor": "processing.LocalProcessor"}
            processor_config.write_text(json.dumps(config))
            (model / "processing.py").write_text(
                "from transformers import LlavaProcessor\n"
                "class LocalProcessor(LlavaProcessor):\n    pass\n"
            )
            write_custom_tokenizer(model)

            with isolated_hf_modules_cache(cache):
                args = ServerArgs(
                    model_path=str(model),
                    trust_remote_code=True,
                    device="cpu",
                    tp_size=1,
                )
                publish(args, role="tokenizer")

                self.assertEqual(
                    type(process_model_config().hf_config).__name__, "LocalConfig"
                )

                def check_at_spawn():
                    package = f"{dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME}.{model.name}"
                    self.assertFalse(
                        any(
                            name.startswith(package + ".")
                            and name.endswith((".custom", ".processing"))
                            for name in sys.modules
                        )
                    )
                    with patch(
                        "shutil.copyfile", side_effect=AssertionError("cold cache")
                    ):
                        tokenizer, processor = load_scheduler_tokenizer(
                            is_multimodal=True
                        )
                    self.assertIs(tokenizer, processor.tokenizer)
                    self.assertEqual(tokenizer.marker, "warm")

                with patch(
                    "multiprocessing.Process.start", side_effect=check_at_spawn
                ) as start:
                    Engine._launch_scheduler_processes(args, None, None)
                    self.assertEqual(start.call_count, 1)

    def test_minimax_detokenizer_reads_warmed_standalone_tokenizer(self):
        from transformers import MiniMaxM3VLConfig

        from sglang.srt.managers.detokenizer_manager import DetokenizerManager

        self.addCleanup(reset_context)
        with (
            tempfile.TemporaryDirectory() as root,
            isolated_hf_modules_cache(Path(root) / "modules"),
        ):
            model = Path(root) / "minimax"
            MiniMaxM3VLConfig(
                architectures=["MiniMaxM3SparseForConditionalGeneration"]
            ).save_pretrained(model)
            BertTokenizer().save_pretrained(model)
            (model / "preprocessor_config.json").write_text("{}")
            write_custom_tokenizer(model)
            args = ServerArgs(
                model_path=str(model),
                device="cpu",
                trust_remote_code=True,
                detokenizer_worker_num=2,
            )
            publish(args, role="tokenizer")
            self.assertTrue(process_model_config().is_multimodal)
            maybe_prewarm_scheduler_tokenizer()
            with patch(
                "shutil.copyfile", side_effect=AssertionError("cold detokenizer cache")
            ):
                worker = DetokenizerManager.__new__(DetokenizerManager)
                worker.init_tokenizer(args)
            self.assertEqual(worker.tokenizer.marker, "warm")

    def test_prewarm_loads_plugin_processor_modules(self):
        self.addCleanup(reset_context)
        with (
            tempfile.TemporaryDirectory() as root,
            isolated_hf_modules_cache(Path(root) / "modules"),
        ):
            model = Path(root) / "llava"
            LlavaConfig(
                architectures=["LlavaForConditionalGeneration"]
            ).save_pretrained(model)
            LlavaProcessor(
                image_processor=CLIPImageProcessor(), tokenizer=BertTokenizer()
            ).save_pretrained(model)
            (model / "plugin_only.py").write_text("class LocalHelper:\n    pass\n")
            # A discoverable plugin using a builtin config, so forgetting to load
            # it silently falls back to the stock LlavaProcessor.
            dist = Path(root) / "prewarm_test-1.0.dist-info"
            dist.mkdir()
            (dist / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: prewarm-test\nVersion: 1.0\n"
            )
            (dist / "entry_points.txt").write_text(
                "[sglang.srt.plugins]\nprewarm_test = prewarm_plugin:register\n"
            )
            (Path(root) / "prewarm_plugin.py").write_text(
                "from transformers import LlavaConfig, LlavaProcessor\n"
                "from transformers.dynamic_module_utils import get_class_from_dynamic_module\n"
                "from sglang.srt.multimodal.customized_mm_processor_utils import register_customized_processor\n"
                "class PluginProcessor(LlavaProcessor):\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, path, **kwargs):\n"
                "        get_class_from_dynamic_module('plugin_only.LocalHelper', path)\n"
                "        return LlavaProcessor.from_pretrained(path, **kwargs)\n"
                "def register():\n"
                "    register_customized_processor(PluginProcessor)(LlavaConfig)\n"
            )
            sys.path.insert(0, root)
            publish(
                ServerArgs(model_path=str(model), device="cpu", trust_remote_code=True),
                role="tokenizer",
            )
            with envs.SGLANG_PLUGINS.override("prewarm_test"):
                maybe_prewarm_scheduler_tokenizer()
            with patch(
                "shutil.copyfile", side_effect=AssertionError("cold plugin cache")
            ):
                helper = dynamic_module_utils.get_class_from_dynamic_module(
                    "plugin_only.LocalHelper", model
                )
            self.assertEqual(helper.__name__, "LocalHelper")

    def test_failed_custom_import_leaves_launcher_and_cache_clean(self):
        """A failed warmup must leave the real worker to report its own error."""
        self.addCleanup(reset_context)
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "broken"
            cache = Path(root) / "modules"
            LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(model)
            BertTokenizer().save_pretrained(model)
            config_path = model / "tokenizer_config.json"
            config = json.loads(config_path.read_text())
            config["auto_map"] = {"AutoTokenizer": [None, "broken.LocalTokenizer"]}
            config_path.write_text(json.dumps(config))
            (model / "broken.py").write_text(
                "import os\n"
                "os.environ['SGLANG_TEST_PREWARM_PID'] = str(os.getpid())\n"
                "raise RuntimeError('custom tokenizer import failed')\n"
            )
            args = ServerArgs(
                model_path=str(model),
                device="cpu",
                trust_remote_code=True,
                tp_size=4,
                nnodes=2,
                node_rank=1,
            )
            publish(args, role="tokenizer")
            with patch.dict(os.environ), isolated_hf_modules_cache(cache):
                os.environ.pop("SGLANG_TEST_PREWARM_PID", None)
                with self.assertLogs(processor.logger, level="WARNING"):
                    maybe_prewarm_scheduler_tokenizer()
                self.assertIsNone(os.environ.get("SGLANG_TEST_PREWARM_PID"))
                self.assertEqual(list(cache.rglob("*.py")), [])
                with self.assertRaisesRegex(
                    RuntimeError, "custom tokenizer import failed"
                ):
                    load_scheduler_tokenizer(is_multimodal=False)
                self.assertEqual(
                    os.environ["SGLANG_TEST_PREWARM_PID"], str(os.getpid())
                )


if __name__ == "__main__":
    unittest.main()
