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
from sglang.srt.runtime_context import publish, reset_context
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
            processor_config = model / "processor_config.json"
            config = json.loads(processor_config.read_text())
            config["auto_map"] = {"AutoProcessor": "processing.LocalProcessor"}
            processor_config.write_text(json.dumps(config))
            (model / "processing.py").write_text(
                "from transformers import LlavaProcessor\n"
                "class LocalProcessor(LlavaProcessor):\n    pass\n"
            )
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

            with isolated_hf_modules_cache(cache):
                args = ServerArgs(
                    model_path=str(model),
                    trust_remote_code=True,
                    device="cpu",
                    tp_size=1,
                )
                publish(args, role="tokenizer")

                def check_at_spawn():
                    package = f"{dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME}.{model.name}"
                    self.assertFalse(
                        any(
                            name == package or name.startswith(package + ".")
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
