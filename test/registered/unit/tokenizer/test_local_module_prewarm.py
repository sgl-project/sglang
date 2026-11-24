import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import (
    BertTokenizer,
    CLIPImageProcessor,
    LlavaConfig,
    LlavaProcessor,
    dynamic_module_utils,
)

from sglang.srt.hf_transformers.processor import (
    load_scheduler_tokenizer,
    maybe_prewarm_scheduler_tokenizer,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalModulePrewarm(CustomTestCase):
    def test_prewarm_only_for_concurrent_local_custom_code(self):
        self.addCleanup(reset_context)
        with tempfile.TemporaryDirectory() as model:
            for overrides in (
                {"trust_remote_code": False},
                {"skip_tokenizer_init": True},
                {"tp_size": 1},
                {"nnodes": 2},  # One scheduler per node.
                {"tokenizer_path": "org/remote-tokenizer"},
            ):
                with self.subTest(overrides=overrides):
                    options = dict(
                        model_path="dummy",
                        tokenizer_path=model,
                        device="cpu",
                        tp_size=2,
                        trust_remote_code=True,
                    )
                    publish(ServerArgs(**(options | overrides)), role="tokenizer")
                    with patch(
                        "sglang.srt.hf_transformers.processor.process_model_config"
                    ) as model_config:
                        maybe_prewarm_scheduler_tokenizer()
                        model_config.assert_not_called()

    def test_worker_loads_custom_tokenizer_without_recopying(self):
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

            with (
                patch.object(dynamic_module_utils, "HF_MODULES_CACHE", str(cache)),
                patch.object(
                    dynamic_module_utils,
                    "TRANSFORMERS_DYNAMIC_MODULE_NAME",
                    Path(root).name,
                ),
                patch.object(sys, "path", sys.path.copy()),
            ):
                args = ServerArgs(
                    model_path=str(model),
                    trust_remote_code=True,
                    device="cpu",
                    tp_size=4,
                    nnodes=2,
                    node_rank=1,
                )
                publish(args, role="tokenizer")

                def check_at_spawn():
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
                    self.assertEqual(start.call_count, 2)


if __name__ == "__main__":
    unittest.main()
