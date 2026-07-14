"""Unit tests for Evo 2 model configuration and NaN safety."""

import inspect
import os
import tempfile

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestEvo2Config(CustomTestCase):
    def test_config_has_model_type(self):
        from sglang.srt.configs.evo2 import Evo2Config

        self.assertEqual(Evo2Config().model_type, "evo2")

    def test_detect_7b_variant(self):
        from sglang.srt.configs.evo2 import _detect_evo2_variant

        self.assertEqual(_detect_evo2_variant("evo2_7b_base"), "7b-8k")
        self.assertEqual(_detect_evo2_variant("evo2_1b_base"), "1b")
        # evo2_7b (no suffix) is the 1M-context flagship
        self.assertEqual(_detect_evo2_variant("evo2_7b"), "7b-1m")

    def test_detect_unknown_raises(self):
        from sglang.srt.configs.evo2 import _detect_evo2_variant

        with self.assertRaises(ValueError):
            _detect_evo2_variant("unknown_model")


class TestEvo2ForwardSanity(CustomTestCase):
    def test_nan_to_num_present(self):
        from sglang.srt.models.evo2 import Evo2ForCausalLM

        source = inspect.getsource(Evo2ForCausalLM.forward)
        self.assertIn("nan_to_num", source)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tokenizer_files_generated(self):
        from sglang.srt.models.evo2 import generate_evo2_tokenizer_files

        generate_evo2_tokenizer_files(self.tmpdir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "tokenizer.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.tmpdir, "tokenizer_config.json"))
        )
