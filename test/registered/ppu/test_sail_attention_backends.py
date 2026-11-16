"""Model-level SAIL FA3 (MHA) and FlashMLA (MLA) sanity tests."""

import os

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import (
    BasicDecodeCorrectnessMixin,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_ppu_ci(est_time=1200, suite="stage-a-test-1-gpu-ppu")


class SailModelMixin(BasicAPIContractMixin, BasicDecodeCorrectnessMixin):
    attention_backend: str
    model_env: str
    served_model_name = "sail-model-sanity"

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(cls.model_env)
        if not cls.model:
            raise RuntimeError(
                f"Set {cls.model_env} to a complete local model checkpoint"
            )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=[
                "--trust-remote-code",
                "--served-model-name",
                cls.served_model_name,
                "--attention-backend",
                cls.attention_backend,
                "--tp-size",
                "1",
                "--cuda-graph-max-bs-decode",
                "4",
                "--cuda-graph-backend-prefill=disabled",
                "--max-running-requests",
                "4",
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)

    def test_color_completion(self):
        # Both RGB (red/green/blue) and traditional RYB (red/yellow/blue)
        # conventions are valid readings of the shared sanity prompt.
        out = self._decode_generate(
            "Q: The three primary colors are red, blue, and ___. "
            "Fill in the blank.\nA:",
            self.sanity_max_new_tokens_short,
        ).lower()
        self.assertTrue(
            "green" in out or "yellow" in out,
            f"expected an RGB or RYB primary color, got: {out!r}",
        )


class TestSailFA3Model(SailModelMixin, CustomTestCase):
    attention_backend = "fa3"
    model_env = "SGLANG_TEST_FA3_MODEL"


class TestSailFlashMLAModel(SailModelMixin, CustomTestCase):
    attention_backend = "flashmla"
    model_env = "SGLANG_TEST_MLA_MODEL"


if __name__ == "__main__":
    import unittest

    unittest.main()
