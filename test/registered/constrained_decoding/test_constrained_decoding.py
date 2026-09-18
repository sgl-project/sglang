import unittest

import openai
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.json_mode_kit import JSONModeMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=161, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=220, suite="stage-b-test-1-gpu-small-amd")


class ServerWithGrammar(CustomTestCase):
    backend = "xgrammar"
    disable_overlap = False
    tp_size = 1

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
        ]

        if cls.tp_size > 1:
            launch_args += ["--tp-size", str(cls.tp_size)]

        if cls.disable_overlap:
            launch_args += ["--disable-overlap-schedule"]

        if is_in_amd_ci():
            launch_args.append("--constrained-json-disable-any-whitespace")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestXGrammarBackend(
    ServerWithGrammar,
    JSONConstrainedMixin,
    JSONModeMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    backend = "xgrammar"


class TestOutlinesBackend(ServerWithGrammar, JSONConstrainedMixin, JSONModeMixin):
    backend = "outlines"


class TestLLGuidanceBackend(
    ServerWithGrammar,
    JSONConstrainedMixin,
    JSONModeMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    backend = "llguidance"


# With TP > 1 only the entry rank compiles grammars and applies the vocab
# mask; the sampled token ids are broadcast in the sampler. These variants
# exercise that path end to end.
@unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
class TestXGrammarBackendTP2(
    ServerWithGrammar,
    JSONConstrainedMixin,
    JSONModeMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    backend = "xgrammar"
    tp_size = 2


@unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
class TestOutlinesBackendTP2(ServerWithGrammar, JSONConstrainedMixin, JSONModeMixin):
    backend = "outlines"
    tp_size = 2


@unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
class TestLLGuidanceBackendTP2(
    ServerWithGrammar,
    JSONConstrainedMixin,
    JSONModeMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    backend = "llguidance"
    tp_size = 2


if __name__ == "__main__":
    unittest.main()
