"""NGRAM speculative-decoding server fixture.

Variants combine this base with `GSM8KMixin` and override `attention_backend`
(required) plus optional `extra_args` to select a backend / pass extra flags.

Example:
    from sglang.test.server_fixtures.ngram_fixture import NgramServerBase
    from sglang.test.kits.eval_accuracy_kit import GSM8KMixin

    class TestNgramSpeculativeDecodingTriton(NgramServerBase, GSM8KMixin):
        attention_backend = "triton"

The base itself is NOT a runnable test (no `test_*` methods until a subclass
mixes in GSM8KMixin), so unittest discovery picks it up as empty.
"""

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_NGRAM,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_NGRAM_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "16",
    "--mem-fraction-static",
    0.8,
]


class NgramServerBase(CustomTestCase):
    model = DEFAULT_TARGET_MODEL_NGRAM
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.79
    gsm8k_accept_length_thres = 1.8

    # Subclasses must set `attention_backend`; `extra_args` is optional.
    attention_backend: str = ""
    extra_args: list = []

    @classmethod
    def get_server_args(cls):
        assert cls.attention_backend, f"{cls.__name__} must set `attention_backend`"
        return (
            DEFAULT_NGRAM_SERVER_ARGS
            + ["--attention-backend", cls.attention_backend]
            + list(cls.extra_args)
        )

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
