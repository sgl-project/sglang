"""DSA model + MTP (EAGLE) speculative-decoding server fixture.

Variants combine this base with `GSM8KMixin` and `SpecDecodingMixin`, set
`model` and any per-variant overrides (e.g. `enable_dp_attention`,
`mem_fraction_static`), and provide a `bs_1_speed_thres`.

Example:
    from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase
    from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
    from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin

    class TestDsv32DP(DsaMtpServerBase, GSM8KMixin, SpecDecodingMixin):
        model = "deepseek-ai/DeepSeek-V3.2"
        enable_dp_attention = True
        bs_1_speed_thres = 90

The base itself is NOT a runnable test (no `test_*` methods until a subclass
mixes in the kits), so unittest discovery picks it up as empty.
"""

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class DsaMtpServerBase(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST

    # Subclasses must set `model`; the others have sensible defaults.
    model: str = ""
    mem_fraction_static: float = 0.7
    enable_dp_attention: bool = False

    # EAGLE MTP config (fixed across DSA-MTP variants).
    speculative_algorithm: str = "EAGLE"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4

    # GSM8KMixin defaults tuned for DSA MTP accuracy regression.
    gsm8k_accuracy_thres = 0.94
    gsm8k_accept_length_thres = 2.7
    gsm8k_num_questions = 500
    gsm8k_num_threads = 500
    gsm8k_num_shots = 20

    # SpecDecodingMixin default; per-variant subclasses set `bs_1_speed_thres`.
    accept_length_thres = 2.7

    @classmethod
    def get_server_args(cls):
        assert cls.model, f"{cls.__name__} must set `model`"
        args = ["--trust-remote-code", "--tp", "8"]
        if cls.enable_dp_attention:
            args += ["--dp", "8", "--enable-dp-attention"]
        args += [
            "--speculative-algorithm",
            cls.speculative_algorithm,
            "--speculative-num-steps",
            str(cls.speculative_num_steps),
            "--speculative-eagle-topk",
            str(cls.speculative_eagle_topk),
            "--speculative-num-draft-tokens",
            str(cls.speculative_num_draft_tokens),
            "--mem-frac",
            str(cls.mem_fraction_static),
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        return args

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
