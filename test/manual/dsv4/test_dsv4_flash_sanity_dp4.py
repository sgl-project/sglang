"""DSV4-Flash 4-GPU server sanity matrix (TP4 variants)."""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.server_sanity_kit import ServerSanityMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


def _launch(other_args, env_extra=None, timeout_mult=1):
    env = dict(DSV4_FLASH_ENV)
    if env_extra:
        env.update(env_extra)
    return popen_launch_server(
        DSV4_FLASH_MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * timeout_mult,
        other_args=other_args,
        env=env,
    )


_EAGLE_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


class TestDSV4FlashTP4DP4(ServerSanityMixin, CustomTestCase):
    """TP4 + DP4 + deepep + EAGLE MTP."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "256",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--mem-fraction-static",
                "0.7",
                *_EAGLE_SPEC_ARGS,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSV4FlashTP4EP(ServerSanityMixin, CustomTestCase):
    """TP attn + EP MoE (no DP attn) — exercises the DeepEP + TP-attn path."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "4",
                "--ep",
                "4",
                # No --enable-dp-attention by design: covers TP-attn path.
                "--moe-a2a-backend",
                "deepep",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "64",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--mem-fraction-static",
                "0.7",
                *_EAGLE_SPEC_ARGS,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSV4FlashTP4DP4ChunkedPrefillLarge(ServerSanityMixin, CustomTestCase):
    """TP4 + DP4 with --chunked-prefill-size 16384 — large chunked prefill."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--chunked-prefill-size",
                "16384",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "256",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--mem-fraction-static",
                "0.7",
                *_EAGLE_SPEC_ARGS,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
