"""Hybrid attention-backend (FA3 prefill + FlashInfer decode) test fixture.

Variants combine `TestHybridAttnBackendBase` with their own
`get_server_args()` / `accuracy_threshold` / `speculative_decode` knobs.

Requires SM 90+ (H100); the base class wraps that in a `skipIf`.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GSM_DATASET_PATH = None

# Default server arguments shared across all hybrid-attn-backend tests
DEFAULT_HYBRID_ATTN_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--prefill-attention-backend",
    "fa3",
    "--decode-attention-backend",
    "flashinfer",
]


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class TestHybridAttnBackendBase(CustomTestCase):

    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.65  # derived tests need to override this
    speculative_decode = False
    spec_decode_threshold = 2.2  # derived spec decoding tests need to override this
    # Appended after DEFAULT_HYBRID_ATTN_SERVER_ARGS in get_server_args.
    extra_args: list = []

    @classmethod
    def get_server_args(cls):
        return DEFAULT_HYBRID_ATTN_SERVER_ARGS + list(cls.extra_args)

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        with (
            envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.override(False),
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
        ):
            if cls.speculative_decode:
                model = DEFAULT_TARGET_MODEL_EAGLE
            else:
                model = cls.model
            cls.process = popen_launch_server(
                model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.get_server_args(),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        model = DEFAULT_TARGET_MODEL_EAGLE if self.speculative_decode else self.model
        args = SimpleNamespace(
            base_url=self.base_url,
            model=model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], self.accuracy_threshold)

        if self.speculative_decode:
            server_info = requests.get(self.base_url + "/server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)
