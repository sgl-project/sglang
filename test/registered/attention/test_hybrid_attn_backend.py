import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Hybrid attention backend tests (FA3 prefill + FlashInfer decode, requires SM 90+ / H100)
# Multiple test classes: base, MLA, TorchCompile, SpecDecode variants
register_cuda_ci(est_time=407, stage="stage-b", runner_config="1-gpu-large")

GSM_DATASET_PATH = None

# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
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

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS

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


class TestHybridAttnBackendMLA(TestHybridAttnBackendBase):
    accuracy_threshold = 0.60
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS


class TestHybridAttnBackendTorchCompile(TestHybridAttnBackendBase):
    accuracy_threshold = 0.65

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--enable-torch-compile"]


class TestHybridAttnBackendSpeculativeDecodingPrefillBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-attention-mode",
            "prefill",
        ]


class TestHybridAttnBackendSpeculativeDecodingDecodeBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-attention-mode",
            "decode",
        ]


if __name__ == "__main__":
    unittest.main()
