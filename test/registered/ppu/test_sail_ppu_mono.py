"""T-Head PPU monokernel boot e2e: qwen4_exp served behind --enable-t-head-monokernel.

Runs on PPU runners only. Needs the sgl-t-head-mega plugin importable (the
docker pin is pending) and a complete qwen4_exp checkpoint named by
SGLANG_PPU_MEGA_MODEL. The server boots the bs=1 mtp3 shape the mega
chain is validated on -- decode steps of 1 request x 4 tokens, full
decode CUDA graphs -- and asserts the standard API contract and decode
correctness kits on top of it: routing through the monokernels must not
change what the served model answers.
"""

import os
from importlib.util import find_spec

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

register_ppu_ci(
    est_time=2400,
    suite="extra-test-1-gpu-ppu",
    disabled="the sgl-t-head-mega wheel and a 4-PPU runner are not available yet; "
    "re-enable once the docker pin lands",
)

MODEL_ENV = "SGLANG_PPU_MEGA_MODEL"


class TestSailPpuMonoE2E(
    BasicAPIContractMixin, BasicDecodeCorrectnessMixin, CustomTestCase
):
    served_model_name = "sail-ppu-mono"

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(MODEL_ENV)
        if not cls.model:
            raise RuntimeError(
                f"Set {MODEL_ENV} to a complete qwen4_exp checkpoint to run"
            )
        if find_spec("sgl_t_head_mega") is None:
            raise RuntimeError(
                "the sgl-t-head-mega plugin must be installed to run the mono e2e"
            )
        # the plugin reads its routing mode at import time, in the server process
        os.environ["SGLANG_PPU_MEGA_BACKBONE"] = "mono"
        os.environ["SGLANG_PPU_MEGA_DRAFT"] = "mono"
        # PPU device images pin flashinfer below the launch-time minimum and the
        # production launchers skip that check; do the same here
        os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=[
                "--trust-remote-code",
                "--served-model-name",
                cls.served_model_name,
                "--tp-size",
                "4",
                "--enable-t-head-monokernel",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--cuda-graph-backend-decode",
                "full",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--cuda-graph-max-bs-decode",
                "1",
                "--max-running-requests",
                "1",
                "--context-length",
                "8192",
                "--mem-fraction-static",
                "0.85",
                # the mega chain runs its own symmetric-memory all-reduce inside the
                # persistent kernels; the launcher always passes these three, and
                # without --disable-custom-all-reduce sglang's custom AR path builds
                # a tvm-ffi wrapper this device's toolchain cannot compile
                "--disable-custom-all-reduce",
                "--watchdog-timeout",
                "1800",
                "--json-model-override-args",
                '{"text_config":{"ple_offload_embedding":true,'
                '"ple_offload_backend":"pinned"}}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)


if __name__ == "__main__":
    import unittest

    unittest.main()
