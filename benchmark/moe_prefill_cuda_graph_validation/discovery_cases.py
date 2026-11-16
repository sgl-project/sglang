"""Discovery variants of the registered e2e classes with an explicit prefill backend.

Run from the SGLang checkout root:
    python3 discovery_cases.py <ClassName>
The registered classes leave the prefill CUDA graph backend to the server's
defaults; these subclasses pin --cuda-graph-backend-prefill=breakable so that
auto-disable rules cannot hide capture or replay failures.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.getcwd(), "test", "registered", "backends"))
sys.path.insert(0, os.path.join(os.getcwd(), "test", "registered", "rl"))

import test_flashinfer_nvfp4_online_moe_backend as online  # noqa: E402
import test_update_weights_from_disk_blackwell as reload_test  # noqa: E402

FORCE_BCG = ["--cuda-graph-backend-prefill", "breakable"]


class ForcedBcgMegaMoeOnlineW4A16(online.TestFlashinferMegaMoeBackendNvFp4OnlineW4A16):
    extra_args = [
        *online.TestFlashinferMegaMoeBackendNvFp4OnlineW4A16.extra_args,
        *FORCE_BCG,
    ]


class ForcedBcgCuteDSLOnlineW4A16(
    online.TestFlashinferCuteDSLMoeBackendNvFp4OnlineW4A16
):
    extra_args = [
        *online.TestFlashinferCuteDSLMoeBackendNvFp4OnlineW4A16.extra_args,
        *FORCE_BCG,
    ]


def _force_bcg_suites(suites):
    return tuple(
        {**suite, "other_args": (*suite["other_args"], *FORCE_BCG)} for suite in suites
    )


class ForcedBcgReloadW4A16MegaMoE(
    reload_test.TestServerUpdateWeightsFromDiskNVFP4W4A16MegaMoE
):
    backend_test_suites = _force_bcg_suites(
        reload_test.TestServerUpdateWeightsFromDiskNVFP4W4A16MegaMoE.backend_test_suites
    )


class ForcedBcgReloadW4A16CuteDSL(
    reload_test.TestServerUpdateWeightsFromDiskNVFP4W4A16CuteDSL
):
    backend_test_suites = _force_bcg_suites(
        reload_test.TestServerUpdateWeightsFromDiskNVFP4W4A16CuteDSL.backend_test_suites
    )


class NemotronFp8DefaultMoeBackend(
    online.FlashinferNvFp4OnlineMoeBackendBase, online.CustomTestCase
):
    """Nemotron-H on its default MoE backend, checkpoint FP8, no speculation.

    Sanity check for enabling Nemotron-H hybrid prefill graphs on SM100-family
    beyond the CuTe DSL configurations.
    """

    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    eval_args = {"max_tokens": 16000, "temperature": 1.0, "top_p": 0.95}

    @classmethod
    def setUpClass(cls):
        cls.base_url = online.DEFAULT_URL_FOR_TEST
        cls.process = online.popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=online.DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--reasoning-parser",
                "nemotron_3",
                "--tool-call-parser",
                "qwen3_coder",
                "--mem-fraction-static",
                "0.7",
            ],
        )


if __name__ == "__main__":
    unittest.main()
