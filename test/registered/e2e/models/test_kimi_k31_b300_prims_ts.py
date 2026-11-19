"""test_kimi_k31_b300.py on the Prims-TS DSA backend, which InferenceX serves with.

Under DCP, Prims-TS decode reads the sharded indexer's local KV slots through a
path the default FlashMLA backend does not share.
"""

import unittest

import test_kimi_k31_b300 as flashmla_case

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-b300")


class TestKimiK31B300PrimsTs(flashmla_case.TestKimiK31B300):
    server_args = flashmla_case.TestKimiK31B300.server_args + [
        "--dsa-prefill-backend",
        "prims_ts",
        "--dsa-decode-backend",
        "prims_ts",
    ]


if __name__ == "__main__":
    unittest.main()
