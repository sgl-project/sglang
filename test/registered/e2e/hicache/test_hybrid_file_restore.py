import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.hicache_fixture import FileHiCacheRestoreMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")


class TestHybridFileRestore(FileHiCacheRestoreMixin, CustomTestCase):
    model = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    revision = "e1df551a447157d4658b573f9a695d57658590e9"
    tp = 4
    dcp = 1
    hybrid = True
    other_args = [
        "--attention-backend",
        "cutedsl_mla",
        "--max-mamba-cache-size",
        "64",
    ]


class TestHybridDcpFileRestore(TestHybridFileRestore):
    dcp = 4
    other_args = [
        *TestHybridFileRestore.other_args,
        "--dcp-comm-backend",
        "a2a",
        "--dcp-replicate-q-proj",
    ]


if __name__ == "__main__":
    unittest.main()
