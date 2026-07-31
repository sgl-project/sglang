import unittest

import test_deepseek_v4_flash_fp4_b200_cp as _dsv4_cp

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=260, stage="extra-b", runner_config="deepep-4-gpu-b200")


class TestDSV4FlashFP4B200CPIndexCache(
    _dsv4_cp.TestDSV4FlashFP4B200Balanced_CP_NonDeepEP
):
    extra_server_args = [
        "--enable-deepseek-v4-fp4-indexer",
        "--json-model-override-args",
        '{"index_topk_freq": 4}',
    ]


if __name__ == "__main__":
    unittest.main()
