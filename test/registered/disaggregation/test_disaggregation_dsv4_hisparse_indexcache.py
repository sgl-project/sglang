import unittest

import test_disaggregation_hisparse as _hisparse

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1000, stage="extra-b", runner_config="deepep-8-gpu-h200")


class TestDisaggregationDSV4HiSparseIndexCache(
    _hisparse.TestDisaggregationDSV4HiSparseBase
):
    extra_server_args = [
        "--json-model-override-args",
        '{"index_topk_freq": 4}',
    ]


if __name__ == "__main__":
    unittest.main()
