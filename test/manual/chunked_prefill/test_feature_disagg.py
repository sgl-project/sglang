import unittest

from sglang.test.chunked_prefill_test_utils import (
    KV_CANARY_ARGS,
    ChunkedRefactorTestBase,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model


class TestChunkedFeatureDisagg(PDDisaggregationServerBase, ChunkedRefactorTestBase):
    @classmethod
    def setUpClass(cls):
        cls.extra_prefill_args = [
            "--chunked-prefill-size",
            str(cls.chunked_prefill_size),
        ] + list(KV_CANARY_ARGS)
        cls.extra_decode_args = list(KV_CANARY_ARGS)

        PDDisaggregationServerBase.setUpClass()
        cls.model = try_cached_model(cls.model)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        PDDisaggregationServerBase.tearDownClass()


if __name__ == "__main__":
    unittest.main()
