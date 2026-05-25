"""Feature (b): PD disaggregation + chunked prefill.

Disagg has its own setup (separate prefill + decode servers + load balancer),
so this fixture overrides ``setUpClass`` to use ``PDDisaggregationServerBase``
instead of the single-server pattern in ChunkedRefactorTestBase. The eval
runs against the LB URL.

Server arg template borrowed from
``test/registered/disaggregation/test_disaggregation_pp.py::TestDisaggregationPrefillPPAccuracy``.

GPU requirement: at least 2 GPUs (prefill on gpu 0, decode on gpu 1; LB on
host). Multi-node disagg works too but isn't required.

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import (
    DEFAULT_CHUNKED_PREFILL_SIZE,
    KV_CANARY_ARGS,
    ChunkedRefactorTestBase,
)
from typing import ClassVar, List

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, try_cached_model


class TestChunkedFeatureB_Disagg(PDDisaggregationServerBase, ChunkedRefactorTestBase):
    """Disagg PD with chunked prefill on the prefill side.

    Multiple inheritance: PDDisaggregationServerBase provides ``setUpClass``
    and ``tearDownClass`` (which launch prefill/decode/LB and clean up);
    ChunkedRefactorTestBase provides ``test_gsm8k_mixed_chunked`` and
    ``_run_gsm8k_mixed``. The MRO puts PDDisaggregationServerBase first so
    its setUpClass wins. We bind ``cls.base_url`` to the LB URL there
    already.
    """

    model: ClassVar[str] = DEFAULT_MODEL_NAME_FOR_TEST

    # PDDisaggregationServerBase reads these to pass per-side args.
    extra_prefill_args: ClassVar[List[str]] = [
        "--chunked-prefill-size",
        str(DEFAULT_CHUNKED_PREFILL_SIZE),
        "--disable-overlap-schedule",
    ] + KV_CANARY_ARGS
    extra_decode_args: ClassVar[List[str]] = list(KV_CANARY_ARGS)

    @classmethod
    def setUpClass(cls):
        # Use the disagg path's setup, not ChunkedRefactorTestBase's.
        PDDisaggregationServerBase.setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        PDDisaggregationServerBase.tearDownClass()


if __name__ == "__main__":
    unittest.main()
