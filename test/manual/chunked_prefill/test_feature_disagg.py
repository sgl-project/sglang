"""PD disaggregation + chunked prefill.

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
    KV_CANARY_ARGS,
    ChunkedRefactorTestBase,
)
from typing import ClassVar

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, try_cached_model


class TestChunkedFeatureDisagg(PDDisaggregationServerBase, ChunkedRefactorTestBase):
    """Disagg PD with chunked prefill on the prefill side.

    Multiple inheritance: PDDisaggregationServerBase provides server launch
    (prefill + decode + LB); ChunkedRefactorTestBase provides
    ``test_gsm8k_mixed_chunked`` and ``_run_gsm8k_mixed``. We override
    setUpClass to drive the disagg setup explicitly — the inherited
    ChunkedRefactorTestBase.setUpClass is *not* called as it would try to
    launch a single server.
    """

    model: ClassVar[str] = DEFAULT_MODEL_NAME_FOR_TEST

    @classmethod
    def setUpClass(cls):
        # Compose disagg-side args here so subclasses overriding
        # ``cls.chunked_prefill_size`` actually take effect (doing this at
        # class-body time would freeze them at import). ``KV_CANARY_ARGS`` is
        # re-read here so flipping that constant in common.py and re-running
        # the suite sees the new value uniformly.
        cls.extra_prefill_args = [
            "--chunked-prefill-size",
            str(cls.chunked_prefill_size),
            "--disable-overlap-schedule",
        ] + list(KV_CANARY_ARGS)
        cls.extra_decode_args = list(KV_CANARY_ARGS)

        PDDisaggregationServerBase.setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        PDDisaggregationServerBase.tearDownClass()


if __name__ == "__main__":
    unittest.main()
