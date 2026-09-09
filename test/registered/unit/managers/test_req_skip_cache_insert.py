"""``Req.skip_radix_cache_insert`` comes only from the explicit request field.

The PD fake bootstrap host selects the fake KV sender and nothing else; a
request that wants to stay out of the prefix cache says so with
``skip_cache_insert``.
"""

import unittest
from array import array

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(**kwargs) -> Req:
    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    return Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=sampling_params,
        vocab_size=128,
        **kwargs,
    )


class TestReqSkipCacheInsert(CustomTestCase):
    def test_default_inserts(self):
        self.assertFalse(_make_req().skip_radix_cache_insert)

    def test_explicit_field_skips(self):
        self.assertTrue(_make_req(skip_cache_insert=True).skip_radix_cache_insert)

    def test_fake_bootstrap_host_alone_does_not_skip(self):
        req = _make_req(bootstrap_host=FAKE_BOOTSTRAP_HOST, bootstrap_room=0)
        self.assertFalse(req.skip_radix_cache_insert)


if __name__ == "__main__":
    unittest.main()
