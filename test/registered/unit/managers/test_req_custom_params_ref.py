import gc
import unittest
import weakref
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_req(custom_logit_processor=None):
    return Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=SamplingParams(custom_params={"thinking_budget": 8}),
        custom_logit_processor=custom_logit_processor,
    )


class TestReqCustomParamsRef(CustomTestCase):
    def test_req_without_a_processor_is_freed_by_refcount(self):
        """custom_params without a custom logit processor must not make the
        request reference itself; refcount alone frees it."""
        gc.disable()
        try:
            req = _make_req()
            self.assertNotIn("__req__", req.sampling_params.custom_params)
            ref = weakref.ref(req)
            del req
            self.assertIsNone(ref())
        finally:
            gc.enable()

    def test_processor_reads_its_request(self):
        """A custom logit processor still finds its request under __req__."""
        req = _make_req(custom_logit_processor="{}")
        self.assertIs(req.sampling_params.custom_params["__req__"], req)


if __name__ == "__main__":
    unittest.main()
