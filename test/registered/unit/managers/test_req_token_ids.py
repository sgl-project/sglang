from array import array

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestReqTokenIds(CustomTestCase):
    def test_list_origin_input_ids_are_normalized_for_fill_ids(self):
        req = Req(
            rid="req-list-origin",
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        req.output_ids.extend([4, 5])

        req.init_next_round_input()

        self.assertIsInstance(req.origin_input_ids, array)
        self.assertIsInstance(req.output_ids, array)
        self.assertIsInstance(req.fill_ids, array)
        self.assertEqual(list(req.fill_ids), [1, 2, 3, 4, 5])

    def test_list_origin_input_ids_unpadded_are_normalized(self):
        req = Req(
            rid="req-list-unpadded",
            origin_input_text="",
            origin_input_ids=array("q", [10, 11]),
            origin_input_ids_unpadded=[10],
            sampling_params=SamplingParams(max_new_tokens=8),
        )

        self.assertIsInstance(req.origin_input_ids_unpadded, array)
        self.assertEqual(list(req.origin_input_ids_unpadded), [10])
