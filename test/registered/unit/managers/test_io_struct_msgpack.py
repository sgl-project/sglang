import unittest

from sglang.srt.managers.io_struct import (
    MooncakeMMUrlItem,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestTokenizedGenerateReqInputMsgpack(unittest.TestCase):
    def _make_request(self, mm_data_mooncake):
        return TokenizedGenerateReqInput(
            input_text="",
            input_ids=None,
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            mm_data_mooncake=mm_data_mooncake,
        )

    def _round_trip(self, req):
        req.wrap_pickle_fields()
        decoded = msgpack_decode(msgpack_encode(req))
        decoded.unwrap_pickle_fields()
        return decoded

    def test_mm_data_mooncake_round_trips_populated(self):
        items = [
            MooncakeMMUrlItem(url="http://x/a.jpg", modality=Modality.IMAGE),
            MooncakeMMUrlItem(url="http://x/b.mp4", modality=Modality.VIDEO),
        ]

        decoded = self._round_trip(self._make_request(items))

        self.assertEqual(decoded.mm_data_mooncake, items)
        self.assertEqual(decoded.mm_data_mooncake[0].url, "http://x/a.jpg")
        self.assertEqual(decoded.mm_data_mooncake[0].modality, Modality.IMAGE)
        self.assertEqual(decoded.mm_data_mooncake[1].url, "http://x/b.mp4")
        self.assertEqual(decoded.mm_data_mooncake[1].modality, Modality.VIDEO)

    def test_mm_data_mooncake_round_trips_none(self):
        decoded = self._round_trip(self._make_request(None))

        self.assertIsNone(decoded.mm_data_mooncake)

    def test_mm_data_mooncake_round_trips_empty_list(self):
        decoded = self._round_trip(self._make_request([]))

        self.assertEqual(decoded.mm_data_mooncake, [])

    def test_mm_data_mooncake_round_trips_all_modalities(self):
        for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
            with self.subTest(modality=modality):
                items = [
                    MooncakeMMUrlItem(
                        url=f"http://x/{modality.name.lower()}",
                        modality=modality,
                    )
                ]

                decoded = self._round_trip(self._make_request(items))

                self.assertEqual(decoded.mm_data_mooncake[0].url, items[0].url)
                self.assertEqual(decoded.mm_data_mooncake[0].modality, modality)


if __name__ == "__main__":
    unittest.main()
