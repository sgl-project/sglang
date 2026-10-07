"""The `/generate` schema gate (`entrypoints/api_contract.py`): the typed media
contract that replaced the raw_json passthrough, checked without a server."""

import unittest

from sglang.srt.entrypoints.api_contract import generate_contract_error
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGenerateContract(CustomTestCase):
    def test_every_media_container_form_is_accepted(self):
        for image_data in (
            "u",
            {"url": "u", "detail": "high"},
            ["u", None, {"url": "v"}],
            [],
            [["a", "b"], None, []],
        ):
            with self.subTest(image_data=image_data):
                self.assertIsNone(
                    generate_contract_error({"text": "hi", "image_data": image_data})
                )
        self.assertIsNone(
            generate_contract_error({"text": "hi", "mm_hashes": ["a1b2", "0xff"]})
        )
        self.assertIsNone(
            generate_contract_error(
                {
                    "text": "hi",
                    "video_data": {"url": "v", "fps": 2.0, "use_audio": True},
                }
            )
        )

    def test_unknown_media_hint_is_rejected(self):
        """A hint the schema does not name used to be silently dropped."""
        error = generate_contract_error(
            {
                "text": "hi",
                "image_data": {"url": "u", "max_dynamic_patc": 4},
            }  # codespell:ignore patc
        )
        self.assertIsNotNone(error)
        self.assertIn("unknown field", error)

    def test_preprocessed_inputs_are_not_an_http_shape(self):
        """processor_output / precomputed_embedding carry tensors: Engine-only."""
        error = generate_contract_error(
            {"input_ids": [1, 2], "image_data": [{"format": "processor_output"}]}
        )
        self.assertIsNotNone(error)

    def test_mixed_list_shapes_are_rejected(self):
        error = generate_contract_error({"text": "hi", "image_data": ["a", ["b"]]})
        self.assertIsNotNone(error)

    def test_sampling_params_follow_the_schema(self):
        self.assertIsNone(
            generate_contract_error(
                {"text": "hi", "sampling_params": {"beam_width": 2, "n": 2}}
            )
        )
        bad = {"temperatur": 1}  # codespell:ignore temperatur
        error = generate_contract_error({"text": "hi", "sampling_params": bad})
        self.assertIsNotNone(error)
        self.assertIn("unknown field", error)


if __name__ == "__main__":
    unittest.main()
