import unittest

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from sglang.srt.models.molmo_d import MolmoVisionBackbone, MolmoVisionBackboneConfig
from sglang.test.runners import HFRunner, SRTRunner

PROMPT = "Describe this image."

MODELS = [
    ("allenai/Molmo-7B-D-0924", "bfloat16", 1e-5),
]

CONVS = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://picsum.photos/id/237/536/354"},
                },
            ],
        }
    ],
]


class TestMolmoD(unittest.TestCase):

    def assert_molmo_d_close(self, model_path, torch_dtype, tolerance, conv):
        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(conv)

        with SRTRunner(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(conv)

        assert torch.allclose(
            hf_outputs.top_input_logprobs,
            srt_outputs.top_input_logprobs,
            atol=tolerance,
        ), "top_input_logprobs are not all close"
        assert torch.allclose(
            hf_outputs.top_output_logprobs,
            srt_outputs.top_output_logprobs,
            atol=tolerance,
        ), "top_output_logprobs are not all close"

    def test_molmo_d(self):
        for conv in CONVS:
            for model_path, torch_dtype, tolerance in MODELS:
                self.assert_molmo_d_close(model_path, torch_dtype, tolerance, conv)


if __name__ == "__main__":
    unittest.main()
