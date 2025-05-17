import multiprocessing as mp
import unittest

import torch

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from sglang.test.runners import HFRunner, SRTRunner, check_close_model_outputs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "AI is a field of computer science focused on",
]


class TestHybridAttnBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_hybrid_attn_backend(self):
        prefill_tolerance: float = 5e-2
        decode_tolerance: float = 5e-2
        rouge_l_tolerance: float = 1
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        max_new_tokens = 32
        torch_dtype = torch.float16

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                DEFAULT_PROMPTS, max_new_tokens=max_new_tokens
            )

        with SRTRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=True,
            prefill_attention_backend="fa3",
            decode_attention_backend="flashinfer",
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                DEFAULT_PROMPTS, max_new_tokens=max_new_tokens
            )

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=prefill_tolerance,
            decode_tolerance=decode_tolerance,
            rouge_l_tolerance=rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={DEFAULT_PROMPTS}",
        )


if __name__ == "__main__":
    unittest.main()
