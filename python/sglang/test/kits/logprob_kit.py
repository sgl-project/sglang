import multiprocessing as mp
from typing import Any, Dict, List, Optional

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.utils.common import kill_process_tree
from sglang.test.runners import HFRunner, ModelOutput, check_close_model_outputs
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def run_hf_logprob(
    model_path: str,
    prompts: List[str],
    max_new_tokens: int,
    token_ids_logprob: List[int],
    torch_dtype: torch.dtype,
) -> ModelOutput:
    with HFRunner(
        model_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        patch_model_do_sample_false=True,
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            token_ids_logprob=token_ids_logprob,
        )

    return hf_outputs


class LogprobTestBase(CustomTestCase):
    model: Optional[str] = None
    other_args: List[str] = []
    prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "The chemical symbol for water is",
    ]
    token_logprob = [" Paris", " Jupiter", " Oxygen"]
    max_new_tokens = 5
    dtype = torch.float16

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn")
        tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.token_ids_logprob = tokenizer(cls.token_logprob, add_special_tokens=False)[
            "input_ids"
        ]
        # Unwrap from list
        cls.token_ids_logprob = [ids[0] for ids in cls.token_ids_logprob]
        print("Tokens for logprob:", cls.token_logprob)
        print("Token IDs for logprob:", cls.token_ids_logprob)

    def get_hf_logprob_outputs(self) -> Dict[str, Any]:
        return run_hf_logprob(
            self.model,
            self.prompts,
            self.max_new_tokens,
            self.token_ids_logprob,
            self.dtype,
        )

    def get_sglang_logprob_outputs(self) -> Dict[str, Any]:
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            self.model,
            base_url,
            timeout=60,
            other_args=self.other_args,
        )

        payload = {
            "text": self.prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.max_new_tokens,
            },
            "return_logprob": True,
            "top_logprobs_num": 5,
            "return_text_in_logprobs": True,
            "logprob_start_len": 0,
        }
        res = requests.post(base_url + "/generate", json=payload).json()
        kill_process_tree(process.pid)

        output_strs = [choice["text"] for choice in res]
        output_ids = [choice["output_ids"] for choice in res]
        output = ModelOutput(output_strs=output_strs, output_ids=output_ids)
        return output


if __name__ == "__main__":

    class LogprobTestLlama(LogprobTestBase):
        model = "meta-llama/Llama-3.1-8B-Instruct"

    testcase = LogprobTestLlama()
    testcase.setUpClass()
    hf_output = testcase.get_hf_logprob_outputs()
    print(f"{hf_output=}")
    sglang_output = testcase.get_sglang_logprob_outputs()
    print(f"{sglang_output=}")

    check_close_model_outputs(
        hf_outputs=hf_output,
        srt_outputs=sglang_output,
        prefill_tolerance=1e-4,
        decode_tolerance=1e-4,
        rouge_l_tolerance=1e-4,
        debug_text="LogprobTestLlama",
    )
