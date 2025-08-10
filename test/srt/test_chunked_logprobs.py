import os
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_logprob_check,
)

model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

fields = [
    "input_token_logprobs",
    "input_token_ids_logprobs",
    "output_token_logprobs",
    "output_token_ids_logprobs",
    "input_top_logprobs",
    "output_top_logprobs",
]


def recursive_allclose(a, b, rtol=0.1, atol=0.1):
    """
    Recursively check if two nested structures are close.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        return torch.isclose(torch.tensor(a), torch.tensor(b), rtol=rtol, atol=atol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(recursive_allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))
    return False


class TestChunkedLogprobsAccuracy(CustomTestCase):
    def test_logprobs_accuracy(self):
        top_logprobs_num = 50
        first_n_token_ids = 5

        prompts = [
            "Hello, my name is",
            "The future of AI is",
            "The president of the United States is",
            "The capital of France is ",
        ]

        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 100,
            "max_new_tokens": 32,
        }

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        # set chunk size to 1 to test the chunked logprobs
        os.environ["SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"] = "1"

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
        )
        outputs = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=list(range(first_n_token_ids)),
        )
        engine.shutdown()
        del engine
        torch.cuda.empty_cache()

        # check the logprobs
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
        for input_id, output in zip(input_ids, outputs):
            input_token_logprobs = output["meta_info"]["input_token_logprobs"]
            input_top_logprobs = output["meta_info"]["input_top_logprobs"]
            input_token_ids_logprobs = output["meta_info"]["input_token_ids_logprobs"]
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            output_top_logprobs = output["meta_info"]["output_top_logprobs"]
            output_token_ids_logprobs = output["meta_info"]["output_token_ids_logprobs"]

            # Here we ignore the first token logprob, which is None
            input_token_logprobs_tensor = torch.tensor(
                [a for a, _, _ in input_token_logprobs[1:]]
            ).to(model.device)
            input_token_logprobs_indices = torch.tensor(
                [b for _, b, _ in input_token_logprobs[1:]]
            ).to(model.device)
            data = [[x[0] for x in inner] for inner in input_top_logprobs[1:]]
            input_top_logprobs_tensor = torch.tensor(data).to(model.device)
            data = [[a for a, _, _ in row] for row in input_token_ids_logprobs[1:]]
            input_token_ids_logprobs_tensor = torch.tensor(data).to(model.device)

            output_token_logprobs_tensor = torch.tensor(
                [a for a, _, _ in output_token_logprobs]
            ).to(model.device)
            output_token_logprobs_indices = torch.tensor(
                [b for _, b, _ in output_token_logprobs]
            ).to(model.device)

            data = [[x[0] for x in inner] for inner in output_top_logprobs]
            output_top_logprobs_tensor = torch.tensor(data).to(model.device)
            data = [[a for a, _, _ in row] for row in output_token_ids_logprobs]
            output_token_ids_logprobs_tensor = torch.tensor(data).to(model.device)

            # Concatenate input and output tensors
            srt_token_logprobs = torch.cat(
                [input_token_logprobs_tensor, output_token_logprobs_tensor], dim=0
            )
            srt_token_logprobs_indices = torch.cat(
                [input_token_logprobs_indices, output_token_logprobs_indices], dim=0
            )
            srt_top_logprobs = torch.cat(
                [input_top_logprobs_tensor, output_top_logprobs_tensor], dim=0
            )
            srt_token_ids_logprobs = torch.cat(
                [input_token_ids_logprobs_tensor, output_token_ids_logprobs_tensor],
                dim=0,
            )

            with torch.inference_mode():
                hf_out = model(
                    torch.tensor(
                        [input_id + output["output_ids"][:-1]], device=model.device
                    ),
                    return_dict_in_generate=True,
                    output_scores=True,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    top_k=sampling_params["top_k"],
                )

            for logits in hf_out.logits:
                # It seems we can only get logits instead of logprobs from HF,
                # so log_softmax is applied on logits to get the log probabilities.
                # using float to align with the sglang. The temperature is 1.0 and
                # top_p is 1.0, so no further processing is needed.
                log_probs = F.log_softmax(logits.float(), dim=-1)

                rtol = 0.2

                print("================")
                # check token_logprobs
                hf_token_logprobs = log_probs[
                    torch.arange(log_probs.shape[0]), srt_token_logprobs_indices
                ]
                self.assertTrue(
                    torch.allclose(
                        hf_token_logprobs, srt_token_logprobs, atol=0, rtol=rtol
                    )
                )
                print(
                    f"Max diff for token_logprobs: {torch.max(torch.abs(hf_token_logprobs - srt_token_logprobs))}"
                )

                # check top_logprobs
                hf_top_logprobs, _ = torch.topk(log_probs, k=top_logprobs_num, dim=-1)
                print(
                    f"Max diff for top_logprobs: {torch.max(torch.abs(hf_top_logprobs - srt_top_logprobs))}"
                )

                self.assertTrue(
                    torch.allclose(
                        hf_top_logprobs,
                        srt_top_logprobs,
                        atol=0,
                        rtol=rtol,
                    )
                )

                # check token_ids_logprobs
                hf_token_ids_logprobs = log_probs[:, :first_n_token_ids]
                self.assertTrue(
                    torch.allclose(
                        hf_token_ids_logprobs, srt_token_ids_logprobs, atol=0, rtol=rtol
                    )
                )
                print(
                    f"Max diff for token_ids_logprobs: {torch.max(torch.abs(hf_token_ids_logprobs - srt_token_ids_logprobs))}"
                )
        del model
        torch.cuda.empty_cache()


class TestChunkedLogprobsChunkSizeVariations(CustomTestCase):
    def test_logprobs_chunk_size_variations(self):
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 8,
        }
        prompts = [
            "Hello, my name is",
            "The future of AI is",
            "The president of the United States is",
            "The capital of France is ",
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # create longer input_ids to test the chunked logprobs
        input_ids = tokenizer(prompts).input_ids
        input_ids = [
            input * (n // len(input))
            for (input, n) in zip(input_ids, [512, 1024, 2048, 4096])
        ]

        for temperature in [0.9, 1.0]:
            sampling_params["temperature"] = temperature
            # result1: chunk size = 256
            # result2: chunk size = 2048
            # result3: chunk size = 4096, test the old code path
            results = []
            for chunk_size in [256, 2048, 4096]:
                print(f"set SGLANG_LOGITS_PROCESSER_CHUNK_SIZE to {chunk_size}")
                os.environ["SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"] = str(chunk_size)
                engine = sgl.Engine(
                    model_path=model_path,
                    random_seed=42,
                    skip_tokenizer_init=True,
                    mem_fraction_static=0.6,
                )
                outputs = engine.generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=0,
                    top_logprobs_num=5,
                    token_ids_logprob=list(range(5)),
                )
                outputs = sorted(outputs, key=lambda x: x["output_ids"])
                results.append(outputs)
                engine.shutdown()
                del engine
                torch.cuda.empty_cache()

            # compare the results
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    print(f"Comparing result[{i}] and result[{j}]:")
                    for output1, output2 in zip(results[i], results[j]):
                        for field in fields:
                            v1 = output1["meta_info"][field]
                            v2 = output2["meta_info"][field]
                            assert recursive_allclose(
                                v1, v2, rtol=0.02, atol=0.0
                            ), f"Mismatch: {field} between result[{i}] and result[{j}]"
                    print(f"result[{i}] and result[{j}] are numerically close")


class TestChunkedLogprobsMix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model_path
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--mem-fraction-static",
                "0.6",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_logprobs_mix(self):
        args = []
        # input_len, output_len, temperature, logprob_start_len, return_logprob, top_logprobs_num
        for input_len in [1000, 5000, 10000, 50000]:
            for output_len in [4, 8]:
                for temperature in [0.1, 1.0]:
                    for logprob_start_len in [0, 500, 2500, 5000, 25000]:
                        for return_logprob in [True, False]:
                            for top_logprobs_num in [0, 5]:

                                if logprob_start_len >= input_len:
                                    continue

                                args.append(
                                    (
                                        input_len,
                                        output_len,
                                        temperature,
                                        logprob_start_len,
                                        return_logprob,
                                        top_logprobs_num,
                                    )
                                )

        random.shuffle(args)

        func = partial(run_logprob_check, self)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(func, args))


if __name__ == "__main__":
    unittest.main()
