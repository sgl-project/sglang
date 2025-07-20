import os
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class TestLogprobs(CustomTestCase):
    def test_logprobs(self):
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
            "top_k": 10,
            "max_new_tokens": 32,
            "n": 1,
        }

        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # model_path = "/workspace/models/Llama/Llama-3.2-1B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
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
                # using float to align with the sglang. the temperature is 1.0 and
                # top_p is 1.0, so no further processing is needed.
                log_probs = F.log_softmax(logits.float(), dim=-1)

                # TODO: the rtol is set to 0.2 because the logprobs from HF is not
                # exactly the same as the sglang. we need to find the reason.
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


if __name__ == "__main__":
    unittest.main()
