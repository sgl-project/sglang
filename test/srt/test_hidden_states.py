import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import is_in_ci


class TestHiddenState(unittest.TestCase):
    def test_return_hidden_states(self):
        prompts = ["Today is", "Today is a sunny day and I like"]
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            return_hidden_states=True,
            skip_tokenizer_init=True,
        )
        outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        engine.shutdown()

        for output in outputs:
            self.assertEqual(len(output["meta_info"]["hidden_states"]), 8)
            for hidden_state in output["meta_info"]["hidden_states"]:
                self.assertIsInstance(hidden_state, torch.Tensor)
        # Checks that splicing of the batch was done correctly
        self.assertGreater(
            outputs[1]["meta_info"]["hidden_states"][0].shape[0],
            outputs[0]["meta_info"]["hidden_states"][0].shape[0],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )

        for input_id, output in zip(input_ids, outputs):
            with torch.inference_mode():
                hf_out = model(
                    torch.tensor(
                        [input_id + output["token_ids"][:-1]], device=model.device
                    ),
                    output_hidden_states=True,
                )
            print("=== HF Hiddens ===")
            print(hf_out["hidden_states"][-1][0])
            sg_hidden_states = torch.cat(
                [
                    i.unsqueeze(0) if len(i.shape) == 1 else i
                    for i in output["meta_info"]["hidden_states"]
                ]
            ).to("cuda")
            print("=== SRT Hiddens ===")
            print(sg_hidden_states)

            print(
                f"Max diff: {torch.max(torch.abs(hf_out['hidden_states'][-1][0] - sg_hidden_states))}"
            )

            atol = 0.8
            self.assertTrue(
                torch.allclose(
                    hf_out["hidden_states"][-1][0],
                    sg_hidden_states,
                    atol=atol,
                    rtol=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
