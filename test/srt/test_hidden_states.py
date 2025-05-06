import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.srt.utils import get_device
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class TestHiddenState(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()

    def test_return_hidden_states(self):
        prompts = ["Today is", "Today is a sunny day and I like"]
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 8,
        }

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
        )
        outputs = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        engine.shutdown()

        for output in outputs:
            self.assertEqual(len(output["meta_info"]["hidden_states"]), 8)
            for i in range(len(output["meta_info"]["hidden_states"])):
                assert isinstance(output["meta_info"]["hidden_states"][i], list)
                output["meta_info"]["hidden_states"][i] = torch.tensor(
                    output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                )
        # Checks that splicing of the batch was done correctly
        self.assertGreater(
            outputs[1]["meta_info"]["hidden_states"][0].shape[0],
            outputs[0]["meta_info"]["hidden_states"][0].shape[0],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device
        )

        for input_id, output in zip(input_ids, outputs):
            with torch.inference_mode():
                hf_out = model(
                    torch.tensor(
                        [input_id + output["output_ids"][:-1]], device=model.device
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
            ).to(self.device)
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

    def test_repeatedly_changes_hidden_states(self):
        prompts = ["Today is", "Today is a sunny day and I like"]
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 8,
        }

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
        )
        outputs_completion_first_round = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        outputs_hidden_state = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=False,
        )

        outputs_completion_last_round = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        engine.shutdown()

        for (
            output_completion_first_round,
            output_hidden_state,
            output_completion_last_round,
        ) in zip(
            outputs_completion_first_round,
            outputs_hidden_state,
            outputs_completion_last_round,
        ):
            self.assertEqual(
                len(output_completion_first_round["meta_info"]["hidden_states"]), 8
            )
            self.assertNotIn("hidden_states", output_hidden_state["meta_info"])
            self.assertEqual(
                len(output_completion_last_round["meta_info"]["hidden_states"]), 8
            )


if __name__ == "__main__":
    unittest.main()
