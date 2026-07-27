import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=55, suite="stage-b-test-1-gpu-small-amd")

_is_hip = is_hip()
if _is_hip:
    import os

    os.environ["SGLANG_USE_AITER"] = "0"


class TestHiddenState(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.prompts = ["Today is", "Today is a sunny day and I like"]
        cls.input_ids = cls.tokenizer(cls.prompts).input_ids
        cls.sampling_params = {"temperature": 0, "max_new_tokens": 8}
        # mem_fraction_static=0.7 leaves headroom for the HF reference
        # model that test_return_hidden_states loads on the same GPU.
        cls.engine = sgl.Engine(
            model_path=cls.model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            enable_return_hidden_states=True,
            mem_fraction_static=0.7,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def setUp(self):
        # Tests share one Engine; flush radix cache so each test sees a
        # cold prefill (test_return_hidden_states asserts on the prefill
        # hidden-state shape, which collapses to 0 on a full cache hit).
        self.engine.flush_cache()

    def test_return_hidden_states(self):
        outputs = self.engine.generate(
            input_ids=self.input_ids,
            sampling_params=self.sampling_params,
            return_hidden_states=True,
        )

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
            self.model_path, torch_dtype=torch.bfloat16, device_map=get_device()
        )

        for input_id, output in zip(self.input_ids, outputs):
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
            ).to(get_device())
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
        outputs_completion_first_round = self.engine.generate(
            input_ids=self.input_ids,
            sampling_params=self.sampling_params,
            return_hidden_states=True,
        )
        outputs_hidden_state = self.engine.generate(
            input_ids=self.input_ids,
            sampling_params=self.sampling_params,
            return_hidden_states=False,
        )

        outputs_completion_last_round = self.engine.generate(
            input_ids=self.input_ids,
            sampling_params=self.sampling_params,
            return_hidden_states=True,
        )

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
