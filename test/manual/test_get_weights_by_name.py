import gc
import unittest

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import terminate_process


def _process_return(ret):
    if isinstance(ret, list) and len(ret) == 2:
        print(f"running assert_allclose on data parallel")
        np.testing.assert_allclose(ret[0], ret[1])
        return np.array(ret[0])
    return np.array(ret)


class TestGetWeightsByName(CustomTestCase):

    def init_hf_model(self, model_name, tie_word_embeddings):
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16", tie_word_embeddings=tie_word_embeddings
        ).to("cuda:0")

    def init_backend(self, backend, dp, tp, model_name):
        self.backend = backend
        self.dp = dp
        self.tp = tp
        if backend == "Engine":
            self.engine = sgl.Engine(
                model_path=model_name,
                random_seed=42,
                tp_size=tp,
                dp_size=dp,
            )
        else:
            self.process = popen_launch_server(
                model_name,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=(
                    "--tp-size",
                    str(tp),
                    "--dp-size",
                    str(dp),
                ),
            )

    def clean_up(self):
        del self.hf_model
        gc.collect()
        torch.cuda.empty_cache()
        if self.backend == "Engine":
            self.engine.shutdown()
        else:
            terminate_process(self.process)

    def assert_tie_word_embeddings(self, truncate_size):
        print("assert_tie_word_embeddings")
        if self.backend == "Engine":
            backend_ret = _process_return(
                self.engine.get_weights_by_name("lm_head.weight", truncate_size)
            )
        else:
            backend_ret = _process_return(
                requests.get(
                    f"{DEFAULT_URL_FOR_TEST}/get_weights_by_name",
                    json={"name": "lm_head.weight", "truncate_size": truncate_size},
                ).json()
            )
        print("assert_tie_word_embeddings of hf and backend")
        assert np.allclose(
            self.hf_model.get_parameter("model.embed_tokens.weight")
            .cpu()
            .detach()
            .float()
            .numpy()[:truncate_size],
            backend_ret,
        )
        assert np.allclose(
            self.hf_model.get_parameter("lm_head.weight")
            .cpu()
            .detach()
            .float()
            .numpy()[:truncate_size],
            self.hf_model.get_parameter("model.embed_tokens.weight")
            .cpu()
            .detach()
            .float()
            .numpy()[:truncate_size],
        )

    def assert_weights_all_close(self, param_name, truncate_size):
        print(
            f"param_name: {param_name}, backend: {self.backend}, dp: {self.dp}, tp: {self.tp}"
        )
        param = self.hf_model.get_parameter(param_name)[:truncate_size]
        param_np = param.cpu().detach().float().numpy()

        if self.backend == "Engine":
            engine_ret = self.engine.get_weights_by_name(param_name, truncate_size)
            engine_ret = _process_return(engine_ret)
            np.testing.assert_allclose(engine_ret, param_np, rtol=1e-5, atol=1e-5)

        if self.backend == "Runtime":
            runtime_ret = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/get_weights_by_name",
                json={"name": param_name, "truncate_size": truncate_size},
            ).json()
            runtime_ret = _process_return(runtime_ret)
            np.testing.assert_allclose(runtime_ret, param_np, rtol=1e-5, atol=1e-5)

    def test_get_weights_by_name(self):
        if is_in_ci():
            test_suits = [
                ("Engine", 1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
            ]
        else:
            test_suits = [
                ("Runtime", 1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
                ("Engine", 1, 1, DEFAULT_MODEL_NAME_FOR_TEST),
            ]
            if torch.cuda.device_count() >= 2:
                test_suits.append(("Engine", 1, 2, DEFAULT_SMALL_MODEL_NAME_FOR_TEST))
                test_suits.append(("Runtime", 2, 1, DEFAULT_MODEL_NAME_FOR_TEST))

            if torch.cuda.device_count() >= 4:
                test_suits.extend(
                    [
                        ("Engine", 2, 2, DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
                        ("Runtime", 2, 2, DEFAULT_MODEL_NAME_FOR_TEST),
                    ]
                )

        parameters = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.6.mlp.up_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.layers.8.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]

        truncate_size = 100

        for test_suit in test_suits:
            if test_suit[-1] == DEFAULT_MODEL_NAME_FOR_TEST:
                tie_word_embeddings = False
            else:
                tie_word_embeddings = True

            self.init_hf_model(test_suit[-1], tie_word_embeddings)
            self.init_backend(*test_suit)

            for param_name in parameters:
                self.assert_weights_all_close(param_name, truncate_size)

            if tie_word_embeddings:
                self.assert_tie_word_embeddings(truncate_size)

            self.clean_up()


if __name__ == "__main__":
    unittest.main()
