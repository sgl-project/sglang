import gc
import unittest

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)
from sglang.utils import terminate_process


class TestUpdateWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST.replace("2157", "1999")
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            cls.model, torch_dtype="bfloat16"
        ).to("cuda:0")

    def init_backend(cls, backend, dp, tp):
        cls.engine = None
        cls.process = None
        cls.backend = backend
        cls.dp = dp
        cls.tp = tp
        if backend == "Engine":
            cls.engine = sgl.Engine(
                model_path=cls.model,
                random_seed=42,
                tp_size=tp,
                dp_size=dp,
                mem_fraction_static=0.85,
            )
        else:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=(
                    "--tp-size",
                    str(tp),
                    "--dp-size",
                    str(dp),
                ),
            )

    def close_engine_and_server(cls):
        if cls.engine:
            cls.engine.shutdown()
        if cls.process:
            terminate_process(cls.process)

    @classmethod
    def tearDownClass(cls):
        del cls.hf_model
        gc.collect()
        torch.cuda.empty_cache()

    def assert_update_weights_all_close(cls, param_name, truncate_size):
        print(
            f"param_name: {param_name}, backend: {cls.backend}, dp: {cls.dp}, tp: {cls.tp}"
        )
        param = cls.hf_model.get_parameter(param_name)[:truncate_size]
        param_np = param.cpu().detach().float().numpy()

        if cls.backend == "Engine":
            engine_ret = cls.engine.get_weights_by_name(param_name, truncate_size)
            engine_ret = cls._process_return(engine_ret)
            np.testing.assert_allclose(engine_ret, param_np, rtol=1e-5, atol=1e-5)

        if cls.backend == "Runtime":
            runtime_ret = requests.get(
                f"{cls.base_url}/get_weights_by_name",
                json={"name": param_name, "truncate_size": truncate_size},
            ).json()
            runtime_ret = cls._process_return(runtime_ret)
            np.testing.assert_allclose(runtime_ret, param_np, rtol=1e-5, atol=1e-5)

    @staticmethod
    def _process_return(ret):
        if isinstance(ret, list) and len(ret) == 2:
            print(f"running assert_allclose on data parallel")
            np.testing.assert_allclose(ret[0], ret[1])
            return np.array(ret[0])
        return np.array(ret)

    def test_update_weights_unexist_model(cls):
        test_suits = [("Engine", 1, 1), ("Runtime", 1, 1)]

        if torch.cuda.device_count() >= 2:
            test_suits.append(("Engine", 1, 2))
            test_suits.append(("Runtime", 2, 1))

        if torch.cuda.device_count() >= 4:
            test_suits.extend([("Engine", 2, 2), ("Runtime", 2, 2)])

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

        for test_suit in test_suits:
            cls.init_backend(*test_suit)
            for param_name in parameters:
                cls.assert_update_weights_all_close(param_name, 100)
            cls.close_engine_and_server()


if __name__ == "__main__":
    unittest.main()
