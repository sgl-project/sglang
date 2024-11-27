import gc
import json
import time
import unittest
from itertools import product
from signal import signal

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
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            cls.model, torch_dtype="bfloat16"
        ).to("cuda:0")

    @classmethod
    def init_engine_and_server(cls, engine_tp, server_tp, engine_dp, server_dp):
        cls.engine = sgl.Engine(
            model_path=cls.model,
            random_seed=42,
            tp_size=engine_tp,
            dp_size=engine_dp,
            base_gpu_id=0,
            mem_fraction_static=0.85,
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--base-gpu-id",
                str(engine_dp * engine_tp),
                "--tp-size",
                str(server_tp),
                "--dp-size",
                str(server_dp),
            ),
        )

    @classmethod
    def close_engine_and_server(cls):
        cls.engine.shutdown()
        terminate_process(cls.process)

    @classmethod
    def tearDownClass(cls):
        del cls.hf_model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def assert_update_weights_all_close(cls, param_name, truncate_size):
        param = cls.hf_model.get_parameter(param_name)[:truncate_size]
        engine_ret = cls.engine.get_weights_by_parameter_name(param_name, truncate_size)

        # 如果 engine_ret 是标量值的列表
        if isinstance(engine_ret, list) and len(engine_ret) == 2:
            print("working in dp, engine_ret is a list of two elements")
            np.testing.assert_allclose(engine_ret[0], engine_ret[1])
            engine_ret = engine_ret[0]

        engine_ret = np.array(engine_ret)

        np.testing.assert_allclose(
            engine_ret, param.cpu().detach().float().numpy(), rtol=1e-5, atol=1e-5
        )

        runtime_ret = requests.get(
            f"{cls.base_url}/get_weights_by_parameter_name",
            json={
                "name": param_name,
                "truncate_size": truncate_size,
            },
        ).json()

        # 处理 runtime_ret 的情况
        if isinstance(runtime_ret, list) and len(runtime_ret) == 2:
            np.testing.assert_allclose(runtime_ret[0], runtime_ret[1])
            runtime_ret = runtime_ret[0]

        np.testing.assert_allclose(
            runtime_ret,
            param.cpu().detach().float().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    @classmethod
    def test_update_weights_unexist_model(cls):

        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        test_suits = [(1, 1, 1, 1)]

        if torch.cuda.device_count() >= 4:
            test_suits.extend([(2, 2, 1, 1), (1, 1, 2, 2)])

        if torch.cuda.device_count() >= 8:
            test_suits.append((2, 2, 2, 2))

        parameters = [
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "lm_head.weight",
        ]

        for test_suit in test_suits:
            cls.init_engine_and_server(*test_suit)
            for param_name in parameters:
                cls.assert_update_weights_all_close(param_name, 100)
            cls.close_engine_and_server()


if __name__ == "__main__":
    unittest.main()
