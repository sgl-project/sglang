import time
import unittest

import requests
import torch

from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


def check_quant_method(model_path: str, use_marlin_kernel: bool):
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
    from sglang.srt.layers.quantization.utils import get_dynamic_override
    from sglang.srt.model_loader import get_model
    from sglang.srt.server_args import ServerArgs

    try:
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:2646",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        monkey_patch_vllm_parallel_state()
    except AssertionError:
        # ignore this error: tensor model parallel group is already initialized
        pass

    server_args = ServerArgs(model_path=model_path, dtype=torch.float16)
    set_global_server_args_for_scheduler(server_args)
    model_config = ModelConfig.from_server_args(server_args)

    load_config = LoadConfig()
    device_config = DeviceConfig(device_type)
    model = get_model(
        model_config=model_config, load_config=load_config, device_config=device_config
    )

    from sglang.srt.layers.linear import UnquantizedLinearMethod
    from sglang.srt.layers.quantization.gptq import (
        GPTQLinearMethod,
        GPTQMarlinLinearMethod,
    )

    linear_method_cls = (
        GPTQMarlinLinearMethod if use_marlin_kernel else (GPTQLinearMethod)
    )

    for name, submodule in model.named_modules():
        if name == "lm_head":
            assert isinstance(submodule.quant_method, linear_method_cls)
        elif name == "model.layers.0.self_attn.qkv_proj":
            # The first layer is quantized using bits=4, group_size=128
            # desc_act=True
            assert isinstance(submodule.quant_method, linear_method_cls)
            config = submodule.quant_method.quant_config
            assert config.weight_bits == 4
            assert config.group_size == 128
            assert config.desc_act
        elif name == "model.layers.1.self_attn.qkv_proj":
            # The second layer is quantized using bits=8, group_size=32
            # desc_act=False
            assert isinstance(submodule.quant_method, linear_method_cls)
            config = submodule.quant_method.quant_config
            assert get_dynamic_override(config, layer_name=name, key="bits") == 8
            assert get_dynamic_override(config, layer_name=name, key="group_size") == 32
            assert not get_dynamic_override(config, layer_name=name, key="desc_act")
        elif (
            name == "model.layers.2.self_attn.qkv_proj"
            or name == "model.layers.2.mlp.gate_up_proj"
        ):
            # All other layers (layer index >= 2) are not quantized
            assert isinstance(submodule.quant_method, UnquantizedLinearMethod)

    del model


# GPTQ with Dynamic Per/Module Quantization Control
# Leverages GPTQModel (pypi) to produce the `dynamic` models
# Test GPTQ fallback kernel that is not Marlin
class TestGPTQModelDynamic(CustomTestCase):
    MODEL_PATH = (
        "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symFalse"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = cls.MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "float16"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.001,
                },
            },
        )
        return response.json()

    def test_throughput(self):
        max_tokens = 256

        tic = time.perf_counter()
        result = self.run_decode(max_tokens)
        tok = time.perf_counter()

        print(f"result = `{result}`")

        self.assertIn("paris", result["text"].lower())

        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        self.assertGreaterEqual(throughput, 140)

    def test_gptq_module(self):
        check_quant_method(self.MODEL_PATH, use_marlin_kernel=False)


# GPTQ with Dynamic Per/Module Quantization Control
# Leverages GPTQModel (pypi) to produce the `dynamic` models
# Test Marlin kernel
class TestGPTQModelDynamicWithMarlin(CustomTestCase):
    MODEL_PATH = (
        "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = cls.MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "bfloat16"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.001,
                },
            },
        )
        return response.json()

    def test_throughput(self):
        max_tokens = 256

        tic = time.perf_counter()
        result = self.run_decode(max_tokens)
        tok = time.perf_counter()

        print(f"result = `{result}`")

        assert "paris" in result["text"].lower()

        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        assert throughput >= 140

    def test_gptq_marlin_module(self):
        check_quant_method(self.MODEL_PATH, use_marlin_kernel=True)


if __name__ == "__main__":
    unittest.main()
