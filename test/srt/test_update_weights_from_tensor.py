import gc
import time
import unittest

import torch

import sglang as sgl
from sglang.srt.utils import get_device, get_device_count
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


def test_update_weights_from_tensor(tp_size, device):
    module = torch.get_device_module(device)
    assert module.device_count() >= tp_size, f"At least {tp_size} GPUs are required"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST, tp_size=tp_size)

    param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]

    _check_param(engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

    memory_before = module.memory_allocated()
    new_tensor = torch.full((16384, 2048), 1.5, device=device)

    time_start = time.time()
    engine.update_weights_from_tensor([(x, new_tensor) for x in param_names])
    print(f"Time delta: {time.time() - time_start:.03f}")

    for param_name in param_names[:3]:
        _check_param(engine, param_name, [1.5] * 5)

    engine.shutdown()

    del new_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    memory_after = module.memory_allocated()
    assert (
        memory_after <= memory_before + 1024
    ), f"Memory leak detected: {memory_after - memory_before} bytes"


class TestUpdateWeightsFromTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.num_gpus = get_device_count()

    def test_update_weights_from_tensor(self):
        tp_sizes = [1, 2]
        for tp_size in tp_sizes:
            if self.num_gpus < tp_size:
                continue

            with self.subTest(tp_size=tp_size):
                test_update_weights_from_tensor(tp_size, self.device)

    def test_update_weights_from_tensor_load_format_direct(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]

        _check_param(
            engine, read_param_names[0], [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178]
        )

        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor(
            [
                (write_param_name, new_tensor.clone())
                for write_param_name in write_param_names
            ],
            load_format="direct",
        )

        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        engine.shutdown()


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


if __name__ == "__main__":
    unittest.main()
