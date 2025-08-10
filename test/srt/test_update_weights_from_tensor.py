import gc
import time
import unittest

import torch

import sglang as sgl
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


def test_update_weights_from_tensor(tp_size):
    assert torch.cuda.device_count() >= tp_size, f"At least {tp_size} GPUs are required"
    torch.cuda.empty_cache()

    engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST, tp_size=tp_size)

    param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]

    _check_param(engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

    memory_before = torch.cuda.memory_allocated()
    new_tensor = torch.full((16384, 2048), 1.5, device="cuda")

    time_start = time.perf_counter()
    engine.update_weights_from_tensor([(x, new_tensor) for x in param_names])
    print(f"Time delta: {time.perf_counter() - time_start:.03f}")

    for param_name in param_names[:3]:
        _check_param(engine, param_name, [1.5] * 5)

    engine.shutdown()

    del new_tensor
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    memory_after = torch.cuda.memory_allocated()
    assert (
        memory_after <= memory_before + 1024
    ), f"Memory leak detected: {memory_after - memory_before} bytes"


class TestUpdateWeightsFromTensor(CustomTestCase):
    def test_update_weights_from_tensor(self):
        tp_sizes = [1, 2]
        for tp_size in tp_sizes:
            if torch.cuda.device_count() < tp_size:
                continue

            with self.subTest(tp_size=tp_size):
                test_update_weights_from_tensor(tp_size)

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

    def test_update_weights_from_tensor_load_format_custom(self):
        custom_loader_name = (
            "sglang.srt.model_executor.model_runner._model_load_weights_direct"
        )
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            custom_weight_loader=[custom_loader_name],
        )

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
            load_format=custom_loader_name,
        )

        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        engine.shutdown()

    def test_update_weights_from_tensor_load_format_flattened_bucket(self):
        """Test updating weights using flattened_bucket format"""
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        # Create a small set of parameters for testing
        param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 10)]

        # Check original values
        _check_param(engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

        # Create new tensors with different values
        new_tensors = []
        for _, name in enumerate(param_names):
            # Create tensors with different values for each parameter
            value = 2.0  # Different value for each parameter
            new_tensor = torch.full((16384, 2048), value, device="cuda")
            new_tensors.append((name, new_tensor))

        # Create a flattened bucket
        flattened_bucket = FlattenedTensorBucket(named_tensors=new_tensors)

        # Extract the flattened tensor and metadata in the format expected by model_runner
        flattened_tensor = flattened_bucket.get_flattened_tensor()
        metadata = flattened_bucket.get_metadata()

        # Create the dict format expected by _update_weights_from_flattened_bucket
        bucket_dict = {"flattened_tensor": flattened_tensor, "metadata": metadata}

        # Serialize the bucket data
        from sglang.srt.utils import MultiprocessingSerializer

        serialized_bucket = MultiprocessingSerializer.serialize(
            bucket_dict, output_str=True
        )

        # Create a list where each rank contains the same serialized data
        # This simulates the distributed environment where each rank has the same data
        serialized_bucket_list = [serialized_bucket]

        # Update weights using flattened_bucket format
        time_start = time.perf_counter()
        engine.update_weights_from_tensor(
            named_tensors=serialized_bucket_list, load_format="flattened_bucket"
        )
        update_time = time.perf_counter() - time_start
        print(f"Flattened bucket update time: {update_time:.03f}")

        # Verify the weights were updated correctly
        for i, param_name in enumerate(param_names):
            _check_param(engine, param_name, [2.0] * 5)

        engine.shutdown()


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


if __name__ == "__main__":
    unittest.main()
