import gc
import time
import unittest

import torch

import sglang as sgl
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_npu,
)


class TestUpdateWeightsFromTensor(CustomTestCase):
    def test_update_weights_from_tensor_load_format_custom(self):
        custom_loader_name = (
            "sglang.srt.model_executor.model_runner._model_load_weights_direct"
        )
        model_path = (
            "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
            if is_npu()
            else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )
        print(".......before start engine.....................")
        engine = sgl.Engine(
            model_path=model_path,
            custom_weight_loader=[custom_loader_name],
            disable_cuda_graph=True,
            mem_fraction_static=0.7,
            base_gpu_id=0,
        )
        print("after start engine.....................")
        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]
        pre_matrix = engine.get_weights_by_name(read_param_names[0], truncate_size=100)
        print("3 start engine.....................")
        expect_values = []
        if pre_matrix:
            expect_values = (
                pre_matrix[0][:5] if len(pre_matrix[0]) > 5 else pre_matrix[0]
            )
            _check_param(engine, read_param_names[0], expect_values)
        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor(
            [
                (write_param_name, new_tensor.clone())
                for write_param_name in write_param_names
            ],
            load_format=custom_loader_name,
            flush_cache=False,
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
