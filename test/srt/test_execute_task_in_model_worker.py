import unittest
from dataclasses import dataclass

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.test_utils import CustomTestCase


def simple_task_func(**kwargs):
    # This function will be run on all model workers
    return 12345


def get_model_param_info(**kwargs):
    # kwargs will contain 'model' and 'model_runner'
    model = kwargs["model"]
    param_info = []
    for name, param in model.named_parameters():
        param_info.append(
            {
                "name": name,
                "numel": param.numel(),
                "shape": tuple(param.shape),
                "dtype": str(param.dtype),
            }
        )
    return param_info


@dataclass
class ParamInfo:
    name: str
    shape: tuple
    numel: int
    dtype: torch.dtype


def get_model_param_info_struct(**kwargs):
    # kwargs will contain 'model' and 'model_runner'
    model = kwargs["model"]
    param_info = []
    for name, param in model.named_parameters():
        param_info.append(
            ParamInfo(
                name=name,
                shape=tuple(param.shape),
                numel=param.numel(),
                dtype=param.dtype,
            )
        )
    return param_info


class TestExecuteTaskInModelWorker(CustomTestCase):

    def _run_test(self, tp_size, dp_size, enable_dp_attention, expected_len):
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        engine = Engine(
            model_path=model_path,
            served_model_name=model_path,
            attention_backend="torch_native",
            tp_size=tp_size,
            trust_remote_code=True,
            dp_size=dp_size,
            enable_dp_attention=enable_dp_attention,
            log_level="error",
            stream_output=False,
        )

        try:
            # Test simple_task_func
            results = engine.execute_task_in_model_worker(simple_task_func)
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), expected_len)
            for r in results:
                self.assertEqual(r, 12345)

            # Test get_model_param_info
            param_info_results = engine.execute_task_in_model_worker(
                get_model_param_info
            )
            self.assertIsInstance(param_info_results, list)
            self.assertEqual(len(param_info_results), expected_len)
            for param_info in param_info_results:
                self.assertIsInstance(param_info, list)
                for param in param_info:
                    self.assertIn("name", param)
                    self.assertIn("numel", param)
                    self.assertIn("shape", param)
                    self.assertIn("dtype", param)

            all_weight_names = [
                param["name"]
                for param_info in param_info_results
                for param in param_info
            ]
            self.assertIn("model.embed_tokens.weight", all_weight_names)
            self.assertIn("model.layers.0.self_attn.qkv_proj.weight", all_weight_names)
            self.assertIn("model.layers.0.mlp.gate_up_proj.weight", all_weight_names)
            self.assertIn("model.layers.1.input_layernorm.weight", all_weight_names)

            # Test struct serialization
            param_info_results = engine.execute_task_in_model_worker(
                get_model_param_info_struct
            )
            for param_info in param_info_results:
                self.assertIsInstance(param_info, list)
                for param in param_info:
                    self.assertTrue(param.name)
                    self.assertGreater(param.numel, 0)
                    self.assertGreaterEqual(len(param.shape), 1)
                    self.assertIsInstance(param.dtype, torch.dtype)
        finally:
            engine.shutdown()

    def test_tp1_dp1(self):
        self._run_test(tp_size=1, dp_size=1, enable_dp_attention=False, expected_len=1)

    def test_tp4_dp1(self):
        self._run_test(tp_size=4, dp_size=1, enable_dp_attention=False, expected_len=4)

    def test_tp4_dp2_attention(self):
        self._run_test(tp_size=4, dp_size=2, enable_dp_attention=True, expected_len=8)


if __name__ == "__main__":
    unittest.main()
