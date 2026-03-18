import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class DummyBackend(BaseLoRABackend):
    name = "triton"


class UnsupportedBackend(BaseLoRABackend):
    name = "custom"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def make_base_hf_config():
    return SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=6,
        vocab_size=16,
    )


def make_lora_config():
    return LoRAConfig.from_dict(
        {
            "peft_type": "LORA",
            "target_modules": ["q_proj", "v_proj", "gate_proj"],
            "r": 2,
            "lora_alpha": 2,
        }
    )


def make_adapter(
    tensors, backend_cls=DummyBackend, enable_zero_copy_load: bool = False
):
    adapter = LoRAAdapter(
        uid="adapter",
        config=make_lora_config(),
        base_hf_config=make_base_hf_config(),
        load_config=LoadConfig(),
        lora_backend=backend_cls(max_loras_per_batch=2, device=torch.device("cpu")),
        enable_zero_copy_load=enable_zero_copy_load,
    )
    adapter.initialize_weights_from_tensors(tensors)
    return adapter


class TestLoRAMemoryPool(unittest.TestCase):
    def test_initialize_weights_normalizes_split_tensors_by_default(self):
        q_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        q_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        v_a = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
        v_b = torch.arange(8, 12, dtype=torch.float32).reshape(2, 2)
        gate_a = torch.arange(16, 24, dtype=torch.float32).reshape(2, 4)
        gate_b = torch.arange(12, dtype=torch.float32).reshape(6, 2)

        adapter = make_adapter(
            {
                "model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
                "model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
                "model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
                "model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
                "model.layers.0.mlp.gate_proj.lora_A.weight": gate_a,
                "model.layers.0.mlp.gate_proj.lora_B.weight": gate_b,
            }
        )

        layer_weights = adapter.layers[0].weights
        self.assertNotIn("model.layers.0.self_attn.q_proj.lora_A.weight", layer_weights)
        self.assertNotIn("model.layers.0.self_attn.q_proj.lora_B.weight", layer_weights)
        self.assertNotIn("model.layers.0.self_attn.v_proj.lora_A.weight", layer_weights)
        self.assertNotIn("model.layers.0.self_attn.v_proj.lora_B.weight", layer_weights)
        self.assertNotIn("model.layers.0.mlp.gate_proj.lora_A.weight", layer_weights)
        self.assertNotIn("model.layers.0.mlp.gate_proj.lora_B.weight", layer_weights)
        self.assertTrue(
            torch.equal(
                layer_weights["model.layers.0.self_attn.qkv_proj.lora_A.weight"],
                torch.cat((q_a, torch.zeros_like(v_a), v_a), dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                layer_weights["model.layers.0.self_attn.qkv_proj.lora_B.weight"],
                torch.cat((q_b, torch.zeros_like(v_b), v_b), dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                layer_weights["model.layers.0.mlp.gate_up_proj.lora_A.weight"],
                torch.cat((gate_a, torch.zeros_like(gate_a)), dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                layer_weights["model.layers.0.mlp.gate_up_proj.lora_B.weight"],
                torch.cat((gate_b, torch.zeros_like(gate_b)), dim=0),
            )
        )

    def test_initialize_weights_keeps_raw_split_tensors_when_zero_copy_enabled(self):
        q_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        q_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        v_a = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
        v_b = torch.arange(8, 12, dtype=torch.float32).reshape(2, 2)
        gate_a = torch.arange(16, 24, dtype=torch.float32).reshape(2, 4)
        gate_b = torch.arange(12, dtype=torch.float32).reshape(6, 2)

        adapter = make_adapter(
            {
                "model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
                "model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
                "model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
                "model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
                "model.layers.0.mlp.gate_proj.lora_A.weight": gate_a,
                "model.layers.0.mlp.gate_proj.lora_B.weight": gate_b,
            },
            enable_zero_copy_load=True,
        )

        layer_weights = adapter.layers[0].weights
        self.assertIs(
            layer_weights["model.layers.0.self_attn.q_proj.lora_A.weight"], q_a
        )
        self.assertIs(
            layer_weights["model.layers.0.self_attn.q_proj.lora_B.weight"], q_b
        )
        self.assertIs(
            layer_weights["model.layers.0.self_attn.v_proj.lora_A.weight"], v_a
        )
        self.assertIs(
            layer_weights["model.layers.0.self_attn.v_proj.lora_B.weight"], v_b
        )
        self.assertIs(
            layer_weights["model.layers.0.mlp.gate_proj.lora_A.weight"], gate_a
        )
        self.assertIs(
            layer_weights["model.layers.0.mlp.gate_proj.lora_B.weight"], gate_b
        )
        self.assertNotIn(
            "model.layers.0.self_attn.qkv_proj.lora_A.weight", layer_weights
        )
        self.assertNotIn("model.layers.0.mlp.gate_up_proj.lora_A.weight", layer_weights)

    def test_memory_pool_assembles_split_and_fused_buffers(self):
        memory_pool = LoRAMemoryPool(
            base_hf_config=make_base_hf_config(),
            max_loras_per_batch=2,
            dtype=torch.float32,
            tp_size=1,
            tp_rank=0,
            max_lora_rank=2,
            target_modules={"qkv_proj", "gate_up_proj"},
            base_model=DummyModel(),
            eviction_policy="lru",
            lora_added_tokens_size=0,
        )

        split_adapter = make_adapter(
            {
                "model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
                ),
                "model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
                    [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]
                ),
                "model.layers.0.self_attn.v_proj.lora_A.weight": torch.tensor(
                    [[21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0]]
                ),
                "model.layers.0.self_attn.v_proj.lora_B.weight": torch.tensor(
                    [[31.0, 32.0], [33.0, 34.0]]
                ),
                "model.layers.0.mlp.gate_proj.lora_A.weight": torch.tensor(
                    [[41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0]]
                ),
                "model.layers.0.mlp.gate_proj.lora_B.weight": torch.tensor(
                    [
                        [51.0, 52.0],
                        [53.0, 54.0],
                        [55.0, 56.0],
                        [57.0, 58.0],
                        [59.0, 60.0],
                        [61.0, 62.0],
                    ]
                ),
            },
            enable_zero_copy_load=True,
        )
        memory_pool.load_lora_weight_to_buffer(
            uid="split",
            buffer_id=0,
            lora_adapter=split_adapter,
            lora_modules=[{}],
            lora_embed_tokens_module={},
            lora_lm_head_module={},
        )

        expected_qkv_a = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0],
            ]
        )
        expected_qkv_b = torch.tensor(
            [
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
                [16.0, 17.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [31.0, 32.0],
                [33.0, 34.0],
            ]
        )
        expected_gate_up_a = torch.tensor(
            [
                [41.0, 42.0, 43.0, 44.0],
                [45.0, 46.0, 47.0, 48.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        expected_gate_up_b = torch.tensor(
            [
                [51.0, 52.0],
                [53.0, 54.0],
                [55.0, 56.0],
                [57.0, 58.0],
                [59.0, 60.0],
                [61.0, 62.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.assertTrue(
            torch.equal(memory_pool.A_buffer["qkv_proj"][0][0, :6, :], expected_qkv_a)
        )
        self.assertTrue(
            torch.equal(memory_pool.B_buffer["qkv_proj"][0][0, :, :2], expected_qkv_b)
        )
        self.assertTrue(
            torch.equal(
                memory_pool.A_buffer["gate_up_proj"][0][0, :4, :], expected_gate_up_a
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.B_buffer["gate_up_proj"][0][0, :, :2],
                expected_gate_up_b,
            )
        )

        fused_adapter = make_adapter(
            {
                "model.layers.0.self_attn.qkv_proj.lora_A.weight": torch.tensor(
                    [[71.0, 72.0, 73.0, 74.0], [75.0, 76.0, 77.0, 78.0]]
                ),
                "model.layers.0.self_attn.qkv_proj.lora_B.weight": torch.tensor(
                    [
                        [81.0, 82.0],
                        [83.0, 84.0],
                        [85.0, 86.0],
                        [87.0, 88.0],
                        [89.0, 90.0],
                        [91.0, 92.0],
                        [93.0, 94.0],
                        [95.0, 96.0],
                    ]
                ),
                "model.layers.0.mlp.gate_up_proj.lora_A.weight": torch.tensor(
                    [[101.0, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0]]
                ),
                "model.layers.0.mlp.gate_up_proj.lora_B.weight": torch.tensor(
                    [
                        [111.0, 112.0],
                        [113.0, 114.0],
                        [115.0, 116.0],
                        [117.0, 118.0],
                        [119.0, 120.0],
                        [121.0, 122.0],
                        [123.0, 124.0],
                        [125.0, 126.0],
                        [127.0, 128.0],
                        [129.0, 130.0],
                        [131.0, 132.0],
                        [133.0, 134.0],
                    ]
                ),
            },
            enable_zero_copy_load=True,
        )
        memory_pool.load_lora_weight_to_buffer(
            uid="fused",
            buffer_id=1,
            lora_adapter=fused_adapter,
            lora_modules=[{}],
            lora_embed_tokens_module={},
            lora_lm_head_module={},
        )

        expected_fused_qkv_a = torch.tensor(
            [
                [71.0, 72.0, 73.0, 74.0],
                [75.0, 76.0, 77.0, 78.0],
                [71.0, 72.0, 73.0, 74.0],
                [75.0, 76.0, 77.0, 78.0],
                [71.0, 72.0, 73.0, 74.0],
                [75.0, 76.0, 77.0, 78.0],
            ]
        )
        expected_fused_gate_up_a = torch.tensor(
            [
                [101.0, 102.0, 103.0, 104.0],
                [105.0, 106.0, 107.0, 108.0],
                [101.0, 102.0, 103.0, 104.0],
                [105.0, 106.0, 107.0, 108.0],
            ]
        )
        self.assertTrue(
            torch.equal(
                memory_pool.A_buffer["qkv_proj"][0][1, :6, :], expected_fused_qkv_a
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.B_buffer["qkv_proj"][0][1, :, :2],
                fused_adapter.layers[0].weights[
                    "model.layers.0.self_attn.qkv_proj.lora_B.weight"
                ],
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.A_buffer["gate_up_proj"][0][1, :4, :],
                expected_fused_gate_up_a,
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.B_buffer["gate_up_proj"][0][1, :, :2],
                fused_adapter.layers[0].weights[
                    "model.layers.0.mlp.gate_up_proj.lora_B.weight"
                ],
            )
        )

    def test_memory_pool_loads_normalized_fused_a_weights(self):
        memory_pool = LoRAMemoryPool(
            base_hf_config=make_base_hf_config(),
            max_loras_per_batch=2,
            dtype=torch.float32,
            tp_size=1,
            tp_rank=0,
            max_lora_rank=2,
            target_modules={"qkv_proj", "gate_up_proj"},
            base_model=DummyModel(),
            eviction_policy="lru",
            lora_added_tokens_size=0,
        )

        adapter = make_adapter(
            {
                "model.layers.0.self_attn.qkv_proj.lora_A.weight": torch.tensor(
                    [[71.0, 72.0, 73.0, 74.0], [75.0, 76.0, 77.0, 78.0]]
                ),
                "model.layers.0.self_attn.qkv_proj.lora_B.weight": torch.tensor(
                    [
                        [81.0, 82.0],
                        [83.0, 84.0],
                        [85.0, 86.0],
                        [87.0, 88.0],
                        [89.0, 90.0],
                        [91.0, 92.0],
                        [93.0, 94.0],
                        [95.0, 96.0],
                    ]
                ),
                "model.layers.0.mlp.gate_up_proj.lora_A.weight": torch.tensor(
                    [[101.0, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0]]
                ),
                "model.layers.0.mlp.gate_up_proj.lora_B.weight": torch.tensor(
                    [
                        [111.0, 112.0],
                        [113.0, 114.0],
                        [115.0, 116.0],
                        [117.0, 118.0],
                        [119.0, 120.0],
                        [121.0, 122.0],
                        [123.0, 124.0],
                        [125.0, 126.0],
                        [127.0, 128.0],
                        [129.0, 130.0],
                        [131.0, 132.0],
                        [133.0, 134.0],
                    ]
                ),
            }
        )
        memory_pool.load_lora_weight_to_buffer(
            uid="normalized_fused",
            buffer_id=0,
            lora_adapter=adapter,
            lora_modules=[{}],
            lora_embed_tokens_module={},
            lora_lm_head_module={},
        )

        expected_qkv_a = torch.tensor(
            [
                [71.0, 72.0, 73.0, 74.0],
                [75.0, 76.0, 77.0, 78.0],
                [71.0, 72.0, 73.0, 74.0],
                [75.0, 76.0, 77.0, 78.0],
                [71.0, 72.0, 73.0, 74.0],
                [75.0, 76.0, 77.0, 78.0],
            ]
        )
        expected_gate_up_a = torch.tensor(
            [
                [101.0, 102.0, 103.0, 104.0],
                [105.0, 106.0, 107.0, 108.0],
                [101.0, 102.0, 103.0, 104.0],
                [105.0, 106.0, 107.0, 108.0],
            ]
        )
        self.assertTrue(
            torch.equal(
                memory_pool.A_buffer["qkv_proj"][0][0, :6, :],
                expected_qkv_a,
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.B_buffer["qkv_proj"][0][0, :, :2],
                adapter.layers[0].weights[
                    "model.layers.0.self_attn.qkv_proj.lora_B.weight"
                ],
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.A_buffer["gate_up_proj"][0][0, :4, :],
                expected_gate_up_a,
            )
        )
        self.assertTrue(
            torch.equal(
                memory_pool.B_buffer["gate_up_proj"][0][0, :, :2],
                adapter.layers[0].weights[
                    "model.layers.0.mlp.gate_up_proj.lora_B.weight"
                ],
            )
        )

    def test_missing_up_proj_requires_supported_backend(self):
        memory_pool = LoRAMemoryPool(
            base_hf_config=make_base_hf_config(),
            max_loras_per_batch=2,
            dtype=torch.float32,
            tp_size=1,
            tp_rank=0,
            max_lora_rank=2,
            target_modules={"gate_up_proj"},
            base_model=DummyModel(),
            eviction_policy="lru",
            lora_added_tokens_size=0,
        )

        adapter = make_adapter(
            {
                "model.layers.0.mlp.gate_proj.lora_A.weight": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
                ),
                "model.layers.0.mlp.gate_proj.lora_B.weight": torch.tensor(
                    [
                        [9.0, 10.0],
                        [11.0, 12.0],
                        [13.0, 14.0],
                        [15.0, 16.0],
                        [17.0, 18.0],
                        [19.0, 20.0],
                    ]
                ),
            },
            backend_cls=UnsupportedBackend,
            enable_zero_copy_load=True,
        )

        with self.assertRaisesRegex(
            AssertionError, "LoRA weight initialization currently only supported"
        ):
            memory_pool.load_lora_weight_to_buffer(
                uid="adapter",
                buffer_id=0,
                lora_adapter=adapter,
                lora_modules=[{}],
                lora_embed_tokens_module={},
                lora_lm_head_module={},
            )


if __name__ == "__main__":
    unittest.main()
