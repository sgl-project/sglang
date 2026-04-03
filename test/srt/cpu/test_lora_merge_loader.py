import unittest

try:
    import torch
    from torch import nn

    from sglang.srt.weight_sync.lora_merge_loader import apply_lora_merge_from_tensors
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Missing test dependency: {exc}")


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2, bias=False)
        self.conv_weight = nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
        )
        self.row_parallel_weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.row_parallel_weight.weight_loader = self._row_parallel_weight_loader
        self.row_parallel_weight.local_shard_start = 2
        self.packed_weight = nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
        self.packed_weight.output_dim = 0
        self.packed_weight.weight_loader = self._packed_weight_loader
        self.expert_w13_weight = nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.float32))
        self.expert_w13_weight.weight_loader = self._expert_weight_loader
        self.expert_w2_weight = nn.Parameter(torch.zeros(2, 3, 2, dtype=torch.float32))
        self.expert_w2_weight.weight_loader = self._expert_weight_loader

    def load_weights(self, weights):
        raise AssertionError("apply_lora_merge_from_tensors should not call model.load_weights.")

    @staticmethod
    def _packed_weight_loader(param, loaded_weight, loaded_shard_id):
        shard_slices = {
            "q": slice(0, 2),
            "v": slice(2, 4),
        }
        param.data[shard_slices[loaded_shard_id]].copy_(loaded_weight)

    @staticmethod
    def _row_parallel_weight_loader(param, loaded_weight):
        shard_start = getattr(param, "local_shard_start", 0)
        shard_width = param.data.shape[1]
        param.data.copy_(loaded_weight[:, shard_start : shard_start + shard_width])

    @staticmethod
    def _expert_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id):
        if weight_name == "expert_w13_weight":
            shard_slices = {
                "w1": slice(0, 2),
                "w3": slice(2, 4),
            }
            param.data[expert_id, shard_slices[shard_id]].copy_(loaded_weight)
            return

        if weight_name == "expert_w2_weight":
            assert shard_id == "w2"
            param.data[expert_id].copy_(loaded_weight)
            return

        raise AssertionError(f"Unexpected expert weight target: {weight_name}")


class TestLoRAMergeLoader(unittest.TestCase):
    def test_merge_with_explicit_tensor_names(self):
        model = DummyModel()
        base_weight = model.layer.weight.detach().clone()
        lora_a = torch.tensor([[1.0, 2.0, 3.0]])
        lora_b = torch.tensor([[4.0], [5.0]])

        apply_lora_merge_from_tensors(
            model,
            [("a", lora_a), ("b", lora_b)],
            loader_metadata={
                "targets": [
                    {
                        "target_name": "layer.weight",
                        "lora_a_name": "a",
                        "lora_b_name": "b",
                        "scaling": 0.5,
                    }
                ]
            },
        )

        expected = base_weight + 0.5 * torch.matmul(lora_b, lora_a)
        self.assertTrue(torch.allclose(model.layer.weight, expected))

    def test_merge_with_inferred_tensor_names(self):
        model = DummyModel()
        base_weight = model.layer.weight.detach().clone()
        lora_a = torch.tensor([[1.0, 0.0, -1.0]])
        lora_b = torch.tensor([[2.0], [3.0]])

        apply_lora_merge_from_tensors(
            model,
            [
                ("layer.weight.lora_A", lora_a),
                ("layer.weight.lora_B", lora_b),
            ],
            loader_metadata={"lora_alpha": 8, "rank": 4},
        )

        expected = base_weight + 2.0 * torch.matmul(lora_b, lora_a)
        self.assertTrue(torch.allclose(model.layer.weight, expected))

    def test_merge_supports_singleton_middle_dim_weights(self):
        model = DummyModel()
        base_weight = model.conv_weight.detach().clone()
        lora_a = torch.tensor([[1.0, 2.0, 3.0]])
        lora_b = torch.tensor([[1.0], [2.0]])

        apply_lora_merge_from_tensors(
            model,
            [
                ("conv_weight.lora_A", lora_a),
                ("conv_weight.lora_B", lora_b),
            ],
        )

        expected = base_weight + torch.matmul(lora_b, lora_a).unsqueeze(1)
        self.assertTrue(torch.allclose(model.conv_weight, expected))

    def test_merge_localizes_non_shard_tagged_loader_targets(self):
        model = DummyModel()
        lora_a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        lora_b = torch.tensor([[5.0], [6.0]])

        apply_lora_merge_from_tensors(
            model,
            [
                ("row_parallel_weight.lora_A", lora_a),
                ("row_parallel_weight.lora_B", lora_b),
            ],
        )

        full_delta = torch.matmul(lora_b, lora_a)
        expected_local = full_delta[:, 2:4]
        self.assertTrue(torch.allclose(model.row_parallel_weight, expected_local))

    def test_merge_supports_multiple_components_for_packed_targets(self):
        model = DummyModel()
        q_lora_a = torch.tensor([[1.0, 0.0, 0.0]])
        q_lora_b = torch.tensor([[2.0], [3.0]])
        v_lora_a = torch.tensor([[0.0, 1.0, 0.0]])
        v_lora_b = torch.tensor([[4.0], [5.0]])

        apply_lora_merge_from_tensors(
            model,
            [
                ("q_a", q_lora_a),
                ("q_b", q_lora_b),
                ("v_a", v_lora_a),
                ("v_b", v_lora_b),
            ],
            loader_metadata={
                "targets": [
                    {
                        "target_name": "packed_weight",
                        "components": [
                            {
                                "lora_a_name": "q_a",
                                "lora_b_name": "q_b",
                                "shard_id": "q",
                            },
                            {
                                "lora_a_name": "v_a",
                                "lora_b_name": "v_b",
                                "shard_id": "v",
                            },
                        ],
                    }
                ]
            },
        )

        expected = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 5.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(model.packed_weight, expected))

    def test_merge_supports_fused_expert_w13_targets(self):
        model = DummyModel()
        w1_lora_a = torch.tensor([[[1.0, 0.0, 0.0]]])
        w1_lora_b = torch.tensor(
            [
                [[2.0], [3.0]],
                [[4.0], [5.0]],
            ]
        )
        w3_lora_a = torch.tensor([[[0.0, 1.0, 0.0]]])
        w3_lora_b = torch.tensor(
            [
                [[6.0], [7.0]],
                [[8.0], [9.0]],
            ]
        )

        apply_lora_merge_from_tensors(
            model,
            [
                ("w1_a", w1_lora_a),
                ("w1_b", w1_lora_b),
                ("w3_a", w3_lora_a),
                ("w3_b", w3_lora_b),
            ],
            loader_metadata={
                "targets": [
                    {
                        "target_name": "expert_w13_weight",
                        "components": [
                            {
                                "lora_a_name": "w1_a",
                                "lora_b_name": "w1_b",
                                "shard_id": "w1",
                                "fused_experts": True,
                            },
                            {
                                "lora_a_name": "w3_a",
                                "lora_b_name": "w3_b",
                                "shard_id": "w3",
                                "fused_experts": True,
                            },
                        ],
                    }
                ]
            },
        )

        expected = torch.tensor(
            [
                [
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 7.0, 0.0],
                ],
                [
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 8.0, 0.0],
                    [0.0, 9.0, 0.0],
                ],
            ]
        )
        self.assertTrue(torch.allclose(model.expert_w13_weight, expected))

    def test_merge_supports_fused_expert_w2_targets(self):
        model = DummyModel()
        lora_a = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        lora_b = torch.tensor([[[2.0], [3.0], [4.0]]])

        apply_lora_merge_from_tensors(
            model,
            [
                ("w2_a", lora_a),
                ("w2_b", lora_b),
            ],
            loader_metadata={
                "targets": [
                    {
                        "target_name": "expert_w2_weight",
                        "lora_a_name": "w2_a",
                        "lora_b_name": "w2_b",
                        "shard_id": "w2",
                        "fused_experts": True,
                    }
                ]
            },
        )

        expected = torch.tensor(
            [
                [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
                [[0.0, 2.0], [0.0, 3.0], [0.0, 4.0]],
            ]
        )
        self.assertTrue(torch.allclose(model.expert_w2_weight, expected))


if __name__ == "__main__":
    unittest.main()
