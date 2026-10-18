"""Exercise GPT-OSS's normal NPU checkpoint loader without accelerator hardware."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import gpt_oss
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "model.layers.0.mlp.experts"
EXPERTS = 128
# Checkpoint matrices are transposed by the normal loader; biases are not.
WEIGHTS = (
    ("gate_up_proj", "w13_weight", "w13", (EXPERTS, 3, 10)),
    ("down_proj", "w2_weight", "w2", (EXPERTS, 5, 3)),
    ("gate_up_proj_bias", "w13_weight_bias", "w13", (EXPERTS, 10)),
    ("down_proj_bias", "w2_weight_bias", "w2", (EXPERTS, 3)),
)


def make_model(ep_size, ep_rank, tp_rank=0):
    model = gpt_oss.GptOssForCausalLM.__new__(gpt_oss.GptOssForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=1, num_local_experts=EXPERTS)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    experts = torch.nn.Module()
    model.model.layers[0].mlp.experts = experts
    experts.moe_ep_size = ep_size
    experts.moe_ep_rank = ep_rank
    experts.moe_tp_rank = tp_rank
    experts.num_global_routed_experts = EXPERTS
    experts.num_local_experts = EXPERTS // ep_size
    experts._num_local_routed = experts.num_local_experts
    experts._expert_storage_rank = ep_rank
    received = {}

    def record(param, weight, name, *, shard_id):
        received[name.rsplit(".", 1)[1]] = (shard_id, weight.clone())

    for _, name, _, shape in WEIGHTS:
        shape = (experts.num_local_experts, *shape[1:])
        if "bias" not in name:
            shape = (shape[0], shape[2], shape[1])
        param = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
        param.weight_loader = record
        experts.register_parameter(name, param)
    return model, received


class TestGptOssNpuExpertLoading(CustomTestCase):
    def check_loading(
        self,
        *,
        npu,
        ep_size,
        ep_rank,
        local=False,
        tp_rank=0,
        storage_rank=None,
        dtype=torch.bfloat16,
    ):
        model, received = make_model(ep_size, ep_rank, tp_rank)
        checkpoint, expected = [], {}
        count = EXPERTS // ep_size
        if storage_rank is None:
            storage_rank = ep_rank
        model.model.layers[0].mlp.experts._expert_storage_rank = storage_rank
        start = storage_rank * count
        for checkpoint_name, param_name, shard_id, shape in WEIGHTS:
            tensor = torch.arange(math.prod(shape)).reshape(shape)
            tensor = tensor.to(dtype)
            if local:
                tensor = tensor[start : start + count].clone()
            checkpoint.append((f"{PREFIX}.{checkpoint_name}", tensor))
            wanted = tensor.clone()
            if npu and not local:
                wanted = wanted[start : start + count]
            if "bias" not in param_name:
                wanted = wanted.transpose(-2, -1)
            elif param_name == "w2_weight_bias" and tp_rank != 0:
                wanted.zero_()
            expected[param_name] = (shard_id, wanted)

        # Model loading must use the destination module's partition, not a
        # temporarily overridden global parallel context.
        with (
            patch.object(gpt_oss, "_is_npu", npu),
            patch.object(
                gpt_oss,
                "get_parallel",
                side_effect=AssertionError("use the destination expert layout"),
            ),
        ):
            model._load_normal_weights(
                checkpoint, is_nextn=False, weight_name_mapping=None
            )

        self.assertEqual(received.keys(), expected.keys())
        for name, (shard_id, wanted) in expected.items():
            actual_shard_id, actual = received[name]
            self.assertEqual(actual_shard_id, shard_id)
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            if npu:
                param = model.get_parameter(f"{PREFIX}.{name}")
                self.assertEqual(actual.shape, param.shape)
                param.data.copy_(actual)

    def test_npu_full_and_presharded_checkpoints(self):
        for ep_size in (1, 2, 4, 8):
            for ep_rank in range(ep_size):
                for local in (False, True):
                    for dtype in (torch.bfloat16, torch.float16):
                        with self.subTest(
                            ep_size=ep_size, rank=ep_rank, local=local, dtype=dtype
                        ):
                            self.check_loading(
                                npu=True,
                                ep_size=ep_size,
                                ep_rank=ep_rank,
                                local=local,
                                dtype=dtype,
                            )

    def test_non_npu_loading_is_unchanged(self):
        for ep_rank in range(4):
            with self.subTest(rank=ep_rank):
                self.check_loading(npu=False, ep_size=4, ep_rank=ep_rank)

    def test_npu_uses_destination_storage_rank(self):
        self.check_loading(npu=True, ep_size=4, ep_rank=0, storage_rank=2)

    def test_npu_rejects_incompatible_expert_counts(self):
        for checkpoint_name, _, _, shape in WEIGHTS:
            with self.subTest(weight=checkpoint_name):
                model, _ = make_model(ep_size=4, ep_rank=1)
                # Neither the global count (128) nor this rank's local count (32).
                tensor = torch.zeros((3, *shape[1:]))
                with (
                    patch.object(gpt_oss, "_is_npu", True),
                    self.assertRaisesRegex(ValueError, "Unexpected expert count"),
                ):
                    model._load_normal_weights(
                        [(f"{PREFIX}.{checkpoint_name}", tensor)],
                        is_nextn=False,
                        weight_name_mapping=None,
                    )

    def test_down_projection_bias_uses_destination_tp_rank(self):
        for npu in (False, True):
            for tp_rank in (0, 1):
                with self.subTest(npu=npu, tp_rank=tp_rank):
                    self.check_loading(npu=npu, ep_size=4, ep_rank=2, tp_rank=tp_rank)


if __name__ == "__main__":
    unittest.main()
