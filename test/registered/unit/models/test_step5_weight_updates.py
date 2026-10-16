"""Step5 load_weights: a strict checkpoint load, then weight updates streamed like any other model's."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.step5 import Step5ForCausalLM, _step5_bf16_trtllm_supported
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_NUM_EXPERTS = 4
_EXPERTS = "model.layers.3.moe.experts."
_PACKED = "model.layers.3.moe."


class _RecordingParam:
    """A parameter that records what its weight loader was handed."""

    def __init__(self, dtype: torch.dtype, calls: list):
        self.dtype = dtype
        self._calls = calls

    def weight_loader(self, param, weight, *args, shard_id=None, expert_id=None):
        self._calls.append((self, weight, shard_id, expert_id))


class _FakeStep5:
    """The attributes Step5ForCausalLM.load_weights reads, around one FP8 MoE layer and a norm."""

    def __init__(self):
        self.calls = []
        fp8, fp32 = torch.float8_e4m3fn, torch.float32
        self.params = {
            _EXPERTS + "w13_weight": _RecordingParam(fp8, self.calls),
            _EXPERTS + "w2_weight": _RecordingParam(fp8, self.calls),
            _EXPERTS + "w13_weight_scale_inv": _RecordingParam(fp32, self.calls),
            _EXPERTS + "w2_weight_scale_inv": _RecordingParam(fp32, self.calls),
            "model.norm.weight": _RecordingParam(torch.bfloat16, self.calls),
        }
        self.config = SimpleNamespace(
            moe_num_experts=_NUM_EXPERTS,
            num_hidden_layers=92,
            num_nextn_predict_layers=0,
        )
        self.nvfp4_expert_layers = frozenset()
        self.nvfp4_expert_layouts = {}
        self.model = SimpleNamespace(layers=[])
        self.checkpoint_path = None
        self._checkpoint_loaded = False
        self.duplicate_scans = 0

    def named_parameters(self):
        return list(self.params.items())

    def _indexed_duplicate_weights(self, weights, expected, model_path):
        self.duplicate_scans += 1
        return Step5ForCausalLM._indexed_duplicate_weights(
            weights, expected, model_path
        )

    def load_weights(self, weights):
        return Step5ForCausalLM.load_weights(self, weights)

    def expert_calls(self, param_name: str) -> list[tuple[str, int]]:
        param = self.params[_EXPERTS + param_name]
        return [(shard, expert) for p, _, shard, expert in self.calls if p is param]


def _checkpoint() -> list[tuple[str, torch.Tensor]]:
    """Every target weight, experts packed as the released checkpoint stores them."""
    weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weights.append(
            (
                f"{_PACKED}{projection}.weight",
                torch.zeros(_NUM_EXPERTS, 4, 4, dtype=torch.float8_e4m3fn),
            )
        )
        weights.append(
            (f"{_PACKED}{projection}.weight_scale_inv", torch.ones(_NUM_EXPERTS, 1, 1))
        )
    return weights


def _loaded() -> _FakeStep5:
    model = _FakeStep5()
    model.load_weights(_checkpoint())
    model.calls.clear()
    return model


class TestStep5CheckpointLoad(CustomTestCase):
    def test_loads_every_expert_of_a_packed_tensor(self):
        model = _FakeStep5()
        model.load_weights(_checkpoint())

        self.assertEqual(
            model.expert_calls("w13_weight"),
            [(s, e) for s in ("w1", "w3") for e in range(4)],
        )
        self.assertEqual(model.duplicate_scans, 1)

    def test_names_what_it_is_missing(self):
        model = _FakeStep5()
        partial = [
            (name, weight) for name, weight in _checkpoint() if "down_proj" not in name
        ]

        with self.assertRaisesRegex(ValueError, "missing 2 weights"):
            model.load_weights(partial)

    def test_refuses_per_expert_names(self):
        model = _FakeStep5()
        per_expert = _EXPERTS + "1.gate_proj.weight"

        with self.assertRaisesRegex(ValueError, "Unknown Step5 target weight"):
            model.load_weights(
                [
                    *_checkpoint(),
                    (per_expert, torch.zeros(4, 4, dtype=torch.float8_e4m3fn)),
                ]
            )


class TestStep5WeightUpdates(CustomTestCase):
    def test_accept_one_expert_at_a_time(self):
        model = _loaded()
        weight = torch.full((4, 4), 2.0).to(torch.float8_e4m3fn)

        model.load_weights(
            [
                (_EXPERTS + "2.gate_proj.weight", weight),
                (_EXPERTS + "2.gate_proj.weight_scale_inv", torch.ones(1, 1)),
            ]
        )

        self.assertEqual(model.expert_calls("w13_weight"), [("w1", 2)])
        self.assertEqual(model.expert_calls("w13_weight_scale_inv"), [("w1", 2)])
        loaded = next(
            w
            for p, w, _, _ in model.calls
            if p is model.params[_EXPERTS + "w13_weight"]
        )
        self.assertTrue(torch.equal(loaded.float(), weight.float()))

    def test_may_send_any_subset_and_skip_the_disk_index(self):
        model = _loaded()

        model.load_weights([("model.norm.weight", torch.ones(8, dtype=torch.bfloat16))])

        self.assertEqual(model.duplicate_scans, 1)
        self.assertEqual(len(model.calls), 1)

    def test_still_refuse_unknown_names(self):
        model = _loaded()
        for name in (
            _EXPERTS + f"{_NUM_EXPERTS}.gate_proj.weight",
            _EXPERTS + "1.gate_proj.bias",
            "model.nope",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "Unknown Step5 target weight"),
            ):
                model.load_weights([(name, torch.zeros(4, 4))])

    def test_refuse_the_same_expert_twice_in_one_call(self):
        model = _loaded()
        name = _EXPERTS + "1.down_proj.weight"
        weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)

        with self.assertRaisesRegex(ValueError, "Duplicate Step5 target weight"):
            model.load_weights([(name, weight), (name, weight)])


class TestStep5Bf16ExpertRunner(CustomTestCase):
    def test_blackwell_uses_trtllm_only_with_the_swiglu_step_patch(self):
        """Blackwell without FlashInfer's SwigluStep patch serves checkpoint-BF16
        experts on the Triton runner instead of refusing to start."""
        for patched in (False, True):
            core = types.ModuleType("flashinfer.fused_moe.core")
            if patched:
                core.TRTLLM_BF16_SWIGLU_STEP_SUPPORTED = True
            fused_moe = types.ModuleType("flashinfer.fused_moe")
            fused_moe.core = core
            with (
                self.subTest(patched=patched),
                patch(
                    "sglang.srt.models.step5.get_platform",
                    return_value=SimpleNamespace(is_sm100=True),
                ),
                patch.dict(
                    sys.modules,
                    {
                        "flashinfer.fused_moe": fused_moe,
                        "flashinfer.fused_moe.core": core,
                    },
                ),
            ):
                self.assertEqual(_step5_bf16_trtllm_supported(), patched)


if __name__ == "__main__":
    unittest.main()
