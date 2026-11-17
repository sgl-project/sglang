"""Online weight updates call load_weights once per bucket.

update_weights_from_tensor / _from_distributed send the weights in buckets and
call ``load_weights`` once per bucket. Laguna and Step-3.5 checked at the end of
every call that the call loaded every routed expert (Laguna) or every parameter
(Step-3.5), so the first bucket that did not carry all of them failed the update.
The check belongs to the startup load only.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.models.laguna import LagunaForCausalLM
from sglang.srt.models.step3p5 import Step3p5ForCausalLM
from sglang.test.test_utils import CustomTestCase


class _RecordingParam:
    def __init__(self, name, calls):
        self._name = name
        self._calls = calls

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self._calls.append((self._name, kwargs.get("expert_id")))


def _laguna_stub(calls):
    # Stands in for an initialized LagunaForCausalLM: one sparse layer, two
    # experts. `_startup_load_done` mirrors the value __init__ sets.
    params = {
        name: _RecordingParam(name, calls)
        for name in (
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w2_weight",
        )
    }
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            num_experts=2, mlp_layer_types=["sparse"], tie_word_embeddings=False
        ),
        start_layer=0,
        end_layer=1,
        named_parameters=lambda: list(params.items()),
        _startup_load_done=False,
    )


def _laguna_expert_weights(expert_ids):
    return [
        (f"model.layers.0.mlp.experts.{e}.{proj}.weight", torch.zeros(1))
        for e in expert_ids
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]


def _step3p5_stub(calls):
    # Stands in for an initialized Step3p5ForCausalLM with two dense parameters.
    params = {
        name: _RecordingParam(name, calls)
        for name in ("model.norm.weight", "lm_head.weight")
    }
    return types.SimpleNamespace(
        config=types.SimpleNamespace(moe_num_experts=2, num_hidden_layers=1),
        num_fused_shared_experts=0,
        named_parameters=lambda: list(params.items()),
        _startup_load_done=False,
    )


class TestLagunaBucketedWeightUpdate(CustomTestCase):
    def test_startup_load_rejects_missing_experts(self):
        model = _laguna_stub([])
        with self.assertRaisesRegex(RuntimeError, "routed-expert tensors"):
            LagunaForCausalLM.load_weights(model, iter(_laguna_expert_weights([0])))

    def test_update_accepts_bucket_with_some_experts(self):
        calls = []
        model = _laguna_stub(calls)
        LagunaForCausalLM.load_weights(model, iter(_laguna_expert_weights([0, 1])))
        calls.clear()

        LagunaForCausalLM.load_weights(model, iter(_laguna_expert_weights([1])))

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(expert_id == 1 for _, expert_id in calls))


class TestStep3p5BucketedWeightUpdate(CustomTestCase):
    def test_startup_load_rejects_missing_params(self):
        model = _step3p5_stub([])
        with self.assertRaisesRegex(AssertionError, "lm_head.weight"):
            Step3p5ForCausalLM.load_weights(
                model, iter([("model.norm.weight", torch.zeros(1))])
            )

    def test_update_accepts_bucket_with_some_params(self):
        calls = []
        model = _step3p5_stub(calls)
        Step3p5ForCausalLM.load_weights(
            model,
            iter(
                [
                    ("model.norm.weight", torch.zeros(1)),
                    ("lm_head.weight", torch.zeros(1)),
                ]
            ),
        )
        calls.clear()

        Step3p5ForCausalLM.load_weights(
            model, iter([("model.norm.weight", torch.zeros(1))])
        )

        self.assertEqual(calls, [("model.norm.weight", None)])


if __name__ == "__main__":
    unittest.main()
