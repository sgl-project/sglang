from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.models.laguna import LagunaForCausalLM
from sglang.srt.models.step3p5 import Step3p5ForCausalLM
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
from sglang.test.test_utils import CustomTestCase


def _stub(param_names, calls, **attrs):
    # Stands in for an initialized model; `_checkpoint_loaded` mirrors __init__.
    param = types.SimpleNamespace(
        weight_loader=lambda p, w, *a, **k: calls.append(k.get("expert_id"))
    )
    return types.SimpleNamespace(
        named_parameters=lambda: [(n, param) for n in param_names],
        _checkpoint_loaded=False,
        **attrs,
    )


def _laguna(calls):
    return _stub(
        [
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w2_weight",
        ],
        calls,
        config=types.SimpleNamespace(
            num_experts=2, mlp_layer_types=["sparse"], tie_word_embeddings=False
        ),
        start_layer=0,
        end_layer=1,
    )


def _laguna_experts(expert_ids):
    return [
        (f"model.layers.0.mlp.experts.{e}.{proj}.weight", torch.zeros(1))
        for e in expert_ids
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]


def _step3p5(calls):
    return _stub(
        ["model.norm.weight", "lm_head.weight"],
        calls,
        config=types.SimpleNamespace(moe_num_experts=2, num_hidden_layers=1),
        num_fused_shared_experts=0,
    )


_NORM_ONLY = [("model.norm.weight", torch.zeros(1))]


class TestLagunaBucketedWeightUpdate(CustomTestCase):
    def test_checkpoint_load_rejects_missing_experts(self):
        model = _laguna([])
        with self.assertRaisesRegex(RuntimeError, "routed-expert tensors"):
            LagunaForCausalLM.load_weights(model, iter(_laguna_experts([0])))
        self.assertFalse(model._checkpoint_loaded)

    def test_update_accepts_partial_bucket(self):
        calls = []
        model = _laguna(calls)
        LagunaForCausalLM.load_weights(model, iter(_laguna_experts([0, 1])))
        calls.clear()

        LagunaForCausalLM.load_weights(model, iter(_laguna_experts([1])))

        self.assertEqual(calls, [1, 1, 1])

    def test_update_after_dummy_load_accepts_partial_bucket(self):
        calls = []
        model = _laguna(calls)
        LagunaForCausalLM.post_load_weights(model)

        LagunaForCausalLM.load_weights(model, iter(_laguna_experts([1])))

        self.assertEqual(calls, [1, 1, 1])


class TestStep3p5BucketedWeightUpdate(CustomTestCase):
    def test_checkpoint_load_rejects_missing_params(self):
        model = _step3p5([])
        with self.assertRaisesRegex(AssertionError, "lm_head.weight"):
            Step3p5ForCausalLM.load_weights(model, iter(_NORM_ONLY))
        self.assertFalse(model._checkpoint_loaded)

    def test_update_accepts_partial_bucket(self):
        calls = []
        model = _step3p5(calls)
        Step3p5ForCausalLM.load_weights(
            model, iter(_NORM_ONLY + [("lm_head.weight", torch.zeros(1))])
        )
        calls.clear()

        Step3p5ForCausalLM.load_weights(model, iter(_NORM_ONLY))

        self.assertEqual(calls, [None])

    def test_step3p7_update_after_dummy_load_accepts_partial_bucket(self):
        calls = []
        language_model = _step3p5(calls)
        language_model.post_load_weights = types.MethodType(
            Step3p5ForCausalLM.post_load_weights, language_model
        )
        Step3p7ForConditionalGeneration.post_load_weights(
            types.SimpleNamespace(language_model=language_model)
        )

        Step3p5ForCausalLM.load_weights(language_model, iter(_NORM_ONLY))

        self.assertEqual(calls, [None])


if __name__ == "__main__":
    unittest.main()
