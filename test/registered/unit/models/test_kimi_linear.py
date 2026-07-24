"""
Unit tests for KimiLinearForCausalLM post-load MLA weight fixup.

Regression coverage for `--load-format dummy`: DummyModelLoader bypasses
`model.load_weights()` and only invokes `model.post_load_weights()` (via
`_post_load_weights` in srt/model_loader/loader.py). KimiLinearForCausalLM
used to inline the MLA w_kc / w_vc extraction at the tail of
`load_weights()`, so dummy loading left `w_kc = None` and crashed at the
first MLA forward. These tests pin the extracted method: it exists, it is
idempotent, the normal `load_weights()` path still performs the extraction,
and the loader-side hook reaches it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_loader.loader import _post_load_weights
from sglang.srt.models.kimi_linear import KimiLinearForCausalLM

NUM_HEADS = 2
QK_NOPE_HEAD_DIM = 4
V_HEAD_DIM = 3
KV_LORA_RANK = 5


def _make_mla_attn(with_weight_scale: bool = False) -> SimpleNamespace:
    """Minimal stand-in for the MLA self_attn of a full-attention layer."""
    weight = torch.arange(
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * KV_LORA_RANK,
        dtype=torch.float32,
    ).reshape(NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK)
    kv_b_proj = SimpleNamespace(weight=weight)
    if with_weight_scale:
        kv_b_proj.weight_scale = torch.tensor(0.5)
    return SimpleNamespace(
        kv_b_proj=kv_b_proj,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
    )


def _make_model(with_weight_scale: bool = False) -> KimiLinearForCausalLM:
    """Skeleton KimiLinearForCausalLM without touching distributed init.

    Layer 0 mimics a KDA (linear attention) layer whose self_attn has no
    kv_b_proj; layer 1 is the full-attention (MLA) layer.
    """
    model = object.__new__(KimiLinearForCausalLM)
    model.config = SimpleNamespace(
        full_attention_layer_ids=[1],
        is_moe=False,
    )
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace()),
        SimpleNamespace(self_attn=_make_mla_attn(with_weight_scale)),
    ]
    model.model = SimpleNamespace(layers=layers)
    model.named_parameters = lambda: iter(())
    return model


def _expected_w_kc_w_vc(self_attn):
    ref = self_attn.kv_b_proj.weight.unflatten(0, (-1, QK_NOPE_HEAD_DIM + V_HEAD_DIM))
    return ref[:, :QK_NOPE_HEAD_DIM, :], ref[:, QK_NOPE_HEAD_DIM:, :].transpose(1, 2)


class TestKimiLinearPostLoadWeights(unittest.TestCase):
    def test_post_load_weights_extracts_w_kc_w_vc(self):
        model = _make_model()
        model.post_load_weights()

        self_attn = model.model.layers[1].self_attn
        expected_w_kc, expected_w_vc = _expected_w_kc_w_vc(self_attn)

        self.assertEqual(
            self_attn.w_kc.shape, (NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK)
        )
        self.assertEqual(self_attn.w_vc.shape, (NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM))
        self.assertTrue(torch.equal(self_attn.w_kc, expected_w_kc))
        self.assertTrue(torch.equal(self_attn.w_vc, expected_w_vc))
        # KDA (linear attention) layer must stay untouched.
        self.assertFalse(hasattr(model.model.layers[0].self_attn, "w_kc"))
        # No weight_scale on kv_b_proj -> no w_scale.
        self.assertFalse(hasattr(self_attn, "w_scale"))

    def test_post_load_weights_is_idempotent(self):
        model = _make_model()
        model.post_load_weights()
        self_attn = model.model.layers[1].self_attn
        first_w_kc = self_attn.w_kc.clone()
        first_w_vc = self_attn.w_vc.clone()

        model.post_load_weights()

        self.assertTrue(torch.equal(self_attn.w_kc, first_w_kc))
        self.assertTrue(torch.equal(self_attn.w_vc, first_w_vc))

    def test_load_weights_still_triggers_extraction(self):
        """The normal checkpoint path must keep its old tail behavior."""
        model = _make_model()
        model.load_weights(iter(()))

        self_attn = model.model.layers[1].self_attn
        expected_w_kc, expected_w_vc = _expected_w_kc_w_vc(self_attn)
        self.assertTrue(torch.equal(self_attn.w_kc, expected_w_kc))
        self.assertTrue(torch.equal(self_attn.w_vc, expected_w_vc))

    def test_dummy_loader_hook_reaches_extraction(self):
        """DummyModelLoader path: loader._post_load_weights(model) must find
        and run the fixup (before the refactor, hasattr() was False and the
        extraction silently never happened)."""
        model = _make_model()
        _post_load_weights(model)

        self.assertTrue(hasattr(model.model.layers[1].self_attn, "w_kc"))
        self.assertIsNotNone(model.model.layers[1].self_attn.w_kc)

    def test_weight_scale_propagates(self):
        model = _make_model(with_weight_scale=True)
        model.post_load_weights()

        self_attn = model.model.layers[1].self_attn
        self.assertTrue(
            torch.equal(self_attn.w_scale, self_attn.kv_b_proj.weight_scale)
        )


if __name__ == "__main__":
    unittest.main()
