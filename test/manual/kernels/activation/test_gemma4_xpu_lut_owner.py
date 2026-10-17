"""Manual checks through both real outer weight loaders.

The conditional-generation loader once bypassed CausalLM-only initialization.
Reload must preserve captured pointers; module-global ownership must not outlive
its model. Small real module trees exercise those loaders without full weights.
"""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import AutoConfig

from sglang.srt.models.gemma4_causal import (
    Gemma4ForCausalLM,
    Gemma4MLP,
    Gemma4TextModel,
)
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.test_utils import CustomTestCase

MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def make_text_owner(device, enabled=True):
    owner = Gemma4TextModel.__new__(Gemma4TextModel)
    torch.nn.Module.__init__(owner)
    owner.start_layer = 0
    owner.end_layer = 0
    for index in range(2):
        mlp = Gemma4MLP.__new__(Gemma4MLP)
        torch.nn.Module.__init__(mlp)
        mlp.gate_up_proj = torch.nn.Linear(
            1, 1, bias=False, device=device, dtype=torch.bfloat16
        )
        mlp._xpu_gelu_mul_lut = object() if enabled else None
        mlp.register_buffer("_xpu_gate_up_gelu_table", None, persistent=False)
        owner.add_module(f"mlp{index}", mlp)
    return owner


def make_outer(model_type):
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True
    )
    outer = model_type.__new__(model_type)
    torch.nn.Module.__init__(outer)
    outer.pp_group = None
    if model_type is Gemma4ForConditionalGeneration:
        outer.config = config
        outer.language_model = make_text_owner("xpu")
        outer.audio_tower = None
        outer.language_model_only = True
        outer.lm_head_is_tied = True
        return outer, outer.language_model
    outer.config = config.text_config
    outer.model = make_text_owner("xpu")
    return outer, outer.model


class TestGemma4LutOwner(CustomTestCase):
    def test_both_outer_loaders_initialize_and_reuse_model_owned_graph_pointer(self):
        for model_type in [Gemma4ForConditionalGeneration, Gemma4ForCausalLM]:
            with self.subTest(model=model_type.__name__):
                outer, owner = make_outer(model_type)
                outer.load_weights([])
                table = owner._xpu_gate_up_lut_owner
                address = table.data_ptr()
                for layer in owner.children():
                    self.assertIs(layer._xpu_gate_up_gelu_table, table)
                outer.load_weights([])
                self.assertEqual(owner._xpu_gate_up_lut_owner.data_ptr(), address)
                self.assertFalse(
                    any(
                        "lut" in name or "gelu_table" in name
                        for name in outer.state_dict()
                    )
                )
                reference = weakref.ref(table)
                del table, layer, owner, outer
                torch.xpu.synchronize()
                gc.collect()
                self.assertIsNone(reference())

    def test_constructor_keeps_lookup_without_native_api_and_skips_forced_backend(self):
        from sglang.srt.models.gemma3_causal import Gemma3MLP

        def initialize_base(module, *args, **kwargs):
            torch.nn.Module.__init__(module)
            from sglang.srt.layers.activation import GeluAndMul

            module.act_fn = GeluAndMul(approximate="tanh")

        for backend in [None, "torch"]:
            with self.subTest(backend=backend):
                with (
                    mock.patch.object(Gemma3MLP, "__init__", initialize_base),
                    mock.patch(
                        "sglang.srt.models.gemma4_causal.get_parallel",
                        return_value=SimpleNamespace(tp_size=1),
                    ),
                    mock.patch(
                        "sglang.srt.models.gemma4_causal.get_fused_op_backend",
                        return_value=backend,
                    ),
                    mock.patch.object(torch.xpu, "is_available", return_value=True),
                    mock.patch.object(torch.xpu, "get_device_name", return_value="B70"),
                    mock.patch.object(torch.xpu, "current_device", return_value=0),
                    mock.patch(
                        "sgl_kernel.bf16_dense_gelu_tanh_mul_lut_xe20",
                        new=None,
                        create=True,
                    ),
                ):
                    mlp = Gemma4MLP(1536, 12288, "gelu_pytorch_tanh")
                    self.assertIsNone(mlp._xpu_gate_up_gelu_table)
                    self.assertEqual(mlp._xpu_gelu_mul_lut is not None, backend is None)

    def test_activation_only_modules_share_owner_without_native_gate_api(self):
        owner = make_text_owner("xpu")
        owner._initialize_xpu_gate_up_lut()
        owner._assert_xpu_gate_up_lut_initialized()
        table = owner._xpu_gate_up_lut_owner
        self.assertTrue(
            all(layer._xpu_gate_up_gelu_table is table for layer in owner.children())
        )
        owner._initialize_xpu_gate_up_lut()
        self.assertIs(owner._xpu_gate_up_lut_owner, table)

    def test_missing_conditional_loader_initialization_is_rejected(self):
        outer, owner = make_outer(Gemma4ForConditionalGeneration)
        with self.assertRaisesRegex(AssertionError, "no GELU lookup table"):
            owner._assert_xpu_gate_up_lut_initialized()
        outer.load_weights([])
        owner._assert_xpu_gate_up_lut_initialized()

    def test_unsupported_and_disabled_models_keep_native_storage(self):
        for device, enabled in [("cpu", True), ("xpu", False)]:
            with self.subTest(device=device, enabled=enabled):
                owner = make_text_owner(device, enabled)
                owner._initialize_xpu_gate_up_lut()
                owner._assert_xpu_gate_up_lut_initialized()
                self.assertFalse(hasattr(owner, "_xpu_gate_up_lut_owner"))
                self.assertTrue(
                    all(
                        layer._xpu_gate_up_gelu_table is None
                        for layer in owner.children()
                    )
                )


if __name__ == "__main__":
    unittest.main()
