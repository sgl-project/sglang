"""Startup-time LoRA contracts kept at weekly cadence.

Each case here fails loudly in production (a `ValueError` at startup, a
mis-sized buffer refused by the load-time shape assert) or pins an order (a
policy's eviction sequence). The per-PR suite keeps the guards whose failures
are silent, including the per-model `should_apply_lora` snapshot in
test_lora_manager_init_modules.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.laguna import LagunaConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora.backend.uno_cublas_backend import UnoCublasLoRABackend
from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.utils import get_default_hidden_dim
from sglang.srt.models.laguna import LagunaModel
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, stage="weekly", runner_config="cpu")


class TestEvictionPolicyOrder(CustomTestCase):
    def test_reuse_changes_lru_but_not_fifo(self):
        """A repeated access refreshes LRU recency, not FIFO insertion order."""
        policies = {name: get_eviction_policy(name) for name in ("lru", "fifo")}
        for policy in policies.values():
            for adapter_id in ("lora1", "lora2", "lora3", "lora1"):
                policy.mark_used(adapter_id)
        candidates = {"lora1", "lora2", "lora3"}
        self.assertEqual(policies["lru"].select_victim(candidates), "lora2")
        self.assertEqual(policies["fifo"].select_victim(candidates), "lora1")


class TestStartupValidation(CustomTestCase):
    """Unsupported configurations are refused before any module is wrapped."""

    def test_uno_rejects_unsupported_targets(self):
        backend = UnoCublasLoRABackend.__new__(UnoCublasLoRABackend)
        backend._pending_lora_a = None
        backend._use_cublas_lora_b = False
        cases = {
            "unknown decoder layer": (
                [("model.layers.0.custom_proj", torch.nn.Linear(2, 2))],
                {"custom_proj"},
                "Linear",
            ),
            "fused MoE": (
                [("model.layers.0.mlp", FusedMoE.__new__(FusedMoE))],
                {"gate_up_proj", "down_proj"},
                "FusedMoE",
            ),
        }
        for name, (modules, targets, expected) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected):
                backend.validate_lora_targets(
                    base_model=SimpleNamespace(named_modules=lambda: modules),
                    target_modules=targets,
                )

    def test_dp_attention_rejects_multi_rank_attention_groups(self):
        with (
            get_context().override_server_args(
                tp_size=4, attn_dp_size=2, enable_lora_overlap_loading=False
            ) as args,
            get_parallel().override(
                tp_size=4, attn_dp_size=2, moe_tp_size=4, attn_tp_size=2, attn_cp_size=1
            ),
            self.assertRaisesRegex(
                ValueError, "requires --attn-dp-size equal to --tp-size"
            ),
        ):
            LoRAManager(
                base_model=torch.nn.Linear(2, 2),
                base_hf_config=SimpleNamespace(),
                max_loras_per_batch=2,
                load_config=None,
                dtype=torch.float32,
                server_args=args,
                lora_backend="triton",
            )

    def test_routed_bank_outside_the_gated_prefix_still_raises(self):
        """Intern-S2-Mobius routed banks sit outside the decoder hierarchy, so
        get_layer_id() returns None; they must raise, not be skipped."""
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = SimpleNamespace(
            named_modules=lambda: [
                ("model.meta_mlp.0.experts", FusedMoE.__new__(FusedMoE))
            ],
            should_apply_lora=lambda name: name.startswith(
                "language_model.model.layers."
            ),
        )
        manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
        manager.target_modules = {"gate_up_proj", "down_proj"}
        with patch.object(manager, "set_lora_module", side_effect=lambda name, _: name):
            with self.assertRaisesRegex(ValueError, "meta_mlp routed banks"):
                manager.init_lora_modules()


class TestLagunaPerLayerAttentionDims(CustomTestCase):
    def test_attention_projections_follow_each_layers_head_count(self):
        """Laguna sizes each layer from `num_attention_heads_per_layer`; the
        global `num_attention_heads` fallback would mis-size qkv_proj/o_proj on
        the wider layer and the load-time shape assert would refuse the adapter.
        `head_dim` (128) is deliberately not hidden_size // heads, as in every
        real checkpoint, so the hook must read it rather than derive it."""
        config = LagunaConfig(
            hidden_size=2048,
            head_dim=128,
            num_key_value_heads=8,
            num_hidden_layers=2,
            num_attention_heads_per_layer=[48, 64],
        )
        model = LagunaModel.__new__(LagunaModel)
        model.config = config
        head_dim, hidden, kv_heads = 128, 2048, 8
        for layer_idx, heads in ((0, 48), (1, 64)):
            with self.subTest(layer=layer_idx):
                self.assertEqual(
                    model.get_hidden_dim("qkv_proj", layer_idx),
                    (hidden, head_dim * (heads + 2 * kv_heads)),
                )
                self.assertEqual(
                    model.get_hidden_dim("o_proj", layer_idx),
                    (head_dim * heads, hidden),
                )
        for module_name in ("qkv_proj", "o_proj"):
            self.assertNotEqual(
                model.get_hidden_dim(module_name, 1),
                get_default_hidden_dim(module_name, config, 1),
                f"{module_name}: the per-layer hook must diverge from the global "
                "fallback on the wider layer",
            )
        with self.assertRaises(NotImplementedError):
            model.get_hidden_dim("not_a_module", layer_idx=0)
        del config.head_dim
        with self.assertRaises(AttributeError):
            model.get_hidden_dim("o_proj", layer_idx=1)


if __name__ == "__main__":
    unittest.main()
