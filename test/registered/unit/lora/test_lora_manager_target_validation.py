"""Target admission: replicated-Q uses the fixed attachment set, not adapter
shorthand; DSA indexer targets need an unfused indexer."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora import lora_manager as manager_module
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestManagerTargetValidation(CustomTestCase):
    @staticmethod
    def _config(targets):
        return LoRAConfig.from_dict(
            {"target_modules": targets, "r": 8, "lora_alpha": 8}
        )

    def _manager(self, adapter_targets=None, extra_modules=()):
        modules = [("model.layers.0.experts", FusedMoE.__new__(FusedMoE))]
        modules += [
            (f"model.layers.0.{target}", ReplicatedLinear.__new__(ReplicatedLinear))
            for target in ("q_b_proj", "kv_b_proj", "q_proj")
        ]
        modules += list(extra_modules)
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = SimpleNamespace(named_modules=lambda: modules)
        manager.configs = (
            {"adapter": self._config(adapter_targets)}
            if adapter_targets is not None
            else {}
        )
        manager.lora_refs = {
            "adapter": SimpleNamespace(lora_name="adapter", lora_path="/unused")
        }
        manager.init_lora_adapters = mock.Mock()
        manager.lora_backend = mock.Mock()
        manager._experts_shared_outer_override = False
        manager.init_lora_modules = mock.Mock()
        manager.init_memory_pool = mock.Mock()
        manager.update_lora_info = mock.Mock()
        manager.attn_dp_enabled = False
        manager.num_pinned_loras = 0
        manager.max_loras_per_batch = 4
        manager.lora_added_tokens_size = None
        return manager

    @staticmethod
    def _pool(targets):
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.target_modules = set(targets)
        pool.max_lora_rank = 8
        pool.lora_added_tokens_size = 0
        return pool

    @staticmethod
    def _parallel(replicate_q):
        return mock.patch.object(
            manager_module,
            "get_parallel",
            return_value=SimpleNamespace(dcp_replicate_q_proj=replicate_q),
        )

    def test_indexer_targets_require_an_unfused_dsa_indexer(self):
        indexer = "model.layers.0.self_attn.indexer"
        fused = self._manager(
            extra_modules=[(indexer, SimpleNamespace(use_dsa_indexer_fusion=True))]
        )
        with self.assertRaisesRegex(ValueError, "DSA indexer Q/K fusion"):
            fused.init_lora_shapes(max_lora_rank=8, target_modules={"wk"})
        unfused = self._manager(
            extra_modules=[(indexer, SimpleNamespace(use_dsa_indexer_fusion=False))]
        )
        unfused.init_lora_shapes(max_lora_rank=8, target_modules={"wk"})
        self.assertEqual(unfused.target_modules, {"indexer.wk"})

    def test_safe_final_moe_binding_preserves_shorthand_and_cli_subset(self):
        for adapter in (None, "all", "all-linear"):
            manager = self._manager(adapter)
            with self.subTest(adapter=adapter), self._parallel(True):
                manager.init_state(
                    max_lora_rank=8, target_modules={"gate_proj", "down_proj"}
                )
            self.assertEqual(manager.target_modules, {"gate_up_proj", "down_proj"})
            manager.init_lora_modules.assert_called_once()
            manager.init_memory_pool.assert_called_once()

    def test_unsafe_final_explicit_inferred_and_model_expanded_targets_reject(self):
        targets = ("q_b_proj", "kv_b_proj", "q_proj", "qkv_proj")
        cases = [({f"model.layers.0.{target}"}, None) for target in targets]
        cases += [(None, [f"model.layers.0.{target}"]) for target in targets]
        cases += [({"all"}, None), (None, "all"), (None, "all-linear")]
        for cli, adapter in cases:
            manager = self._manager(adapter)
            with self.subTest(cli=cli, adapter=adapter), self._parallel(True):
                with self.assertRaisesRegex(ValueError, "--no-dcp-replicate-q-proj"):
                    manager.init_state(
                        max_lora_rank=8, target_modules=cli, lora_paths=[object()]
                    )
            # Backend target checks run first; the rejection still precedes wrapping.
            manager.lora_backend.validate_lora_targets.assert_called_once()
            manager.init_lora_modules.assert_not_called()
            manager.init_memory_pool.assert_not_called()
        manager = self._manager()
        with self._parallel(False):
            manager.init_state(
                max_lora_rank=8, target_modules={"q_b_proj", "kv_b_proj", "qkv_proj"}
            )
        manager.init_lora_modules.assert_called_once()

    def test_dynamic_shorthand_keeps_the_fixed_pool_and_rank_contract(self):
        manager = self._manager()
        manager.memory_pool = self._pool({"gate_up_proj", "down_proj"})
        ref = SimpleNamespace(lora_name="new", lora_path="/new", pinned=False)
        for targets in ("all", "all-linear", ["gate_proj", "down_proj"]):
            config = self._config(targets)
            with self.subTest(targets=targets), self._parallel(True):
                manager.validate_new_adapter(config, ref)
            self.assertEqual(config.target_modules, targets)
            self.assertEqual(
                manager.memory_pool.target_modules, {"gate_up_proj", "down_proj"}
            )
        for config in (self._config(["q_b_proj"]), self._config("all")):
            if config.target_modules == "all":
                config.r = 9
            with (
                self._parallel(True),
                self.assertRaisesRegex(ValueError, "incompatible"),
            ):
                manager.validate_new_adapter(config, ref)

    def test_extra_unbound_mla_weights_fail_before_any_buffer_copy(self):
        pool = self._pool({"gate_up_proj", "down_proj"})
        pool.num_layer = 1
        pool.strict_loading = False
        pool.A_buffer = {"down_proj": [torch.tensor([17.0])]}
        pool.B_buffer = {"down_proj": [torch.tensor([19.0])]}
        adapter = SimpleNamespace(
            config=self._config("all-linear"),
            embedding_layers={},
            layers=[
                SimpleNamespace(
                    weights={"model.layers.0.q_b_proj.lora_A.weight": torch.ones(1)},
                    pinned_weights={},
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "Cannot find target module name"):
            pool.load_lora_weight_to_buffer("new", 0, adapter, [{}], None, None)
        self.assertEqual(pool.A_buffer["down_proj"][0].item(), 17.0)
        self.assertEqual(pool.B_buffer["down_proj"][0].item(), 19.0)

    def test_file_and_tensor_loads_do_not_expand_binding(self):
        for tensors in (False, True):
            manager = self._manager()
            manager.memory_pool = self._pool({"down_proj"})
            manager.loras = {}
            manager.base_hf_config = SimpleNamespace(vocab_size=100)
            manager.load_lora_weights = mock.Mock()
            manager.load_lora_weights_from_tensors = mock.Mock()
            manager.create_lora_update_result = lambda **kwargs: SimpleNamespace(
                **kwargs
            )
            ref = SimpleNamespace(
                lora_id="new", lora_name="new", lora_path="/new", pinned=False
            )
            config = self._config("all-linear")
            before = set(manager.memory_pool.target_modules)
            with self.subTest(tensors=tensors), self._parallel(True):
                if tensors:
                    result = manager._load_lora_adapter_from_tensors(
                        ref, {}, config.hf_config
                    )
                else:
                    with mock.patch.object(
                        manager_module, "LoRAConfig", return_value=config
                    ):
                        result = manager._load_lora_adapter(ref)
            self.assertTrue(result.success)
            self.assertEqual(manager.memory_pool.target_modules, before)
            self.assertIn("new", manager.configs)
            loader = (
                manager.load_lora_weights_from_tensors
                if tensors
                else manager.load_lora_weights
            )
            loader.assert_called_once()
            manager.init_lora_modules.assert_not_called()
            manager.init_memory_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
