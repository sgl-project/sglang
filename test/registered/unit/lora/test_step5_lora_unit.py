"""Step5 LoRA geometry and the routed-expert decision.

An MoE layer's gate_up_proj/down_proj is its shared expert (share_expert_dim wide),
a dense layer's is its MLP (intermediate_size wide); the generic fallback would size
every layer from intermediate_size. Those leaf names cannot tell the routed experts
apart, so Step5 adapts them only when a target names moe.experts: the shared-outer
expert tensors then stack into the *_moe buffers, and only the Triton MoE runner,
whose hooks add the deltas, may serve them.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import lora_adapts_routed_experts
from sglang.srt.models import step5 as step5_module
from sglang.srt.models.step5 import Step5ForCausalLM, Step5MoE
from sglang.srt.models.step5_vl import Step5ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_FROZEN_EXPERT_TARGETS = [
    *(f"model.layers.*.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj")),
    *(
        f"model.layers.*.{block}.{leaf}"
        for block in ("mlp", "share_expert")
        for leaf in ("gate_proj", "up_proj", "down_proj")
    ),
]
_EXPERT_TARGETS = [
    f"model.layers.*.moe.experts.{leaf}"
    for leaf in ("gate_proj", "up_proj", "down_proj")
]


def _moe(layer_id=1, *, mega=False, nvfp4=False, bf16_trtllm=False):
    moe = Step5MoE.__new__(Step5MoE)
    torch.nn.Module.__init__(moe)
    moe.layer_id = layer_id
    moe.use_mega_moe, moe.use_nvfp4_experts, moe.bf16_trtllm = mega, nvfp4, bf16_trtllm
    return moe


def _step5(**moe_kwargs):
    model = Step5ForCausalLM.__new__(Step5ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        intermediate_size=13824,
        share_expert_dim=1536,
        moe_intermediate_size=1536,
        moe_layer_list=[1, 2, 3],
    )
    model.model = SimpleNamespace(
        layers=[
            SimpleNamespace(use_moe=False),
            SimpleNamespace(use_moe=True, moe=_moe(1, **moe_kwargs)),
        ]
    )
    return model


def _runner(name):
    return mock.patch.object(
        step5_module,
        "get_moe_runner_backend",
        return_value=SimpleNamespace(value=name, is_triton=lambda: name == "triton"),
    )


class TestStep5LoRAHiddenDim(unittest.TestCase):
    def test_attention_widths(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("qkv_proj", 0), (4096, 192 * (64 + 8)))
        self.assertEqual(model.get_hidden_dim("o_proj", 1), (192 * 64, 4096))

    def test_dense_layer_mlp(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 0), (4096, 2 * 13824))
        self.assertEqual(model.get_hidden_dim("down_proj", 4), (13824, 4096))

    def test_moe_layer_shared_expert(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 2), (4096, 2 * 1536))
        self.assertEqual(model.get_hidden_dim("down_proj", 3), (1536, 4096))

    def test_routed_experts(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("gate_up_proj_moe", 1), (4096, 2 * 1536))
        self.assertEqual(model.get_hidden_dim("down_proj_moe", 1), (1536, 4096))

    def test_the_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            _step5().get_hidden_dim("g_proj", 1)

    def test_vl_wrapper_delegates_to_the_language_model(self):
        wrapper = Step5ForConditionalGeneration.__new__(Step5ForConditionalGeneration)
        torch.nn.Module.__init__(wrapper)
        wrapper.language_model = _step5()
        self.assertEqual(wrapper.get_hidden_dim("down_proj", 1), (1536, 4096))
        with _runner("triton"):
            self.assertTrue(lora_adapts_routed_experts(wrapper, _EXPERT_TARGETS))
        self.assertFalse(lora_adapts_routed_experts(wrapper, _FROZEN_EXPERT_TARGETS))


class TestStep5RoutedExpertDecision(unittest.TestCase):
    def test_targets_without_moe_experts_keep_them_frozen(self):
        # Leaf names alone are the dense and shared-expert MLPs.
        for targets in (_FROZEN_EXPERT_TARGETS, ["gate_proj", "up_proj", "down_proj"]):
            self.assertFalse(lora_adapts_routed_experts(_step5(), targets))

    def test_moe_experts_or_all_linear_adapt_them(self):
        with _runner("triton"):
            for targets in (
                _FROZEN_EXPERT_TARGETS + _EXPERT_TARGETS,
                ["model.layers.*.moe.experts.*.down_proj"],
                ["all"],
                ["all-linear"],
            ):
                self.assertTrue(lora_adapts_routed_experts(_step5(), targets))

    def test_expert_targets_without_the_moe_prefix_adapt_them(self):
        """`experts.<proj>` targets the same LoRA modules as `moe.experts.<proj>`;
        it must not leave the routed experts silently frozen."""
        with _runner("triton"):
            self.assertTrue(
                lora_adapts_routed_experts(
                    _step5(),
                    [
                        f"experts.{leaf}"
                        for leaf in ("gate_proj", "up_proj", "down_proj")
                    ],
                )
            )

    def test_paths_bypassing_the_lora_hooks_are_refused(self):
        cases = (
            ("triton", dict(mega=True), "MegaMoE"),
            ("triton", dict(nvfp4=True), "NVFP4"),
            ("triton", dict(bf16_trtllm=True), "TRT-LLM BF16"),
            ("flashinfer_trtllm", {}, "flashinfer_trtllm MoE runner"),
        )
        for runner, moe_kwargs, message in cases:
            with self.subTest(message), _runner(runner):
                with self.assertRaisesRegex(ValueError, message):
                    lora_adapts_routed_experts(_step5(**moe_kwargs), _EXPERT_TARGETS)
                # Frozen experts never reach those paths' missing hooks.
                self.assertFalse(
                    lora_adapts_routed_experts(
                        _step5(**moe_kwargs), _FROZEN_EXPERT_TARGETS
                    )
                )

    def test_other_models_keep_routed_expert_lora(self):
        self.assertTrue(lora_adapts_routed_experts(torch.nn.Module(), ["gate_proj"]))


class TestStep5AdapterWithoutRoutedExpertLoRA(unittest.TestCase):
    def test_routed_expert_weights_are_refused(self):
        """When startup left the routed experts without LoRA, a later adapter that
        carries routed-expert weights is refused instead of loading weights that
        never apply; shared-expert weights still load."""
        manager = LoRAManager.__new__(LoRAManager)
        manager.adapts_routed_experts = False
        manager.base_model = None
        manager.base_hf_config = SimpleNamespace(num_hidden_layers=2)
        manager.load_config = None
        manager.lora_backend = None
        manager.lora_no_cpu_backup = False
        manager.speculative_algorithm = None
        config = SimpleNamespace(
            target_modules=["share_expert.down_proj", "experts.down_proj"],
            hf_config={"peft_type": "lora"},
            lora_alpha=4,
            r=2,
        )
        ref = SimpleNamespace(lora_id="late", lora_name="late")
        prefix = "base_model.model.model.layers.1."
        shared = {
            f"{prefix}share_expert.down_proj.lora_A.weight": torch.zeros(2, 3),
        }
        routed = {
            f"{prefix}moe.experts.down_proj.lora_A.weight": torch.zeros(1, 2, 3),
        }

        with self.assertRaisesRegex(ValueError, "1 routed-expert weights"):
            manager._create_lora_adapter_from_tensors(ref, config, shared | routed)
        manager._create_lora_adapter_from_tensors(ref, config, shared)


class TestStep5ExpertPool(unittest.TestCase):
    def test_step5_config_sizes_the_expert_and_shared_buffers(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(moe_num_experts=352, share_expert_dim=1536)
        self.assertEqual(LoRAMemoryPool._get_num_experts(model), 352)

    def test_shared_outer_names_stack_into_the_expert_buffers(self):
        """Routed experts stack gate/up into gate_up_proj; the shared expert stays per member."""
        rank, hidden, inter, experts = 2, 8, 3, 4
        prefix = "model.layers.1."
        weights = {
            f"{prefix}moe.experts.gate_proj.lora_A.weight": torch.randn(
                1, rank, hidden
            ),
            f"{prefix}moe.experts.up_proj.lora_A.weight": torch.randn(1, rank, hidden),
            f"{prefix}moe.experts.gate_proj.lora_B.weight": torch.randn(
                experts, inter, rank
            ),
            f"{prefix}moe.experts.up_proj.lora_B.weight": torch.randn(
                experts, inter, rank
            ),
            f"{prefix}share_expert.gate_proj.lora_A.weight": torch.randn(rank, hidden),
            f"{prefix}share_expert.up_proj.lora_A.weight": torch.randn(rank, hidden),
        }
        expected_a = torch.cat(
            [
                weights[f"{prefix}moe.experts.{leaf}.lora_A.weight"]
                for leaf in ("gate_proj", "up_proj")
            ],
            dim=1,
        )
        adapter = LoRAAdapter.__new__(LoRAAdapter)
        object.__setattr__(adapter, "_moe_is_gated_by_layer", {1: True})
        adapter.normalize_gate_up_proj(list(weights), weights)

        self.assertEqual(
            sorted(weights),
            sorted(
                [
                    f"{prefix}moe.experts.gate_up_proj.lora_A.weight",
                    f"{prefix}moe.experts.gate_up_proj.lora_B.weight",
                    f"{prefix}share_expert.gate_proj.lora_A.weight",
                    f"{prefix}share_expert.up_proj.lora_A.weight",
                ]
            ),
        )
        torch.testing.assert_close(
            weights[f"{prefix}moe.experts.gate_up_proj.lora_A.weight"], expected_a
        )
        self.assertEqual(
            weights[f"{prefix}moe.experts.gate_up_proj.lora_B.weight"].shape,
            (experts, 2 * inter, rank),
        )


if __name__ == "__main__":
    unittest.main()
