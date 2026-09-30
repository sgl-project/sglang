"""CPU-only tests for out-of-tree MoE expert executor selection."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.ep_moe import layer as ep_moe_layer
from sglang.srt.layers.moe.expert_executor import MoeExpertExecutorContext
from sglang.srt.models import gemma4_causal
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _context(model_family: str = "gemma4") -> MoeExpertExecutorContext:
    return MoeExpertExecutorContext(
        model_family=model_family,
        layer_id=3,
        num_experts=4,
        num_redundant_experts=0,
        hidden_size=8,
        intermediate_size=16,
        top_k=2,
        activation="gelu",
        reduce_results=True,
        quant_config=None,
        prefix="model.layers.3.moe.experts",
        tp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
    )


class _FakeExecutor(torch.nn.Module):
    last_instance = None

    def __init__(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
        layer_id,
        top_k,
        quant_config,
        prefix,
        activation,
        reduce_results,
    ):
        super().__init__()
        self.kwargs = {
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "layer_id": layer_id,
            "top_k": top_k,
            "quant_config": quant_config,
            "prefix": prefix,
            "activation": activation,
            "reduce_results": reduce_results,
        }
        self.topk_output = None
        _FakeExecutor.last_instance = self

    def forward(self, hidden_states, topk_output):
        self.topk_output = topk_output
        return hidden_states + 1


class _FakeTopK(torch.nn.Module):
    last_instance = None

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.output = object()
        self.inputs = None
        _FakeTopK.last_instance = self

    def forward(self, hidden_states, router_logits):
        self.inputs = (hidden_states, router_logits)
        return self.output


class _OutOfTreePlatformWithoutExecutor(SRTPlatform):
    _enum = PlatformEnum.OOT


class _StandardA2ABackend:
    def is_mori(self):
        return False

    def is_deepep(self):
        return False

    def is_deepep_v2(self):
        return False

    def is_mooncake(self):
        return False

    def is_nixl(self):
        return False

    def is_pplx(self):
        return False


class _DeepEPA2ABackend(_StandardA2ABackend):
    def is_deepep(self):
        return True


class TestMoeExpertExecutorResolution(CustomTestCase):
    def test_external_executor_selection_is_model_scoped(self):
        platform = mock.Mock(spec=SRTPlatform)
        platform.is_out_of_tree.return_value = True
        platform.get_moe_expert_executor_cls.side_effect = lambda context: (
            _FakeExecutor if context.model_family == "gemma4" else None
        )
        with (
            mock.patch.object(ep_moe_layer, "current_platform", platform),
            mock.patch.object(
                ep_moe_layer,
                "get_moe_a2a_backend",
                return_value=_StandardA2ABackend(),
            ),
        ):
            gemma_executor = ep_moe_layer.get_moe_impl_class(
                None, context=_context("gemma4")
            )
            qwen_executor = ep_moe_layer.get_moe_impl_class(
                None, context=_context("qwen")
            )

        self.assertIs(gemma_executor, _FakeExecutor)
        self.assertIs(qwen_executor, ep_moe_layer.FusedMoE)
        self.assertEqual(
            platform.get_moe_expert_executor_cls.call_args_list,
            [mock.call(_context("gemma4")), mock.call(_context("qwen"))],
        )

    def test_out_of_tree_platform_can_decline_and_use_default(self):
        platform = _OutOfTreePlatformWithoutExecutor()

        with (
            mock.patch.object(ep_moe_layer, "current_platform", platform),
            mock.patch.object(
                ep_moe_layer,
                "get_moe_a2a_backend",
                return_value=_StandardA2ABackend(),
            ),
        ):
            actual = ep_moe_layer.get_moe_impl_class(None, context=_context())

        self.assertIs(actual, ep_moe_layer.FusedMoE)

    def test_out_of_tree_platform_decline_preserves_builtin_backend_selection(self):
        platform = _OutOfTreePlatformWithoutExecutor()

        with (
            mock.patch.object(ep_moe_layer, "current_platform", platform),
            mock.patch.object(
                ep_moe_layer,
                "get_moe_a2a_backend",
                return_value=_DeepEPA2ABackend(),
            ),
        ):
            actual = ep_moe_layer.get_moe_impl_class(None, context=_context())

        self.assertIs(actual, ep_moe_layer.DeepEPMoE)

    def test_in_tree_platform_keeps_default_without_consulting_extension(self):
        platform = mock.Mock(spec=SRTPlatform)
        platform.is_out_of_tree.return_value = False
        with (
            mock.patch.object(ep_moe_layer, "current_platform", platform),
            mock.patch.object(
                ep_moe_layer,
                "get_moe_a2a_backend",
                return_value=_StandardA2ABackend(),
            ),
        ):
            actual = ep_moe_layer.get_moe_impl_class(None, context=_context())

        self.assertIs(actual, ep_moe_layer.FusedMoE)
        platform.get_moe_expert_executor_cls.assert_not_called()


class TestGemma4MoeExpertExecutor(CustomTestCase):
    def test_gemma4_passes_context_and_routes_before_selected_executor(self):
        config = SimpleNamespace(
            num_experts=4,
            top_k_experts=2,
            hidden_size=8,
            moe_intermediate_size=16,
        )
        parallel = SimpleNamespace(tp_size=2, moe_tp_size=1, moe_ep_size=2)
        exec_config = SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0))
        quant_config = object()

        with (
            mock.patch.object(gemma4_causal, "TopK", _FakeTopK),
            mock.patch.object(gemma4_causal, "get_parallel", return_value=parallel),
            mock.patch.object(gemma4_causal, "get_exec", return_value=exec_config),
            mock.patch.object(
                gemma4_causal,
                "get_moe_impl_class",
                return_value=_FakeExecutor,
            ) as get_moe_impl_class,
        ):
            moe = gemma4_causal.Gemma4MoE(
                hidden_size=8,
                layer_id=3,
                config=config,
                quant_config=quant_config,
                prefix="model.layers.3.moe",
            )

        self.assertEqual(get_moe_impl_class.call_args.args, (quant_config,))
        context = get_moe_impl_class.call_args.kwargs["context"]
        self.assertEqual(context.model_family, "gemma4")
        self.assertEqual(context.layer_id, 3)
        self.assertEqual(context.num_experts, 4)
        self.assertEqual(context.num_redundant_experts, 0)
        self.assertEqual(context.hidden_size, 8)
        self.assertEqual(context.intermediate_size, 16)
        self.assertEqual(context.top_k, 2)
        self.assertEqual(context.activation, "gelu")
        self.assertTrue(context.reduce_results)
        self.assertIs(context.quant_config, quant_config)
        self.assertEqual(context.prefix, "model.layers.3.moe.experts")
        self.assertEqual(context.tp_size, 2)
        self.assertEqual(context.moe_tp_size, 1)
        self.assertEqual(context.moe_ep_size, 2)

        executor = _FakeExecutor.last_instance
        self.assertEqual(executor.kwargs["num_experts"], 4)
        self.assertEqual(executor.kwargs["hidden_size"], 8)
        self.assertEqual(executor.kwargs["intermediate_size"], 16)
        self.assertEqual(executor.kwargs["layer_id"], 3)
        self.assertEqual(executor.kwargs["top_k"], 2)
        self.assertIs(executor.kwargs["quant_config"], quant_config)
        self.assertEqual(executor.kwargs["prefix"], context.prefix)
        self.assertEqual(executor.kwargs["activation"], "gelu")
        self.assertTrue(executor.kwargs["reduce_results"])

        hidden_states = torch.zeros(2, 8)
        router_logits = torch.zeros(2, 4)
        output = moe(hidden_states, router_logits)

        topk = _FakeTopK.last_instance
        get_moe_impl_class.assert_called_once()
        self.assertIs(topk.inputs[0], hidden_states)
        self.assertIs(topk.inputs[1], router_logits)
        self.assertIs(executor.topk_output, topk.output)
        torch.testing.assert_close(output, torch.ones_like(hidden_states))


if __name__ == "__main__":
    unittest.main()
