"""Kimi dense and shared MLPs freeze placement without changing row policy."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers import linear
from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def build_mlp(group=None, *, width=16, quant_config=None, reduce=True):
    kwargs = {} if group is None else dict(parallel_group=group)
    return kimi_k3.KimiK3MLP(
        width,
        2 * width,
        "silu",
        quant_config=quant_config,
        reduce_results=reduce,
        **kwargs,
    )


class _RoutedExperts(torch.nn.Module):
    should_fuse_routed_scaling_factor_in_topk = False

    def __init__(self, **kwargs):
        super().__init__()


class _Gate(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.e_score_correction_bias = None


def build_shared_experts(*, ep_a2a=True, width=16, shared=1, quant_config=None):
    # Routed experts and their router do not participate in this placement
    # test. The real MoE constructor still builds its real shared MLP.
    backend = SimpleNamespace(
        is_megamoe=lambda: False,
        is_deepep=lambda: ep_a2a,
        is_mooncake=lambda: False,
        is_ascend_fuseep=lambda: False,
        is_mori=lambda: False,
    )
    config = SimpleNamespace(
        hidden_size=width,
        moe_intermediate_size=2 * width,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        num_shared_experts=shared,
        routed_expert_hidden_size=None,
        num_experts=4,
        num_experts_per_token=1,
        hidden_act="silu",
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
    )
    with (
        patch.object(kimi_k3, "MoEGate", _Gate),
        patch.object(kimi_k3, "get_moe_impl_class", return_value=_RoutedExperts),
        patch.object(kimi_k3, "get_moe_a2a_backend", return_value=backend),
    ):
        module = kimi_k3.KimiK3MoE(config, quant_config=quant_config)
    return module, module.shared_experts


def load_projection(layer):
    full = (
        torch.arange(layer.output_size * layer.input_size, device=layer.weight.device)
        % 29
        - 14
    ).reshape(layer.output_size, layer.input_size).to(layer.weight.dtype) / 128
    rank, size = layer.tp_rank, layer.tp_size
    with get_parallel().override(
        tp_rank=0, attn_tp_rank=0, attn_dp_rank=0, shared_experts_tp_group=None
    ):
        layer.weight.weight_loader(layer.weight, full)
    if isinstance(layer, linear.RowParallelLinear):
        expected = full.chunk(size, dim=1)[rank]
    else:
        expected = torch.cat(
            [v.chunk(size)[rank] for v in full.split(layer.output_sizes)]
        )
    return expected, None


class TestKimiMLPParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_dense_and_explicit_group_layout_and_row_policy(self):
        for dp in (1, 2):
            for dense_attn in (False, True):
                for rank in range(4):
                    reset_context()
                    server = ServerArgs(
                        model_path="dummy",
                        device="cpu",
                        tp_size=4,
                        attn_dp_size=dp,
                        enable_dense_mlp_attn_tp=dense_attn,
                    )
                    publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
                    initialize_dp_attention_flags(server)
                    shared = SimpleNamespace(rank_in_group=rank % 2, world_size=2)
                    with get_parallel().override(shared_experts_tp_group=shared):
                        for group in (
                            None,
                            "tp",
                            "attn_tp",
                            "replicated",
                            "shared_experts_tp",
                        ):
                            for reduce in (False, True):
                                module = build_mlp(group, reduce=reduce)
                                auto_attn = group is None and dense_attn and dp > 1
                                r, s = (
                                    (0, 1)
                                    if group == "replicated"
                                    else (rank % 2, 2)
                                    if group == "shared_experts_tp"
                                    else (rank % (4 // dp), 4 // dp)
                                    if group == "attn_tp" or auto_attn
                                    else (rank, 4)
                                )
                                self.assertEqual(module._dense_attn_tp, auto_attn)
                                for name in ("gate_up_proj", "down_proj"):
                                    layer = getattr(module, name)
                                    self.assertEqual(
                                        (layer.tp_rank, layer.tp_size), (r, s)
                                    )
                                    expected, _ = load_projection(layer)
                                    torch.testing.assert_close(layer.weight, expected)
                                    x = (
                                        (
                                            torch.arange(2 * expected.shape[1]).reshape(
                                                2, -1
                                            )
                                            % 17
                                            - 8
                                        )
                                        / 128
                                    ).to(layer.weight.dtype)
                                    tp, attn = Mock(), Mock()
                                    tp.all_reduce.side_effect = lambda v: v * 4
                                    attn.all_reduce.side_effect = lambda v: (
                                        v * (4 // dp)
                                    )
                                    row = name == "down_proj"
                                    reference = F.linear(x, expected)
                                    if row and reduce and s > 1:
                                        reference *= (4 // dp) if auto_attn else 4
                                    with (
                                        get_parallel().override(
                                            tp_group=tp, attn_tp_group=attn
                                        ),
                                        patch.object(
                                            linear,
                                            "use_symmetric_memory",
                                            return_value=nullcontext(),
                                        ) as allocation,
                                    ):
                                        actual = layer(x)[0]
                                    torch.testing.assert_close(actual, reference)
                                    if row:
                                        self.assertEqual(layer.reduce_results, reduce)
                                        self.assertEqual(
                                            layer.use_dp_attention_reduce, auto_attn
                                        )
                                        self.assertIs(
                                            allocation.call_args.args[0],
                                            attn if auto_attn else tp,
                                        )
                                    else:
                                        allocation.assert_not_called()

    def test_moe_shared_consumer_preserves_selection_and_execution_handle(self):
        for rank in range(4):
            reset_context()
            server = ServerArgs(
                model_path="dummy", device="cpu", tp_size=4, attn_dp_size=1
            )
            publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
            initialize_dp_attention_flags(server)
            attn = SimpleNamespace(world_size=4, rank_in_group=rank)
            shared = SimpleNamespace(world_size=2, rank_in_group=rank % 2)
            for ep, requested, enabled, expected_size in (
                (False, None, False, 4),
                (True, None, False, 1),
                (True, 1, False, 1),
                (True, 2, False, 2),
                (True, None, True, 4),
            ):
                with get_parallel().override(
                    shared_experts_tp_size=requested,
                    enable_shared_experts_attn_tp=enabled,
                    shared_experts_tp_group=shared,
                    attn_tp_group=attn,
                ):
                    owner, module = build_shared_experts(ep_a2a=ep)
                    comm = ep and expected_size > 1
                    self.assertEqual(owner._shared_experts_tp_comm, comm)
                    self.assertEqual(
                        owner._shared_experts_tp1, ep and expected_size == 1
                    )
                    self.assertIs(
                        owner._shared_experts_tp_group,
                        shared if requested == 2 else attn if comm else None,
                    )
                    self.assertFalse(module._dense_attn_tp)
                    for name in ("gate_up_proj", "down_proj"):
                        layer = getattr(module, name)
                        self.assertEqual(
                            (layer.tp_rank, layer.tp_size),
                            (rank % expected_size, expected_size),
                        )
                        expected, _ = load_projection(layer)
                        torch.testing.assert_close(layer.weight, expected)
                    self.assertFalse(module.down_proj.reduce_results)
                    self.assertFalse(module.down_proj.use_dp_attention_reduce)
            with get_parallel().override(shared_experts_tp_size=2):
                with self.assertRaisesRegex(ValueError, "requires an EP a2a"):
                    build_shared_experts(ep_a2a=False)
            with get_parallel().override(
                shared_experts_tp_size=3,
                shared_experts_tp_group=SimpleNamespace(
                    world_size=3, rank_in_group=rank % 3
                ),
            ):
                with self.assertRaisesRegex(ValueError, "must be divisible"):
                    build_shared_experts()
            _, disabled = build_shared_experts(shared=0)
            self.assertIsNone(disabled)

    def test_shared_selector_requires_a_built_group(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu"),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        with get_parallel().override(shared_experts_tp_group=None):
            with self.assertRaisesRegex(ValueError, "must exist before construction"):
                linear.resolve_linear_parallel_group("shared_experts_tp")


if __name__ == "__main__":
    unittest.main()
