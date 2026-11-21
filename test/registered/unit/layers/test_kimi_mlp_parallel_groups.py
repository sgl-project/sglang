"""Kimi dense and shared MLPs freeze placement without changing row policy."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers import linear
from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, stage="weekly", runner_config="cpu")


def build_mlp(group=None, *, width=16, quant_config=None):
    kwargs = {} if group is None else dict(parallel_group=group)
    return kimi_k3.KimiK3MLP(
        width,
        2 * width,
        "silu",
        quant_config=quant_config,
        **kwargs,
    )


class _RoutedExperts(torch.nn.Module):
    routing_method_type = None
    runner_backend = MoeRunnerBackend.TRITON
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
        is_flashinfer_megamoe=lambda: False,
        is_deepep=lambda: ep_a2a,
        is_ascend_ops=lambda: False,
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
    rank, size = rank_size(layer)[0], rank_size(layer)[1]
    with parallel_scope(
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
                    with parallel_scope(shared_experts_tp_group=shared):
                        for group in (
                            None,
                            "tp",
                            "attn_tp",
                            "replicated",
                            "shared_experts_tp",
                        ):
                            module = build_mlp(group)
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
                                self.assertEqual(rank_size(layer), (r, s))
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
                                attn.all_reduce.side_effect = lambda v: v * (4 // dp)
                                row = name == "down_proj"
                                reference = F.linear(x, expected)
                                with (
                                    parallel_scope(tp_group=tp, attn_tp_group=attn),
                                    patch.object(
                                        linear,
                                        "is_allocation_symmetric",
                                        return_value=True,
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
                                    # The stage boundary completes the sum.
                                    self.assertFalse(layer.reduce_results)
                                    self.assertEqual(
                                        layer.use_dp_attention_reduce, auto_attn
                                    )
                                    self.assertIs(
                                        allocation.call_args.args[0],
                                        attn if auto_attn else tp,
                                    )
                                    if not auto_attn:
                                        self.assertEqual(
                                            allocation.call_args.kwargs["disabled"],
                                            layer.tp_group is None,
                                        )
                                else:
                                    allocation.assert_not_called()

    def test_moe_shared_consumer_preserves_selection_and_execution_handle(self):
        def publish_case(rank, dp, **shared_flags):
            reset_context()
            # Under DP attention the dense MLP shards over attention TP;
            # without an a2a backend the shared experts stay on full TP.
            server = ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=4,
                attn_dp_size=dp,
                enable_dense_mlp_attn_tp=dp > 1,
                **shared_flags,
            )
            publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
            initialize_dp_attention_flags(server)

        for rank in range(4):
            for dp in (1, 2):
                attn_tp = 4 // dp
                attn = SimpleNamespace(world_size=attn_tp, rank_in_group=rank % attn_tp)
                shared = SimpleNamespace(world_size=2, rank_in_group=rank % 2)
                for ep, requested, enabled, expected_size in (
                    (False, None, False, 4),
                    (True, None, False, 1),
                    (True, 1, False, 1),
                    (True, 2, False, 2),
                    (True, None, True, attn_tp),
                ):
                    publish_case(
                        rank,
                        dp,
                        shared_experts_tp_size=requested,
                        enable_shared_experts_attn_tp=enabled,
                    )
                    with parallel_scope(
                        shared_experts_tp_group=attn if enabled else shared,
                        attn_tp_group=attn,
                    ):
                        owner, module = build_shared_experts(ep_a2a=ep)
                        comm = ep and expected_size > 1
                        self.assertEqual(owner._shared_experts_tp is not None, comm)
                        self.assertEqual(
                            owner._shared_experts_tp1, ep and expected_size == 1
                        )
                        if comm:
                            self.assertIs(
                                owner._shared_experts_tp.group,
                                shared if requested == 2 else attn,
                            )
                            self.assertFalse(owner._shared_experts_tp.across_dp)
                        self.assertFalse(module._dense_attn_tp)
                        for name in ("gate_up_proj", "down_proj"):
                            layer = getattr(module, name)
                            self.assertEqual(
                                rank_size(layer),
                                (rank % expected_size, expected_size),
                            )
                            expected, _ = load_projection(layer)
                            torch.testing.assert_close(layer.weight, expected)
                        self.assertFalse(module.down_proj.reduce_results)
                        self.assertFalse(module.down_proj.use_dp_attention_reduce)
                publish_case(rank, dp, shared_experts_tp_size=2)
                with self.assertRaisesRegex(ValueError, "requires an EP a2a"):
                    build_shared_experts(ep_a2a=False)
                publish_case(rank, dp, shared_experts_tp_size=3)
                with parallel_scope(
                    shared_experts_tp_group=SimpleNamespace(
                        world_size=3, rank_in_group=rank % 3
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "must be divisible"):
                        build_shared_experts()
                publish_case(rank, dp)
                _, disabled = build_shared_experts(shared=0)
                self.assertIsNone(disabled)

    def test_handle_shared_experts_tp_dp_spanning(self):
        from sglang.srt.arg_groups import parallel_hook

        model_cfg = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
        )

        def run_hook(tp_size, attn_dp_size, shared_size, attn_cp_size=1):
            cfg = SimpleNamespace(
                tp_size=tp_size,
                attn_dp_size=attn_dp_size,
                shared_experts_tp_size=shared_size,
                model_path="dummy",
                moe_a2a_backend="megamoe",
            )
            view = SimpleNamespace(attn_cp_size=attn_cp_size)
            with (
                patch.object(parallel_hook, "resolving_view", return_value=cfg),
                patch.object(parallel_hook, "resolved_view", return_value=view),
                patch.object(parallel_hook, "model_config_of", return_value=model_cfg),
            ):
                parallel_hook.handle_shared_experts_tp(SimpleNamespace())

        # ADP8 with attention TP 1 and CP 1 accepts full TP size, rejects partial DP spans.
        run_hook(tp_size=8, attn_dp_size=8, shared_size=8)
        for bad in (2, 4):
            with self.assertRaisesRegex(ValueError, "equal TP size"):
                run_hook(tp_size=8, attn_dp_size=8, shared_size=bad)
        # ADP4 has attention TP 2: accepts 2 (within replica), rejects 8 (spans DP).
        run_hook(tp_size=8, attn_dp_size=4, shared_size=2)
        with self.assertRaisesRegex(ValueError, "equal TP size"):
            run_hook(tp_size=8, attn_dp_size=4, shared_size=8)

    def test_dp_spanning_shared_experts_tp(self):
        from sglang.srt.layers.dp_attention import DpPaddingMode

        reset_context()
        server = ServerArgs(
            model_path="dummy",
            device="cpu",
            tp_size=4,
            attn_dp_size=4,
            shared_experts_tp_size=4,
        )
        publish(server, role="test", ranks=SpawnRanks(world_rank=0))
        initialize_dp_attention_flags(server)

        tp_group = SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            unique_name="tp:0",
            all_reduce=Mock(side_effect=lambda t: t),
        )
        attn_group = SimpleNamespace(world_size=1, rank_in_group=0)
        with (
            parallel_scope(
                tp_group=tp_group,
                attn_tp_group=attn_group,
                shared_experts_tp_group=tp_group,
            ),
            patch.object(kimi_k3.k3_sp_collective, "GroupCollective") as k3_cls,
        ):
            k3 = k3_cls.return_value
            owner, _ = build_shared_experts(ep_a2a=True)

        shared_tp = owner._shared_experts_tp
        self.assertIsInstance(shared_tp, kimi_k3._DpSharedExpertsTp)
        self.assertTrue(shared_tp.across_dp)

        # NPU dual-stream overlap stays off for DP-spanning shared-expert TP.
        owner._sbo_shared_overlap = True
        owner.use_latent_moe = True
        with (
            patch.object(kimi_k3, "_is_npu", True),
            patch.object(
                kimi_k3.envs.SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM,
                "get",
                return_value=True,
            ),
            patch.object(
                kimi_k3,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_deepep=lambda: True),
            ),
        ):
            self.assertFalse(
                owner._can_overlap_shared_experts_npu(torch.zeros((2, 16)))
            )

        max_fb = SimpleNamespace(dp_padding_mode=DpPaddingMode.MAX_LEN)
        sum_fb = SimpleNamespace(dp_padding_mode=DpPaddingMode.SUM_LEN)
        x = torch.zeros((2, 16))
        y = torch.zeros((8, 16))

        # MAX_LEN: k3 fast path.
        k3.all_gather.return_value = y
        k3.reduce_scatter.return_value = x
        with patch.object(kimi_k3, "_note_dp_gather_in_prefill_graph") as note:
            self.assertIs(shared_tp.gather(x, max_fb), y)
            note.assert_called_once()
        self.assertIs(shared_tp.reduce_scatter(y, x, max_fb), x)

        # MAX_LEN: fallback to DP gather / DP reduce-scatter.
        k3.all_gather.return_value = None
        k3.reduce_scatter.return_value = None
        with (
            patch.object(kimi_k3, "get_global_dp_buffer", return_value=y),
            patch.object(kimi_k3, "dp_gather_partial") as dp_gather,
            patch.object(kimi_k3, "dp_reduce_scatter_tensor") as dp_rs,
        ):
            self.assertIs(shared_tp.gather(x, max_fb), y)
            dp_gather.assert_called_once_with(y, x, max_fb)
            out = shared_tp.reduce_scatter(y, x, max_fb)
            self.assertEqual(out.shape, x.shape)
            dp_rs.assert_called_once()

        # SUM_LEN with an idle (0-row) rank: DP gather + all-reduce + DP scatter.
        idle = torch.zeros((0, 16))
        with (
            patch.object(kimi_k3, "get_global_dp_buffer", return_value=y),
            patch.object(kimi_k3, "dp_gather_partial") as dp_gather,
            patch.object(kimi_k3, "dp_scatter") as dp_scatter,
        ):
            self.assertIs(shared_tp.gather(idle, sum_fb), y)
            dp_gather.assert_called_once_with(y, idle, sum_fb)
            out = shared_tp.reduce_scatter(y, idle, sum_fb)
            self.assertEqual(out.shape, idle.shape)
            tp_group.all_reduce.assert_called_once_with(y)
            dp_scatter.assert_called_once()

        with self.assertRaises(AssertionError):
            shared_tp.gather(x, None)


if __name__ == "__main__":
    unittest.main()
