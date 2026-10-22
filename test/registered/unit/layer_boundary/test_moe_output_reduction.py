"""When a MoE block all-reduces its own output, and why it does not."""

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import (
    post_experts_output_is_complete,
    reduce_moe_output,
    should_add_replicated_moe_output,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def a2a(name=None):
    names = ("flashinfer", "pplx", "flashinfer_megamoe")
    return types.SimpleNamespace(
        **{f"is_{n}": (lambda n=n: n == name) for n in names},
        is_none=lambda: name is None,
    )


@contextlib.contextmanager
def moe_config(
    *,
    tp_size=2,
    tp_rank=0,
    dwdp_size=1,
    backend=None,
    fp4_allgather=False,
    reduce_scatterv=False,
):
    with (
        patch.object(
            moe_utils,
            "get_parallel",
            return_value=types.SimpleNamespace(
                tp_size=tp_size, tp_rank=tp_rank, dwdp_size=dwdp_size
            ),
        ),
        patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a(backend)),
        patch.object(
            moe_utils,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=fp4_allgather,
        ),
        patch.object(
            moe_utils, "should_use_dp_reduce_scatterv", return_value=reduce_scatterv
        ),
    ):
        yield


class TestReduceMoeOutput(CustomTestCase):
    def all_reduces(self, **flags):
        calls = []
        with (
            patch(
                "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
                side_effect=lambda x: calls.append(x) or x * 2,
            ),
            get_forward().scoped(**flags),
        ):
            output = reduce_moe_output(torch.ones(2, 3))
        return len(calls), output

    def test_partial_output_is_reduced_once(self):
        with moe_config():
            count, output = self.all_reduces()
        self.assertEqual(count, 1)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))

    def test_single_rank_has_nothing_to_sum(self):
        with moe_config(tp_size=1):
            self.assertEqual(self.all_reduces()[0], 0)

    def test_a_later_step_owns_the_sum(self):
        for flags, config in (
            ({"mlp_reduce_scatter": True}, {}),
            ({"mlp_reduce_scatter": True}, {"reduce_scatterv": True}),
        ):
            with self.subTest(flags=flags, config=config), moe_config(**config):
                self.assertEqual(self.all_reduces(**flags)[0], 0)
                # Who runs the sum does not change whether one is owed.
                self.assertFalse(post_experts_output_is_complete(is_tp_path=True))

    def test_output_is_already_complete(self):
        for config in (
            {"backend": "flashinfer"},
            {"backend": "pplx"},
            {"backend": "flashinfer_megamoe"},
            {"dwdp_size": 2},
            {"fp4_allgather": True},
        ):
            with self.subTest(**config), moe_config(**config):
                self.assertEqual(self.all_reduces()[0], 0)
                self.assertTrue(post_experts_output_is_complete(is_tp_path=True))

    def test_fp4_allgather_also_completes_the_ep_sum(self):
        # Its combine reduce-scatters over _TP, which spans the EP group.
        with moe_config(fp4_allgather=True):
            self.assertTrue(post_experts_output_is_complete(is_tp_path=False))
        with moe_config():
            self.assertFalse(post_experts_output_is_complete(is_tp_path=False))


class TestReplicatedMoeOutput(CustomTestCase):
    """A shared expert replicated with tp_size=1 holds its full output on every
    rank; whatever sums the MoE output over TP must count it once."""

    def summed_over_two_ranks(self, *, flags=None, **config):
        routed = [torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)]
        shared = torch.full((2, 3), 10.0)
        outputs = []
        for rank in (0, 1):
            with (
                moe_config(tp_rank=rank, **config),
                get_forward().scoped(**(flags or {})),
            ):
                output = routed[rank]
                if should_add_replicated_moe_output():
                    output = output + shared
            outputs.append(output)
        return outputs

    def test_a_later_sum_counts_it_once(self):
        for flags, config in (
            ({"mlp_reduce_scatter": True}, {}),
            ({"mlp_reduce_scatter": True}, {"reduce_scatterv": True}),
        ):
            with self.subTest(flags=flags, config=config):
                outputs = self.summed_over_two_ranks(flags=flags, **config)
                torch.testing.assert_close(sum(outputs), torch.full((2, 3), 13.0))

    def test_every_rank_adds_it_to_a_reduced_or_complete_output(self):
        for config in (
            {},
            {"backend": "flashinfer"},
            {"fp4_allgather": True},
            {"reduce_scatterv": True},
        ):
            with self.subTest(**config):
                outputs = self.summed_over_two_ranks(**config)
                torch.testing.assert_close(outputs[1], torch.full((2, 3), 12.0))

    def test_a_single_rank_adds_it(self):
        with (
            moe_config(tp_size=1, tp_rank=0),
            get_forward().scoped(mlp_reduce_scatter=True),
        ):
            self.assertTrue(should_add_replicated_moe_output())


class TestModelsWithExplicitDpCompletion(CustomTestCase):
    def test_direct_dp_exits_publish_their_selected_sum(self):
        from sglang.srt.layers import dp_attention
        from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup, TokenAxis
        from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
        from sglang.srt.layers.layer_boundary.row_transport import (
            TransportExit,
            TransportRows,
        )
        from sglang.srt.models import deepseek_v4
        from sglang.srt.models import deepseek_v4_boundary as boundary
        from sglang.srt.models.deepseek_common import moe_transport

        for non_cuda in (False, True):
            for use_rsv in (False, True):
                with self.subTest(non_cuda=non_cuda, reduce_scatterv=use_rsv):
                    trace = []

                    def rsv(value, *, output, sizes):
                        trace.append("RSv")
                        output.copy_(value[:2] * 2)

                    def scatter(output, value, batch):
                        trace.append("slice")
                        output.copy_(value[:2])

                    def reduce(value):
                        self.assertFalse(get_forward().mlp_reduce_scatter)
                        trace.append("AR")
                        return value * 2

                    def compute(value, batch, **kwargs):
                        self.assertEqual(get_forward().mlp_reduce_scatter, use_rsv)
                        return reduce(value) if non_cuda and not use_rsv else value

                    group = types.SimpleNamespace(reduce_scatterv=rsv)
                    parallel = types.SimpleNamespace(
                        attn_dp_size=2, attn_tp_size=1, tp_size=2, tp_group=group
                    )
                    model = types.SimpleNamespace(
                        dsa_enable_prefill_cp=False,
                        boundary_mhc_fusions=False,
                        config=types.SimpleNamespace(model_type="deepseek_v4"),
                        mlp=compute,
                    )
                    batch = types.SimpleNamespace(
                        dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: True)
                    )
                    rows = TransportRows(
                        Layout(frozenset({TokenAxis.ATTN_DP})),
                        Layout(frozenset()),
                        Layout(frozenset({TokenAxis.ATTN_DP})),
                        SumGroup.MOE_OUTPUT,
                    )
                    with contextlib.ExitStack() as patches:
                        patches.enter_context(
                            get_forward().scoped(mlp_reduce_scatter=False)
                        )
                        for module in (moe_transport, boundary):
                            patches.enter_context(
                                patch.object(
                                    module, "get_parallel", return_value=parallel
                                )
                            )
                        for context in (
                            patch.object(
                                moe_transport, "get_moe_a2a_backend", return_value=a2a()
                            ),
                            patch.object(moe_transport, "SHARED_EXPERT_LOCAL", False),
                            patch.object(
                                moe_transport, "is_cp_active", return_value=False
                            ),
                            patch.object(
                                moe_transport,
                                "is_dp_gatherv_active",
                                return_value=False,
                            ),
                            patch.object(
                                moe_transport,
                                "should_use_dp_reduce_scatterv",
                                return_value=use_rsv,
                            ),
                            patch.object(
                                moe_transport,
                                "get_global_dp_buffer",
                                side_effect=lambda g: torch.empty(4, 3),
                            ),
                            patch.object(
                                moe_transport,
                                "get_local_dp_buffer",
                                side_effect=lambda g: torch.empty(2, 3),
                            ),
                            patch.object(
                                moe_transport,
                                "get_dp_global_num_tokens",
                                return_value=[2, 2],
                            ),
                            patch.object(
                                moe_transport,
                                "dp_gather_replicate",
                                side_effect=lambda out, value, batch: out.fill_(1),
                            ),
                            patch.object(
                                moe_transport, "dp_scatter", side_effect=scatter
                            ),
                            patch(
                                "sglang.srt.layers.moe.should_use_dp_reduce_scatterv",
                                return_value=use_rsv,
                            ),
                            deepseek_v4.envs.SGLANG_DP_USE_REDUCE_SCATTER.override(
                                False
                            ),
                            patch.object(
                                dp_attention, "is_dp_gatherv_active", return_value=False
                            ),
                            patch.object(
                                dp_attention,
                                "get_global_dp_buffer",
                                side_effect=lambda g: torch.empty(4, 3),
                            ),
                            patch.object(
                                dp_attention,
                                "get_local_dp_buffer",
                                side_effect=lambda g: torch.empty(2, 3),
                            ),
                            patch.object(
                                dp_attention,
                                "get_dp_global_num_tokens",
                                return_value=[2, 2],
                            ),
                            patch.object(
                                dp_attention, "dp_scatter", side_effect=scatter
                            ),
                            patch(
                                "sglang.srt.layers.moe.post_experts_all_reduce",
                                side_effect=reduce,
                            ),
                        ):
                            patches.enter_context(context)
                        if non_cuda:
                            output = (
                                deepseek_v4.NonCudaDecoderMixin._run_moe_ffn_dp_sync(
                                    model,
                                    torch.ones(2, 3),
                                    batch,
                                    input_ids=None,
                                    input_ids_global=None,
                                )
                            )
                        else:
                            transport = boundary.MoeTransport(
                                model,
                                torch.ones(2, 3),
                                batch,
                                boundary.LayerCall(None),
                                rows,
                            )
                            stage = types.SimpleNamespace(
                                plan=types.SimpleNamespace(
                                    path_for=lambda b: types.SimpleNamespace(
                                        output=types.SimpleNamespace(update=PLAIN_ADD),
                                        exit=types.SimpleNamespace(
                                            writes_residual=False
                                        ),
                                    )
                                )
                            )
                            batch.residual_stream = ResidualStream(torch.ones(2, 3))
                            batch.residual_stream.transport = (stage, transport)
                            with TransportExit(stage, batch) as completion:
                                value = model.mlp(transport.input, batch)
                                output = completion.finish(value)
                        self.assertFalse(get_forward().mlp_reduce_scatter)
                    self.assertEqual(trace, ["RSv"] if use_rsv else ["AR", "slice"])
                    torch.testing.assert_close(output, torch.full((2, 3), 2.0))


if __name__ == "__main__":
    unittest.main()
