"""Output operations and collective fallbacks share one boundary decision."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.runtime_context import get_forward
from sglang.test.boundary_fixtures import finish_exit
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBoundaryOutputPolicy(unittest.TestCase):
    def batch(self, max_len=True):
        return SimpleNamespace(
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: max_len),
            forward_mode=SimpleNamespace(
                is_context_parallel_extend=lambda: False, is_decode_or_idle=lambda: True
            ),
        )

    def test_npu_weight_cache_belongs_to_one_prepare_call(self):
        from sglang.srt.layers.layer_boundary import prepare as ops
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2)
        layer = fixture.build(fixture.layer_case(0, 2), parallel)
        batch = self.batch()
        cache = object()
        events = []

        def reduce(value):
            events.append("reduce")
            return value * 2

        def preload(value, weights):
            events.append("cache")
            self.assertIs(weights, cache)
            torch.testing.assert_close(value, torch.full((2, 4), 2.0))

        with (
            fixture.planning(parallel),
            patch.object(ops, "_is_npu", True),
            patch.object(ops, "prepare_weight_cache", side_effect=preload, create=True),
            patch_communicator("attention_tensor_model_parallel_all_reduce", reduce),
        ):
            for weights in (cache, None):
                batch.residual_stream = ResidualStream(torch.zeros(2, 4))
                hidden = layer.attn.finish(torch.ones(2, 4), batch)
                result = layer.ffn.prepare(hidden, batch, cache=weights)
                torch.testing.assert_close(result, torch.full((2, 4), 4.0))
        self.assertEqual(events, ["reduce", "cache", "reduce"])

    def test_token_slice_reads_the_current_parallel_rank(self):
        from sglang.srt.layers.layer_boundary.ops import tp_slice

        hidden = torch.arange(8).reshape(4, 2)
        for rank in (0, 1):
            with fixture.planning(
                fixture.parallel_of(attn_dp=1, attn_tp=2, tp_rank=rank)
            ):
                value, residual = tp_slice(hidden, hidden)
            torch.testing.assert_close(value, hidden.chunk(2)[rank])
            torch.testing.assert_close(residual, hidden.chunk(2)[rank])

    def test_cli_policy_reaches_execution_config(self):
        import argparse

        from sglang.srt.runtime_context import get_exec, publish, reset_context
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for policy in ("auto", "ar", "rs", "rsv", "rs+rsv"):
            with self.subTest(policy=policy):
                args = ["--model", "dummy", "--boundary-reduction", policy]
                record = ServerArgs.from_cli_args(parser.parse_args(args))
                reset_context()
                try:
                    record.resolve_once()
                    publish(record, role="test")
                    self.assertEqual(
                        get_exec().comm.boundary_reduction,
                        "rs+rsv" if policy == "auto" else policy,
                    )
                finally:
                    reset_context()

    def test_step3_shared_and_routed_outputs_have_one_completion(self):
        from sglang.srt.models.step3_vl import Step3TextDecoderLayer

        parallel = SimpleNamespace(tp_size=2)
        for fused_shared in (False, True):
            for deferred in (False, True):
                calls = []
                layer = SimpleNamespace(
                    num_fused_shared_experts=int(fused_shared),
                    moe=lambda x: x * (0.75 if fused_shared else 0.25),
                    share_expert=lambda x: x * 0.5,
                )
                with (
                    patch.object(moe_utils, "get_parallel", return_value=parallel),
                    patch.object(
                        moe_utils, "post_experts_output_is_complete", return_value=False
                    ),
                    patch(
                        "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
                        side_effect=lambda x: calls.append("AR") or x * 2,
                    ),
                    get_forward().scoped(
                        fuse_mlp_allreduce=deferred, mlp_reduce_scatter=False
                    ),
                ):
                    result = Step3TextDecoderLayer.moe_mlp_forward(
                        layer, torch.ones(2, 4)
                    )
                self.assertEqual(calls, [] if deferred else ["AR"])
                torch.testing.assert_close(
                    result, torch.full((2, 4), 0.75 if deferred else 1.5)
                )

    def test_dp_skip_and_collective_are_selected_together(self):
        import itertools

        for policy, varlen, max_len, can_rs in itertools.product(
            ("ar", "rs", "rsv", "rs+rsv"), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                policy=policy, varlen=varlen, max_len=max_len, can_rs=can_rs
            ):
                parallel = fixture.parallel_of(attn_dp=2, attn_tp=1)
                layer = fixture.build(
                    fixture.layer_case(2, 3), parallel, boundary_reduction=policy
                )
                calls = []

                def move(step, batch, value):
                    calls.append(step.__name__)
                    return value[:1]

                selected = (
                    "dp_reduce_scatterv"
                    if varlen and policy in ("rsv", "rs+rsv")
                    else "dp_reduce_scatter"
                    if max_len and can_rs and policy in ("rs", "rs+rsv")
                    else "_dp_scatter_step"
                )
                with (
                    fixture.planning(parallel),
                    patch_communicator("should_use_dp_reduce_scatterv", lambda: varlen),
                    patch_communicator("can_use_dp_reduce_scatter", lambda: can_rs),
                    patch_communicator("to_dp_local", move),
                ):
                    with layer.ffn.plan.output.ffn_exit(
                        self.batch(max_len), stream=ResidualStream()
                    ) as output:
                        self.assertEqual(
                            get_forward().mlp_reduce_scatter,
                            selected != "_dp_scatter_step",
                        )
                    result, _ = finish_exit(output, torch.ones(2, 4), torch.zeros(1, 4))
                self.assertEqual(calls, [selected])
                self.assertEqual(result.shape, (1, 4))

    def test_postprocess_only_scatters_an_already_reduced_output(self):
        # The operation-scheduled API used by LongCat NextN receives an MLP
        # output that was already summed. An enabled RSv must not sum it again.
        for policy in ("ar", "rs", "rsv", "rs+rsv"):
            for varlen in (False, True):
                with self.subTest(policy=policy, varlen=varlen):
                    parallel = fixture.parallel_of(attn_dp=2, attn_tp=1)
                    layer = fixture.build(
                        fixture.layer_case(2, 3), parallel, boundary_reduction=policy
                    )
                    calls = []
                    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
                    residual = torch.ones(1, 4)
                    batch = self.batch()
                    batch.residual_stream = ResidualStream(residual)

                    def move(step, batch, value):
                        calls.append(step.__name__)
                        return value[:1]

                    with (
                        fixture.planning(parallel),
                        patch_communicator(
                            "should_use_dp_reduce_scatterv", lambda: varlen
                        ),
                        patch_communicator("can_use_dp_reduce_scatter", lambda: True),
                        patch_communicator("to_dp_local", move),
                    ):
                        output = layer.ffn.finish_complete_output(hidden, batch)
                        value, saved_residual = batch.residual_stream.export(output)
                    self.assertEqual(calls, ["_dp_scatter_step"])
                    torch.testing.assert_close(value, hidden[:1])
                    self.assertIs(saved_residual, residual)

    def test_moe_cannot_skip_from_topology_without_boundary_request(self):
        with (
            patch.object(moe_utils, "should_use_dp_reduce_scatterv", return_value=True),
            patch.object(
                moe_utils, "post_experts_output_is_complete", return_value=False
            ),
        ):
            for skip in (False, True):
                with get_forward().scoped(
                    fuse_mlp_allreduce=False, mlp_reduce_scatter=skip
                ):
                    self.assertEqual(
                        moe_utils.should_skip_post_experts_all_reduce(is_tp_path=True),
                        skip,
                    )

    def test_output_transform_stays_before_transport_and_after_producer_ar(self):
        # The compute stub obeys the same skip flag as RowParallelLinear.
        # Its trace distinguishes AR -> multiply from multiply -> RS.
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                calls = []

                def scale(value):
                    calls.append("multiply")
                    return value * 3

                parallel = fixture.parallel_of(attn_dp=2, attn_tp=1)
                layer = fixture.build(
                    fixture.layer_case(1, 3),
                    parallel,
                    output=OutputTransform(scale, before_reduce_scatter=True),
                    boundary_reduction="ar" if disabled else "rs+rsv",
                )

                def move(step, batch, value):
                    if "reduce" in step.__name__:
                        calls.append("RS")
                        value = value * 2
                    else:
                        calls.append("slice")
                    return value[:1]

                with (
                    fixture.planning(parallel),
                    patch_communicator("should_use_dp_reduce_scatterv", lambda: False),
                    patch_communicator("can_use_dp_reduce_scatter", lambda: True),
                    patch_communicator("to_dp_local", move),
                ):
                    with layer.ffn.plan.output.ffn_exit(
                        self.batch(), stream=ResidualStream()
                    ) as output:
                        value = torch.ones(2, 4)
                        if not get_forward().mlp_reduce_scatter:
                            calls.append("AR")
                            value = value * 2
                    value, _ = finish_exit(output, value, torch.zeros(1, 4))
                self.assertEqual(
                    calls,
                    ["AR", "multiply", "slice"] if disabled else ["multiply", "RS"],
                )
                torch.testing.assert_close(value, torch.full((1, 4), 6.0))

    def test_transform_without_pre_rs_implementation_uses_ar(self):
        parallel = fixture.parallel_of(attn_dp=2, attn_tp=1)
        layer = fixture.build(
            fixture.layer_case(1, 3),
            parallel,
            output=OutputTransform(lambda x: x.square()),
        )
        output = layer.ffn.plan.paths.get(BatchVariant.ORDINARY).output
        self.assertFalse(output.may_defer_to_next)
        self.assertFalse(output.may_reduce_scatter)
        self.assertFalse(output.may_reduce_scatterv)


if __name__ == "__main__":
    unittest.main()
