"""prepare_attn completes a sum the previous layer left, then adds the residual
and applies the input norm in the form the attention's quant format wants."""

import contextlib
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary import StageKind
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.fusions.allreduce import attn_input_fusions
from sglang.srt.layers.layer_boundary.ops import keep_output
from sglang.test.boundary_fixtures import prepare_input, stub_plan, stub_stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

KERNELS = (
    "fused_rms_mxfp4_quant",
    "fused_rms_fp8_group_quant",
    "_fused_rmsnorm_fp8_per_token_quant",
)


class Norm:
    weight = torch.ones(4)
    variance_epsilon = 1e-6

    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, residual=None, post_residual_addition=None):
        self.calls.append("norm")
        if residual is None:
            return hidden_states * 2
        return hidden_states * 2, hidden_states + residual

    def forward_with_allreduce_fusion(self, hidden_states, residual, **_):
        self.calls.append("fused_all_reduce_norm")
        return hidden_states * 3, hidden_states + residual


def communicator(norm):
    c = stub_plan()
    c.norm = norm
    # A layer inside the stack: an absent residual was written back before.
    c.enters_stack = False
    c.paths[BatchVariant.SEQUENCE_PARALLEL] = None
    c.paths[BatchVariant.INPUT_SCATTERED] = None
    c.paths[BatchVariant.CONTEXT_PARALLEL] = None
    c.qkv_latent_func = None
    # Construction picks the fused entries; call it under the platform patches.
    c._attn_input_fusions = attn_input_fusions(c)
    # A layer whose attention takes its input as it is and owes nothing on it.
    c.paths[BatchVariant.ORDINARY] = comm.StagePath(
        entry=comm.EntryPath(
            prepare=partial(
                comm_ops._run_entry,
                is_plain_add=True,
                step=partial(
                    comm_ops._update_read,
                    pre_move=None,
                    enters_stack=False,
                    read=comm.NORM_QUANT_READOUT,
                    update=comm.PLAIN_ADD,
                ),
                carried_fusions=c._attn_input_fusions,
            ),
            input_rows=comm.Layout(frozenset()),
            input_move=lambda hidden_states, **_: hidden_states,
            attn_input_adapter=comm_ops._attn_input_default,
        ),
        output=comm.OutputContract(comm.Layout(frozenset())),
        output_move=keep_output,
    )
    return c


@contextlib.contextmanager
def platform(*, use_aiter=False, gfx95=False, fusion=False, kernel_group=True):
    kernels = {name: MagicMock(name=name) for name in KERNELS}
    kernels["fused_rms_mxfp4_quant"].return_value = ("q", None, None, "r")
    kernels["fused_rms_fp8_group_quant"].return_value = (("q", "s"), None, None, "r")
    kernels["_fused_rmsnorm_fp8_per_token_quant"].return_value = ("q", "r")
    all_reduce = MagicMock(side_effect=lambda h: h * 5)
    # The group an unreduced output owes its sum over.
    group = SimpleNamespace(all_reduce=all_reduce)
    with contextlib.ExitStack() as stack:
        for name, mock in kernels.items():
            stack.enter_context(patch_communicator(name, mock, create=True))
        stack.enter_context(patch_communicator("_use_aiter", use_aiter))
        stack.enter_context(patch_communicator("_is_gfx95_supported", gfx95))
        stack.enter_context(
            patch_communicator("_use_aiter_bpreshuffle_gfx95", False, create=True)
        )
        stack.enter_context(
            patch_communicator("aiter_ar_fusion_applies", return_value=False)
        )
        stack.enter_context(
            patch_communicator("flashinfer_ar_fusion_applies", return_value=fusion)
        )
        # The group the fused kernel reduces over.
        stack.enter_context(
            patch_communicator(
                "post_experts_reduction_group",
                return_value=group if kernel_group else SimpleNamespace(),
            )
        )
        stack.enter_context(
            patch_communicator(
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False, is_dsa=False),
            )
        )
        stack.enter_context(
            patch_communicator("_batch_shards_over_cp", return_value=False)
        )
        yield kernels, all_reduce, group


class TestPrepareAttnSteps(CustomTestCase):
    def test_each_quant_format_uses_its_kernel(self):
        for quant_format, flags, kernel in (
            ("mxfp4", {"use_aiter": True, "gfx95": True}, "fused_rms_mxfp4_quant"),
            ("fp8", {"use_aiter": True, "gfx95": True}, "fused_rms_fp8_group_quant"),
            (
                "fp8_per_token",
                {"use_aiter": True},
                "_fused_rmsnorm_fp8_per_token_quant",
            ),
            ("fp8", {"use_aiter": True}, None),  # the group kernel needs gfx95
            ("mxfp4", {}, None),
        ):
            for residual in (None, torch.zeros(2, 4)):
                with (
                    self.subTest(
                        quant_format=quant_format,
                        flags=flags,
                        residual=residual is not None,
                    ),
                    platform(**flags) as (kernels, _, _),
                ):
                    norm = Norm()
                    prepare_input(
                        stub_stage(communicator(norm), StageKind.ATTENTION),
                        torch.ones(2, 4),
                        residual,
                        None,
                        quant_format=quant_format,
                    )
                    for name, mock in kernels.items():
                        self.assertEqual(mock.called, name == kernel, name)
                    self.assertEqual(norm.calls, [] if kernel else ["norm"])

    def test_missing_residual_takes_the_input_as_residual(self):
        for quant_format, flags in (
            ("", {}),
            ("mxfp4", {"use_aiter": True, "gfx95": True}),
            ("fp8", {"use_aiter": True, "gfx95": True}),
            ("fp8_per_token", {"use_aiter": True}),
        ):
            with self.subTest(quant_format=quant_format), platform(**flags):
                hidden_states = torch.ones(2, 4)
                _, residual = prepare_input(
                    stub_stage(communicator(Norm()), StageKind.ATTENTION),
                    hidden_states,
                    None,
                    None,
                    quant_format=quant_format,
                )
                self.assertIs(residual.residual, hidden_states)

    def test_a_pending_sum_is_completed_once(self):
        for fusion in (False, True):
            with (
                self.subTest(fusion=fusion),
                platform(fusion=fusion) as (_, all_reduce, group),
            ):
                norm = Norm()
                hidden_states = comm.UnreducedOutput(torch.ones(2, 4), group=group)
                prepare_input(
                    stub_stage(communicator(norm), StageKind.ATTENTION),
                    hidden_states,
                    torch.zeros(2, 4),
                    None,
                )
                self.assertEqual(all_reduce.call_count, 0 if fusion else 1)
                self.assertEqual(
                    norm.calls, ["fused_all_reduce_norm"] if fusion else ["norm"]
                )

    def test_a_pending_sum_keeps_the_quant_format_without_fusion(self):
        with platform(use_aiter=True) as (kernels, all_reduce, group):
            norm = Norm()
            prepare_input(
                stub_stage(communicator(norm), StageKind.ATTENTION),
                comm.UnreducedOutput(torch.ones(2, 4), group=group),
                torch.zeros(2, 4),
                None,
                quant_format="fp8_per_token",
            )
            self.assertEqual(all_reduce.call_count, 1)
            self.assertTrue(kernels["_fused_rmsnorm_fp8_per_token_quant"].called)
            self.assertEqual(norm.calls, [])

    def test_a_pending_sum_over_another_group_skips_the_fused_kernel(self):
        with platform(fusion=True, kernel_group=False) as (_, all_reduce, group):
            norm = Norm()
            prepare_input(
                stub_stage(communicator(norm), StageKind.ATTENTION),
                comm.UnreducedOutput(torch.ones(2, 4), group=group),
                torch.zeros(2, 4),
                None,
            )
            self.assertEqual(all_reduce.call_count, 1)
            self.assertEqual(norm.calls, ["norm"])

    def test_a_post_residual_addition_bypasses_the_fused_kernel(self):
        with platform(fusion=True) as (_, all_reduce, group):
            norm = Norm()
            prepare_input(
                stub_stage(communicator(norm), StageKind.ATTENTION),
                comm.UnreducedOutput(torch.ones(2, 4), group=group),
                torch.zeros(2, 4),
                None,
                post_residual_addition=torch.ones(2, 4),
            )
            self.assertEqual(all_reduce.call_count, 1)
            self.assertEqual(norm.calls, ["norm"])

    def test_an_owed_reduce_scatter_reaches_local_tokens_before_the_norm(self):
        for local_rows in (1, 0):
            with (
                self.subTest(local_rows=local_rows),
                platform(fusion=True) as (_, all_reduce, _),
            ):
                norm = Norm()
                partial = torch.ones(3, 4)
                local = torch.full((local_rows, 4), 7.0)
                step = MagicMock(return_value=local)
                hidden_states, residual = prepare_input(
                    stub_stage(communicator(norm), StageKind.ATTENTION),
                    comm.UnreducedOutput(partial, reduce_to_dp_local=step),
                    torch.zeros(local_rows, 4),
                    None,
                )
                step.assert_called_once_with(partial)
                self.assertEqual(all_reduce.call_count, 0)
                if local_rows:
                    self.assertEqual(norm.calls, ["norm"])
                    torch.testing.assert_close(hidden_states, local * 2)
                else:
                    self.assertEqual(norm.calls, [])
                    self.assertIs(residual.residual, local)

    def test_an_owed_reduce_scatter_is_not_offered_to_a_fused_kernel(self):
        with platform(fusion=True) as (_, all_reduce, _):
            c = communicator(Norm())
            takes_anything = MagicMock(return_value=("fused", "fused"))
            c.paths[BatchVariant.ORDINARY] = msgspec.structs.replace(
                c.paths.get(BatchVariant.ORDINARY),
                entry=msgspec.structs.replace(
                    c.paths.get(BatchVariant.ORDINARY).entry,
                    prepare=partial(
                        comm_ops._run_entry,
                        is_plain_add=True,
                        step=partial(
                            comm_ops._update_read,
                            pre_move=None,
                            enters_stack=False,
                            read=comm.NORM_QUANT_READOUT,
                            update=comm.PLAIN_ADD,
                        ),
                        carried_fusions=(takes_anything,),
                    ),
                ),
            )
            partial_sum = torch.ones(3, 4)
            step = MagicMock(return_value=torch.full((1, 4), 7.0))
            prepare_input(
                stub_stage(c, StageKind.ATTENTION),
                comm.UnreducedOutput(partial_sum, reduce_to_dp_local=step),
                torch.zeros(1, 4),
                None,
            )
            step.assert_called_once_with(partial_sum)
            takes_anything.assert_not_called()
            self.assertEqual(all_reduce.call_count, 0)


class TestFusedReadForms(CustomTestCase):
    def test_consumer_form_and_backend_policy_are_both_required(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            Fp8Input,
            NormQuantReadout,
        )

        for form in (None, Fp8Input.TUPLE, Fp8Input.TUPLE_AND_BF16):
            for enabled in (False, True):
                for disabled_by_env in (False, True):
                    with (
                        self.subTest(
                            form=form, enabled=enabled, disabled_by_env=disabled_by_env
                        ),
                        patch_communicator("_use_aiter", True),
                        patch_communicator(
                            "get_exec",
                            return_value=SimpleNamespace(
                                comm=SimpleNamespace(
                                    enable_aiter_allreduce_fusion=enabled
                                )
                            ),
                        ),
                        patch_communicator(
                            "get_bool_env_var", return_value=disabled_by_env
                        ),
                    ):
                        plan = SimpleNamespace(norm=MagicMock(), fusions=None)
                        (candidate,) = attn_input_fusions(
                            plan, NormQuantReadout(fp8_input=form)
                        )
                        expected = form is not None and enabled and not disabled_by_env
                        self.assertEqual(candidate.keywords["fuses_quant"], expected)
                        self.assertEqual(
                            candidate.keywords["keep_bf16"],
                            expected and form is Fp8Input.TUPLE_AND_BF16,
                        )


if __name__ == "__main__":
    unittest.main()
