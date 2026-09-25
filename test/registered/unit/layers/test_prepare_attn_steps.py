"""prepare_attn completes a sum the previous layer left, then adds the residual
and applies the input norm in the form the attention's quant format wants."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
    c = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
    c.input_layernorm = norm
    c._sp_variant = None
    c.qkv_latent_func = None
    c._context = None
    c.enable_fused_ar_quant = False
    c.fused_ar_quant_keep_bf16 = False
    c._communicate_simple_fn = lambda hidden_states, **_: hidden_states
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
            stack.enter_context(patch.object(comm, name, mock, create=True))
        stack.enter_context(patch.object(comm, "_use_aiter", use_aiter))
        stack.enter_context(patch.object(comm, "_is_gfx95_supported", gfx95))
        stack.enter_context(
            patch.object(comm, "_use_aiter_bpreshuffle_gfx95", False, create=True)
        )
        stack.enter_context(
            patch.object(comm, "apply_aiter_all_reduce_fusion", return_value=False)
        )
        stack.enter_context(
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=fusion)
        )
        # The group the fused kernel reduces over.
        stack.enter_context(
            patch.object(
                comm,
                "post_experts_reduction_group",
                return_value=group if kernel_group else SimpleNamespace(),
            )
        )
        stack.enter_context(
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False, is_dsa=False),
            )
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
                    communicator(norm).prepare_attn(
                        torch.ones(2, 4), residual, None, quant_format=quant_format
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
                _, residual = communicator(Norm()).prepare_attn(
                    hidden_states, None, None, quant_format=quant_format
                )
                self.assertIs(residual, hidden_states)

    def test_a_pending_sum_is_completed_once(self):
        for fusion in (False, True):
            with (
                self.subTest(fusion=fusion),
                platform(fusion=fusion) as (_, all_reduce, group),
            ):
                norm = Norm()
                hidden_states = comm.UnreducedOutput(torch.ones(2, 4), group=group)
                communicator(norm).prepare_attn(hidden_states, torch.zeros(2, 4), None)
                self.assertEqual(all_reduce.call_count, 0 if fusion else 1)
                self.assertEqual(
                    norm.calls, ["fused_all_reduce_norm"] if fusion else ["norm"]
                )

    def test_a_pending_sum_keeps_the_quant_format_without_fusion(self):
        with platform(use_aiter=True) as (kernels, all_reduce, group):
            norm = Norm()
            communicator(norm).prepare_attn(
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
            communicator(norm).prepare_attn(
                comm.UnreducedOutput(torch.ones(2, 4), group=group),
                torch.zeros(2, 4),
                None,
            )
            self.assertEqual(all_reduce.call_count, 1)
            self.assertEqual(norm.calls, ["norm"])

    def test_a_post_residual_addition_bypasses_the_fused_kernel(self):
        with platform(fusion=True) as (_, all_reduce, group):
            norm = Norm()
            communicator(norm).prepare_attn(
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
                hidden_states, residual = communicator(norm).prepare_attn(
                    comm.UnreducedOutput(partial, reduce_and_redistribute=step),
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
                    self.assertIs(residual, local)

    def test_an_unreduced_output_without_a_residual_is_rejected(self):
        for step in (None, MagicMock()):
            with (
                self.subTest(redistributes=step is not None),
                platform(fusion=True) as (_, all_reduce, _),
            ):
                norm = Norm()
                with self.assertRaises(RuntimeError):
                    communicator(norm).prepare_attn(
                        comm.UnreducedOutput(
                            torch.ones(2, 4), reduce_and_redistribute=step
                        ),
                        None,
                        None,
                    )
                self.assertEqual(all_reduce.call_count, 0)
                if step is not None:
                    step.assert_not_called()
                self.assertEqual(norm.calls, [])


if __name__ == "__main__":
    unittest.main()
