"""Aux owns retained storage, including captures from reusable gather buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList, AuxHiddenStatePacker
from sglang.srt.layers.layer_boundary import StageKind
from sglang.srt.layers.layer_boundary.residual.access import final_norm_pair
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.boundary_fixtures import prepare_attention, stub_plan, stub_stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestAuxStorage(CustomTestCase):
    def test_breakable_graph_copies_aux_collector_into_plain_list(self):
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )

        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        source = AuxHiddenStateList([torch.full((4, 3), 7.0)])
        buffers = [torch.zeros(4, 3)]
        backend._copy_output_to_buffer(
            (torch.ones(4, 3), source), (torch.zeros(4, 3), buffers), 2
        )
        torch.testing.assert_close(buffers[0][:2], torch.full((2, 3), 7.0))
        self.assertEqual(buffers[0][2:].count_nonzero(), 0)
        with self.assertRaises(TypeError):
            backend._copy_output_to_buffer(source, tuple(buffers), 2)

    def test_capture_move_owns_gather_but_not_slice(self):
        from sglang.srt.layers.layer_boundary.boundary import _capture_move
        from sglang.srt.layers.layer_boundary.layout import Layout, TokenAxis
        from sglang.test.communicator_patch import patch_communicator

        full = Layout(frozenset())
        sharded = Layout(frozenset({TokenAxis.ATTN_TP}))
        source = torch.ones(2, 3)
        for rows, target, owns in ((sharded, full, True), (full, sharded, False)):
            edge = SimpleNamespace(
                residual_to=rows, produced=SimpleNamespace(layout=target)
            )
            capture_move, allocates = _capture_move(edge)
            self.assertEqual(allocates, owns)
            with (
                patch_communicator(
                    "attn_tp_gather",
                    side_effect=lambda x: torch.cat((x, x)),
                ),
                patch_communicator(
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_size=2, attn_tp_rank=0),
                ),
            ):
                value = capture_move(source, forward_batch=None)
            outputs = AuxHiddenStateList()
            if owns:
                with patch.object(
                    torch.Tensor, "clone", side_effect=AssertionError("extra clone")
                ):
                    outputs.capture(value, owned=allocates)
                self.assertIs(outputs[0], value)
            else:
                outputs.capture(value, owned=allocates)
                self.assertIsNot(outputs[0], value)

    def test_list_snapshots_borrowed_views_and_reused_buffers(self):
        outputs = AuxHiddenStateList()
        buffer = torch.full((4, 3), 2.0)
        outputs.capture(buffer[:2])
        buffer.fill_(5)
        outputs.capture(buffer[:2])
        buffer.zero_()
        torch.testing.assert_close(outputs[0], torch.full((2, 3), 2.0))
        torch.testing.assert_close(outputs[1], torch.full((2, 3), 5.0))

    def test_list_adopts_owned_output_without_copying(self):
        outputs = AuxHiddenStateList()
        value = torch.ones(2, 3)
        with patch.object(
            torch.Tensor, "clone", side_effect=AssertionError("extra clone")
        ):
            outputs.capture(value, owned=True)
        self.assertIs(outputs[0], value)

    def boundary(self, stream, *, move=None):
        boundary = stub_plan()
        boundary.norm = None
        stub_stage(boundary, StageKind.ATTENTION)._prepare_input = Mock(
            return_value=(stream.residual, stream)
        )
        stub_stage(boundary, StageKind.ATTENTION).entry = lambda _: SimpleNamespace(
            input_move=move,
            capture_move=None,
            capture_move_allocates=False,
            capture_preserves_residual=None,
        )
        return stub_stage(boundary, StageKind.ATTENTION)

    def test_packer_receives_borrowed_boundary_value_without_intermediate_clone(self):
        for callback in (False, True):
            with self.subTest(callback=callback):
                value = torch.full((2, 3), 4.0)
                stream = ResidualStream(value)
                stage = self.boundary(stream)
                outputs = AuxHiddenStatePacker(1)
                kwargs = (
                    {"capture": outputs.capture}
                    if callback
                    else {"capture_gathered": outputs}
                )
                with patch.object(
                    torch.Tensor, "clone", side_effect=AssertionError("extra clone")
                ):
                    prepare_attention(stage, value, stream, None, **kwargs)
                value.zero_()
                torch.testing.assert_close(outputs.finalize(), torch.full((2, 3), 4.0))

    def test_gather_storage_is_borrowed_even_if_it_is_a_different_tensor(self):
        source = torch.ones(2, 3)
        gathered = torch.full((4, 3), 7.0)
        stream = ResidualStream(source)
        move = Mock(return_value=gathered)
        stage = self.boundary(stream, move=move)
        outputs = AuxHiddenStateList()
        prepare_attention(stage, source, stream, None, outputs)
        gathered.zero_()
        source.zero_()
        torch.testing.assert_close(outputs[0], torch.full((4, 3), 7.0))
        move.assert_called_once()

    def test_embedding_and_final_norm_capture_write_directly_to_packer(self):
        for residual in (None, torch.ones(2, 3)):
            with self.subTest(residual=residual):
                hidden = torch.full((2, 3), 4.0)
                outputs = AuxHiddenStatePacker(1)
                updated = torch.full((2, 3), 5.0)
                norm = Mock(
                    return_value=hidden if residual is None else (hidden, updated)
                )
                with patch.object(
                    torch.Tensor, "clone", side_effect=AssertionError("extra clone")
                ):
                    final_norm_pair(hidden, residual, norm, outputs.capture)
                hidden.zero_()
                updated.zero_()
                torch.testing.assert_close(
                    outputs.finalize(),
                    torch.full((2, 3), 4.0 if residual is None else 5.0),
                )

    def test_initial_capture_precedes_in_place_prepare_without_intermediate_clone(self):
        hidden = torch.full((2, 3), 4.0)
        stream = ResidualStream()
        outputs = AuxHiddenStatePacker(1)
        stage = self.boundary(stream)

        def prepare(value, stream, batch, **kwargs):
            value.zero_()
            stream.write(value)
            return value, stream

        stage._prepare_input = prepare
        with patch.object(
            torch.Tensor, "clone", side_effect=AssertionError("extra clone")
        ):
            prepare_attention(stage, hidden, stream, None, capture=outputs.capture)
        torch.testing.assert_close(outputs.finalize(), torch.full((2, 3), 4.0))


class TestBoundCaptureOwnership(CustomTestCase):
    def exercise(self, *, enabled, backend_accepts=True, custom=False, callback=False):
        import test_declared_decoder_boundary as fixture

        from sglang.srt.layers import layernorm
        from sglang.srt.layers.layer_boundary import (
            PLAIN_ADD,
            FfnInputFusion,
            SumGroup,
            declare_attn,
            declare_ffn,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.test.boundary_fixtures import build_stages
        from sglang.test.communicator_patch import patch_communicator

        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2)
        norm = layernorm.RMSNorm(4)

        def inplace_norm(value, residual, unused=None):
            residual.add_(value)
            return residual * 2, residual

        norm.forward = inplace_norm

        def fused_kernel(*, input_tensor, residual, **kwargs):
            if not backend_accepts:
                return None, None
            updated = residual + input_tensor * 2
            return updated * 2, updated

        def custom_fused(value, residual, batch):
            return inplace_norm(value * 2, residual)

        custom_fusions = (
            SimpleNamespace(
                ffn_input_fusions=lambda plan: (
                    FfnInputFusion(completes=SumGroup.ATTN_TP, run=custom_fused),
                ),
                attn_input_fusions=lambda plan: (),
            )
            if custom
            else None
        )
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            residual_stream=ResidualStream(torch.full((2, 4), 2.0)),
        )
        hidden = fb.residual_stream.record(torch.full((2, 4), 3.0), PLAIN_ADD)
        outputs = AuxHiddenStateList()
        with (
            fixture.planning(parallel),
            patch_communicator("_use_aiter", False),
            patch_communicator("aiter_ar_fusion_applies", return_value=False),
            patch_communicator("flashinfer_ar_fusion_applies", return_value=enabled),
            patch_communicator(
                "attention_tensor_model_parallel_all_reduce",
                side_effect=lambda x: x * 2,
            ),
            patch.object(layernorm, "_use_aiter", False),
            patch.object(layernorm, "get_parallel", return_value=parallel),
            patch(
                "sglang.srt.distributed.attention_tensor_model_parallel_all_reduce",
                side_effect=lambda x: x * 2,
            ),
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm",
                side_effect=fused_kernel,
            ),
        ):
            attn, ffn = build_stages(
                (declare_attn(), fixture.Norm()),
                (declare_ffn(), norm, {"fusions": custom_fusions}),
                previous=declare_ffn(),
            )
            predicate = attn.entry(fb).capture_preserves_residual
            self.assertEqual(
                predicate is not None and predicate(hidden, fb), enabled and not custom
            )
            kwargs = (
                {"capture": outputs.capture}
                if callback
                else {"capture_gathered": outputs}
            )
            if enabled and not custom:
                # Only capture is forbidden from cloning. The real FlashInfer
                # wrapper's declined-kernel fallback may copy to preserve residual.
                with patch.object(
                    torch.Tensor, "clone", side_effect=AssertionError("extra aux clone")
                ):
                    attn.prepare(hidden, fb, **kwargs)
                self.assertIs(outputs[0], fb.residual_stream.residual)
            else:
                attn.prepare(hidden, fb, **kwargs)
                self.assertIsNot(outputs[0], fb.residual_stream.residual)
            hidden = attn.finish(torch.ones(2, 4), fb)
            result = ffn.prepare(hidden, fb)
            torch.testing.assert_close(result, torch.full((2, 4), 14.0))
            torch.testing.assert_close(outputs[0], torch.full((2, 4), 5.0))
            torch.testing.assert_close(
                fb.residual_stream.residual, torch.full((2, 4), 7.0)
            )

    def test_selected_flashinfer_wrapper_preserves_capture_including_fallback(self):
        for accepts in (False, True):
            for callback in (False, True):
                with self.subTest(accepts=accepts, callback=callback):
                    self.exercise(
                        enabled=True, backend_accepts=accepts, callback=callback
                    )

    def test_in_place_or_custom_read_keeps_capture_owned_by_aux(self):
        for custom in (False, True):
            for callback in (False, True):
                with self.subTest(custom=custom, callback=callback):
                    self.exercise(enabled=custom, custom=custom, callback=callback)


if __name__ == "__main__":
    unittest.main()
