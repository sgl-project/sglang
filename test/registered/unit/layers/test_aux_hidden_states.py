from sglang.srt.layers.communicator import StageKind
from sglang.test.boundary_fixtures import stub_plan, stub_stage

"""Aux owns retained storage, including captures from reusable gather buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList, AuxHiddenStatePacker
from sglang.srt.layers.communicator.residual.access import norm_output
from sglang.srt.layers.communicator.residual.stream import ResidualStream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        from sglang.srt.layers.communicator.boundary import Boundary
        from sglang.srt.layers.communicator.layout import Layout, TokenAxis
        from sglang.test.communicator_patch import patch_communicator

        full = Layout(frozenset())
        sharded = Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
        source = torch.ones(2, 3)
        for rows, target, owns in ((sharded, full, True), (full, sharded, False)):
            edge = SimpleNamespace(
                residual_to=rows, produced=SimpleNamespace(layout=target)
            )
            boundary = Boundary(edge=edge, prepare=None)
            self.assertEqual(boundary.capture_move_allocates, owns)
            with (
                patch_communicator(
                    "_redistribute_from_attn_tp_shards",
                    side_effect=lambda x: torch.cat((x, x)),
                ),
                patch_communicator(
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_size=2, attn_tp_rank=0),
                ),
            ):
                value = boundary.capture_move(source, forward_batch=None)
            outputs = AuxHiddenStateList()
            if owns:
                with patch.object(
                    torch.Tensor, "clone", side_effect=AssertionError("extra clone")
                ):
                    outputs.capture(value, owned=boundary.capture_move_allocates)
                self.assertIs(outputs[0], value)
            else:
                outputs.capture(value, owned=boundary.capture_move_allocates)
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
                    {"capture_output": outputs.capture}
                    if callback
                    else {"captured_last_layer_outputs": outputs}
                )
                with patch.object(
                    torch.Tensor, "clone", side_effect=AssertionError("extra clone")
                ):
                    stage._prepare_attention(value, stream, None, **kwargs)
                value.zero_()
                torch.testing.assert_close(outputs.finalize(), torch.full((2, 3), 4.0))

    def test_gather_storage_is_borrowed_even_if_it_is_a_different_tensor(self):
        source = torch.ones(2, 3)
        gathered = torch.full((4, 3), 7.0)
        stream = ResidualStream(source)
        move = Mock(return_value=gathered)
        stage = self.boundary(stream, move=move)
        outputs = AuxHiddenStateList()
        stage._prepare_attention(source, stream, None, outputs)
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
                    norm_output(hidden, residual, norm, outputs.capture)
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
            stage._prepare_attention(
                hidden, stream, None, capture_output=outputs.capture
            )
        torch.testing.assert_close(outputs.finalize(), torch.full((2, 3), 4.0))


if __name__ == "__main__":
    unittest.main()
