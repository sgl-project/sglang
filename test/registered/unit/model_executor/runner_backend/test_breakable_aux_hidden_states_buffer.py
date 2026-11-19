"""Breakable prefill capture keeps packed aux hidden states in the runner's buffer.

A DFlash-family target that packs its captures with
``AuxHiddenStatePacker.for_batch`` writes them straight into the prefill
runner's aux buffer. The breakable backend must hand that buffer back as the
graph output instead of allocating a second ``[tokens, K * hidden]`` leaf and
copying into it. CUDA capture is mocked, so this runs on CPU.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import breakable_cuda_graph_backend as bcg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_CAPTURES = 3
HIDDEN = 4
MAX_TOKENS = 8


def _make_backend(runner):
    backend = bcg.BreakableCudaGraphBackend.__new__(bcg.BreakableCudaGraphBackend)
    backend._cuda_graph_runner = runner
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._debug_eager = False
    backend._hicache_graph_events = None
    backend._shared_output_buffer = None
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=mock.Mock()
    )
    backend._device_module = SimpleNamespace(synchronize=mock.Mock())
    backend._tp_group = SimpleNamespace(barrier=mock.Mock())
    return backend


def _make_runner(*, width=NUM_CAPTURES * HIDDEN, dflash=True):
    # Skip __init__ (it needs a model runner and CUDA); capture reads only
    # these fields.
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        model=SimpleNamespace(get_prefill_aux_hidden_states_width=lambda: width),
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: dflash),
        attn_tp_sequence_sharded=lambda num_tokens: False,
        dtype=torch.float32,
    )
    runner.max_num_tokens = MAX_TOKENS
    runner.device = "cpu"
    runner.pp_size = 1
    runner.enable_cp_bcg_capture = False
    runner.aux_hidden_states_buffer = None
    runner.backend = _make_backend(runner)
    return runner


@contextlib.contextmanager
def _mocked_capture():
    with (
        mock.patch.object(bcg, "BreakableCUDAGraph", return_value=mock.Mock()),
        mock.patch.object(
            bcg,
            "BreakableCUDAGraphCapture",
            side_effect=lambda **kw: contextlib.nullcontext(),
        ),
        mock.patch.object(bcg, "graph_pool_capture_scope", contextlib.nullcontext),
    ):
        yield


def _body(runner, num_tokens):
    forward_batch = SimpleNamespace(
        aux_hidden_states_buffer=runner.aux_hidden_states_buffer[:num_tokens]
    )
    aux = AuxHiddenStatePacker.for_batch(forward_batch, NUM_CAPTURES)
    hidden = torch.full((num_tokens, HIDDEN), float(num_tokens))
    for i in range(NUM_CAPTURES):
        aux.capture(hidden + i)
    return hidden, aux.finalize()


class TestBreakableAuxHiddenStatesBuffer(CustomTestCase):
    def test_packed_aux_output_is_the_runner_buffer(self):
        runner = _make_runner()
        runner.aux_hidden_states_buffer = torch.zeros(MAX_TOKENS, NUM_CAPTURES * HIDDEN)
        sizes = [MAX_TOKENS, MAX_TOKENS - 2]
        with _mocked_capture():
            for size in sizes:
                runner.backend.capture_one(
                    ShapeKey(size=size), lambda size=size: _body(runner, size)
                )

        buffer = runner.aux_hidden_states_buffer
        shared_hidden, shared_aux = runner.backend._shared_output_buffer
        self.assertEqual(shared_aux.data_ptr(), buffer.data_ptr())
        self.assertNotEqual(shared_hidden.data_ptr(), buffer.data_ptr())
        for size in sizes:
            _, aux = runner.backend._outputs[ShapeKey(size=size)]
            self.assertEqual(aux.data_ptr(), buffer.data_ptr())
            self.assertEqual(tuple(aux.shape), (size, NUM_CAPTURES * HIDDEN))

        last = sizes[-1]
        expected = torch.cat(
            [torch.full((last, HIDDEN), float(last + i)) for i in range(NUM_CAPTURES)],
            dim=-1,
        )
        torch.testing.assert_close(buffer[:last], expected)

    def test_output_outside_the_runner_buffer_gets_its_own_leaf(self):
        runner = _make_runner()
        runner.aux_hidden_states_buffer = torch.zeros(MAX_TOKENS, NUM_CAPTURES * HIDDEN)

        def body():
            hidden = torch.ones(MAX_TOKENS, HIDDEN)
            return hidden, torch.cat([hidden] * NUM_CAPTURES, dim=-1)

        with _mocked_capture():
            runner.backend.capture_one(ShapeKey(size=MAX_TOKENS), body)

        _, shared_aux = runner.backend._shared_output_buffer
        self.assertNotEqual(
            shared_aux.data_ptr(), runner.aux_hidden_states_buffer.data_ptr()
        )
        torch.testing.assert_close(
            shared_aux, torch.ones(MAX_TOKENS, NUM_CAPTURES * HIDDEN)
        )

    def test_buffer_only_for_local_rows_of_a_dflash_target(self):
        runner = _make_runner()
        rows = runner._capture_aux_hidden_states_buffer(4, gathered=False)
        self.assertEqual(tuple(rows.shape), (4, NUM_CAPTURES * HIDDEN))
        self.assertEqual(
            tuple(runner.aux_hidden_states_buffer.shape),
            (MAX_TOKENS, NUM_CAPTURES * HIDDEN),
        )
        self.assertIsNone(
            _make_runner()._capture_aux_hidden_states_buffer(4, gathered=True)
        )
        self.assertIsNone(
            _make_runner(dflash=False)._capture_aux_hidden_states_buffer(
                4, gathered=False
            )
        )
        self.assertIsNone(
            _make_runner(width=0)._capture_aux_hidden_states_buffer(4, gathered=False)
        )
        for field, value in (("pp_size", 2), ("enable_cp_bcg_capture", True)):
            runner = _make_runner()
            setattr(runner, field, value)
            self.assertIsNone(
                runner._capture_aux_hidden_states_buffer(4, gathered=False), field
            )
        runner = _make_runner()
        runner.model_runner.attn_tp_sequence_sharded = lambda num_tokens: True
        self.assertIsNone(runner._capture_aux_hidden_states_buffer(4, gathered=False))

    def test_qwen3_5_width_skips_subclasses_with_their_own_forward(self):
        from sglang.srt.models.interns2_mobius import InternS2MobiusForCausalLM
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

        for cls, expected in (
            (Qwen3_5ForCausalLM, NUM_CAPTURES * HIDDEN),
            (InternS2MobiusForCausalLM, 0),
        ):
            model = cls.__new__(cls)
            model.layers_to_capture = list(range(NUM_CAPTURES))
            model.hidden_size = HIDDEN
            self.assertEqual(
                model.get_prefill_aux_hidden_states_width(), expected, cls.__name__
            )

    def test_qwen3_5_packs_into_the_batch_buffer(self):
        from sglang.srt.models import qwen3_5

        class _Layer:
            def __init__(self, capture):
                self._is_layer_to_capture = capture

            def __call__(self, *, hidden_states, captured_last_layer_outputs, **kw):
                hidden_states = hidden_states + 1
                if captured_last_layer_outputs is not None:
                    captured_last_layer_outputs.capture(hidden_states)
                return hidden_states

        model = qwen3_5.Qwen3_5ForCausalLM.__new__(qwen3_5.Qwen3_5ForCausalLM)
        model.flashinfer_mnnvl_cutedsl_fusion = None
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.embed_tokens = lambda input_ids: torch.zeros(len(input_ids), HIDDEN)
        model.layers = [_Layer(i % 2 == 1) for i in range(2 * NUM_CAPTURES)]
        model._start_layer, model._end_layer = 0, len(model.layers)
        model.layers_to_capture = [1, 3, 5]
        model.norm = None
        model.hidden_size = HIDDEN
        input_ids = torch.zeros(MAX_TOKENS, dtype=torch.long)
        expected = torch.cat(
            [torch.full((MAX_TOKENS, HIDDEN), float(i)) for i in (2, 4, 6)], dim=-1
        )
        recorder = SimpleNamespace(
            with_current_layer=lambda layer_idx: contextlib.nullcontext()
        )
        with (
            mock.patch.object(
                qwen3_5,
                "residual_batch",
                SimpleNamespace(
                    start=lambda forward_batch: None,
                    final_norm=lambda hidden, *args, **kw: hidden,
                ),
            ),
            mock.patch.object(
                qwen3_5,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            buffer = torch.empty(MAX_TOKENS, NUM_CAPTURES * HIDDEN)
            _, aux = model.forward(
                input_ids, None, SimpleNamespace(aux_hidden_states_buffer=buffer)
            )
            self.assertIsInstance(aux, torch.Tensor)
            self.assertEqual(aux.data_ptr(), buffer.data_ptr())
            torch.testing.assert_close(aux, expected)

            _, aux = model.forward(
                input_ids, None, SimpleNamespace(aux_hidden_states_buffer=None)
            )
            self.assertIsInstance(aux, list)
            torch.testing.assert_close(torch.cat(aux, dim=-1), expected)


if __name__ == "__main__":
    unittest.main()
