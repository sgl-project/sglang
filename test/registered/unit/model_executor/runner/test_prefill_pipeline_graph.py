"""Pipeline activations must retain token geometry and owned replay inputs."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
import torch
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_prefill_registry
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillPipelineGraph(CustomTestCase):
    @staticmethod
    def buffers(**kwargs):
        return PrefillInputBuffers.create(
            device="cpu",
            max_bs=2,
            max_num_tokens=8,
            cache_loc_dtype=torch.int64,
            is_multimodal=False,
            hidden_size=6,
            dtype=torch.float32,
            enable_mamba_track=False,
            **kwargs,
        )

    def test_token_axis_geometry_and_optional_fields(self):
        self.assertIsNone(self.buffers().pp_proxy_tensors)
        ordinary = self.buffers(pp_size=2).pp_proxy_tensors
        self.assertEqual(ordinary["hidden_states"].shape, (8, 6))
        self.assertEqual(ordinary["residual"].shape, (8, 6))
        blocked = self.buffers(
            pp_size=2, pp_proxy_residual_num_blocks=3, pp_proxy_topk_size=4
        ).pp_proxy_tensors
        self.assertEqual(blocked["residual"].shape, (8, 3, 6))
        self.assertEqual(blocked["topk_indices"].shape, (8, 4))
        self.assertEqual(blocked["topk_indices"].dtype, torch.int32)
        mhc = self.buffers(pp_size=2, hc_hidden_size=24).pp_proxy_tensors
        self.assertEqual(set(mhc), {"hidden_states"})
        self.assertEqual(mhc["hidden_states"].shape, (8, 24))

    def test_registry_owns_input_and_clears_reused_padding(self):
        buffers = self.buffers(pp_size=2)
        registry = build_prefill_registry(
            device="cpu",
            max_bs=2,
            max_num_token=8,
            cache_loc_dtype=torch.int64,
            source=buffers,
            share_pool=False,
        )
        for count in (8, 5, 7, 3):
            source = {
                name: torch.full((count, 6), float(count))
                for name in ("hidden_states", "residual")
            }
            registry.fill_from(
                SimpleNamespace(),
                raw_bs=2,
                padded_bs=2,
                raw_num_tokens=count,
                padded_num_tokens=8,
                pp_proxy_tensors=PPProxyTensors(source),
            )
            for name, value in source.items():
                backing = buffers.pp_proxy_tensors[name]
                self.assertEqual(
                    registry.get_slot(f"pp_proxy_tensors.{name}").buffer.data_ptr(),
                    backing.data_ptr(),
                )
                value.fill_(-1)
                torch.testing.assert_close(
                    backing[:count], torch.full((count, 6), float(count))
                )
                self.assertEqual(torch.count_nonzero(backing[count:]).item(), 0)

    def test_capture_passes_owned_activations_to_body_and_outer_model(self):
        for backend in (Backend.FULL, Backend.BREAKABLE):
            with self.subTest(backend=backend):
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                runner.pp_size = 2
                runner.buffers = self.buffers(pp_size=2)
                for value in runner.buffers.pp_proxy_tensors.values():
                    value.fill_(3)
                runner.prefill_backend_name = backend
                runner._prefill_forward_context = lambda _: nullcontext()
                runner._get_layer_model_positions = lambda batch: batch.positions

                def forward(
                    ids, positions, batch, input_embeds=None, *, pp_proxy_tensors=None
                ):
                    self.assertIsNotNone(pp_proxy_tensors)
                    self.assertEqual(pp_proxy_tensors["hidden_states"].shape, (5, 6))
                    pp_proxy_tensors["residual"].add_(7)
                    return pp_proxy_tensors

                runner.layer_model = SimpleNamespace(forward=forward)
                runner.model_runner = SimpleNamespace(model=runner.layer_model)
                batch = SimpleNamespace(
                    global_dp_buffer_len=None,
                    dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
                    global_num_tokens_cpu=None,
                    input_ids=torch.arange(5),
                    positions=torch.arange(5),
                    input_embeds=None,
                )
                with (
                    patch.object(runner_module, "set_dp_buffer_len"),
                    patch.object(runner_module, "set_is_extend_in_batch"),
                ):
                    first = runner._run_forward(batch, 5)
                    second = runner._run_forward(batch, 5)
                torch.testing.assert_close(first["residual"], torch.full((5, 6), 10.0))
                torch.testing.assert_close(second["residual"], first["residual"])
                torch.testing.assert_close(
                    runner.buffers.pp_proxy_tensors["residual"], torch.full((8, 6), 3.0)
                )

    def test_piecewise_replay_preserves_captured_input_addresses(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.pp_size = 2
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.buffers = self.buffers(pp_size=2)
        runner._prefill_forward_context = lambda *args, **kwargs: nullcontext()
        captured = None

        def replay(shape, batch, *, pp_proxy_tensors):
            nonlocal captured
            self.assertEqual(shape.size, 8)
            for name, value in pp_proxy_tensors.tensors.items():
                self.assertEqual(value.shape, (8, 6))
                self.assertEqual(
                    value.data_ptr(), runner.buffers.pp_proxy_tensors[name].data_ptr()
                )
            if captured is None:
                captured = pp_proxy_tensors
            # CUDA graphs read capture-time pointers, not the new Python args.
            captured["residual"].add_(7)
            return captured

        runner.backend = SimpleNamespace(replay=replay)
        incoming = PPProxyTensors({"residual": torch.full((5, 6), -1.0)})
        for value in (3.0, 5.0):
            for buffer in runner.buffers.pp_proxy_tensors.values():
                buffer.fill_(value)
            output = runner._execute_tc_piecewise(
                SimpleNamespace(), 8, 5, pp_proxy_tensors=incoming
            )
            torch.testing.assert_close(
                output["residual"], torch.full((8, 6), value + 7)
            )
        torch.testing.assert_close(incoming["residual"], torch.full((5, 6), -1.0))

    def test_pipeline_output_trims_every_token_axis(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 5
        output = PPProxyTensors(
            {
                "hidden_states": torch.arange(48).reshape(8, 6),
                "residual": torch.arange(144).reshape(8, 3, 6),
                "topk_indices": torch.arange(32).reshape(8, 4),
            }
        )
        trimmed = runner._finalize_execute_output(output)
        for name, value in output.tensors.items():
            torch.testing.assert_close(trimmed[name], value[:5])
            self.assertEqual(trimmed[name].data_ptr(), value.data_ptr())

    def test_missing_or_misaligned_stage_inputs_fail_before_replay(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.pp_size = 2
        runner.buffers = self.buffers(pp_size=2)
        runner.model_runner = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False)
        )
        batch = SimpleNamespace(input_ids=torch.arange(5))
        with self.assertRaisesRegex(ValueError, "preceding-stage activations"):
            runner.load_batch(batch)
        for name in ("hidden_states", "residual"):
            with self.subTest(name=name):
                values = {
                    key: torch.zeros(5, 6) for key in ("hidden_states", "residual")
                }
                values[name] = torch.zeros(4, 6)
                with self.assertRaisesRegex(ValueError, "wrong shape"):
                    runner.load_batch(batch, pp_proxy_tensors=PPProxyTensors(values))


if __name__ == "__main__":
    unittest.main()
