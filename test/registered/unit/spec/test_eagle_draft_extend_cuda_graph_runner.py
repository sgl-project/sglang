"""CPU contracts for the common draft-extend runner and device backends."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as npu
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend import utils as backend_utils
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEagleDraftExtendCudaGraphRunner(unittest.TestCase):
    def _runner(self, *, npu_backend=False, steps=2, width=6):
        attn_backend = SimpleNamespace(
            init_cuda_graph_state=mock.Mock(),
            get_cuda_graph_seq_len_fill_value=lambda: width,
            init_forward_metadata_out_graph=mock.Mock(),
            shared_read_ends=lambda mode: SharedReadEnds.POST_REPLAY,
        )
        model_runner = SimpleNamespace(
            device="cpu",
            model=SimpleNamespace(),
            device_timer=None,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                vocab_size=8,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                model_is_mrope=False,
            ),
            graph_shared_output=SimpleNamespace(
                get_logits_buffer=lambda vocab, rows: torch.zeros(rows, vocab)
            ),
        )
        worker = SimpleNamespace(
            draft_runner=model_runner,
            megakernel=None,
            draft_extend_attn_backend=attn_backend,
            speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
            seed_dsa_topk_from_draft_extend=False,
        )
        device_module = SimpleNamespace(
            current_device=lambda: 0, set_device=mock.Mock(), Event=mock.Mock()
        )
        runtime = SimpleNamespace(
            graph=SimpleNamespace(
                disable_cuda_graph_padding=False,
                enable_profile_cuda_graph=False,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend=Backend.FULL)
                ),
            ),
            mamba=SimpleNamespace(enable_mamba_extra_buffer=False),
            features=SimpleNamespace(enable_memory_saver=False),
        )

        def resolve_backend(runner):
            # This must run before allocation; no constructor may require buffers.
            self.assertFalse(hasattr(runner, "buffers"))
            self.assertEqual(runner.captured_req_width, width)
            if npu_backend:
                # Exercise the real factory with its unchanged signature while
                # keeping all allocations on CPU and device methods mocked.
                with (
                    mock.patch.object(model_runner, "device", "npu"),
                    mock.patch.object(backend_utils, "get_exec", return_value=runtime),
                ):
                    backend = backend_utils.resolve_decode_backend(runner)
                self.addCleanup(backend.cleanup)
                self.assertIs(backend._decode_runner, runner)
                return backend
            # No graph capture runs in this fixture; exercise the inherited
            # dtype property without creating a CUDA graph.
            return FullCudaGraphBackend.__new__(FullCudaGraphBackend)

        with contextlib.ExitStack() as stack:
            replacements = {
                "megakernel_graph_variants": None,
                "get_exec": runtime,
                "get_parallel": SimpleNamespace(pp_size=1),
                "deployment_attn_dp_size": 1,
                "get_flags": SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=False)
                ),
                "get_spec": SimpleNamespace(speculative_num_steps=steps),
                "get_batch_sizes_to_capture": ([1, 4], []),
                "resolve_num_tokens_per_req": width,
                "get_draft_input_from_target_hidden_dim": 3,
                "require_gathered_buffer": False,
                "require_mlp_tp_gather": False,
                "require_mlp_sync": False,
                "require_attn_tp_gather": False,
            }
            for name, result in replacements.items():
                stack.enter_context(mock.patch.object(mod, name, return_value=result))
            stack.enter_context(
                mock.patch.object(
                    mod, "resolve_decode_backend", side_effect=resolve_backend
                )
            )
            stack.enter_context(
                mock.patch.object(
                    mod.torch, "get_device_module", return_value=device_module
                )
            )
            stack.enter_context(
                mock.patch.object(mod, "model_capture_mode", contextlib.nullcontext)
            )
            stack.enter_context(mock.patch.object(mod, "DeepEPCudaGraphRunnerAdapter"))
            stack.enter_context(
                mock.patch.object(mod.EagleDraftExtendInputBuffers, "share_buffers")
            )
            stack.enter_context(
                mock.patch.object(mod.EAGLEDraftExtendCudaGraphRunner, "capture")
            )
            stack.enter_context(
                mock.patch.object(
                    npu, "get_parallel", return_value=SimpleNamespace(tp_group=object())
                )
            )
            stack.enter_context(
                mock.patch.object(
                    npu.TorchMemorySaverAdapter, "create", return_value=None
                )
            )
            runner = mod.EAGLEDraftExtendCudaGraphRunner(worker)
        runner.model_context = contextlib.nullcontext
        return runner

    def test_backend_dtype_is_available_before_buffer_allocation(self):
        # D=6 deliberately differs from S+1, including the S=1 case.
        for npu_backend, dtype in ((False, torch.int64), (True, torch.int32)):
            for steps in (1, 2):
                with self.subTest(npu=npu_backend, steps=steps):
                    runner = self._runner(npu_backend=npu_backend, steps=steps)
                    if npu_backend:
                        self.assertIs(runner.backend._decode_runner, runner)
                    self.assertEqual(runner.buffers.out_cache_loc.dtype, dtype)
                    self.assertEqual(runner.buffers.out_cache_loc.numel(), 4 * 6)
                    self.assertEqual(
                        runner.buffers.select_index.tolist(), [5, 11, 17, 23]
                    )

    def test_execute_refreshes_metadata_before_host_update_and_publishes_event(self):
        runner = self._runner(npu_backend=True)
        order = []
        metadata = []

        def record_metadata(fb):
            metadata.append(fb.seq_lens_cpu.clone())
            order.append("metadata")

        runner.draft_extend_attn_backend.init_forward_metadata_out_graph.side_effect = (
            record_metadata
        )
        runner._publish_read_done = lambda **kwargs: order.append("read_done")
        key = runner._make_graph_key(4)
        graph = mock.Mock()
        graph.update.side_effect = lambda **kwargs: order.append("update")
        graph.replay.side_effect = lambda: order.append("replay")
        captured_hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        runner.backend._graphs[key] = graph
        runner.backend._outputs[key] = LogitsProcessorOutput(
            next_token_logits=torch.zeros(4, 8), hidden_states=captured_hidden
        )
        fb = SimpleNamespace(
            batch_size=2,
            input_ids=torch.arange(12),
            seq_lens=torch.tensor([17, 23]),
            seq_lens_cpu=torch.tensor([17, 23]),
            seq_lens_sum=40,
            out_cache_loc=torch.arange(12),
            positions=torch.arange(12),
            req_pool_indices=torch.tensor([1, 2]),
            mamba_track_indices=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            out_cache_loc_virtual=None,
            kv_loc_plan=None,
            kv_loc_cols=None,
            spec_info=SimpleNamespace(
                num_tokens_per_req=6,
                num_correct_drafts=None,
                hidden_states=torch.ones(12, 3),
            ),
        )

        output = runner.execute(fb, select_index=torch.tensor([5, 11]))

        self.assertEqual(metadata[0].tolist(), [17, 23, 6, 6])
        graph.update.assert_called_once_with(
            cpu_update_input=[{"actual_seq_lengths_kv": [17, 23, 0, 0]}]
        )
        self.assertEqual(order[0], "metadata")
        self.assertEqual(order[-1], "read_done")
        self.assertEqual(runner.buffers.select_index[:2].tolist(), [5, 11])
        self.assertEqual(output.next_token_logits.shape, (2, 8))
        self.assertTrue(torch.equal(output.hidden_states, captured_hidden[:2]))
        self.assertNotEqual(output.hidden_states.data_ptr(), captured_hidden.data_ptr())

    def test_common_admission_keeps_padding_and_width_checks(self):
        runner = self._runner()
        fb = SimpleNamespace(
            seq_lens=torch.ones(2), spec_info=SimpleNamespace(num_tokens_per_req=6)
        )
        self.assertTrue(runner.can_run_graph(fb))
        fb.spec_info.num_tokens_per_req = 3
        self.assertFalse(runner.can_run_graph(fb))
        fb.spec_info.num_tokens_per_req = 6
        runner.disable_padding = True
        runner.backend.can_run = mock.Mock(return_value=False)
        self.assertFalse(runner.can_run_graph(fb))
        runner.backend.can_run.assert_called_once_with(fb, runner._make_graph_key(2))


if __name__ == "__main__":
    unittest.main()
