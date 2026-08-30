"""CPU coverage for chunked-prefix Full prefill CUDA-graph state."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.model_runner_components.cuda_graph_setup as graph_setup
import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_prefill_graph,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeAttentionBackend:
    supports_full_cuda_graph_chunked_prefix = True

    def __init__(self):
        self.calls = []

    def prepare_full_cuda_graph_chunked_prefix(self, forward_batch, *, in_capture):
        self.calls.append((forward_batch, in_capture))


class _FakeKVIndexKernel:
    def __getitem__(self, grid):
        del grid

        def run(
            req_to_token,
            req_pool_indices,
            starts,
            seq_lens,
            cu_seq_lens,
            output,
            req_to_token_stride,
        ):
            del cu_seq_lens, req_to_token_stride
            cursor = 0
            for row in range(seq_lens.numel()):
                seq_len = int(seq_lens[row])
                start = int(starts[row])
                req = int(req_pool_indices[row])
                output[cursor : cursor + seq_len].copy_(
                    req_to_token[req, start : start + seq_len]
                )
                cursor += seq_len

        return run


class _FakeGraphSlot:
    def __init__(self, buffer):
        self.buffer = buffer

    def slice_for(self, _batch_size, num_tokens):
        return self.buffer[:num_tokens]


class _FakeBatchRegistry:
    def __init__(self):
        self.slots = {
            "input_ids": _FakeGraphSlot(torch.arange(4, dtype=torch.int64)),
            "positions": _FakeGraphSlot(torch.arange(4, dtype=torch.int64)),
            "out_cache_loc": _FakeGraphSlot(torch.arange(4, dtype=torch.int64)),
        }

    def fill_from(self, *_args, **_kwargs):
        return None

    def has_slot(self, name):
        return name in self.slots

    def get_slot(self, name):
        return self.slots[name]


class TestPrefillCudaGraphRunnerChunkedPrefix(CustomTestCase):
    def test_low_free_memory_still_captures_prefill_graph(self):
        eager_runner = object()
        prefill_runner = object()
        # The capture decision reads the graph configuration and the LoRA flag
        # out of the bags.
        override = get_context().override_server_args(
            enable_lora=False,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(bs=[1], backend=Backend.BREAKABLE)
            ),
        )
        override.install()
        self.addCleanup(override.restore)
        model_runner = SimpleNamespace(
            device="cuda",
            gpu_id=0,
            is_draft_worker=False,
            # A real ModelRunner always has this attribute; the prefill gate
            # reads it rather than the process-wide LoRA config.
            lora_manager=None,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
            server_args=SimpleNamespace(),
            model=SimpleNamespace(),
            model_config=SimpleNamespace(context_len=8192, num_hidden_layers=1),
            layer_info=SimpleNamespace(start_layer=0, end_layer=1),
            req_to_token_pool=SimpleNamespace(size=1),
        )
        language_model = SimpleNamespace(layers=[object()])

        with (
            patch.object(graph_setup, "check_cuda_graph_backend", return_value=False),
            patch.object(
                graph_setup, "resolve_language_model", return_value=language_model
            ),
            patch.object(
                graph_setup,
                "compute_attention_and_moe_layers",
                return_value=([object()], [], [], [], [None]),
            ),
            patch.object(
                graph_setup,
                "get_available_gpu_memory",
                side_effect=[3.99, 3.5],
            ),
            patch.object(
                graph_setup,
                "PrefillCudaGraphRunner",
                return_value=prefill_runner,
            ),
        ):
            capture = capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=eager_runner,
            )

        self.assertIs(capture.runner, prefill_runner)

    def test_eagle_target_tc_piecewise_skips_last_mode_capture(self):
        eager_runner = object()
        # The server-side hidden-state ceiling is a bag leaf.
        override = get_context().override_server_args(
            enable_return_hidden_states=True,
            return_hidden_states_mode="last",
        )
        override.install()
        self.addCleanup(override.restore)
        model_runner = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            server_args=SimpleNamespace(),
        )

        with patch.object(
            graph_setup,
            "check_cuda_graph_backend",
            return_value=False,
        ):
            capture = capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=eager_runner,
            )

        self.assertIs(capture.runner, eager_runner)

    def test_pp_proxy_output_is_trimmed_to_raw_prefill_tokens(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.raw_num_tokens = 3
        output = PPProxyTensors(
            {
                "hidden_states": torch.arange(32).view(8, 4),
                "residual": torch.arange(32, 64).view(8, 4),
            }
        )

        trimmed = runner._finalize_execute_output(output)

        self.assertIsInstance(trimmed, PPProxyTensors)
        self.assertEqual(tuple(trimmed["hidden_states"].shape), (3, 4))
        self.assertEqual(tuple(trimmed["residual"].shape), (3, 4))

    def test_static_batch_preserves_consumed_multimodal_embeddings(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.capture_num_tokens = [4]
        runner.buffer_registry = _FakeBatchRegistry()
        runner.enable_cp_v2_bcg_capture = False
        runner._is_full_backend = False
        runner.backend = SimpleNamespace()
        runner.has_mha_companion_layers = False
        runner._prefill_static_buffers = None
        runner.static_draft_hidden_states = None
        runner.capture_return_pooled_hidden_states = False
        runner._next_token_logits_buffer = lambda _rows: None
        runner._prefill_logits_buffer_rows = lambda _batch: 1
        runner._prepare_forward_metadata_for_replay = lambda *_args: None

        mm_input_embeds = torch.randn(3, 8)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.arange(3, dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            out_cache_loc=torch.arange(3, dtype=torch.int64),
            seq_lens_sum=3,
            positions=torch.arange(3, dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            extend_seq_lens=torch.tensor([3], dtype=torch.int32),
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
            extend_start_loc=torch.zeros(1, dtype=torch.int32),
            extend_seq_lens_cpu=[3],
            extend_prefix_lens_cpu=[0],
            mm_inputs=None,
            mm_input_embeds=mm_input_embeds,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_forward_mode=ForwardMode.EXTEND,
        )

        static_batch = runner.load_batch(forward_batch)

        self.assertIs(static_batch.mm_input_embeds, mm_input_embeds)

    def test_prefix_chunk_capacity_is_aggregate_and_can_be_overridden(self):
        graph_config = SimpleNamespace(
            prefill=SimpleNamespace(full_prefill_prefix_chunk_tokens=None, max_bs=8)
        )
        # Both leaves come from the bags; the published object is this one, so
        # the cases below still drive them by mutating it.
        override = get_context().override_server_args(
            chunked_prefill_size=16, cuda_graph_config=graph_config
        )
        published = override.install()
        self.addCleanup(override.restore)
        model_runner = SimpleNamespace(
            server_args=SimpleNamespace(),
            # Wider than the token table, so the table is the binding limit.
            model_config=SimpleNamespace(context_len=4096),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((1, 32), dtype=torch.int32)
            ),
        )

        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (4, 16),
        )

        get_context().override("test", chunked_prefill_size=-1)
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (2, 8),
        )
        get_context().override("test", chunked_prefill_size=16)

        graph_config.prefill.full_prefill_prefix_chunk_tokens = 24
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (6, 24),
        )

        graph_config.prefill.full_prefill_prefix_chunk_tokens = 256
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (32, 128),
        )

        # At least one token is reserved per request lane even if the requested
        # aggregate capacity is smaller than the fixed request-slot count.
        graph_config.prefill.full_prefill_prefix_chunk_tokens = 2
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (1, 4),
        )

        # A context shorter than the token table binds instead: a draft runner
        # capped at the target's context, or a short --context-length.
        model_runner.model_config.context_len = 8
        graph_config.prefill.full_prefill_prefix_chunk_tokens = 256
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (8, 32),
        )

        graph_config.prefill.full_prefill_prefix_chunk_tokens = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4)

    def test_buffers_are_shared_across_token_buckets(self):
        backend = _FakeAttentionBackend()
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 3
        runner._prefix_chunk_len = 2
        runner._prefix_chunk_capacity = 6
        runner._prefix_max_len = 8
        runner._prefix_capture_variants = (1, 2, 4)
        runner.device = torch.device("cpu")
        runner._prefill_static_buffers = {
            "extend_prefix_lens": torch.zeros(3, dtype=torch.int64),
            "req_pool_indices": torch.tensor([2, 0, 1], dtype=torch.int64),
        }
        runner._prefix_capture_batches = {}
        runner._prefix_capture_buffers = runner._create_chunked_prefix_buffers()
        runner.model_runner = SimpleNamespace(
            attn_backend=backend,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(24, dtype=torch.int32).view(3, 8)
            ),
        )

        first = SimpleNamespace()
        second = SimpleNamespace()
        first_key = ShapeKey(size=8, variant_label="chunked_prefix:4")
        second_key = ShapeKey(size=16, variant_label="chunked_prefix:4")

        with patch.object(
            runner_module,
            "create_chunked_prefix_cache_kv_indices",
            _FakeKVIndexKernel(),
        ):
            runner._prepare_chunked_prefix_capture(first, first_key, 4)
            runner._prepare_chunked_prefix_capture(second, second_key, 4)

            buffers = runner._prefix_capture_buffers
            self.assertIsNotNone(buffers)
            # Chunk starts are constant and prefilled at allocation.
            self.assertEqual(
                buffers.starts_cpu.tolist(),
                [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]],
            )
            self.assertEqual(first.extend_prefix_lens_cpu, [8, 8, 8])
            self.assertEqual(first.prefix_chunk_num_tokens, [6, 6, 6, 6])
            self.assertIs(first.prefix_chunk_starts, buffers.starts)
            self.assertIs(first.prefix_chunk_seq_lens, buffers.seq_lens)
            self.assertIs(first.prefix_chunk_cu_seq_lens, buffers.cu_seq_lens)
            self.assertIs(first.prefix_chunk_starts, second.prefix_chunk_starts)
            self.assertIs(first.prefix_chunk_seq_lens, second.prefix_chunk_seq_lens)
            self.assertIs(
                first.prefix_chunk_cu_seq_lens,
                second.prefix_chunk_cu_seq_lens,
            )
            # Per-chunk KV indices are views of one shared 2-D buffer; what
            # capture bakes into the graph is the address, so compare pointers.
            for kv_chunk_idx in (0, 3):
                self.assertEqual(
                    first.prefix_chunk_kv_indices[kv_chunk_idx].data_ptr(),
                    buffers.kv_indices[kv_chunk_idx].data_ptr(),
                )
                self.assertEqual(
                    first.prefix_chunk_kv_indices[kv_chunk_idx].data_ptr(),
                    second.prefix_chunk_kv_indices[kv_chunk_idx].data_ptr(),
                )

            runner._prepare_chunked_prefix_replay(
                second_key,
                SimpleNamespace(batch_size=2, extend_prefix_lens_cpu=[5, 1]),
            )

        self.assertEqual(
            second.prefix_chunk_seq_lens.tolist(),
            [[2, 1, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[0].tolist(),
            [16, 17, 0, 0, 0, 0],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[1].tolist(),
            [18, 19, 0, 0, 0, 0],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[2].tolist(),
            [20, 0, 0, 0, 0, 0],
        )
        self.assertEqual(second.prefix_chunk_kv_indices[3].tolist(), [0] * 6)
        self.assertEqual(
            backend.calls,
            [(first, True), (second, True), (second, False)],
        )

    def test_prefix_gate_only_applies_to_chunked_prefix_variant(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 4
        runner.enable_lora = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.max_num_tokens = 32
        runner.capture_num_tokens = [4]
        runner.backend = SimpleNamespace()
        runner.prefill_backend_name = Backend.FULL
        runner.has_mha_companion_layers = False
        runner._prefix_chunk_len = 2
        runner._prefix_capture_variants = (1, 2, 4)

        forward_batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.zeros(4, dtype=torch.int64),
            input_embeds=None,
            replace_embeds=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            extend_prefix_lens_cpu=[8],
        )

        # Prefix hits in BCG/TC-piecewise and ordinary non-MLA FullCG use the
        # normal graph topology and must retain their existing eligibility.
        runner._capture_chunked_prefix = False
        for is_full_backend in (False, True):
            with self.subTest(is_full_backend=is_full_backend):
                runner._is_full_backend = is_full_backend
                self.assertTrue(runner.can_run_graph(forward_batch))

        # The dedicated chunked-prefix topology has a fixed captured capacity.
        runner._is_full_backend = True
        runner._capture_chunked_prefix = True
        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(
            runner._shape_key(4, forward_batch).variant_label,
            "chunked_prefix:4",
        )
        forward_batch.batch_size = 2
        # Capacity is per request, not a sum: three real chunks round up to the
        # four-chunk graph even though the aggregate prefix has eight tokens.
        forward_batch.extend_prefix_lens_cpu = [5, 3]
        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(
            runner._shape_key(4, forward_batch).variant_label,
            "chunked_prefix:4",
        )
        forward_batch.extend_prefix_lens_cpu = [9, 1]
        self.assertFalse(runner.can_run_graph(forward_batch))


if __name__ == "__main__":
    unittest.main()
