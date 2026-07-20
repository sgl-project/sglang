import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.sparsity.cuda_graph_support import (
    create_cuda_graph_runtime_provider,
    is_runtime_sparse_cuda_graph_available,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBreakableCudaGraphStructuredOutput(unittest.TestCase):
    def setUp(self):
        self.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)

    def test_dataclass_tensor_fields_use_stable_buffers(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(6, dtype=torch.float32).view(2, 3),
            hidden_states=torch.arange(8, dtype=torch.float32).view(2, 4),
            customized_info={"source": ["decode"]},
        )

        self.assertEqual(self.backend._output_rows(output, cap=4), 2)
        buffer = self.backend._alloc_full_buffer(output, size=4)
        self.backend._copy_output_to_buffer(output, buffer, num_tokens=2)
        sliced = self.backend._slice_output(buffer, num_tokens=2)

        torch.testing.assert_close(sliced.next_token_logits, output.next_token_logits)
        torch.testing.assert_close(sliced.hidden_states, output.hidden_states)
        self.assertEqual(sliced.customized_info, {"source": ["decode"]})

    def test_non_tensor_dataclass_fields_must_remain_stable(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.ones((1, 3)),
            hidden_states=None,
            customized_info={"mode": "decode"},
        )
        buffer = self.backend._alloc_full_buffer(output, size=2)
        changed = dataclasses.replace(output, customized_info={"mode": "extend"})

        with self.assertRaisesRegex(ValueError, "customized_info"):
            self.backend._copy_output_to_buffer(changed, buffer, num_tokens=1)


class _RuntimeExtension:
    def __init__(self):
        self.calls = []

    def cuda_graph_capture_variants(self):
        return (256, 1024)

    def select_cuda_graph_variant(self, forward_batch):
        return forward_batch.supported, forward_batch.variant

    def prepare_cuda_graph_capture(self, forward_batch, runtime_variant):
        self.calls.append(("capture", runtime_variant))

    def prepare_cuda_graph_replay(self, forward_batch, runtime_variant):
        self.calls.append(("replay", runtime_variant))

    def prepare_cuda_graph_capture_context(self, context):
        self.calls.append(("context", context))
        return "extended-context"


class TestDecodeCudaGraphRuntimeExtension(unittest.TestCase):
    def _runner(self, extension=None):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.model_runner = SimpleNamespace(cuda_graph_runtime_extension=extension)
        return runner

    def test_absent_extension_preserves_the_default_graph_key(self):
        runner = self._runner()
        forward_batch = SimpleNamespace()

        self.assertEqual(runner._runtime_graph_capture_variants(), (None,))
        self.assertEqual(
            runner._select_runtime_graph_variant(forward_batch), (True, None)
        )
        self.assertIsNone(runner._make_graph_key(8).runtime_variant)

    def test_extension_selects_and_prepares_an_isolated_variant(self):
        extension = _RuntimeExtension()
        runner = self._runner(extension)
        forward_batch = SimpleNamespace(supported=True, variant=1024)

        self.assertEqual(runner._runtime_graph_capture_variants(), (256, 1024))
        supported, variant = runner._select_runtime_graph_variant(forward_batch)
        self.assertTrue(supported)
        self.assertEqual(variant, 1024)
        runner._prepare_runtime_graph_capture(forward_batch, variant)
        runner._prepare_runtime_graph_replay(forward_batch, variant)
        context = SimpleNamespace(attn_backend="backend")
        self.assertEqual(
            runner._prepare_runtime_graph_capture_context(context),
            "extended-context",
        )

        self.assertEqual(
            runner._make_graph_key(8, runtime_variant=256).runtime_variant, 256
        )
        self.assertNotEqual(
            runner._make_graph_key(8, runtime_variant=256),
            runner._make_graph_key(8, runtime_variant=1024),
        )
        self.assertEqual(
            [call[0] for call in extension.calls], ["capture", "replay", "context"]
        )

    def test_ragged_verify_rejects_an_uncaptured_runtime_variant(self):
        extension = _RuntimeExtension()
        runner = self._runner(extension)
        runner.attn_backend = SimpleNamespace(supports_ragged_verify_graph=True)
        runner.capture_num_tokens = [8]
        runner._ragged_capture_slots = lambda _: 2
        runner.require_mlp_sync = False
        runner.is_encoder_decoder = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        forward_batch = SimpleNamespace(
            supported=True,
            variant=2048,
            batch_size=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_info=None,
        )

        self.assertFalse(
            runner._can_run_ragged_verify_graph(
                forward_batch, SimpleNamespace(graph_num_tokens=8)
            )
        )


@dataclasses.dataclass(frozen=True)
class _Context:
    attn_backend: str
    runtime_sparse_coordinator: object = None


class _FixedCapacityAlgorithm:
    supports_fixed_cuda_graph_capacity = True


class TestSparseCudaGraphRuntimeProvider(unittest.TestCase):
    def _provider(self):
        coordinator = SimpleNamespace(
            page_size=16,
            device=torch.device("cuda"),
            req_to_token_pool=SimpleNamespace(max_context_len=40 * 1024),
            config=SimpleNamespace(
                backend="fa3",
                sparse_extra_config={
                    "enable_cuda_graph_retrieval": True,
                    "cuda_graph_context_buckets": [8 * 1024, 32 * 1024],
                },
            ),
            algorithm=_FixedCapacityAlgorithm(),
        )
        return coordinator, create_cuda_graph_runtime_provider(coordinator)

    def test_preflight_requires_a_class_level_fixed_capacity_capability(self):
        coordinator, provider = self._provider()
        self.assertTrue(is_runtime_sparse_cuda_graph_available(coordinator))
        self.assertIsNotNone(provider)

        coordinator.algorithm = SimpleNamespace(supports_fixed_cuda_graph_capacity=True)
        self.assertFalse(is_runtime_sparse_cuda_graph_available(coordinator))
        self.assertIsNone(create_cuda_graph_runtime_provider(coordinator))

    def test_bucket_selection_and_out_of_range_fallback(self):
        _, provider = self._provider()
        self.assertEqual(provider.cuda_graph_capture_variants(), (512, 2048, 2560))

        cases = (
            ([1, 8192], (True, 512)),
            (torch.tensor([8193, 32768]), (True, 2048)),
            ([40 * 1024], (True, 2560)),
            ([40 * 1024 + 1], (False, None)),
        )
        for seq_lens, expected in cases:
            with self.subTest(seq_lens=seq_lens):
                self.assertEqual(
                    provider.select_cuda_graph_variant(
                        SimpleNamespace(seq_lens_cpu=seq_lens)
                    ),
                    expected,
                )

    def test_capture_and_replay_publish_capacity_and_context(self):
        coordinator, provider = self._provider()
        forward_batch = SimpleNamespace()
        provider.prepare_cuda_graph_capture(forward_batch, 512)
        self.assertEqual(forward_batch.runtime_sparse_page_capacity, 512)

        provider.prepare_cuda_graph_replay(forward_batch, 2048)
        self.assertEqual(forward_batch.runtime_sparse_page_capacity, 2048)

        context = provider.prepare_cuda_graph_capture_context(_Context("backend"))
        self.assertIs(context.runtime_sparse_coordinator, coordinator)


if __name__ == "__main__":
    unittest.main()
