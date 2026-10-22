"""Policy graph captures must retain the attention metadata already captured."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor import input_buffers
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.speculative import eagle_draft_cuda_graph_runner, eagle_worker_v2
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.watermarking.synthid import SynthIDConfig, SynthIDWatermarkProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AttentionBackend:
    def __init__(self):
        self.attn_backends = [self]
        self.page_table: torch.Tensor | None = None

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        self.page_table = torch.zeros(max_bs, 4, dtype=torch.int32)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


def _stub_capture_hardware(*, patches, backend):
    captured: dict[str, torch.Tensor] = {}

    def record_capture(runner):
        assert backend.page_table is not None
        kind = (
            "off"
            if runner.synthid_watermark is None
            else "on" if runner.sparse_synthid_draft_probs else "mixed"
        )
        captured[kind] = backend.page_table

    patches.enter_context(
        patch.dict(input_buffers._forward_input_buffer_pool, clear=True)
    )
    for name in ("_is_cpu", "_is_cuda", "_is_hip", "_is_musa", "_is_npu", "_is_xpu"):
        patches.enter_context(patch.object(eagle_worker_v2, name, False))
    patches.enter_context(
        patch.object(eagle_worker_v2, "get_available_gpu_memory", return_value=100)
    )
    for module in (eagle_worker_v2, eagle_draft_cuda_graph_runner):
        patches.enter_context(
            patch.object(
                module, "get_batch_sizes_to_capture", return_value=([1, 2], [])
            )
        )
    for name in (
        "require_gathered_buffer",
        "require_mlp_tp_gather",
        "require_mlp_sync",
        "require_attn_tp_gather",
    ):
        patches.enter_context(
            patch.object(eagle_draft_cuda_graph_runner, name, return_value=False)
        )
    patches.enter_context(
        patch.object(
            eagle_draft_cuda_graph_runner,
            "get_draft_recurrent_hidden_state_spec",
            return_value=(2, torch.float32),
        )
    )
    patches.enter_context(
        patch.object(
            eagle_draft_cuda_graph_runner,
            "DeepEPCudaGraphRunnerAdapter",
            return_value=Mock(),
        )
    )
    patches.enter_context(
        patch.object(
            eagle_draft_cuda_graph_runner,
            "resolve_decode_backend",
            return_value=Mock(),
        )
    )
    patches.enter_context(
        patch.object(
            eagle_draft_cuda_graph_runner,
            "model_capture_mode",
            side_effect=nullcontext,
        )
    )
    patches.enter_context(
        patch.object(
            EAGLEDraftCudaGraphRunner,
            "capture",
            record_capture,
        )
    )
    patches.enter_context(
        patch.object(torch.cuda, "get_rng_state_all", return_value=[])
    )
    patches.enter_context(patch.object(torch.cuda, "set_rng_state_all"))
    patches.enter_context(
        patch.object(
            SynthIDWatermarkProcessor,
            "capture_mixed_target_graphs",
            return_value=None,
            create=True,
        )
    )
    return captured


def _build_worker(*, backend, processor):
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.device = "cpu"
    worker.gpu_id = 0
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4
    worker.seed_dsa_topk_from_draft_extend = False
    worker.draft_attn_backend = backend
    worker.draft_extend_attn_backend = None
    worker._specialized_graph_memory_usage = {}
    worker._specialized_graph_time_usage = {}
    worker.target_worker = SimpleNamespace(
        device="cuda",
        model_runner=SimpleNamespace(
            sampler=SimpleNamespace(synthid_watermark=processor),
            model_config=SimpleNamespace(vocab_size=8),
        ),
    )
    worker.draft_runner = SimpleNamespace(
        device="cpu",
        ps=SimpleNamespace(tp_size=1, attn_dp_size=1),
        server_args=get_server_args(),
        model_config=SimpleNamespace(vocab_size=8),
        draft_attn_backend=backend,
    )
    return worker


class TestEagleWatermarkGraphState(CustomTestCase):
    def test_plain_capture_does_not_replace_marked_graph_attention_storage(self):
        """A second capture must not leave the first graph reading stale KV pages."""
        config = SynthIDConfig(keys=(1, 2), top_k=3, context_history_size=4)
        processor = SynthIDWatermarkProcessor(config, "cpu", num_slots=2)
        graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        )

        for enabled, enforce in ((True, False), (True, True), (False, False)):
            with self.subTest(enabled=enabled, enforce=enforce):
                backend = _AttentionBackend()
                with (
                    get_context().override_server_args(
                        speculative_algorithm="EAGLE",
                        speculative_eagle_topk=1,
                        speculative_num_steps=3,
                        speculative_num_draft_tokens=4,
                        speculative_use_rejection_sampling=True,
                        cuda_graph_config=graph_config,
                        enable_watermark=enabled,
                        watermark_algorithm="synthid" if enabled else None,
                        watermark_enforce_all=enforce,
                    ),
                    ExitStack() as patches,
                ):
                    worker = _build_worker(
                        backend=backend, processor=processor if enabled else None
                    )
                    captured = _stub_capture_hardware(patches=patches, backend=backend)
                    worker._capture_cuda_graphs()

                expected = torch.tensor(
                    [[5, 7, 11, 13], [17, 19, 23, 29]], dtype=torch.int32
                )
                assert backend.page_table is not None
                backend.page_table.copy_(expected)
                for pages in captured.values():
                    torch.testing.assert_close(pages, expected)
                self.assertEqual(
                    set(captured),
                    (
                        {"on", "off", "mixed"}
                        if enabled and not enforce
                        else {"on"} if enabled else {"off"}
                    ),
                )

    def test_mixed_mask_preserves_capture_alias_and_clears_padding(self):
        """Replays must update the captured tensor, including newly padded rows."""
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.synthid_request_mask = torch.ones(4, dtype=torch.bool)
        captured = runner.synthid_request_mask[:]
        runner._stage_synthid_request_mask(
            enabled=torch.tensor([True, False, True]), raw_bs=3
        )
        self.assertEqual(captured.tolist(), [True, False, True, False])
        runner._stage_synthid_request_mask(
            enabled=torch.tensor([False, True]), raw_bs=2
        )
        self.assertEqual(captured.tolist(), [False, True, False, False])


if __name__ == "__main__":
    unittest.main()
