import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device_sm
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    replace_backend,
    run_dense_fixture_eager,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    _prepare_spec_verify_batch,
)

register_cuda_ci(est_time=11, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTboAttnDenseAttentionBackendCorrectness(CustomTestCase):
    """Compose TboAttnBackend(primary=triton, children=[triton, triton]) and
    verify the eager dispatch matches the dense reference.

    The TBO wrapper only orchestrates two-batch splitting when
    ``forward_batch.tbo_children`` is set (driven by the scheduler and CUDA
    graph capture paths). Without children, ``init_forward_metadata`` and
    ``forward`` delegate to ``self.primary``, so composition correctness is
    what's covered here. Sub-batched orchestration through the TBO children
    requires scheduler-level batch splitting and CUDA-graph helpers
    (``two_batch_overlap.compute_split_indices_for_cuda_graph_replay``) that
    aren't present in the unit fixture; that path stays for Phase 3 graph
    expansion.
    """

    EXTEND_CASE = DenseAttentionCase(
        name="tbo_extend_no_prefix",
        backend="triton",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=4,
        page_size=16,
        prefix_lens=(0,),
        extend_lens=(16,),
    )

    # Mirrors ``runner_fa3_eagle_verify_chain`` in test_fa3.py — the smallest
    # case shape that drives a real TARGET_VERIFY CUDA-graph capture through
    # FlashAttention's per-bs metadata dicts.
    TARGET_VERIFY_CAPTURE_CASE = DenseAttentionCase(
        name="tbo_fa3_target_verify_chain_capture",
        backend="fa3",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_heads=4,
        num_kv_heads=4,
        page_size=16,
        prefix_lens=(4, 7),
        extend_lens=(3, 3),
    )

    def _build_and_wrap(self, case: DenseAttentionCase):
        fixture = build_dense_attention_fixture(self, case)
        try:
            primary = ATTENTION_BACKENDS[case.backend](fixture.runner)
            children = [
                ATTENTION_BACKENDS[case.backend](fixture.runner) for _ in range(2)
            ]
        except (AssertionError, ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"tbo child backend unavailable: {exc}")
        wrapper = TboAttnBackend(primary=primary, children=children)
        return replace_backend(fixture, wrapper)

    def test_tbo_extend_delegates_to_primary(self):
        fixture = self._build_and_wrap(self.EXTEND_CASE)
        actual = run_dense_fixture_eager(fixture)
        expected = expected_dense_fixture_output(fixture)
        torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)

    @unittest.skipIf(
        get_device_sm() >= 100 or get_device_sm() < 80,
        "FA3 backend requires SM 80-90",
    )
    def test_tbo_target_verify_cuda_graph_capture_delegates_to_primary_capture(self):
        """TBO capture must dispatch primary's
        ``init_forward_metadata_out_graph(fb, in_capture=True)`` (the capture
        path), not the replay path.

        Backends like FlashAttention store per-bs metadata in dicts populated
        only by the in_capture=True branch (via ``_bind_metadata_buffers``).
        If TBO short-circuits its capture to its own replay path, those dicts
        are empty and replay raises ``KeyError: bs``. Reproduces the
        deepep-4-gpu-h100 failure where
        ``flashattention_backend.target_verify_metadata[bs]`` lookup blew up
        during ``init_device_graphs``.

        Asserts capture completes without raising — numerical correctness of
        the captured graph is covered by per-backend spec-verify tests.
        """
        case = self.TARGET_VERIFY_CAPTURE_CASE
        fixture = self._build_and_wrap(case)
        wrapper = fixture.backend
        batch = fixture.forward_batch

        # Wire TARGET_VERIFY batch state + EAGLE chain (topk=1) spec_info,
        # mirroring what the per-backend spec-verify runner sets up.
        _prepare_spec_verify_batch(
            case,
            batch,
            topk=1,
            spec_kind="eagle",
            device=str(batch.seq_lens.device),
        )

        capture_bs = case.batch_size
        num_tokens = sum(case.extend_lens)
        wrapper.init_cuda_graph_state(max_bs=capture_bs, max_num_tokens=num_tokens)
        wrapper.init_forward_metadata_out_graph(batch, in_capture=True)

    @unittest.skipIf(
        get_device_sm() >= 100 or get_device_sm() < 80,
        "FA3 backend requires SM 80-90",
    )
    def test_tbo_target_verify_cuda_graph_replay_splits_children(self):
        """TBO replay must split the padded capture-time buffers into per-child
        views before dispatching to ``init_forward_metadata_out_graph(fb_view,
        in_capture=False)``.

        ``cuda_graph_runner.replay_prepare`` constructs a ``SimpleNamespace``
        fb_view via ``build_replay_fb_view`` — it has no ``tbo_children``
        attribute because ``tbo_plugin.replay_prepare`` does not call
        ``prepare_raw``. Without the in-line split, TBO either crashes
        (AttributeError on ``fb_view.tbo_children``) or silently leaves child
        metadata stale.

        Asserts both children's ``init_forward_metadata_out_graph`` is invoked
        with sliced ``req_pool_indices`` / ``seq_lens`` / ``seq_lens_cpu``
        whose lengths match the split derived from
        ``compute_split_indices_for_cuda_graph_replay``.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from sglang.srt.batch_overlap.two_batch_overlap import (
            compute_split_indices_for_cuda_graph_replay,
        )

        case = self.TARGET_VERIFY_CAPTURE_CASE
        fixture = self._build_and_wrap(case)
        wrapper = fixture.backend
        batch = fixture.forward_batch
        _prepare_spec_verify_batch(
            case,
            batch,
            topk=1,
            spec_kind="eagle",
            device=str(batch.seq_lens.device),
        )

        capture_bs = case.batch_size
        num_tokens_per_bs = sum(case.extend_lens) // capture_bs
        num_tokens = capture_bs * num_tokens_per_bs
        split_seq_index, split_token_index = (
            compute_split_indices_for_cuda_graph_replay(
                forward_mode=batch.forward_mode,
                cuda_graph_num_tokens=num_tokens,
                spec_info=batch.spec_info,
            )
        )
        self.assertGreater(split_seq_index, 0)
        self.assertLess(split_seq_index, capture_bs)

        # fb_view shaped like build_replay_fb_view's output: a SimpleNamespace
        # with no tbo_children attribute.
        fb_view = SimpleNamespace(
            batch_size=capture_bs,
            forward_mode=batch.forward_mode,
            actual_forward_mode=batch.forward_mode,
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            seq_lens_sum=int(batch.seq_lens_cpu.sum()),
            seq_lens_cpu=batch.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=batch.out_cache_loc,
            spec_info=batch.spec_info,
        )

        # Pure mocks (no `wraps=...`) so the dispatcher's slicing/contract is
        # observed without invoking real backend bodies.
        primary_mock = MagicMock()
        child_mocks = [MagicMock(), MagicMock()]
        wrapper.primary = primary_mock
        wrapper.children = child_mocks

        wrapper.init_forward_metadata_out_graph(fb_view, in_capture=False)

        primary_mock.init_forward_metadata_out_graph.assert_called_once()
        for child_mock in child_mocks:
            child_mock.init_forward_metadata_out_graph.assert_called_once()
        child_fbs = [
            m.init_forward_metadata_out_graph.call_args.kwargs["forward_batch"]
            for m in child_mocks
        ]
        self.assertEqual(child_fbs[0].batch_size, split_seq_index)
        self.assertEqual(child_fbs[1].batch_size, capture_bs - split_seq_index)
        self.assertEqual(child_fbs[0].req_pool_indices.shape[0], split_seq_index)
        self.assertEqual(
            child_fbs[1].req_pool_indices.shape[0], capture_bs - split_seq_index
        )


if __name__ == "__main__":
    unittest.main()
