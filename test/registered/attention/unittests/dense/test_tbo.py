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

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
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
        """TBO capture must invoke ``primary.init_forward_metadata_capture_cuda_graph``,
        not ``primary.init_forward_metadata_replay_cuda_graph``.

        Backends like FlashAttention store per-bs metadata in dicts populated
        only by their capture path (via ``_bind_metadata_buffers``). If TBO
        short-circuits its capture to its own replay (which delegates to
        ``primary.replay``), those dicts are empty and replay raises
        ``KeyError: bs``. Reproduces the deepep-4-gpu-h100 failure where
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
        # This is the failing call before the fix: TBO.capture delegating to
        # primary.replay (instead of primary.capture) reads an unpopulated
        # ``target_verify_metadata[bs]`` dict and raises KeyError.
        wrapper.init_forward_metadata_capture_cuda_graph(
            bs=capture_bs,
            num_tokens=num_tokens,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            encoder_lens=batch.encoder_lens,
            forward_mode=batch.forward_mode,
            spec_info=batch.spec_info,
        )


if __name__ == "__main__":
    unittest.main()
