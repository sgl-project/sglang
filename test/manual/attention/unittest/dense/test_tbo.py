import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    replace_backend,
    run_dense_fixture_eager,
)


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

    def _build_and_wrap(self, case: DenseAttentionCase):
        fixture = build_dense_attention_fixture(self, case)
        try:
            primary = ATTENTION_BACKENDS["triton"](fixture.runner)
            children = [ATTENTION_BACKENDS["triton"](fixture.runner) for _ in range(2)]
        except (AssertionError, ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"tbo child backend unavailable: {exc}")
        wrapper = TboAttnBackend(primary=primary, children=children)
        return replace_backend(fixture, wrapper)

    def test_tbo_extend_delegates_to_primary(self):
        fixture = self._build_and_wrap(self.EXTEND_CASE)
        actual = run_dense_fixture_eager(fixture)
        expected = expected_dense_fixture_output(fixture)
        torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


if __name__ == "__main__":
    unittest.main()
