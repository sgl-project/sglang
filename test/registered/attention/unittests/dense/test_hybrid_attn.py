import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
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

register_cuda_ci(est_time=12, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestHybridAttnDenseAttentionBackendCorrectness(CustomTestCase):
    """Compose HybridAttnBackend(prefill=triton, decode=flashinfer) and verify
    dispatch produces the dense reference for both EXTEND and DECODE."""

    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    EXTEND_CASE = DenseAttentionCase(
        name="hybrid_extend_no_prefix",
        backend="triton",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=4,
        page_size=16,
        prefix_lens=(0,),
        extend_lens=(16,),
    )
    DECODE_CASE = DenseAttentionCase(
        name="hybrid_decode_nonzero_prefix",
        backend="flashinfer",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=4,
        page_size=16,
        prefix_lens=(7,),
    )

    def _build_and_wrap(self, case: DenseAttentionCase):
        fixture = build_dense_attention_fixture(
            self,
            case,
            head_dim=self.HEAD_DIM,
            hidden_size=self.HIDDEN_SIZE,
        )
        try:
            prefill_backend = ATTENTION_BACKENDS["triton"](fixture.runner)
            decode_backend = ATTENTION_BACKENDS["flashinfer"](fixture.runner)
        except (AssertionError, ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"hybrid_attn child backend unavailable: {exc}")
        wrapper = HybridAttnBackend(
            fixture.runner,
            prefill_backend=prefill_backend,
            decode_backend=decode_backend,
        )
        return replace_backend(fixture, wrapper)

    def test_hybrid_extend_dispatches_prefill_backend(self):
        fixture = self._build_and_wrap(self.EXTEND_CASE)
        actual = run_dense_fixture_eager(fixture)
        expected = expected_dense_fixture_output(fixture)
        torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)

    def test_hybrid_decode_dispatches_decode_backend(self):
        fixture = self._build_and_wrap(self.DECODE_CASE)
        actual = run_dense_fixture_eager(fixture)
        expected = expected_dense_fixture_output(fixture)
        torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


if __name__ == "__main__":
    unittest.main()
