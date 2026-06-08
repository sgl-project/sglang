"""Unit tests for the init contract that piecewise + breakable cuda graph
capture rely on with ``forward_mode=EXTEND``.

The piecewise + breakable runners (unlike the full ``cuda_graph_runner``)
capture prefill chunks, which means their capture path passes
``forward_mode=ForwardMode.EXTEND`` to ``attn_backend.init_forward_metadata*``.

Backends like FlashInfer and FA3 implement two separate init bodies:

- ``init_forward_metadata(fb)`` — the eager entry. Handles all modes
  including plain ``EXTEND`` (full prefill / chunked prefill).
- ``init_forward_metadata_out_graph(fb, in_capture=True)`` — the
  bucket-keyed wrapper prep used by the full cuda graph runner. Only
  handles modes the full runner captures: ``DECODE`` / ``IDLE`` /
  ``TARGET_VERIFY`` / ``DRAFT_EXTEND`` / ``DLLM_EXTEND``. Plain
  ``EXTEND`` is not in scope here and the body raises on it.

These tests pin both halves of that contract so a future refactor that
incorrectly routes piecewise/breakable capture through ``_out_graph(in_capture=True)``
fails at unit-test time instead of at GPU CI / e2e time. This is the
specific regression #26735 introduced and then fixed
(see ``piecewise_cuda_graph_runner.py`` /
``breakable_cuda_graph_runner.py`` capture sites).
"""

import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device_sm
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
)

register_cuda_ci(est_time=9, stage="base-a", runner_config="1-gpu-small")

_EXTEND_CASE = DenseAttentionCase(
    name="extend_no_prefix_smoke",
    backend="fa3",  # overridden per backend below
    forward_mode=ForwardMode.EXTEND,
    num_heads=4,
    num_kv_heads=4,
    page_size=16,
    prefix_lens=(0,),
    extend_lens=(16,),
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestExtendInitContract(CustomTestCase):

    def _make_case(self, backend: str) -> DenseAttentionCase:
        return DenseAttentionCase(
            name=f"extend_no_prefix_{backend}",
            backend=backend,
            forward_mode=_EXTEND_CASE.forward_mode,
            num_heads=_EXTEND_CASE.num_heads,
            num_kv_heads=_EXTEND_CASE.num_kv_heads,
            page_size=_EXTEND_CASE.page_size,
            prefix_lens=_EXTEND_CASE.prefix_lens,
            extend_lens=_EXTEND_CASE.extend_lens,
        )

    def _build_fixture(self, backend: str, *, head_dim: int = 16):
        case = self._make_case(backend)
        try:
            return build_dense_attention_fixture(self, case, head_dim=head_dim)
        except (AssertionError, ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"backend {backend} unavailable: {exc}")

    def _assert_extend_eager_init_well_formed(
        self, backend: str, *, head_dim: int = 16
    ):
        fixture = self._build_fixture(backend, head_dim=head_dim)
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        meta = fixture.backend.forward_metadata
        self.assertIsNotNone(
            meta,
            f"{backend}: init_forward_metadata(EXTEND) left forward_metadata as None — "
            "piecewise/breakable capture would crash on the first forward_extend call.",
        )
        page_table = getattr(meta, "page_table", None)
        if page_table is not None:
            self.assertIsInstance(page_table, torch.Tensor)

    @unittest.skipIf(
        get_device_sm() >= 100 or get_device_sm() < 80,
        "FA3 backend requires SM 80-90",
    )
    def test_fa3_extend_eager_init(self):
        self._assert_extend_eager_init_well_formed("fa3")

    def test_flashinfer_extend_eager_init(self):
        # FlashInfer's JIT prefill kernel needs head_dim ≥ 64.
        self._assert_extend_eager_init_well_formed("flashinfer", head_dim=128)

    def test_triton_extend_eager_init(self):
        self._assert_extend_eager_init_well_formed("triton")

    def _assert_out_graph_in_capture_rejects_extend(
        self, backend: str, *, head_dim: int = 16
    ):
        """``init_forward_metadata_out_graph(fb, in_capture=True)`` is the
        bucket-prep path. Plain EXTEND mode isn't in its supported set —
        piecewise/breakable capture used to route through here and crash.
        This pins the constraint: if the path starts handling EXTEND
        (raises become passes), revisit the piecewise/breakable capture
        wiring and consider unifying the API.
        """
        fixture = self._build_fixture(backend, head_dim=head_dim)
        try:
            fixture.backend.init_forward_metadata_out_graph(
                fixture.forward_batch, in_capture=True
            )
        except (ValueError, AttributeError, KeyError, AssertionError):
            return
        meta = fixture.backend.forward_metadata
        self.assertIsNotNone(
            meta,
            f"{backend}: _out_graph(in_capture=True) accepted EXTEND but left "
            "forward_metadata as None — this is the specific bug pattern that "
            "broke piecewise capture (FA3._apply_cuda_graph_metadata falls "
            "through for unsupported modes and sets self.forward_metadata = None).",
        )

    @unittest.skipIf(
        get_device_sm() >= 100 or get_device_sm() < 80,
        "FA3 backend requires SM 80-90",
    )
    def test_fa3_out_graph_capture_rejects_extend(self):
        self._assert_out_graph_in_capture_rejects_extend("fa3")

    def test_flashinfer_out_graph_capture_rejects_extend(self):
        self._assert_out_graph_in_capture_rejects_extend("flashinfer", head_dim=128)


if __name__ == "__main__":
    unittest.main()
