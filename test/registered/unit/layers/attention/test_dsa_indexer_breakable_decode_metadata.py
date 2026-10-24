import unittest
from unittest import mock

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeForwardBatch:
    def __init__(self, forward_mode):
        self.forward_mode = forward_mode
        self.attn_cp_metadata = None


class TestDsaIndexerBreakableDecodeMetadata(CustomTestCase):
    """Indexer.forward_cuda must not null out ``metadata`` for forward modes
    that fall through to ``_get_topk_paged``/``_get_topk_ragged`` instead of
    the graph DSA split-op dispatch, even while running under a
    piecewise/breakable CUDA graph -- otherwise those call sites crash with
    ``'NoneType' object has no attribute 'get_page_table_64'`` (observed on
    ROCm decode under ``--cuda-graph-backend-decode breakable``, since the
    split-op dispatch is CUDA-only and decode is never on that surface).
    """

    def _run(self, forward_mode, *, in_breakable_graph, dsa_enable_prefill_cp, is_cuda):
        fake_self = mock.MagicMock()
        fake_self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        fake_self.alt_stream = None
        fake_self.use_dsa_indexer_fusion = False

        fake_backend = mock.MagicMock()
        fake_backend.get_indexer_metadata.return_value = object()
        forward_batch = _FakeForwardBatch(forward_mode)

        with (
            mock.patch.object(
                dsa_indexer, "get_attn_backend", return_value=fake_backend
            ),
            mock.patch.object(
                dsa_indexer,
                "_is_in_piecewise_or_breakable_cuda_graph",
                return_value=in_breakable_graph,
            ),
            mock.patch.object(
                dsa_utils, "is_in_breakable_cuda_graph", return_value=in_breakable_graph
            ),
            mock.patch.object(dsa_utils, "is_cuda", return_value=is_cuda),
        ):
            try:
                dsa_indexer.Indexer.forward_cuda(
                    fake_self,
                    mock.MagicMock(),
                    mock.MagicMock(),
                    mock.MagicMock(),
                    forward_batch,
                    0,
                )
            except Exception:
                # forward_cuda does real tensor work past the metadata check;
                # a mocked self/x cannot get further, and this test only cares
                # about the metadata fetch that happens before any of that.
                pass

        return fake_backend.get_indexer_metadata.call_count

    def test_decode_under_breakable_graph_still_fetches_metadata(self):
        call_count = self._run(
            ForwardMode.DECODE,
            in_breakable_graph=True,
            dsa_enable_prefill_cp=False,
            is_cuda=False,
        )
        self.assertEqual(
            call_count,
            1,
            "decode under a breakable CUDA graph must fetch DSA indexer "
            "metadata instead of leaving it None",
        )

    def test_prefill_cp_under_breakable_graph_still_fetches_metadata(self):
        call_count = self._run(
            ForwardMode.EXTEND,
            in_breakable_graph=True,
            dsa_enable_prefill_cp=True,
            is_cuda=True,
        )
        self.assertEqual(
            call_count,
            1,
            "prefill context-parallel forwards are not on the graph DSA "
            "split-op surface and must fetch metadata too",
        )

    def test_split_op_surface_prefill_still_skips_eager_metadata_fetch(self):
        call_count = self._run(
            ForwardMode.EXTEND,
            in_breakable_graph=True,
            dsa_enable_prefill_cp=False,
            is_cuda=True,
        )
        self.assertEqual(
            call_count,
            0,
            "non-CP prefill on the graph DSA split-op surface fetches its "
            "own metadata inside the split op, not eagerly here",
        )


if __name__ == "__main__":
    unittest.main()
