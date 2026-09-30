"""Manual metadata routing tests; mocked APIs are not GPU correctness evidence.

This bounded import environment omits the serving/evaluation dependencies used
by registered unit tests and CustomTestCase. Run this module with pytest.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4 import topk, topk_flashinfer


def _api(
    logits,
    seq_lens,
    top_k,
    compress_ratio=1,
    next_n=1,
    return_values=False,
    out_indices=None,
    backend="auto",
):
    raise AssertionError("Resolution must not execute the API")


class TestSparseFlashInferRouting(unittest.TestCase):
    def _resolve(self, api):
        with (
            patch.dict(sys.modules, {"flashinfer": SimpleNamespace(top_k_varlen=api)}),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            return topk.resolve_flashinfer_sparse_topk(torch.device("cuda", 0))

    def test_complete_contract_without_suitable_backends_resolves(self):
        _api.has_backend = Mock(return_value=True)
        _api.is_backend_supported = Mock(side_effect=lambda name, cc: cc in {103, 107})
        self.assertFalse(hasattr(_api, "suitable_auto_backends"))
        self.assertIs(self._resolve(_api), _api)
        _api.has_backend.assert_called_once_with("cudnn")
        _api.is_backend_supported.assert_called_once_with("cudnn", 103)

    def test_old_api_missing_hooks_or_parameters_stays_stock(self):
        self.assertIsNone(self._resolve(None))

        def old(logits, seq_lens, top_k):
            pass

        self.assertIsNone(self._resolve(old))
        old.has_backend = lambda name: True
        old.is_backend_supported = lambda name, cc: True
        self.assertIsNone(self._resolve(old))
        _api.has_backend = lambda name: False
        _api.is_backend_supported = lambda name, cc: True
        self.assertIsNone(self._resolve(_api))
        _api.has_backend = lambda name: True
        _api.is_backend_supported = lambda name, cc: False
        self.assertIsNone(self._resolve(_api))

    def test_stock_utility_does_not_import_or_resolve_flashinfer(self):
        values = [object() for _ in range(4)]
        with patch.object(topk, "topk_transform_bf16_small") as stock:
            topk.topk_transform_sparse(*values)
        stock.assert_called_once_with(*values, 8)

    def test_metadata_decline_uses_stock_without_calling_flashinfer(self):
        values = (
            torch.empty(3, 16, dtype=torch.bfloat16),
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(3, 2, dtype=torch.int32),
            torch.empty(3, 8, dtype=torch.int32),
        )
        op = Mock(side_effect=AssertionError("CPU input must decline before execution"))
        with patch.object(topk, "topk_transform_bf16_small") as stock:
            topk.topk_transform_sparse(*values, topk_op=op)
        op.assert_not_called()
        stock.assert_called_once_with(*values, 8)

    def test_execution_uses_operand_device_and_restores_context(self):
        # Metadata-only fake tensors: this tests host ownership, not GPU math.
        class Tensor:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
                self.device = torch.device("cuda", 1)
                self.ndim, self.is_cuda = len(shape), True

            def stride(self, dimension):
                return self.shape[1] if dimension == 0 and self.ndim == 2 else 1

            def is_contiguous(self):
                return True

            def is_neg(self):
                return False

            def is_conj(self):
                return False

            def data_ptr(self):
                return 4096

        args = (
            Tensor((3, 16), torch.bfloat16),
            Tensor((3,), torch.int32),
            Tensor((3, 2), torch.int32),
            Tensor((3, 8), torch.int32),
        )
        raw = Tensor((3, 8), torch.int32)
        current, events = [0], []

        class Device:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                self.previous = current[0]
                current[0] = self.device.index

            def __exit__(self, *ignored):
                current[0] = self.previous

        def allocate(*args, **kwargs):
            events.append(("allocate", current[0]))
            return raw

        def op(*args, **kwargs):
            events.append(("select", current[0]))
            self.assertIs(kwargs["out_indices"], raw)
            self.assertEqual(
                (kwargs["next_n"], kwargs["compress_ratio"], kwargs["backend"]),
                (1, 1, "auto"),
            )
            return raw, None

        class Kernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: events.append(("remap", current[0]))

        with (
            patch.object(torch.cuda, "device", Device),
            patch.object(torch, "empty", allocate),
            patch.object(topk_flashinfer, "_remap_sparse_topk", Kernel()),
        ):
            self.assertTrue(topk_flashinfer.topk_transform_sparse_flashinfer(*args, op))
        self.assertEqual(events, [("allocate", 1), ("select", 1), ("remap", 1)])
        self.assertEqual(current[0], 0)

    def test_backend_resolves_once_only_when_explicitly_selected(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import sparse_table

        with (
            patch.object(torch.cuda, "Stream"),
            patch.object(
                sparse_table, "resolve_flashinfer_sparse_topk", return_value=_api
            ) as resolve,
        ):
            kwargs = dict(
                token_to_kv_pool=None,
                req_to_token=torch.empty(1),
                page_size=256,
                candidate_topk_blocks=2048,
                candidate_block_size=8,
            )
            stock = sparse_table.SparseTableBackend(**kwargs)
            self.assertIsNone(stock._sparse_topk_op)
            resolve.assert_not_called()
            optional = sparse_table.SparseTableBackend(
                **kwargs, use_flashinfer_topk=True
            )
            self.assertIs(optional._sparse_topk_op, _api)
            resolve.assert_called_once_with(kwargs["req_to_token"].device)

    def test_consumer_waits_before_scores_then_passes_physical_table(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import sparse_table

        backend = object.__new__(sparse_table.SparseTableBackend)
        backend.token_to_kv_pool = object()
        backend._sparse_topk_op = _api
        inputs = object()
        data = SimpleNamespace(
            q_fp4=object(), q_sf=object(), k_cache=object(), weights=torch.empty(1)
        )
        published = SimpleNamespace(
            ready=object(),
            blocks=torch.empty(3, 2048),
            valid_lens=object(),
            phys_blocks=object(),
        )
        out = SimpleNamespace(raw_indices=None, page_indices=object())
        sequence = []
        stream = SimpleNamespace(
            wait_event=lambda event: sequence.append(("wait", event))
        )
        logits = object()

        def scores(*args):
            sequence.append(("score", args[-1]))
            return logits

        def select(*args, **kwargs):
            sequence.append(("select", args, kwargs))

        with (
            patch.object(sparse_table, "get_deep_gemm_decode_data", return_value=data),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(sparse_table, "sparse_logits", side_effect=scores),
            patch.object(sparse_table, "topk_transform_sparse", side_effect=select),
        ):
            backend.consume_decode(inputs, published, out)
        self.assertEqual(
            sequence,
            [
                ("wait", published.ready),
                ("score", 2048),
                (
                    "select",
                    (
                        logits,
                        published.valid_lens,
                        published.phys_blocks,
                        out.page_indices,
                    ),
                    {"topk_op": _api},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
