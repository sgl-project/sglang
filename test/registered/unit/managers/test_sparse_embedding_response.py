"""Contract tests for sparse embedding output and response conversion."""

import math
import unittest

import torch

from sglang.srt.layers.sparse_pooler import SparseEmbeddingOutput
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _scalar_reference(tensor: torch.Tensor) -> list[dict[int, float]]:
    batch_ids, token_ids = tensor.indices()
    values = tensor.values()
    output = [{} for _ in range(tensor.size(0))]
    for index in range(values.shape[0]):
        output[batch_ids[index].item()][token_ids[index].item()] = values[index].item()
    return output


def _bulk(tensor: torch.Tensor) -> list[dict[int, float]]:
    return batch_result_processor._sparse_embeddings_to_python(tensor)


def _linear_sparse(batch: int, nnz: int) -> torch.Tensor:
    linear = torch.arange(nnz, dtype=torch.int64)
    indices = torch.stack((linear % batch, linear // batch))
    return torch.sparse_coo_tensor(
        indices,
        linear.to(torch.float32),
        size=(batch, math.ceil(nnz / batch)),
    ).coalesce()


class TestSparseEmbeddingResponse(CustomTestCase):
    def test_sparse_output_satisfies_scheduler_contract(self):
        output = SparseEmbeddingOutput(embeddings=torch.tensor([[1.0]]))
        self.assertIsNone(output.pooled_hidden_states)

    def test_empty_preserves_batch_rows(self):
        tensor = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.int64),
            torch.empty((0,), dtype=torch.float32),
            size=(5, 128),
        ).coalesce()
        self.assertEqual(_bulk(tensor), [{}, {}, {}, {}, {}])

    def test_singleton(self):
        tensor = torch.sparse_coo_tensor(
            torch.tensor([[0], [2]]), torch.tensor([1.25]), size=(1, 4)
        ).coalesce()
        self.assertEqual(_bulk(tensor), [{2: 1.25}])

    def test_exact_dictionary_parity_across_value_dtypes(self):
        indices = torch.tensor([[0, 0, 2, 3], [1, 7, 4, 9]])
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                tensor = torch.sparse_coo_tensor(
                    indices,
                    torch.tensor([0.5, -2.0, 3.25, 8.0], dtype=dtype),
                    size=(4, 16),
                ).coalesce()
                self.assertEqual(_bulk(tensor), _scalar_reference(tensor))

    def test_signed_zero_is_preserved(self):
        tensor = torch.sparse_coo_tensor(
            torch.tensor([[0, 0], [2, 3]]),
            torch.tensor([-0.0, 0.0]),
            size=(1, 8),
        ).coalesce()
        result = _bulk(tensor)[0]
        self.assertTrue(torch.signbit(torch.tensor(result[2])).item())
        self.assertFalse(torch.signbit(torch.tensor(result[3])).item())

    def test_coalesced_duplicate_semantics(self):
        tensor = torch.sparse_coo_tensor(
            torch.tensor([[1, 1], [3, 3]]),
            torch.tensor([1.25, 2.75]),
            size=(2, 8),
        ).coalesce()
        self.assertEqual(_bulk(tensor), [{}, {3: 4.0}])

    def test_chunk_boundary(self):
        tensor = _linear_sparse(batch=3, nnz=515)
        self.assertEqual(_bulk(tensor), _scalar_reference(tensor))

    def test_large_nnz(self):
        tensor = _linear_sparse(batch=32, nnz=8192)
        result = _bulk(tensor)
        self.assertEqual(result, _scalar_reference(tensor))
        self.assertEqual(sum(map(len, result)), 8192)

    def test_input_metadata_is_not_mutated(self):
        tensor = _linear_sparse(batch=4, nnz=37)
        indices_before = tensor.indices().clone()
        values_before = tensor.values().clone()
        _bulk(tensor)
        self.assertTrue(torch.equal(tensor.indices(), indices_before))
        self.assertTrue(torch.equal(tensor.values(), values_before))

    def test_uncoalesced_error_is_preserved(self):
        tensor = torch.sparse_coo_tensor(
            torch.tensor([[0, 0], [1, 1]]),
            torch.tensor([1.0, 2.0]),
            size=(1, 8),
        )
        with self.assertRaises(RuntimeError):
            _bulk(tensor)


if __name__ == "__main__":
    unittest.main()
