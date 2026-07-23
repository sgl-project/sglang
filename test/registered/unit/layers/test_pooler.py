import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_forward_batch(extend_seq_lens, dimensions=None, extend_seq_lens_cpu=None):
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=(
            list(extend_seq_lens)
            if extend_seq_lens_cpu is None
            else extend_seq_lens_cpu
        ),
        dimensions=dimensions,
    )


class TestPooler(CustomTestCase):
    def test_last_pooling_returns_last_token(self):
        hidden_states = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 3])

        output = Pooler(PoolingType.LAST, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 20.0],
                    [5.0, 50.0],
                ]
            ),
        )

    def test_cls_pooling_returns_first_token(self):
        hidden_states = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 3])

        output = Pooler(PoolingType.CLS, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [1.0, 10.0],
                    [3.0, 30.0],
                ]
            ),
        )

    def test_mean_pooling_returns_segment_means(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 3])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 3.0],
                    [7.0, 8.0],
                ]
            ),
        )

    def test_mean_pooling_normalizes_output(self):
        hidden_states = torch.tensor(
            [
                [2.0, 2.0],
                [4.0, 6.0],
                [0.0, 3.0],
                [0.0, 7.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 2])

        output = Pooler(PoolingType.MEAN, normalize=True)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [0.6, 0.8],
                    [0.0, 1.0],
                ]
            ),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_mean_pooling_supports_tensor_extend_seq_lens_cpu(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ]
        )
        forward_batch = _make_forward_batch(
            [1, 4], extend_seq_lens_cpu=torch.tensor([2, 3], dtype=torch.int32)
        )

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 3.0],
                    [7.0, 8.0],
                ]
            ),
        )

    def test_mean_pooling_prefers_extend_seq_lens_cpu(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ]
        )
        forward_batch = _make_forward_batch([1, 4], extend_seq_lens_cpu=[2, 3])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0, 3.0],
                    [7.0, 8.0],
                ]
            ),
        )

    def test_mean_pooling_accumulates_in_float32(self):
        torch.manual_seed(2)
        hidden_states = (torch.randn(4096, 2, dtype=torch.float32) * 100).to(
            torch.bfloat16
        )
        forward_batch = _make_forward_batch([2048, 2048])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        expected = torch.stack(
            [
                hidden_states[:2048].float().mean(dim=0),
                hidden_states[2048:].float().mean(dim=0),
            ]
        )
        self.assertEqual(output.embeddings.dtype, torch.float32)
        torch.testing.assert_close(output.embeddings, expected, atol=1e-6, rtol=0.0)

    def test_dimensions_same_truncates_tensor_output(self):
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [6.0, 8.0, 10.0],
                [8.0, 10.0, 12.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 2], dimensions=[1, 1])

        output = Pooler(PoolingType.MEAN, normalize=False)(hidden_states, forward_batch)

        torch.testing.assert_close(
            output.embeddings,
            torch.tensor(
                [
                    [2.0],
                    [7.0],
                ]
            ),
        )

    def test_dimensions_mixed_returns_list_and_normalizes_each(self):
        hidden_states = torch.tensor(
            [
                [2.0, 2.0, 0.0],
                [4.0, 6.0, 0.0],
                [0.0, 2.0, 2.0],
                [0.0, 4.0, 6.0],
            ]
        )
        forward_batch = _make_forward_batch([2, 2], dimensions=[1, 2])

        output = Pooler(PoolingType.MEAN, normalize=True)(hidden_states, forward_batch)

        self.assertIsInstance(output.embeddings, list)
        self.assertEqual(len(output.embeddings), 2)
        torch.testing.assert_close(output.embeddings[0], torch.tensor([1.0]))
        torch.testing.assert_close(
            output.embeddings[1],
            torch.tensor([0.0, 1.0]),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
