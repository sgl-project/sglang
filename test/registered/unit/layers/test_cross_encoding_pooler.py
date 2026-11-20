"""Packed cross-encoder scores and host synchronization in the real model heads."""

import itertools
import unittest

import torch
from transformers import BertConfig, RobertaConfig

from sglang.srt.layers.pooler import CrossEncodingPooler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.bert import BertPooler
from sglang.srt.models.roberta import RobertaClassificationHead
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


def make_case(model, num_labels, lengths, device, dtype, host_lengths, strided=False):
    config_class = BertConfig if model == "bert" else RobertaConfig
    config = config_class(hidden_size=3, num_labels=num_labels)
    if model == "bert":
        head = BertPooler(config)
        classifier = torch.nn.Linear(3, num_labels)
        pooler = CrossEncodingPooler(config, classifier, head)
    else:
        head = RobertaClassificationHead(config)
        classifier = head.out_proj
        pooler = CrossEncodingPooler(config, head)

    # Both real heads now return tanh(first token) in the first num_labels columns.
    with torch.no_grad():
        head.dense.weight.copy_(torch.eye(3))
        head.dense.bias.zero_()
        classifier.weight.copy_(torch.eye(3)[:num_labels])
        classifier.bias.zero_()
    pooler = pooler.to(device=device, dtype=dtype).eval()

    tokens = sum(lengths)
    hidden = torch.empty((tokens, 6 if strided else 3), device=device, dtype=dtype)
    if strided:
        hidden = hidden[:, ::2]
    hidden.copy_(
        torch.tensor(
            [
                [0.0, 0.5, -0.5],
                [1.0, -1.0, 0.25],
                [9.0, 9.0, 9.0],
                [8.0, 8.0, 8.0],
                [-0.25, 0.75, -1.0],
                [7.0, 7.0, 7.0],
            ][:tokens],
            device=device,
            dtype=dtype,
        )
    )
    device_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(lengths),
        input_ids=torch.zeros(tokens, dtype=torch.long, device=device),
        req_pool_indices=torch.arange(len(lengths), device=device),
        seq_lens=device_lengths,
        out_cache_loc=torch.arange(tokens, device=device),
        seq_lens_sum=tokens,
        extend_seq_lens=device_lengths,
        extend_seq_lens_cpu=list(lengths) if host_lengths else None,
    )
    return pooler, hidden, batch


class TestCrossEncodingPooler(CustomTestCase):
    @torch.inference_mode()
    def check_scores(self, device, dtypes):
        for (
            model,
            num_labels,
            host_lengths,
            lengths,
            dtype,
            strided,
        ) in itertools.product(
            ("bert", "roberta"),
            (1, 2),
            (True, False),
            ([1], [1, 3, 2]),
            dtypes,
            (True, False),
        ):
            with self.subTest(
                model=model,
                labels=num_labels,
                host_lengths=host_lengths,
                lengths=lengths,
                dtype=dtype,
                strided=strided,
            ):
                pooler, hidden, batch = make_case(
                    model, num_labels, lengths, device, dtype, host_lengths, strided
                )
                actual = pooler(hidden, batch).embeddings
                # Hand-computed tanh of the first token at packed offsets 0, 1, 4.
                expected = torch.tensor(
                    [
                        [0.0, 0.46211715726000974],
                        [0.7615941559557649, -0.7615941559557649],
                        [-0.24491866240370913, 0.6351489523872873],
                    ][: len(lengths)],
                    device=device,
                    dtype=dtype,
                )[:, :num_labels].squeeze(-1)
                torch.testing.assert_close(actual, expected)

    def test_cpu_scores_respect_packed_boundaries(self):
        self.check_scores("cpu", (torch.float32,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_scores_respect_packed_boundaries(self):
        self.check_scores("cuda", (torch.float32, torch.float16, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_cuda_pooling_does_not_read_individual_lengths(self):
        for model, host_lengths in itertools.product(
            ("bert", "roberta"), (True, False)
        ):
            with self.subTest(model=model, host_lengths=host_lengths):
                pooler, hidden, batch = make_case(
                    model, 1, [1, 3, 2], "cuda", torch.float32, host_lengths
                )
                pooler(hidden, batch)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    pooler(hidden, batch)
                torch.cuda.synchronize()
                scalar_reads = sum(
                    event.count
                    for event in prof.key_averages()
                    if event.key == "aten::_local_scalar_dense"
                )
                self.assertEqual(
                    scalar_reads, 0, "Pooling reads CUDA lengths one by one"
                )


if __name__ == "__main__":
    unittest.main()
