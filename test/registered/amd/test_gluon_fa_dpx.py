"""Graph replay regressions for long BF16 attention on 96-CU gfx1250."""

import itertools
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, suite="nightly-amd-1-gpu-mi45x", nightly=True)


def _is_dpx_gfx1250():
    if not torch.cuda.is_available() or not torch.version.hip:
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        "gfx1250" in getattr(properties, "gcnArchName", "")
        and properties.multi_processor_count == 96
    )


def _reference_attention(query, key, value):
    # Independent stable FP64 oracle: sampled queries attend to every key.
    q, k, v = (tensor.double().transpose(1, 2) for tensor in (query, key, value))
    probabilities = torch.softmax((q @ k.transpose(-1, -2)) * 128**-0.5, dim=-1)
    return (probabilities @ v).transpose(1, 2)


@unittest.skipUnless(_is_dpx_gfx1250(), "requires 96-CU AMD gfx1250")
class TestDpxBf16GraphReplay(unittest.TestCase):
    def _check_replay_and_recovery(self, case):
        from sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward import (
            flash_attention,
        )

        for seqlen, storage_offset, value_stride in itertools.product(
            (65_536, 65_537), (0, 1), (1, 3)
        ):
            with (
                self.subTest(
                    seqlen=seqlen,
                    storage_offset=storage_offset,
                    value_stride=value_stride,
                ),
                torch.random.fork_rng(devices=[0]),
            ):
                torch.manual_seed(41972)
                inputs = [
                    torch.empty(
                        2 * seqlen * 3 * 128 + storage_offset,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )[storage_offset:]
                    .view(2, seqlen, 3, 128)
                    .normal_()
                    for _ in range(3)
                ]
                if value_stride == 3:
                    # Wan's value view comes from a packed QKV projection.
                    inputs[2] = (
                        torch.empty(
                            2 * seqlen * 3 * 3 * 128 + storage_offset,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )[storage_offset:]
                        .view(2, seqlen, 3, 3, 128)[:, :, 2]
                        .normal_()
                    )
                query, key, value = inputs
                original = [tensor.clone() for tensor in inputs]
                flash_attention(*inputs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = flash_attention(*inputs)

                if case in (
                    "late-key",
                    "single-query",
                    "zero-values",
                    "finite-denominator-overflow",
                ):
                    query.fill_(1)
                    if case == "single-query":
                        query.zero_()
                        query[1, seqlen // 2, 2].fill_(1)
                    key.zero_()
                    key[:, -1].fill_(1 if case == "finite-denominator-overflow" else 16)
                    value.zero_()
                    if case != "zero-values":
                        value[:, -1].fill_(
                            2.0**120 if case == "finite-denominator-overflow" else 1
                        )
                elif case == "large-uniform":
                    query.fill_(66048)
                    query[1].neg_()
                    key.fill_(66048)
                    value.fill_(1)

                rows = [0, 255, 256, seqlen // 2, seqlen - 1]
                # Each test covers its named input and recovery to random input
                # on the same graph, so retry flags must refresh in both directions.
                for restore in (False, True):
                    if restore:
                        for tensor, saved in zip(inputs, original):
                            tensor.copy_(saved)
                    expected = _reference_attention(query[:, rows], key, value)
                    rms = expected.square().mean().sqrt().item()
                    for _ in range(10):
                        graph.replay()
                        self.assertTrue(torch.isfinite(output).all().item())
                        actual = output[:, rows].double()
                        error = (actual - expected).square().mean().sqrt().item()
                        self.assertLessEqual(error, 0.003 * rms + 1e-12)
                        torch.testing.assert_close(
                            actual, expected, rtol=0.01, atol=0.01 * max(rms, 1e-7)
                        )
                torch.cuda.synchronize()

    def test_random_graph_replay(self):
        self._check_replay_and_recovery("random")

    def test_late_key_graph_replay(self):
        self._check_replay_and_recovery("late-key")

    def test_single_query_retry_graph_replay(self):
        self._check_replay_and_recovery("single-query")

    def test_zero_value_retry_graph_replay(self):
        self._check_replay_and_recovery("zero-values")

    def test_finite_denominator_overflow_graph_replay(self):
        self._check_replay_and_recovery("finite-denominator-overflow")

    def test_large_uniform_logits_graph_replay(self):
        self._check_replay_and_recovery("large-uniform")


if __name__ == "__main__":
    unittest.main()
