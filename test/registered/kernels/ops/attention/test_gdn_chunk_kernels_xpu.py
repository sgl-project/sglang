"""Functional check of the Gated DeltaNet chunk prefill kernels on Intel XPU.

On XPU these kernels load and store their tiles through tensor descriptors. Each case
compares one kernel against a float32 torch reference on shapes where descriptor
bounds handling matters: ragged varlen tails (sequences of 1, 63, 65 tokens), grouped
q/k heads, and K != V. Regression guarded: with masked pointer loads, recompute_w_u
left part of u unwritten on XPU whenever K != V.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla import chunk as fla_chunk
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=150, suite="stage-b-test-1-gpu-xpu")

CHUNK_SIZE = 64
SEQ_LENS = (1, 63, 64, 65, 130)
NUM_QK_HEADS = 2
NUM_V_HEADS = 4
# (K, V): equal, K < V, K > V
HEAD_DIMS = ((128, 128), (64, 128), (128, 64))
ATOL = 2e-2
RTOL = 2e-2


def _randn(generator: torch.Generator, *shape: int) -> torch.Tensor:
    # CPU generator + copy: torch.Generator(device="xpu") is not portable across the
    # XPU torch builds CI runs.
    return torch.randn(*shape, generator=generator).to("xpu", torch.bfloat16)


def _make_inputs(key_dim: int, value_dim: int) -> dict:
    generator = torch.Generator().manual_seed(key_dim * 1000 + value_dim)
    total = sum(SEQ_LENS)
    k = _randn(generator, 1, total, NUM_QK_HEADS, key_dim)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()], device="xpu"
    )
    log_decay = -torch.rand(1, total, NUM_V_HEADS, generator=generator).to("xpu")
    return {
        "q": _randn(generator, 1, total, NUM_QK_HEADS, key_dim),
        "k": k,
        "v": _randn(generator, 1, total, NUM_V_HEADS, value_dim),
        "beta": torch.rand(1, total, NUM_V_HEADS, generator=generator).to(
            "xpu", torch.bfloat16
        ),
        "g": chunk_local_cumsum(
            log_decay * 0.1, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens
        ),
        "cu_seqlens": cu_seqlens,
    }


def _chunks():
    """(start, end, chunk_index) over every chunk of every sequence."""
    start, chunk_index = 0, 0
    for seq_len in SEQ_LENS:
        for chunk_start in range(start, start + seq_len, CHUNK_SIZE):
            yield (
                chunk_start,
                min(chunk_start + CHUNK_SIZE, start + seq_len),
                chunk_index,
            )
            chunk_index += 1
        start += seq_len


def _per_value_head(x: torch.Tensor) -> torch.Tensor:
    """[T, Hg, D] q/k rows to [T, H, D], each q/k head shared by H // Hg value heads."""
    return x.repeat_interleave(NUM_V_HEADS // NUM_QK_HEADS, dim=1)


def _reference_solved_a(inputs: dict) -> torch.Tensor:
    """(I + strict_lower(beta_i k_i.k_j exp(g_i - g_j)))^-1 per chunk, [T, H, BT]."""
    k = _per_value_head(inputs["k"][0].float())
    beta, g = inputs["beta"][0].float(), inputs["g"][0].float()
    solved = torch.zeros(k.shape[0], NUM_V_HEADS, CHUNK_SIZE, device="xpu")
    for start, end, _ in _chunks():
        n = end - start
        k_c = k[start:end].transpose(0, 1)  # [H, n, K]
        g_c = g[start:end].transpose(0, 1)  # [H, n]
        lower = (k_c @ k_c.transpose(1, 2)) * torch.exp(g_c[:, :, None] - g_c[:, None])
        lower = lower * beta[start:end].transpose(0, 1)[:, :, None]
        lower = torch.tril(lower, diagonal=-1)
        eye = torch.eye(n, device="xpu").expand_as(lower)
        solved[start:end, :, :n] = torch.linalg.inv(eye + lower).transpose(0, 1)
    return solved


def _reference_w_u(inputs: dict, solved: torch.Tensor):
    """u = A (beta v), w = A (beta exp(g) k) per chunk."""
    k = _per_value_head(inputs["k"][0].float())
    v, beta, g = inputs["v"][0].float(), inputs["beta"][0].float(), inputs["g"][0]
    w = torch.zeros(*k.shape, device="xpu")
    u = torch.zeros(*v.shape, device="xpu")
    for start, end, _ in _chunks():
        n = end - start
        a = solved[start:end, :, :n].float().transpose(0, 1)  # [H, n, n]
        scaled_v = (v[start:end] * beta[start:end, :, None]).transpose(0, 1)
        scaled_k = (
            k[start:end] * (beta[start:end] * torch.exp(g[start:end]))[:, :, None]
        )
        u[start:end] = (a @ scaled_v).transpose(0, 1)
        w[start:end] = (a @ scaled_k.transpose(0, 1)).transpose(0, 1)
    return w, u


def _reference_o(inputs: dict, h: torch.Tensor, scale: float) -> torch.Tensor:
    """scale * (exp(g_t) q_t h^T + sum_{j<=t in chunk} q_t.k_j exp(g_t - g_j) v_j)."""
    q = _per_value_head(inputs["q"][0].float())
    k = _per_value_head(inputs["k"][0].float())
    v, g = inputs["v"][0].float(), inputs["g"][0].float()
    o = torch.zeros(*v.shape, device="xpu")
    for start, end, chunk_index in _chunks():
        q_c, k_c = q[start:end].transpose(0, 1), k[start:end].transpose(0, 1)
        v_c, g_c = v[start:end].transpose(0, 1), g[start:end].transpose(0, 1)
        inter = (q_c @ h[0, chunk_index].float().transpose(1, 2)) * torch.exp(g_c)[
            :, :, None
        ]
        scores = (q_c @ k_c.transpose(1, 2)) * torch.exp(g_c[:, :, None] - g_c[:, None])
        intra = torch.tril(scores) @ v_c
        o[start:end] = ((inter + intra) * scale).transpose(0, 1)
    return o


@unittest.skipUnless(torch.xpu.is_available(), "requires Intel XPU")
class TestGdnChunkKernelsXpu(CustomTestCase):
    def assert_close(self, actual: torch.Tensor, expected: torch.Tensor, name: str):
        actual = actual.float()
        self.assertFalse(actual.isnan().any().item(), f"{name} has NaN")
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=name)

    def test_intra_chunk_solve_and_recompute_w_u(self):
        """kkt_solve (XPU) and recompute_w_u against the torch reference."""
        for key_dim, value_dim in HEAD_DIMS:
            with self.subTest(K=key_dim, V=value_dim):
                inputs = _make_inputs(key_dim, value_dim)
                w, u, solved = fla_chunk.chunk_gated_delta_rule_fwd_intra(
                    k=inputs["k"],
                    v=inputs["v"],
                    g=inputs["g"],
                    beta=inputs["beta"],
                    cu_seqlens=inputs["cu_seqlens"],
                )
                expected_solved = _reference_solved_a(inputs)
                self.assert_close(solved[0], expected_solved, "A")
                expected_w, expected_u = _reference_w_u(inputs, solved[0])
                self.assert_close(w[0], expected_w, "w")
                self.assert_close(u[0], expected_u, "u")

    def test_recompute_w_u_alone(self):
        """recompute_w_u with a random A, so a wrong w or u cannot hide behind kkt."""
        for key_dim, value_dim in HEAD_DIMS:
            with self.subTest(K=key_dim, V=value_dim):
                inputs = _make_inputs(key_dim, value_dim)
                generator = torch.Generator().manual_seed(7)
                solved = (
                    _randn(generator, 1, sum(SEQ_LENS), NUM_V_HEADS, CHUNK_SIZE) * 0.1
                )
                w, u = recompute_w_u_fwd(
                    k=inputs["k"],
                    v=inputs["v"],
                    beta=inputs["beta"],
                    g_cumsum=inputs["g"],
                    A=solved,
                    cu_seqlens=inputs["cu_seqlens"],
                )
                expected_w, expected_u = _reference_w_u(inputs, solved[0])
                self.assert_close(w[0], expected_w, "w")
                self.assert_close(u[0], expected_u, "u")

    def test_chunk_output(self):
        """chunk_fwd_o, contiguous output and output_indices row mapping."""
        for key_dim, value_dim in HEAD_DIMS:
            with self.subTest(K=key_dim, V=value_dim):
                inputs = _make_inputs(key_dim, value_dim)
                total, num_chunks = sum(SEQ_LENS), len(list(_chunks()))
                generator = torch.Generator().manual_seed(11)
                h = (
                    _randn(generator, 1, num_chunks, NUM_V_HEADS, value_dim, key_dim)
                    * 0.1
                )
                scale = key_dim**-0.5
                kwargs = dict(
                    q=inputs["q"],
                    k=inputs["k"],
                    v=inputs["v"],
                    h=h,
                    g=inputs["g"],
                    scale=scale,
                    cu_seqlens=inputs["cu_seqlens"],
                    chunk_indices=prepare_chunk_indices(
                        inputs["cu_seqlens"], CHUNK_SIZE
                    ),
                )
                expected = _reference_o(inputs, h, scale)
                self.assert_close(chunk_fwd_o(**kwargs)[0], expected, "o")

                # Rows go to a permutation of a larger buffer; unmapped rows stay put.
                destinations = torch.randperm(total + 3, generator=generator)[:total]
                out = torch.full(
                    (total + 3, NUM_V_HEADS, value_dim), 7.0, dtype=torch.bfloat16
                ).to("xpu")
                chunk_fwd_o(**kwargs, out=out, output_indices=destinations.to("xpu"))
                self.assert_close(out[destinations.to("xpu")], expected, "mapped o")
                unmapped = torch.ones(total + 3, dtype=torch.bool)
                unmapped[destinations] = False
                self.assertTrue(bool((out[unmapped.to("xpu")] == 7.0).all()))


if __name__ == "__main__":
    unittest.main()
