"""Numerical coverage of the real Triton kernels in an isolated CPU interpreter.

Run with ``python3 -m pytest -q test/registered/unit/npu/attention/test_npu_mojo_mtp_kernel.py``.
The child process sets TRITON_INTERPRET before importing Triton, leaving other
tests' compiler settings untouched. This does not replace NPU hardware tests.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def test_mojo_paged_attention_in_triton_interpreter():
    pytest.importorskip("triton")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env={
            **os.environ,
            "TRITON_INTERPRET": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "numeric cases passed" in result.stdout


def run_numeric_cases():
    import math

    import torch

    # The kernel ships in sgl-kernel-npu, not in this tree.
    from sgl_kernel_npu.attention import mojo_mtp as module
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(24)
    cases = 0
    # Non-power-of-two query/head groups, empty requests, split tails and page
    # boundaries. Unused physical pages and KV tails are poisoned with NaNs.
    for gamma, group, dim, page, lengths in (
        (1, 1, 16, 16, [0, 1, 17]),
        (7, 3, 48, 16, [7, 15, 17]),
        (7, 4, 16, 128, [0, 127, 134]),
        (8, 2, 32, 32, [31, 32, 33]),
        (7, 8, 128, 128, [0, 128, 262]),
        (1, 1, 64, 128, [1, 129, 1025]),
        (7, 1, 128, 32, [0, 7, 65]),
        (8, 4, 256, 128, [8, 127, 256]),
    ):
        kv_heads = 2
        heads = kv_heads * group
        batch_size = len(lengths)
        pages_per_req = math.ceil(max(lengths) / page) + 1
        total_pages = batch_size * pages_per_req
        pages = (
            torch.randperm(total_pages, generator=generator)
            .reshape(batch_size, pages_per_req)
            .int()
        )
        keys = torch.full(
            (total_pages, page, kv_heads * dim), float("nan"), dtype=torch.float16
        )
        values = keys.clone()
        query = torch.randn(
            batch_size * gamma, heads, dim, generator=generator, dtype=torch.float16
        )
        logical = []
        for b, length in enumerate(lengths):
            k = torch.randn(
                length, kv_heads, dim, generator=generator, dtype=torch.float16
            )
            v = torch.randn(
                length, kv_heads, dim, generator=generator, dtype=torch.float16
            )
            logical.append((k, v))
            for token in range(length):
                physical = int(pages[b, token // page])
                keys[physical, token % page] = k[token].flatten()
                values[physical, token % page] = v[token].flatten()
            if length == 0:
                pages[b].fill_(-1)  # An inactive row must not dereference any page.

        for causal in (False, True):
            reference = torch.zeros_like(query)
            for b, length in enumerate(lengths):
                k, v = logical[b]
                k = k.repeat_interleave(group, dim=1).float()
                v = v.repeat_interleave(group, dim=1).float()
                for row in range(gamma):
                    end = max(0, length - gamma + row + 1) if causal else length
                    if end:
                        q = query[b * gamma + row].float()
                        scores = torch.einsum("hd,khd->hk", q, k[:end]) / math.sqrt(dim)
                        reference[b * gamma + row] = torch.einsum(
                            "hk,khd->hd", scores.softmax(-1), v[:end]
                        ).half()
            for split in (False, True):
                module._get_num_cores = lambda: (32, 8) if split else (1, 1)
                kwargs = dict(
                    query=query,
                    k_cache=keys,
                    v_cache=values,
                    seq_lens=torch.tensor(lengths, dtype=torch.int32),
                    cu_q_lens=torch.arange(
                        0, (batch_size + 1) * gamma, gamma, dtype=torch.int32
                    ),
                    block_tables=pages,
                    page_size=page,
                    num_q_heads=heads,
                    num_kv_heads=kv_heads,
                    head_dim=dim,
                    gamma=gamma,
                    softmax_scale=dim**-0.5,
                    max_kv_len=4096,
                    causal=causal,
                )
                output = module.mojo_paged_mtp_decode(**kwargs)
                assert torch.isfinite(output).all(), (gamma, group, causal, split)
                torch.testing.assert_close(output, reference, atol=0.003, rtol=0.02)
                # Same storage, different lengths: emulate graph replay inputs.
                kwargs["seq_lens"].zero_()
                output = module.mojo_paged_mtp_decode(**kwargs)
                assert torch.count_nonzero(output) == 0
                cases += 2
    print(f"{cases} numeric cases passed")


if __name__ == "__main__":
    run_numeric_cases()
