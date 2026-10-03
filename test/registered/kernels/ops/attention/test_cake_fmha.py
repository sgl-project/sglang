"""Cake dense FMHA / GQA / DCP decode through sglang.kernels.

Checks the registry resolves the explicit FlashInfer backend for every
``attention_fmha`` adapter, that the facade result is bitwise identical to
calling FlashInfer directly, and that each kernel matches an FP32 torch
reference within BF16 tolerance. GPU tests skip (with the reason) when the
installed FlashInfer lacks the Cake modules or the device is not sm_100a /
sm_103a. The SM110 (Thor) exports are registry-tested only: no Thor runner.
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_fmha as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_dcp_spec_decode,
    cake_fmha_batch_context_with_kv_cache,
    cake_fmha_batch_decode_with_kv_cache,
    cake_prepare_balanced_batch_decode_with_kv_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "fmha_batch_decode_with_kv_cache",
    "fmha_batch_context_with_kv_cache",
    "dcp_spec_decode",
    "prepare_balanced_batch_decode_with_kv_cache",
    "sm110_gqa_decode",
    "prepare_sm110_gqa_decode",
    "launch_sm110_gqa_decode_prepared",
    "sm110_xqa_prepare",
    "sm110_xqa_attention",
)
HEAD_DIM = 128
LOG2_E = 1.0 / math.log(2.0)


@pytest.mark.parametrize("name", OPS)
def test_registry_resolves_flashinfer_backend(name):
    spec = select_kernel(f"attention.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_fmha:")
    assert spec.load() is not None


def _skip_unless(archs, *modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in archs:
        pytest.skip(f"Cake kernel is built for {archs}, device is {cc}")


def _ceil_div(a, b):
    return -(-a // b)


def _paged_cache(seq_lens, num_kv_heads, page_size, *, gen, device, dummy=False):
    """HND BF16 K/V pools, one shuffled page table, optional trailing dummy page."""
    batch = len(seq_lens)
    pages_per = [_ceil_div(s, page_size) for s in seq_lens]
    max_pages = max(pages_per)
    total = sum(pages_per)
    num_pages = total + (1 if dummy else 0)
    k_cache = torch.randn(
        (num_pages, num_kv_heads, page_size, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    block_tables = torch.full(
        (batch, max_pages), total if dummy else 0, dtype=torch.int32, device=device
    )
    off = 0
    for b, n in enumerate(pages_per):
        block_tables[b, :n] = perm[off : off + n]
        off += n
    return k_cache, v_cache, block_tables


def _gather_kv(cache, block_tables, b, seq_len, page_size):
    """[Hkv, seq_len, D] FP32 rows of request ``b``."""
    pages = block_tables[b, : _ceil_div(seq_len, page_size)].long()
    rows = cache[pages].permute(1, 0, 2, 3).reshape(cache.shape[1], -1, HEAD_DIM)
    return rows[:, :seq_len].float()


def _reference_rows(q, k, v, *, scale, visible):
    """q [R, Hq, D]; k/v [Hq, S, D]; visible [R] -> (out [R, Hq, D], lse2 [R, Hq])."""
    scores = torch.einsum("rhd,hsd->hrs", q.float(), k) * scale
    pos = torch.arange(k.shape[1], device=q.device)
    mask = pos[None, :] < visible[:, None]  # [R, S]
    scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hrs,hsd->rhd", probs, v)
    lse2 = torch.logsumexp(scores, dim=-1).transpose(0, 1) * LOG2_E
    return out, lse2


def test_fmha_decode_bf16_matches_flashinfer_and_reference():
    _skip_unless(cake.ARCHS, cake.FI_MODULE, cake.FI_JIT_MODULE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    seq_lens = [31, 45, 200]
    batch, num_q_heads, num_kv_heads, page_size = len(seq_lens), 8, 2, 16
    query = torch.randn(
        (batch, num_q_heads, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    k_cache, v_cache, block_tables = _paged_cache(
        seq_lens, num_kv_heads, page_size, gen=gen, device=device
    )
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    workspace = torch.empty(32 << 20, dtype=torch.uint8, device=device)
    scale = HEAD_DIM**-0.5
    assert cake.supports_batch_decode_with_kv_cache(query, (k_cache, v_cache))

    out = cake_fmha_batch_decode_with_kv_cache(
        query,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens_dev,
        max(seq_lens),
        bmm1_scale=scale,
        bmm2_scale=1.0,
    )
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    out_fi = trtllm_batch_decode_with_kv_cache(
        query,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens_dev,
        max(seq_lens),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        backend="cake",
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)

    group = num_q_heads // num_kv_heads
    for b, s in enumerate(seq_lens):
        k = _gather_kv(k_cache, block_tables, b, s, page_size).repeat_interleave(
            group, dim=0
        )
        v = _gather_kv(v_cache, block_tables, b, s, page_size).repeat_interleave(
            group, dim=0
        )
        ref, _ = _reference_rows(
            query[b : b + 1],
            k,
            v,
            scale=scale,
            visible=torch.tensor([s], device=device),
        )
        torch.testing.assert_close(out[b : b + 1].float(), ref, atol=1e-2, rtol=1e-2)


def test_fmha_context_bf16_matches_flashinfer_and_reference():
    _skip_unless(cake.ARCHS, cake.FI_MODULE, cake.FI_JIT_MODULE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1)
    q_lens = [5, 7]
    kv_lens = [21, 40]
    batch, num_q_heads, num_kv_heads, page_size = len(q_lens), 8, 2, 16
    total_q = sum(q_lens)
    query = torch.randn(
        (total_q, num_q_heads, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    k_cache, v_cache, block_tables = _paged_cache(
        kv_lens, num_kv_heads, page_size, gen=gen, device=device
    )
    seq_lens_dev = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device=device)
    cum_kv = torch.tensor(
        [0, kv_lens[0], kv_lens[0] + kv_lens[1]], dtype=torch.int32, device=device
    )
    workspace = torch.empty(32 << 20, dtype=torch.uint8, device=device)
    scale = HEAD_DIM**-0.5
    assert cake.supports_batch_context_with_kv_cache(query, (k_cache, v_cache))

    kwargs = dict(
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=batch,
        cum_seq_lens_q=cum_q,
        cum_seq_lens_kv=cum_kv,
    )
    out = cake_fmha_batch_context_with_kv_cache(
        query, (k_cache, v_cache), workspace, block_tables, seq_lens_dev, **kwargs
    )
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    out_fi = trtllm_batch_context_with_kv_cache(
        query,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens_dev,
        backend="cake",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)

    group = num_q_heads // num_kv_heads
    row = 0
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        k = _gather_kv(k_cache, block_tables, b, kl, page_size).repeat_interleave(
            group, dim=0
        )
        v = _gather_kv(v_cache, block_tables, b, kl, page_size).repeat_interleave(
            group, dim=0
        )
        visible = kl - ql + 1 + torch.arange(ql, device=device)
        ref, _ = _reference_rows(
            query[row : row + ql], k, v, scale=scale, visible=visible
        )
        torch.testing.assert_close(
            out[row : row + ql].float(), ref, atol=1e-2, rtol=1e-2
        )
        row += ql


@pytest.mark.parametrize("q_len", [1, 4])
def test_balanced_gqa_decode_prepare_launch_relaunch(q_len):
    _skip_unless(cake.BALANCED_ARCHS, cake.FI_BALANCED_MODULE)
    from flashinfer.experimental.balanced_gqa_decode.cake_backend import (
        generated_program_available,
    )

    device = torch.device("cuda")
    if not generated_program_available(device, q_len):
        pytest.skip(f"no generated balanced GQA decode program for q_len={q_len}")
    gen = torch.Generator(device=device).manual_seed(2)
    seq_lens = [130, 519, 64]
    batch, num_kv_heads = len(seq_lens), 1
    num_q_heads = cake.BALANCED_GROUP_RATIO * num_kv_heads
    page_size = cake.BALANCED_PAGE_SIZE
    query = torch.randn(
        (batch * q_len, num_q_heads, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    k_cache, v_cache, block_tables = _paged_cache(
        seq_lens, num_kv_heads, page_size, gen=gen, device=device
    )
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    assert cake.supports_balanced_batch_decode_with_kv_cache(
        query, (k_cache, v_cache), block_tables, seq_lens_dev, q_len_per_req=q_len
    )
    workspace = torch.empty(
        cake.balanced_gqa_decode_workspace_size(
            device, batch=batch, max_pages=int(block_tables.shape[1])
        ),
        dtype=torch.uint8,
        device=device,
    )
    scale = HEAD_DIM**-0.5
    out = torch.full_like(query, float("nan"))
    runner = cake_prepare_balanced_batch_decode_with_kv_cache(
        query,
        (k_cache, v_cache),
        block_tables,
        seq_lens_dev,
        workspace,
        sm_scale=scale,
        q_len_per_req=q_len,
        out=out,
    )
    assert runner.launch() is out
    torch.cuda.synchronize()

    def reference():
        ref = torch.empty_like(query, dtype=torch.float32)
        for b, s in enumerate(seq_lens):
            k = _gather_kv(k_cache, block_tables, b, s, page_size).repeat_interleave(
                cake.BALANCED_GROUP_RATIO, dim=0
            )
            v = _gather_kv(v_cache, block_tables, b, s, page_size).repeat_interleave(
                cake.BALANCED_GROUP_RATIO, dim=0
            )
            visible = s - q_len + 1 + torch.arange(q_len, device=device)
            ref[b * q_len : (b + 1) * q_len], _ = _reference_rows(
                query[b * q_len : (b + 1) * q_len], k, v, scale=scale, visible=visible
            )
        return ref

    torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=1e-2)

    # Bitwise parity with FlashInfer called directly on cloned bindings.
    from flashinfer.decode import prepare_balanced_batch_decode_with_kv_cache

    out_fi = torch.full_like(query, float("nan"))
    fi_runner = prepare_balanced_batch_decode_with_kv_cache(
        query,
        (k_cache, v_cache),
        block_tables,
        seq_lens_dev,
        workspace.clone(),
        sm_scale=scale,
        q_len_per_req=q_len,
        out=out_fi,
        backend="cake",
    )
    fi_runner.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)

    # Second launch with changed contents: the runner is allocation-free.
    query.copy_(torch.randn(query.shape, generator=gen, device=device))
    k_cache.copy_(torch.randn(k_cache.shape, generator=gen, device=device))
    before = torch.cuda.memory_allocated()
    runner.launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("q_len", [1, 3])
def test_dcp_spec_decode_world1_bf16_matches_flashinfer_and_reference(q_len):
    _skip_unless(cake.ARCHS, cake.FI_DCP_MODULE, cake.FI_DCP_JIT_MODULE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(3)
    prefix = [37, 70]
    batch, num_q_heads, num_kv_heads, page_size = len(prefix), 8, 1, 16
    seq_lens = [p + q_len for p in prefix]
    query = torch.randn(
        (batch * q_len, num_q_heads, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    k_cache, v_cache, block_tables = _paged_cache(
        seq_lens, num_kv_heads, page_size, gen=gen, device=device, dummy=True
    )
    # Page-table rows padded to whole 128-token loop blocks (even count).
    loop_blocks = _ceil_div(max(seq_lens), 128)
    loop_blocks += loop_blocks % 2
    width = loop_blocks * (128 // page_size)
    padded = torch.full(
        (batch, width), int(k_cache.shape[0]) - 1, dtype=torch.int32, device=device
    )
    padded[:, : block_tables.shape[1]] = block_tables
    block_tables = padded
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prefix_dev = torch.tensor(prefix, dtype=torch.int32, device=device)
    assert cake.supports_dcp_spec_decode(
        query, (k_cache, v_cache), q_len_per_req=q_len, cp_world=1
    )
    workspace = torch.empty(
        cake.dcp_spec_workspace_size_bytes(batch, q_len, num_q_heads),
        dtype=torch.uint8,
        device=device,
    )
    counter = torch.zeros(
        cake.dcp_spec_counter_bytes(batch, q_len, num_kv_heads),
        dtype=torch.uint8,
        device=device,
    )
    scale = HEAD_DIM**-0.5
    out, lse = cake_dcp_spec_decode(
        query,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens_dev,
        max(seq_lens),
        prefix_dev,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        cp_world=1,
        cp_rank=0,
        q_len_per_req=q_len,
        multi_ctas_kv_counter_buffer=counter,
    )
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    out_fi, lse_fi = trtllm_batch_decode_with_kv_cache(
        query,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens_dev,
        max(seq_lens),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        kv_layout="HND",
        backend="cake",
        q_len_per_req=q_len,
        return_lse=True,
        multi_ctas_kv_counter_buffer=torch.zeros_like(counter),
        cp_world=1,
        cp_rank=0,
        causal_seqlens_kv_global=prefix_dev,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(lse, lse_fi)

    for b, (p, s) in enumerate(zip(prefix, seq_lens)):
        k = _gather_kv(k_cache, block_tables, b, s, page_size).repeat_interleave(
            num_q_heads // num_kv_heads, dim=0
        )
        v = _gather_kv(v_cache, block_tables, b, s, page_size).repeat_interleave(
            num_q_heads // num_kv_heads, dim=0
        )
        visible = p + 1 + torch.arange(q_len, device=device)
        ref, ref_lse2 = _reference_rows(
            query[b * q_len : (b + 1) * q_len], k, v, scale=scale, visible=visible
        )
        torch.testing.assert_close(
            out[b * q_len : (b + 1) * q_len].float(), ref, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            lse[b * q_len : (b + 1) * q_len], ref_lse2, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
