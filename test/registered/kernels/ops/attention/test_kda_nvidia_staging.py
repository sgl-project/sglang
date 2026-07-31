import pytest
import torch
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.linear.kernels.kda_nvidia import NvidiaKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_nvidia_staging import (
    gather_nvidia_kda_state,
    pack_nvidia_kda_inputs,
    scatter_nvidia_kda_state,
    unpack_nvidia_kda_output,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="NVIDIA KDA prefill requires datacenter Blackwell",
)


def _staging(rows: int, bucket: int, heads: int, dim: int):
    return {
        "q": torch.empty(rows, bucket, heads, dim, device="cuda", dtype=torch.bfloat16),
        "k": torch.empty(rows, bucket, heads, dim, device="cuda", dtype=torch.bfloat16),
        "v": torch.empty(rows, bucket, heads, dim, device="cuda", dtype=torch.bfloat16),
        "g": torch.empty(rows, bucket, heads, dim, device="cuda", dtype=torch.bfloat16),
        "beta": torch.empty(rows, bucket, heads, device="cuda", dtype=torch.bfloat16),
    }


def test_fused_input_and_output_staging_matches_reference():
    torch.manual_seed(32541)
    lengths = [1, 17, 128]
    rows, bucket, heads, dim = len(lengths), 256, 2, 128
    total = sum(lengths)
    q = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.rand(1, total, heads, device="cuda", dtype=torch.float32)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    expected = _staging(rows, bucket, heads, dim)
    for tensor in expected.values():
        tensor.zero_()
    expected["g"].fill_(-1000.0)
    start = 0
    for row, length in enumerate(lengths):
        end = start + length
        expected["q"][row, :length] = l2norm_fwd(q[0, start:end].contiguous())
        expected["k"][row, :length] = l2norm_fwd(k[0, start:end].contiguous())
        expected["v"][row, :length] = v[0, start:end]
        expected["g"][row, :length] = g[0, start:end]
        expected["beta"][row, :length] = beta[0, start:end]
        start = end

    actual = _staging(rows, bucket, heads, dim)
    pack_nvidia_kda_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        seq_start=0,
        group_size=rows,
        staging=actual,
    )
    torch.cuda.synchronize()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1e-3)

    source = torch.randn_like(actual["v"])
    output = torch.zeros_like(v)
    unpack_nvidia_kda_output(source, output, cu, seq_start=0)
    expected_output = torch.cat(
        [source[row, :length] for row, length in enumerate(lengths)], dim=0
    ).unsqueeze(0)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


def test_fused_state_staging_supports_envelope_strides():
    torch.manual_seed(32542)
    slots, rows, heads, dim = 7, 3, 2, 128
    elements_per_slot = heads * dim * dim
    storage = torch.randn(
        slots * (elements_per_slot + 1024), device="cuda", dtype=torch.float32
    )
    pool = torch.as_strided(
        storage,
        size=(slots, heads, dim, dim),
        stride=(elements_per_slot + 1024, dim * dim, dim, 1),
    )
    pool_before = pool.clone()
    selected = torch.tensor([5, 1, 3], device="cuda", dtype=torch.int64)
    vendor_state = torch.empty(
        rows, heads, dim, dim, device="cuda", dtype=torch.float32
    )

    gather_nvidia_kda_state(pool, selected, vendor_state)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        vendor_state, pool[selected].transpose(-1, -2), rtol=0, atol=0
    )

    vendor_state.add_(1.0)
    scatter_nvidia_kda_state(vendor_state, selected, pool)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        pool[selected].transpose(-1, -2), vendor_state, rtol=0, atol=0
    )
    untouched = torch.ones(slots, device="cuda", dtype=torch.bool)
    untouched[selected] = False
    torch.testing.assert_close(pool[untouched], pool_before[untouched], rtol=0, atol=0)


def test_real_nvidia_kda_prefill_matches_triton_tp4():
    torch.manual_seed(32543)
    batch, length, heads, dim = 4, 1024, 24, 128
    total = batch * length
    q = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q) * 0.1
    beta = torch.sigmoid(
        torch.randn(1, total, heads, device="cuda", dtype=torch.float32)
    )
    a_log = torch.randn(1, 1, heads, 1, device="cuda", dtype=torch.float32) * 0.5 - 1.5
    dt_bias = torch.randn(heads * dim, device="cuda", dtype=torch.float32) * 0.1
    cu = torch.arange(0, total + 1, length, device="cuda", dtype=torch.int32)
    slots = torch.tensor([5, 1, 7, 3], device="cuda", dtype=torch.int32)
    state_seed = (
        torch.randn(9, heads, dim, dim, device="cuda", dtype=torch.float32) * 0.01
    )
    common = {
        "cache_indices": slots,
        "query_start_loc": cu,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": -5.0,
        "extend_seq_lens_cpu": [length] * batch,
        "is_spec_decode": False,
        "return_intermediate_states": False,
    }

    ref_state = state_seed.clone()
    ref = TritonKDAKernel().extend(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        ssm_states=ref_state,
        **common,
    )
    actual_state = state_seed.clone()
    actual = NvidiaKDAKernel().extend(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        ssm_states=actual_state,
        **common,
    )
    torch.cuda.synchronize()

    output_cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), actual.float().flatten(), dim=0
    )
    state_cos = torch.nn.functional.cosine_similarity(
        ref_state[slots.long()].flatten(),
        actual_state[slots.long()].flatten(),
        dim=0,
    )
    assert output_cos > 0.999
    assert state_cos > 0.9999
    assert torch.equal(
        ref_state[torch.tensor([0, 2, 4, 6, 8], device="cuda", dtype=torch.long)],
        actual_state[torch.tensor([0, 2, 4, 6, 8], device="cuda", dtype=torch.long)],
    )
