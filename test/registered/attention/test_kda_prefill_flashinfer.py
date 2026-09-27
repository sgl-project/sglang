"""FlashInfer KDA prefill integration against SGLang's Triton reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
):
    pytest.skip("FlashInfer KDA prefill requires B200/B300", allow_module_level=True)

flashinfer = pytest.importorskip("flashinfer")
pytest.importorskip("flashinfer.kda")
if Version(flashinfer.__version__) < Version("0.7.0"):
    pytest.skip(
        "KDA prefill checkpoints require FlashInfer 0.7.0", allow_module_level=True
    )

from sglang.srt.layers.attention.linear.kda_backend import (  # noqa: E402
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (  # noqa: E402
    FlashInferKDAPrefillKernel,
    build_flashinfer_kda_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)
from sglang.srt.layers.attention.linear.utils import (  # noqa: E402
    LinearAttnKernelBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.kits.attention_unittest.attention_methods.kda_attention import (  # noqa: E402
    KDAAttentionCase,
    build_kda_attention_fixture,
    run_kda_fixture_eager,
)


@pytest.mark.parametrize(
    "state_dtype,padded_slots,strided_beta",
    [
        (torch.bfloat16, False, False),
        (torch.float32, False, False),
        (torch.bfloat16, True, False),
        (torch.float32, False, True),
    ],
)
def test_kda_prefill_indexed_state_and_130_token_checkpoint(
    state_dtype, padded_slots, strided_beta
):
    torch.manual_seed(7)
    lengths = [130, 128]
    total_tokens = sum(lengths)
    heads = 12
    dim = 128

    def randn(*shape, scale=1):
        return (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
        ).contiguous()

    q = randn(1, total_tokens, heads, dim, scale=0.01)
    k = randn(1, total_tokens, heads, dim, scale=0.01)
    v = randn(1, total_tokens, heads, dim, scale=0.01)
    g = randn(1, total_tokens + 14, heads, dim, scale=0.1)
    beta = (
        randn(1, total_tokens + 14, heads * 12)[..., :heads]
        if strided_beta
        else randn(1, total_tokens + 14, heads)
    )
    a_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(heads, dim, device="cuda", dtype=torch.float32)
    initial_values = (
        torch.randn(4, heads, dim, dim, device="cuda", dtype=torch.float32) * 0.01
    ).to(state_dtype)
    initial = initial_values
    if padded_slots:
        fi_state = torch.empty_strided(
            initial_values.shape,
            (heads * dim * dim + 16, dim * dim, dim, 1),
            device="cuda",
            dtype=state_dtype,
        )
        fi_state.copy_(initial_values)
    else:
        fi_state = initial.clone()
    cu_seqlens = torch.tensor([0, 130, total_tokens], device="cuda", dtype=torch.int32)
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

    forward_batch = SimpleNamespace(
        extend_seq_lens_cpu=lengths,
        extend_seq_lens=None,
        mamba_track_seqlens_cpu=lengths,
        mamba_track_seqlens=None,
        extend_prefix_lens_cpu=[0, 0],
        extend_prefix_lens=None,
        mamba_prefill_track_mask_cpu=[True, True],
        mamba_track_mask=None,
    )
    metadata = SimpleNamespace(
        track_ssm_h_src=torch.tensor([2], device="cuda"),
        track_ssm_h_batch_src=torch.tensor([0], device="cuda"),
    )
    build_flashinfer_kda_checkpoint_plan(forward_batch, metadata, "cuda", 64)
    assert metadata.state_checkpoint_cu_starts.tolist() == [0, 2, 4]
    assert metadata.state_checkpoint_track_src.tolist() == [1]
    assert metadata.state_checkpoint_indices.tolist() == [0, 1, 2, 3]

    triton = TritonKDAKernel()
    flashinfer = FlashInferKDAPrefillKernel(triton)
    ref_state = initial.clone()
    ref_track = torch.full(
        (2, heads, dim, dim), torch.nan, device="cuda", dtype=torch.float32
    )
    fi_track = torch.full_like(ref_track, torch.nan)
    common = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        beta_is_raw=True,
        return_intermediate_states=True,
        extend_seq_lens_cpu=lengths,
        track_chunk_idx=torch.tensor([2, -1], device="cuda", dtype=torch.int32),
    )
    ref_output, _ = triton.extend(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        ssm_states=ref_state,
        cache_indices=slots,
        query_start_loc=cu_seqlens,
        track_state=ref_track,
        **common,
    )
    with patch.object(triton, "extend", side_effect=AssertionError("Triton fallback")):
        fi_output, _ = flashinfer.extend(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta,
            ssm_states=fi_state,
            cache_indices=slots,
            query_start_loc=cu_seqlens,
            track_state=fi_track,
            state_checkpoint_cu_starts=metadata.state_checkpoint_cu_starts,
            num_state_checkpoints=metadata.num_state_checkpoints,
            state_checkpoint_every_n_tokens=metadata.state_checkpoint_every_n_tokens,
            state_checkpoint_track_src=metadata.state_checkpoint_track_src,
            state_checkpoint_indices=metadata.state_checkpoint_indices,
            track_ssm_h_batch_src=metadata.track_ssm_h_batch_src,
            **common,
        )

    assert torch.isfinite(fi_output).all()
    assert torch.isfinite(fi_state).all()
    assert torch.isfinite(fi_track[0]).all()
    assert torch.isnan(fi_track[1]).all()
    assert (fi_output.float() - ref_output.float()).abs().max() < 1e-2
    assert (fi_state.float() - ref_state.float()).abs().max() < 1e-2
    relative_track_error = torch.linalg.vector_norm(fi_track[0] - ref_track[0]) / (
        torch.linalg.vector_norm(ref_track[0]) + 1e-12
    )
    assert relative_track_error < 5e-2

    # The reusable state for 130 tokens is the state after token 128.
    prefix_state = initial[1:2].clone()
    flashinfer.extend(
        q[:, :128].clone(),
        k[:, :128].clone(),
        v[:, :128].clone(),
        g[:, :128].clone(),
        beta[:, :128].clone(),
        ssm_states=prefix_state,
        cache_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor([0, 128], device="cuda", dtype=torch.int32),
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        beta_is_raw=True,
        extend_seq_lens_cpu=[128],
    )
    torch.testing.assert_close(fi_track[0], prefix_state[0].float(), atol=0, rtol=0)


@pytest.fixture
def single_dcp_rank():
    with get_parallel().override(attn_dcp_rank=0, attn_dcp_size=1):
        yield


def test_kda_backend_prefill_dispatch_and_tracked_state(single_dcp_rank):
    case = KDAAttentionCase(
        name="flashinfer_kda_tracked_extend",
        backend="triton",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=2,
        num_v_heads=2,
        page_size=16,
        prefix_lens=(0, 0),
        extend_lens=(130, 128),
    )
    fixture = build_kda_attention_fixture(
        unittest.TestCase(),
        case,
        head_k_dim=128,
        head_v_dim=128,
        max_context_len=256,
        runner_batch_size=6,
    )
    batch = fixture.forward_batch
    batch.mamba_track_mask = torch.tensor([True, True], device="cuda")
    batch.mamba_track_indices = torch.tensor([4, 5], device="cuda", dtype=torch.int32)
    batch.mamba_track_seqlens = torch.tensor(
        [130, 128], device="cuda", dtype=torch.int32
    )
    batch.mamba_prefill_track_mask_cpu = [True, True]
    batch.mamba_track_seqlens_cpu = [130, 128]
    fixture.actual_module.attn.lower_bound = -5.0
    cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
    initial_conv, initial_ssm = cache.conv[0].clone(), cache.temporal.clone()

    triton_output = run_kda_fixture_eager(fixture)
    triton_state = cache.temporal.clone()

    cache.conv[0].copy_(initial_conv)
    cache.temporal.copy_(initial_ssm)
    fixture.b = fixture.b_raw.unsqueeze(0)  # FlashInfer consumes beta logits.
    fixture.backend.linear_attn_backend.kernel_dispatcher = KDAKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.FLASHINFER,
        LinearAttnKernelBackend.TRITON,
    )
    assert isinstance(
        fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel,
        FlashInferKDAPrefillKernel,
    )
    flashinfer_output = run_kda_fixture_eager(fixture)

    torch.testing.assert_close(
        flashinfer_output.float(), triton_output.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        cache.temporal.float(), triton_state.float(), atol=3e-2, rtol=3e-2
    )
