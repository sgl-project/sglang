"""CUDA correctness and CUDA Graph tests for GDN compact ragged speculative verify."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_after_verify,
)
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    ragged_verify_dense_scatter_indices,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _make_gdn_layer(
    *,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_dim: int = 128,
    conv_width: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    q_dim = num_k_heads * head_dim
    k_dim = num_k_heads * head_dim
    v_dim = num_v_heads * head_dim
    conv_dim = q_dim + k_dim + v_dim
    return SimpleNamespace(
        layer_id=0,
        num_q_heads=num_k_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_q_dim=head_dim,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        conv_dim=conv_dim,
        conv_weights=torch.randn(conv_dim, conv_width, device=device, dtype=dtype)
        * 0.1,
        bias=torch.randn(conv_dim, device=device, dtype=dtype) * 0.05,
        activation="silu",
        A_log=torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.5 - 1.0,
        dt_bias=torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.2,
    )


def _make_backend_stub(
    *,
    layer,
    pool_size: int,
    draft_token_num: int,
    ssm_dtype: torch.dtype,
    replayssm: bool,
    verify_backend: LinearAttnKernelBackend = LinearAttnKernelBackend.TRITON,
):
    conv_width = layer.conv_weights.shape[-1]
    state_width = conv_width - 1
    device = layer.conv_weights.device
    dtype = layer.conv_weights.dtype

    temporal = (
        torch.randn(
            1,
            pool_size,
            layer.num_v_heads,
            layer.head_v_dim,
            layer.head_k_dim,
            device=device,
            dtype=ssm_dtype,
        )
        * 0.05
    )
    conv = (
        torch.randn(
            1,
            pool_size,
            layer.conv_dim,
            state_width,
            device=device,
            dtype=dtype,
        )
        * 0.05
    )
    intermediate_conv = torch.zeros(
        1,
        pool_size + 1,
        draft_token_num,
        layer.conv_dim,
        state_width,
        device=device,
        dtype=dtype,
    )

    if replayssm:
        ring_len = max(16, 1 << (draft_token_num - 1).bit_length())
        intermediate_ssm = None
        replayssm_d = torch.zeros(
            1,
            pool_size + 1,
            layer.num_v_heads,
            ring_len,
            layer.head_v_dim,
            device=device,
            dtype=dtype,
        )
        replayssm_k = torch.zeros(
            1,
            pool_size + 1,
            layer.num_k_heads,
            ring_len,
            layer.head_k_dim,
            device=device,
            dtype=dtype,
        )
        replayssm_g = torch.zeros(
            1,
            pool_size + 1,
            layer.num_v_heads,
            ring_len,
            device=device,
            dtype=torch.float32,
        )
        replayssm_rawv = torch.zeros(
            1,
            pool_size + 1,
            layer.num_v_heads,
            ring_len,
            layer.head_v_dim,
            device=device,
            dtype=dtype,
        )
        replayssm_rawk = torch.zeros(
            1,
            pool_size + 1,
            layer.num_k_heads,
            ring_len,
            layer.head_k_dim,
            device=device,
            dtype=dtype,
        )
        replayssm_beta = None
        write_pos = torch.zeros(pool_size + 1, device=device, dtype=torch.int32)
        cache_base = torch.zeros(pool_size + 1, device=device, dtype=torch.int32)
        is_flush = torch.zeros(pool_size + 1, device=device, dtype=torch.int8)
    else:
        intermediate_ssm = torch.zeros(
            1,
            pool_size + 1,
            draft_token_num,
            layer.num_v_heads,
            layer.head_v_dim,
            layer.head_k_dim,
            device=device,
            dtype=ssm_dtype,
        )
        replayssm_d = replayssm_k = replayssm_g = None
        replayssm_rawv = replayssm_rawk = replayssm_beta = None
        write_pos = cache_base = is_flush = None

    spec_state = MambaPool.SpeculativeState(
        conv=[conv],
        temporal=temporal,
        intermediate_conv_window=[intermediate_conv],
        intermediate_ssm=intermediate_ssm,
        replayssm_d=replayssm_d,
        replayssm_k=replayssm_k,
        replayssm_g=replayssm_g,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_beta=replayssm_beta,
    )
    layer_cache = MambaPool.SpeculativeState(
        conv=[conv[0]],
        temporal=temporal[0],
        intermediate_conv_window=[intermediate_conv[0]],
        intermediate_ssm=None if intermediate_ssm is None else intermediate_ssm[0],
        replayssm_d=None if replayssm_d is None else replayssm_d[0],
        replayssm_k=None if replayssm_k is None else replayssm_k[0],
        replayssm_g=None if replayssm_g is None else replayssm_g[0],
        replayssm_rawv=None if replayssm_rawv is None else replayssm_rawv[0],
        replayssm_rawk=None if replayssm_rawk is None else replayssm_rawk[0],
        replayssm_beta=None if replayssm_beta is None else replayssm_beta[0],
    )

    mamba_pool = SimpleNamespace(
        replayssm_is_kda=False,
        replayssm_spec_fold=False,
        replayssm_cache_base=cache_base,
        replayssm_spec_write_pos=write_pos,
        replayssm_is_flush=is_flush,
        get_speculative_mamba2_params_all_layers=lambda: spec_state,
    )

    backend = object.__new__(GDNAttnBackend)
    backend.prefill_graph_metadata = None
    backend.mis_metadata = None
    backend._use_strided_target_verify_qkv = False
    backend.requires_contiguous_prefill_state = False
    backend.state_pool = mamba_pool
    backend.state_layer_cache = lambda _layer_id: layer_cache
    backend.req_to_token_pool = SimpleNamespace(size=pool_size)
    backend.verify_intermediate_state_indices = torch.arange(
        pool_size + 1, device=device, dtype=torch.int32
    )
    backend.kernel_dispatcher = GDNKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.TRITON,
        verify_backend,
    )
    return backend, spec_state, mamba_pool


def _run_single_seq_reference(
    *,
    layer,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    mixed_qkv_i: torch.Tensor,
    a_i: torch.Tensor,
    b_i: torch.Tensor,
    draft_token_num: int,
):
    """Run dense single-request GDN verify on one sequence of length L_i <= draft_token_num."""
    L_i = mixed_qkv_i.shape[0]
    backend, spec_state, _ = _make_backend_stub(
        layer=layer,
        pool_size=2,
        draft_token_num=L_i,
        ssm_dtype=ssm_state.dtype,
        replayssm=False,
    )
    spec_state.conv[0][0, 0].copy_(conv_state)
    spec_state.temporal[0, 0].copy_(ssm_state)

    qsl = torch.tensor([0, L_i], device="cuda", dtype=torch.int32)
    cache_idx = torch.tensor([0], device="cuda", dtype=torch.int32)
    backend.forward_metadata = ForwardMetadata(
        query_start_loc=qsl,
        mamba_cache_indices=cache_idx,
        mamba_track_indices=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=None,
        spec_info=SimpleNamespace(draft_token_num=L_i),
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
    )
    out = backend.forward_extend(layer, fb, mixed_qkv_i, a_i, b_i)
    return (
        out[0],
        spec_state.intermediate_conv_window[0][0, 0, :L_i].clone(),
        spec_state.intermediate_ssm[0, 0, :L_i].clone(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_gdn_compact_ragged_verify_recurrent_matches_per_request_ref(ssm_dtype):
    """Recurrent GDN compact ragged verify with variable lengths + padded tail matches per-request reference."""
    torch.manual_seed(17)
    layer = _make_gdn_layer()
    verify_lens = [1, 4, 8, 16]
    draft_token_num = 16
    bucket_tokens = 32  # 29 real tokens + 3 padded tail tokens
    bs = len(verify_lens)

    backend, spec_state, _ = _make_backend_stub(
        layer=layer,
        pool_size=bs + 2,
        draft_token_num=draft_token_num,
        ssm_dtype=ssm_dtype,
        replayssm=False,
    )
    init_conv = spec_state.conv[0][0, :bs].clone()
    init_ssm = spec_state.temporal[0, :bs].clone()

    qsl = torch.tensor([0, 1, 5, 13, 29], device="cuda", dtype=torch.int32)
    cache_idx = torch.arange(bs, device="cuda", dtype=torch.int32)
    mixed_qkv = (
        torch.randn(bucket_tokens, layer.conv_dim, device="cuda", dtype=torch.bfloat16)
        * 0.2
    )
    a = (
        torch.randn(
            bucket_tokens, layer.num_v_heads, device="cuda", dtype=torch.bfloat16
        )
        * 0.2
    )
    b = (
        torch.randn(
            bucket_tokens, layer.num_v_heads, device="cuda", dtype=torch.bfloat16
        )
        * 0.2
    )

    backend.forward_metadata = ForwardMetadata(
        query_start_loc=qsl,
        mamba_cache_indices=cache_idx,
        mamba_track_indices=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=SimpleNamespace(qo_indptr_device=qsl),
        spec_info=SimpleNamespace(draft_token_num=draft_token_num),
        req_pool_indices=cache_idx,
    )
    out = backend.forward_extend(layer, fb, mixed_qkv, a, b)[0]
    assert out.shape == (bucket_tokens, layer.num_v_heads, layer.head_v_dim)
    # Padded tail tokens must be finite zeros.
    assert torch.equal(out[29:], torch.zeros_like(out[29:]))

    for i, L_i in enumerate(verify_lens):
        s, e = int(qsl[i]), int(qsl[i + 1])
        ref_out, ref_conv, ref_ssm = _run_single_seq_reference(
            layer=layer,
            conv_state=init_conv[i],
            ssm_state=init_ssm[i],
            mixed_qkv_i=mixed_qkv[s:e],
            a_i=a[s:e],
            b_i=b[s:e],
            draft_token_num=draft_token_num,
        )
        torch.testing.assert_close(out[s:e], ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            spec_state.intermediate_conv_window[0][0, i, :L_i],
            ref_conv,
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            spec_state.intermediate_ssm[0, i, :L_i],
            ref_ssm,
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_gdn_compact_ragged_verify_replayssm_block16_multistep(ssm_dtype):
    """ReplaySSM GDN compact ragged verify + commit (block-16 ring, 0 intermediate_ssm) matches recurrent reference across multiple rounds."""
    torch.manual_seed(29)
    layer = _make_gdn_layer()
    draft_token_num = 16
    bs = 4

    backend, spec_state, mamba_pool = _make_backend_stub(
        layer=layer,
        pool_size=bs + 2,
        draft_token_num=draft_token_num,
        ssm_dtype=ssm_dtype,
        replayssm=True,
    )
    assert spec_state.intermediate_ssm is None

    ref_convs = [spec_state.conv[0][0, i].clone() for i in range(bs)]
    ref_ssms = [spec_state.temporal[0, i].clone() for i in range(bs)]
    cache_idx = torch.arange(bs, device="cuda", dtype=torch.int32)

    rounds = [
        ([1, 4, 8, 16], [1, 3, 5, 11], 32),
        ([16, 2, 6, 3], [9, 2, 4, 2], 32),
    ]
    for verify_lens, accept_lens_list, bucket_tokens in rounds:
        cu = [0]
        for vlen in verify_lens:
            cu.append(cu[-1] + vlen)
        total_real = cu[-1]
        qsl = torch.tensor(cu, device="cuda", dtype=torch.int32)
        mixed_qkv = (
            torch.randn(
                bucket_tokens, layer.conv_dim, device="cuda", dtype=torch.bfloat16
            )
            * 0.15
        )
        a = (
            torch.randn(
                bucket_tokens, layer.num_v_heads, device="cuda", dtype=torch.bfloat16
            )
            * 0.15
        )
        b = (
            torch.randn(
                bucket_tokens, layer.num_v_heads, device="cuda", dtype=torch.bfloat16
            )
            * 0.15
        )

        backend.forward_metadata = ForwardMetadata(
            query_start_loc=qsl,
            mamba_cache_indices=cache_idx,
            mamba_track_indices=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
            replayssm_ring_indices=cache_idx,
        )
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            ragged_verify_layout=SimpleNamespace(qo_indptr_device=qsl),
            spec_info=SimpleNamespace(draft_token_num=draft_token_num),
            req_pool_indices=cache_idx,
        )
        out = backend.forward_extend(layer, fb, mixed_qkv, a, b)[0]
        assert torch.isfinite(out).all()
        assert torch.equal(out[total_real:], torch.zeros_like(out[total_real:]))

        accept_lens = torch.tensor(accept_lens_list, device="cuda", dtype=torch.int32)
        last_step = accept_lens - 1
        commit_gdn_replayssm_after_verify(
            spec_state=spec_state,
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            state_batch_indices=cache_idx,
            replay_indices=cache_idx,
            accept_lens=accept_lens,
            last_correct_step_indices=last_step,
            fold_every_commit=True,
            src_indices_raw=cache_idx,
            null_block_id=-1,
        )

        tol = 2e-2 if ssm_dtype == torch.bfloat16 else 1e-2
        for i, (L_i, acc_i) in enumerate(zip(verify_lens, accept_lens_list)):
            s, e = cu[i], cu[i + 1]
            ref_out, ref_conv, ref_ssm = _run_single_seq_reference(
                layer=layer,
                conv_state=ref_convs[i],
                ssm_state=ref_ssms[i],
                mixed_qkv_i=mixed_qkv[s:e],
                a_i=a[s:e],
                b_i=b[s:e],
                draft_token_num=draft_token_num,
            )
            torch.testing.assert_close(out[s:e], ref_out, rtol=tol, atol=tol)
            ref_convs[i] = ref_conv[acc_i - 1].clone()
            ref_ssms[i] = ref_ssm[acc_i - 1].clone()
            torch.testing.assert_close(
                spec_state.conv[0][0, i], ref_convs[i], rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                spec_state.temporal[0, i], ref_ssms[i], rtol=tol, atol=tol
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gdn_compact_ragged_verify_replayssm_cuda_graph():
    """Captured ragged verify CUDA Graph replays accurately with changing per-request lengths."""
    torch.manual_seed(41)
    layer = _make_gdn_layer()
    draft_token_num = 16
    bs = 4
    bucket_tokens = 24

    backend, spec_state, _ = _make_backend_stub(
        layer=layer,
        pool_size=bs + 2,
        draft_token_num=draft_token_num,
        ssm_dtype=torch.float32,
        replayssm=True,
    )
    init_conv = spec_state.conv[0][0, :bs].clone()
    init_ssm = spec_state.temporal[0, :bs].clone()

    qsl_buf = torch.tensor([0, 2, 7, 8, 16], device="cuda", dtype=torch.int32)
    cache_idx = torch.arange(bs, device="cuda", dtype=torch.int32)
    mixed_qkv_buf = (
        torch.randn(bucket_tokens, layer.conv_dim, device="cuda", dtype=torch.bfloat16)
        * 0.15
    )
    a_buf = (
        torch.randn(
            bucket_tokens, layer.num_v_heads, device="cuda", dtype=torch.bfloat16
        )
        * 0.15
    )
    b_buf = (
        torch.randn(
            bucket_tokens, layer.num_v_heads, device="cuda", dtype=torch.bfloat16
        )
        * 0.15
    )

    backend.forward_metadata = ForwardMetadata(
        query_start_loc=qsl_buf,
        mamba_cache_indices=cache_idx,
        mamba_track_indices=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        replayssm_ring_indices=cache_idx,
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=SimpleNamespace(qo_indptr_device=qsl_buf),
        spec_info=SimpleNamespace(draft_token_num=draft_token_num),
        req_pool_indices=cache_idx,
    )

    # Warmup & capture (fresh ForwardMetadata for capture so hoisted scatter
    # indices are recorded inside the captured graph, matching _capture_metadata).
    _ = backend.forward_extend(layer, fb, mixed_qkv_buf, a_buf, b_buf)
    torch.cuda.synchronize()
    backend.forward_metadata.ragged_dense_token_indices = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf = backend.forward_extend(layer, fb, mixed_qkv_buf, a_buf, b_buf)
    torch.cuda.synchronize()

    # Replay with completely different ragged lengths inside the same static graph
    for lens in ([2, 5, 1, 8], [12, 1, 3, 4]):
        cu = [0]
        for vlen in lens:
            cu.append(cu[-1] + vlen)
        qsl_buf.copy_(torch.tensor(cu, device="cuda", dtype=torch.int32))
        mixed_qkv_buf.normal_(0.0, 0.15)
        a_buf.normal_(0.0, 0.15)
        b_buf.normal_(0.0, 0.15)
        spec_state.conv[0][0, :bs].copy_(init_conv)
        spec_state.temporal[0, :bs].copy_(init_ssm)

        g.replay()
        torch.cuda.synchronize()

        out = out_buf[0]
        assert torch.equal(out[cu[-1] :], torch.zeros_like(out[cu[-1] :]))
        for i, L_i in enumerate(lens):
            s, e = cu[i], cu[i + 1]
            ref_out, _, _ = _run_single_seq_reference(
                layer=layer,
                conv_state=init_conv[i],
                ssm_state=init_ssm[i],
                mixed_qkv_i=mixed_qkv_buf[s:e],
                a_i=a_buf[s:e],
                b_i=b_buf[s:e],
                draft_token_num=draft_token_num,
            )
            torch.testing.assert_close(out[s:e], ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gdn_dispatcher_flashinfer_falls_back_to_triton_for_ragged_layout():
    """FlashInfer GDN verify (rectangular) falls back to varlen Triton when ragged_layout is set."""
    dispatcher = GDNKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.TRITON,
    )
    sentinel_fi = object()
    dispatcher.verify_kernel = sentinel_fi
    dispatcher.verify_kernel_is_flashinfer = True

    assert dispatcher._get_target_verify_kernel(None, ragged_layout=None) is sentinel_fi
    assert (
        dispatcher._get_target_verify_kernel(None, ragged_layout=object())
        is dispatcher.tree_verify_kernel
    )
    assert ragged_verify_dense_scatter_indices(
        query_start_loc=torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32),
        seq_len=5,
        draft_token_num=4,
    ).tolist() == [0, 1, 4, 8, 8]
