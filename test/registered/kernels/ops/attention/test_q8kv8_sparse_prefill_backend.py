"""DeepSeek-V4 Q8KV8 sparse-prefill backend helper tests.

These tests avoid starting a full server. They construct the minimum V4
metadata and token-pool surface consumed by the sparse-prefill helpers, then
compare the BF16 sparse path's gathered workspace against the Q8 path's FP8
workspace after dequantizing it back to BF16.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.index_buf_accessor import SetKAndS
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillWorkspace,
    use_dsv4_q8kv8_sparse_prefill,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_q8kv8_sparse_prefill_backend_selector_uses_cli_value():
    assert not use_dsv4_q8kv8_sparse_prefill()
    assert not use_dsv4_q8kv8_sparse_prefill("auto")
    assert not use_dsv4_q8kv8_sparse_prefill("flashmla_sparse")
    assert use_dsv4_q8kv8_sparse_prefill("flashmla_sparse_q8")


class _Pool:
    def __init__(self, page_size: int):
        self.page_size = page_size


class _Capture:
    def __init__(self):
        self.calls = []

    def record(self, **kwargs):
        cloned = {}
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                cloned[name] = value.detach().clone()
            else:
                cloned[name] = value
        self.calls.append(cloned)


class _TokenToKVPool:
    def __init__(
        self,
        *,
        swa_key_buffer: torch.Tensor,
        full_to_swa_index_mapping: torch.Tensor,
        page_size: int,
    ):
        self._swa_key_buffer = swa_key_buffer
        self.full_to_swa_index_mapping = full_to_swa_index_mapping
        self.swa_window_size = page_size

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        _ = layer_id
        return self._swa_key_buffer


def _sm90_available() -> bool:
    return torch.cuda.is_available() and is_sm90_supported()


def _make_v4_paged_kv_cache(
    *,
    total_slots: int,
    page_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_pages = (total_slots + page_size - 1) // page_size
    total_slots = num_pages * page_size
    bytes_per_token = 448 + 64 * 2 + 8
    quant_k_cache = torch.zeros(
        num_pages,
        page_size * bytes_per_token,
        dtype=torch.uint8,
        device=device,
    )

    k_bf16 = (torch.randn(total_slots, 512, device=device) * 0.25).to(torch.bfloat16)
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
    loc = torch.arange(total_slots, dtype=torch.int32, device=device)
    SetKAndS.torch(_Pool(page_size), quant_k_cache, loc, pack)
    return quant_k_cache


def _make_forward_batch_and_mapping(
    device: torch.device,
) -> tuple[ForwardBatch, torch.Tensor]:
    seq_lens = torch.tensor([96, 144], dtype=torch.int32, device=device)
    extend_seq_lens = torch.tensor([3, 2], dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seq0 = int(seq_lens[0].item())
    seq1 = int(seq_lens[1].item())

    req_to_token = torch.zeros(
        (2, int(seq_lens.max().item())), dtype=torch.int32, device=device
    )
    req_to_token[0, :seq0] = torch.arange(seq0, dtype=torch.int32, device=device)
    req1_base = 192
    req_to_token[1, :seq1] = req1_base + torch.arange(
        seq1, dtype=torch.int32, device=device
    )

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.zeros(
            int(extend_seq_lens.sum().item()), dtype=torch.int32, device=device
        ),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=torch.zeros(
            int(extend_seq_lens.sum().item()), dtype=torch.int32, device=device
        ),
        seq_lens_sum=int(seq_lens.sum().item()),
        seq_lens_cpu=seq_lens.detach().cpu(),
        extend_num_tokens=int(extend_seq_lens.sum().item()),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=[int(x) for x in extend_seq_lens.detach().cpu().tolist()],
    )
    return forward_batch, req_to_token


def _make_backend(
    device: torch.device,
    req_to_token: torch.Tensor,
    dsv4_prefill_backend: str = "auto",
) -> DeepseekV4AttnBackend:
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(sparse_prefill_cache=None)
    backend.req_to_token = req_to_token
    backend.sparse_prefill_workspace = SparsePrefillWorkspace(device)
    backend.softmax_scale = 512**-0.5
    backend.head_dim_v = 512
    backend.dsv4_prefill_backend = dsv4_prefill_backend
    return backend


def _make_sparse_prefill_case(
    device: torch.device,
    local_heads: int = 64,
):
    page_size = 64
    total_slots = 384
    forward_batch, req_to_token = _make_forward_batch_and_mapping(device)
    backend = _make_backend(device, req_to_token)
    quant_k_cache = _make_v4_paged_kv_cache(
        total_slots=total_slots,
        page_size=page_size,
        seed=3,
        device=device,
    )
    token_to_kv_pool = _TokenToKVPool(
        swa_key_buffer=quant_k_cache,
        full_to_swa_index_mapping=torch.arange(
            total_slots, dtype=torch.int64, device=device
        ),
        page_size=page_size,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(11)
    q = (
        torch.randn(
            forward_batch.extend_num_tokens,
            1,
            local_heads,
            512,
            device=device,
            generator=generator,
        )
        * 0.05
    ).to(torch.bfloat16)
    attn_sink = torch.zeros(local_heads, dtype=torch.float32, device=device)
    core_attn_metadata = SimpleNamespace()
    return backend, forward_batch, token_to_kv_pool, q, attn_sink, core_attn_metadata


@contextmanager
def _patched_sparse_kernels(
    bf16_capture: _Capture,
    q8_capture: _Capture,
):
    def fake_flash_mla_sparse_fwd(
        *,
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
    ):
        bf16_capture.record(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        out = torch.zeros(
            (q.shape[0], q.shape[1], d_v), dtype=torch.bfloat16, device=q.device
        )
        meta = torch.zeros(
            (q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device
        )
        return out, meta, meta

    def fake_sparse_mla_q8kv8_prefill_fwd(
        *,
        q,
        kv,
        indices,
        sm_scale,
        q_scale,
        kv_scale,
        d_v,
        attn_sink,
        topk_length,
    ):
        q8_capture.record(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        out = torch.zeros(
            (q.shape[0], q.shape[1], d_v), dtype=torch.bfloat16, device=q.device
        )
        meta = torch.zeros(
            (q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device
        )
        return out, meta, meta

    sgl_kernel_pkg = types.ModuleType("sgl_kernel")
    flash_mla_mod = types.ModuleType("sgl_kernel.flash_mla")
    flash_mla_mod.flash_mla_sparse_fwd = fake_flash_mla_sparse_fwd
    sgl_kernel_pkg.flash_mla = flash_mla_mod

    q8_mod = types.ModuleType("sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90")
    q8_mod.sparse_mla_q8kv8_prefill_fwd = fake_sparse_mla_q8kv8_prefill_fwd

    old_modules = {
        name: sys.modules.get(name)
        for name in (
            "sgl_kernel",
            "sgl_kernel.flash_mla",
            "sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90",
        )
    }
    sys.modules["sgl_kernel"] = sgl_kernel_pkg
    sys.modules["sgl_kernel.flash_mla"] = flash_mla_mod
    sys.modules["sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90"] = q8_mod
    try:
        yield
    finally:
        for name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_q8kv8_sparse_prefill_helper_builds_fp8_workspace_matching_bf16_path():
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import fp8_dtype

    device = torch.device("cuda")
    backend, forward_batch, token_to_kv_pool, q, attn_sink, core_attn_metadata = (
        _make_sparse_prefill_case(device, local_heads=16)
    )

    bf16_capture = _Capture()
    q8_capture = _Capture()
    with _patched_sparse_kernels(bf16_capture, q8_capture):
        bf16_out = backend._forward_prefill_sparse(
            q=q,
            layer_id=0,
            compress_ratio=0,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            core_attn_metadata=core_attn_metadata,
            attn_sink=attn_sink,
        )
        sparse_cache = backend.forward_metadata.sparse_prefill_cache
        q8_out = backend._forward_prefill_sparse_q8kv8(
            q=q,
            layer_id=0,
            compress_ratio=0,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            core_attn_metadata=core_attn_metadata,
            attn_sink=attn_sink,
        )

    assert backend.forward_metadata.sparse_prefill_cache is sparse_cache
    assert (
        bf16_out.shape
        == q8_out.shape
        == (
            forward_batch.extend_num_tokens,
            16,
            512,
        )
    )
    assert len(bf16_capture.calls) == 1
    assert len(q8_capture.calls) == 1

    bf16_call = bf16_capture.calls[0]
    q8_call = q8_capture.calls[0]
    bf16_kv = bf16_call["kv"]
    q8_kv = q8_call["kv"]

    assert bf16_kv.dtype == torch.bfloat16
    assert q8_kv.dtype == fp8_dtype
    assert q8_call["q"].dtype == fp8_dtype
    assert q8_call["q"].shape[1] == 64
    assert torch.count_nonzero(q8_call["q"][:, 16:]).item() == 0
    assert q8_call["attn_sink"].shape == (64,)
    assert q8_kv.shape[0] == bf16_kv.shape[0] + 1
    torch.testing.assert_close(
        q8_kv[:-1].to(torch.bfloat16).float(),
        bf16_kv.float(),
        atol=3e-2,
        rtol=2e-1,
    )
    assert torch.equal(
        q8_kv[-1].to(torch.bfloat16),
        torch.zeros_like(q8_kv[-1].to(torch.bfloat16)),
    )

    bf16_indices = bf16_call["indices"]
    q8_indices = q8_call["indices"]
    sentinel_row = q8_kv.shape[0] - 1
    valid_mask = bf16_indices >= 0
    assert torch.equal(q8_indices[valid_mask], bf16_indices[valid_mask])
    assert torch.equal(
        q8_indices[~valid_mask],
        torch.full_like(q8_indices[~valid_mask], sentinel_row),
    )
    assert torch.equal(q8_call["topk_length"], bf16_call["topk_length"])
    assert q8_call["q_scale"].item() == pytest.approx(1.0)
    assert q8_call["kv_scale"].item() == pytest.approx(1.0)


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
def test_q8kv8_sparse_prefill_real_kernel_matches_bf16_sparse_path():
    device = torch.device("cuda")
    backend, forward_batch, token_to_kv_pool, q, attn_sink, core_attn_metadata = (
        _make_sparse_prefill_case(device, local_heads=64)
    )

    bf16_out = backend._forward_prefill_sparse(
        q=q,
        layer_id=0,
        compress_ratio=0,
        forward_batch=forward_batch,
        token_to_kv_pool=token_to_kv_pool,
        core_attn_metadata=core_attn_metadata,
        attn_sink=attn_sink,
    )
    sparse_cache = backend.forward_metadata.sparse_prefill_cache
    q8_out = backend._forward_prefill_sparse_q8kv8(
        q=q,
        layer_id=0,
        compress_ratio=0,
        forward_batch=forward_batch,
        token_to_kv_pool=token_to_kv_pool,
        core_attn_metadata=core_attn_metadata,
        attn_sink=attn_sink,
    )
    torch.cuda.synchronize()

    assert backend.forward_metadata.sparse_prefill_cache is sparse_cache
    assert (
        bf16_out.shape
        == q8_out.shape
        == (
            forward_batch.extend_num_tokens,
            64,
            512,
        )
    )
    assert bf16_out.dtype == torch.bfloat16
    assert q8_out.dtype == torch.bfloat16
    assert torch.isfinite(bf16_out.float()).all()
    assert torch.isfinite(q8_out.float()).all()

    abs_diff = (q8_out.float() - bf16_out.float()).abs()
    assert abs_diff.mean().item() < 0.03
    assert torch.quantile(abs_diff.flatten(), 0.99).item() < 0.2
    torch.testing.assert_close(
        q8_out.float(),
        bf16_out.float(),
        atol=2.5e-1,
        rtol=3.0e-1,
    )
