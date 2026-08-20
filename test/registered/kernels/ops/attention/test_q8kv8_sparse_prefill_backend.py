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
from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import (
    sparse_mla_q8kv8_prefill_fwd,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
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
        extra_key_buffer: torch.Tensor | None = None,
    ):
        self._swa_key_buffer = swa_key_buffer
        self._extra_key_buffer = (
            extra_key_buffer if extra_key_buffer is not None else swa_key_buffer
        )
        self.full_to_swa_index_mapping = full_to_swa_index_mapping
        self.swa_window_size = page_size

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        _ = layer_id
        return self._swa_key_buffer

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _ = layer_id
        return self.swa_window_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor:
        _ = layer_id
        return self._extra_key_buffer


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
    extra_k_cache = _make_v4_paged_kv_cache(
        total_slots=total_slots,
        page_size=page_size,
        seed=7,
        device=device,
    )
    token_to_kv_pool = _TokenToKVPool(
        swa_key_buffer=quant_k_cache,
        extra_key_buffer=extra_k_cache,
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


def _populate_compress_metadata(
    core_attn_metadata: SimpleNamespace,
    *,
    compress_ratio: int,
    device: torch.device,
) -> None:
    if compress_ratio == 4:
        core_attn_metadata.page_table = torch.zeros(
            (2, 4), dtype=torch.int32, device=device
        )
        core_attn_metadata.c4_sparse_raw_indices = torch.zeros(
            (16, 1), dtype=torch.int32, device=device
        )
    elif compress_ratio == 128:
        core_attn_metadata.c128_page_indices = torch.zeros(
            (16, 1), dtype=torch.int32, device=device
        )


@contextmanager
def _patched_compressed_sparse_cache_paths(compress_ratio: int):
    if compress_ratio == 0:
        yield
        return

    old_ensure_c4 = SparsePrefillChunkCache.ensure_c4
    old_ensure_c128 = SparsePrefillChunkCache.ensure_c128
    old_combine_c4_layer = SparsePrefillChunkCache.combine_c4_layer

    def _with_compressed_prefix(cache: SparsePrefillChunkCache, n_compressed: int):
        shifted_swa = torch.where(
            cache.c0_combined_indices >= 0,
            cache.c0_combined_indices + n_compressed,
            cache.c0_combined_indices,
        )
        n_prefix = min(n_compressed, shifted_swa.shape[1])
        if n_prefix > 0:
            shifted_swa[:, :n_prefix] = torch.arange(
                n_prefix, dtype=shifted_swa.dtype, device=shifted_swa.device
            )
        combined_lens = torch.clamp(
            cache.c0_combined_lens + n_prefix,
            max=shifted_swa.shape[1],
        )
        return shifted_swa, combined_lens

    def fake_ensure_c128(self, c128_page_indices):
        _ = c128_page_indices
        n_compressed = 8
        self.c128_flat_token_ids = torch.arange(
            n_compressed, dtype=torch.int64, device=self.swa_token_ids.device
        )
        self.c128_combined_indices, self.c128_combined_lens = _with_compressed_prefix(
            self, n_compressed
        )

    def fake_ensure_c4(self, page_table, extra_page_size):
        _ = page_table, extra_page_size
        n_compressed = 8
        self.c4_flat_token_ids = torch.arange(
            n_compressed, dtype=torch.int64, device=self.swa_token_ids.device
        )

    def fake_combine_c4_layer(self, c4_sparse_raw_indices):
        _ = c4_sparse_raw_indices
        return _with_compressed_prefix(self, self.c4_flat_token_ids.shape[0])

    SparsePrefillChunkCache.ensure_c128 = fake_ensure_c128
    SparsePrefillChunkCache.ensure_c4 = fake_ensure_c4
    SparsePrefillChunkCache.combine_c4_layer = fake_combine_c4_layer
    try:
        yield
    finally:
        SparsePrefillChunkCache.ensure_c4 = old_ensure_c4
        SparsePrefillChunkCache.ensure_c128 = old_ensure_c128
        SparsePrefillChunkCache.combine_c4_layer = old_combine_c4_layer


def _make_q8kv8_kernel_args(
    *,
    device: torch.device,
    s_q: int = 4,
    h_q: int = 64,
    d_qk: int = 512,
    s_kv: int = 256,
    h_kv: int = 1,
    topk: int = 128,
):
    q = (torch.randn(s_q, h_q, d_qk, device=device) * 0.05).to(torch.float8_e4m3fn)
    kv = (torch.randn(s_kv, h_kv, d_qk, device=device) * 0.05).to(torch.float8_e4m3fn)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device
    )
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    return {
        "q": q.contiguous(),
        "kv": kv.contiguous(),
        "indices": indices.contiguous(),
        "sm_scale": 512**-0.5,
        "q_scale": torch.ones((), dtype=torch.float32, device=device),
        "kv_scale": torch.ones((), dtype=torch.float32, device=device),
        "d_v": 512,
        "attn_sink": torch.zeros(h_q, dtype=torch.float32, device=device),
        "topk_length": topk_length,
    }


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

    q8_module_name = "sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90"
    q8_mod = types.ModuleType(q8_module_name)
    q8_mod.sparse_mla_q8kv8_prefill_fwd = fake_sparse_mla_q8kv8_prefill_fwd

    old_modules = {
        name: sys.modules.get(name)
        for name in (
            "sgl_kernel",
            "sgl_kernel.flash_mla",
            q8_module_name,
        )
    }
    sys.modules["sgl_kernel"] = sgl_kernel_pkg
    sys.modules["sgl_kernel.flash_mla"] = flash_mla_mod
    sys.modules[q8_module_name] = q8_mod
    try:
        yield
    finally:
        for name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_q8kv8_sparse_prefill_helper_builds_fp8_workspace_matching_bf16_path(
    compress_ratio: int,
):
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import fp8_dtype

    device = torch.device("cuda")
    backend, forward_batch, token_to_kv_pool, q, attn_sink, core_attn_metadata = (
        _make_sparse_prefill_case(device, local_heads=16)
    )
    _populate_compress_metadata(
        core_attn_metadata,
        compress_ratio=compress_ratio,
        device=device,
    )

    bf16_capture = _Capture()
    q8_capture = _Capture()
    with _patched_sparse_kernels(
        bf16_capture, q8_capture
    ), _patched_compressed_sparse_cache_paths(compress_ratio):
        bf16_out = backend._forward_prefill_sparse(
            q=q,
            layer_id=0,
            compress_ratio=compress_ratio,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            core_attn_metadata=core_attn_metadata,
            attn_sink=attn_sink,
        )
        sparse_cache = backend.forward_metadata.sparse_prefill_cache
        q8_out = backend._forward_prefill_sparse_q8kv8(
            q=q,
            layer_id=0,
            compress_ratio=compress_ratio,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_q8kv8_sparse_prefill_rejects_topk_64_before_cuda_launch():
    args = _make_q8kv8_kernel_args(device=torch.device("cuda"), topk=64)

    with pytest.raises(ValueError, match="positive multiple of 128"):
        sparse_mla_q8kv8_prefill_fwd(**args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (
            lambda args: args.update(q=args["q"].float()),
            "q must be torch.float8_e4m3fn",
        ),
        (
            lambda args: args.update(kv=args["kv"].float()),
            "kv must be torch.float8_e4m3fn",
        ),
        (
            lambda args: args.update(
                q=torch.empty(
                    args["q"].shape[0],
                    args["q"].shape[1],
                    args["q"].shape[2] + 1,
                    dtype=args["q"].dtype,
                    device=args["q"].device,
                )[:, :, : args["q"].shape[2]]
            ),
            "q must be contiguous",
        ),
        (
            lambda args: args.update(
                q_scale=torch.ones(2, dtype=torch.float32, device=args["q"].device)
            ),
            "q_scale must be a scalar tensor",
        ),
        (
            lambda args: args.update(
                kv_scale=torch.ones((), dtype=torch.float16, device=args["q"].device)
            ),
            "kv_scale must be float32",
        ),
    ],
)
def test_q8kv8_sparse_prefill_rejects_invalid_tensor_contracts(
    mutate,
    error_match: str,
):
    args = _make_q8kv8_kernel_args(device=torch.device("cuda"), topk=128)
    mutate(args)

    with pytest.raises(ValueError, match=error_match):
        sparse_mla_q8kv8_prefill_fwd(**args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("bad_length", [-1, 129])
def test_q8kv8_sparse_prefill_rejects_invalid_topk_length_bounds(
    bad_length: int,
):
    args = _make_q8kv8_kernel_args(device=torch.device("cuda"), topk=128)
    args["topk_length"][0] = bad_length

    with pytest.raises(ValueError, match="0 <= topk_length <= topk"):
        sparse_mla_q8kv8_prefill_fwd(**args)


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


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
def test_q8kv8_sparse_prefill_real_kernel_repeated_launch_stable():
    args = _make_q8kv8_kernel_args(
        device=torch.device("cuda"),
        s_q=512,
        h_q=64,
        d_qk=512,
        s_kv=1024,
        h_kv=1,
        topk=256,
    )

    baseline = None
    for _ in range(10):
        out, max_logits, lse = sparse_mla_q8kv8_prefill_fwd(**args)
        torch.cuda.synchronize()

        assert out.shape == (512, 64, 512)
        assert max_logits.shape == (512, 64)
        assert lse.shape == (512, 64)
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(max_logits).all()
        assert torch.isfinite(lse).all()

        current = out.float().detach().clone()
        if baseline is None:
            baseline = current
        else:
            torch.testing.assert_close(
                current,
                baseline,
                atol=1e-2,
                rtol=1e-2,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
