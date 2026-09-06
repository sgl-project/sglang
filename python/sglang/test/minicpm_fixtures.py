"""Shared construction scaffold for MiniCPMSparseBackend unit tests."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.runtime_context import get_schedule, override_platform
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend

POOL_SIZE = 8


def _sparse_token_table(*, pages, device):
    """One compression level's request-to-chunk table, distinct value per slot."""
    return torch.arange(POOL_SIZE * pages, dtype=torch.int32, device=device).view(
        POOL_SIZE, pages
    )


def make_runner_scaffold(
    *,
    max_context_len=1024,
    max_running_requests=1,
):
    """Build the (model_runner, flash_attn_backend) SimpleNamespace pair
    that MiniCPMSparseBackend construction reads;
    callers add the fields only their tests consume
    before patching in the backend."""
    kernel_size, kernel_stride = 32, 16
    # attach_compressed_cache is stubbed out here, so the scaffold sizes the K1/K2
    # tables the way cache.py does.
    req_pool = SimpleNamespace(
        req_to_sparse_k1_token=_sparse_token_table(
            pages=(max_context_len - kernel_size) // kernel_stride + 1, device="cpu"
        ),
        req_to_sparse_k2_token=_sparse_token_table(
            pages=(max_context_len - kernel_size * 4) // (kernel_stride * 4) + 1,
            device="cpu",
        ),
    )
    flash_attn_backend = SimpleNamespace(
        max_context_len=max_context_len,
        device="cpu",
        decode_cuda_graph_metadata={},
        req_to_token_pool=req_pool,
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
    )
    server_args = SimpleNamespace(enable_memory_saver=False)
    model_runner = SimpleNamespace(
        dtype=torch.float16,
        max_running_requests=max_running_requests,
        token_to_kv_pool_allocator=SimpleNamespace(),
        server_args=server_args,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                has_minicpm_sparse_attention=True,
                sparse_config={
                    "kernel_size": kernel_size,
                    "kernel_stride": kernel_stride,
                    "init_blocks": 1,
                    "block_size": 64,
                    "window_size": 64,
                    "dense_len": 128,
                    "topk": 1,
                },
            ),
            num_attention_heads=16,
            head_dim=128,
            get_num_kv_heads=lambda _tp: 1,
        ),
    )
    return model_runner, flash_attn_backend


class FakeFlashInferAdapter:
    def __init__(self, *args, **kwargs):
        pass


def make_sparse_backend(
    *,
    max_context_len=256,
    chunked_prefill_size=64,
    max_running_requests=1,
    use_flashinfer=False,
    blackwell=False,
    kv_cache_dtype_str="bfloat16",
):
    """A CPU MiniCPMSparseBackend over the mocked runner scaffold; returns
    (backend, model_runner, flash_attn_backend, flash_attention_ctor)."""
    model_runner, flash_attn_backend = make_runner_scaffold(
        max_context_len=max_context_len,
        max_running_requests=max_running_requests,
    )
    flash_attn_backend.kv_cache_dtype_str = kv_cache_dtype_str
    flash_attn_backend.init_forward_metadata = Mock()
    flash_attn_backend.forward_metadata = None
    with (
        get_schedule().override(chunked_prefill_size=chunked_prefill_size),
        patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
        override_platform(is_blackwell=blackwell),
        patch.object(
            backend_module, "FlashAttentionBackend", return_value=flash_attn_backend
        ) as flash_attention,
        patch.object(backend_module, "MiniCPMFlashInferAdapter", FakeFlashInferAdapter),
        patch.object(
            backend_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1),
        ),
        patch.object(backend_module, "attach_compressed_cache"),
    ):
        backend = MiniCPMSparseBackend(model_runner, use_flashinfer=use_flashinfer)
    return backend, model_runner, flash_attn_backend, flash_attention
