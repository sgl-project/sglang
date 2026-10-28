# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Optional FlashInfer MLA+SWA decode; prefill and other backends are unchanged."""

import torch

from sglang.srt.layers.attention.swa_mla_fallback.forward import (
    forward_dense_kvlora_swa_torch_fallback,
)


class FlashInferMLASWA:
    """Own workspace for one serial attention backend, prepared before capture."""

    def __init__(self, run, workspace):
        self.run = run
        self.workspace = workspace

    @classmethod
    def create(cls, device):
        if not torch.cuda.is_available():
            return None
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            return None
        props = torch.cuda.get_device_properties(device)
        if props.shared_memory_per_block_optin < 223584:
            return None
        try:
            import flashinfer.mla as mla
        except ModuleNotFoundError as exc:
            if exc.name in ("flashinfer", "flashinfer.mla"):
                return None
            raise
        run = getattr(mla, "batch_decode_with_kv_cache_mla_swa", None)
        size = getattr(mla, "get_mla_swa_workspace_size_bytes", None)
        if run is None or size is None:
            return None  # Older optional dependency, not a CUDA/ABI failure.
        from flashinfer.jit import MissingJITCacheError

        workspace = torch.empty(size(16, 4, 64), dtype=torch.uint8, device=device)
        # One real eager call populates module/device caches through the public
        # API. No private wrapper, tensor-value D2H, or capture-time compilation.
        q = torch.zeros((1, 1, 64, 1088), dtype=torch.bfloat16, device=device)
        kv = torch.zeros((1, 64, 1088), dtype=torch.bfloat16, device=device)
        table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        lengths = torch.ones(1, dtype=torch.int32, device=device)
        try:
            run(q, kv, workspace, table, lengths, window_left=512, sm_scale=0.0625)
        except MissingJITCacheError:
            # JIT explicitly disabled with no matching optional AOT artifact.
            # Do not catch compiler, loader/ABI, or CUDA execution failures.
            return None
        return cls(run, workspace)

    def supports(
        self, q, kv, table, lengths, layer, kv_cache_dim, head_dim_v, window_size
    ):
        return (
            q.is_cuda
            and q.device == self.workspace.device
            and kv.device == q.device
            and table.device == q.device
            and lengths.device == q.device
            and q.ndim == 4
            and q.shape[0] in (1, 2, 4, 8, 16)
            and q.shape[1] in (1, 4)
            and q.shape[2:] == (64, 1088)
            and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and q.stride()[-3:] == (64 * 1088, 1088, 1)
            and q.stride(0) >= q.shape[1] * 64 * 1088
            and kv.ndim == 3
            and kv.shape[1:] == (1, 1088)
            and kv.shape[0] > 0
            and kv.shape[0] % 64 == 0
            and kv.is_contiguous()
            and table.ndim == 2
            and table.shape[0] == q.shape[0]
            and table.shape[1] > 0
            and table.stride(1) == 1
            and table.stride(0) >= table.shape[1]
            and lengths.shape == (q.shape[0],)
            and lengths.is_contiguous()
            and table.dtype == torch.int32
            and lengths.dtype == torch.int32
            and q.data_ptr() % 16 == 0
            and kv.data_ptr() % 16 == 0
            and kv_cache_dim == 1088
            and head_dim_v == 1024
            and window_size == 513
            and layer.tp_k_head_num == 1
            and not getattr(layer, "is_cross_attention", False)
        )

    def forward(
        self,
        reshape_q,
        k_cache,
        block_table,
        cache_seqlens,
        layer,
        kv_cache_dim,
        head_dim_v,
        window_size,
    ):
        if not self.supports(
            reshape_q,
            k_cache,
            block_table,
            cache_seqlens,
            layer,
            kv_cache_dim,
            head_dim_v,
            window_size,
        ):
            return forward_dense_kvlora_swa_torch_fallback(
                reshape_q,
                k_cache,
                block_table,
                cache_seqlens,
                layer,
                kv_cache_dim,
                head_dim_v,
                window_size,
            )
        return self.run(
            reshape_q,
            k_cache.view(-1, 64, 1088),
            self.workspace,
            block_table,
            cache_seqlens,
            window_left=window_size - 1,
            sm_scale=float(layer.scaling),
        )
