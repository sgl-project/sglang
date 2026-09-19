"""Shared compressed-KV workspace validity across layers and forwards."""
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    CompressedGather,
    SparsePrefillWorkspace,
    WORKSPACE_DIM,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_shared_compressed_dequant_lifetime():
    device = "cuda"
    obj = object.__new__(backend.DeepseekV4AttnBackend)
    obj.sparse_prefill_workspace = SparsePrefillWorkspace(device)
    obj.shared_compressed_prefill_workspaces = {
        ratio: SparsePrefillWorkspace(device) for ratio in (1, 2)
    }
    obj.softmax_scale = 0.1
    obj.head_dim_v = WORKSPACE_DIM
    sources = {0: 0, 1: 1, 2: 0, 3: 1, 4: 4, 5: 4, 6: 6, 7: 7}
    ratios = {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 1, 6: 0, 7: 4}
    compressed = {
        source: torch.full(
            (128, 1, WORKSPACE_DIM),
            float(source + 1),
            device=device,
            dtype=torch.bfloat16,
        )
        for source in (0, 1, 4, 7)
    }
    swa = torch.empty((16, 1, WORKSPACE_DIM), device=device, dtype=torch.bfloat16)
    pool = NS(
        source_layer_of=lambda layer: sources[layer],
        get_extra_key_page_size=lambda layer: 1,
        get_extra_key_buffer=lambda layer: compressed[sources[layer]],
        get_extra_key_layout=lambda layer: None,
        get_swa_key_buffer_radix=lambda layer: swa,
        get_swa_key_layout=lambda: None,
    )
    calls = []
    active = [True]

    def dequant(src, indices, *, out, **kwargs):
        calls.append("swa" if src is swa else sources_by_ptr[src.data_ptr()])
        out.copy_(src.index_select(0, indices.long()))

    sources_by_ptr = {v.data_ptr(): k for k, v in compressed.items()}

    def make_cache(n):
        gathers = {
            ratio: CompressedGather(
                flat_token_ids=torch.arange(n // ratio, device=device, dtype=torch.int32),
                compressed_base=torch.zeros(1, device=device, dtype=torch.int32),
                swa_base=torch.zeros(1, device=device, dtype=torch.int32),
            )
            for ratio in (1, 2, 4)
        }
        indices = torch.zeros((4, 128), device=device, dtype=torch.int32)
        lengths = torch.full((4,), 3, device=device, dtype=torch.int32)
        cache = NS(
            compressed=gathers,
            swa_token_ids=torch.arange(3, device=device),
            swa_page_size=1,
            c0_combined_indices=indices,
            c0_combined_lens=lengths,
        )
        cache.layer_inputs = lambda ratio, core, page: (
            gathers[ratio].flat_token_ids,
            indices,
            lengths,
        )
        return cache

    def forward(layer):
        return obj._forward_prefill_sparse(
            torch.empty((4, 1, 1, WORKSPACE_DIM), device=device, dtype=torch.bfloat16),
            layer,
            ratios[layer],
            NS(),
            pool,
            NS(),
            torch.zeros(1, device=device),
        )

    with (
        patch.object(backend, "is_cp_active", side_effect=lambda _: active[0]),
        patch.object(backend, "dequantize_k_cache_paged", side_effect=dequant),
        patch(
            "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
            side_effect=lambda **kw: (kw["kv"].clone(), None, None),
        ),
    ):
        # Same-size replay, then growth and shrink must all read freshly written KV.
        for step, n in enumerate((8, 8, 40, 4)):
            obj.forward_metadata = NS(sparse_prefill_cache=make_cache(n))
            for source, tensor in compressed.items():
                tensor.fill_(source + 1 + step * 10)
            for layer, should_dequant in (
                (0, True),
                (1, True),
                (2, False),
                (3, False),
                (4, True),
                (5, False),
                (2, True),
                (3, False),
                (6, False),
                (7, True),
                (7, True),
            ):
                swa.fill_(100 + layer + step)
                before = len(calls)
                active[0] = True
                actual = forward(layer)
                actual_calls = calls[before:]
                assert actual_calls.count("swa") == 1
                assert len(actual_calls) == 1 + int(should_dequant)
                active[0] = False
                expected = forward(layer)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            # Workspace replacement invalidates even an unchanged source identity.
            obj.shared_compressed_prefill_workspaces[1].get(256 + step * 256)
            active[0] = True
            before = len(calls)
            actual = forward(2)
            assert calls[before:] == [0, "swa"]
            active[0] = False
            torch.testing.assert_close(actual, forward(2), rtol=0, atol=0)

            # Re-executing a producer may mutate the same cache address in place.
            compressed[0].add_(1)
            active[0] = True
            before = len(calls)
            actual = forward(0)
            assert calls[before:] == [0, "swa"]
            active[0] = False
            torch.testing.assert_close(actual, forward(0), rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
