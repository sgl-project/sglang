"""Node storage placement must not change the calculation rank's row ownership."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.layers.engram import EngramEmbedding
from sglang.srt.mem_cache.engram import create_storage
from sglang.srt.mem_cache.engram.deepseek import _dequant


def options(shard):
    return json.dumps(
        dict(
            shard_id=shard,
            num_shards=2,
            master_addr="http://localhost:1/metadata",
            session_id="unit",
            max_lookup_rows=4096,
        )
    )


def test_head_placement_and_compute_partition_are_independent():
    bounds = (0, 17, 46, 65, 96)
    for node in range(2):
        memory = SimpleNamespace(
            enable_hi_engram="remote", hi_engram_config=options(node)
        )
        for rank in range(4):
            with (
                patch(
                    "sglang.srt.layers.engram.get_parallel",
                    return_value=SimpleNamespace(tp_size=4, tp_rank=rank),
                ),
                patch("sglang.srt.layers.engram.get_memory", return_value=memory),
            ):
                module = EngramEmbedding(96, 256, 14, bounds)
            assert (module.row_start, module.rows) == (
                bounds[rank],
                bounds[rank + 1] - bounds[rank],
            )
            begin, end = (0, 46) if node == 0 else (46, 96)
            assert module._storage_row_start == begin
            assert module.weight.shape == (end - begin, 256)
            full = (torch.arange(96 * 256).reshape(96, 256) % 16).to(
                torch.float8_e4m3fn
            )
            module._load_rows(module.weight, full)
            assert torch.equal(
                module.weight.view(torch.uint8), full[begin:end].view(torch.uint8)
            )


def test_reject_remote_missing_head_layout_before_weight_allocation():
    with pytest.raises(ValueError, match="head boundaries"):
        create_storage("remote", options(0), namespace="layer")


def test_qwen_weight_loader_uses_node_heads_without_changing_vocab_shard():
    from sglang.srt.mem_cache.engram.embedding import HiCacheVocabParallelEmbedding

    memory = SimpleNamespace(enable_hi_engram="remote", hi_engram_config=options(1))
    with patch("sglang.srt.mem_cache.engram.embedding.get_memory", return_value=memory):
        module = HiCacheVocabParallelEmbedding(
            96,
            32,
            params_dtype=torch.bfloat16,
            enable_tp=False,
            engram_head_bounds=(0, 17, 46, 65, 96),
            engram_namespace="qwen/ple0",
        )
    assert module.shard_indices.org_vocab_start_index == 0
    assert module.shard_indices.org_vocab_end_index == 96
    full = torch.arange(96 * 32).reshape(96, 32).to(torch.bfloat16)
    module.weight.weight_loader(module.weight, full)
    assert torch.equal(module.weight, full[46:96])


def test_qwen_remote_lookup_keeps_global_ids_and_compute_mask():
    from sglang.srt.mem_cache.engram.embedding import HiCacheVocabParallelEmbedding

    memory = SimpleNamespace(enable_hi_engram="remote", hi_engram_config=options(1))
    with patch("sglang.srt.mem_cache.engram.embedding.get_memory", return_value=memory):
        module = HiCacheVocabParallelEmbedding(
            96,
            32,
            params_dtype=torch.bfloat16,
            enable_tp=False,
            engram_head_bounds=(0, 17, 46, 65, 96),
            engram_namespace="qwen/ple1",
        )
    full = torch.arange(96 * 32).reshape(96, 32).to(torch.bfloat16)
    seen = []

    def read(ids):
        seen.append(ids.clone())
        values = full[ids.clamp_min(0)].masked_fill((ids == -1).unsqueeze(-1), 0)
        return values.view(torch.uint8)

    module._rows.lookup = read
    module._rows_initialized = True
    module.shard_indices = SimpleNamespace(
        org_vocab_start_index=17, org_vocab_end_index=65
    )
    ids = torch.tensor([[3, 19, 48, 70], [5, 45, 63, 95]])
    with (
        patch(
            "sglang.srt.mem_cache.engram.embedding.get_parallel",
            return_value=SimpleNamespace(tp_group=None),
        ),
        patch(
            "sglang.srt.mem_cache.engram.embedding.use_symmetric_memory",
            return_value=nullcontext(),
        ),
        patch(
            "sglang.srt.mem_cache.engram.embedding.is_allocation_symmetric",
            return_value=False,
        ),
    ):
        output = module._embed_local_shard(ids)
    expected = full[ids].clone()
    expected[:, (0, 3)] = 0
    assert torch.equal(output, expected)
    # IDs 19/45 are physically on node 0; 48/63 on node 1. The computational
    # rank retains both and delegates their storage location to EngramStore.
    assert seen[0].tolist() == [[-1, 19, 48, -1], [-1, 45, 63, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dequantization")
def test_packed_dequant_matches_native_including_zero_scale_exponent():
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    rows, dim, block = 13, 256, 32
    weight = (
        (torch.arange(rows * dim, device="cuda").reshape(rows, dim) % 17 - 8) / 4
    ).to(torch.float8_e4m3fn)
    exponents = torch.tensor(
        [0, 1, 126, 127, 128, 129, 130, 131], dtype=torch.uint8, device="cuda"
    ).repeat(rows, 1)
    packed = torch.cat((weight.view(torch.uint8), exponents), dim=1)
    expected = torch.empty((rows, dim), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty_like(expected)
    ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    engram_gather(weight.data_ptr(), exponents.data_ptr(), ids, expected, dim, block)
    _dequant[(rows,)](packed, actual, dim, block, packed.shape[1], dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
