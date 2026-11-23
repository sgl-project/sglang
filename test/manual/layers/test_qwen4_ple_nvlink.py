"""Check Qwen PLE peer lookup with synthetic weights in two attention-TP groups.

From the repo root on one NVLink host (4 GPUs also work):
    torchrun --standalone --nproc-per-node=8 test/manual/layers/test_qwen4_ple_nvlink.py
"""

import os

import torch
import torch.distributed as dist

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_attn_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPeerEmbedding,
    Qwen4ExpPinnedHostEmbedding,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.test.test_utils import publish_build_topology


def _make_ngram(dtype):
    config = Qwen4ExpTextConfig(
        vocab_size=64,
        eos_token_id=1,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=53,
        make_ngram_vocab_size_divisible_by=1,
        ple_offload_embedding=True,
        ple_embedding_dtype="float8_e4m3fn" if dtype == torch.float8_e4m3fn else None,
    )
    with torch.device("cuda", torch.cuda.current_device()):
        return Qwen4ExpNGramEmbedding(config, embedding_dim=4 * 160)


@torch.no_grad()
def _check_lookup(dtype, group_index):
    ngram = _make_ngram(dtype)
    embedding = ngram.ngram_embedding
    group = get_attn_tp_group()
    assert isinstance(embedding, Qwen4ExpPeerEmbedding)
    assert embedding.tp_group is group and not ngram.gather_dp_tokens
    assert embedding.weight.is_cuda and embedding.weight.dtype == dtype
    row_count, dim = embedding.org_vocab_size, embedding.embedding_dim
    device = embedding.weight.device
    row_numbers = torch.arange(row_count, device=device).unsqueeze(1)
    cols = torch.arange(dim, device=device)
    # Each row encodes its ID. Give the two DP groups opposite signs so
    # pointers accidentally imported from the other group fail the comparison.
    rows = ((row_numbers >> (cols % 8)) & 1) * 2 + 1
    rows = (rows * (cols % 4 + 1) * (1 if group_index == 0 else -1)).to(dtype)
    embedding.weight.weight_loader(embedding.weight, rows)
    embedding.finish_loading()

    shard_size = embedding.num_embeddings_per_partition
    starts = list(range(0, row_count, shard_size))
    ends = [min(start + shard_size, row_count) - 1 for start in starts]
    local_row = starts[group.rank_in_group] + group.rank_in_group
    ids = torch.tensor(
        [starts + [-1, local_row], ends + [row_count, local_row]],
        dtype=torch.int32,
        device=device,
    ).T
    rows_bf16 = rows.to(torch.bfloat16)

    def check(actual, requested):
        valid = (requested >= 0) & (requested < row_count)
        expected = rows_bf16[requested.clamp(0, row_count - 1).long()]
        expected = expected.masked_fill(~valid.unsqueeze(-1), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    check(embedding(ids), ids)
    assert embedding(ids[:0]).shape == (0, 2, dim)

    static_ids = ids.long()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        embedding(static_ids)
        with torch.cuda.graph(graph, stream=stream):
            captured = embedding(static_ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    check(captured, ids)
    changed_ids = ids.flip((0, 1))
    static_ids.copy_(changed_ids)
    graph.replay()
    check(captured, changed_ids)
    # Keep the table and its graph alive until every peer has finished reading.
    dist.barrier(group=group.device_group)


def main():
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world not in (4, 8) or int(os.environ["LOCAL_WORLD_SIZE"]) != world:
        raise ValueError("Run with torchrun on one host with 4 or 8 GPUs")
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
    try:
        publish_build_topology(world_rank=rank, tp_size=world, attn_dp_size=2)
        initialize_model_parallel()
        initialize_dp_attention(get_server_args())
        group_index = rank // (world // 2)
        assert get_parallel().attn_dp_rank == group_index
        assert get_attn_tp_group().world_size == world // 2
        with envs.SGLANG_USE_ATTN_TP_NGRAM.override(False):
            with envs.SGLANG_USE_QWEN4_PLE_NVLINK.override(False):
                native = _make_ngram(torch.bfloat16)
                assert isinstance(native.ngram_embedding, Qwen4ExpPinnedHostEmbedding)
                assert native.gather_dp_tokens
            with envs.SGLANG_USE_QWEN4_PLE_NVLINK.override(True):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    _check_lookup(dtype, group_index)
        dist.barrier()
        if rank == 0:
            print("Qwen PLE NVLink eager and CUDA Graph checks passed")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
