"""Real multi-GPU Engram lookup parity (no model weights).

PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \
    test/manual/layers/test_engram_cp_all_to_all.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.ops.embeddings.engram_gather import (
    engram_rows_to_token_major,
    unpack_mxfp4_rows,
)
from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import initialize_model_parallel
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.layers.engram import EngramEmbedding
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.test_utils import publish_build_topology


def batch_for(counts, real=None):
    total = sum(counts) if real is None else real
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=[total],
        attn_cp_metadata=SimpleNamespace(
            per_rank_actual_token=counts,
            max_rank_len=[max(counts)] * len(counts),
            total_seq_lens=total,
        ),
    )


def equal(a, b):
    # Check every BF16 bit, including the sign of zero.
    assert torch.equal(
        a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    )


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=local, distributed_init_method="env://"
    )
    dp = "--dp" in sys.argv
    publish_build_topology(
        world_rank=rank,
        tp_size=world,
        attn_cp_size=1 if dp else world,
        dp_size=world if dp else 1,
        enable_dp_attention=dp,
        disable_custom_all_reduce=True,
    )
    initialize_model_parallel()
    get_context().override(
        "cp-test", engram_config=EngramConfig(placement="cpu", lookup="all_reduce")
    )
    device = torch.device("cuda", local)
    bounds = [0]
    for c in range(24):
        bounds.append(bounds[-1] + 11 + c * 2)
    lo = torch.tensor(bounds[:-1], device=device)
    lengths = torch.tensor([b - a for a, b in zip(bounds, bounds[1:])], device=device)
    checks = 0
    for fp4 in [False, True]:
        embed = EngramEmbedding(bounds[-1], 256, 0, tuple(bounds), mxfp4=fp4)
        g = torch.Generator().manual_seed(10)
        raw = torch.randint(
            0,
            256 if fp4 else 127,
            (bounds[-1], 128 if fp4 else 256),
            generator=g,
            dtype=torch.uint8,
        )
        if not fp4:
            raw[:, 0] = 128  # negative zero
        weight = raw if fp4 else raw.view(torch.float8_e4m3fn)
        scale = torch.randint(
            124, 130, (bounds[-1], 8), generator=g, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        embed.weight.weight_loader(embed.weight, weight)
        embed.scale.weight_loader(embed.scale, scale)
        embed.finish_load("cp-test")
        assert torch.equal(
            embed.weight.view(torch.uint8),
            raw[embed.row_start : embed.row_start + embed.rows],
        )
        assert torch.equal(
            embed.scale.view(torch.uint8),
            scale.view(torch.uint8)[embed.row_start : embed.row_start + embed.rows],
        )
        decoded = unpack_mxfp4_rows(weight) if fp4 else weight
        ref_table = (
            (decoded.float().reshape(-1, 8, 32) * scale.float().unsqueeze(-1))
            .flatten(1)
            .to(torch.bfloat16)
        )
        # Match the original SUM over one owner and zero-valued nonowners.
        ref_table.add_(0)
        if dp:
            for counts in [
                [3] * world,
                [2048] * world,
                list(range(world)),
                [0] * world,
            ]:
                n = counts[rank]
                ids = lo + (torch.rand(n, 24, device=device) * lengths).long()
                set_dp_buffer_len(sum(counts), n, False, counts)
                batch = SimpleNamespace(dp_padding_mode=DpPaddingMode.SUM_LEN)
                embed.lookup = "all_to_all"
                actual = embed(ids, batch)
                # Frozen pre-refactor DP implementation, using real collectives.
                if sum(counts):
                    group = get_parallel().tp_group
                    gathered = ids.new_empty(sum(counts), 24)
                    group.all_gatherv(ids, sizes=counts, output=gathered)
                    k = embed._cols_per_rank
                    own = gathered.flatten(1)[:, rank * k : (rank + 1) * k].contiguous()
                    values = embed._owned_rows(own).view(own.shape[0], -1)
                    recv = values.new_empty(world * n, values.shape[1])
                    if len(set(counts)) > 1:
                        dist.all_to_all_single(
                            recv, values, [n] * world, counts, group=group.device_group
                        )
                    else:
                        group.all_to_all_single(recv.view(-1), values.view(-1))
                    expected = engram_rows_to_token_major(recv, world).view(
                        *ids.shape, 256
                    )
                    equal(actual, expected)
                else:
                    assert actual.shape == (0, 24, 256)
                checks += 1
            continue
        for n in [1, 3, 128, 2048, 7, 0, 64]:
            torch.manual_seed(80 + n + rank)
            ids = lo + (torch.rand(n, 24, device=device) * lengths).long()
            if n:
                ids[0] = lo
                ids[-1] = lo + lengths - 1
            # Exercise non-contiguous IDs, as from the model's CP strided slice.
            storage = torch.empty(n * 2, 24, device=device, dtype=torch.int64)
            storage[::2] = ids
            ids = storage[::2]
            batch = batch_for([n] * world)
            embed.lookup = "all_reduce"
            a = embed(ids, batch, cp_all_tokens=True)
            embed.lookup = "all_to_all"
            assert embed._cp_all_to_all_enabled(batch) == (n >= 64)
            b = embed(ids, batch, cp_all_tokens=True)
            equal(a, b)
            expected = ref_table[ids.cpu()].to(device)
            equal(b, expected)
            if n:
                torch.manual_seed(9)
                wkv = torch.randn(48, 24 * 256, device=device, dtype=torch.bfloat16)
                equal(
                    torch.nn.functional.linear(a.flatten(1), wkv),
                    torch.nn.functional.linear(b.flatten(1), wkv),
                )
            checks += 1
        # The uneven CP tail is physically padded to the same rows on every rank.
        n = 3
        counts = [n] * (world - 1) + [n - 1]
        ids = (lo[None, :].expand(n, -1)).clone()
        if rank == world - 1:
            ids[-1].zero_()
        batch = batch_for(counts)
        assert not embed._cp_all_to_all_enabled(batch)
        embed.lookup = "all_reduce"
        a = embed(ids, batch, cp_all_tokens=True)
        embed.lookup = "all_to_all"
        equal(a, embed(ids, batch, cp_all_tokens=True))
        checks += 1
        # Use production metadata and padding: physical counts become uniform,
        # so the fallback must not rely only on uneven counts or the <64 guard.
        strategy = InterleaveCPStrategy(world)
        for total in [128 * world - 1, 129 * world]:
            metadata = strategy.build_metadata(total, [total])
            logical = list(metadata.per_rank_actual_token)
            pad_logical_token_to_physical(metadata)
            counts = metadata.per_rank_actual_token
            assert min(logical) >= 64
            assert counts == [counts[0]] * world == metadata.max_rank_len
            assert sum(counts) > metadata.total_seq_lens == total
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens_cpu=[total],
                attn_cp_metadata=metadata,
            )
            global_ids = lo + torch.arange(total, device=device)[:, None] % lengths
            ids = strategy.shard_hidden_states(global_ids, batch)
            assert ids.shape[0] == counts[rank]
            assert torch.count_nonzero(ids[logical[rank] :]).item() == 0
            embed.lookup = "all_reduce"
            expected = embed(ids, batch, cp_all_tokens=True)
            embed.lookup = "all_to_all"
            assert not embed._cp_all_to_all_enabled(batch)
            actual = embed(ids, batch, cp_all_tokens=True)
            equal(actual, expected)
            equal(actual, ref_table[ids.cpu()].to(device))
            checks += 1
        # Real graph capture takes the old collective; changed replay inputs
        # must not reuse stale lookup results or enter the eager A2A path.
        n = 64
        batch = batch_for([n] * world)
        ids = lo[None, :].expand(n, -1).clone()
        embed.lookup = "all_reduce"
        expected = embed(ids, batch, cp_all_tokens=True)
        torch.cuda.synchronize()
        embed.lookup = "all_to_all"
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                embed(ids, batch, cp_all_tokens=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                assert not embed._cp_all_to_all_enabled(batch)
                graph_out = embed(ids, batch, cp_all_tokens=True)
        torch.cuda.current_stream().wait_stream(stream)
        for delta in [0, 1, 2]:
            ids.copy_(lo[None, :] + delta)
            graph.replay()
            torch.cuda.synchronize()
            embed.lookup = "all_reduce"
            equal(graph_out, embed(ids, batch, cp_all_tokens=True))
            embed.lookup = "all_to_all"
        del graph_out, graph
        torch.cuda.synchronize()
        checks += 1
        # Group membership/order, rather than size alone, must match.
        cp = get_parallel().attn_cp_group
        with get_parallel().override(
            attn_cp_group=SimpleNamespace(ranks=list(reversed(cp.ranks)))
        ):
            assert not embed._cp_all_to_all_enabled(batch_for([64] * world))
        checks += 1
    dist.barrier()
    if rank == 0:
        print(
            f"PASS mode={'DP' if dp else 'CP'} checks={checks} world={world}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
