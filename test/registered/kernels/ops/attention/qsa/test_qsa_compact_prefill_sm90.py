"""Offline QSA prefill coverage for varlen, prefix, paging and fallback."""

import sys
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("q_heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)])
def test_sm90_paged_prefill_varlen_prefix_and_fallback(q_heads, kv_heads, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an SM90 GPU")
    pytest.importorskip("tilelang")

    from sglang.kernels.ops.attention.qsa_sm90 import paged_prefill
    from sglang.srt.layers.attention.qsa.config import QSAProfile
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.test.mem_cache_utils import routed

    device = "cuda"
    generator = torch.Generator().manual_seed(714)
    lengths = [17, 67, 0, 25]
    prefixes = [0, 4097, 8192, 65535]
    seq_lengths = [length + prefix for length, prefix in zip(lengths, prefixes)]
    rows = sum(lengths)
    width = (max(seq_lengths) + 63) // 64 * 64
    pages_per_request = width // 64
    pages = torch.randperm(len(lengths) * pages_per_request, generator=generator)
    table = (
        (pages.reshape(len(lengths), pages_per_request, 1) * 64 + torch.arange(64))
        .flatten(1)
        .to(device=device, dtype=torch.int32)
    )
    positions_cpu = [
        pos
        for length, prefix in zip(lengths, prefixes)
        for pos in range(prefix, prefix + length)
    ]
    positions = torch.tensor(positions_cpu, dtype=torch.int32, device=device)
    seq = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    extend = torch.tensor(lengths, dtype=torch.int32, device=device)
    request = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), extend, output_size=rows
    )
    row_lens = seq[request]
    new_slots = table[request, positions.long()].long()
    blocks_cpu = torch.full((rows, 512), -1, dtype=torch.int32)
    for row, pos in enumerate(positions_cpu):
        available = (pos + 1) // 4
        picks = torch.randperm(available, generator=generator)[:512]
        blocks_cpu[row, : picks.numel()] = picks
    blocks = blocks_cpu.to(device)
    expanded = expand_qsa_block_indices(blocks, positions, row_lens, 4, 2048)
    q = torch.randn(rows + 3, q_heads, 256, device=device, dtype=torch.bfloat16)
    keys = torch.randn(table.numel(), kv_heads, 256, device=device, dtype=q.dtype)
    values = torch.randn_like(keys)
    new_keys = torch.randn(rows, kv_heads, 256, device=device, dtype=q.dtype)
    new_values = torch.randn_like(new_keys)

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            keys.index_copy_(0, loc, k)
            values.index_copy_(0, loc, v)

        def get_key_buffer(self, layer_id):
            return keys

        def get_value_buffer(self, layer_id):
            return values

    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=q_heads, head_dim=256, scaling=0.0625
    )
    backend = QwenSparseAttnBackend()
    backend._sm90_bf16_prefill = True
    backend.qsa_profile = QSAProfile(4, 1, 128, 2048, 4)
    backend.token_to_kv_pool = routed(Pool())
    backend.req_to_token_pool = SimpleNamespace(req_to_token=table)
    backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=False,
        sequence_lengths=seq,
        token_slot_table=table,
        indexer_metadata=SimpleNamespace(get_seqlens_expanded=lambda: row_lens),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        positions=positions,
        extend_seq_lens=extend,
        extend_seq_lens_cpu=lengths,
        seq_lens_cpu=seq_lengths,
        req_pool_indices=torch.arange(len(lengths), device=device),
        out_cache_loc=new_slots,
    )
    launches = []
    launch = paged_prefill.sparse_prefill_blocks

    def record_launch(*args):
        launches.append(args[0].shape[0])
        return launch(*args)

    monkeypatch.setattr(paged_prefill, "sparse_prefill_blocks", record_launch)

    def run(indices, *, save=True):
        output = backend.forward_extend(
            q, new_keys, new_values, layer, batch, save, indices
        )
        return output.reshape_as(q)

    actual = run(blocks)
    assert launches == [rows]
    expected = run(expanded)
    fallback = run(blocks, save=False)
    assert launches == [rows]
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
    torch.testing.assert_close(fallback, expected, atol=0, rtol=0)
    assert not actual[rows:].count_nonzero()

    # Independent FP32 reference. Read logical tokens directly from the input
    # blocks and append the causal tail, then apply the physical page map.
    first_rows = [0, *accumulate(lengths)]
    kv_head = torch.arange(q_heads, device=device) // (q_heads // kv_heads)
    for req, (start, end) in enumerate(zip(first_rows, first_rows[1:])):
        for row in range(start, end):
            visible = positions_cpu[row] + 1
            chosen = blocks_cpu[row][blocks_cpu[row] >= 0].to(device)
            logical = (chosen[:, None] * 4 + torch.arange(4, device=device)).flatten()
            tail = torch.arange(visible // 4 * 4, visible, device=device)
            slots = table[req, torch.cat((logical, tail))].long()
            k = keys[slots][:, kv_head].transpose(0, 1).float()
            v = values[slots][:, kv_head].transpose(0, 1).float()
            logits = (q[row].float()[:, None, :] * k).sum(-1) * layer.scaling
            reference = (logits.softmax(-1)[..., None] * v).sum(1)
            torch.testing.assert_close(
                actual[row].float(), reference, atol=0.025, rtol=0.025
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x"]))
