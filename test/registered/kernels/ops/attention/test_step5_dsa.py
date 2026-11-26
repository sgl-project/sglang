import sys

import pytest
import torch

from sglang.kernels.ops.attention.deep_select import is_deep_select_supported
from sglang.kernels.ops.attention.step5_dsa_indexer import (
    STEP5_PAGE_SIZE,
    step5_dsa_append_ring,
    step5_dsa_indexer_norm_rope,
    step5_dsa_select_blocks,
)
from sglang.kernels.ops.attention.step5_dsa_sparse_attention import (
    step5_dsa_sparse_attention,
)
from sglang.srt.layers.attention.step5_backend import (
    Step5CompressionPlan,
    Step5SelectionWorkspace,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels import step5_dsa_reference as ref

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_deep_select_supported(), reason="requires a DeepSelect-capable GPU"
)

REGION, TOPK, INDEX_DIM, ROPE_DIM = 8, 512, 256, 32


def _request_mapping(num_requests: int, capacity: int) -> torch.Tensor:
    starts = STEP5_PAGE_SIZE * (1 + torch.arange(num_requests + 1) * capacity // 64)
    mapping = starts[:, None] + torch.arange(capacity)
    mapping[0] = 0
    return mapping.to(device="cuda", dtype=torch.int32)


def _rope_cache(max_positions: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    angles = torch.arange(max_positions).float()[:, None] * inv_freq
    return torch.cat((angles.cos(), angles.sin()), -1).bfloat16().cuda()


@pytest.mark.parametrize("heads", [2, 4])
def test_indexer_norm_rope_matches_reference(heads):
    torch.manual_seed(0)
    tokens = 37
    q = torch.randn(tokens, heads, INDEX_DIM, device="cuda").bfloat16()
    k = torch.randn(tokens, 1, INDEX_DIM, device="cuda").bfloat16()
    weights = dict(
        q_weight=torch.randn(INDEX_DIM, device="cuda").bfloat16() * 0.1,
        k_weight=torch.randn(INDEX_DIM, device="cuda").bfloat16(),
        k_bias=torch.randn(INDEX_DIM, device="cuda").bfloat16() * 0.1,
        cache=_rope_cache(4096),
        positions=torch.randint(0, 4096, (tokens,), device="cuda"),
        eps=1e-5,
    )
    q_out, k_out = step5_dsa_indexer_norm_rope(q, k, **weights)
    q_ref, k_ref = ref.indexer_norm_rope(q, k, **weights)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


def _append(plan, state, index_key, logits):
    rows = plan.request_slots.numel()
    step5_dsa_append_ring(
        slots=plan.request_slots,
        bases=plan.input_start_positions,
        sources=plan.input_start_rows,
        lengths=plan.input_lengths,
        active=plan.request_active,
        admitted=plan.request_mapping_valid,
        count_valid=plan.request_count_valid,
        physical_blocks=plan.compressed_key_write_indices,
        enabled=torch.empty(rows, dtype=torch.bool, device="cuda"),
        success=torch.empty(rows, dtype=torch.bool, device="cuda"),
        index_key=index_key,
        compression_logits=logits,
        **state,
    )


def test_ring_compression_matches_reference():
    torch.manual_seed(1)
    capacity, prefill, decode_steps = 256, (20, 13), 25
    mapping = _request_mapping(len(prefill), capacity)
    num_pages = int(mapping.max()) // STEP5_PAGE_SIZE + 2
    compressed = torch.zeros(
        num_pages, STEP5_PAGE_SIZE // REGION, 1, INDEX_DIM, dtype=torch.uint8
    ).cuda()
    num_slots = mapping.shape[0]
    state = dict(
        index_key_ring=torch.empty(num_slots, REGION, 1, INDEX_DIM).bfloat16().cuda(),
        compression_logits_ring=torch.empty(num_slots, REGION, 1, INDEX_DIM)
        .bfloat16()
        .cuda(),
        ring_positions=torch.full((num_slots, REGION), -1, device="cuda"),
        compressed_index_keys=compressed,
    )
    lengths = [n + decode_steps for n in prefill]
    keys = [torch.randn(n, 1, INDEX_DIM, device="cuda").bfloat16() for n in lengths]
    logits = [torch.randn(n, 1, INDEX_DIM, device="cuda").bfloat16() for n in lengths]
    slots = torch.arange(1, len(prefill) + 1, device="cuda")

    plan = Step5CompressionPlan.for_prefill(
        request_slots=slots,
        prefix_lengths=[0] * len(prefill),
        append_lengths=list(prefill),
        req_to_token=mapping,
        compressed_index_key_pages=num_pages,
        region_size=REGION,
    )
    _append(
        plan,
        state,
        torch.cat([k[:n] for k, n in zip(keys, prefill)]),
        torch.cat([z[:n] for z, n in zip(logits, prefill)]),
    )
    for step in range(decode_steps):
        positions = torch.tensor([n + step for n in prefill], device="cuda")
        plan, _, _ = Step5CompressionPlan.for_decode(
            request_slots=slots,
            positions=positions,
            valid_rows=torch.tensor([len(prefill)], dtype=torch.int32, device="cuda"),
            req_to_token=mapping,
            compressed_index_key_pages=num_pages,
            context_len=capacity,
            region_size=REGION,
        )
        _append(
            plan,
            state,
            torch.stack([k[n + step] for k, n in zip(keys, prefill)]).contiguous(),
            torch.stack([z[n + step] for z, n in zip(logits, prefill)]).contiguous(),
        )

    blocks = compressed.view(-1, INDEX_DIM).view(torch.float8_e4m3fn)
    for request, (key, logit) in enumerate(zip(keys, logits)):
        for region in range(key.shape[0] // REGION):
            tokens = slice(region * REGION, (region + 1) * REGION)
            address = int(mapping[request + 1, region * REGION]) // REGION
            expected = ref.compress_region(key[tokens, 0], logit[tokens, 0])
            torch.testing.assert_close(
                blocks[address].float(), expected.float(), atol=1e-2, rtol=0.13
            )


def _selection_inputs(positions):
    torch.manual_seed(2)
    rows = len(positions)
    capacity = (max(positions) // 64 + 2) * 64
    mapping = _request_mapping(rows, capacity)
    num_pages = int(mapping.max()) // STEP5_PAGE_SIZE + 2
    num_blocks = num_pages * (STEP5_PAGE_SIZE // REGION)
    compressed = (torch.rand(num_blocks, INDEX_DIM, device="cuda") + 0.1).to(
        torch.float8_e4m3fn
    )
    return dict(
        mapping=mapping,
        compressed=compressed,
        row_to_req=torch.arange(1, rows + 1, dtype=torch.int32, device="cuda"),
        positions=torch.tensor(positions, dtype=torch.int32, device="cuda"),
        index_q=(torch.rand(rows, 4, INDEX_DIM, device="cuda") + 0.1).bfloat16(),
        index_weights=torch.rand(rows, 4, device="cuda") + 0.1,
    )


def _select(inputs):
    rows = inputs["positions"].numel()
    workspace = Step5SelectionWorkspace.allocate(
        rows=rows,
        score_width=-(-(int(inputs["positions"].max()) + 1) // REGION),
        max_score_width=inputs["mapping"].shape[1] // REGION,
        row_to_req=inputs["row_to_req"],
        positions=inputs["positions"],
        topk=TOPK,
        region_size=REGION,
    )
    step5_dsa_select_blocks(
        index_q=inputs["index_q"],
        index_weights=inputs["index_weights"],
        compressed_index_keys=inputs["compressed"]
        .view(torch.uint8)
        .view(-1, STEP5_PAGE_SIZE // REGION, 1, INDEX_DIM),
        req_to_token=inputs["mapping"],
        row_to_req=inputs["row_to_req"],
        positions=inputs["positions"],
        tiles=workspace.tiles,
        indexer_logits=workspace.indexer_logits,
        ids=workspace.ids,
        packed=workspace.packed,
        counts=workspace.counts,
        lengths=workspace.lengths,
        use_deepselect=workspace.use_deepselect,
        topk_lengths=workspace.topk_lengths,
        topk_ids=workspace.topk_ids,
        topk_size=TOPK,
        region_size=REGION,
    )
    return workspace


def _expected_blocks(inputs, row):
    position = int(inputs["positions"][row])
    history = position // REGION
    if history <= TOPK:
        selected = list(range(history))
    else:
        mapping = inputs["mapping"][row + 1]
        addresses = mapping[torch.arange(history, device="cuda") * REGION].long()
        scores = ref.block_scores(
            inputs["index_q"][row],
            inputs["index_weights"][row],
            inputs["compressed"][addresses // REGION],
        )
        selected = sorted(scores.topk(TOPK).indices.tolist())
    return selected + [history]


@pytest.mark.parametrize("positions", [[5, 100, 300], [100, 4100, 6000]])
def test_selection_and_sparse_attention_match_reference(positions):
    inputs = _selection_inputs(positions)
    workspace = _select(inputs)
    rows, heads, head_dim = len(positions), 16, 192
    num_tokens = int(inputs["mapping"].max()) + 1
    q = torch.randn(rows, heads, head_dim, device="cuda").bfloat16()
    k_cache = torch.randn(num_tokens, 1, head_dim, device="cuda").bfloat16()
    v_cache = torch.randn(num_tokens, 1, head_dim, device="cuda").bfloat16()
    scale = head_dim**-0.5
    out = step5_dsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        packed=workspace.packed,
        counts=workspace.counts,
        lengths=workspace.lengths,
        out=torch.empty_like(q),
        scale=scale,
        region_size=REGION,
    )
    for row in range(rows):
        expected = _expected_blocks(inputs, row)
        count = int(workspace.counts[row])
        assert workspace.ids[row, :count].tolist() == expected
        position = int(inputs["positions"][row])
        logical = torch.tensor(expected, device="cuda")[:, None] * REGION
        logical = (logical + torch.arange(REGION, device="cuda")).flatten()
        logical = logical[logical <= position]
        slots = inputs["mapping"][row + 1, logical].long()
        expected_out = ref.sparse_attention(
            q[row], k_cache, v_cache, token_slots=slots, scale=scale
        )
        torch.testing.assert_close(out[row], expected_out, atol=2e-2, rtol=2e-2)


def test_selection_is_deterministic():
    inputs = _selection_inputs([100, 4100, 6000])
    first = _select(inputs)
    second = _select(inputs)
    assert torch.equal(first.packed, second.packed)
    assert torch.equal(first.counts, second.counts)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
