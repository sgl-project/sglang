"""QSA backend with the K/V host offload vs. the full GPU K/V pool.

Two QSA pools receive the same K/V: one keeps it on the GPU (upstream), the other
offloads it to pinned host memory (``kv_offload_layout``). The QSA backend runs
the same forwards on both: three prefill chunks per request (the later chunks
gather their cached context from the host) and decode steps whose selections
evict and reload hot-cache blocks and cross the tail ring. The attention outputs
must be bit-identical. Breaking the scratch or ring rows the backend stores to,
the host write-through, the prefill context gather or the decode row table
turns this red.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.qsa_kv_offload import compute_offload_layout
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

CR, PAGE, Q_HEADS, KV_HEADS, DIM = 4, 64, 24, 2, 256
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // CR
# The minimum: one forward's selection fills the hot cache, so steps evict.
HOT_BLOCKS = BLOCK_TOPK
REQUESTS, POOL_TOKENS, CONTEXT, EXTEND_TOKENS = 2, 16384, 8192, 2048
CHUNKS = ((1024, 1024), (1024, 1024), (1024, 512))
DECODE_STEPS = 8


@pytest.fixture
def server_context():
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield
    finally:
        reset_context()


def _backend(req_to_token, kv_offload_layout):
    pool = QSATokenToKVPool(
        size=POOL_TOKENS,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=KV_HEADS,
        head_dim=DIM,
        full_attention_layer_ids=[0],
        device="cuda",
        mamba_pool=None,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=128,
        qsa_compress_ratio=CR,
        qsa_token_topk=TOKEN_TOPK,
        num_request_slots=REQUESTS + 1,
        kv_offload_layout=kv_offload_layout,
    )
    runner = SimpleNamespace(
        device="cuda",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(
            context_len=CONTEXT,
            hf_config=SimpleNamespace(indexer_compress_ratio=CR),
        ),
    )
    return QwenSparseAttnBackend(runner)


def _select(positions: torch.Tensor) -> torch.Tensor:
    """Expanded top-k token positions: up to ``BLOCK_TOPK`` random complete
    blocks before each query position, plus its own partial block."""
    complete = (positions + 1) // CR
    scores = torch.rand(
        positions.numel(), max(int(complete.max()), BLOCK_TOPK), device="cuda"
    )
    columns = torch.arange(scores.shape[1], device="cuda")
    scores[columns[None, :] >= complete[:, None]] = -1.0
    top = scores.topk(BLOCK_TOPK, dim=1)
    blocks = torch.where(top.values >= 0, top.indices, -1).to(torch.int32)
    return expand_qsa_block_indices(blocks, positions, positions + 1, CR, TOKEN_TOPK)


def _forward_batch(mode, *, lengths, new_tokens, positions, out_cache_loc):
    seq_lens = [lengths[r] + new_tokens[r] for r in range(REQUESTS)]
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=REQUESTS,
        req_pool_indices=torch.arange(1, REQUESTS + 1, device="cuda"),
        seq_lens=torch.tensor(seq_lens, device="cuda"),
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens=torch.tensor(new_tokens, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=list(new_tokens),
        extend_prefix_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        positions=positions,
        mrope_positions=None,
        input_ids=torch.zeros(positions.numel(), dtype=torch.int32, device="cuda"),
        out_cache_loc=out_cache_loc,
        spec_info=None,
        _original_forward_mode=None,
    )


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, what: str):
    assert actual.shape == expected.shape, what
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), f"{what}: offloaded attention differs from the full pool"


@pytest.mark.parametrize("torch_resolve", [False, True])
def test_backend_offload_matches_full_pool(server_context, torch_resolve):
    torch.manual_seed(11)
    req_to_token = torch.zeros(
        (REQUESTS + 1, CONTEXT), dtype=torch.int32, device="cuda"
    )
    layout = compute_offload_layout(
        compress_ratio=CR,
        token_topk=TOKEN_TOPK,
        hot_blocks=HOT_BLOCKS,
        num_request_slots=REQUESTS + 1,
        rows_per_request=1,
        max_extend_tokens=EXTEND_TOKENS,
        page_size=PAGE,
    )
    full = _backend(req_to_token, kv_offload_layout=None)
    with envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.override(torch_resolve):
        offloaded = _backend(req_to_token, kv_offload_layout=layout)
    assert full._kv_offload is None
    assert offloaded._kv_offload.use_fused is not torch_resolve
    try:
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=Q_HEADS,
            tp_k_head_num=KV_HEADS,
            tp_v_head_num=KV_HEADS,
            head_dim=DIM,
            scaling=DIM**-0.5,
            k_scale=None,
            v_scale=None,
        )
        # Logical slots in random order; slot 0 is the padding slot.
        free = (torch.randperm(POOL_TOKENS - 1, device="cuda") + 1).tolist()
        lengths = [0] * REQUESTS

        def run(mode, new_tokens, positions, topk_indices, what):
            nonlocal free
            num = sum(new_tokens)
            loc = torch.tensor(free[:num], dtype=torch.int64, device="cuda")
            free = free[num:]
            start = 0
            for r, n in enumerate(new_tokens):
                req_to_token[r + 1, lengths[r] : lengths[r] + n] = loc[
                    start : start + n
                ].int()
                start += n
            q = torch.randn(num, Q_HEADS * DIM, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(num, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            outputs = []
            for backend in (full, offloaded):
                batch = _forward_batch(
                    mode,
                    lengths=lengths,
                    new_tokens=new_tokens,
                    positions=positions,
                    out_cache_loc=loc,
                )
                backend.init_forward_metadata(batch)
                forward = (
                    backend.forward_extend
                    if mode == ForwardMode.EXTEND
                    else backend.forward_decode
                )
                outputs.append(
                    forward(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        layer,
                        batch,
                        topk_indices=topk_indices,
                    )
                )
            _assert_bitwise_equal(outputs[1], outputs[0], what)
            for r, n in enumerate(new_tokens):
                lengths[r] += n

        for chunk, new_tokens in enumerate(CHUNKS):
            positions = torch.cat(
                [
                    torch.arange(lengths[r], lengths[r] + n, device="cuda")
                    for r, n in enumerate(new_tokens)
                ]
            )
            run(
                ForwardMode.EXTEND,
                new_tokens,
                positions,
                _select(positions),
                f"extend chunk {chunk}",
            )
        for step in range(DECODE_STEPS):
            positions = torch.tensor(lengths, device="cuda")
            run(
                ForwardMode.DECODE,
                [1] * REQUESTS,
                positions,
                _select(positions),
                f"decode step {step}",
            )
        if offloaded._kv_offload.fused is not None:
            assert int(offloaded._kv_offload.fused.errors) == 0
    finally:
        offloaded._kv_offload.destroy()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
