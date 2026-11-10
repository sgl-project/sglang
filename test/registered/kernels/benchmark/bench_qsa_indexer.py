"""Region benchmark: eager QSA indexer-prep chain vs fused kernels.

Measures the per-layer indexer-prep region (index-Q norm+rope, raw-K/RoPE
position stores, compressed-K mean+norm+rope+store) at the decode shape
(bs=1) and the prefill shape (M=8000). Multiply the decode number by 12 for
the per-step total across Qwen4-Exp's 12 QSA layers.
"""

from types import SimpleNamespace

import torch
from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

from sglang.srt.layers.attention.qsa.kernel import average_pool_qsa_keys
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

# MRotaryEmbedding reads the exec config bag at init; publish a minimal
# process context for the bare benchmark process.
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy"), role="test")

HEAD_DIM = 128
NUM_Q_HEADS = 4
RATIO = 4
HIDDEN = 2560
EPS = 1e-6


class FakePool:
    def __init__(self, num_slots, num_compressed, device, dtype=torch.bfloat16):
        self.key_state = torch.zeros(num_slots, 1, HEAD_DIM, dtype=dtype, device=device)
        self.rope_position_buffer = torch.zeros(
            num_slots, 3, dtype=torch.int64, device=device
        )
        self.compressed = torch.zeros(
            num_compressed, 1, HEAD_DIM, dtype=dtype, device=device
        )

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.key_state[loc.long()] = token_k.to(self.key_state.dtype)

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc):
        return self.rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed

    def set_qsa_compressed_k_buffer(self, layer_id, loc, compressed_k):
        self.compressed[loc.long()] = compressed_k.to(self.compressed.dtype)


def _build(num_tokens, num_groups, device):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    rotary = MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position_embeddings=32768,
        base=1000000,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=[24, 20, 20],
        mrope_interleaved=True,
    )
    # Production capture sets sglang's capture-mode flag, which makes
    # apply_rope skip the host-side cache-length check (positions.max().item()
    # is a D2H sync that breaks graph capture). Mirror that here; the cos/sin
    # cache is pre-built long enough.
    import sglang.srt.layers.attention.qsa.qsa_indexer as _qsa_mod

    _qsa_mod.get_is_capture_mode = lambda: True
    rotary._ensure_cos_sin_cache_length = lambda needed: None
    # Build under the model dtype like ModelRunner does; device-only .to()
    # afterwards so the fp32 cos_sin_cache buffer keeps its dtype.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        indexer = QSAIndexer(
            SimpleNamespace(
                indexer_n_heads=NUM_Q_HEADS,
                indexer_kv_heads=1,
                indexer_head_dim=HEAD_DIM,
                indexer_budget=2048,
                indexer_compress_ratio=RATIO,
                hidden_size=HIDDEN,
                rms_norm_eps=EPS,
            ),
            layer_id=0,
            quant_config=None,
            rotary_emb=rotary,
        )
        indexer.to(device=device)
    finally:
        torch.set_default_dtype(prev_dtype)

    qk = torch.randn(
        num_tokens, (NUM_Q_HEADS + 1) * HEAD_DIM, device=device, dtype=dtype
    )
    token_k = qk[:, NUM_Q_HEADS * HEAD_DIM :].reshape(num_tokens, 1, HEAD_DIM)
    positions = (
        torch.arange(num_tokens, device=device).unsqueeze(0).expand(3, -1).contiguous()
    )
    cache_loc = torch.randperm(num_tokens + 8, device=device)[:num_tokens].long() + 1
    group_locs = torch.randint(1, 8192, (num_groups, RATIO), device=device).to(
        torch.int32
    )
    write_locs = torch.randperm(4096, device=device)[:num_groups].to(torch.int32)
    pool = FakePool(8192, 4096, device, dtype)
    return indexer, pool, qk, token_k, positions, cache_loc, group_locs, write_locs


def eager_region(
    indexer, pool, qk, token_k, positions, cache_loc, group_locs, write_locs
):
    num_tokens = qk.shape[0]
    q_raw = qk[:, : NUM_Q_HEADS * HEAD_DIM]
    q = indexer.q_layernorm(q_raw.reshape(-1, HEAD_DIM)).reshape(
        num_tokens, NUM_Q_HEADS, HEAD_DIM
    )
    q = indexer.apply_rope(positions, q)
    pool.set_qsa_key_state_buffer(0, cache_loc, token_k)
    pool.set_qsa_rope_position_buffer(cache_loc, positions)
    key_groups = pool.get_qsa_key_state_buffer(0)[group_locs.long()]
    pooled = average_pool_qsa_keys(key_groups)
    rope_positions = indexer._get_group_rope_positions(pool, group_locs[:, 0])
    normalized = indexer.normalize_compressed_keys(pooled, rope_positions)
    pool.set_qsa_compressed_k_buffer(0, write_locs, normalized.contiguous())
    return q


def fused_region(
    indexer, pool, qk, token_k, positions, cache_loc, group_locs, write_locs
):
    from sglang.kernels.ops.attention.qsa_indexer import (
        qsa_index_k_compress_store,
        qsa_index_q_norm_rope_store,
    )

    key_state = pool.key_state.view(pool.key_state.shape[0], -1)
    compressed = pool.compressed.view(pool.compressed.shape[0], -1)
    q = qsa_index_q_norm_rope_store(
        qk,
        positions,
        indexer.rotary_emb.cos_sin_cache,
        indexer._rope_axis_map(qk.device),
        indexer.q_layernorm.weight.data,
        cache_loc,
        key_state,
        pool.rope_position_buffer,
        NUM_Q_HEADS,
        indexer.rotary_emb.rotary_dim,
        EPS,
        True,
        q_heads_padded=8,
    )
    qsa_index_k_compress_store(
        key_state,
        group_locs,
        pool.rope_position_buffer,
        indexer.rotary_emb.cos_sin_cache,
        indexer._rope_axis_map(qk.device),
        indexer.k_layernorm.weight.data,
        write_locs,
        compressed,
        RATIO,
        indexer.rotary_emb.rotary_dim,
        EPS,
        True,
    )
    return q


FN_MAP = {"eager": eager_region, "fused": fused_region}


@marker.parametrize(
    "num_tokens,num_groups",
    [(1, 1), (8000, 2000)],  # decode bs=1; prefill M=8000 (ratio 4)
    [(1, 1), (8000, 2000)],
)
@marker.benchmark("impl", ["eager", "fused"], unit="us")
def benchmark(num_tokens: int, num_groups: int, impl: str):
    device = torch.device("cuda")
    args = _build(num_tokens, num_groups, device)

    def fn():
        return FN_MAP[impl](*args)

    return marker.do_bench(
        fn,
        input_args=(),
        graph_clone_args=None,
        memory_args=None,
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
