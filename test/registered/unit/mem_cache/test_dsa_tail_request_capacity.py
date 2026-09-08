import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _request_pool(kind, extra_slots, device):
    kwargs = dict(
        size=32,
        max_context_len=8,
        device=device,
        enable_memory_saver=False,
    )
    if kind == "regular":
        return ReqToTokenPool(**kwargs)
    if kind == "decode":
        return DecodeReqToTokenPool(**kwargs, pre_alloc_size=extra_slots)

    # Initialize the shared target pool's request table without allocating
    # unrelated Mamba state. The NextN draft uses the plain DSA KV builder.
    pool = HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
    DecodeReqToTokenPool.__init__(pool, **kwargs, pre_alloc_size=extra_slots)
    return pool


def _build_pool(req_pool, *, compress=True, device="cpu"):
    cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
    cfg.device = device
    cfg.is_draft_worker = True
    cfg.use_mla_backend = True
    cfg.is_hybrid_swa = False
    cfg.mambaish_config = None
    cfg.kv_cache_dtype = torch.bfloat16
    cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=2, num_effective_layers=2)
    cfg.model_config = SimpleNamespace(
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        hf_config=SimpleNamespace(
            architectures=["Glm5NextForConditionalGenerationNextN"],
            index_topk=8,
            index_head_dim=128,
            index_kpool=4,
            index_kpool_compress=compress,
        ),
    )
    with get_context().override_server_args(
        page_size=64,
        attention_backend="triton",
        enable_hisparse=False,
        enable_unified_memory=False,
        enable_page_major_kv_layout=False,
        enable_memory_saver=False,
        speculative_algorithm="EAGLE",
        speculative_num_draft_tokens=2,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        disaggregation_mode="decode",
        dsa_decode_backend="tilelang",
    ):
        return cfg._build_token_to_kv_pool(
            sizes=SimpleNamespace(max_total_num_tokens=64, max_running_requests=32),
            is_dsa_model=True,
            is_dsv4_model=False,
            req_to_token_pool=req_pool,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "kind,extra_slots", [("regular", 0), ("decode", 0), ("decode", 64), ("hybrid", 64)]
)
@pytest.mark.parametrize("compress", [False, True])
def test_dsa_builder_covers_request_slots(kind, extra_slots, compress, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    req_pool = _request_pool(kind, extra_slots, device)
    pool = _build_pool(req_pool, compress=compress, device=device)
    if not compress:
        assert not pool.kpool_use_compress
        assert pool._compress_tail_k is None
        assert pool._compress_tail_score is None
        return

    highest_slot = max(req_pool.free_slots)
    assert highest_slot == 32 + extra_slots
    if kind == "hybrid":
        # The inherited row allocator can issue the highest slot immediately,
        # even when only one request is admitted.
        assert req_pool.alloc_rows(1) == [highest_slot]

    key = torch.full((1, 128), 7, dtype=torch.bfloat16, device=device)
    score = torch.full_like(key, 11)
    for layer_id in range(2):
        tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
        assert tail_k.shape == tail_score.shape == (highest_slot + 1, 6, 128)
        for slot in (1, highest_slot):
            pool.set_compress_tail_for_request(
                layer_id, torch.tensor(slot, device=device), key, score, 1, 0
            )
            torch.testing.assert_close(tail_k[slot, 0], key[0])
            torch.testing.assert_close(tail_score[slot, 0], score[0])
        assert torch.count_nonzero(tail_k[0]) == 0
        assert torch.count_nonzero(tail_score[0]) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
