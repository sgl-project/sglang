from __future__ import annotations

import torch
import triton

from sglang.jit_kernel.benchmark import marker
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

try:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4
except Exception:
    deep_gemm = None
    per_token_cast_to_fp4 = None

HEAD_DIM = 128
NUM_HEADS = 64
BLOCK_KV = 64
NEXT_N = 1

_SM100_SUPPORTED = is_sm100_supported()
_DEEPGEMM_AVAILABLE = deep_gemm is not None and per_token_cast_to_fp4 is not None


def _pack_fp8_cache(k: torch.Tensor, *, num_blocks: int) -> torch.Tensor:
    k = k.view(num_blocks, BLOCK_KV, 1, HEAD_DIM)
    scale = k.abs().float().amax(dim=3, keepdim=True).clamp(1.0e-4) / 448.0
    k_fp8 = (k * (1.0 / scale)).to(torch.float8_e4m3fn)
    buf = torch.empty(
        (num_blocks, BLOCK_KV * (HEAD_DIM + 4)), dtype=torch.uint8, device="cuda"
    )
    buf[:, : BLOCK_KV * HEAD_DIM].copy_(
        k_fp8.view(num_blocks, BLOCK_KV * HEAD_DIM).view(torch.uint8)
    )
    buf[:, BLOCK_KV * HEAD_DIM :].copy_(
        scale.view(num_blocks, BLOCK_KV).view(torch.uint8)
    )
    return buf.view(num_blocks, BLOCK_KV, 1, HEAD_DIM + 4)


def _pack_fp4_cache(
    k_fp4: torch.Tensor,
    k_sf: torch.Tensor,
    *,
    num_blocks: int,
) -> torch.Tensor:
    buf = torch.empty((num_blocks, BLOCK_KV * 68), dtype=torch.uint8, device="cuda")
    buf[:, : BLOCK_KV * 64].view(num_blocks, BLOCK_KV, 64).copy_(
        k_fp4.view(torch.uint8).view(num_blocks, BLOCK_KV, 64)
    )
    buf[:, BLOCK_KV * 64 :].view(num_blocks, BLOCK_KV, 4).copy_(
        k_sf.contiguous().view(torch.uint8).view(num_blocks, BLOCK_KV, 4)
    )
    return buf.view(num_blocks, BLOCK_KV, 1, 68)


def _make_case(batch: int, seq_len_kv: int):
    if deep_gemm is None or per_token_cast_to_fp4 is None:
        raise RuntimeError("DeepGEMM is required for this benchmark.")

    blocks_per_seq = triton.cdiv(seq_len_kv, BLOCK_KV)
    padded_len = blocks_per_seq * BLOCK_KV
    num_blocks = batch * blocks_per_seq
    num_cache_tokens = num_blocks * BLOCK_KV
    page_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        batch, blocks_per_seq
    )
    context_lens = torch.full(
        (batch, NEXT_N), seq_len_kv, dtype=torch.int32, device="cuda"
    )
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, BLOCK_KV, deep_gemm.get_num_sms(), indices=None
    )

    q = torch.randn(
        batch, NEXT_N, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(num_cache_tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch * NEXT_N, NUM_HEADS, device="cuda", dtype=torch.float32)

    q_scale = q.abs().float().amax(dim=-1, keepdim=True).clamp(1.0e-4) / 448.0
    q_fp8 = (q.float() / q_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    weights_fp8 = (
        weights.view(batch, NEXT_N, NUM_HEADS)[:, :, :, None] * q_scale
    ).view(batch * NEXT_N, NUM_HEADS)
    k_cache_fp8 = _pack_fp8_cache(k, num_blocks=num_blocks)

    q_fp4_flat, q_sf_flat = per_token_cast_to_fp4(
        q.view(-1, HEAD_DIM), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_fp4 = q_fp4_flat.view(batch, NEXT_N, NUM_HEADS, HEAD_DIM // 2)
    q_sf = q_sf_flat.view(batch, NEXT_N, NUM_HEADS)
    k_fp4, k_sf = per_token_cast_to_fp4(
        k, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    k_cache_fp4 = _pack_fp4_cache(k_fp4, k_sf, num_blocks=num_blocks)

    return {
        "padded_len": padded_len,
        "page_table": page_table,
        "context_lens": context_lens,
        "schedule": schedule,
        "q_fp8": q_fp8,
        "weights_fp8": weights_fp8,
        "k_cache_fp8": k_cache_fp8,
        "q_fp4": q_fp4,
        "q_sf": q_sf,
        "weights": weights,
        "k_cache_fp4": k_cache_fp4,
    }


def _fp8_indexer(case):
    return deep_gemm.fp8_paged_mqa_logits(
        case["q_fp8"],
        case["k_cache_fp8"],
        case["weights_fp8"],
        case["context_lens"],
        case["page_table"],
        case["schedule"],
        case["padded_len"],
        clean_logits=False,
        indices=None,
    )


def _fp4_indexer(case):
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        (case["q_fp4"], case["q_sf"]),
        case["k_cache_fp4"],
        case["weights"],
        case["context_lens"],
        case["page_table"],
        case["schedule"],
        case["padded_len"],
        clean_logits=False,
        logits_dtype=torch.float32,
        indices=None,
    )


FN_MAP = {
    "fp8": _fp8_indexer,
    "fp4": _fp4_indexer,
}


@marker.parametrize(
    "batch,seq_len_kv",
    [(256, 8192), (256, 32768)],
    [(256, 8192)],
)
@marker.benchmark("impl", ["fp8", "fp4"])
def benchmark(batch: int, seq_len_kv: int, impl: str):
    if not _DEEPGEMM_AVAILABLE:
        marker.skip("DeepGEMM is unavailable.")

    case = _make_case(batch, seq_len_kv)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(case,),
        # paged MQA logits cannot be CUDA-graph captured
        use_cuda_graph=False,
        disable_log_bandwidth=True,  # report us only
    )


if __name__ == "__main__":
    if not _SM100_SUPPORTED:
        print(
            "[skip] DeepSeek V4 FP4 indexer benchmark requires SM100 (Blackwell) CUDA."
        )
    else:
        benchmark.run()
