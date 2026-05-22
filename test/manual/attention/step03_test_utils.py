"""
Shared test utilities for step-03 attention backend init tests.

These tests target the sglang-tests branch (cheng/test/attn-coverage),
which uses the current main-branch API:

  - init_forward_metadata(fb)                                         [eager]
  - init_forward_metadata_capture_cuda_graph(bs, num_tokens, ...)    [graph capture]
  - init_forward_metadata_replay_cuda_graph(bs, ...)                 [graph replay]
  - init_cuda_graph_state(max_bs, max_num_tokens)                    [graph setup]

When step-03 lands the callers switch to init_forward_data{,_out_graph,_in_graph}
and these helpers will be updated accordingly.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

# ---------------------------------------------------------------------------
# MockModelRunner factories
# ---------------------------------------------------------------------------


def build_mha_runner(
    *,
    num_heads: int = 4,
    num_kv_heads: Optional[int] = None,
    head_dim: int = 16,
    max_bs: int = 16,
    max_context_len: int = 64,
    page_size: int = 1,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Return a mock ModelRunner for standard MHA attention backends.

    Covers: TritonAttnBackend, FlashInferAttnBackend, FlashAttentionBackend,
            TRTLLMMHABackend, TorchNativeAttnBackend.
    """
    _num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

    max_total = max_bs * max_context_len
    kv_pool = MHATokenToKVPool(
        size=max_total,
        page_size=page_size,
        dtype=dtype,
        head_num=_num_kv_heads,
        head_dim=head_dim,
        layer_num=2,
        device=device,
        enable_memory_saver=False,
    )

    class _HFTextConfig:
        # Fields FA3 reads (others default to None via getattr)
        num_attention_heads = num_heads
        attn_logit_softcapping = None

    class _HFConfig:
        architectures = ["LlamaForCausalLM"]
        hf_text_config = _HFTextConfig()

    class _ModelConfig:
        attention_arch = AttentionArch.MHA
        context_len = max_context_len
        is_multimodal = False
        is_encoder_decoder = False
        is_local_attention_model = False
        is_hybrid_swa_model = False
        v_head_dim = head_dim
        swa_v_head_dim = None
        num_attention_heads = num_heads
        hidden_size = num_heads * head_dim
        head_dim = head_dim  # FA3 reads model_config.head_dim
        hf_config = _HFConfig()
        hf_text_config = (
            _HFTextConfig()
        )  # FA3 reads model_runner.model_config.hf_text_config

        def get_num_kv_heads(self, tp_size: int = 1) -> int:
            return _num_kv_heads // tp_size

    class _ServerArgs:
        kv_cache_dtype = "auto"
        page_size = page_size  # TritonMultiStepDraft reads server_args.page_size
        speculative_eagle_topk = None
        speculative_num_draft_tokens = 0
        speculative_num_steps = 0
        enable_deterministic_inference = False
        triton_attention_num_kv_splits = 8
        triton_attention_split_tile_size = None
        disable_cuda_graph = False
        disable_piecewise_cuda_graph = True  # FlashInfer reads this on B200
        chunked_prefill_size = -1
        enable_flashinfer_autotune = False
        enable_mis = False
        dllm_algorithm = None
        is_embedding = False  # FA3 reads server_args.is_embedding
        disable_radix_cache = False  # FA3 reads server_args.disable_radix_cache
        enable_dp_attention = False  # FA3 reads via getattr

    class _ReqToTokenPool:
        pass

    req_pool = _ReqToTokenPool()
    req_pool.size = max_bs
    req_pool.req_to_token = torch.zeros(
        max_bs, max_context_len, dtype=torch.int32, device=device
    )

    class _MR:
        pass

    mr = _MR()
    mr.model_config = _ModelConfig()
    mr.server_args = _ServerArgs()
    mr.req_to_token_pool = req_pool
    mr.token_to_kv_pool = kv_pool
    mr.token_to_kv_pool_allocator = kv_pool
    mr.kv_cache_dtype = dtype
    mr.dtype = dtype
    mr.page_size = page_size
    mr.tp_size = 1  # FA3 reads model_runner.tp_size
    mr.gpu_id = 0
    mr.sliding_window_size = None
    mr.attn_cp_size = 1
    mr.is_hybrid_swa = False
    mr.attention_chunk_size = None
    mr.hybrid_gdn_config = None
    mr.kimi_linear_config = None
    mr.linear_attn_model_spec = None
    mr.hisparse_coordinator = None
    mr.device = device
    return mr


def build_mla_runner(
    *,
    num_heads: int = 16,
    kv_lora_rank: int = 64,
    qk_rope_head_dim: int = 32,
    qk_nope_head_dim: int = 64,
    v_head_dim: int = 64,
    max_bs: int = 8,
    max_context_len: int = 32,
    page_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Return a mock ModelRunner for MLA attention backends.

    Covers: FlashInferMLAAttnBackend, FlashMLAAttnBackend,
            CutlassMLABackend, TRTLLMMLABackend.
    """
    max_total = max_bs * max_context_len
    kv_pool = MLATokenToKVPool(
        size=max_total,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=2,
        device=device,
        enable_memory_saver=False,
    )

    class _HFConfig:
        architectures = ["DeepseekV3ForCausalLM"]
        rope_scaling = {
            "type": "yarn",
            "rope_type": "deepseek_yarn",
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        rope_theta = 10000.0

    class _ModelConfig:
        attention_arch = AttentionArch.MLA
        context_len = max_context_len
        is_multimodal = False
        is_encoder_decoder = False
        is_local_attention_model = False
        is_hybrid_swa_model = False
        kv_lora_rank = kv_lora_rank
        qk_rope_head_dim = qk_rope_head_dim
        qk_nope_head_dim = qk_nope_head_dim
        v_head_dim = v_head_dim
        swa_v_head_dim = None
        num_attention_heads = num_heads
        num_kv_heads = 1
        tp_q_head_num = num_heads
        hidden_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        # FlashInferMLA reads model_config.scaling (attention softmax scale)
        scaling = qk_rope_head_dim**-0.5
        hf_config = _HFConfig()

        def get_num_kv_heads(self, tp_size: int = 1) -> int:
            return 1

    class _ServerArgs:
        kv_cache_dtype = "auto"
        page_size = page_size
        speculative_eagle_topk = None
        speculative_num_draft_tokens = 0
        speculative_num_steps = 0
        enable_deterministic_inference = False
        triton_attention_num_kv_splits = 8
        triton_attention_split_tile_size = None
        disable_cuda_graph = False
        disable_piecewise_cuda_graph = True
        chunked_prefill_size = -1
        enable_flashinfer_autotune = False
        enable_mis = False
        dllm_algorithm = None
        is_embedding = False
        disable_radix_cache = False
        enable_dp_attention = False

    class _ReqToTokenPool:
        pass

    req_pool = _ReqToTokenPool()
    req_pool.size = max_bs
    req_pool.req_to_token = torch.zeros(
        max_bs, max_context_len, dtype=torch.int32, device=device
    )

    class _MR:
        pass

    mr = _MR()
    mr.model_config = _ModelConfig()
    mr.server_args = _ServerArgs()
    mr.req_to_token_pool = req_pool
    mr.token_to_kv_pool = kv_pool
    mr.token_to_kv_pool_allocator = kv_pool
    mr.kv_cache_dtype = dtype
    mr.dtype = dtype
    mr.page_size = page_size
    mr.tp_size = 1
    mr.gpu_id = 0
    mr.sliding_window_size = None
    mr.attn_cp_size = 1
    mr.is_hybrid_swa = False
    mr.attention_chunk_size = None
    mr.hybrid_gdn_config = None
    mr.kimi_linear_config = None
    mr.linear_attn_model_spec = None
    mr.hisparse_coordinator = None
    mr.device = device
    return mr


# ---------------------------------------------------------------------------
# ForwardBatch builders
# ---------------------------------------------------------------------------


def fill_req_to_token(mr, bs: int, seq_len: int):
    """Write sequential token indices into req_to_token_pool."""
    for i in range(bs):
        mr.req_to_token_pool.req_to_token[i, :seq_len] = torch.arange(
            i * seq_len, (i + 1) * seq_len, dtype=torch.int32
        )


def make_decode_batch(bs: int, seq_len: int, device: str = "cuda") -> ForwardBatch:
    """Build a ForwardBatch for DECODE mode (bs requests, each at seq_len)."""
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=torch.randint(0, 1000, (bs,), device=device),
        req_pool_indices=torch.arange(bs, dtype=torch.int32, device=device),
        seq_lens=torch.full((bs,), seq_len, dtype=torch.int32, device=device),
        out_cache_loc=torch.arange(
            bs * seq_len, bs * (seq_len + 1), dtype=torch.int32, device=device
        ),
        seq_lens_sum=bs * seq_len,
        seq_lens_cpu=torch.full((bs,), seq_len, dtype=torch.int32),
    )


def make_extend_batch(
    bs: int,
    extend_len: int,
    prefix_len: int = 0,
    device: str = "cuda",
) -> ForwardBatch:
    """Build a ForwardBatch for EXTEND mode."""
    total_len = prefix_len + extend_len
    num_new = bs * extend_len
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.randint(0, 1000, (num_new,), device=device),
        req_pool_indices=torch.arange(bs, dtype=torch.int32, device=device),
        seq_lens=torch.full((bs,), total_len, dtype=torch.int32, device=device),
        out_cache_loc=torch.arange(
            bs * prefix_len, bs * total_len, dtype=torch.int32, device=device
        ),
        seq_lens_sum=bs * total_len,
        seq_lens_cpu=torch.full((bs,), total_len, dtype=torch.int32),
        extend_seq_lens=torch.full((bs,), extend_len, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=[extend_len] * bs,
        extend_prefix_lens=torch.full(
            (bs,), prefix_len, dtype=torch.int32, device=device
        ),
        extend_prefix_lens_cpu=[prefix_len] * bs,
    )


# ---------------------------------------------------------------------------
# Q/K/V and layer helpers
# ---------------------------------------------------------------------------


def make_qkv(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Return (q, k, v) random tensors for attention forward."""
    nkv = num_kv_heads if num_kv_heads is not None else num_heads
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_tokens, nkv, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_tokens, nkv, head_dim, dtype=dtype, device=device)
    return q, k, v


def make_radix_attention(
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
) -> RadixAttention:
    """Create a minimal RadixAttention layer for testing."""
    nkv = num_kv_heads if num_kv_heads is not None else num_heads
    return RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        num_kv_heads=nkv,
        layer_id=0,
    )


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_no_nan_inf(test_case, tensor: torch.Tensor, tag: str = ""):
    """Assert the tensor contains no NaN or Inf values."""
    prefix = f"[{tag}] " if tag else ""
    test_case.assertFalse(
        torch.isnan(tensor).any().item(), f"{prefix}output contains NaN"
    )
    test_case.assertFalse(
        torch.isinf(tensor).any().item(), f"{prefix}output contains Inf"
    )


def gpu_arch_sm() -> Optional[int]:
    """Return the SM version of GPU 0, or None if no CUDA device."""
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return major * 10 + minor


# ---------------------------------------------------------------------------
# API-compatibility helpers
# ---------------------------------------------------------------------------
# Step-03 removed init_forward_metadata_capture/replay_cuda_graph (they raise
# NotImplementedError) and replaced them with init_forward_data_out_graph(fb).
# These wrappers try the old API first, then fall back to the new API, so the
# same test file works on both the pre-step-03 branch and the step-03 branch.


def init_graph_capture(
    backend,
    fb: "ForwardBatch",
    bs: int,
    num_tokens: int,
    req_pool_indices: "torch.Tensor",
    seq_lens: "torch.Tensor",
    forward_mode: "ForwardMode",
) -> None:
    """Graph capture init: old API or step-03 init_forward_data_out_graph."""
    try:
        backend.init_forward_metadata_capture_cuda_graph(
            bs, num_tokens, req_pool_indices, seq_lens, None, forward_mode, None
        )
    except NotImplementedError:
        backend.init_forward_data_out_graph(fb)


def init_graph_replay(
    backend,
    fb: "ForwardBatch",
    bs: int,
    req_pool_indices: "torch.Tensor",
    seq_lens: "torch.Tensor",
    seq_lens_sum: int,
    forward_mode: "ForwardMode",
    seq_lens_cpu: "torch.Tensor",
) -> None:
    """Graph replay init: old API or step-03 init_forward_data_out_graph."""
    try:
        backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            None,
            forward_mode,
            None,
            seq_lens_cpu,
        )
    except NotImplementedError:
        backend.init_forward_data_out_graph(fb)


def _model_exists(model_id: str) -> bool:
    """Return True if the model appears accessible (local path or HF offline cache)."""
    import os

    if os.path.isdir(model_id):
        return True
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    slug = "models--" + model_id.replace("/", "--")
    return os.path.isdir(os.path.join(hf_home, "hub", slug))
