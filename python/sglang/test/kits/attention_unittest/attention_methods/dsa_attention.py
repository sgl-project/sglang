from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import set_global_server_args_for_scheduler

from ..mock_server_args import make_mock_server_args
from .dense_attention import (
    DEFAULT_DEVICE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    ReferenceDenseAttention,
    _copy_dense_weights,
    _dense_attention_reference,
    _make_forward_batch,
    _split_by_lens,
    _token_loc,
)

# Unit tests run without distributed initialization. DSA context-parallel probes
# should see the single-rank default.
_parallel_override = get_parallel().override(attn_cp_size=1, attn_cp_rank=0)
_parallel_override.__enter__()

DSA_PAGE_SIZE = 64
DSA_INDEX_HEAD_DIM = 128
DSA_INDEX_TOPK = 8
DSA_SPARSE_QK_NOPE_HEAD_DIM = 512
DSA_SPARSE_QK_ROPE_HEAD_DIM = 64
DSA_SPARSE_INDEX_TOPK = 128
DSA_SPARSE_ATOL = 1.6e-1
DSA_SPARSE_RTOL = 1.6e-1
# Tolerance for FP8 KV cache. The actual path stores K as FP8 (with
# per-128-channel scales) and the kernel reads from that quantized
# cache; the reference compares against the original BF16 K (so a
# silent pack/write bug can't self-cancel — same separation principle
# as the DSV4 SWA reference). Empirically max_diff lands around
# 0.05–0.1 vs the BF16 reference; 0.2 absorbs that headroom.
DSA_SPARSE_FP8_ATOL = 2.0e-1
DSA_SPARSE_FP8_RTOL = 2.0e-1


@dataclass(frozen=True)
class DSAAttentionCase(DenseAttentionCase):
    pass


def make_dsa_dense_fallback_cases(backend: str) -> tuple[DSAAttentionCase, ...]:
    common = dict(
        backend=backend,
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=4,
        page_size=DSA_PAGE_SIZE,
    )
    return (
        DSAAttentionCase(
            name="dsa_mha_one_shot_no_prefix_ragged",
            prefix_lens=(0, 0, 0),
            extend_lens=(3, 8, 17),
            **common,
        ),
        DSAAttentionCase(
            name="dsa_mha_one_shot_no_prefix_exact_page",
            prefix_lens=(0,),
            extend_lens=(DSA_PAGE_SIZE,),
            **common,
        ),
        # Zero-prefix extend with seq_len exactly one token below the page
        # boundary. Paired with `_no_prefix_exact_page` (seq_len == page) and
        # `_cross_page_boundary` (seq_len == page + 1), this exercises the
        # three-way `< page`, `== page`, `> page` partition required by
        # PLAN.md's "Required input cases" list while staying under the
        # MHA_ONE_SHOT KV threshold (2048).
        DSAAttentionCase(
            name="dsa_mha_one_shot_no_prefix_seq_below_page",
            prefix_lens=(0,),
            extend_lens=(DSA_PAGE_SIZE - 1,),
            **common,
        ),
        # Ragged batch whose three requests span below / exactly at / above
        # the page boundary in a single forward. The dense-fallback K-write
        # walks the full per-request KV concatenation, so the page-aligned
        # request must allocate a fresh page without spilling into the next
        # request's page table.
        DSAAttentionCase(
            name="dsa_mha_one_shot_ragged_below_at_above_page",
            prefix_lens=(0, 0, 0),
            extend_lens=(DSA_PAGE_SIZE - 1, DSA_PAGE_SIZE, DSA_PAGE_SIZE + 1),
            **common,
        ),
        DSAAttentionCase(
            name="dsa_mha_one_shot_prefix_ragged",
            prefix_lens=(3, 8),
            extend_lens=(2, 3),
            **common,
        ),
        # Prefix + extend crosses a page boundary (`page_size=64`), so the dense
        # fallback path must read both the existing page and the freshly-allocated
        # next page during the MHA_ONE_SHOT projection-and-attention.
        DSAAttentionCase(
            name="dsa_mha_one_shot_cross_page_boundary",
            prefix_lens=(DSA_PAGE_SIZE - 1,),
            extend_lens=(2,),
            **common,
        ),
        # Prefix exactly fills one page and extend opens the next: covers the
        # page-aligned prefix branch of `_token_loc` / `req_to_token` setup.
        DSAAttentionCase(
            name="dsa_mha_one_shot_prefix_exact_page",
            prefix_lens=(DSA_PAGE_SIZE,),
            extend_lens=(2,),
            **common,
        ),
        # prefix + extend exactly equals one page so total length lands on the
        # boundary without crossing it.
        DSAAttentionCase(
            name="dsa_mha_one_shot_total_exact_page",
            prefix_lens=(DSA_PAGE_SIZE - 16,),
            extend_lens=(16,),
            **common,
        ),
    )


def make_dsa_sparse_cases(backend: str) -> tuple[DSAAttentionCase, ...]:
    common = dict(
        backend=backend,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
    )
    return (
        DSAAttentionCase(
            name="dsa_sparse_prefill_flashmla_sparse_topk",
            forward_mode=ForwardMode.EXTEND,
            # Keep this above the default dense one-shot threshold so the backend
            # naturally selects the DSA sparse prefill implementation.
            prefix_lens=(2048,),
            extend_lens=(1,),
            **common,
        ),
        # Sparse prefill with multi-token extend: per-query trailing-topk rows must
        # advance with `offset`, exercising the prefill dispatch on more than one
        # query token while staying above the dense one-shot threshold.
        DSAAttentionCase(
            name="dsa_sparse_prefill_long_extend",
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(2048,),
            extend_lens=(4,),
            **common,
        ),
        # Sparse prefill with multiple requests above the dense one-shot threshold,
        # so the flashmla_sparse path runs with bsz > 1.
        DSAAttentionCase(
            name="dsa_sparse_prefill_multi_request",
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(2048, 2048),
            extend_lens=(1, 1),
            **common,
        ),
        DSAAttentionCase(
            name="dsa_sparse_decode_flashmla_kv_topk",
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(127, 128),
            **common,
        ),
        # Decode with prefix < topk so trailing-row indices include the -1 padding
        # tail and the kernel must mask the unused topk slots.
        DSAAttentionCase(
            name="dsa_sparse_decode_short_prefix_padding",
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(64, 96),
            **common,
        ),
        # Decode with ragged prefix across 3 requests: covers (key_count < topk),
        # (key_count == topk), and (key_count > topk) at the same time so the
        # per-request topk slicing must vary across the batch.
        DSAAttentionCase(
            name="dsa_sparse_decode_ragged_prefix",
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(64, 128, 192),
            **common,
        ),
        # Long-prefix decode: prefix >> topk so the trailing topk window walks
        # deep into the KV cache and exercises page-table indexing past many pages.
        DSAAttentionCase(
            name="dsa_sparse_decode_long_prefix",
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(2048,),
            **common,
        ),
    )


class TinyDSAModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        context_len: int,
        num_kv_heads: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int = 0,
        kv_lora_rank: int | None = None,
        index_topk: int = DSA_INDEX_TOPK,
    ):
        qk_nope_head_dim = (
            qk_nope_head_dim if qk_nope_head_dim is not None else head_dim
        )
        kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else qk_nope_head_dim
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.hf_config = SimpleNamespace(
            architectures=["DeepseekV32ForCausalLM"],
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            index_head_dim=DSA_INDEX_HEAD_DIM,
            index_n_heads=1,
            index_topk=index_topk,
            num_hidden_layers=1,
        )
        self.hf_text_config = self.hf_config


class DSAMockModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: DSAAttentionCase,
        model_config: TinyDSAModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        head_dim: int,
        disable_cuda_graph: bool = True,
        disable_piecewise_cuda_graph: bool = True,
        runner_batch_size: int | None = None,
        dsa_prefill_backend: str = "flashmla_auto",
        dsa_decode_backend: str = "flashmla_kv",
        fp8_kv_cache: bool = False,
    ):
        pool_batch_size = runner_batch_size or case.batch_size
        self.device = device
        self.dtype = dtype
        # `kv_cache_dtype` is the dtype the *storage* uses. For FP8 KV
        # cache the pool stores packed FP8 nope + scales + BF16 rope at
        # 656 bytes/token while the model still projects K/V in BF16;
        # `set_mla_kv_buffer` does the quantize on the way in.
        self.kv_cache_dtype = torch.float8_e4m3fn if fp8_kv_cache else dtype
        # For TARGET_VERIFY / DRAFT_EXTEND, the DSA backend uses
        # `self.speculative_num_draft_tokens` to size `seqlens_expanded`
        # (`dsa_backend.py:482-486,510-515`). When zero, deep_gemm's
        # `paged_mqa_logits_metadata` JIT-compiles with
        # `kAlignedBatchSize=0U`, which fails to compile. We auto-derive
        # the draft-token count from `case.extend_lens` so the
        # speculative paths produce a non-empty `seqlens_expanded`.
        if (
            case.forward_mode.is_target_verify()
            or case.forward_mode.is_draft_extend_v2()
        ):
            spec_num_draft_tokens = max(case.extend_lens) if case.extend_lens else 1
        else:
            spec_num_draft_tokens = 0
        self.gpu_id = 0
        self.canary_manager = None
        self.page_size = case.page_size
        self.model_config = model_config
        self.tp_size = 1
        self._kernel_warmed_up = True
        self.dp_size = 1
        self.pp_size = 1
        self.ps = SimpleNamespace(
            tp_size=1, dp_size=1, pp_size=1, tp_rank=0, pp_rank=0, dp_rank=0, gpu_id=0
        )
        self.server_args = make_mock_server_args(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(
                    backend=Backend.DISABLED if disable_cuda_graph else Backend.FULL,
                ),
                prefill=PhaseConfig(
                    backend=(
                        Backend.DISABLED
                        if (disable_cuda_graph or disable_piecewise_cuda_graph)
                        else Backend.TC_PIECEWISE
                    ),
                ),
            ),
            disable_radix_cache=False,
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            dp_size=1,
            dsa_decode_backend=dsa_decode_backend,
            dsa_prefill_cp_mode="round-robin-split",
            dsa_prefill_backend=dsa_prefill_backend,
            device=device,
            enable_deterministic_inference=False,
            enable_dp_attention=False,
            enable_dsa_prefill_context_parallel=False,
            enable_mis=False,
            is_embedding=False,
            kv_cache_dtype="auto",
            max_running_requests=None,
            mem_fraction_static=0.8,
            model_path=None,
            pp_size=1,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=0,
            speculative_num_draft_tokens=spec_num_draft_tokens,
            speculative_num_steps=0,
            tp_size=1,
            triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None,
        )
        set_global_server_args_for_scheduler(self.server_args)
        self.req_to_token_pool = ReqToTokenPool(
            size=pool_batch_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        max_token_loc = case.page_size + pool_batch_size * max_context_len
        # FP8 KV cache: packed nope_fp8 (dim_nope) + scales (num_tiles*4) +
        # rope_bf16_bytes (dim_rope*2) = 528 + 128 = 656 bytes/token for
        # the production DSA shape (dim_nope=512, dim_rope=64). The pool
        # flips `dsa_kv_cache_store_fp8=True` iff
        # `dtype=torch.float8_e4m3fn AND override_kv_cache_dim is not None`
        # (`DSATokenToKVPool.__init__`), so both must be passed in tandem.
        if fp8_kv_cache:
            pool_dtype = torch.float8_e4m3fn
            dim_nope = model_config.kv_lora_rank
            dim_rope = model_config.qk_rope_head_dim
            num_tiles = dim_nope // DSATokenToKVPool.quant_block_size
            # uint8 byte layout: [nope_fp8 (dim_nope B)] + [scales (num_tiles*4 B)] +
            # [rope_bf16 (dim_rope*2 B)]
            pool_kv_cache_dim = dim_nope + num_tiles * 4 + dim_rope * 2
        else:
            pool_dtype = dtype
            pool_kv_cache_dim = (
                model_config.kv_lora_rank + model_config.qk_rope_head_dim
            )
        self.token_to_kv_pool = DSATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            kv_lora_rank=model_config.kv_lora_rank,
            dtype=pool_dtype,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            layer_num=1,
            device=device,
            index_head_dim=DSA_INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=pool_kv_cache_dim,
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=case.page_size)
        self.attn_cp_size = 1
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = False
        self.use_mla_backend = True

    @property
    def hybrid_gdn_config(self):
        return None

    @property
    def hybrid_lightning_config(self):
        return None

    @property
    def kimi_linear_config(self):
        return None

    @property
    def linear_attn_model_spec(self):
        return None

    @property
    def mamba2_config(self):
        return None

    @property
    def mambaish_config(self):
        return None


class ProjectedDSADenseFallbackAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=num_heads,
            layer_id=0,
        )

    def project_qkv(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v


class ProjectedDSASparseAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.q_nope_proj = nn.Linear(
            hidden_size,
            num_heads * qk_nope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_rope_proj = nn.Linear(
            hidden_size,
            num_heads * qk_rope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_nope_proj = nn.Linear(
            hidden_size,
            qk_nope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_rope_proj = nn.Linear(
            hidden_size,
            qk_rope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            num_heads * qk_nope_head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=qk_nope_head_dim,
        )

    def project_q(self, hidden_states: torch.Tensor):
        q_nope = self.q_nope_proj(hidden_states)
        q_rope = self.q_rope_proj(hidden_states).view(
            -1, self.num_heads, self.qk_rope_head_dim
        )
        return q_nope, q_rope

    def project_k(self, hidden_states: torch.Tensor):
        k_nope = self.k_nope_proj(hidden_states)
        k_rope = self.k_rope_proj(hidden_states).view(
            -1, self.num_kv_heads, self.qk_rope_head_dim
        )
        return k_nope, k_rope


@dataclass
class DSAAttentionFixture:
    case: DSAAttentionCase
    runner: DSAMockModelRunner
    backend: object
    actual_module: ProjectedDSADenseFallbackAttention
    reference_module: ReferenceDenseAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


@dataclass
class DSASparseAttentionFixture:
    case: DSAAttentionCase
    runner: DSAMockModelRunner
    backend: object
    actual_module: ProjectedDSASparseAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor
    topk_indices: torch.Tensor
    topk_rows: list[list[int]]
    # The fixture's per-row trailing-topk index width. Defaults to the
    # production-shape `DSA_SPARSE_INDEX_TOPK=128`; the tilelang variant
    # bumps it to 2048 (`tilelang_sparse_fwd` asserts `topk == 2048`).
    # Carried on the fixture so re-derivation paths (e.g. CG-runner
    # `make_dsa_sparse_random_inputs`) can rebuild rows with the same
    # width.
    index_topk: int = DSA_SPARSE_INDEX_TOPK


def build_dsa_attention_fixture(
    testcase,
    case: DSAAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    dsa_prefill_backend: str = "flashmla_auto",
    dsa_decode_backend: str = "flashmla_kv",
    loc_layout: str = "shuffled_pages",
) -> DSAAttentionFixture:
    max_context_len = max(max_context_len, max(case.seq_lens))
    if max_context_len % case.page_size:
        max_context_len = (
            (max_context_len + case.page_size - 1) // case.page_size
        ) * case.page_size

    seed = 4026 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyDSAModelConfig(
        num_heads=case.num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        context_len=max_context_len,
    )
    runner = DSAMockModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        head_dim=head_dim,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
        dsa_prefill_backend=dsa_prefill_backend,
        dsa_decode_backend=dsa_decode_backend,
    )
    try:
        backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    actual_module = ProjectedDSADenseFallbackAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    reference_module = ReferenceDenseAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    _copy_dense_weights(actual_module, reference_module)
    prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in case.prefix_lens
    ]
    input_hidden = torch.randn(
        case.num_input_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    from .dense_attention import make_loc_fn as _dense_make_loc_fn

    loc_fn = _dense_make_loc_fn(
        loc_layout,
        batch_size=case.batch_size,
        seq_lens=case.seq_lens,
        prefix_lens=case.prefix_lens,
        page_size=case.page_size,
        max_context_len=max_context_len,
        seed=seed,
    )
    forward_batch = _make_forward_batch(
        case,
        runner,
        max_context_len=max_context_len,
        device=device,
        loc_fn=loc_fn,
    )
    return DSAAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def _make_dsa_sparse_topk_rows(
    case: DSAAttentionCase,
    *,
    index_topk: int = DSA_SPARSE_INDEX_TOPK,
    pattern: str = "trailing",
) -> list[list[int]]:
    """Build per-query topk index rows.

    The reference (`expected_dsa_sparse_fixture_output`) gathers Q/K via the
    same `topk_rows`, so any valid permutation of keys in `[0, key_count)`
    produces a matching reference. Patterns:

    - ``trailing``: last `topk` keys, i.e. `[key_count - topk, key_count)`.
      Mirrors the production indexer's most common selection for short prefixes.
    - ``strided``: every other key from `[0, key_count)` until `topk` slots
      are filled, then `-1` padding. Exercises the kernel's non-contiguous
      gather path (top-k by attention score is not naturally trailing in
      production for long prefixes).
    - ``head_tail``: first `topk/2` keys + last `topk/2` keys. Forces a
      genuinely sparse layout that drops the middle of the KV window.
    """
    rows = []
    for req_idx, input_len in enumerate(case.input_lens):
        prefix_len = case.prefix_lens[req_idx]
        for offset in range(input_len):
            key_count = prefix_len + offset + 1
            if pattern == "trailing":
                key_start = max(0, key_count - index_topk)
                row = list(range(key_start, key_count))
            elif pattern == "strided":
                # Stride-2 from 0, falling back to trailing if the strided
                # range can't fill topk slots.
                strided = list(range(0, key_count, 2))[:index_topk]
                if len(strided) < min(index_topk, key_count):
                    extra = [
                        k for k in range(key_count - 1, -1, -1) if k not in strided
                    ]
                    needed = min(index_topk, key_count) - len(strided)
                    strided.extend(extra[:needed])
                row = sorted(strided)
            elif pattern == "head_tail":
                # First topk/2 + last topk/2, clipped to key_count and
                # deduplicated to avoid double-counting when key_count < topk.
                half = max(1, index_topk // 2)
                head = list(range(0, min(half, key_count)))
                tail = list(range(max(half, key_count - half), key_count))
                row = sorted(set(head) | set(tail))
            else:
                raise ValueError(f"unknown topk index pattern: {pattern!r}")
            row.extend([-1] * (index_topk - len(row)))
            rows.append(row)
    return rows


def _populate_dsa_sparse_prefix_kv(
    module: ProjectedDSASparseAttention,
    case: DSAAttentionCase,
    runner: DSAMockModelRunner,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
    loc_fn=None,
):
    if loc_fn is None:

        def loc_fn(req_idx: int, pos: int) -> int:
            return _token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )

    locs = []
    k_nope_parts = []
    k_rope_parts = []
    for req_idx, prefix in enumerate(prefix_hidden):
        if prefix.shape[0] == 0:
            continue
        k_nope, k_rope = module.project_k(prefix)
        k_nope_parts.append(k_nope.view(-1, 1, module.qk_nope_head_dim))
        k_rope_parts.append(k_rope)
        for pos in range(prefix.shape[0]):
            locs.append(loc_fn(req_idx, pos))

    if not locs:
        return

    runner.token_to_kv_pool.set_mla_kv_buffer(
        module.attn,
        torch.tensor(locs, dtype=torch.int64, device=runner.device),
        torch.cat(k_nope_parts, dim=0),
        torch.cat(k_rope_parts, dim=0),
    )


def build_dsa_sparse_attention_fixture(
    testcase,
    case: DSAAttentionCase,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    dsa_prefill_backend: str = "flashmla_auto",
    dsa_decode_backend: str = "flashmla_kv",
    fp8_kv_cache: bool = False,
    index_topk: int = DSA_SPARSE_INDEX_TOPK,
    index_pattern: str = "trailing",
    loc_layout: str = "shuffled_pages",
) -> DSASparseAttentionFixture:
    max_context_len = max(max_context_len, max(case.seq_lens))
    if max_context_len % case.page_size:
        max_context_len = (
            (max_context_len + case.page_size - 1) // case.page_size
        ) * case.page_size

    seed = 5026 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    head_dim = DSA_SPARSE_QK_NOPE_HEAD_DIM + DSA_SPARSE_QK_ROPE_HEAD_DIM
    model_config = TinyDSAModelConfig(
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        context_len=max_context_len,
        qk_nope_head_dim=DSA_SPARSE_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=DSA_SPARSE_QK_ROPE_HEAD_DIM,
        kv_lora_rank=DSA_SPARSE_QK_NOPE_HEAD_DIM,
        index_topk=index_topk,
    )
    runner = DSAMockModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        head_dim=head_dim,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
        dsa_prefill_backend=dsa_prefill_backend,
        dsa_decode_backend=dsa_decode_backend,
        fp8_kv_cache=fp8_kv_cache,
    )
    try:
        backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    actual_module = ProjectedDSASparseAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        qk_nope_head_dim=DSA_SPARSE_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=DSA_SPARSE_QK_ROPE_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in case.prefix_lens
    ]
    input_hidden = torch.randn(
        case.num_input_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    from .dense_attention import make_loc_fn as _dense_make_loc_fn

    loc_fn = _dense_make_loc_fn(
        loc_layout,
        batch_size=case.batch_size,
        seq_lens=case.seq_lens,
        prefix_lens=case.prefix_lens,
        page_size=case.page_size,
        max_context_len=max_context_len,
        seed=seed,
    )
    forward_batch = _make_forward_batch(
        case,
        runner,
        max_context_len=max_context_len,
        device=device,
        loc_fn=loc_fn,
    )
    _populate_dsa_sparse_prefix_kv(
        actual_module,
        case,
        runner,
        prefix_hidden,
        max_context_len=max_context_len,
        loc_fn=loc_fn,
    )
    topk_rows = _make_dsa_sparse_topk_rows(
        case, index_topk=index_topk, pattern=index_pattern
    )
    topk_indices = torch.tensor(topk_rows, dtype=torch.int32, device=device)

    return DSASparseAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
        topk_indices=topk_indices,
        topk_rows=topk_rows,
        index_topk=index_topk,
    )


def run_dsa_fixture_eager(fixture: DSAAttentionFixture, testcase) -> torch.Tensor:
    case = fixture.case
    input_parts = _split_by_lens(fixture.input_hidden, case.input_lens)
    kv_hidden = torch.cat(
        [
            torch.cat([fixture.prefix_hidden[req_idx], input_part], dim=0)
            for req_idx, input_part in enumerate(input_parts)
        ],
        dim=0,
    )
    q, _, _ = fixture.actual_module.project_qkv(fixture.input_hidden)
    _, k, v = fixture.actual_module.project_qkv(kv_hidden)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        if not fixture.backend.use_mha:
            testcase.skipTest("DSA MHA_ONE_SHOT dense fallback is not selected here.")
        attn_output = fixture.actual_module.attn(
            q,
            k,
            v,
            fixture.forward_batch,
            save_kv_cache=False,
        )
        attn_output = attn_output.reshape(
            -1, fixture.case.num_heads * fixture.actual_module.head_dim
        )
        return fixture.actual_module.o_proj(attn_output)


def expected_dsa_fixture_output(fixture: DSAAttentionFixture) -> torch.Tensor:
    return _dense_attention_reference(
        fixture.reference_module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def run_dsa_sparse_fixture_eager(
    fixture: DSASparseAttentionFixture, testcase
) -> torch.Tensor:
    module = fixture.actual_module
    q_nope, q_rope = module.project_q(fixture.input_hidden)
    k_nope, k_rope = module.project_k(fixture.input_hidden)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        if fixture.case.forward_mode.is_extend_without_speculative():
            testcase.assertFalse(
                fixture.backend.use_mha,
                "DSA sparse prefill case unexpectedly selected dense MHA fallback.",
            )
        attn_output = module.attn(
            q_nope,
            k_nope,
            k_nope,
            fixture.forward_batch,
            k_rope=k_rope,
            q_rope=q_rope,
            topk_indices=fixture.topk_indices,
        )
        attn_output = attn_output.reshape(
            -1, fixture.case.num_heads * module.qk_nope_head_dim
        )
        return module.o_proj(attn_output)


def expected_dsa_sparse_fixture_output(
    fixture: DSASparseAttentionFixture,
) -> torch.Tensor:
    module = fixture.actual_module
    dtype = fixture.input_hidden.dtype
    q_nope, q_rope = module.project_q(fixture.input_hidden)
    q_nope = q_nope.view(-1, fixture.case.num_heads, module.qk_nope_head_dim)
    input_parts = _split_by_lens(fixture.input_hidden, fixture.case.input_lens)
    outputs = []
    q_idx = 0

    for req_idx, prefix in enumerate(fixture.prefix_hidden):
        req_hidden = torch.cat([prefix, input_parts[req_idx]], dim=0)
        req_k_nope, req_k_rope = module.project_k(req_hidden)
        req_k_nope = req_k_nope.view(-1, module.qk_nope_head_dim)
        req_k_rope = req_k_rope.view(-1, module.qk_rope_head_dim)
        req_k = torch.cat([req_k_nope, req_k_rope], dim=-1)

        for _ in range(fixture.case.input_lens[req_idx]):
            selected = torch.tensor(
                fixture.topk_rows[q_idx],
                dtype=torch.int64,
                device=fixture.input_hidden.device,
            )
            selected = selected[selected >= 0]
            query = torch.cat([q_nope[q_idx], q_rope[q_idx]], dim=-1).float()
            keys = req_k[selected].float()
            values = req_k_nope[selected].float()
            scores = torch.einsum("hd,kd->hk", query, keys) * module.attn.scaling
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,kd->hd", probs, values)
            outputs.append(out.reshape(-1))
            q_idx += 1

    return module.o_proj(torch.stack(outputs, dim=0).to(dtype))


def run_dsa_attention_case(
    testcase,
    case: DSAAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
    loc_layout: str = "shuffled_pages",
) -> None:
    fixture = build_dsa_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        loc_layout=loc_layout,
    )
    actual = run_dsa_fixture_eager(fixture, testcase)
    expected = expected_dsa_fixture_output(fixture)
    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_dsa_sparse_attention_case(
    testcase,
    case: DSAAttentionCase,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
    dsa_prefill_backend: str = "flashmla_auto",
    dsa_decode_backend: str = "flashmla_kv",
    fp8_kv_cache: bool = False,
    index_topk: int = DSA_SPARSE_INDEX_TOPK,
    index_pattern: str = "trailing",
    loc_layout: str = "shuffled_pages",
) -> None:
    fixture = build_dsa_sparse_attention_fixture(
        testcase,
        case,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dsa_prefill_backend=dsa_prefill_backend,
        dsa_decode_backend=dsa_decode_backend,
        fp8_kv_cache=fp8_kv_cache,
        index_topk=index_topk,
        index_pattern=index_pattern,
        loc_layout=loc_layout,
    )
    actual = run_dsa_sparse_fixture_eager(fixture, testcase)
    expected = expected_dsa_sparse_fixture_output(fixture)
    atol = DSA_SPARSE_FP8_ATOL if fp8_kv_cache else DSA_SPARSE_ATOL
    rtol = DSA_SPARSE_FP8_RTOL if fp8_kv_cache else DSA_SPARSE_RTOL
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Implementation-variant matrix
# ---------------------------------------------------------------------------
#
# DSA exposes multiple kernel implementations selectable via
# `--dsa-prefill-backend` and `--dsa-decode-backend`. Production picks one of:
#
#   `flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`, `trtllm`, `aiter`
#
# (`flashmla_auto` resolves to `flashmla_sparse` or `flashmla_kv` per the
# `dsa_kv_cache_store_fp8` flag; see `set_dsa_prefill_impl`.) Each impl maps
# to a distinct kernel path in `dsa_backend.py`; the hardware/SDK
# availability differs and is gated below.
#
# `dsa_impl_capability(impl)` returns `(supported: bool, skip_reason: str)`
# so test methods can iterate over a flat list of impls and emit `skipTest`
# per impl on hardware that doesn't expose it. Capability is independent of
# whether the fixture's specific shape (topk, dtype, num_heads) matches the
# impl's accepted inputs — that is documented at the per-impl call site.


# Hardware/SDK availability for each DSA implementation variant.
# Values are `(supported: bool, reason: str)` — `reason` is shown in
# `skipTest` when `supported=False`.
def dsa_impl_capability(impl: str) -> tuple[bool, str]:
    """Return `(supported, reason)` for a DSA prefill/decode implementation.

    Capability checks are conservative: a returned `supported=True` means
    the kernel can be constructed and dispatched on this device; the
    fixture must still match the impl's shape contract (e.g., tilelang's
    `topk == 2048` requirement)."""
    import torch as _torch

    from sglang.srt.utils import is_hip

    major, minor = _torch.cuda.get_device_capability()

    if impl == "flashmla_sparse" or impl == "flashmla_kv":
        try:
            from sgl_kernel.flash_mla import (  # noqa: F401
                flash_mla_sparse_fwd,
                flash_mla_with_kvcache,
            )
        except ImportError as exc:
            return False, f"sgl_kernel.flash_mla unavailable: {exc}"
        if major < 9:
            return False, f"{impl} requires SM>=9.0, got SM{major}.x"
        return True, ""

    if impl == "fa3":
        try:
            from sglang.jit_kernel.flash_attention import (  # noqa: F401
                flash_attn_with_kvcache,
            )
        except ImportError as exc:
            return False, f"sglang.jit_kernel.flash_attention unavailable: {exc}"
        # sgl-kernel flash_attn is compiled for SM9.x (Hopper) only;
        # it raises NotImplementedError on Blackwell (SM10.x+).
        if major < 9 or major >= 10:
            return False, f"fa3 requires SM9.x (Hopper), got SM{major}.x"
        return True, ""

    if impl == "tilelang":
        try:
            from sglang.srt.layers.attention.dsa.tilelang_kernel import (  # noqa: F401
                tilelang_sparse_fwd,
            )
        except ImportError as exc:
            return False, f"tilelang_kernel unavailable: {exc}"
        # Container gate (KNOWN_FAILURES.md §2): on SM10.x the tilelang JIT
        # generates a `wait_wgmma` WGMMA-sync intrinsic that the container's
        # MMA template library doesn't ship, raising `RuntimeError: namespace
        # "tl" has no member "wait_wgmma"` at PTX compilation time. Skip
        # tilelang on SM>=10 until the container is re-imaged with an SM10.x
        # tilelang version. Override with `SGLANG_TEST_DSA_TILELANG_FORCE=1`
        # if you've verified the wait_wgmma intrinsic is present.
        import os as _os

        if major >= 10 and not _os.environ.get("SGLANG_TEST_DSA_TILELANG_FORCE"):
            return (
                False,
                f"tilelang JIT on SM{major}.{minor} needs `wait_wgmma` template "
                f"that the container doesn't ship "
                f"(KNOWN_FAILURES.md §2). Re-image or set "
                f"SGLANG_TEST_DSA_TILELANG_FORCE=1 to override.",
            )
        # `tilelang_sparse_fwd` asserts `topk == 2048`; our existing sparse
        # fixture uses `DSA_SPARSE_INDEX_TOPK=128`. Tests requesting the
        # tilelang variant must build a topk=2048 fixture variant.
        return True, ""

    if impl == "trtllm":
        # TRT-LLM Gen FMHA / MLA require Blackwell SM10.0 (B200 NVL).
        # SM10.3 (GB300) raises "Missing TRTLLM-GEN kernel" at runtime because
        # the kernel binary in the container isn't compiled for sm_103.
        # Require exactly SM10.0 (same constraint as cutlass_mla) until the
        # container ships sm_103-compiled TRTLLM-GEN kernels.
        if major != 10 or minor != 0:
            return (
                False,
                f"trtllm requires SM10.0 (Blackwell B200), got SM{major}.{minor}",
            )
        try:
            import flashinfer  # noqa: F401
        except ImportError as exc:
            return False, f"flashinfer unavailable: {exc}"
        return True, ""

    if impl == "aiter":
        if not is_hip():
            return False, "aiter is HIP/AMD only"
        try:
            from aiter.mla import (  # noqa: F401
                mla_decode_fwd,
                mla_prefill_fwd,
            )
        except ImportError as exc:
            return False, f"aiter unavailable: {exc}"
        return True, ""

    if impl == "flashmla_auto":
        # `flashmla_auto` resolves to flashmla_sparse / flashmla_kv at
        # forward time depending on `dsa_kv_cache_store_fp8`; both leaf
        # impls share the same SDK requirement, so flag based on those.
        return dsa_impl_capability("flashmla_sparse")

    return False, f"unknown DSA impl `{impl}`"


# Sets of impls covered by the variant matrix. Test methods iterate over
# these and `skipTest` per impl when the capability gate trips.
DSA_PREFILL_IMPL_VARIANTS: tuple[str, ...] = (
    "flashmla_sparse",
    "flashmla_kv",
    "fa3",
    "tilelang",
    "trtllm",
    "aiter",
)
DSA_DECODE_IMPL_VARIANTS: tuple[str, ...] = (
    "flashmla_sparse",
    "flashmla_kv",
    "fa3",
    "tilelang",
    "trtllm",
    "aiter",
)

# Impls that accept an FP8-stored K cache. The flashmla *sparse* and FA3
# kernels require BF16 K (`kv must have dtype torch::kBFloat16`), so they
# fall back to the inline-quantize-of-bf16 path that production *doesn't*
# take in FP8 deployments. The `flashmla_kv` decode kernel and *both*
# flashmla prefill kernels are the production-relevant FP8 paths.
DSA_FP8_COMPATIBLE_PREFILL_IMPLS: frozenset[str] = frozenset(
    {"flashmla_sparse", "flashmla_kv", "flashmla_auto"}
)
DSA_FP8_COMPATIBLE_DECODE_IMPLS: frozenset[str] = frozenset(
    {"flashmla_kv", "flashmla_auto"}
)


def run_dsa_sparse_prefill_impl_variant_case(
    testcase,
    case: DSAAttentionCase,
    impl: str,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
) -> None:
    """Run a sparse-prefill case under a forced `dsa_prefill_backend=impl`.

    `tilelang` requires `topk == 2048`, which the default sparse fixture
    (`DSA_SPARSE_INDEX_TOPK=128`) does not satisfy; the call site skips
    tilelang explicitly with that reason so the gate does not silently
    pass.
    """
    supported, reason = dsa_impl_capability(impl)
    if not supported:
        testcase.skipTest(f"DSA prefill impl `{impl}` not supported: {reason}")
    if impl == "tilelang":
        testcase.skipTest(
            "DSA tilelang prefill requires topk=2048; the shared sparse fixture "
            f"uses topk={DSA_SPARSE_INDEX_TOPK}. A topk=2048 fixture variant is "
            "needed to exercise this path."
        )
    if not case.forward_mode.is_extend_without_speculative():
        raise ValueError(
            "run_dsa_sparse_prefill_impl_variant_case expects an EXTEND case."
        )
    run_dsa_sparse_attention_case(
        testcase,
        case,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dsa_prefill_backend=impl,
    )


def run_dsa_sparse_decode_impl_variant_case(
    testcase,
    case: DSAAttentionCase,
    impl: str,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
) -> None:
    """Run a sparse-decode case under a forced `dsa_decode_backend=impl`."""
    supported, reason = dsa_impl_capability(impl)
    if not supported:
        testcase.skipTest(f"DSA decode impl `{impl}` not supported: {reason}")
    if impl == "tilelang":
        testcase.skipTest(
            "DSA tilelang decode requires topk=2048; the shared sparse fixture "
            f"uses topk={DSA_SPARSE_INDEX_TOPK}. A topk=2048 fixture variant is "
            "needed to exercise this path."
        )
    if not case.forward_mode.is_decode():
        raise ValueError(
            "run_dsa_sparse_decode_impl_variant_case expects a DECODE case."
        )
    run_dsa_sparse_attention_case(
        testcase,
        case,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dsa_decode_backend=impl,
    )


DSA_SPARSE_TILELANG_INDEX_TOPK = 2048
# Tilelang's `tilelang_sparse_fwd` asserts `topk == 2048` at
# `dsa/tilelang_kernel.py:1345`, so the tilelang variant runs against a
# separate fixture instance with this index width. Cases need
# `prefix >= 2048` so the trailing-topk-row builder fills out a real
# 2048-wide row (rather than `[0, ..., key_count - 1, -1, ..., -1]`).


def run_dsa_sparse_tilelang_prefill_case(
    testcase,
    case: DSAAttentionCase,
) -> None:
    """Tilelang sparse-prefill on its dedicated topk=2048 fixture. The
    other prefill kernels (`flashmla_sparse`, `flashmla_kv`, `fa3`) are
    capable of running against topk=2048 too but are already covered by
    the topk=128 variant matrix; this method is scoped to the kernel
    that *requires* topk=2048."""
    supported, reason = dsa_impl_capability("tilelang")
    if not supported:
        testcase.skipTest(f"DSA tilelang impl not supported: {reason}")
    if not case.forward_mode.is_extend_without_speculative():
        raise ValueError("run_dsa_sparse_tilelang_prefill_case expects an EXTEND case.")
    run_dsa_sparse_attention_case(
        testcase,
        case,
        dsa_prefill_backend="tilelang",
        index_topk=DSA_SPARSE_TILELANG_INDEX_TOPK,
    )


def run_dsa_sparse_tilelang_decode_case(
    testcase,
    case: DSAAttentionCase,
) -> None:
    """Tilelang sparse-decode on its dedicated topk=2048 fixture."""
    supported, reason = dsa_impl_capability("tilelang")
    if not supported:
        testcase.skipTest(f"DSA tilelang impl not supported: {reason}")
    if not case.forward_mode.is_decode():
        raise ValueError("run_dsa_sparse_tilelang_decode_case expects a DECODE case.")
    run_dsa_sparse_attention_case(
        testcase,
        case,
        dsa_decode_backend="tilelang",
        index_topk=DSA_SPARSE_TILELANG_INDEX_TOPK,
    )


def run_dsa_sparse_fp8_prefill_case(
    testcase,
    case: DSAAttentionCase,
    *,
    dsa_prefill_backend: str = "flashmla_auto",
) -> None:
    """FP8-KV-cache prefill. With `flashmla_sparse` + EXTEND + non-empty
    prefix, `get_topk_transform_method` returns `RAGGED` (the only path
    that exercises `dequantize_k_cache_paged` + the
    `topk_indices_offset` shift). With `flashmla_kv` or `flashmla_auto`
    it stays on `PAGED` topk; the auto resolver picks `flashmla_kv` for
    FP8 KV cache (`set_dsa_prefill_impl`), so `flashmla_auto` and
    `flashmla_kv` test the same code path."""
    if dsa_prefill_backend not in DSA_FP8_COMPATIBLE_PREFILL_IMPLS:
        testcase.skipTest(
            f"DSA prefill impl `{dsa_prefill_backend}` does not support FP8 KV "
            f"cache (only `flashmla_sparse`, `flashmla_kv`, and `flashmla_auto` "
            f"read FP8 K directly; others require BF16 K)."
        )
    if not case.forward_mode.is_extend_without_speculative():
        raise ValueError("run_dsa_sparse_fp8_prefill_case expects an EXTEND case.")
    run_dsa_sparse_attention_case(
        testcase,
        case,
        dsa_prefill_backend=dsa_prefill_backend,
        fp8_kv_cache=True,
    )


def run_dsa_sparse_fp8_decode_case(
    testcase,
    case: DSAAttentionCase,
    *,
    dsa_decode_backend: str = "flashmla_kv",
) -> None:
    """FP8-KV-cache decode. Only `flashmla_kv` (and `flashmla_auto`
    which resolves to it for FP8) accepts an FP8-stored K cache;
    `flashmla_sparse` and `fa3` decode kernels assert BF16 K and would
    fall back to the inline-quantize-of-bf16 path that production
    doesn't take in FP8 deployments."""
    if dsa_decode_backend not in DSA_FP8_COMPATIBLE_DECODE_IMPLS:
        testcase.skipTest(
            f"DSA decode impl `{dsa_decode_backend}` does not support FP8 KV "
            f"cache (only `flashmla_kv` / `flashmla_auto` read FP8 K directly)."
        )
    if not case.forward_mode.is_decode():
        raise ValueError("run_dsa_sparse_fp8_decode_case expects a DECODE case.")
    run_dsa_sparse_attention_case(
        testcase,
        case,
        dsa_decode_backend=dsa_decode_backend,
        fp8_kv_cache=True,
    )


def run_dsa_sparse_cuda_graph_decode_impl_variant_case(
    testcase,
    case: DSAAttentionCase,
    impl: str,
):
    """CUDA-graph decode replay parametrized over `dsa_decode_backend=impl`.

    Imported lazily because the runner module imports this module — the
    circular dependency only resolves at call time.
    """
    supported, reason = dsa_impl_capability(impl)
    if not supported:
        testcase.skipTest(f"DSA CG decode impl `{impl}` not supported: {reason}")
    if impl == "tilelang":
        testcase.skipTest(
            "DSA tilelang decode requires topk=2048; the shared sparse fixture "
            f"uses topk={DSA_SPARSE_INDEX_TOPK}."
        )
    if not case.forward_mode.is_decode():
        raise ValueError(
            "run_dsa_sparse_cuda_graph_decode_impl_variant_case expects a "
            "DECODE case."
        )
    from ..runner_modes.cuda_graph_decode_runner import (
        run_dsa_sparse_cuda_graph_decode_case,
    )

    run_dsa_sparse_cuda_graph_decode_case(
        testcase,
        case,
        dsa_decode_backend=impl,
    )


def run_dsa_sparse_speculative_forward_mode_case(
    testcase,
    case: DSAAttentionCase,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DSA_PAGE_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEFAULT_DEVICE,
    dsa_decode_backend: str = "flashmla_kv",
) -> None:
    """Run a sparse case with a speculative forward mode (TARGET_VERIFY,
    DRAFT_EXTEND, or DRAFT_EXTEND_V2). DSA dispatches both
    `is_target_verify()` and `is_draft_extend_v2()` through
    `dsa_decode_impl` (`dsa_backend.py:1352-1358`), so the kernel
    selection matches plain DECODE but `seqlens_expanded` is computed
    differently per forward mode (`dsa_backend.py:469-529`).
    `DSAMockModelRunner.__init__` derives
    `speculative_num_draft_tokens` from `case.extend_lens` for the
    speculative modes so deep_gemm's `paged_mqa_logits_metadata` JIT
    compiles with a non-zero `kAlignedBatchSize`."""
    if not (
        case.forward_mode.is_target_verify() or case.forward_mode.is_draft_extend_v2()
    ):
        raise ValueError(
            "run_dsa_sparse_speculative_forward_mode_case expects a "
            "TARGET_VERIFY, DRAFT_EXTEND, or DRAFT_EXTEND_V2 case."
        )
    run_dsa_sparse_attention_case(
        testcase,
        case,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dsa_decode_backend=dsa_decode_backend,
    )


# ---------------------------------------------------------------------------
# Runner-mode helpers for DSA dense fallback split-op extend
# ---------------------------------------------------------------------------


def make_dsa_case_with_prefix_lens(
    case: DSAAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> DSAAttentionCase:
    """Build a variant case with new `prefix_lens`. For DECODE we drop
    `extend_lens` (input_lens derives `(1,) * batch_size`); for EXTEND we
    clip/pad the original `extend_lens` to match the new batch shape."""
    if case.forward_mode.is_decode():
        extend_lens: tuple[int, ...] = ()
    else:
        base = case.extend_lens or (1,)
        if len(prefix_lens) <= len(base):
            extend_lens = base[: len(prefix_lens)]
        else:
            extend_lens = base + (base[-1],) * (len(prefix_lens) - len(base))
    return DSAAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def dsa_fixture_inputs(fixture: DSAAttentionFixture) -> dict[str, Any]:
    return {
        "prefix_hidden": fixture.prefix_hidden,
        "input_hidden": fixture.input_hidden,
    }


def make_dsa_random_inputs(
    case: DSAAttentionCase,
    fixture: DSAAttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    hidden_size = fixture.actual_module.hidden_size
    prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in case.prefix_lens
    ]
    input_hidden = torch.randn(
        case.num_input_tokens, hidden_size, dtype=dtype, device=device
    )
    return {"prefix_hidden": prefix_hidden, "input_hidden": input_hidden}


def make_dsa_token_padded_inputs(
    _case: DSAAttentionCase,
    fixture: DSAAttentionFixture,
    static_num_tokens: int,
    base_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    """Pad `input_hidden` to a fixed static token count. Prefix is kept
    unchanged because DSA dense fallback uses inline K (projected from
    prefix+input each call) — there's no K-cache write at attn time."""
    del fixture
    hidden_size = base_inputs["input_hidden"].shape[1]
    raw_num_tokens = base_inputs["input_hidden"].shape[0]
    if static_num_tokens < raw_num_tokens:
        raise ValueError("static_num_tokens must cover the live input token count.")
    if static_num_tokens == raw_num_tokens:
        return base_inputs
    pad_num_tokens = static_num_tokens - raw_num_tokens
    return {
        "prefix_hidden": base_inputs["prefix_hidden"],
        "input_hidden": torch.cat(
            [
                base_inputs["input_hidden"],
                torch.randn(pad_num_tokens, hidden_size, dtype=dtype, device=device),
            ],
            dim=0,
        ),
    }


def prepare_dsa_runner_inputs(
    fixture: DSAAttentionFixture,
    case: DSAAttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, Any],
    *,
    max_context_len: int,
) -> None:
    """Write the new inputs onto the fixture. DSA dense fallback doesn't
    pre-populate K cache (K is passed inline via `attn(q, k, v, ...)`),
    so this just rebinds `prefix_hidden`/`input_hidden`."""
    del max_context_len
    fixture.case = case
    fixture.forward_batch = batch
    fixture.prefix_hidden = inputs["prefix_hidden"]
    fixture.input_hidden = inputs["input_hidden"]


def run_dsa_forward(
    fixture: DSAAttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, Any],
) -> torch.Tensor:
    """DSA dense fallback forward. Mirrors `run_dsa_fixture_eager` but
    takes `(fixture, batch, inputs)` to fit the generic runner adapter
    contract, and does not call `testcase.skipTest` — case selection is
    the caller's responsibility."""
    case = fixture.case
    module = fixture.actual_module
    input_hidden = inputs["input_hidden"]
    # `input_hidden` may have trailing padding for split-op static-token
    # contracts; project only the live token rows for QKV. The kernel
    # respects `num_token_non_padded_cpu` via the metadata.
    live_input_hidden = input_hidden[: case.num_input_tokens]
    input_parts = _split_by_lens(live_input_hidden, case.input_lens)
    kv_hidden = torch.cat(
        [
            torch.cat([inputs["prefix_hidden"][req_idx], input_part], dim=0)
            for req_idx, input_part in enumerate(input_parts)
        ],
        dim=0,
    )
    q, _, _ = module.project_qkv(input_hidden)
    _, k, v = module.project_qkv(kv_hidden)
    backend = fixture.backend
    attn_output = module.attn(q, k, v, batch, save_kv_cache=False)
    attn_output = attn_output.reshape(-1, case.num_heads * module.head_dim)
    return module.o_proj(attn_output)


def expected_dsa_output_from_inputs(
    fixture: DSAAttentionFixture,
    case: DSAAttentionCase,
    inputs: dict[str, Any],
    state,
) -> torch.Tensor:
    """Pure-PyTorch dense-attention reference (DSA dense fallback IS plain
    MHA, no sparse selection). The `state` arg is unused — dense fallback
    has no recurrent state."""
    del state
    return _dense_attention_reference(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"][: case.num_input_tokens],
    )


def dsa_attention_layers(fixture: DSAAttentionFixture) -> list:
    """Return the RadixAttention layers the backend forwards through. The
    split-op runner uses this to install per-layer
    `num_token_non_padded_cpu` metadata before forward."""
    return [fixture.actual_module.attn]


def _clone_dsa_cache(fixture: DSAAttentionFixture):
    """No-op snapshot — DSA dense fallback has no recurrent state. The
    K cache is populated inline per forward call via `save_kv_cache=False`,
    so capture/replay independence doesn't require state snapshotting."""
    del fixture
    return None


def _restore_dsa_cache(fixture: DSAAttentionFixture, state) -> None:
    del fixture, state


# ---------------------------------------------------------------------------
# Runner-mode helpers for DSA SPARSE attention (DECODE / EXTEND via flashmla)
# ---------------------------------------------------------------------------
# These mirror the dense-fallback helpers above but consume the sparse
# fixture (`DSASparseAttentionFixture`) which carries `topk_indices` /
# `topk_rows` and uses a different `module.attn(...)` signature with
# `q_rope=`, `k_rope=`, `topk_indices=` kwargs.


def make_dsa_sparse_case_with_prefix_lens(
    case: DSAAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> DSAAttentionCase:
    """Build a sparse-case variant with new `prefix_lens`. Mirrors the
    dense-fallback shape but uses `num_kv_heads=1` (sparse always uses
    MLA-style latent KV)."""
    if case.forward_mode.is_decode():
        extend_lens: tuple[int, ...] = ()
    else:
        base = case.extend_lens or (1,)
        if len(prefix_lens) <= len(base):
            extend_lens = base[: len(prefix_lens)]
        else:
            extend_lens = base + (base[-1],) * (len(prefix_lens) - len(base))
    return DSAAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def dsa_sparse_fixture_inputs(
    fixture: DSASparseAttentionFixture,
) -> dict[str, Any]:
    return {
        "input_hidden": fixture.input_hidden,
        "topk_indices": fixture.topk_indices,
        "topk_rows": fixture.topk_rows,
    }


def make_dsa_sparse_random_inputs(
    case: DSAAttentionCase,
    fixture: DSASparseAttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    hidden_size = fixture.actual_module.hidden_size
    input_hidden = torch.randn(
        case.num_input_tokens, hidden_size, dtype=dtype, device=device
    )
    topk_rows = _make_dsa_sparse_topk_rows(case, index_topk=fixture.index_topk)
    topk_indices = torch.tensor(topk_rows, dtype=torch.int32, device=device)
    return {
        "input_hidden": input_hidden,
        "topk_indices": topk_indices,
        "topk_rows": topk_rows,
    }


def make_dsa_sparse_replay_inputs(
    _case: DSAAttentionCase,
    fixture: DSASparseAttentionFixture,
    _pad_prefix_lens: tuple[int, ...],
    base_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    del fixture, dtype, device
    return base_inputs


def prepare_dsa_sparse_runner_inputs(
    fixture: DSASparseAttentionFixture,
    case: DSAAttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, Any],
    *,
    max_context_len: int,
) -> None:
    """Rebind sparse inputs onto the fixture and re-populate prefix KV
    cache for the (possibly re-shaped) case so the kernel reads the
    expected MLA latent values."""
    fixture.case = case
    fixture.forward_batch = batch
    fixture.input_hidden = inputs["input_hidden"]
    fixture.topk_indices = inputs["topk_indices"]
    if "topk_rows" in inputs:
        fixture.topk_rows = inputs["topk_rows"]
    _populate_dsa_sparse_prefix_kv(
        fixture.actual_module,
        case,
        fixture.runner,
        fixture.prefix_hidden,
        max_context_len=max_context_len,
    )


def run_dsa_sparse_forward(
    fixture: DSASparseAttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, Any],
) -> torch.Tensor:
    """DSA sparse forward — mirrors `run_dsa_sparse_fixture_eager` but
    takes `(fixture, batch, inputs)` and re-passes `topk_indices` from
    the inputs dict so capture and replay see consistent values."""
    module = fixture.actual_module
    input_hidden = inputs["input_hidden"]
    q_nope, q_rope = module.project_q(input_hidden)
    k_nope, k_rope = module.project_k(input_hidden)
    attn_output = module.attn(
        q_nope,
        k_nope,
        k_nope,
        batch,
        k_rope=k_rope,
        q_rope=q_rope,
        topk_indices=inputs["topk_indices"],
    )
    attn_output = attn_output.reshape(
        -1, fixture.case.num_heads * module.qk_nope_head_dim
    )
    return module.o_proj(attn_output)


def expected_dsa_sparse_output_from_inputs(
    fixture: DSASparseAttentionFixture,
    case: DSAAttentionCase,
    inputs: dict[str, Any],
    state,
) -> torch.Tensor:
    """Pure-PyTorch sparse-topk reference. The reference reads
    `fixture.topk_rows` (already updated by `prepare_dsa_sparse_runner_inputs`),
    so `inputs` and `state` are unused."""
    del case, inputs, state
    return expected_dsa_sparse_fixture_output(fixture)


def dsa_sparse_attention_layers(fixture: DSASparseAttentionFixture) -> list:
    return [fixture.actual_module.attn]


def _clone_dsa_sparse_cache(fixture: DSASparseAttentionFixture):
    """Snapshot the MLA KV cache so capture's per-decode-token write
    doesn't bleed into replay state. Returns a clone of the layer's
    K buffer."""
    layer_id = fixture.actual_module.attn.layer_id
    kv_buf = fixture.runner.token_to_kv_pool.get_key_buffer(layer_id)
    return kv_buf.clone()


def _restore_dsa_sparse_cache(fixture: DSASparseAttentionFixture, state) -> None:
    layer_id = fixture.actual_module.attn.layer_id
    fixture.runner.token_to_kv_pool.get_key_buffer(layer_id).copy_(state)
