from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.dsa import utils as _dsa_utils
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import set_global_server_args_for_scheduler

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
_dsa_utils.get_attention_cp_size = lambda: 1
_dsa_utils.get_attention_cp_rank = lambda: 0

DSA_PAGE_SIZE = 64
DSA_INDEX_HEAD_DIM = 128
DSA_INDEX_TOPK = 8
DSA_SPARSE_QK_NOPE_HEAD_DIM = 512
DSA_SPARSE_QK_ROPE_HEAD_DIM = 64
DSA_SPARSE_INDEX_TOPK = 128
DSA_SPARSE_ATOL = 1.6e-1
DSA_SPARSE_RTOL = 1.6e-1


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
    ):
        pool_batch_size = runner_batch_size or case.batch_size
        self.device = device
        self.dtype = dtype
        self.kv_cache_dtype = dtype
        self.gpu_id = 0
        self.page_size = case.page_size
        self.model_config = model_config
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.server_args = SimpleNamespace(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            disable_cuda_graph=disable_cuda_graph,
            disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
            disable_radix_cache=False,
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            dp_size=1,
            dsa_decode_backend="flashmla_kv",
            dsa_prefill_cp_mode="round-robin-split",
            dsa_prefill_backend="flashmla_auto",
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
            speculative_num_draft_tokens=0,
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
        self.token_to_kv_pool = DSATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            kv_lora_rank=model_config.kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            layer_num=1,
            device=device,
            index_head_dim=DSA_INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=model_config.kv_lora_rank + model_config.qk_rope_head_dim,
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
    forward_batch = _make_forward_batch(
        case,
        runner,
        max_context_len=max_context_len,
        device=device,
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


def _make_dsa_sparse_topk_rows(case: DSAAttentionCase) -> list[list[int]]:
    rows = []
    for req_idx, input_len in enumerate(case.input_lens):
        prefix_len = case.prefix_lens[req_idx]
        for offset in range(input_len):
            key_count = prefix_len + offset + 1
            key_start = max(0, key_count - DSA_SPARSE_INDEX_TOPK)
            row = list(range(key_start, key_count))
            row.extend([-1] * (DSA_SPARSE_INDEX_TOPK - len(row)))
            rows.append(row)
    return rows


def _populate_dsa_sparse_prefix_kv(
    module: ProjectedDSASparseAttention,
    case: DSAAttentionCase,
    runner: DSAMockModelRunner,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
):
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
            locs.append(
                _token_loc(
                    req_idx,
                    pos,
                    page_size=case.page_size,
                    max_context_len=max_context_len,
                )
            )

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
        index_topk=DSA_SPARSE_INDEX_TOPK,
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
    forward_batch = _make_forward_batch(
        case,
        runner,
        max_context_len=max_context_len,
        device=device,
    )
    _populate_dsa_sparse_prefix_kv(
        actual_module,
        case,
        runner,
        prefix_hidden,
        max_context_len=max_context_len,
    )
    topk_rows = _make_dsa_sparse_topk_rows(case)
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
) -> None:
    fixture = build_dsa_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
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
) -> None:
    fixture = build_dsa_sparse_attention_fixture(
        testcase,
        case,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    actual = run_dsa_sparse_fixture_eager(fixture, testcase)
    expected = expected_dsa_sparse_fixture_output(fixture)
    torch.testing.assert_close(
        actual, expected, atol=DSA_SPARSE_ATOL, rtol=DSA_SPARSE_RTOL
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
    topk_rows = _make_dsa_sparse_topk_rows(case)
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


def _restore_dsa_sparse_cache(
    fixture: DSASparseAttentionFixture, state
) -> None:
    layer_id = fixture.actual_module.attn.layer_id
    fixture.runner.token_to_kv_pool.get_key_buffer(layer_id).copy_(state)
