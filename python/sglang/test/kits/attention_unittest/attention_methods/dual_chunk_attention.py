from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.attention import (
    dual_chunk_flashattention_backend as _dual_chunk_backend,
)
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import set_global_server_args_for_scheduler
from ..mock_server_args import make_mock_server_args

from .dense_attention import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_CONTEXT_LEN,
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    ReferenceDenseAttention,
    _copy_dense_weights,
    _expand_gqa,
    _make_forward_batch,
    _populate_prefix_kv,
    _split_by_lens,
)

DUAL_CHUNK_CONFIG = {
    "chunk_size": 64,
    "local_size": 16,
    "original_max_position_embeddings": 32768,
    "sparse_attention_enabled": False,
}
DUAL_CHUNK_SPARSE_ALL_COLUMN_CONFIG = {
    **DUAL_CHUNK_CONFIG,
    "sparse_attention_enabled": True,
    "sparse_attention_threshold": 0,
    "sparse_attention_last_q": 16,
    "sparse_attention_config": {
        0: {str(head_id): ("vertical_and_slash", 16, 16, None) for head_id in range(4)}
    },
}
# Same vertical/slash sizes as all-column, but with a threshold so short
# sequences bypass the sparse kernel and fall back to dense prefill. This
# exercises the `current_orig_seq_len > self.sparse_attention_threshold` gate.
DUAL_CHUNK_SPARSE_THRESHOLD_GATED_CONFIG = {
    **DUAL_CHUNK_SPARSE_ALL_COLUMN_CONFIG,
    "sparse_attention_threshold": 100,
}
# Sub-context-window sparse: vertical_size + slash_size < intra K count, so
# the kernel's vertical+slash topk genuinely prunes (the union of selected
# columns + slashes does NOT cover every K column). We can't predict the
# exact selection because it's content-aware top-k by softmax-summed scores,
# but we can verify the sparse path runs, produces finite output, and
# differs from the dense reference (proving pruning happened, not silent
# fallback). See dual_chunk/README.md for the engineering paths to a strict
# correctness reference.
DUAL_CHUNK_SPARSE_SUB_WINDOW_CONFIG = {
    **DUAL_CHUNK_CONFIG,
    "sparse_attention_enabled": True,
    "sparse_attention_threshold": 0,
    "sparse_attention_last_q": 8,
    "sparse_attention_config": {
        0: {str(head_id): ("vertical_and_slash", 8, 8, None) for head_id in range(4)}
    },
}
# `vertical_size=8` (not 4): the production fallback at
# dual_chunk_flashattention_backend.py:1110-1122 appends
# `torch.arange(0, k_states_intra.size(0), max(1, k_states_intra.size(0)/5))`
# when a chunk gets zero vertical indices, which can produce 5 elements
# into a `vertical_size`-slot buffer. vertical_size >= 8 avoids that
# overflow path. This is a known production edge case, not a test bug.

# Unit tests run without distributed initialization. Sparse dual-chunk config
# lookup should see the single-rank default.
_dual_chunk_backend.get_tensor_model_parallel_rank = lambda: 0


@dataclass(frozen=True)
class DualChunkAttentionCase(DenseAttentionCase):
    pass


def make_dual_chunk_cases(backend: str) -> tuple[DualChunkAttentionCase, ...]:
    common = dict(backend=backend, num_heads=4, num_kv_heads=4)
    return (
        DualChunkAttentionCase(
            name="dual_chunk_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_extend_succ_chunk",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(46,),
            extend_lens=(4,),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_decode_succ_chunk",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(48,),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_extend_inter_chunk",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(94,),
            extend_lens=(4,),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_decode_inter_chunk",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(96,),
            **common,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_gqa_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
            backend=backend,
        ),
        DualChunkAttentionCase(
            name="dual_chunk_gqa_decode_inter_chunk",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(96,),
            backend=backend,
        ),
    )


def make_dual_chunk_sparse_cases(backend: str) -> tuple[DualChunkAttentionCase, ...]:
    common = dict(backend=backend, num_heads=4, num_kv_heads=4)
    return (
        DualChunkAttentionCase(
            name="dual_chunk_sparse_prefill_all_columns",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        # Multi-request batch within the first chunk: every request's seq_len <= 16
        # so the sparse path's per-request all-column selection still covers all keys
        # and matches the dense reference. Exercises per-request `cu_seqlens_*` slicing
        # in `_dual_chunk_flash_attn_prefill_func` under sparse enabled.
        DualChunkAttentionCase(
            name="dual_chunk_sparse_prefill_multi_request_first_chunk",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(8, 12),
            **common,
        ),
        # Page-boundary extend (prefix + extend crosses page=16) while staying within
        # one chunk (chunk_size=64). Sparse path still sees <= 16 keys per request so
        # last_q + vertical/slash select all → dense-equivalent.
        DualChunkAttentionCase(
            name="dual_chunk_sparse_prefill_cross_page_first_chunk",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(1,),
            **common,
        ),
    )


def make_dual_chunk_sparse_threshold_gated_cases(
    backend: str,
) -> tuple[DualChunkAttentionCase, ...]:
    common = dict(backend=backend, num_heads=4, num_kv_heads=4)
    return (
        # sparse_attention_enabled=True with threshold=100; with seq_len=16 the
        # backend's `current_orig_seq_len > threshold` check should disable sparse
        # per request and fall back to the dense chunk-flash kernel. The output
        # must match the dense reference exactly.
        DualChunkAttentionCase(
            name="dual_chunk_sparse_threshold_gated_short_seq",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
    )


class TinyDualChunkModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        context_len: int,
        dual_chunk_attention_config: dict | None = None,
    ):
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.hf_config = SimpleNamespace(
            architectures=["TinyDualChunkForCausalLM"],
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            dual_chunk_attention_config=(
                dual_chunk_attention_config or DUAL_CHUNK_CONFIG
            ),
        )
        self.hf_text_config = self.hf_config

    def get_num_attention_heads(self, tp_size: int) -> int:
        assert self.num_attention_heads % tp_size == 0
        return self.num_attention_heads // tp_size

    def get_num_kv_heads(self, tp_size: int) -> int:
        assert self.num_key_value_heads % tp_size == 0
        return self.num_key_value_heads // tp_size


class DualChunkMockModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: DualChunkAttentionCase,
        model_config: TinyDualChunkModelConfig,
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
        self.server_args = make_mock_server_args(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            disable_cuda_graph=disable_cuda_graph,
            disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
            disable_radix_cache=False,
            dp_size=1,
            enable_dp_attention=False,
            kv_cache_dtype="auto",
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
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            dtype=dtype,
            head_num=case.num_kv_heads,
            head_dim=head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=case.page_size)
        self.attn_cp_size = 1
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = False
        self.use_mla_backend = False

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


class ProjectedDualChunkAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_succ_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_inter_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_succ_critical_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_inter_critical_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
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
            num_kv_heads=num_kv_heads,
            layer_id=0,
        )

    def project_qkv(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v

    def project_dual_qkv(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        q_succ = self.q_succ_proj(hidden_states)
        q_inter = self.q_inter_proj(hidden_states)
        q_succ_critical = self.q_succ_critical_proj(hidden_states)
        q_inter_critical = self.q_inter_critical_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, q_succ, q_inter, q_succ_critical, q_inter_critical, k, v

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q, q_succ, q_inter, q_succ_critical, q_inter_critical, k, v = (
            self.project_dual_qkv(hidden_states)
        )
        packed_q = torch.cat(
            (q, q_succ, q_inter, q_succ_critical, q_inter_critical), dim=-1
        )
        attn_output = self.attn(packed_q, k, v, forward_batch)
        return self.o_proj(attn_output)


class ReferenceDualChunkAttention(ReferenceDenseAttention):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        self.q_succ_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_inter_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_succ_critical_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_inter_critical_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def project_dual_qkv(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        q_succ = self.q_succ_proj(hidden_states)
        q_inter = self.q_inter_proj(hidden_states)
        q_succ_critical = self.q_succ_critical_proj(hidden_states)
        q_inter_critical = self.q_inter_critical_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, q_succ, q_inter, q_succ_critical, q_inter_critical, k, v


@dataclass
class DualChunkAttentionFixture:
    case: DualChunkAttentionCase
    runner: DualChunkMockModelRunner
    backend: object
    actual_module: ProjectedDualChunkAttention
    reference_module: ReferenceDualChunkAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


def _set_orig_seq_lens(batch: ForwardBatch, case: DualChunkAttentionCase) -> None:
    batch.orig_seq_lens = torch.tensor(
        case.seq_lens,
        dtype=torch.int32,
        device=batch.seq_lens.device,
    )


def build_dual_chunk_attention_fixture(
    testcase,
    case: DualChunkAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    dual_chunk_attention_config: dict | None = None,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    loc_layout: str = "shuffled_pages",
) -> DualChunkAttentionFixture:
    max_context_len = max(max_context_len, max(case.seq_lens))
    if max_context_len % case.page_size:
        max_context_len = (
            (max_context_len + case.page_size - 1) // case.page_size
        ) * case.page_size

    seed = 3026 + len(case.name) + case.num_kv_heads
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyDualChunkModelConfig(
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        context_len=max_context_len,
        dual_chunk_attention_config=dual_chunk_attention_config,
    )
    runner = DualChunkMockModelRunner(
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

    actual_module = ProjectedDualChunkAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    reference_module = ReferenceDualChunkAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    _copy_dual_chunk_weights(actual_module, reference_module)
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
    _set_orig_seq_lens(forward_batch, case)
    _populate_prefix_kv(
        actual_module,
        case,
        runner,
        prefix_hidden,
        max_context_len=max_context_len,
        loc_fn=loc_fn,
    )

    return DualChunkAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def run_dual_chunk_fixture_eager(fixture: DualChunkAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(fixture.input_hidden, fixture.forward_batch)


def expected_dual_chunk_fixture_output(
    fixture: DualChunkAttentionFixture,
) -> torch.Tensor:
    return _dual_chunk_attention_reference(
        fixture.reference_module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def _copy_dual_chunk_weights(
    actual: ProjectedDualChunkAttention,
    reference: ReferenceDualChunkAttention,
) -> None:
    _copy_dense_weights(actual, reference)
    with torch.no_grad():
        reference.q_succ_proj.weight.copy_(actual.q_succ_proj.weight)
        reference.q_inter_proj.weight.copy_(actual.q_inter_proj.weight)
        reference.q_succ_critical_proj.weight.copy_(actual.q_succ_critical_proj.weight)
        reference.q_inter_critical_proj.weight.copy_(
            actual.q_inter_critical_proj.weight
        )


def _dual_chunk_attention_reference(
    module: ReferenceDualChunkAttention,
    case: DualChunkAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, q_succ, q_inter, _, _, k, v = module.project_dual_qkv(input_hidden)
    q_parts = _split_by_lens(
        q.view(-1, case.num_heads, module.head_dim), case.input_lens
    )
    q_succ_parts = _split_by_lens(
        q_succ.view(-1, case.num_heads, module.head_dim), case.input_lens
    )
    q_inter_parts = _split_by_lens(
        q_inter.view(-1, case.num_heads, module.head_dim), case.input_lens
    )
    k_parts = _split_by_lens(
        k.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    v_parts = _split_by_lens(
        v.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    outputs = []
    chunk_len = DUAL_CHUNK_CONFIG["chunk_size"] - DUAL_CHUNK_CONFIG["local_size"]

    for req_idx, prefix in enumerate(prefix_hidden):
        _, _, _, _, _, prefix_k, prefix_v = module.project_dual_qkv(prefix)
        prefix_k = prefix_k.view(-1, case.num_kv_heads, module.head_dim)
        prefix_v = prefix_v.view(-1, case.num_kv_heads, module.head_dim)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0)
        req_v = torch.cat([prefix_v, v_parts[req_idx]], dim=0)

        for offset, query in enumerate(q_parts[req_idx]):
            query_pos = case.prefix_lens[req_idx] + offset
            current_chunk_start = (query_pos // chunk_len) * chunk_len
            previous_chunk_start = current_chunk_start - chunk_len
            groups = [
                (
                    query,
                    req_k[current_chunk_start : query_pos + 1],
                    req_v[current_chunk_start : query_pos + 1],
                )
            ]

            if previous_chunk_start >= 0:
                groups.append(
                    (
                        q_succ_parts[req_idx][offset],
                        req_k[previous_chunk_start:current_chunk_start],
                        req_v[previous_chunk_start:current_chunk_start],
                    )
                )

            if previous_chunk_start > 0:
                groups.append(
                    (
                        q_inter_parts[req_idx][offset],
                        req_k[:previous_chunk_start],
                        req_v[:previous_chunk_start],
                    )
                )

            score_parts = []
            value_parts = []
            for group_query, group_k, group_v in groups:
                keys = _expand_gqa(group_k.movedim(0, 1), case.num_heads)
                values = _expand_gqa(group_v.movedim(0, 1), case.num_heads)
                scores = (
                    torch.einsum("hd,hkd->hk", group_query.float(), keys.float())
                    * module.scaling
                )
                score_parts.append(scores)
                value_parts.append(values.float())

            scores = torch.cat(score_parts, dim=-1)
            values = torch.cat(value_parts, dim=1)
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, values)
            outputs.append(out.reshape(-1))

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def run_dual_chunk_attention_case(
    testcase,
    case: DualChunkAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    loc_layout: str = "shuffled_pages",
) -> None:
    fixture = build_dual_chunk_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        loc_layout=loc_layout,
    )
    actual = run_dual_chunk_fixture_eager(fixture)
    expected = expected_dual_chunk_fixture_output(fixture)
    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_dual_chunk_sparse_attention_case(
    testcase,
    case: DualChunkAttentionCase,
    *,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> None:
    fixture = build_dual_chunk_attention_fixture(
        testcase,
        case,
        # The local sparse FlashAttention build only includes head_dim=128.
        head_dim=128,
        hidden_size=128,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dual_chunk_attention_config=DUAL_CHUNK_SPARSE_ALL_COLUMN_CONFIG,
    )
    actual = run_dual_chunk_fixture_eager(fixture)
    expected = expected_dual_chunk_fixture_output(fixture)
    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_dual_chunk_sparse_threshold_gated_case(
    testcase,
    case: DualChunkAttentionCase,
    *,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> None:
    fixture = build_dual_chunk_attention_fixture(
        testcase,
        case,
        head_dim=128,
        hidden_size=128,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dual_chunk_attention_config=DUAL_CHUNK_SPARSE_THRESHOLD_GATED_CONFIG,
    )
    actual = run_dual_chunk_fixture_eager(fixture)
    expected = expected_dual_chunk_fixture_output(fixture)
    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_dual_chunk_sparse_sub_window_case(
    testcase,
    case: DualChunkAttentionCase,
    *,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> None:
    """Smoke test for genuine sub-context-window sparse pruning.

    The vertical+slash topk in `_dual_chunk_flash_attn_prefill_func` is
    content-aware (per-head top-k by softmax-summed attention scores), so we
    can't predict the exact v_idx/s_idx and thus can't build a strict
    PyTorch reference without re-implementing ~300 lines of inline production
    logic (see `dual_chunk/README.md` for the engineering paths). This case
    instead verifies:

    1. The sparse path runs without crash on a sub-window config (4 vertical
       + 4 slash, intra K count > 8).
    2. The output is finite (no NaN/inf).
    3. The output shape matches the dense reference.
    4. The output **differs** from the dense reference — proving the kernel
       genuinely pruned rather than silently falling back to dense.
    """
    fixture = build_dual_chunk_attention_fixture(
        testcase,
        case,
        head_dim=128,
        hidden_size=128,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        dual_chunk_attention_config=DUAL_CHUNK_SPARSE_SUB_WINDOW_CONFIG,
    )
    actual = run_dual_chunk_fixture_eager(fixture)
    expected = expected_dual_chunk_fixture_output(fixture)
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertTrue(
        torch.isfinite(actual).all(),
        f"sparse sub-window output has non-finite values: {actual}",
    )
    # Bound the absolute magnitude to catch runaway softmax/scaling bugs.
    max_abs = actual.abs().max().item()
    testcase.assertLess(
        max_abs,
        1e3,
        f"sparse sub-window output magnitude {max_abs} suggests a numerical bug",
    )
    # The sparse path must differ from dense for at least one element by
    # more than bf16 FP-noise (~1e-3 at typical accumulation depth). A
    # silent fallback to dense produces diff ~0 on identical inputs, so the
    # 5e-4 floor cleanly distinguishes "kernel pruned something" from
    # "fallback to dense + FP noise".
    abs_diff = (actual.float() - expected.float()).abs()
    max_diff = abs_diff.max().item()
    testcase.assertGreater(
        max_diff,
        5e-4,
        "sparse sub-window output is too close to dense — the sparse path "
        "may not have actually pruned. Check `sparse_attn_enabled` gate and "
        "config (vertical_size + slash_size should be < intra K count, and "
        "seq_len should exceed the production-hardcoded vertical[:30]=inf "
        "and slash[-100:]=inf always-include heuristics).",
    )


# ---------------------------------------------------------------------------
# Runner-mode helpers (mirror dense conventions; dual-chunk wraps RadixAttention
# so the K-write happens via `save_kv_cache=True` inside the backend forward).
# ---------------------------------------------------------------------------


def make_dual_chunk_case_with_prefix_lens(
    case: DualChunkAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> DualChunkAttentionCase:
    if case.forward_mode.is_decode():
        extend_lens: tuple[int, ...] = ()
    else:
        base = case.extend_lens or (1,)
        if len(prefix_lens) <= len(base):
            extend_lens = base[: len(prefix_lens)]
        else:
            extend_lens = base + (base[-1],) * (len(prefix_lens) - len(base))
    return DualChunkAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def dual_chunk_fixture_inputs(
    fixture: DualChunkAttentionFixture,
) -> dict:
    return {
        "prefix_hidden": fixture.prefix_hidden,
        "input_hidden": fixture.input_hidden,
    }


def make_dual_chunk_random_inputs(
    case: DualChunkAttentionCase,
    fixture: DualChunkAttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict:
    hidden_size = fixture.actual_module.hidden_size
    prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in case.prefix_lens
    ]
    input_hidden = torch.randn(
        case.num_input_tokens, hidden_size, dtype=dtype, device=device
    )
    return {"prefix_hidden": prefix_hidden, "input_hidden": input_hidden}


def make_dual_chunk_replay_inputs(
    case: DualChunkAttentionCase,
    fixture: DualChunkAttentionFixture,
    pad_prefix_lens: tuple[int, ...],
    base_inputs: dict,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict:
    """Pad the base inputs with random prefix/input hidden for the trailing
    padding requests so the replay batch matches the capture-batch shape."""
    hidden_size = fixture.actual_module.hidden_size
    pad_prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in pad_prefix_lens
    ]
    extra_input_tokens = case.num_input_tokens - base_inputs["input_hidden"].shape[0]
    if extra_input_tokens < 0:
        raise ValueError("padded case must have at least as many input tokens as base.")
    pad_input_hidden = torch.randn(
        extra_input_tokens, hidden_size, dtype=dtype, device=device
    )
    return {
        "prefix_hidden": base_inputs["prefix_hidden"] + pad_prefix_hidden,
        "input_hidden": torch.cat(
            [base_inputs["input_hidden"], pad_input_hidden], dim=0
        ),
    }


def prepare_dual_chunk_runner_inputs(
    fixture: DualChunkAttentionFixture,
    case: DualChunkAttentionCase,
    batch: ForwardBatch,
    inputs: dict,
    *,
    max_context_len: int,
) -> None:
    """Rebind inputs on the fixture, set `batch.orig_seq_lens` (dual-chunk
    reads it during forward), and re-populate prefix K cache for the
    (possibly re-shaped) case."""
    fixture.case = case
    fixture.forward_batch = batch
    fixture.prefix_hidden = inputs["prefix_hidden"]
    fixture.input_hidden = inputs["input_hidden"]
    _set_orig_seq_lens(batch, case)
    _populate_prefix_kv(
        fixture.actual_module,
        case,
        fixture.runner,
        fixture.prefix_hidden,
        max_context_len=max_context_len,
    )


def run_dual_chunk_forward(
    fixture: DualChunkAttentionFixture,
    batch: ForwardBatch,
    inputs: dict,
) -> torch.Tensor:
    return fixture.actual_module(inputs["input_hidden"], batch)


def expected_dual_chunk_output_from_inputs(
    fixture: DualChunkAttentionFixture,
    case: DualChunkAttentionCase,
    inputs: dict,
    state,
) -> torch.Tensor:
    del state
    return _dual_chunk_attention_reference(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"],
    )


def dual_chunk_attention_layers(fixture: DualChunkAttentionFixture) -> list:
    return [fixture.actual_module.attn]


def _clone_dual_chunk_cache(fixture: DualChunkAttentionFixture):
    """Snapshot the layer's K cache buffer. Dual-chunk writes K cache via
    `set_kv_buffer` at decode time, so the capture forward's K write
    persists into replay; the snapshot lets us roll it back."""
    layer_id = fixture.actual_module.attn.layer_id
    kv_buf = fixture.runner.token_to_kv_pool.get_key_buffer(layer_id)
    v_buf = fixture.runner.token_to_kv_pool.get_value_buffer(layer_id)
    return (kv_buf.clone(), v_buf.clone())


def _restore_dual_chunk_cache(
    fixture: DualChunkAttentionFixture, state
) -> None:
    layer_id = fixture.actual_module.attn.layer_id
    fixture.runner.token_to_kv_pool.get_key_buffer(layer_id).copy_(state[0])
    fixture.runner.token_to_kv_pool.get_value_buffer(layer_id).copy_(state[1])
