from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.attention import (
    dual_chunk_flashattention_backend as _dual_chunk_backend,
)
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_parallel

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
# columns + slashes does NOT cover every K column). The selection is
# deterministic but content-aware; this case compares the sgl-kernel sparse
# attention output against a torch implementation that consumes the same
# block/column metadata.
DUAL_CHUNK_SPARSE_SUB_WINDOW_CONFIG = {
    **DUAL_CHUNK_CONFIG,
    "sparse_attention_enabled": True,
    "sparse_attention_threshold": 0,
    "sparse_attention_last_q": 8,
    "sparse_attention_config": {
        0: {str(head_id): ("vertical_and_slash", 4, 4, None) for head_id in range(4)}
    },
}

# Unit tests run without distributed initialization. Sparse dual-chunk config
# lookup should see the single-rank default.
_parallel_override = get_parallel().override(tp_rank=0)
_parallel_override.__enter__()


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


def make_dual_chunk_sparse_sub_window_cases(
    backend: str,
) -> tuple[DualChunkAttentionCase, ...]:
    common = dict(backend=backend, num_heads=4, num_kv_heads=4)
    return (
        DualChunkAttentionCase(
            name="dual_chunk_sparse_prefill_sub_window_seq128",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(128,),
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
        self.quantization = None
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
        self.hf_config.get_text_config = lambda: self.hf_config
        self.hf_text_config = self.hf_config
        self.linear_attn_registry_result = None

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
        self.canary_manager = None
        self.page_size = case.page_size
        self.model_config = model_config
        self.tp_size = 1
        self._kernel_warmed_up = True
        self.dp_size = 1
        self.pp_size = 1
        self.ps = ParallelState.trivial()
        self._server_args_override = get_context().override_server_args(
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
        self.server_args = self._server_args_override.install()
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


@dataclass(frozen=True)
class _DualChunkSparseStageSelection:
    stage: str
    q_len: int
    kv_len: int
    vertical_indices: tuple[tuple[int, ...], ...]
    slash_indices: tuple[tuple[int, ...], ...]


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


def _dual_chunk_sparse_fallback_indices_reference(
    seq_len: int, max_count: int, device: torch.device
) -> torch.Tensor:
    count = min(int(max_count), seq_len)
    if count <= 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    step = max(1, (seq_len + count - 1) // count)
    return torch.arange(0, seq_len, step, dtype=torch.int64, device=device)[:count]


def _dual_chunk_sum_all_diagonal_matrix(mat: torch.Tensor) -> torch.Tensor:
    h, n, m = mat.shape
    zero_mat = torch.zeros((h, n, n), dtype=mat.dtype, device=mat.device)
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided(
        (1, n, n + m), (n * (2 * n + m), 2 * n + m + 1, 1)
    )
    return torch.sum(mat_strided, 1)[:, 1:]


def _normalise_sparse_stage_indices(
    indices: torch.Tensor,
    counts: torch.Tensor,
    *,
    descending: bool,
) -> tuple[tuple[int, ...], ...]:
    indices = indices.detach().reshape(1, counts.numel(), -1)
    counts = counts.detach().to(torch.int64).cpu().tolist()
    stage_indices = []
    for head_i, count in enumerate(counts):
        head_indices = indices[0, head_i, :count].to(torch.int64)
        head_indices = head_indices.sort(descending=descending).values
        stage_indices.append(tuple(int(idx) for idx in head_indices.cpu().tolist()))
    return tuple(stage_indices)


def _make_sparse_stage_selection(
    stage: str,
    q_len: int,
    kv_len: int,
    vertical_indices: list[torch.Tensor],
    slash_indices: list[torch.Tensor],
) -> _DualChunkSparseStageSelection:
    return _DualChunkSparseStageSelection(
        stage=stage,
        q_len=q_len,
        kv_len=kv_len,
        vertical_indices=tuple(
            tuple(int(idx) for idx in head_indices.sort().values.cpu().tolist())
            for head_indices in vertical_indices
        ),
        slash_indices=tuple(
            tuple(
                int(idx)
                for idx in head_indices.sort(descending=True).values.cpu().tolist()
            )
            for head_indices in slash_indices
        ),
    )


def _capture_sparse_stage_selection(
    stage: str,
    query: torch.Tensor,
    key: torch.Tensor,
    vertical_indices: torch.Tensor,
    slash_indices: torch.Tensor,
    vertical_counts: torch.Tensor | None,
    slash_counts: torch.Tensor | None,
) -> _DualChunkSparseStageSelection:
    assert vertical_counts is not None
    assert slash_counts is not None
    return _DualChunkSparseStageSelection(
        stage=stage,
        q_len=query.shape[2],
        kv_len=key.shape[2],
        vertical_indices=_normalise_sparse_stage_indices(
            vertical_indices, vertical_counts, descending=False
        ),
        slash_indices=_normalise_sparse_stage_indices(
            slash_indices, slash_counts, descending=True
        ),
    )


def _dual_chunk_sparse_selection_reference(
    fixture: DualChunkAttentionFixture,
) -> list[_DualChunkSparseStageSelection]:
    """Reference DCA's content-aware top-k split and fallback indices.

    This covers the DCA-specific part of sparse prefill: vertical/slash top-k
    selection, intra/succ/inter splitting, and empty-stage fallback. The lower
    level 64x64 block conversion and sparse kernel math are covered separately.
    """
    case = fixture.case
    assert case.batch_size == 1
    assert case.prefix_lens == (0,)
    assert case.num_heads == case.num_kv_heads

    module = fixture.reference_module
    (
        q,
        q_succ,
        q_inter,
        q_succ_critical,
        q_inter_critical,
        k,
        _,
    ) = module.project_dual_qkv(fixture.input_hidden)
    q = q.view(-1, case.num_heads, module.head_dim)
    q_succ = q_succ.view(-1, case.num_heads, module.head_dim)
    q_inter = q_inter.view(-1, case.num_heads, module.head_dim)
    q_succ_critical = q_succ_critical.view(-1, case.num_heads, module.head_dim)
    q_inter_critical = q_inter_critical.view(-1, case.num_heads, module.head_dim)
    k = k.view(-1, case.num_kv_heads, module.head_dim)

    config = DUAL_CHUNK_SPARSE_SUB_WINDOW_CONFIG
    chunk_len = config["chunk_size"] - config["local_size"]
    softmax_scale = module.scaling
    scaling_factor = (
        0.1
        * torch.log(
            torch.tensor(case.seq_lens[0] / config["original_max_position_embeddings"])
        )
        + 1.0
    ).clamp(min=1)
    softmax_scale *= float(scaling_factor.item())

    head_config = config["sparse_attention_config"][0]
    heads_vertical_size = []
    heads_slash_size = []
    for head_i in range(case.num_heads):
        ty, vertical_size, slash_size, _ = head_config[str(head_i)]
        assert ty == "vertical_and_slash"
        if vertical_size == 30:
            vertical_size += 100
        heads_vertical_size.append(vertical_size)
        heads_slash_size.append(slash_size)

    selections = []
    k_length = k.shape[0]
    begin = k_length - q.shape[0]
    while begin < k_length:
        prev_chunk_end_pos = (begin // chunk_len) * chunk_len
        next_chunk_end_pos = prev_chunk_end_pos + chunk_len
        end = min(next_chunk_end_pos, k_length)
        qbegin = begin - (k_length - q.shape[0])
        qend = end - (k_length - q.shape[0])
        chunk_q_len = qend - qbegin
        last_q_size = min(chunk_q_len, config["sparse_attention_last_q"])

        q_states_intra = q[qbegin:qend]
        k_states_intra = k[prev_chunk_end_pos:end]
        qk_chunks = [
            (q_states_intra.transpose(0, 1)[:, -last_q_size:] * softmax_scale)
            @ k_states_intra.permute(1, 2, 0)
        ]
        stage_kv_lens = {"intra": k_states_intra.size(0)}

        if prev_chunk_end_pos - chunk_len >= 0:
            q_states_succ_critical = q_succ_critical[qbegin:qend]
            k_states_succ = k[prev_chunk_end_pos - chunk_len : prev_chunk_end_pos]
            qk_chunks.append(
                (
                    q_states_succ_critical.transpose(0, 1)[:, -last_q_size:]
                    * softmax_scale
                )
                @ k_states_succ.permute(1, 2, 0)
            )
            stage_kv_lens["succ"] = k_states_succ.size(0)

        if prev_chunk_end_pos - chunk_len * 2 >= 0:
            q_states_inter_critical = q_inter_critical[qbegin:qend]
            k_states_inter = k[: prev_chunk_end_pos - chunk_len]
            qk_chunks.append(
                (
                    q_states_inter_critical.transpose(0, 1)[:, -last_q_size:]
                    * softmax_scale
                )
                @ k_states_inter.permute(1, 2, 0)
            )
            stage_kv_lens["inter"] = k_states_inter.size(0)

        qk = torch.cat(qk_chunks[::-1], dim=-1)
        arange = torch.arange(last_q_size, device=q.device)
        last_q_mask = arange[:, None] >= arange[None, :]
        qk[:, :, -last_q_size:] = torch.where(
            last_q_mask.unsqueeze(0),
            qk[:, :, -last_q_size:],
            -torch.inf,
        )
        qk = torch.softmax(qk, dim=-1, dtype=torch.float32)

        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :30] = torch.inf
        vertical = vertical.reshape(case.num_heads, -1)
        max_vertical_topk = min(vertical.shape[-1], max(heads_vertical_size))
        max_slash_topk = max(heads_slash_size)
        vertical_topk_buffer = torch.topk(vertical, max_vertical_topk, -1).indices

        slash_topk_buffer = torch.empty(
            (case.num_heads, max_slash_topk), dtype=torch.int64, device=q.device
        )
        current_vertical_size = [
            min(head_vertical_size, max_vertical_topk)
            for head_vertical_size in heads_vertical_size
        ]
        current_slash_size = []
        for head_i in range(case.num_heads):
            head_score = qk[head_i : head_i + 1, :, :]
            slash_scores = _dual_chunk_sum_all_diagonal_matrix(head_score)
            if head_score.size(1) != 1:
                slash_scores = slash_scores[..., : -last_q_size + 1]
            slash_scores[..., -100:] = torch.inf

            head_slash_size = min(heads_slash_size[head_i], vertical.size(-1))
            current_slash_size.append(head_slash_size)
            slash_topk = torch.topk(slash_scores, head_slash_size, -1).indices
            slash_topk_buffer[head_i, :head_slash_size] = slash_topk.reshape(-1)

        stage_vertical_indices = {stage: [] for stage in stage_kv_lens}
        stage_slash_indices = {stage: [] for stage in stage_kv_lens}
        for head_i in range(case.num_heads):
            vertical_topk = vertical_topk_buffer[
                head_i, : current_vertical_size[head_i]
            ]
            slash_topk = slash_topk_buffer[head_i, : current_slash_size[head_i]]

            intra_vertical_indices = (
                vertical_topk[vertical_topk >= prev_chunk_end_pos] - prev_chunk_end_pos
            )
            if intra_vertical_indices.nelement() == 0:
                intra_vertical_indices = _dual_chunk_sparse_fallback_indices_reference(
                    stage_kv_lens["intra"], current_vertical_size[head_i], q.device
                )
            intra_slash_indices = (qk.size(-1) - 1) - slash_topk[
                slash_topk >= prev_chunk_end_pos
            ]
            if intra_slash_indices.nelement() == 0:
                intra_slash_indices = _dual_chunk_sparse_fallback_indices_reference(
                    stage_kv_lens["intra"], current_slash_size[head_i], q.device
                )
            stage_vertical_indices["intra"].append(intra_vertical_indices)
            stage_slash_indices["intra"].append(intra_slash_indices)

            if "succ" in stage_kv_lens:
                succ_vertical_indices = vertical_topk[
                    (vertical_topk < prev_chunk_end_pos)
                    & (vertical_topk >= prev_chunk_end_pos - chunk_len)
                ] - (prev_chunk_end_pos - chunk_len)
                if succ_vertical_indices.nelement() == 0:
                    succ_vertical_indices = (
                        _dual_chunk_sparse_fallback_indices_reference(
                            stage_kv_lens["succ"],
                            current_vertical_size[head_i],
                            q.device,
                        )
                    )
                succ_slash_indices = (
                    prev_chunk_end_pos + chunk_q_len - 1
                ) - slash_topk[
                    (slash_topk >= (prev_chunk_end_pos - chunk_len))
                    & (slash_topk < (prev_chunk_end_pos + chunk_q_len))
                ]
                if succ_slash_indices.nelement() == 0:
                    succ_slash_indices = _dual_chunk_sparse_fallback_indices_reference(
                        stage_kv_lens["succ"],
                        current_slash_size[head_i],
                        q.device,
                    )
                stage_vertical_indices["succ"].append(succ_vertical_indices)
                stage_slash_indices["succ"].append(succ_slash_indices)

            if "inter" in stage_kv_lens:
                inter_vertical_indices = vertical_topk[
                    vertical_topk < prev_chunk_end_pos - chunk_len
                ]
                if inter_vertical_indices.nelement() == 0:
                    inter_vertical_indices = (
                        _dual_chunk_sparse_fallback_indices_reference(
                            stage_kv_lens["inter"],
                            current_vertical_size[head_i],
                            q.device,
                        )
                    )
                inter_slash_indices = (
                    prev_chunk_end_pos - chunk_len + chunk_q_len - 1
                ) - slash_topk[
                    slash_topk < (prev_chunk_end_pos - chunk_len + chunk_q_len)
                ]
                if inter_slash_indices.nelement() == 0:
                    inter_slash_indices = _dual_chunk_sparse_fallback_indices_reference(
                        stage_kv_lens["inter"],
                        current_slash_size[head_i],
                        q.device,
                    )
                stage_vertical_indices["inter"].append(inter_vertical_indices)
                stage_slash_indices["inter"].append(inter_slash_indices)

        for stage in ("intra", "succ", "inter"):
            if stage not in stage_kv_lens:
                continue
            selections.append(
                _make_sparse_stage_selection(
                    stage,
                    chunk_q_len,
                    stage_kv_lens[stage],
                    stage_vertical_indices[stage],
                    stage_slash_indices[stage],
                )
            )
        begin = end

    return selections


def _assert_sparse_stage_selections_match(
    testcase,
    actual: list[_DualChunkSparseStageSelection],
    expected: list[_DualChunkSparseStageSelection],
) -> None:
    testcase.assertEqual(
        len(actual),
        len(expected),
        f"expected {len(expected)} sparse stage calls, got {len(actual)}",
    )
    for index, (actual_stage, expected_stage) in enumerate(zip(actual, expected)):
        testcase.assertEqual(
            actual_stage,
            expected_stage,
            f"sparse stage selection mismatch at call {index}",
        )


def _run_dual_chunk_fixture_with_sparse_selection_capture(
    fixture: DualChunkAttentionFixture,
) -> tuple[torch.Tensor, list[_DualChunkSparseStageSelection]]:
    original_sparse_attention = _dual_chunk_backend._vertical_slash_sparse_attention
    selections = []

    def capture_sparse_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        vertical_indices: torch.Tensor,
        slash_indices: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        stage: str = "intra",
        block_size_M: int = 64,
        block_size_N: int = 64,
        vertical_indices_count: torch.Tensor | None = None,
        slash_indices_count: torch.Tensor | None = None,
    ):
        selections.append(
            _capture_sparse_stage_selection(
                stage,
                query,
                key,
                vertical_indices,
                slash_indices,
                vertical_indices_count,
                slash_indices_count,
            )
        )
        return original_sparse_attention(
            query,
            key,
            value,
            vertical_indices,
            slash_indices,
            softmax_scale,
            causal=causal,
            stage=stage,
            block_size_M=block_size_M,
            block_size_N=block_size_N,
            vertical_indices_count=vertical_indices_count,
            slash_indices_count=slash_indices_count,
        )

    try:
        _dual_chunk_backend._vertical_slash_sparse_attention = capture_sparse_attention
        output = run_dual_chunk_fixture_eager(fixture)
    finally:
        _dual_chunk_backend._vertical_slash_sparse_attention = original_sparse_attention
    return output, selections


def _torch_sparse_attn_metadata_mask(
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    q_len: int,
    kv_len: int,
    *,
    block_size_m: int = 64,
    block_size_n: int = 64,
) -> torch.Tensor:
    batch_size, num_heads, num_rows = block_count.shape
    mask = torch.zeros(
        (batch_size, num_heads, q_len, kv_len),
        dtype=torch.bool,
        device=block_count.device,
    )

    for batch_i in range(batch_size):
        for head_i in range(num_heads):
            for row_i in range(num_rows):
                row_start = row_i * block_size_m
                row_end = min(row_start + block_size_m, q_len)
                if row_start >= row_end:
                    continue

                for block_i in range(int(block_count[batch_i, head_i, row_i].item())):
                    col_start = int(
                        block_offset[batch_i, head_i, row_i, block_i].item()
                    )
                    col_end = min(col_start + block_size_n, kv_len)
                    if 0 <= col_start < col_end:
                        mask[batch_i, head_i, row_start:row_end, col_start:col_end] = (
                            True
                        )

                col_count = int(column_count[batch_i, head_i, row_i].item())
                cols = column_index[batch_i, head_i, row_i, :col_count].to(torch.long)
                cols = cols[(cols >= 0) & (cols < kv_len)]
                if cols.numel() > 0:
                    mask[batch_i, head_i, row_start:row_end, cols] = True

    return mask


def _torch_sparse_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
):
    assert dropout_p == 0.0
    assert softcap == 0.0
    assert alibi_slopes is None
    assert not deterministic
    assert not return_attn_probs
    assert out is None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    dtype = q.dtype
    _, q_len, num_heads, _ = q.shape
    kv_len = k.shape[1]
    if k.shape[2] != num_heads:
        group_size = num_heads // k.shape[2]
        k = torch.repeat_interleave(k, group_size, dim=2)
        v = torch.repeat_interleave(v, group_size, dim=2)

    sparse_mask = _torch_sparse_attn_metadata_mask(
        block_count, block_offset, column_count, column_index, q_len, kv_len
    )
    if causal:
        q_pos = torch.arange(q_len, device=q.device) + (kv_len - q_len)
        k_pos = torch.arange(kv_len, device=q.device)
        sparse_mask &= k_pos.view(1, 1, 1, kv_len) <= q_pos.view(1, 1, q_len, 1)

    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * softmax_scale
    scores = scores.masked_fill(~sparse_mask, -torch.inf)
    softmax_lse = torch.logsumexp(scores, dim=-1)
    valid_rows = sparse_mask.any(dim=-1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid_rows.unsqueeze(-1), probs, torch.zeros_like(probs))
    output = torch.einsum("bhqk,bkhd->bqhd", probs, v.float()).to(dtype)

    if return_softmax_lse:
        return output, softmax_lse
    return output


def _run_dual_chunk_fixture_with_torch_sparse_kernel(
    fixture: DualChunkAttentionFixture,
) -> torch.Tensor:
    original_sparse_attn_func = _dual_chunk_backend.sparse_attn_func
    try:
        _dual_chunk_backend.sparse_attn_func = _torch_sparse_attn_func
        return run_dual_chunk_fixture_eager(fixture)
    finally:
        _dual_chunk_backend.sparse_attn_func = original_sparse_attn_func


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
    """Correctness test for genuine sub-context-window sparse pruning."""
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
    actual, actual_selections = _run_dual_chunk_fixture_with_sparse_selection_capture(
        fixture
    )
    expected_selections = _dual_chunk_sparse_selection_reference(fixture)
    _assert_sparse_stage_selections_match(
        testcase, actual_selections, expected_selections
    )
    expected = _run_dual_chunk_fixture_with_torch_sparse_kernel(fixture)
    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


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


def _restore_dual_chunk_cache(fixture: DualChunkAttentionFixture, state) -> None:
    layer_id = fixture.actual_module.attn.layer_id
    fixture.runner.token_to_kv_pool.get_key_buffer(layer_id).copy_(state[0])
    fixture.runner.token_to_kv_pool.get_value_buffer(layer_id).copy_(state[1])
