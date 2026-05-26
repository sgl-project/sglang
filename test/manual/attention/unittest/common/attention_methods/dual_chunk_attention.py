from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import set_global_server_args_for_scheduler

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
    _dense_attention_reference,
    _make_forward_batch,
    _populate_prefix_kv,
)

DUAL_CHUNK_CONFIG = {
    "chunk_size": 64,
    "local_size": 16,
    "original_max_position_embeddings": 32768,
    "sparse_attention_enabled": False,
}


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
            name="dual_chunk_gqa_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
            backend=backend,
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
            dual_chunk_attention_config=DUAL_CHUNK_CONFIG,
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
    ):
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
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
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
            size=case.batch_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        max_token_loc = case.page_size + case.batch_size * max_context_len
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

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q, k, v = self.project_qkv(hidden_states)
        packed_q = torch.cat((q, q, q, q, q), dim=-1)
        attn_output = self.attn(packed_q, k, v, forward_batch)
        return self.o_proj(attn_output)


@dataclass
class DualChunkAttentionFixture:
    case: DualChunkAttentionCase
    runner: DualChunkMockModelRunner
    backend: object
    actual_module: ProjectedDualChunkAttention
    reference_module: ReferenceDenseAttention
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
) -> DualChunkAttentionFixture:
    chunk_len = DUAL_CHUNK_CONFIG["chunk_size"] - DUAL_CHUNK_CONFIG["local_size"]
    if max(case.seq_lens) > chunk_len:
        raise ValueError("Dual-chunk unit cases must stay inside the first chunk.")

    seed = 3026 + len(case.name) + case.num_kv_heads
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyDualChunkModelConfig(
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        context_len=max_context_len,
    )
    runner = DualChunkMockModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        head_dim=head_dim,
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
    _set_orig_seq_lens(forward_batch, case)
    _populate_prefix_kv(
        actual_module,
        case,
        runner,
        prefix_hidden,
        max_context_len=max_context_len,
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
    return _dense_attention_reference(
        fixture.reference_module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def run_dual_chunk_attention_case(
    testcase,
    case: DualChunkAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> None:
    fixture = build_dual_chunk_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    actual = run_dual_chunk_fixture_eager(fixture)
    expected = expected_dual_chunk_fixture_output(fixture)
    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)
