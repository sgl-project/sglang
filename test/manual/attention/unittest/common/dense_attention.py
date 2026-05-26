from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers import dp_attention as _dp_attention
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner

# Unit tests run without distributed initialization. Backends that size buffers by
# attention tensor-parallel degree should see the single-rank default.
_dp_attention.get_attention_tp_size = lambda: 1

DEFAULT_HEAD_DIM = 16
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "cuda"
DENSE_ATOL = 3e-2
DENSE_RTOL = 3e-2


@dataclass(frozen=True)
class DenseAttentionCase:
    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    num_kv_heads: int
    page_size: int
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...] = ()
    sliding_window_size: int | None = None

    @property
    def batch_size(self) -> int:
        return len(self.prefix_lens)

    @property
    def input_lens(self) -> tuple[int, ...]:
        if self.forward_mode.is_decode():
            return (1,) * self.batch_size
        return self.extend_lens

    @property
    def seq_lens(self) -> tuple[int, ...]:
        return tuple(p + q for p, q in zip(self.prefix_lens, self.input_lens))

    @property
    def num_input_tokens(self) -> int:
        return sum(self.input_lens)


def make_dense_input_config_cases(backend: str) -> tuple[DenseAttentionCase, ...]:
    """MHA cases that focus on input-layout coverage, not head-layout coverage."""
    common = dict(backend=backend, num_heads=4, num_kv_heads=4)
    return (
        DenseAttentionCase(
            name="mha_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_zero_prefix_input_page_edges",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 16, 17),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_total_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_page32_cross_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        DenseAttentionCase(
            name="mha_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


def make_dense_attention_config_cases(backend: str) -> tuple[DenseAttentionCase, ...]:
    """Head-layout variants. Keep these separate from input-layout coverage."""
    return (
        DenseAttentionCase(
            name="gqa_decode_page_boundary",
            backend=backend,
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="mqa_extend_total_exact_page",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
        ),
    )


def make_dense_cases(backend: str) -> tuple[DenseAttentionCase, ...]:
    return make_dense_input_config_cases(backend) + make_dense_attention_config_cases(
        backend
    )


def make_swa_no_prefix_input_config_cases(
    backend: str,
) -> tuple[DenseAttentionCase, ...]:
    """SWA no-prefix cases with lengths below, exactly at, and above the window."""
    return (
        DenseAttentionCase(
            name="swa_extend_no_prefix_window_edges",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(3, 4, 5),
            sliding_window_size=4,
        ),
    )


def make_swa_prefix_input_config_cases(
    backend: str,
) -> tuple[DenseAttentionCase, ...]:
    """SWA prefix cases with prefix lengths below, at, and above the window."""
    return (
        DenseAttentionCase(
            name="swa_extend_prefix_window_edges",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(3, 4, 5),
            extend_lens=(2, 2, 2),
            sliding_window_size=4,
        ),
    )


class TinyModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        context_len: int,
        sliding_window_size: int | None = None,
    ):
        self.attention_arch = AttentionArch.MHA
        self.context_len = context_len
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.swa_v_head_dim = head_dim
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = sliding_window_size is not None
        self.is_local_attention_model = sliding_window_size is not None
        self.attention_chunk_size = None
        self.sliding_window_size = sliding_window_size
        self.hf_config = SimpleNamespace(architectures=["TinyForCausalLM"])
        self.hf_text_config = self.hf_config

    def get_num_kv_heads(self, tp_size: int) -> int:
        assert self.num_key_value_heads % tp_size == 0
        return self.num_key_value_heads // tp_size


class MockModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: DenseAttentionCase,
        model_config: TinyModelConfig,
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
        self.server_args = SimpleNamespace(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            disable_cuda_graph=disable_cuda_graph,
            disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            enable_deterministic_inference=False,
            enable_mis=False,
            max_running_requests=None,
            model_path=None,
            revision=None,
            speculative_algorithm=None,
            speculative_num_draft_tokens=0,
            speculative_num_steps=0,
            triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None,
        )
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
        self.is_hybrid_swa = case.sliding_window_size is not None
        self.sliding_window_size = case.sliding_window_size
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


class ProjectedDenseAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        sliding_window_size: int | None = None,
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
            sliding_window_size=(
                sliding_window_size if sliding_window_size is not None else -1
            ),
        )

    def project_qkv(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q, k, v = self.project_qkv(hidden_states)
        attn_output = self.attn(q, k, v, forward_batch)
        return self.o_proj(attn_output)


class ReferenceDenseAttention(nn.Module):
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
        self.scaling = head_dim**-0.5
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

    def project_qkv(self, hidden_states: torch.Tensor):
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def reconstruct_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return F.linear(attn_output, self.o_proj.weight)


@dataclass
class DenseAttentionFixture:
    case: DenseAttentionCase
    runner: MockModelRunner
    backend: object
    actual_module: ProjectedDenseAttention
    reference_module: ReferenceDenseAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    return page_size + req_idx * max_context_len + pos


def _make_forward_batch(
    case: DenseAttentionCase,
    runner: MockModelRunner,
    *,
    max_context_len: int,
    device: str,
) -> ForwardBatch:
    seq_lens = case.seq_lens
    input_lens = case.input_lens
    req_pool_indices = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    out_cache_locs: List[int] = []
    positions: List[int] = []

    for req_idx, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            runner.req_to_token_pool.req_to_token[req_idx, pos] = _token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )

        if case.forward_mode.is_decode():
            positions.append(seq_len - 1)
            out_cache_locs.append(
                _token_loc(
                    req_idx,
                    seq_len - 1,
                    page_size=case.page_size,
                    max_context_len=max_context_len,
                )
            )
        else:
            prefix_len = case.prefix_lens[req_idx]
            for offset in range(input_lens[req_idx]):
                positions.append(prefix_len + offset)
                out_cache_locs.append(
                    _token_loc(
                        req_idx,
                        prefix_len + offset,
                        page_size=case.page_size,
                        max_context_len=max_context_len,
                    )
                )

    batch = ForwardBatch(
        forward_mode=case.forward_mode,
        batch_size=case.batch_size,
        input_ids=torch.arange(case.num_input_tokens, dtype=torch.int64, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(out_cache_locs, dtype=torch.int64, device=device),
        seq_lens_sum=sum(seq_lens),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
    )

    if case.forward_mode.is_extend():
        batch.extend_prefix_lens = torch.tensor(
            case.prefix_lens, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens_cpu = list(case.prefix_lens)
        batch.extend_seq_lens = torch.tensor(
            input_lens, dtype=torch.int32, device=device
        )
        batch.extend_seq_lens_cpu = list(input_lens)
        batch.extend_num_tokens = case.num_input_tokens

    return batch


def _split_by_lens(tensor: torch.Tensor, lens: tuple[int, ...]):
    parts = []
    start = 0
    for length in lens:
        parts.append(tensor[start : start + length])
        start += length
    return parts


def _expand_gqa(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[0]
    if num_kv_heads == num_heads:
        return x
    assert num_heads % num_kv_heads == 0
    return x.repeat_interleave(num_heads // num_kv_heads, dim=0)


def _dense_attention_reference(
    module: ReferenceDenseAttention,
    case: DenseAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, k, v = module.project_qkv(input_hidden)
    q_parts = _split_by_lens(
        q.view(-1, case.num_heads, module.head_dim), case.input_lens
    )
    k_parts = _split_by_lens(
        k.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    v_parts = _split_by_lens(
        v.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    outputs = []

    for req_idx, prefix in enumerate(prefix_hidden):
        _, prefix_k, prefix_v = module.project_qkv(prefix)
        prefix_k = prefix_k.view(-1, case.num_kv_heads, module.head_dim)
        prefix_v = prefix_v.view(-1, case.num_kv_heads, module.head_dim)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0)
        req_v = torch.cat([prefix_v, v_parts[req_idx]], dim=0)

        for offset, query in enumerate(q_parts[req_idx]):
            query_pos = case.prefix_lens[req_idx] + offset
            key_start = 0
            if case.sliding_window_size is not None:
                # SGLang stores model sliding-window sizes as the number of tokens
                # to the left of the current query, so the current token is extra.
                key_start = max(0, query_pos - case.sliding_window_size)
            keys = _expand_gqa(
                req_k[key_start : query_pos + 1].movedim(0, 1), case.num_heads
            )
            values = _expand_gqa(
                req_v[key_start : query_pos + 1].movedim(0, 1), case.num_heads
            )
            query = query.float()
            keys = keys.float()
            scores = torch.einsum("hd,hkd->hk", query, keys) * module.scaling
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, values.float())
            outputs.append(out.reshape(-1))

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def _copy_dense_weights(
    actual: ProjectedDenseAttention,
    reference: ReferenceDenseAttention,
):
    with torch.no_grad():
        reference.q_proj.weight.copy_(actual.q_proj.weight)
        reference.k_proj.weight.copy_(actual.k_proj.weight)
        reference.v_proj.weight.copy_(actual.v_proj.weight)
        reference.o_proj.weight.copy_(actual.o_proj.weight)


def _populate_prefix_kv(
    module: ProjectedDenseAttention,
    case: DenseAttentionCase,
    runner: MockModelRunner,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
):
    locs = []
    keys = []
    values = []
    for req_idx, prefix in enumerate(prefix_hidden):
        if prefix.shape[0] == 0:
            continue
        _, k, v = module.project_qkv(prefix)
        keys.append(k.view(-1, case.num_kv_heads, module.head_dim))
        values.append(v.view(-1, case.num_kv_heads, module.head_dim))
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

    loc_tensor = torch.tensor(locs, dtype=torch.int64, device=runner.device)
    runner.token_to_kv_pool.set_kv_buffer(
        module.attn,
        loc_tensor,
        torch.cat(keys, dim=0),
        torch.cat(values, dim=0),
    )


def build_dense_attention_fixture(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
) -> DenseAttentionFixture:
    seed = 2026 + len(case.name) + case.num_kv_heads
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyModelConfig(
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        context_len=max_context_len,
        sliding_window_size=case.sliding_window_size,
    )
    runner = MockModelRunner(
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

    actual_module = ProjectedDenseAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        sliding_window_size=case.sliding_window_size,
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
    _populate_prefix_kv(
        actual_module,
        case,
        runner,
        prefix_hidden,
        max_context_len=max_context_len,
    )

    return DenseAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def run_dense_fixture_eager(fixture: DenseAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(fixture.input_hidden, fixture.forward_batch)


def expected_dense_fixture_output(fixture: DenseAttentionFixture) -> torch.Tensor:
    return _dense_attention_reference(
        fixture.reference_module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def run_dense_attention_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    actual = run_dense_fixture_eager(fixture)
    expected = expected_dense_fixture_output(fixture)

    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)
