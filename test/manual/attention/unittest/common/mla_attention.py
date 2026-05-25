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
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner

_dp_attention.get_attention_tp_size = lambda: 1

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_KV_LORA_RANK = 32
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "cuda"
MLA_ATOL = 3e-2
MLA_RTOL = 3e-2
DEFAULT_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4


@dataclass(frozen=True)
class MLAAttentionCase:
    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    page_size: int
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...] = ()

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


def make_mla_cases(backend: str) -> tuple[MLAAttentionCase, ...]:
    common = dict(backend=backend, num_heads=4)
    return (
        MLAAttentionCase(
            name="mla_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        MLAAttentionCase(
            name="mla_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        MLAAttentionCase(
            name="mla_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


class TinyMLAModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        kv_lora_rank: int,
        context_len: int,
    ):
        self.attention_arch = AttentionArch.MLA
        self.context_len = context_len
        self.num_attention_heads = num_heads
        self.num_key_value_heads = 1
        self.head_dim = kv_lora_rank
        self.v_head_dim = kv_lora_rank
        self.swa_v_head_dim = kv_lora_rank
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.hf_config = SimpleNamespace(architectures=["TinyMLAForCausalLM"])
        self.hf_text_config = self.hf_config

    def get_num_kv_heads(self, tp_size: int) -> int:
        return 1


class MockMLAModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: MLAAttentionCase,
        model_config: TinyMLAModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        kv_lora_rank: int,
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
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=0,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=case.page_size)
        self.attn_cp_size = 1
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        self.use_mla_backend = True
        self.is_draft_worker = False

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


class ProjectedMLAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * kv_lora_rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_proj = nn.Linear(
            hidden_size,
            kv_lora_rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            num_heads * kv_lora_rank,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=kv_lora_rank,
            scaling=kv_lora_rank**-0.5,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=kv_lora_rank,
        )

    def project_qk(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        k = self.kv_proj(hidden_states).unsqueeze(1)
        return q, k

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q, k = self.project_qk(hidden_states)
        attn_output = self.attn(q, k, k, forward_batch)
        return self.o_proj(attn_output)


@dataclass
class MLAAttentionFixture:
    case: MLAAttentionCase
    runner: MockMLAModelRunner
    backend: object
    module: ProjectedMLAAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    return page_size + req_idx * max_context_len + pos


def _make_forward_batch(
    case: MLAAttentionCase,
    runner: MockMLAModelRunner,
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
        extend_seq_lens = torch.tensor(input_lens, dtype=torch.int32, device=device)
        batch.extend_prefix_lens = torch.tensor(
            case.prefix_lens, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens_cpu = list(case.prefix_lens)
        batch.extend_seq_lens = extend_seq_lens
        batch.extend_seq_lens_cpu = list(input_lens)
        batch.extend_start_loc = torch.zeros_like(extend_seq_lens)
        if case.batch_size > 1:
            batch.extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        batch.extend_num_tokens = case.num_input_tokens

    return batch


def _split_by_lens(tensor: torch.Tensor, lens: tuple[int, ...]):
    parts = []
    start = 0
    for length in lens:
        parts.append(tensor[start : start + length])
        start += length
    return parts


def _mla_attention_reference(
    module: ProjectedMLAAttention,
    case: MLAAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, k = module.project_qk(input_hidden)
    q_parts = _split_by_lens(
        q.view(-1, case.num_heads, module.kv_lora_rank), case.input_lens
    )
    k_parts = _split_by_lens(k, case.input_lens)
    outputs = []

    for req_idx, prefix in enumerate(prefix_hidden):
        _, prefix_k = module.project_qk(prefix)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0).squeeze(1)

        for offset, query in enumerate(q_parts[req_idx]):
            query_pos = case.prefix_lens[req_idx] + offset
            keys = req_k[: query_pos + 1].movedim(0, 1)
            query = query.float()
            keys = keys.float()
            scores = torch.einsum("hd,dk->hk", query, keys) * module.attn.scaling
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,kd->hd", probs, req_k[: query_pos + 1].float())
            outputs.append(out.reshape(-1))

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return F.linear(attn_output, module.o_proj.weight)


def _populate_prefix_kv(
    module: ProjectedMLAAttention,
    case: MLAAttentionCase,
    runner: MockMLAModelRunner,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
):
    locs = []
    keys = []
    for req_idx, prefix in enumerate(prefix_hidden):
        if prefix.shape[0] == 0:
            continue
        _, k = module.project_qk(prefix)
        keys.append(k)
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
    cache_k = torch.cat(keys, dim=0)
    runner.token_to_kv_pool.set_kv_buffer(module.attn, loc_tensor, cache_k, cache_k)


def build_mla_attention_fixture(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
) -> MLAAttentionFixture:
    seed = 3090 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyMLAModelConfig(
        num_heads=case.num_heads,
        kv_lora_rank=kv_lora_rank,
        context_len=max_context_len,
    )
    runner = MockMLAModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        kv_lora_rank=kv_lora_rank,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
    )
    try:
        backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    module = ProjectedMLAAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        kv_lora_rank=kv_lora_rank,
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
    _populate_prefix_kv(
        module,
        case,
        runner,
        prefix_hidden,
        max_context_len=max_context_len,
    )

    return MLAAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        module=module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def run_mla_fixture_eager(fixture: MLAAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.module(fixture.input_hidden, fixture.forward_batch)


def expected_mla_fixture_output(fixture: MLAAttentionFixture) -> torch.Tensor:
    return _mla_attention_reference(
        fixture.module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def run_mla_attention_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    actual = run_mla_fixture_eager(fixture)
    expected = expected_mla_fixture_output(fixture)

    torch.testing.assert_close(actual, expected, atol=MLA_ATOL, rtol=MLA_RTOL)


def run_mla_cuda_graph_decode_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = DEFAULT_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    if not case.forward_mode.is_decode():
        raise ValueError(
            "CUDA graph runner integration currently expects decode cases."
        )
    if case.batch_size > cuda_graph_capture_batch_size:
        raise ValueError(
            "CUDA graph capture batch size must be at least the replay batch size."
        )

    eager_fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    eager_actual = run_mla_fixture_eager(eager_fixture)
    expected = expected_mla_fixture_output(eager_fixture)
    torch.testing.assert_close(eager_actual, expected, atol=MLA_ATOL, rtol=MLA_RTOL)

    graph_fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    backend = graph_fixture.backend
    capture_batch_size = cuda_graph_capture_batch_size
    seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    capture_prefix_len = max(0, seq_len_fill_value - 1)
    capture_case = MLAAttentionCase(
        name=f"{case.name}_cuda_graph_capture",
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=(capture_prefix_len,) * capture_batch_size,
    )
    capture_prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in capture_case.prefix_lens
    ]
    capture_input_hidden = torch.randn(
        capture_case.num_input_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    capture_batch = _make_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _populate_prefix_kv(
        graph_fixture.module,
        capture_case,
        graph_fixture.runner,
        capture_prefix_hidden,
        max_context_len=max_context_len,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        backend.init_cuda_graph_state(
            max_bs=capture_batch_size,
            max_num_tokens=capture_case.num_input_tokens,
        )
        backend.init_forward_metadata_capture_cuda_graph(
            bs=capture_batch_size,
            num_tokens=capture_case.num_input_tokens,
            req_pool_indices=capture_batch.req_pool_indices,
            seq_lens=capture_batch.seq_lens,
            encoder_lens=capture_batch.encoder_lens,
            forward_mode=capture_batch.forward_mode,
            spec_info=capture_batch.spec_info,
        )
        capture_actual = graph_fixture.module(capture_input_hidden, capture_batch)
        backend.on_after_cuda_graph_warmup()
        capture_expected = _mla_attention_reference(
            graph_fixture.module,
            capture_case,
            capture_prefix_hidden,
            capture_input_hidden,
        )

        replay_pad_prefix_lens = (capture_prefix_len,) * (
            capture_batch_size - case.batch_size
        )
        replay_case = MLAAttentionCase(
            name=f"{case.name}_cuda_graph_replay",
            backend=case.backend,
            forward_mode=case.forward_mode,
            num_heads=case.num_heads,
            page_size=case.page_size,
            prefix_lens=case.prefix_lens + replay_pad_prefix_lens,
        )
        replay_pad_prefix_hidden = [
            torch.randn(length, hidden_size, dtype=dtype, device=device)
            for length in replay_pad_prefix_lens
        ]
        replay_prefix_hidden = graph_fixture.prefix_hidden + replay_pad_prefix_hidden
        replay_pad_input_hidden = torch.randn(
            replay_case.num_input_tokens - case.num_input_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        replay_input_hidden = torch.cat(
            [graph_fixture.input_hidden, replay_pad_input_hidden], dim=0
        )
        replay_batch = _make_forward_batch(
            replay_case,
            graph_fixture.runner,
            max_context_len=max_context_len,
            device=device,
        )
        _populate_prefix_kv(
            graph_fixture.module,
            replay_case,
            graph_fixture.runner,
            replay_prefix_hidden,
            max_context_len=max_context_len,
        )
        backend.init_forward_metadata_replay_cuda_graph(
            bs=capture_batch_size,
            req_pool_indices=replay_batch.req_pool_indices,
            seq_lens=replay_batch.seq_lens,
            seq_lens_sum=replay_batch.seq_lens_sum,
            encoder_lens=replay_batch.encoder_lens,
            forward_mode=replay_batch.forward_mode,
            spec_info=replay_batch.spec_info,
            seq_lens_cpu=replay_batch.seq_lens_cpu,
        )
        replay_actual = graph_fixture.module(replay_input_hidden, replay_batch)

    torch.testing.assert_close(
        capture_actual, capture_expected, atol=MLA_ATOL, rtol=MLA_RTOL
    )
    replay_expected = _mla_attention_reference(
        graph_fixture.module,
        replay_case,
        replay_prefix_hidden,
        replay_input_hidden,
    )
    torch.testing.assert_close(
        replay_actual, replay_expected, atol=MLA_ATOL, rtol=MLA_RTOL
    )
    torch.testing.assert_close(
        replay_actual[: case.num_input_tokens],
        eager_actual,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
