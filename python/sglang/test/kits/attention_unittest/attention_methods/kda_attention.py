from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MHATokenToKVPool,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel

from ..mock_server_args import make_mock_server_args

_parallel_override = get_parallel().override(attn_tp_size=1)
_parallel_override.__enter__()

DEFAULT_HEAD_K_DIM = 32
DEFAULT_HEAD_V_DIM = 32
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"
KDA_ATOL = 3e-2
KDA_RTOL = 3e-2
KDA_TREE_ATOL = 5e-2
# CUDA-graph replay through the KDA Triton kernel accumulates small drift
# that pushes per-element diff above eager `KDA_ATOL`. Loose tolerance for
# graph-replay coverage where the goal is buffer/metadata compatibility
# rather than exact numerical reproduction.
KDA_GRAPH_ATOL = 1e-1
KDA_GRAPH_RTOL = 1e-1


@dataclass(frozen=True)
class KDAAttentionCase:
    name: str
    backend: str
    forward_mode: ForwardMode
    num_k_heads: int
    num_v_heads: int
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


def make_kda_cases(backend: str) -> tuple[KDAAttentionCase, ...]:
    common = dict(backend=backend, num_k_heads=2, num_v_heads=2)
    return (
        KDAAttentionCase(
            name="kda_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_zero_prefix_input_page_edges",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 16, 17),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_total_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        KDAAttentionCase(
            name="kda_extend_page32_cross_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
            **common,
        ),
        KDAAttentionCase(
            name="kda_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        KDAAttentionCase(
            name="kda_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


class TinyKDAModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        context_len: int,
    ):
        self.attention_arch = AttentionArch.MHA
        self.context_len = context_len
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.swa_v_head_dim = head_dim
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.hf_config = SimpleNamespace(architectures=["TinyKDAForCausalLM"])
        self.hf_text_config = self.hf_config

    def get_num_kv_heads(self, tp_size: int) -> int:
        assert self.num_key_value_heads % tp_size == 0
        return self.num_key_value_heads // tp_size


class MockKDAModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: KDAAttentionCase,
        model_config: TinyKDAModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        head_dim: int,
        head_k_dim: int,
        head_v_dim: int,
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
        speculative_num_draft_tokens = (
            case.input_lens[0]
            if case.forward_mode.is_target_verify()
            or case.forward_mode.is_draft_extend_v2()
            else 0
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
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            enable_deterministic_inference=False,
            enable_mis=False,
            linear_attn_backend="triton",
            linear_attn_decode_backend=None,
            linear_attn_prefill_backend=None,
            mamba_cache_chunk_size=64,
            max_running_requests=None,
            model_path=None,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=1 if case.forward_mode.is_target_verify() else 0,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_num_steps=max(0, speculative_num_draft_tokens - 1),
            triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None,
        )
        # KDA uses the KimiLinear cache layout (conv_kernel-1, conv_dim) and a
        # temporal state of (num_heads, head_dim, head_dim). The KDA backend's
        # forward_extend splits conv by [q_dim, k_dim, v_dim] along the conv_dim
        # axis after transpose, which requires this layout.
        cache_shape = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=case.num_v_heads,
            head_dim=head_v_dim,
            num_k_heads=case.num_k_heads,
            head_k_dim=head_k_dim,
            conv_kernel_size=2,
        )
        cache_params = KimiLinearCacheParams(
            shape=cache_shape,
            layers=[0],
            dtype=Mamba2StateDType(conv=dtype, temporal=torch.float32),
        )
        self.req_to_token_pool = HybridReqToTokenPool(
            size=pool_batch_size,
            mamba_size=pool_batch_size,
            mamba_spec_state_size=pool_batch_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=speculative_num_draft_tokens or None,
            enable_overlap_schedule=False,
        )
        max_token_loc = case.page_size + pool_batch_size * max_context_len
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            dtype=dtype,
            head_num=model_config.num_key_value_heads,
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
        self.sliding_window_size = None
        self.use_mla_backend = False
        self.is_draft_worker = False
        self._kernel_warmed_up = True

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


class ProjectedKDAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        mixed_qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        conv_weights = torch.zeros(mixed_qkv_dim, 2, dtype=dtype, device=device)
        conv_weights[:, 1] = 1
        # KDA's A_log is per-head ([HV]); dt_bias is per-head-channel ([HV*K]).
        self.A_log = nn.Parameter(
            torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
        )
        self.dt_bias = nn.Parameter(
            torch.randn(num_v_heads * head_k_dim, dtype=dtype, device=device) * 0.1
        )
        self.attn = RadixLinearAttention(
            layer_id=0,
            num_q_heads=num_k_heads,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_q_dim=head_k_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_weights=conv_weights.contiguous(),
            bias=None,
            activation=None,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    @property
    def mixed_qkv_dim(self) -> int:
        return (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )

    def split_qkv(self, mixed_qkv: torch.Tensor):
        q, k, v = torch.split(
            mixed_qkv,
            [
                self.num_k_heads * self.head_k_dim,
                self.num_k_heads * self.head_k_dim,
                self.num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        q = q.view(1, mixed_qkv.shape[0], self.num_k_heads, self.head_k_dim)
        k = k.view(1, mixed_qkv.shape[0], self.num_k_heads, self.head_k_dim)
        v = v.view(1, mixed_qkv.shape[0], self.num_v_heads, self.head_v_dim)
        return q, k, v

    def forward(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        return self.attn(forward_batch, mixed_qkv=mixed_qkv, a=a, b=b)


class ReferenceKDAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.A_log = nn.Parameter(
            torch.empty(num_v_heads, dtype=torch.float32, device=device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(num_v_heads * head_k_dim, dtype=dtype, device=device)
        )

    @property
    def mixed_qkv_dim(self) -> int:
        return (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )

    def split_qkv(self, mixed_qkv: torch.Tensor):
        q, k, v = torch.split(
            mixed_qkv,
            [
                self.num_k_heads * self.head_k_dim,
                self.num_k_heads * self.head_k_dim,
                self.num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        q = q.view(1, mixed_qkv.shape[0], self.num_k_heads, self.head_k_dim)
        k = k.view(1, mixed_qkv.shape[0], self.num_k_heads, self.head_k_dim)
        v = v.view(1, mixed_qkv.shape[0], self.num_v_heads, self.head_v_dim)
        return q, k, v


@dataclass
class KDAAttentionFixture:
    case: KDAAttentionCase
    runner: MockKDAModelRunner
    backend: HybridLinearAttnBackend
    actual_module: ProjectedKDAAttention
    reference_module: ReferenceKDAAttention
    forward_batch: ForwardBatch
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    # Raw [T, HV*K] gate and [T, HV] beta used by the reference math. The
    # kernel-shaped `a`/`b` above are derived from these in build time.
    a_raw: torch.Tensor
    b_raw: torch.Tensor


@dataclass
class KDAReferenceOutput:
    output: torch.Tensor
    final_states: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    return page_size + req_idx * max_context_len + pos


def _make_forward_batch(
    case: KDAAttentionCase,
    runner: MockKDAModelRunner,
    *,
    max_context_len: int,
    device: str,
    loc_fn=None,
) -> ForwardBatch:
    seq_lens = case.seq_lens
    input_lens = case.input_lens
    req_pool_indices = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    out_cache_locs: list[int] = []
    positions: list[int] = []

    if loc_fn is None:

        def loc_fn(req_idx: int, pos: int) -> int:
            return _token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )

    mamba_indices = torch.arange(
        1, case.batch_size + 1, dtype=torch.int32, device=device
    )
    runner.req_to_token_pool.req_index_to_mamba_index_mapping[req_pool_indices] = (
        mamba_indices
    )

    for req_idx, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            runner.req_to_token_pool.req_to_token[req_idx, pos] = loc_fn(req_idx, pos)

        if case.forward_mode.is_decode():
            positions.append(seq_len - 1)
            out_cache_locs.append(loc_fn(req_idx, seq_len - 1))
        else:
            prefix_len = case.prefix_lens[req_idx]
            for offset in range(input_lens[req_idx]):
                positions.append(prefix_len + offset)
                out_cache_locs.append(loc_fn(req_idx, prefix_len + offset))

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

    if case.forward_mode.is_extend(include_draft_extend_v2=True):
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


def build_kda_attention_fixture(
    testcase,
    case: KDAAttentionCase,
    *,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    loc_layout: str = "shuffled_pages",
) -> KDAAttentionFixture:
    seed = 4096 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyKDAModelConfig(
        num_heads=case.num_k_heads,
        head_dim=head_k_dim,
        context_len=max_context_len,
    )
    runner = MockKDAModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        head_dim=head_k_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
    )
    try:
        full_backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    initialize_linear_attn_config(runner.server_args)
    linear_backend = KDAAttnBackend(runner)
    backend = HybridLinearAttnBackend(full_backend, linear_backend, full_attn_layers=[])
    actual_module = ProjectedKDAAttention(
        num_k_heads=case.num_k_heads,
        num_v_heads=case.num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        dtype=dtype,
        device=device,
    )
    reference_module = ReferenceKDAAttention(
        num_k_heads=case.num_k_heads,
        num_v_heads=case.num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        dtype=dtype,
        device=device,
    )
    _copy_kda_parameters(actual_module, reference_module)
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
    mixed_qkv = torch.randn(
        case.num_input_tokens,
        actual_module.mixed_qkv_dim,
        dtype=dtype,
        device=device,
    )
    # KDA gate input is per-head-channel ([T, HV*K] raw); beta is per-head ([T, HV]).
    # For extend, the production model unflattens gate to [1, T, HV, K] and
    # sigmoid-then-unsqueezes beta to [1, T, HV] before calling the attn layer.
    # For decode, both stay flat and beta is sigmoid'd inside the fused kernel.
    a_raw = torch.randn(
        case.num_input_tokens,
        case.num_v_heads * head_k_dim,
        dtype=dtype,
        device=device,
    )
    b_raw = torch.randn(
        case.num_input_tokens, case.num_v_heads, dtype=dtype, device=device
    )
    if case.forward_mode.is_decode():
        a = a_raw
        b = b_raw.unsqueeze(0)
    else:
        a = a_raw.unflatten(-1, (case.num_v_heads, head_k_dim)).unsqueeze(0)
        b = b_raw.float().sigmoid().unsqueeze(0).to(dtype)

    fixture = KDAAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        a_raw=a_raw,
        b_raw=b_raw,
    )
    _populate_kda_prefix_state(fixture)
    return fixture


def _populate_kda_prefix_state(fixture: "KDAAttentionFixture") -> None:
    """Seed per-request KDA SSM state for `prefix_lens > 0` so both backend
    and reference start from a non-trivial initial state. Without this the
    pool's default zero state would let cases with prefix match trivially.
    Save/restores the global RNG to avoid perturbing downstream consumers.
    """
    case = fixture.case
    cache_indices = fixture.runner.req_to_token_pool.req_index_to_mamba_index_mapping[
        fixture.forward_batch.req_pool_indices
    ]
    temporal = _ssm_states(fixture)
    device = temporal.device

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device=device)
    try:
        seed = 5601 + len(case.name) * 19
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prefix_scale = 0.05  # bf16 accumulation tolerance — see GDN twin
        for req_idx, prefix_len in enumerate(case.prefix_lens):
            if prefix_len <= 0:
                continue
            state_idx = int(cache_indices[req_idx].item())
            slot_shape = temporal[state_idx].shape
            temporal[state_idx] = (
                torch.randn(slot_shape, dtype=temporal.dtype, device=device)
                * prefix_scale
            )
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device=device)


def _copy_kda_parameters(
    actual: ProjectedKDAAttention,
    reference: ReferenceKDAAttention,
):
    with torch.no_grad():
        reference.A_log.copy_(actual.A_log)
        reference.dt_bias.copy_(actual.dt_bias)


def _ssm_states(fixture: KDAAttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.mamba2_layer_cache(0).temporal


def _conv_states(fixture: KDAAttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.mamba2_layer_cache(0).conv[0]


def _clone_kda_cache(fixture: KDAAttentionFixture):
    return _conv_states(fixture).clone(), _ssm_states(fixture).clone()


def _restore_kda_cache(fixture: KDAAttentionFixture, cache) -> None:
    conv_states, ssm_states = cache
    _conv_states(fixture).copy_(conv_states)
    _ssm_states(fixture).copy_(ssm_states)


def _cache_indices(fixture: KDAAttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.get_mamba_indices(
        fixture.forward_batch.req_pool_indices
    )


def run_kda_fixture_eager(fixture: KDAAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(
            fixture.forward_batch,
            fixture.mixed_qkv,
            fixture.a,
            fixture.b,
        )


def _pure_torch_kda_gating(
    module: ReferenceKDAAttention,
    a_raw_per_token_head_k: torch.Tensor,
    b_per_token_head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # KDA's gate is per (token, v_head, k_channel): A_log is [HV], dt_bias is [HV*K].
    # g[t,h,k] = -exp(A_log[h]) * softplus(a[t,h,k] + dt_bias[h,k]); lower_bound=None.
    H = module.num_v_heads
    K = module.head_k_dim
    a_f = a_raw_per_token_head_k.float().view(-1, H, K)
    dt = module.dt_bias.float().view(H, K)
    A = module.A_log.float().view(H, 1)
    g = -torch.exp(A) * torch.nn.functional.softplus(a_f + dt)
    beta = torch.sigmoid(b_per_token_head.float())
    return g, beta


def _pure_torch_kda_reference(
    fixture: KDAAttentionFixture,
    initial_ssm_states: torch.Tensor,
) -> KDAReferenceOutput:
    module = fixture.reference_module
    # KDA backend hard-codes activation="silu" on the causal_conv1d. With identity
    # conv weights, the conv output equals silu(mixed_qkv).
    mixed_qkv_act = torch.nn.functional.silu(fixture.mixed_qkv.float()).to(
        fixture.mixed_qkv.dtype
    )
    q, k, v = module.split_qkv(mixed_qkv_act)
    cache_indices = _cache_indices(fixture)
    # g has shape [T, HV, K]; beta has shape [T, HV].
    g, beta = _pure_torch_kda_gating(module, fixture.a_raw, fixture.b_raw)
    q = q.float()
    k = k.float()
    v = v.float()

    outputs = torch.empty(
        1,
        fixture.case.num_input_tokens,
        fixture.case.num_v_heads,
        module.head_v_dim,
        dtype=torch.float32,
        device=fixture.runner.device,
    )
    final_states = initial_ssm_states.clone()
    q_head_ratio = fixture.case.num_v_heads // fixture.case.num_k_heads
    start = 0

    for req_idx, input_len in enumerate(fixture.case.input_lens):
        state_idx = cache_indices[req_idx]
        state = initial_ssm_states[state_idx].float().clone()

        for offset in range(input_len):
            token_idx = start + offset
            for v_head in range(fixture.case.num_v_heads):
                k_head = v_head // q_head_ratio
                q_vec = q[0, token_idx, k_head]
                k_vec = k[0, token_idx, k_head]
                v_vec = v[0, token_idx, v_head]

                q_norm = q_vec / torch.sqrt(torch.sum(q_vec * q_vec) + 1e-6)
                k_norm = k_vec / torch.sqrt(torch.sum(k_vec * k_vec) + 1e-6)
                q_norm = q_norm * (module.head_k_dim**-0.5)

                head_state = state[v_head]
                # State per head is (V, K). KDA's gate is per-channel along K:
                # multiply each column by exp(g[t, h, k]).
                head_state = head_state * torch.exp(g[token_idx, v_head]).unsqueeze(0)
                residual_v = v_vec - torch.sum(head_state * k_norm.unsqueeze(0), dim=1)
                residual_v = residual_v * beta[token_idx, v_head]
                head_state = head_state + residual_v.unsqueeze(1) * k_norm.unsqueeze(0)
                state[v_head] = head_state
                outputs[0, token_idx, v_head] = torch.sum(
                    head_state * q_norm.unsqueeze(0), dim=1
                )

        final_states[state_idx] = state.to(final_states.dtype)
        start += input_len

    return KDAReferenceOutput(
        output=outputs.to(fixture.mixed_qkv.dtype),
        final_states=final_states,
    )


def make_kda_case_with_prefix_lens(
    case: KDAAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> KDAAttentionCase:
    extend_lens = ()
    if not case.forward_mode.is_decode():
        if not case.input_lens:
            raise ValueError("Non-decode cases require input lengths.")
        if len(prefix_lens) <= len(case.input_lens):
            extend_lens = case.input_lens[: len(prefix_lens)]
        else:
            extend_lens = case.input_lens + (case.input_lens[-1],) * (
                len(prefix_lens) - len(case.input_lens)
            )

    return KDAAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_k_heads=case.num_k_heads,
        num_v_heads=case.num_v_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def kda_fixture_inputs(fixture: KDAAttentionFixture) -> dict[str, torch.Tensor]:
    # `a, b` are the per-forward-mode shaped tensors the actual module
    # consumes (see `build_kda_attention_fixture`: for DECODE
    # `a = a_raw [T, HV*K]` and `b = b_raw.unsqueeze(0) [1, T, HV]`; for
    # non-DECODE `a = a_raw.unflatten(-1, (HV, K)).unsqueeze(0)` and
    # `b = b_raw.sigmoid().unsqueeze(0)`). The verify reference
    # (`expected_kda_verify_output_from_inputs` →
    # `_pure_torch_kda_gating`) expects raw `[T, HV*K]` / `[T, HV]`
    # instead, so we expose both shapes through the inputs dict.
    return {
        "mixed_qkv": fixture.mixed_qkv,
        "a": fixture.a,
        "b": fixture.b,
        "a_raw": fixture.a_raw,
        "b_raw": fixture.b_raw,
    }


def make_kda_random_inputs(
    case: KDAAttentionCase,
    fixture: KDAAttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    head_k_dim = fixture.reference_module.head_k_dim
    a_raw = torch.randn(
        case.num_input_tokens,
        case.num_v_heads * head_k_dim,
        dtype=dtype,
        device=device,
    )
    b_raw = torch.randn(
        case.num_input_tokens,
        case.num_v_heads,
        dtype=dtype,
        device=device,
    )
    if case.forward_mode.is_decode():
        a = a_raw
        b = b_raw.unsqueeze(0)
    else:
        a = a_raw.unflatten(-1, (case.num_v_heads, head_k_dim)).unsqueeze(0)
        b = b_raw.float().sigmoid().unsqueeze(0).to(dtype)
    return {
        "mixed_qkv": torch.randn(
            case.num_input_tokens,
            fixture.actual_module.mixed_qkv_dim,
            dtype=dtype,
            device=device,
        ),
        "a": a,
        "b": b,
        "a_raw": a_raw,
        "b_raw": b_raw,
    }


def make_kda_replay_inputs(
    _case: KDAAttentionCase,
    fixture: KDAAttentionFixture,
    _pad_prefix_lens: tuple[int, ...],
    base_inputs: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    del fixture, dtype, device
    return base_inputs


def make_kda_token_padded_inputs(
    _case: KDAAttentionCase,
    fixture: KDAAttentionFixture,
    static_num_tokens: int,
    base_inputs: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    """Pad each input tensor along its token-axis to `static_num_tokens`.

    `kda_fixture_inputs` carries both the shaped `a/b` (which may be 2D
    `[T, HV*K]` for DECODE or 4D `[1, T, HV, K]` for non-DECODE) and the
    raw `a_raw/b_raw` (always 2D). Pad along whichever axis corresponds to
    `num_input_tokens` for each tensor.
    """
    del fixture
    raw_num_tokens = base_inputs["mixed_qkv"].shape[0]
    if static_num_tokens < raw_num_tokens:
        raise ValueError("static_num_tokens must cover the live input token count.")
    if static_num_tokens == raw_num_tokens:
        return base_inputs
    pad_num_tokens = static_num_tokens - raw_num_tokens

    def _pad_token_axis(t: torch.Tensor, token_axis: int) -> torch.Tensor:
        pad_shape = list(t.shape)
        pad_shape[token_axis] = pad_num_tokens
        return torch.cat(
            [t, torch.randn(*pad_shape, dtype=dtype, device=device)],
            dim=token_axis,
        )

    padded: dict[str, torch.Tensor] = {}
    for key, t in base_inputs.items():
        if key in ("mixed_qkv", "a_raw", "b_raw"):
            padded[key] = _pad_token_axis(t, token_axis=0)
        elif key in ("a", "b"):
            # DECODE shape `[T, HV*K]` / `[1, T, HV]`; non-DECODE
            # `[1, T, HV, K]` / `[1, T, HV]`. Pad whichever axis has T.
            token_axis = 0 if t.shape[0] == raw_num_tokens else 1
            padded[key] = _pad_token_axis(t, token_axis=token_axis)
        else:
            padded[key] = t
    return padded


def prepare_kda_runner_inputs(
    fixture: KDAAttentionFixture,
    case: KDAAttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, torch.Tensor],
    *,
    max_context_len: int,
) -> None:
    del max_context_len
    fixture.case = case
    fixture.forward_batch = batch
    fixture.mixed_qkv = inputs["mixed_qkv"]
    fixture.a = inputs["a"]
    fixture.b = inputs["b"]
    # Keep `a_raw, b_raw` in sync so the verify reference (which reads them
    # off the fixture in non-runner tests) stays consistent with `a, b`.
    if "a_raw" in inputs:
        fixture.a_raw = inputs["a_raw"]
    if "b_raw" in inputs:
        fixture.b_raw = inputs["b_raw"]


def run_kda_forward(
    fixture: KDAAttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    return fixture.actual_module(
        batch,
        inputs["mixed_qkv"],
        inputs["a"],
        inputs["b"],
    )


def kda_attention_layers(fixture: KDAAttentionFixture) -> list[RadixLinearAttention]:
    return [fixture.actual_module.attn]


def expected_kda_output_from_inputs(
    fixture: KDAAttentionFixture,
    _case: KDAAttentionCase,
    _inputs: dict[str, torch.Tensor],
    state,
) -> torch.Tensor:
    return _pure_torch_kda_reference(fixture, state[1]).output


def _kda_verify_parent_indices(draft_token_num: int, topk: int) -> tuple[int, ...]:
    if topk == 1:
        return tuple(range(-1, draft_token_num - 1))
    if draft_token_num != 3:
        raise ValueError("Tree KDA verify reference currently expects 3 draft tokens.")
    return (-1, 0, 0)


def expected_kda_verify_output_from_inputs(
    fixture: KDAAttentionFixture,
    case: KDAAttentionCase,
    inputs: dict[str, torch.Tensor],
    state,
    *,
    topk: int,
) -> torch.Tensor:
    module = fixture.reference_module
    q, k, v = module.split_qkv(inputs["mixed_qkv"])
    cache_indices = _cache_indices(fixture)
    # `_pure_torch_kda_gating` expects raw `[T, HV*K]` / `[T, HV]` shapes
    # (matching `fixture.a_raw / b_raw`). `inputs["a_raw"]` / `inputs["b_raw"]`
    # are surfaced by `kda_fixture_inputs` and `make_kda_random_inputs` for
    # this purpose. Falling back to `inputs["a"] / inputs["b"]` keeps
    # backwards compatibility for callers that haven't been updated to pass
    # the raw keys.
    a_for_gating = inputs.get("a_raw", inputs["a"])
    b_for_gating = inputs.get("b_raw", inputs["b"])
    g, beta = _pure_torch_kda_gating(module, a_for_gating, b_for_gating)
    q = q.float()
    k = k.float()
    v = v.float()

    initial_ssm_states = state[1]
    outputs = torch.empty(
        1,
        case.num_input_tokens,
        case.num_v_heads,
        module.head_v_dim,
        dtype=torch.float32,
        device=fixture.runner.device,
    )
    q_head_ratio = case.num_v_heads // case.num_k_heads
    start = 0

    for req_idx, input_len in enumerate(case.input_lens):
        parent_indices = _kda_verify_parent_indices(input_len, topk)
        state_idx = cache_indices[req_idx]
        root_state = initial_ssm_states[state_idx].float().clone()
        token_states = []

        for offset, parent_idx in enumerate(parent_indices):
            token_idx = start + offset
            state_for_token = (
                root_state.clone()
                if parent_idx < 0
                else token_states[parent_idx].clone()
            )

            for v_head in range(case.num_v_heads):
                k_head = v_head // q_head_ratio
                q_vec = q[0, token_idx, k_head]
                k_vec = k[0, token_idx, k_head]
                v_vec = v[0, token_idx, v_head]

                q_norm = q_vec / torch.sqrt(torch.sum(q_vec * q_vec) + 1e-6)
                k_norm = k_vec / torch.sqrt(torch.sum(k_vec * k_vec) + 1e-6)
                q_norm = q_norm * (module.head_k_dim**-0.5)

                head_state = state_for_token[v_head]
                head_state = head_state * torch.exp(g[token_idx, v_head])
                residual_v = v_vec - torch.sum(head_state * k_norm.unsqueeze(0), dim=1)
                residual_v = residual_v * beta[token_idx, v_head]
                head_state = head_state + residual_v.unsqueeze(1) * k_norm.unsqueeze(0)
                state_for_token[v_head] = head_state
                outputs[0, token_idx, v_head] = torch.sum(
                    head_state * q_norm.unsqueeze(0), dim=1
                )

            token_states.append(state_for_token)

        start += input_len

    return outputs.to(inputs["mixed_qkv"].dtype)


def run_kda_attention_case(
    testcase,
    case: KDAAttentionCase,
    *,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    loc_layout: str = "shuffled_pages",
):
    fixture = build_kda_attention_fixture(
        testcase,
        case,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        loc_layout=loc_layout,
    )
    initial_ssm_states = _ssm_states(fixture).clone()
    actual = run_kda_fixture_eager(fixture)
    expected = _pure_torch_kda_reference(fixture, initial_ssm_states)

    torch.testing.assert_close(actual, expected.output, atol=KDA_ATOL, rtol=KDA_RTOL)
    if case.forward_mode.is_decode():
        torch.testing.assert_close(
            _ssm_states(fixture)[_cache_indices(fixture)],
            expected.final_states[_cache_indices(fixture)],
            atol=KDA_ATOL,
            rtol=KDA_RTOL,
        )
