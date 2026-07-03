from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.linear.lightning_backend import (
    LightningAttentionBackend,
)
from sglang.srt.layers.attention.linear.utils import initialize_linear_attn_config
from sglang.srt.layers.radix_attention import RadixAttention
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

_parallel_override = get_parallel().override(attn_tp_size=1, attn_tp_rank=0)
_parallel_override.__enter__()

# seg_la kernel constraints (see seg_la.py:683-694):
#   - decode (`seg_la_d_kernel`): K_SPLIT_DIM = 128, so head_dim must be >= 128
#     for `k_dim_block = head_dim // K_SPLIT_DIM` to be at least 1.
#   - prefill with bs > 2 (`seg_la_p_kernel`): V_SPLIT_DIM = 64, so head_dim must
#     be >= 64 for `v_dim_block = head_dim // V_SPLIT_DIM` to be at least 1.
# We use 128 so both decode and ragged multi-request extend exercise valid kernel grids.
DEFAULT_HEAD_DIM = 128
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"
LIGHTNING_ATOL = 3e-2
LIGHTNING_RTOL = 3e-2
# CUDA-graph replay through the seg_la Triton kernel accumulates small
# drift; loose tolerance for graph-replay coverage where the goal is
# buffer/metadata compatibility rather than exact numerical match.
LIGHTNING_GRAPH_ATOL = 1e-1
LIGHTNING_GRAPH_RTOL = 1e-1


@dataclass(frozen=True)
class LightningAttentionCase:
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


def make_lightning_cases(backend: str) -> tuple[LightningAttentionCase, ...]:
    common = dict(backend=backend, num_heads=2)
    return (
        LightningAttentionCase(
            name="lightning_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_zero_prefix_input_page_edges",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 16, 17),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_total_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_extend_page32_cross_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        LightningAttentionCase(
            name="lightning_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


class TinyLightningModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        context_len: int,
        num_hidden_layers: int = 1,
        linear_backend: str = "seg_la",
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
        self.quantization = None
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        # LightningAttentionBackend.__init__ reads num_attention_heads, num_hidden_layers,
        # and linear_backend directly from hf_config.
        self.hf_config = SimpleNamespace(
            architectures=["TinyLightningForCausalLM"],
            num_attention_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
            linear_backend=linear_backend,
        )
        self.hf_text_config = self.hf_config

    def get_num_kv_heads(self, tp_size: int) -> int:
        assert self.num_key_value_heads % tp_size == 0
        return self.num_key_value_heads // tp_size


class MockLightningModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: LightningAttentionCase,
        model_config: TinyLightningModelConfig,
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
            kv_cache_dtype="auto",
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
        # Lightning seg_la temporal state is [num_heads, head_dim, head_dim]; Bailing's
        # mamba2_cache_params sets intermediate_size=0, n_groups=0, conv_kernel=1
        # because seg_la does not use a conv state (the conv shape collapses to (0, 0)).
        cache_shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=0,
            n_groups=0,
            num_heads=case.num_heads,
            head_dim=head_dim,
            state_size=head_dim,
            conv_kernel=1,
        )
        cache_params = Mamba2CacheParams(
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

    # Return None so the attention_registry wrapper is bypassed and we drive
    # LightningAttentionBackend directly. The real wrapper uses HybridLinearAttnBackend,
    # whose _is_full_attn isinstance check routes Lightning's RadixAttention layer to
    # the full backend.
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


class ProjectedLightningAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Lightning's BailingMoELinearAttention uses a plain RadixAttention, with
        # forward_extend(q, k, v, ...) receiving already-projected q/k/v.
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=num_heads,
            layer_id=0,
        )
        # Move buffers so they live on `device` if RadixAttention adds any.
        self.to(device=device, dtype=dtype)

    def forward(
        self,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        return self.attn(q, k, v, forward_batch)


class ReferenceLightningAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_hidden_layers: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.dtype = dtype
        self.device = device
        # slopes[h] follows _build_slope_tensor for layer 0: base ALiBi slopes scaled
        # by (1 - layer_id/(L-1) + 1e-5) with L==num_hidden_layers (or 1+1e-5 when L==1).
        slopes = torch.tensor(
            _alibi_slopes(num_heads), dtype=torch.float32, device=device
        )
        self.register_buffer("slopes", slopes, persistent=False)

    def slope_for_layer(self, layer_id: int) -> torch.Tensor:
        if self.num_hidden_layers <= 1:
            scale = 1.0 + 1e-5
        else:
            scale = 1.0 - layer_id / (self.num_hidden_layers - 1) + 1e-5
        return self.slopes * scale


def _alibi_slopes(n: int) -> list[float]:
    import math

    def slopes_pow2(p):
        start = 2 ** (-(2 ** -(math.log2(p) - 3)))
        return [start * (start**i) for i in range(p)]

    if math.log2(n).is_integer():
        return slopes_pow2(n)
    closest = 2 ** math.floor(math.log2(n))
    extra = _alibi_slopes(2 * closest)[0::2][: n - closest]
    return slopes_pow2(closest) + extra


@dataclass
class LightningAttentionFixture:
    case: LightningAttentionCase
    runner: MockLightningModelRunner
    backend: LightningAttentionBackend
    actual_module: ProjectedLightningAttention
    reference_module: ReferenceLightningAttention
    forward_batch: ForwardBatch
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class LightningReferenceOutput:
    output: torch.Tensor
    final_states: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    return page_size + req_idx * max_context_len + pos


def _make_forward_batch(
    case: LightningAttentionCase,
    runner: MockLightningModelRunner,
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


def build_lightning_attention_fixture(
    testcase,
    case: LightningAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    num_hidden_layers: int = 1,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    loc_layout: str = "shuffled_pages",
) -> LightningAttentionFixture:
    seed = 4096 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyLightningModelConfig(
        num_heads=case.num_heads,
        head_dim=head_dim,
        context_len=max_context_len,
        num_hidden_layers=num_hidden_layers,
    )
    runner = MockLightningModelRunner(
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
        # Validate the named full backend can be constructed (matches GDN/KDA pattern);
        # for Lightning we drive LightningAttentionBackend directly below.
        ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    initialize_linear_attn_config(runner.server_args)
    backend = LightningAttentionBackend(runner)
    actual_module = ProjectedLightningAttention(
        num_heads=case.num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    reference_module = ReferenceLightningAttention(
        num_heads=case.num_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
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
    q = torch.randn(
        case.num_input_tokens, case.num_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        case.num_input_tokens, case.num_heads, head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        case.num_input_tokens, case.num_heads, head_dim, dtype=dtype, device=device
    )

    fixture = LightningAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        q=q,
        k=k,
        v=v,
    )
    _populate_lightning_prefix_state(fixture)
    return fixture


def _ssm_states(fixture: LightningAttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.mamba2_layer_cache(0).temporal


def _populate_lightning_prefix_state(fixture: LightningAttentionFixture) -> None:
    """Seed per-request seg_la SSM state for `prefix_lens > 0`. Without this
    the pool's default zero state lets cases with prefix match trivially in
    both actual and reference paths regardless of backend correctness.
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
        seed = 5701 + len(case.name) * 23
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prefix_scale = 0.05  # match GDN/KDA — bf16 accumulation tolerance
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


def _cache_indices(fixture: LightningAttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.get_mamba_indices(
        fixture.forward_batch.req_pool_indices
    )


def run_lightning_fixture_eager(fixture: LightningAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(
            fixture.forward_batch,
            fixture.q,
            fixture.k,
            fixture.v,
        )


def _pure_torch_lightning_reference(
    fixture: LightningAttentionFixture,
    initial_ssm_states: torch.Tensor,
) -> LightningReferenceOutput:
    # seg_la per-token recurrence (single layer, layer_id=0):
    #   state_t = state_{t-1} * exp(-slope_h) + outer(k_t, v_t)
    #   o_t = q_t @ state_t * softmax_scale
    # where slope_h = base_alibi_slope[h] * (1 - 0/(L-1) + 1e-5) for layer 0.
    case = fixture.case
    head_dim = fixture.reference_module.head_dim
    slopes = fixture.reference_module.slope_for_layer(0)
    decay = torch.exp(-slopes)  # per-head per-step decay
    softmax_scale = head_dim**-0.5

    q = fixture.q.float()
    k = fixture.k.float()
    v = fixture.v.float()

    outputs = torch.empty(
        case.num_input_tokens,
        case.num_heads,
        head_dim,
        dtype=torch.float32,
        device=fixture.runner.device,
    )
    final_states = initial_ssm_states.clone()
    cache_indices = _cache_indices(fixture)
    start = 0

    for req_idx, input_len in enumerate(case.input_lens):
        state_idx = cache_indices[req_idx]
        # Initial state is zero when has_initial_states is False (e.g. prefix_lens==0).
        has_initial = case.prefix_lens[req_idx] > 0
        if has_initial:
            state = initial_ssm_states[state_idx].float().clone()
        else:
            state = torch.zeros(
                case.num_heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=fixture.runner.device,
            )

        for offset in range(input_len):
            t = start + offset
            for h in range(case.num_heads):
                state[h] = state[h] * decay[h] + torch.outer(k[t, h], v[t, h])
                outputs[t, h] = (q[t, h] @ state[h]) * softmax_scale

        final_states[state_idx] = state.to(final_states.dtype)
        start += input_len

    return LightningReferenceOutput(
        output=outputs.to(fixture.q.dtype),
        final_states=final_states,
    )


def run_lightning_attention_case(
    testcase,
    case: LightningAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    num_hidden_layers: int = 1,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    loc_layout: str = "shuffled_pages",
):
    fixture = build_lightning_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        max_context_len=max_context_len,
        num_hidden_layers=num_hidden_layers,
        dtype=dtype,
        device=device,
        loc_layout=loc_layout,
    )
    initial_ssm_states = _ssm_states(fixture).clone()
    actual = run_lightning_fixture_eager(fixture)
    expected = _pure_torch_lightning_reference(fixture, initial_ssm_states)

    # Backend returns shape [num_tokens, num_heads * head_dim]; reshape to per-head.
    actual_per_head = actual.view(case.num_input_tokens, case.num_heads, head_dim)
    torch.testing.assert_close(
        actual_per_head,
        expected.output,
        atol=LIGHTNING_ATOL,
        rtol=LIGHTNING_RTOL,
    )


# ---------------------------------------------------------------------------
# Runner-mode helpers (mirror GDN/KDA conventions for cuda_graph_decode_runner)
# ---------------------------------------------------------------------------


def _clone_lightning_cache(fixture: LightningAttentionFixture):
    """Snapshot the SSM state for CG capture/replay isolation."""
    return _ssm_states(fixture).clone()


def _restore_lightning_cache(
    fixture: LightningAttentionFixture, state: torch.Tensor
) -> None:
    _ssm_states(fixture).copy_(state)


def make_lightning_case_with_prefix_lens(
    case: LightningAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> LightningAttentionCase:
    """Build a variant case with new `prefix_lens`. For DECODE, `extend_lens`
    is empty (input_lens derives `(1,) * batch_size`); for EXTEND we keep
    the original `extend_lens` clipped/padded to the new batch shape."""
    if case.forward_mode.is_decode():
        extend_lens: tuple[int, ...] = ()
    else:
        base = case.extend_lens or (1,)
        if len(prefix_lens) <= len(base):
            extend_lens = base[: len(prefix_lens)]
        else:
            extend_lens = base + (base[-1],) * (len(prefix_lens) - len(base))
    return LightningAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def lightning_fixture_inputs(
    fixture: LightningAttentionFixture,
) -> dict[str, torch.Tensor]:
    return {"q": fixture.q, "k": fixture.k, "v": fixture.v}


def make_lightning_random_inputs(
    case: LightningAttentionCase,
    fixture: LightningAttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    head_dim = fixture.reference_module.head_dim
    return {
        "q": torch.randn(
            case.num_input_tokens,
            case.num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        "k": torch.randn(
            case.num_input_tokens,
            case.num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        "v": torch.randn(
            case.num_input_tokens,
            case.num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
    }


def make_lightning_replay_inputs(
    _case: LightningAttentionCase,
    fixture: LightningAttentionFixture,
    _pad_prefix_lens: tuple[int, ...],
    base_inputs: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    del fixture, dtype, device
    return base_inputs


def prepare_lightning_runner_inputs(
    fixture: LightningAttentionFixture,
    _case: LightningAttentionCase,
    _batch: ForwardBatch,
    inputs: dict[str, torch.Tensor],
    *,
    max_context_len: int,
) -> None:
    del max_context_len
    fixture.q = inputs["q"]
    fixture.k = inputs["k"]
    fixture.v = inputs["v"]


def run_lightning_forward(
    fixture: LightningAttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    return fixture.actual_module(batch, inputs["q"], inputs["k"], inputs["v"])


def expected_lightning_output_from_inputs(
    fixture: LightningAttentionFixture,
    case: LightningAttentionCase,
    _inputs: dict[str, torch.Tensor],
    state,
) -> torch.Tensor:
    """Reference output for runner-mode tests. `state` is the cloned initial
    SSM state. The reference reshapes the per-head output back to the flat
    `[T, num_heads * head_dim]` shape the backend returns so the runner can
    compare directly."""
    out = _pure_torch_lightning_reference(fixture, state).output
    return out.reshape(case.num_input_tokens, -1)


def _lightning_verify_parent_indices(
    draft_token_num: int, topk: int
) -> tuple[int, ...]:
    """Parent indices for the EAGLE draft tree shape used by the verify tests.
    Matches KDA's `_kda_verify_parent_indices` convention: chain (topk=1) is a
    linear chain `(-1, 0, 1, ...)`; tree (topk=2 with 3 draft tokens) is the
    root + two-branch shape `(-1, 0, 0)`."""
    if topk == 1:
        return tuple(range(-1, draft_token_num - 1))
    if draft_token_num != 3:
        raise ValueError(
            "Tree Lightning verify reference currently expects 3 draft tokens."
        )
    return (-1, 0, 0)


def expected_lightning_verify_output_from_inputs(
    fixture: LightningAttentionFixture,
    case: LightningAttentionCase,
    inputs: dict[str, torch.Tensor],
    state: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Per-draft-token seg_la recurrence with parent-index sharing.

    Mirrors `expected_kda_verify_output_from_inputs`: for each draft token,
    start from the parent's post-recurrence state (or the request's root
    state for the first), apply the per-head decay+outer-product update,
    save the resulting state under the token's index so child draft tokens
    in the tree can read it.

    Returns shape `[num_input_tokens, num_heads * head_dim]` to match the
    backend's flat output.
    """
    head_dim = fixture.reference_module.head_dim
    slopes = fixture.reference_module.slope_for_layer(0)
    decay = torch.exp(-slopes)
    softmax_scale = head_dim**-0.5

    q = inputs["q"].float()
    k = inputs["k"].float()
    v = inputs["v"].float()

    outputs = torch.empty(
        case.num_input_tokens,
        case.num_heads,
        head_dim,
        dtype=torch.float32,
        device=fixture.runner.device,
    )
    cache_indices = _cache_indices(fixture)
    start = 0

    for req_idx, input_len in enumerate(case.input_lens):
        parent_indices = _lightning_verify_parent_indices(input_len, topk)
        state_idx = cache_indices[req_idx]
        has_initial = case.prefix_lens[req_idx] > 0
        if has_initial:
            root_state = state[state_idx].float().clone()
        else:
            root_state = torch.zeros(
                case.num_heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=fixture.runner.device,
            )

        token_states: list[torch.Tensor] = []
        for offset, parent_idx in enumerate(parent_indices):
            t = start + offset
            parent_state = (
                root_state.clone()
                if parent_idx < 0
                else token_states[parent_idx].clone()
            )
            new_state = torch.empty_like(parent_state)
            for h in range(case.num_heads):
                new_state[h] = parent_state[h] * decay[h] + torch.outer(
                    k[t, h], v[t, h]
                )
                outputs[t, h] = (q[t, h] @ new_state[h]) * softmax_scale
            token_states.append(new_state)
        start += input_len

    return outputs.to(fixture.q.dtype).reshape(case.num_input_tokens, -1)


def make_lightning_token_padded_inputs(
    _case: LightningAttentionCase,
    fixture: LightningAttentionFixture,
    static_num_tokens: int,
    base_inputs: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    """Pad inputs to a fixed static token count for split-op runner tests.
    The static count is the upper bound the backend's token-padding contract
    must cover; live tokens come first, padding follows."""
    del fixture
    raw_num_tokens = base_inputs["q"].shape[0]
    if static_num_tokens < raw_num_tokens:
        raise ValueError("static_num_tokens must cover the live input token count.")
    if static_num_tokens == raw_num_tokens:
        return base_inputs
    pad_num_tokens = static_num_tokens - raw_num_tokens

    def _pad(t: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                t,
                torch.randn(
                    pad_num_tokens,
                    *t.shape[1:],
                    dtype=dtype,
                    device=device,
                ),
            ],
            dim=0,
        )

    return {
        "q": _pad(base_inputs["q"]),
        "k": _pad(base_inputs["k"]),
        "v": _pad(base_inputs["v"]),
    }


def lightning_attention_layers(fixture: LightningAttentionFixture) -> list:
    """Return the RadixAttention layers the backend forwards through. The
    split-op runner uses this list to install per-layer
    `num_token_non_padded_cpu` metadata before forward."""
    return [fixture.actual_module.attn]


def expected_lightning_split_op_output_from_inputs(
    fixture: LightningAttentionFixture,
    case: LightningAttentionCase,
    _inputs: dict[str, torch.Tensor],
    state,
) -> torch.Tensor:
    """Per-head-shape reference for the split-op runner. In piecewise CG
    context, `RadixAttention.forward` writes through `output =
    torch.empty_like(q)` (shape `[T, num_heads, head_dim]`) instead of the
    flat backend return — so the split-op `actual` is per-head and the
    expected must match that shape."""
    return _pure_torch_lightning_reference(fixture, state).output
