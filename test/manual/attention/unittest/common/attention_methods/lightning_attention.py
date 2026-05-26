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
from sglang.srt.layers import dp_attention as _dp_attention
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner

_dp_attention.get_attention_tp_size = lambda: 1
_dp_attention.get_attention_tp_rank = lambda: 0

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
        self.page_size = case.page_size
        self.model_config = model_config
        speculative_num_draft_tokens = (
            case.input_lens[0]
            if case.forward_mode.is_target_verify()
            or case.forward_mode.is_draft_extend(include_v2=True)
            else 0
        )
        self.server_args = SimpleNamespace(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            disable_cuda_graph=disable_cuda_graph,
            disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
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
) -> ForwardBatch:
    seq_lens = case.seq_lens
    input_lens = case.input_lens
    req_pool_indices = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    out_cache_locs: list[int] = []
    positions: list[int] = []

    mamba_indices = torch.arange(
        1, case.batch_size + 1, dtype=torch.int32, device=device
    )
    runner.req_to_token_pool.req_index_to_mamba_index_mapping[req_pool_indices] = (
        mamba_indices
    )

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
    forward_batch = _make_forward_batch(
        case,
        runner,
        max_context_len=max_context_len,
        device=device,
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
):
    fixture = build_lightning_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        max_context_len=max_context_len,
        num_hidden_layers=num_hidden_layers,
        dtype=dtype,
        device=device,
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
