from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

# Set the single-rank topology before importing the attention implementation.
from sglang.srt.runtime_context import get_context, get_parallel

_parallel_override = get_parallel().override(
    tp_size=1,
    tp_rank=0,
    attn_tp_size=1,
    attn_tp_rank=0,
    tp_group=SimpleNamespace(rank_in_group=0, world_size=1),
)
_parallel_override.__enter__()

from sglang.srt.configs.falcon_h1 import FalconH1Config  # noqa: E402
from sglang.srt.configs.mamba_utils import (  # noqa: E402
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.configs.model_config import AttentionArch  # noqa: E402
from sglang.srt.layers.attention.attention_registry import (  # noqa: E402
    ATTENTION_BACKENDS,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # noqa: E402
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2  # noqa: E402
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    HybridReqToTokenPool,
    MHATokenToKVPool,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

# Tiny dims chosen to be the minimum that satisfies MambaMixer2's TP/chunk asserts:
#   - num_heads % tp_size == 0  (tp_size=1)
#   - intermediate_size = num_heads * head_dim
#   - mamba_chunk_size>=8 for chunked-scan kernel internals; we use 16
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_HEADS = 2
DEFAULT_HEAD_DIM = 16
DEFAULT_STATE_SIZE = 16
DEFAULT_N_GROUPS = 1
DEFAULT_CONV_KERNEL = 4
DEFAULT_MAMBA_CHUNK_SIZE = 16
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"
# Mamba2 has more accumulation steps than GDN/KDA/Lightning (chunked-scan,
# softplus-bounded dt, optional fp32 state) so we use 5e-2 instead of 3e-2.
MAMBA2_ATOL = 5e-2
MAMBA2_RTOL = 5e-2
# CUDA-graph replay through the Mamba2 SSD kernel accumulates drift that
# pushes per-element diff above eager `MAMBA2_ATOL`. Loose tolerance for
# graph-replay coverage where the goal is buffer/metadata compatibility.
MAMBA2_GRAPH_ATOL = 1e-1
MAMBA2_GRAPH_RTOL = 1e-1


@dataclass(frozen=True)
class Mamba2AttentionCase:
    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    mamba_chunk_size: int
    hidden_size: int
    page_size: int
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...] = ()

    @property
    def intermediate_size(self) -> int:
        return self.num_heads * self.head_dim

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


def make_mamba2_cases(backend: str) -> tuple[Mamba2AttentionCase, ...]:
    common = dict(
        backend=backend,
        num_heads=DEFAULT_NUM_HEADS,
        head_dim=DEFAULT_HEAD_DIM,
        state_size=DEFAULT_STATE_SIZE,
        n_groups=DEFAULT_N_GROUPS,
        conv_kernel=DEFAULT_CONV_KERNEL,
        mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
    )
    # DECODE coverage requires `initialize_mamba_selective_state_update_backend()`
    # to install the global selective-state-update backend that
    # `MambaMixer2.forward_decode` calls into. The fixture's
    # `MockMamba2ModelRunner.__init__` mirrors what the scheduler does
    # at startup, so DECODE is reachable.
    return (
        Mamba2AttentionCase(
            name="mamba2_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_extend_zero_prefix_below_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(8,),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_extend_zero_prefix_above_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(32,),
            **common,
        ),
        # Page-boundary edge sweep at the extend size: one below, exactly at,
        # one above page boundary. Mamba2 doesn't read paged KV, but the
        # req_to_token_pool still indexes by page; this exercises the metadata
        # builder under different per-request seq layouts. Use bsz=3 so the
        # batched metadata path sees mixed lengths.
        Mamba2AttentionCase(
            name="mamba2_extend_zero_prefix_input_page_edges",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 16, 17),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_extend_with_prefix",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(16,),
            **common,
        ),
        # Required input case: prefix + extend that lands exactly at one page
        # (total == page_size) with nonzero prefix.
        Mamba2AttentionCase(
            name="mamba2_extend_total_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
            **common,
        ),
        # Required input case: prefix + extend that crosses a page boundary,
        # with prefix just below the boundary.
        Mamba2AttentionCase(
            name="mamba2_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_extend_multi_request_zero_prefix",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_extend_multi_request_ragged",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 16),
            extend_lens=(16, 16),
            **common,
        ),
        # Required ragged case: requests with sequences below/at/above a page
        # boundary in the same batch.
        Mamba2AttentionCase(
            name="mamba2_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        # Required representative page-size-32 cross-page-boundary case.
        Mamba2AttentionCase(
            name="mamba2_extend_page32_cross_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        Mamba2AttentionCase(
            name="mamba2_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


class TinyMamba2ModelConfig:
    def __init__(
        self,
        *,
        case: Mamba2AttentionCase,
        context_len: int,
    ):
        self.attention_arch = AttentionArch.MHA
        self.context_len = context_len
        self.num_attention_heads = case.num_heads
        self.num_key_value_heads = case.num_heads
        self.head_dim = case.head_dim
        self.v_head_dim = case.head_dim
        self.swa_v_head_dim = case.head_dim
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.model_is_mrope = False
        self.is_generation = True
        self.quantization = None
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        # Mamba2AttnBackend reads mamba2_config(model_config).mamba_chunk_size; expose it
        self.hf_config = FalconH1Config(
            architectures=["TinyMamba2ForCausalLM"],
            mamba_chunk_size=case.mamba_chunk_size,
        )
        self.hf_config.get_text_config = lambda: self.hf_config
        self.hf_text_config = self.hf_config
        self.linear_attn_registry_result = None

    def get_max_num_attention_heads(self) -> int:
        return self.num_attention_heads

    def get_num_kv_heads(self, tp_size: int, dcp_size: int = 1) -> int:
        kv_tp_size = tp_size // dcp_size
        assert self.num_key_value_heads % kv_tp_size == 0
        return self.num_key_value_heads // kv_tp_size


class MockMamba2ModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: Mamba2AttentionCase,
        model_config: TinyMamba2ModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        disable_cuda_graph: bool = True,
        disable_piecewise_cuda_graph: bool = True,
        runner_batch_size: int | None = None,
        enable_mis: bool = False,
    ):
        pool_batch_size = runner_batch_size or case.batch_size
        self.device = device
        self.dtype = dtype
        self.kv_cache_dtype = dtype
        self.kv_cache_dtype_str = "auto"
        # This runner's own resolved backends (production stamps these in
        # ModelRunner.initialize); a draft runner would carry its own.
        self.prefill_attention_backend_str = case.backend
        self.decode_attention_backend_str = case.backend
        self.draft_attention_backend = None
        self.gpu_id = 0
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.canary_manager = None
        self.page_size = case.page_size
        self.model_config = model_config
        # MambaMixer2 asserts the layer_cache is a `SpeculativeState`
        # whenever `spec_info` is present (TARGET_VERIFY / DRAFT_EXTEND).
        # The HybridReqToTokenPool only allocates the extra
        # `intermediate_ssm` / `intermediate_conv_window` buffers when
        # `speculative_num_draft_tokens is not None`, so auto-derive the
        # count from `case.extend_lens` for the speculative modes.
        if (
            case.forward_mode.is_target_verify()
            or case.forward_mode.is_draft_extend_v2()
        ):
            speculative_num_draft_tokens = (
                max(case.extend_lens) if case.extend_lens else 1
            )
        else:
            speculative_num_draft_tokens = 0
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
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            enable_deterministic_inference=False,
            enable_mis=enable_mis,
            # `RowParallelLinear.forward` (called by the production
            # `MambaMixer2.out_proj`) consults
            # `get_server_args().enable_symm_mem` to decide whether
            # to wrap allocations in a symmetric-memory context. With
            # `world_size=1` the wrapper short-circuits, but the
            # attribute read still happens, so it must exist on the mock
            # server_args.
            enable_symm_mem=False,
            kv_cache_dtype="auto",
            linear_attn_backend="triton",
            linear_attn_decode_backend=None,
            linear_attn_prefill_backend=None,
            # `initialize_mamba_selective_state_update_backend` consults
            # `server_args.mamba_backend` (defaults to "triton") to install
            # the global selective-state-update backend that
            # `MambaMixer2.forward_decode` calls into. Set it explicitly so
            # the DECODE fixture path becomes reachable.
            mamba_backend="triton",
            max_running_requests=None,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=0,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_num_steps=0,
            triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None,
            # Pin the lazy mamba_cache_chunk_size property cache: production
            # derives it from hf_config + page_size, which needs a real model.
            _mamba_cache_chunk_size=64,
        )
        # install() publishes this fixture's config, so production reads
        # like `is_symmetric_memory_enabled()` (RowParallelLinear.forward)
        # see our `enable_symm_mem=False` for the fixture's lifetime.
        self.server_args = self._server_args_override.install()

        # Install the selective-state-update backend that
        # `MambaMixer2.forward_decode` requires. In production the
        # scheduler calls this during initialization; the fixture must
        # mirror that or DECODE crashes with a missing-backend error.
        from sglang.kernels.ops.mamba.triton_ops import (
            initialize_mamba_selective_state_update_backend,
        )

        initialize_mamba_selective_state_update_backend(self.server_args)
        cache_shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=case.intermediate_size,
            n_groups=case.n_groups,
            num_heads=case.num_heads,
            head_dim=case.head_dim,
            state_size=case.state_size,
            conv_kernel=case.conv_kernel,
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
            # Pass through so the pool allocates the SpeculativeState
            # intermediate buffers (required by MambaMixer2 when
            # `spec_info` is set).
            speculative_num_draft_tokens=(
                speculative_num_draft_tokens
                if speculative_num_draft_tokens > 0
                else None
            ),
            enable_overlap_schedule=False,
        )
        max_token_loc = case.page_size + pool_batch_size * max_context_len
        # Mamba2 doesn't use KV; the pool is required only because ModelRunner
        # contract expects it. Use a minimal MHA pool with a single layer.
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            dtype=dtype,
            head_num=model_config.num_key_value_heads,
            head_dim=case.head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.req_to_token_pool.state.bind_routes(self.token_to_kv_pool.layer_router)
        # A static allocator: ids index the pools as they are.
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=case.page_size,
            hands_out_virtual_ids=False,
            state_id_spaces=lambda: {},
            request_id_spaces=lambda: {},
        )
        self.is_draft_worker = False
        self.init_runner_ids()
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        self.use_mla_backend = False
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
        # Non-None so Mamba2AttnBackend reads mamba_chunk_size.
        return self.model_config.hf_config

    @property
    def mambaish_config(self):
        return self.mamba2_config


class ProjectedMamba2Attention(nn.Module):
    """Wraps a real MambaMixer2 and drives it through Mamba2AttnBackend."""

    def __init__(
        self,
        *,
        case: Mamba2AttentionCase,
        backend: Mamba2AttnBackend,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        cache_shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=case.intermediate_size,
            n_groups=case.n_groups,
            num_heads=case.num_heads,
            head_dim=case.head_dim,
            state_size=case.state_size,
            conv_kernel=case.conv_kernel,
        )
        cache_params = Mamba2CacheParams(
            shape=cache_shape,
            layers=[0],
            dtype=Mamba2StateDType(conv=dtype, temporal=torch.float32),
        )
        self.mixer = MambaMixer2(
            cache_params=cache_params,
            hidden_size=case.hidden_size,
            use_conv_bias=True,
            use_bias=False,
            n_groups=case.n_groups,
            rms_norm_eps=1e-5,
            activation="silu",
            use_rms_norm=True,
        )
        # Random-initialize the parameters that MambaMixer2 leaves as empty/ones.
        with torch.no_grad():
            self.mixer.in_proj.weight.copy_(
                torch.randn_like(self.mixer.in_proj.weight) * 0.1
            )
            self.mixer.conv1d.weight.copy_(
                torch.randn_like(self.mixer.conv1d.weight) * 0.1
            )
            self.mixer.conv1d.bias.copy_(torch.randn_like(self.mixer.conv1d.bias) * 0.1)
            self.mixer.out_proj.weight.copy_(
                torch.randn_like(self.mixer.out_proj.weight) * 0.1
            )
            # A is loaded as -exp(raw); use a negative random value to match.
            self.mixer.A.copy_(-torch.exp(torch.randn_like(self.mixer.A) * 0.1))
            self.mixer.D.copy_(torch.randn_like(self.mixer.D) * 0.1)
            self.mixer.dt_bias.copy_(torch.randn_like(self.mixer.dt_bias) * 0.1)
            self.mixer.norm.weight.copy_(
                torch.ones_like(self.mixer.norm.weight)
                + torch.randn_like(self.mixer.norm.weight) * 0.05
            )
        self.mixer.to(device=device, dtype=dtype)
        # Keep accumulator-sensitive params in their expected dtype.
        self.mixer.A.data = self.mixer.A.data.float()
        self.mixer.D.data = self.mixer.D.data.float()
        self.mixer.dt_bias.data = self.mixer.dt_bias.data.float()
        self.mixer.norm.weight.data = self.mixer.norm.weight.data.float()
        self.backend = backend
        self.case = case

    def forward(
        self,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
        *,
        use_triton_causal_conv_override: bool | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        # `MambaMixer2.forward` asserts `use_triton_causal_conv=True` whenever
        # `spec_info` is present (target-verify / draft-extend paths), because
        # the kernel needs the Triton causal-conv variant for intermediate
        # state support. The dense-extend path leaves it False.
        use_triton_causal_conv = (
            (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
            if use_triton_causal_conv_override is None
            else use_triton_causal_conv_override
        )
        self.backend.forward(
            self.mixer,
            hidden_states,
            output,
            layer_id=0,
            forward_batch=forward_batch,
            use_triton_causal_conv=use_triton_causal_conv,
        )
        return output


@dataclass
class Mamba2AttentionFixture:
    case: Mamba2AttentionCase
    runner: MockMamba2ModelRunner
    backend: Mamba2AttnBackend
    actual_module: ProjectedMamba2Attention
    forward_batch: ForwardBatch
    hidden_states: torch.Tensor


@dataclass
class Mamba2ReferenceOutput:
    output: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    return page_size + req_idx * max_context_len + pos


def _make_forward_batch(
    case: Mamba2AttentionCase,
    runner: MockMamba2ModelRunner,
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
    # Production batches take their KV ids from a plan (`init_new`).
    runner.runner_ids.bind_own_plan(batch)

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


def build_mamba2_attention_fixture(
    testcase,
    case: Mamba2AttentionCase,
    *,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    loc_layout: str = "shuffled_pages",
    enable_mis: bool = False,
) -> Mamba2AttentionFixture:
    seed = 4096 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyMamba2ModelConfig(case=case, context_len=max_context_len)
    runner = MockMamba2ModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
        enable_mis=enable_mis,
    )
    try:
        ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    backend = Mamba2AttnBackend(runner)
    actual_module = ProjectedMamba2Attention(
        case=case,
        backend=backend,
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
    hidden_states = torch.randn(
        case.num_input_tokens, case.hidden_size, dtype=dtype, device=device
    )
    return Mamba2AttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        forward_batch=forward_batch,
        hidden_states=hidden_states,
    )


def _ssm_states(fixture: Mamba2AttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.mamba_pool.mamba2_layer_cache(0).temporal


def _conv_states(fixture: Mamba2AttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.mamba_pool.mamba2_layer_cache(0).conv[0]


def _cache_indices(fixture: Mamba2AttentionFixture) -> torch.Tensor:
    return fixture.runner.req_to_token_pool.get_mamba_indices(
        fixture.forward_batch.req_pool_indices
    )


def _pure_torch_mamba2_reference(
    fixture: Mamba2AttentionFixture,
    initial_conv_states: torch.Tensor,
    initial_ssm_states: torch.Tensor,
) -> Mamba2ReferenceOutput:
    """Pure-PyTorch SSM scan that mirrors Mamba2AttnBackend->MambaMixer2.

    Uses the trained MambaMixer2 weights directly (we are testing the kernel
    path, not the projections), but recomputes every step with torch ops only.
    """
    case = fixture.case
    mixer = fixture.actual_module.mixer
    hidden_states = fixture.hidden_states  # [T, hidden]
    T = case.num_input_tokens
    H = case.num_heads
    P = case.head_dim
    N = case.state_size
    G = case.n_groups
    K = case.conv_kernel

    # 1. in_proj -> split into [gate, x_BC, dt]
    projected, _ = mixer.in_proj(hidden_states)  # [T, intermediate + conv_dim + H]
    intermediate_size = case.intermediate_size
    conv_dim = intermediate_size + 2 * G * N
    gate, x_BC, dt_in = torch.split(projected, [intermediate_size, conv_dim, H], dim=-1)

    # 2. Depthwise causal conv1d over x_BC. weight stored as (conv_dim, 1, K).
    conv_w = mixer.conv1d.weight  # (conv_dim, 1, K)
    conv_b = mixer.conv1d.bias  # (conv_dim,)
    cache_idx = _cache_indices(fixture)
    outputs_BC = torch.empty_like(x_BC)
    start = 0
    for req_idx, input_len in enumerate(case.input_lens):
        prefix_len = case.prefix_lens[req_idx]
        # Per-token padding on the left to make conv causal.
        x_seg = x_BC[start : start + input_len].transpose(0, 1).unsqueeze(0)
        # initial conv state lives in conv_states[cache_idx]; for zero-prefix
        # initial state is the zero-buffer the pool already has.
        if prefix_len > 0:
            init = initial_conv_states[cache_idx[req_idx]].unsqueeze(0)  # (1, dim, K-1)
            x_full = torch.cat([init.to(x_seg.dtype), x_seg], dim=-1)
            y = F.conv1d(x_full, conv_w, conv_b, padding=0, groups=conv_dim)
        else:
            y = F.conv1d(x_seg, conv_w, conv_b, padding=K - 1, groups=conv_dim)
            y = y[..., :input_len]
        y = F.silu(y).squeeze(0).transpose(0, 1)  # (input_len, conv_dim)
        outputs_BC[start : start + input_len] = y.to(outputs_BC.dtype)
        start += input_len

    x, B, C = torch.split(outputs_BC, [intermediate_size, G * N, G * N], dim=-1)
    x = x.view(T, H, P).float()
    B = B.view(T, G, N).float()
    C = C.view(T, G, N).float()

    # 3. dt = softplus(dt + dt_bias); broadcast per-head onto head_dim later.
    dt = F.softplus(dt_in.float() + mixer.dt_bias.float())  # [T, H]
    A = mixer.A.float()  # [H], negative
    D = mixer.D.float()  # [H]

    head_to_group = lambda h: h * G // H  # noqa: E731

    ssm_out = torch.empty(T, H, P, dtype=torch.float32, device=hidden_states.device)
    start = 0
    for req_idx, input_len in enumerate(case.input_lens):
        prefix_len = case.prefix_lens[req_idx]
        if prefix_len > 0:
            state = initial_ssm_states[cache_idx[req_idx]].float().clone()
        else:
            state = torch.zeros(
                H, P, N, dtype=torch.float32, device=hidden_states.device
            )
        for offset in range(input_len):
            t = start + offset
            for h in range(H):
                g = head_to_group(h)
                dt_h = dt[t, h]
                dA = torch.exp(dt_h * A[h])
                # state[h]: (P, N); B[t,g]: (N,); x[t,h]: (P,)
                state[h] = state[h] * dA + dt_h * torch.outer(x[t, h], B[t, g])
                # y[h,p] = C[g] @ state[h, p, :] + D[h] * x[t, h, p]
                ssm_out[t, h] = state[h] @ C[t, g] + D[h] * x[t, h]
        start += input_len

    # 4. norm(ssm_out, gate) -> out_proj
    ssm_out_2d = ssm_out.view(T, H * P).to(hidden_states.dtype)
    normed = mixer.norm.forward_native(ssm_out_2d, gate)
    out, _ = mixer.out_proj(normed.to(hidden_states.dtype))
    return Mamba2ReferenceOutput(output=out)


def run_mamba2_fixture_eager(fixture: Mamba2AttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(fixture.forward_batch, fixture.hidden_states)


def run_mamba2_attention_case(
    testcase,
    case: Mamba2AttentionCase,
    *,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    loc_layout: str = "shuffled_pages",
):
    fixture = build_mamba2_attention_fixture(
        testcase,
        case,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        loc_layout=loc_layout,
    )
    initial_conv = _conv_states(fixture).clone()
    initial_ssm = _ssm_states(fixture).clone()
    with torch.no_grad():
        expected = _pure_torch_mamba2_reference(fixture, initial_conv, initial_ssm)
    actual = run_mamba2_fixture_eager(fixture)
    torch.testing.assert_close(
        actual, expected.output, atol=MAMBA2_ATOL, rtol=MAMBA2_RTOL
    )


# ---------------------------------------------------------------------------
# Runner-mode helpers (mirror GDN/KDA/Lightning conventions)
# ---------------------------------------------------------------------------


def _clone_mamba2_cache(
    fixture: Mamba2AttentionFixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snapshot both SSM state and conv state for CG capture/replay isolation."""
    return (_ssm_states(fixture).clone(), _conv_states(fixture).clone())


def _restore_mamba2_cache(
    fixture: Mamba2AttentionFixture,
    state: tuple[torch.Tensor, torch.Tensor],
) -> None:
    ssm, conv = state
    _ssm_states(fixture).copy_(ssm)
    _conv_states(fixture).copy_(conv)


def make_mamba2_case_with_prefix_lens(
    case: Mamba2AttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> Mamba2AttentionCase:
    """Build a variant case with new `prefix_lens`. For DECODE,
    `extend_lens` is empty (input_lens derives `(1,) * batch_size`); for
    EXTEND we keep the original `extend_lens` clipped/padded to the new
    batch shape."""
    if case.forward_mode.is_decode():
        extend_lens: tuple[int, ...] = ()
    else:
        base = case.extend_lens or (1,)
        if len(prefix_lens) <= len(base):
            extend_lens = base[: len(prefix_lens)]
        else:
            extend_lens = base + (base[-1],) * (len(prefix_lens) - len(base))
    return Mamba2AttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        head_dim=case.head_dim,
        state_size=case.state_size,
        n_groups=case.n_groups,
        conv_kernel=case.conv_kernel,
        mamba_chunk_size=case.mamba_chunk_size,
        hidden_size=case.hidden_size,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def mamba2_fixture_inputs(
    fixture: Mamba2AttentionFixture,
) -> dict[str, torch.Tensor]:
    return {"hidden_states": fixture.hidden_states}


def make_mamba2_random_inputs(
    case: Mamba2AttentionCase,
    fixture: Mamba2AttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(
            case.num_input_tokens,
            case.hidden_size,
            dtype=dtype,
            device=device,
        ),
    }


def make_mamba2_replay_inputs(
    _case: Mamba2AttentionCase,
    fixture: Mamba2AttentionFixture,
    _pad_prefix_lens: tuple[int, ...],
    base_inputs: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    del fixture, dtype, device
    return base_inputs


def prepare_mamba2_runner_inputs(
    fixture: Mamba2AttentionFixture,
    case: Mamba2AttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, torch.Tensor],
    *,
    max_context_len: int,
) -> None:
    del max_context_len
    fixture.case = case
    fixture.forward_batch = batch
    fixture.hidden_states = inputs["hidden_states"]


def run_mamba2_forward(
    fixture: Mamba2AttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    return fixture.actual_module(batch, inputs["hidden_states"])


def expected_mamba2_output_from_inputs(
    fixture: Mamba2AttentionFixture,
    _case: Mamba2AttentionCase,
    _inputs: dict[str, torch.Tensor],
    state,
) -> torch.Tensor:
    """Reference output for runner-mode tests. `state` is the cloned
    (ssm_states, conv_states) snapshot; the reference walks the actual
    fixture's recurrence using the same hidden_states stored on the
    fixture (set via `prepare_mamba2_runner_inputs`)."""
    initial_ssm, initial_conv = state
    return _pure_torch_mamba2_reference(
        fixture,
        initial_conv_states=initial_conv,
        initial_ssm_states=initial_ssm,
    ).output


def expected_mamba2_verify_output_from_inputs(
    fixture: Mamba2AttentionFixture,
    case: Mamba2AttentionCase,
    inputs: dict[str, torch.Tensor],
    state,
    *,
    topk: int,
) -> torch.Tensor:
    """Reference output for chain (topk=1) target-verify cases.

    This reference (`_pure_torch_mamba2_reference`) is a chain recurrence.
    For `topk == 1` it matches the chain semantics the EAGLE verifier
    expects, so it doubles as the verify reference. For `topk > 1` the
    production SSM kernel DOES follow the draft tree (it consumes the
    parent-indices plumbing), but this test has no tree-aware reference to
    compare against, so tree verify is skipped here rather than validated.
    """
    if topk != 1:
        raise ValueError(
            "Mamba2 tree verify (topk>1) is not exercised here: this "
            "reference is chain-only. The production kernel supports tree "
            "verify; a tree-aware reference is future work."
        )
    del inputs
    # `state` is the (ssm_states, conv_states) snapshot captured before
    # the forward; same shape contract as
    # `expected_mamba2_output_from_inputs`.
    initial_ssm, initial_conv = state
    return _pure_torch_mamba2_reference(
        fixture,
        initial_conv_states=initial_conv,
        initial_ssm_states=initial_ssm,
    ).output


def make_mamba2_token_padded_inputs(
    _case: Mamba2AttentionCase,
    fixture: Mamba2AttentionFixture,
    static_num_tokens: int,
    base_inputs: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    """Pad `hidden_states` to a fixed static token count for split-op
    runner tests. Live tokens come first, padding follows."""
    del fixture
    raw_num_tokens = base_inputs["hidden_states"].shape[0]
    if static_num_tokens < raw_num_tokens:
        raise ValueError("static_num_tokens must cover the live input token count.")
    if static_num_tokens == raw_num_tokens:
        return base_inputs
    pad_num_tokens = static_num_tokens - raw_num_tokens
    return {
        "hidden_states": torch.cat(
            [
                base_inputs["hidden_states"],
                torch.randn(
                    pad_num_tokens,
                    base_inputs["hidden_states"].shape[1],
                    dtype=dtype,
                    device=device,
                ),
            ],
            dim=0,
        ),
    }


def mamba2_attention_layers(fixture: Mamba2AttentionFixture) -> list:
    """Return the layer list the backend forwards through. For Mamba2 the
    "layer" is the MambaMixer2 itself; there is no separate RadixAttention
    wrapper. Returns an empty list because `piecewise_forward_context`
    doesn't need to install per-layer hooks — Mamba2's forward writes
    output directly to an `empty_like(hidden_states)` buffer, bypassing
    the RadixAttention dispatch path that other backends use."""
    del fixture
    return []


def run_mamba2_mis_case(
    *,
    testcase,
    requests: Sequence[tuple[int, Sequence[int]]],
    backend: str = "triton",
    mamba_chunk_size: int = DEFAULT_MAMBA_CHUNK_SIZE,
    use_triton_causal_conv: bool = False,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> None:
    seq_lens, delimiters_list, req_spans = _build_mis_request_layouts(requests=requests)
    max_context_len = max(DEFAULT_MAX_CONTEXT_LEN, max(seq_lens) + 32)
    case = _make_mis_case(
        backend=backend,
        mamba_chunk_size=mamba_chunk_size,
        extend_lens=seq_lens,
    )
    fixture = build_mamba2_attention_fixture(
        testcase=testcase,
        case=case,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        enable_mis=True,
    )
    try:
        fixture.forward_batch.is_prefill_only = True
        fixture.forward_batch.multi_item_delimiter_indices = [
            torch.tensor(d, dtype=torch.int64) for d in delimiters_list
        ]

        # Seed state pools with non-zero garbage to verify stale state isolation.
        _conv_states(fixture).fill_(7.5)
        _ssm_states(fixture).fill_(7.5)
        actual = _run_mamba2_mis_forward(
            fixture=fixture,
            use_triton_causal_conv=use_triton_causal_conv,
            enforce_zero_sync=True,
        )
        testcase.assertTrue(torch.isfinite(actual).all().item())
        post_conv = _conv_states(fixture).clone()
        post_ssm = _ssm_states(fixture).clone()
        cache_indices = _cache_indices(fixture).tolist()

        _assert_mis_perturbation_invariance(
            testcase=testcase,
            fixture=fixture,
            actual=actual,
            delimiters_list=delimiters_list,
            req_spans=req_spans,
            use_triton_causal_conv=use_triton_causal_conv,
        )
        _assert_mis_item_outputs(
            testcase=testcase,
            fixture=fixture,
            actual=actual,
            req_spans=req_spans,
            zero_conv=torch.zeros_like(post_conv[:1]),
            zero_ssm=torch.zeros_like(post_ssm[:1]),
        )
        for req_idx, (req_start, query_len, _) in enumerate(req_spans):
            slot = cache_indices[req_idx]
            _assert_mis_query_pool_state(
                testcase=testcase,
                fixture=fixture,
                query_hidden=fixture.hidden_states[req_start : req_start + query_len],
                post_conv_slot=post_conv[slot],
                post_ssm_slot=post_ssm[slot],
                max_context_len=max_context_len,
                dtype=dtype,
                device=device,
                use_triton_causal_conv=use_triton_causal_conv,
            )
    finally:
        fixture.runner._server_args_override.restore()


def run_mamba2_mis_guard_cases(
    *,
    testcase,
    backend: str = "triton",
    mamba_chunk_size: int = DEFAULT_MAMBA_CHUNK_SIZE,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> None:
    case = _make_mis_case(
        backend=backend,
        mamba_chunk_size=mamba_chunk_size,
        extend_lens=(6,),
    )
    fixture_no_mis = build_mamba2_attention_fixture(
        testcase=testcase,
        case=case,
        max_context_len=DEFAULT_MAX_CONTEXT_LEN,
        dtype=dtype,
        device=device,
        enable_mis=False,
    )
    try:
        fixture_no_mis.forward_batch.is_prefill_only = True
        fixture_no_mis.forward_batch.multi_item_delimiter_indices = [
            torch.tensor([2, 5], dtype=torch.int64)
        ]
        with testcase.assertRaisesRegex(ValueError, "--enable-mis"):
            fixture_no_mis.backend.init_forward_metadata(fixture_no_mis.forward_batch)
    finally:
        fixture_no_mis.runner._server_args_override.restore()

    fixture_mis = build_mamba2_attention_fixture(
        testcase=testcase,
        case=case,
        max_context_len=DEFAULT_MAX_CONTEXT_LEN,
        dtype=dtype,
        device=device,
        enable_mis=True,
    )
    try:
        fb = fixture_mis.forward_batch
        fb.is_prefill_only = True
        fb.multi_item_delimiter_indices = [torch.tensor([2, 3, 5], dtype=torch.int64)]
        with testcase.assertRaisesRegex(ValueError, "does not support empty items"):
            fixture_mis.backend.init_forward_metadata(fb)

        fb.multi_item_delimiter_indices = [torch.tensor([2, 5], dtype=torch.int64)]
        fb.mamba_prefill_track_mask_cpu = [True]
        with testcase.assertRaisesRegex(ValueError, "disable-radix-cache"):
            fixture_mis.backend.init_forward_metadata(fb)
        fb.mamba_prefill_track_mask_cpu = None

        fb.mamba_track_indices = torch.tensor([1], dtype=torch.int32, device=device)
        with testcase.assertRaisesRegex(ValueError, "disable-radix-cache"):
            fixture_mis.backend.init_forward_metadata(fb)
        fb.mamba_track_indices = None

        fb.is_prefill_only = False
        with testcase.assertRaisesRegex(ValueError, "prefill-only"):
            fixture_mis.backend.init_forward_metadata(fb)
        fb.is_prefill_only = True

        fb.forward_mode = ForwardMode.DECODE
        with testcase.assertRaisesRegex(ValueError, "prefill-only"):
            fixture_mis.backend.init_forward_metadata(fb)
    finally:
        fixture_mis.runner._server_args_override.restore()


@contextmanager
def _cuda_zero_sync_guard(*, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        yield
        return
    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


def _make_mis_case(
    *,
    backend: str,
    mamba_chunk_size: int,
    extend_lens: Sequence[int],
) -> Mamba2AttentionCase:
    return Mamba2AttentionCase(
        name=f"mamba2_mis_chunk{mamba_chunk_size}_reqs{len(extend_lens)}",
        backend=backend,
        forward_mode=ForwardMode.EXTEND,
        num_heads=DEFAULT_NUM_HEADS,
        head_dim=DEFAULT_HEAD_DIM,
        state_size=DEFAULT_STATE_SIZE,
        n_groups=DEFAULT_N_GROUPS,
        conv_kernel=DEFAULT_CONV_KERNEL,
        mamba_chunk_size=mamba_chunk_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        page_size=16,
        prefix_lens=(0,) * len(extend_lens),
        extend_lens=tuple(extend_lens),
    )


def _single_extend_ref_case(
    *,
    base_case: Mamba2AttentionCase,
    suffix: str,
    seq_len: int,
) -> Mamba2AttentionCase:
    return replace(
        base_case,
        name=f"{base_case.name}_{suffix}",
        forward_mode=ForwardMode.EXTEND,
        prefix_lens=(0,),
        extend_lens=(seq_len,),
    )


def _build_mis_request_layouts(
    *,
    requests: Sequence[tuple[int, Sequence[int]]],
) -> tuple[list[int], list[list[int]], list[tuple[int, int, list[tuple[int, int]]]]]:
    seq_lens: list[int] = []
    delimiters_list: list[list[int]] = []
    req_spans: list[tuple[int, int, list[tuple[int, int]]]] = []
    req_start = 0
    for query_len, item_lens in requests:
        d = query_len
        delimiters = [d]
        item_spans: list[tuple[int, int]] = []
        for item_len in item_lens:
            if item_len <= 0:
                raise ValueError("Mamba2 MIS test layouts require positive item_len")
            item_start = req_start + d + 1
            item_end = item_start + item_len
            item_spans.append((item_start, item_end))
            d += 1 + item_len
            delimiters.append(d)
        seq_len = d + 1
        seq_lens.append(seq_len)
        delimiters_list.append(delimiters)
        req_spans.append((req_start, query_len, item_spans))
        req_start += seq_len
    return seq_lens, delimiters_list, req_spans


def _run_mamba2_mis_forward(
    *,
    fixture: Mamba2AttentionFixture,
    use_triton_causal_conv: bool,
    enforce_zero_sync: bool = False,
) -> torch.Tensor:
    with (
        torch.no_grad(),
        forward_context(ForwardContext(attn_backend=fixture.backend)),
        _cuda_zero_sync_guard(enabled=enforce_zero_sync),
    ):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(
            forward_batch=fixture.forward_batch,
            hidden_states=fixture.hidden_states,
            use_triton_causal_conv_override=use_triton_causal_conv,
        )


def _assert_mis_perturbation_invariance(
    *,
    testcase,
    fixture: Mamba2AttentionFixture,
    actual: torch.Tensor,
    delimiters_list: list[list[int]],
    req_spans: list[tuple[int, int, list[tuple[int, int]]]],
    use_triton_causal_conv: bool,
) -> None:
    items_0 = req_spans[0][2]
    if len(items_0) < 2:
        return
    orig_hidden = fixture.hidden_states
    perturbed_hidden = orig_hidden.clone()
    for (req_start, _, _), delims in zip(req_spans, delimiters_list):
        for d in delims:
            perturbed_hidden[req_start + d] += 50.0
    sib_start, sib_end = items_0[1]
    perturbed_hidden[sib_start:sib_end] += 50.0
    try:
        fixture.hidden_states = perturbed_hidden
        _conv_states(fixture).fill_(-3.25)
        _ssm_states(fixture).fill_(-3.25)
        perturbed_actual = _run_mamba2_mis_forward(
            fixture=fixture,
            use_triton_causal_conv=use_triton_causal_conv,
            enforce_zero_sync=True,
        )
    finally:
        fixture.hidden_states = orig_hidden
    item0_start, item0_end = items_0[0]
    testcase.assertTrue(
        torch.equal(
            perturbed_actual[item0_start:item0_end],
            actual[item0_start:item0_end],
        )
    )


def _assert_mis_item_outputs(
    *,
    testcase,
    fixture: Mamba2AttentionFixture,
    actual: torch.Tensor,
    req_spans: list[tuple[int, int, list[tuple[int, int]]]],
    zero_conv: torch.Tensor,
    zero_ssm: torch.Tensor,
) -> None:
    orig_case = fixture.case
    orig_hidden = fixture.hidden_states
    try:
        for req_start, query_len, item_spans in req_spans:
            query_hidden = orig_hidden[req_start : req_start + query_len]
            for item_start, item_end in item_spans:
                testcase.assertGreater(item_end, item_start)
                ref_hidden = torch.cat(
                    [query_hidden, orig_hidden[item_start:item_end]], dim=0
                )
                fixture.case = _single_extend_ref_case(
                    base_case=orig_case,
                    suffix="item_ref",
                    seq_len=ref_hidden.shape[0],
                )
                fixture.hidden_states = ref_hidden
                with torch.no_grad():
                    ref_out = _pure_torch_mamba2_reference(
                        fixture=fixture,
                        initial_conv_states=zero_conv,
                        initial_ssm_states=zero_ssm,
                    ).output
                torch.testing.assert_close(
                    actual[item_start:item_end],
                    ref_out[query_len:],
                    atol=MAMBA2_ATOL,
                    rtol=MAMBA2_RTOL,
                )
    finally:
        fixture.case = orig_case
        fixture.hidden_states = orig_hidden


def _assert_mis_query_pool_state(
    *,
    testcase,
    fixture: Mamba2AttentionFixture,
    query_hidden: torch.Tensor,
    post_conv_slot: torch.Tensor,
    post_ssm_slot: torch.Tensor,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
    use_triton_causal_conv: bool,
) -> None:
    if query_hidden.shape[0] == 0:
        torch.testing.assert_close(
            post_conv_slot, torch.zeros_like(post_conv_slot), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            post_ssm_slot, torch.zeros_like(post_ssm_slot), atol=0.0, rtol=0.0
        )
        return

    q_case = _single_extend_ref_case(
        base_case=fixture.case,
        suffix="query_ref",
        seq_len=query_hidden.shape[0],
    )
    q_fixture = build_mamba2_attention_fixture(
        testcase=testcase,
        case=q_case,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        enable_mis=False,
    )
    try:
        q_fixture.actual_module.mixer.load_state_dict(
            fixture.actual_module.mixer.state_dict()
        )
        q_fixture.hidden_states = query_hidden.clone()
        _conv_states(q_fixture).zero_()
        _ssm_states(q_fixture).zero_()
        _run_mamba2_mis_forward(
            fixture=q_fixture, use_triton_causal_conv=use_triton_causal_conv
        )
        q_slot = int(_cache_indices(q_fixture)[0].item())
        torch.testing.assert_close(
            post_conv_slot,
            _conv_states(q_fixture)[q_slot],
            atol=MAMBA2_ATOL,
            rtol=MAMBA2_RTOL,
        )
        torch.testing.assert_close(
            post_ssm_slot,
            _ssm_states(q_fixture)[q_slot],
            atol=MAMBA2_ATOL,
            rtol=MAMBA2_RTOL,
        )
    finally:
        q_fixture.runner._server_args_override.restore()
