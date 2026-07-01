from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
from sglang.srt.model_executor.runner import set_global_graph_memory_pool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPContext,
    FrozenKVMTPDraftInput,
)
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from ..attention_methods.dense_attention import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_CONTEXT_LEN,
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
)
from ..attention_methods.dense_attention import _token_loc as _dense_token_loc
from ..attention_methods.dense_attention import (
    build_dense_attention_fixture,
    prepare_dense_runner_inputs,
)
from ..attention_methods.dsa_attention import (
    DSA_PAGE_SIZE,
    DSA_SPARSE_ATOL,
    DSA_SPARSE_INDEX_TOPK,
    DSA_SPARSE_RTOL,
    DSAAttentionCase,
)
from ..attention_methods.dsa_attention import _token_loc as _dsa_token_loc
from ..attention_methods.dsa_attention import (
    build_dsa_sparse_attention_fixture,
)
from ..attention_methods.dsv4_attention import (
    DSV4_ATOL,
    DSV4_HEAD_DIM,
    DSV4_PAGE_SIZE,
    DSV4_RTOL,
    DSV4_SWA_WINDOW,
    DSV4AttentionCase,
)
from ..attention_methods.dsv4_attention import _token_loc as _dsv4_token_loc
from ..attention_methods.dsv4_attention import (
    build_dsv4_attention_fixture,
)
from ..attention_methods.mla_attention import DEFAULT_DEVICE as MLA_DEFAULT_DEVICE
from ..attention_methods.mla_attention import DEFAULT_DTYPE as MLA_DEFAULT_DTYPE
from ..attention_methods.mla_attention import (
    DEFAULT_HIDDEN_SIZE as MLA_DEFAULT_HIDDEN_SIZE,
)
from ..attention_methods.mla_attention import (
    DEFAULT_KV_LORA_RANK,
)
from ..attention_methods.mla_attention import (
    DEFAULT_MAX_CONTEXT_LEN as MLA_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.mla_attention import (
    DEFAULT_QK_ROPE_HEAD_DIM,
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
)
from ..attention_methods.mla_attention import _token_loc as _mla_token_loc
from ..attention_methods.mla_attention import (
    build_mla_attention_fixture,
    prepare_mla_runner_inputs,
)


def _assert_draft_outputs_close(actual, expected, settings) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=settings.atol,
            rtol=settings.rtol,
        )


@dataclass(frozen=True)
class EagleDraftRunnerSettings:
    topk: int
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    capture_batch_size: int
    hidden_size: int
    vocab_size: int
    max_context_len: int
    dtype: torch.dtype
    device: str
    atol: float
    rtol: float


@dataclass(frozen=True)
class EagleDraftCudaGraphRunnerAdapter:
    build_fixture: Callable[..., Any]
    make_model_forward: Callable[[Any, EagleDraftRunnerSettings], Callable[..., Any]]
    make_draft_inputs: Callable[[Any, EagleDraftRunnerSettings], Any]
    prepare_replay_state: Callable[[Any, Any, Any, EagleDraftRunnerSettings], None]
    make_forward_batch: Callable[[Any, Any, EagleDraftRunnerSettings], ForwardBatch]
    check_case: Callable[[Any, EagleDraftRunnerSettings], None] = (
        lambda _case, _settings: None
    )
    assert_outputs_close: Callable[[Any, Any, EagleDraftRunnerSettings], None] = (
        _assert_draft_outputs_close
    )
    # Optional override for the eager pre-draft init path. Default mirrors
    # dense/MLA: one `init_forward_metadata` call before the multi-step
    # loop. DSV4 needs to override this because its
    # `init_forward_metadata_decode` strictly asserts
    # `out_cache_loc.shape[0] == bs`, which is not the case for the
    # multi-step batch (`shape = bs * topk * num_steps`).
    init_eager_metadata: Callable[
        [Any, ForwardBatch, EagleDraftRunnerSettings], None
    ] = None


@dataclass(frozen=True)
class _DummyTpGroup:
    ca_comm = None

    def barrier(self) -> None:
        return None


class _TinyDraftModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("EAGLEDraftCudaGraphRunner should call worker.draft_forward")


class _EagleDraftWorkerHarness:
    def __init__(
        self,
        *,
        fixture,
        draft_attn_backend,
        model_forward: Callable[..., Any],
        settings: EagleDraftRunnerSettings,
    ):
        self.model_runner = fixture.runner
        self.draft_attn_backend = draft_attn_backend
        self.topk = settings.topk
        self.speculative_num_steps = settings.speculative_num_steps
        self.speculative_num_draft_tokens = settings.speculative_num_draft_tokens
        self.server_args = fixture.runner.server_args
        self.model_config = fixture.runner.model_config
        self.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        self.hot_token_id = None
        self.model_runner.forward = model_forward
        self.draft_forward = MethodType(EagleDraftWorker.draft_forward, self)
        # draft_forward's topk=1 fast path reads these prealloc buffers (built
        # in EagleDraftWorker.__init__, which the harness skips), so build them
        # here. _rebuild_topk1_chain_buffers asserts num_draft_tokens ==
        # num_steps + 1; the fast path never reads num_draft_tokens, so pin it.
        self.device = self.model_runner.device
        if self.topk == 1:
            self.speculative_num_draft_tokens = self.speculative_num_steps + 1
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        EagleDraftWorker._rebuild_topk1_chain_buffers(self)
        # draft_forward reads this (set in EagleDraftWorker.__init__, skipped here).
        self.index_share_for_mtp_iteration = (
            getattr(
                self.model_config.hf_config,
                "index_share_for_mtp_iteration",
                False,
            )
            and self.topk == 1
        )
        self.dsa_index_topk = getattr(
            self.draft_runner.model_config.hf_config, "index_topk", None
        )
        self.seed_topk_from_extend = (
            self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
        )

    @property
    def draft_model_runner(self):
        return self.model_runner

    @property
    def draft_runner(self):
        # V2 draft_forward reads self.draft_runner (forward / model_config /
        # canary_manager); for the harness that's the fixture runner.
        return self.model_runner


class _FrozenKVMTPWorkerHarness:
    def __init__(
        self,
        *,
        fixture,
        draft_attn_backend,
        model_forward: Callable[..., Any],
        settings: EagleDraftRunnerSettings,
    ):
        self.model_runner = fixture.runner
        self.draft_attn_backend = draft_attn_backend
        self.topk = settings.topk
        self.speculative_num_steps = settings.speculative_num_steps
        self.speculative_num_draft_tokens = settings.speculative_num_draft_tokens
        self.server_args = fixture.runner.server_args
        self.model_config = fixture.runner.model_config
        self.speculative_algorithm = SpeculativeAlgorithm.FROZEN_KV_MTP
        self.hot_token_id = None
        self.target_worker = SimpleNamespace(
            device=fixture.runner.device,
            model_runner=fixture.runner,
        )
        self.kv_context = FrozenKVMTPContext(
            target_token_to_kv_pool=fixture.runner.token_to_kv_pool,
            physical_layer_ids={0: 0},
        )
        self.model_runner.forward = model_forward
        self._hidden_size = settings.hidden_size
        self.draft_forward = MethodType(FrozenKVMTPDraftWorker.draft_forward, self)
        self._frozen_kv_target_view = MethodType(
            FrozenKVMTPDraftWorker._frozen_kv_target_view,
            self,
        )
        self._target_kv_pool_view = MethodType(
            FrozenKVMTPDraftWorker._target_kv_pool_view,
            self,
        )
        self._set_positions = MethodType(FrozenKVMTPDraftWorker._set_positions, self)
        self._init_frozen_kv_metadata = MethodType(
            FrozenKVMTPDraftWorker._init_frozen_kv_metadata,
            self,
        )
        self._init_frozen_kv_metadata_capture_cuda_graph = MethodType(
            FrozenKVMTPDraftWorker._init_frozen_kv_metadata_capture_cuda_graph,
            self,
        )
        self._init_frozen_kv_metadata_replay_cuda_graph = MethodType(
            FrozenKVMTPDraftWorker._init_frozen_kv_metadata_replay_cuda_graph,
            self,
        )

    @property
    def draft_model_runner(self):
        return self.model_runner

    @property
    def _recurrent_hidden_size(self) -> int:
        return self._hidden_size


@contextmanager
def _seeded_rng(seed: int, *, device: str | torch.device):
    """Set CPU+CUDA RNG to `seed` for the body, restore on exit. The draft
    input builders below need deterministic randomness per case but must not
    leak that seed into the rest of the test process (build_*_fixture relies
    on its own seeding for parameter init)."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device=device)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device=device)


@contextmanager
def _single_rank_graph_capture():
    stream = torch.cuda.Stream()
    yield SimpleNamespace(stream=stream)


def _reset_cuda_graph_test_buffers() -> None:
    set_global_graph_memory_pool(None)
    _forward_input_buffer_pool.clear()


def _configure_runner_for_eagle_draft(
    runner,
    case,
    settings: EagleDraftRunnerSettings,
    *,
    speculative_attention_mode: str = "decode",
) -> None:
    server_args = runner.server_args
    updates = {
        "attention_backend": case.backend,
        "cuda_graph_config": CudaGraphConfig(
            decode=PhaseConfig(
                bs=[settings.capture_batch_size],
                max_bs=settings.capture_batch_size,
            ),
        ),
        "debug_cuda_graph": False,
        "decode_attention_backend": case.backend,
        "disable_cuda_graph_padding": False,
        "enable_cudagraph_gc": True,
        "enable_dp_lm_head": False,
        "enable_memory_saver": False,
        "enable_pdmux": False,
        "enable_profile_cuda_graph": False,
        "enable_torch_compile": False,
        "enable_two_batch_overlap": False,
        "moe_dense_tp_size": None,
        "page_size": runner.page_size,
        "prefill_attention_backend": case.backend,
        "speculative_algorithm": "EAGLE",
        "speculative_attention_mode": speculative_attention_mode,
        "speculative_draft_attention_backend": None,
        "speculative_eagle_topk": settings.topk,
        "speculative_num_draft_tokens": settings.speculative_num_draft_tokens,
        "speculative_num_steps": settings.speculative_num_steps,
        "torch_compile_max_bs": 0,
        "use_mla_backend": runner.use_mla_backend,
    }
    for key, value in updates.items():
        setattr(server_args, key, value)

    runner.spec_algorithm = SpeculativeAlgorithm.EAGLE
    runner.is_draft_worker = True
    runner.model = _TinyDraftModel()
    runner.tp_group = _DummyTpGroup()
    runner.device_timer = None
    runner.model_config.spec_hidden_size = settings.hidden_size
    runner.model_config.dtype = runner.dtype
    runner.model_config.vocab_size = settings.vocab_size
    runner.model_config.hf_config.vocab_size = settings.vocab_size
    set_global_server_args_for_scheduler(server_args)


def _build_eagle_draft_fixture(
    testcase,
    case,
    *,
    adapter: EagleDraftCudaGraphRunnerAdapter,
    build_kwargs: dict,
    settings: EagleDraftRunnerSettings,
):
    fixture = adapter.build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=settings.capture_batch_size,
    )
    _configure_runner_for_eagle_draft(fixture.runner, case, settings)
    draft_attn_backend = DraftBackendFactory(
        fixture.runner.server_args,
        fixture.runner,
        settings.topk,
        settings.speculative_num_steps,
    ).create_decode_backend()
    fixture.runner.draft_attn_backend = draft_attn_backend
    worker = _EagleDraftWorkerHarness(
        fixture=fixture,
        draft_attn_backend=draft_attn_backend,
        model_forward=adapter.make_model_forward(fixture, settings),
        settings=settings,
    )
    return fixture, worker, draft_attn_backend


def _build_frozen_kv_mtp_fixture(
    testcase,
    case,
    *,
    adapter,
    build_kwargs: dict,
    settings: EagleDraftRunnerSettings,
):
    fixture = adapter.build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=settings.capture_batch_size,
    )
    _configure_runner_for_eagle_draft(fixture.runner, case, settings)
    fixture.runner.server_args.speculative_algorithm = "FROZEN_KV_MTP"
    fixture.runner.spec_algorithm = SpeculativeAlgorithm.FROZEN_KV_MTP
    fixture.runner.draft_attn_backend = fixture.backend
    fixture.runner.attn_backend = fixture.backend
    worker = _FrozenKVMTPWorkerHarness(
        fixture=fixture,
        draft_attn_backend=fixture.backend,
        model_forward=adapter.make_model_forward(fixture, settings),
        settings=settings,
    )
    return fixture, worker, fixture.backend


def _run_eagle_draft_eager(
    worker: _EagleDraftWorkerHarness,
    batch: ForwardBatch,
    *,
    init_eager_metadata: Callable[..., None] | None = None,
    settings: EagleDraftRunnerSettings | None = None,
):
    if init_eager_metadata is not None:
        init_eager_metadata(worker, batch, settings)
    else:
        worker.draft_attn_backend.init_forward_metadata(batch)
    batch.mark_forward_metadata_ready()  # mirror production: pre-plan marks
    return worker.draft_forward(batch)


def _run_frozen_kv_mtp_eager(
    worker: _FrozenKVMTPWorkerHarness,
    batch: ForwardBatch,
):
    return worker.draft_forward(batch)


def _capture_eagle_draft_graph_runner(
    worker: _EagleDraftWorkerHarness,
    draft_attn_backend,
    settings: EagleDraftRunnerSettings,
) -> EAGLEDraftCudaGraphRunner:
    with (
        patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.graph_capture",
            _single_rank_graph_capture,
        ),
        patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.get_tensor_model_parallel_rank",
            lambda: 0,
        ),
        patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.get_available_gpu_memory",
            lambda *args, **kwargs: 0.0,
        ),
        get_parallel().override(attn_cp_size=1),
    ):
        _reset_cuda_graph_test_buffers()
        return EAGLEDraftCudaGraphRunner(
            worker,
            draft_attn_backend=draft_attn_backend,
            speculative_num_steps=settings.speculative_num_steps,
        )


def _capture_frozen_kv_mtp_graph_runner(
    worker: _FrozenKVMTPWorkerHarness,
) -> FrozenKVMTPCudaGraphRunner:
    with (
        patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.graph_capture",
            _single_rank_graph_capture,
        ),
        patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.get_tensor_model_parallel_rank",
            lambda: 0,
        ),
        patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.get_available_gpu_memory",
            lambda *args, **kwargs: 0.0,
        ),
        get_parallel().override(attn_cp_size=1),
    ):
        _reset_cuda_graph_test_buffers()
        return FrozenKVMTPCudaGraphRunner(worker)


def _check_eagle_draft_cuda_graph_runner_case(
    case,
    *,
    adapter: EagleDraftCudaGraphRunnerAdapter,
    settings: EagleDraftRunnerSettings,
) -> None:
    if not case.forward_mode.is_decode():
        raise ValueError("EAGLE draft CUDA graph runner coverage expects DECODE cases.")
    if case.batch_size > settings.capture_batch_size:
        raise ValueError("Capture batch size must cover the replay batch size.")
    adapter.check_case(case, settings)


def run_eagle_draft_cuda_graph_runner_case(
    testcase,
    case,
    *,
    adapter: EagleDraftCudaGraphRunnerAdapter,
    build_kwargs: dict,
    settings: EagleDraftRunnerSettings,
):
    try:
        _check_eagle_draft_cuda_graph_runner_case(
            case,
            adapter=adapter,
            settings=settings,
        )
        draft_inputs = adapter.make_draft_inputs(case, settings)

        eager_fixture, eager_worker, _ = _build_eagle_draft_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(eager_fixture, case, draft_inputs, settings)
        eager_batch = adapter.make_forward_batch(case, draft_inputs, settings)
        expected = _run_eagle_draft_eager(
            eager_worker,
            eager_batch,
            init_eager_metadata=adapter.init_eager_metadata,
            settings=settings,
        )

        graph_fixture, graph_worker, graph_backend = _build_eagle_draft_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        graph_batch = adapter.make_forward_batch(case, draft_inputs, settings)
        graph_runner = _capture_eagle_draft_graph_runner(
            graph_worker,
            graph_backend,
            settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)

        testcase.assertTrue(graph_runner.can_run_graph(graph_batch))
        actual = graph_runner.execute(graph_batch)
        adapter.assert_outputs_close(actual, expected, settings)
    finally:
        _reset_cuda_graph_test_buffers()


def run_frozen_kv_mtp_cuda_graph_runner_case(
    testcase,
    case,
    *,
    adapter: EagleDraftCudaGraphRunnerAdapter,
    build_kwargs: dict,
    settings: EagleDraftRunnerSettings,
):
    try:
        if not case.forward_mode.is_decode():
            raise ValueError("Frozen-KV MTP CUDA graph runner coverage expects DECODE.")
        if case.batch_size > settings.capture_batch_size:
            raise ValueError("Capture batch size must cover the replay batch size.")
        adapter.check_case(case, settings)
        draft_inputs = adapter.make_draft_inputs(case, settings)

        eager_fixture, eager_worker, _ = _build_frozen_kv_mtp_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(eager_fixture, case, draft_inputs, settings)
        eager_batch = adapter.make_forward_batch(case, draft_inputs, settings)
        expected = _run_frozen_kv_mtp_eager(eager_worker, eager_batch)

        graph_fixture, graph_worker, _ = _build_frozen_kv_mtp_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        graph_batch = adapter.make_forward_batch(case, draft_inputs, settings)
        graph_runner = _capture_frozen_kv_mtp_graph_runner(graph_worker)
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)

        testcase.assertTrue(graph_runner.can_run_graph(graph_batch))
        actual = graph_runner.execute(graph_batch)
        adapter.assert_outputs_close(actual, expected, settings)
    finally:
        _reset_cuda_graph_test_buffers()


class _DenseEagleDraftForward:
    def __init__(
        self,
        *,
        module,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.module = module
        self.token_embed = nn.Embedding(
            vocab_size, hidden_size, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype, device=device
        )

    def __call__(self, forward_batch: ForwardBatch):
        assert (
            forward_batch.forward_metadata_ready
        ), "draft-loop forward reached the runner without a pre-planned batch"
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states
        if hidden_states is None:
            raise ValueError("EAGLE draft runner tests expect hidden-state drafts.")

        token_hidden = self.token_embed(forward_batch.input_ids)
        hidden_states = hidden_states + token_hidden
        hidden_states = self.module(hidden_states, forward_batch)
        logits = self.lm_head(hidden_states).float()
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=hidden_states,
            )
        )


class _FrozenKVMTPDenseDraftForward:
    def __init__(
        self,
        *,
        module,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.module = module
        self.token_embed = nn.Embedding(
            vocab_size, hidden_size, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype, device=device
        )

    def __call__(self, forward_batch: ForwardBatch):
        assert (
            forward_batch.forward_metadata_ready
        ), "draft-loop forward reached the runner without a pre-planned batch"
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states
        if hidden_states is None:
            raise ValueError("Frozen-KV MTP runner tests expect hidden-state drafts.")

        token_hidden = self.token_embed(forward_batch.input_ids)
        hidden_states = hidden_states + token_hidden
        q = self.module.q_proj(hidden_states)
        attn_output = self.module.attn(
            q,
            None,
            None,
            forward_batch,
            save_kv_cache=False,
        )
        hidden_states = self.module.o_proj(attn_output)
        logits = self.lm_head(hidden_states).float()
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=hidden_states,
            )
        )


def _make_dense_model_forward(fixture, settings: EagleDraftRunnerSettings):
    return _DenseEagleDraftForward(
        module=fixture.actual_module,
        hidden_size=settings.hidden_size,
        vocab_size=settings.vocab_size,
        dtype=settings.dtype,
        device=settings.device,
    )


def _make_dense_frozen_kv_mtp_model_forward(
    fixture,
    settings: EagleDraftRunnerSettings,
):
    return _FrozenKVMTPDenseDraftForward(
        module=fixture.actual_module,
        hidden_size=settings.hidden_size,
        vocab_size=settings.vocab_size,
        dtype=settings.dtype,
        device=settings.device,
    )


def _make_dense_draft_inputs(
    case: DenseAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    with _seeded_rng(4080 + len(case.name) + settings.topk, device=settings.device):
        topk_index = (
            torch.arange(
                case.batch_size * settings.topk,
                dtype=torch.int64,
                device=settings.device,
            ).view(case.batch_size, settings.topk)
            + 3
        ) % settings.vocab_size
        topk_p = torch.linspace(
            0.6,
            0.9,
            steps=case.batch_size * settings.topk,
            dtype=torch.float32,
            device=settings.device,
        ).view(case.batch_size, settings.topk)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        return {
            "hidden_states": torch.randn(
                case.batch_size,
                settings.hidden_size,
                dtype=settings.dtype,
                device=settings.device,
            ),
            "topk_p": topk_p,
            "topk_index": topk_index,
        }


def _make_dense_frozen_kv_mtp_draft_inputs(
    case: DenseAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    draft_inputs = _make_dense_draft_inputs(case, settings)
    # `draft_forward` now runs the assistant seed iter in-graph: it consumes the
    # per-req bonus token + target hidden and derives iter-0 topk_p/topk_index
    # itself, so the fixture supplies `bonus_tokens` rather than topk_p/index.
    with _seeded_rng(4090 + len(case.name) + settings.topk, device=settings.device):
        bonus_tokens = torch.randint(
            0,
            settings.vocab_size,
            (case.batch_size,),
            dtype=torch.int64,
            device=settings.device,
        )
    return {
        "hidden_states": draft_inputs["hidden_states"],
        "bonus_tokens": bonus_tokens,
    }


def _prepare_dense_draft_replay_state(
    fixture,
    case: DenseAttentionCase,
    _draft_inputs,
    settings: EagleDraftRunnerSettings,
) -> None:
    prepare_dense_runner_inputs(
        fixture,
        case,
        fixture.forward_batch,
        {"prefix_hidden": fixture.prefix_hidden},
        max_context_len=settings.max_context_len,
    )
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for branch in range(settings.topk):
            for step in range(settings.speculative_num_steps):
                position = prefix_len + branch * settings.speculative_num_steps + step
                fixture.runner.req_to_token_pool.req_to_token[
                    req_idx,
                    position,
                ] = _dense_token_loc(
                    req_idx,
                    position,
                    page_size=case.page_size,
                    max_context_len=settings.max_context_len,
                )


def _dense_draft_cache_position(
    *,
    prefix_len: int,
    branch: int,
    step: int,
    topk: int,
    speculative_num_steps: int,
) -> int:
    if topk == 1:
        return prefix_len + step
    return prefix_len + branch * speculative_num_steps + step


def _check_dense_draft_cache_layout(
    case: DenseAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> None:
    if settings.topk > 1 and case.page_size != 1:
        raise ValueError(
            "The dense EAGLE draft runner fixture covers tree draft with "
            "page_size=1, where branch cache slots are laid out linearly."
        )
    if settings.topk > 1:
        for prefix_len in case.prefix_lens:
            if (
                prefix_len + settings.topk * settings.speculative_num_steps
                > settings.max_context_len
            ):
                raise ValueError(
                    "Draft cache layout exceeds the configured context len."
                )


def _make_dense_eagle_draft_forward_batch(
    case: DenseAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    out_cache_locs = []
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for branch in range(settings.topk):
            for step in range(settings.speculative_num_steps):
                position = _dense_draft_cache_position(
                    prefix_len=prefix_len,
                    branch=branch,
                    step=step,
                    topk=settings.topk,
                    speculative_num_steps=settings.speculative_num_steps,
                )
                out_cache_locs.append(
                    _dense_token_loc(
                        req_idx,
                        position,
                        page_size=case.page_size,
                        max_context_len=settings.max_context_len,
                    )
                )

    spec_info = EagleDraftInput(
        topk_p=draft_inputs["topk_p"].clone(),
        topk_index=draft_inputs["topk_index"].clone(),
        hidden_states=draft_inputs["hidden_states"].clone(),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=settings.topk,
        num_tokens_for_logprob_per_req=settings.topk,
    )
    seq_lens = torch.tensor(
        case.prefix_lens,
        dtype=torch.int32,
        device=settings.device,
    )
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=case.batch_size,
        input_ids=None,
        req_pool_indices=torch.arange(
            case.batch_size,
            dtype=torch.int32,
            device=settings.device,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(
            out_cache_locs,
            dtype=torch.int64,
            device=settings.device,
        ),
        seq_lens_sum=sum(case.prefix_lens),
        positions=seq_lens.repeat_interleave(settings.topk).to(torch.int64),
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def _make_dense_frozen_kv_mtp_forward_batch(
    case: DenseAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    spec_info = FrozenKVMTPDraftInput(
        hidden_states=draft_inputs["hidden_states"].clone(),
        bonus_tokens=draft_inputs["bonus_tokens"].clone(),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=settings.topk,
        num_tokens_for_logprob_per_req=settings.topk,
    )
    seq_lens = torch.tensor(
        case.prefix_lens,
        dtype=torch.int32,
        device=settings.device,
    )
    positions = torch.clamp(seq_lens - 1, min=0).to(torch.int64)
    spec_info.positions = positions
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=case.batch_size,
        input_ids=None,
        req_pool_indices=torch.arange(
            case.batch_size,
            dtype=torch.int32,
            device=settings.device,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=None,
        seq_lens_sum=sum(case.prefix_lens),
        positions=positions,
        spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def run_dense_eagle_draft_cuda_graph_runner_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int = 3,
    cuda_graph_capture_batch_size: int = 4,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    settings = EagleDraftRunnerSettings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    adapter = EagleDraftCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_model_forward,
        make_draft_inputs=_make_dense_draft_inputs,
        prepare_replay_state=_prepare_dense_draft_replay_state,
        make_forward_batch=_make_dense_eagle_draft_forward_batch,
        check_case=_check_dense_draft_cache_layout,
    )
    run_eagle_draft_cuda_graph_runner_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        settings=settings,
    )


def run_dense_frozen_kv_mtp_cuda_graph_runner_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int = 3,
    cuda_graph_capture_batch_size: int = 4,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    settings = EagleDraftRunnerSettings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    adapter = EagleDraftCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_frozen_kv_mtp_model_forward,
        make_draft_inputs=_make_dense_frozen_kv_mtp_draft_inputs,
        prepare_replay_state=_prepare_dense_draft_replay_state,
        make_forward_batch=_make_dense_frozen_kv_mtp_forward_batch,
        check_case=_check_dense_draft_cache_layout,
    )
    run_frozen_kv_mtp_cuda_graph_runner_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        settings=settings,
    )


class _MLAEagleDraftForward:
    def __init__(
        self,
        *,
        module,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.module = module
        self.token_embed = nn.Embedding(
            vocab_size, hidden_size, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype, device=device
        )

    def __call__(self, forward_batch: ForwardBatch):
        assert (
            forward_batch.forward_metadata_ready
        ), "draft-loop forward reached the runner without a pre-planned batch"
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states
        if hidden_states is None:
            raise ValueError("EAGLE draft runner tests expect hidden-state drafts.")

        token_hidden = self.token_embed(forward_batch.input_ids)
        hidden_states = hidden_states + token_hidden
        hidden_states = self.module(hidden_states, forward_batch)
        logits = self.lm_head(hidden_states).float()
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=hidden_states,
            )
        )


def _make_mla_model_forward(fixture, settings: EagleDraftRunnerSettings):
    return _MLAEagleDraftForward(
        module=fixture.actual_module,
        hidden_size=settings.hidden_size,
        vocab_size=settings.vocab_size,
        dtype=settings.dtype,
        device=settings.device,
    )


def _make_mla_draft_inputs(
    case: MLAAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    with _seeded_rng(5080 + len(case.name) + settings.topk, device=settings.device):
        topk_index = (
            torch.arange(
                case.batch_size * settings.topk,
                dtype=torch.int64,
                device=settings.device,
            ).view(case.batch_size, settings.topk)
            + 5
        ) % settings.vocab_size
        topk_p = torch.linspace(
            0.55,
            0.95,
            steps=case.batch_size * settings.topk,
            dtype=torch.float32,
            device=settings.device,
        ).view(case.batch_size, settings.topk)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        return {
            "hidden_states": torch.randn(
                case.batch_size,
                settings.hidden_size,
                dtype=settings.dtype,
                device=settings.device,
            ),
            "topk_p": topk_p,
            "topk_index": topk_index,
        }


def _prepare_mla_draft_replay_state(
    fixture,
    case: MLAAttentionCase,
    _draft_inputs,
    settings: EagleDraftRunnerSettings,
) -> None:
    prepare_mla_runner_inputs(
        fixture,
        case,
        fixture.forward_batch,
        {"prefix_hidden": fixture.prefix_hidden},
        max_context_len=settings.max_context_len,
    )
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for branch in range(settings.topk):
            for step in range(settings.speculative_num_steps):
                position = _dense_draft_cache_position(
                    prefix_len=prefix_len,
                    branch=branch,
                    step=step,
                    topk=settings.topk,
                    speculative_num_steps=settings.speculative_num_steps,
                )
                fixture.runner.req_to_token_pool.req_to_token[
                    req_idx,
                    position,
                ] = _mla_token_loc(
                    req_idx,
                    position,
                    page_size=case.page_size,
                    max_context_len=settings.max_context_len,
                )


def _check_mla_draft_cache_layout(
    case: MLAAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> None:
    if settings.topk > 1 and case.page_size != 1:
        raise ValueError(
            "The MLA EAGLE draft runner fixture covers tree draft with "
            "page_size=1, where branch cache slots are laid out linearly."
        )
    if settings.topk > 1:
        for prefix_len in case.prefix_lens:
            if (
                prefix_len + settings.topk * settings.speculative_num_steps
                > settings.max_context_len
            ):
                raise ValueError(
                    "Draft cache layout exceeds the configured context len."
                )


def _make_mla_eagle_draft_forward_batch(
    case: MLAAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    out_cache_locs = []
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for branch in range(settings.topk):
            for step in range(settings.speculative_num_steps):
                position = _dense_draft_cache_position(
                    prefix_len=prefix_len,
                    branch=branch,
                    step=step,
                    topk=settings.topk,
                    speculative_num_steps=settings.speculative_num_steps,
                )
                out_cache_locs.append(
                    _mla_token_loc(
                        req_idx,
                        position,
                        page_size=case.page_size,
                        max_context_len=settings.max_context_len,
                    )
                )

    spec_info = EagleDraftInput(
        topk_p=draft_inputs["topk_p"].clone(),
        topk_index=draft_inputs["topk_index"].clone(),
        hidden_states=draft_inputs["hidden_states"].clone(),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=settings.topk,
        num_tokens_for_logprob_per_req=settings.topk,
    )
    seq_lens = torch.tensor(
        case.prefix_lens,
        dtype=torch.int32,
        device=settings.device,
    )
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=case.batch_size,
        input_ids=None,
        req_pool_indices=torch.arange(
            case.batch_size,
            dtype=torch.int32,
            device=settings.device,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(
            out_cache_locs,
            dtype=torch.int64,
            device=settings.device,
        ),
        seq_lens_sum=sum(case.prefix_lens),
        positions=seq_lens.repeat_interleave(settings.topk).to(torch.int64),
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def run_mla_eagle_draft_cuda_graph_runner_case(
    testcase,
    case: MLAAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int = 3,
    cuda_graph_capture_batch_size: int = 4,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
):
    settings = EagleDraftRunnerSettings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
    adapter = EagleDraftCudaGraphRunnerAdapter(
        build_fixture=build_mla_attention_fixture,
        make_model_forward=_make_mla_model_forward,
        make_draft_inputs=_make_mla_draft_inputs,
        prepare_replay_state=_prepare_mla_draft_replay_state,
        make_forward_batch=_make_mla_eagle_draft_forward_batch,
        check_case=_check_mla_draft_cache_layout,
    )
    run_eagle_draft_cuda_graph_runner_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        settings=settings,
    )


class _DSV4EagleDraftForward:
    """Minimal DSV4 draft model forward.

    Mirrors `_MLAEagleDraftForward` but routes through
    `ProjectedDSV4Attention.forward` which production-style writes K to the
    SWA pool before invoking the active step backend.
    """

    def __init__(
        self,
        *,
        module,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.module = module
        self.token_embed = nn.Embedding(
            vocab_size, hidden_size, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype, device=device
        )

    def __call__(self, forward_batch: ForwardBatch):
        assert (
            forward_batch.forward_metadata_ready
        ), "draft-loop forward reached the runner without a pre-planned batch"
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states
        if hidden_states is None:
            raise ValueError("EAGLE draft runner tests expect hidden-state drafts.")

        token_hidden = self.token_embed(forward_batch.input_ids)
        hidden_states = hidden_states + token_hidden
        hidden_states = self.module(hidden_states, forward_batch)
        logits = self.lm_head(hidden_states).float()
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=hidden_states,
            )
        )


def _make_dsv4_model_forward(fixture, settings: EagleDraftRunnerSettings):
    return _DSV4EagleDraftForward(
        module=fixture.actual_module,
        hidden_size=settings.hidden_size,
        vocab_size=settings.vocab_size,
        dtype=settings.dtype,
        device=settings.device,
    )


def _make_dsv4_draft_inputs(
    case: DSV4AttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    with _seeded_rng(9180 + len(case.name), device=settings.device):
        topk_index = (
            torch.arange(
                case.batch_size * settings.topk,
                dtype=torch.int64,
                device=settings.device,
            ).view(case.batch_size, settings.topk)
            + 5
        ) % settings.vocab_size
        topk_p = torch.linspace(
            0.55,
            0.95,
            steps=case.batch_size * settings.topk,
            dtype=torch.float32,
            device=settings.device,
        ).view(case.batch_size, settings.topk)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        return {
            "hidden_states": torch.randn(
                case.batch_size,
                settings.hidden_size,
                dtype=settings.dtype,
                device=settings.device,
            ),
            "topk_p": topk_p,
            "topk_index": topk_index,
        }


def _prepare_dsv4_draft_replay_state(
    fixture,
    case: DSV4AttentionCase,
    _draft_inputs,
    settings: EagleDraftRunnerSettings,
) -> None:
    # Map every (req, position) the draft will write/read to a real slot in
    # the SWA pool. DSV4 chain draft writes one new token per step at
    # `prefix_len + step` (topk == 1).
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for pos in range(prefix_len):
            runner.req_to_token_pool.req_to_token[req_idx, pos] = _dsv4_token_loc(
                req_idx, pos, max_context_len=max_context_len
            )
        for step in range(settings.speculative_num_steps):
            pos = prefix_len + step
            runner.req_to_token_pool.req_to_token[req_idx, pos] = _dsv4_token_loc(
                req_idx, pos, max_context_len=max_context_len
            )


def _check_dsv4_draft_cache_layout(
    case: DSV4AttentionCase,
    settings: EagleDraftRunnerSettings,
) -> None:
    if case.compress_ratio != 0:
        raise ValueError(
            "DSV4 EAGLE draft runner coverage is SWA-only. Production "
            "`DeepseekV4ModelNextN` hardcodes `compress_ratio_override=0` so "
            "C4/C128 + draft is unreachable."
        )
    if settings.topk != 1:
        raise ValueError(
            "DSV4 speculative decoding asserts `topk in [0, 1]` "
            "(`deepseek_v4_backend.py:369`); tree draft is structurally "
            "impossible."
        )
    if case.page_size != DSV4_PAGE_SIZE:
        raise ValueError(
            f"DSV4 backend asserts page_size == {DSV4_PAGE_SIZE} "
            f"(got {case.page_size})."
        )
    for prefix_len in case.prefix_lens:
        if prefix_len + settings.speculative_num_steps > DSV4_SWA_WINDOW:
            raise ValueError(
                "Prefix + speculative steps exceed the SWA window; the "
                "fixture currently only covers within-window draft."
            )


def _init_dsv4_eager_metadata(
    worker,
    batch: ForwardBatch,
    settings: EagleDraftRunnerSettings,
) -> None:
    """Per-step DSV4 init for the eager comparison path.

    After PR #26239 `DeepseekV4AttnBackend.init_forward_metadata` slices the
    multi-step `out_cache_loc` internally using `self.speculative_step_id`
    (each per-step backend in `multi_step_backend.attn_backends[i]` was
    constructed with `speculative_step_id=i`). Pass the full
    `bs * topk * num_steps` buffer through each per-step init unchanged.
    """
    multi_step_backend = worker.draft_attn_backend
    for attn_backend in multi_step_backend.attn_backends:
        attn_backend.init_forward_metadata(batch)


def _make_dsv4_eagle_draft_forward_batch(
    case: DSV4AttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    out_cache_locs = []
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for step in range(settings.speculative_num_steps):
            out_cache_locs.append(
                _dsv4_token_loc(
                    req_idx,
                    prefix_len + step,
                    max_context_len=settings.max_context_len,
                )
            )

    spec_info = EagleDraftInput(
        topk_p=draft_inputs["topk_p"].clone(),
        topk_index=draft_inputs["topk_index"].clone(),
        hidden_states=draft_inputs["hidden_states"].clone(),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=settings.topk,
        num_tokens_for_logprob_per_req=settings.topk,
    )
    seq_lens = torch.tensor(
        case.prefix_lens,
        dtype=torch.int32,
        device=settings.device,
    )
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=case.batch_size,
        input_ids=None,
        req_pool_indices=torch.arange(
            case.batch_size,
            dtype=torch.int32,
            device=settings.device,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(
            out_cache_locs,
            dtype=torch.int64,
            device=settings.device,
        ),
        seq_lens_sum=sum(case.prefix_lens),
        positions=seq_lens.repeat_interleave(settings.topk).to(torch.int64),
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def run_dsv4_eagle_draft_cuda_graph_runner_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int = 3,
    cuda_graph_capture_batch_size: int = 4,
    hidden_size: int = DSV4_HEAD_DIM,
    max_context_len: int = 256,
    vocab_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    settings = EagleDraftRunnerSettings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        atol=DSV4_ATOL,
        rtol=DSV4_RTOL,
    )
    adapter = EagleDraftCudaGraphRunnerAdapter(
        build_fixture=build_dsv4_attention_fixture,
        make_model_forward=_make_dsv4_model_forward,
        make_draft_inputs=_make_dsv4_draft_inputs,
        prepare_replay_state=_prepare_dsv4_draft_replay_state,
        make_forward_batch=_make_dsv4_eagle_draft_forward_batch,
        check_case=_check_dsv4_draft_cache_layout,
        init_eager_metadata=_init_dsv4_eager_metadata,
    )
    run_eagle_draft_cuda_graph_runner_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# DSV4 EAGLE draft-extend CUDA-graph runner adapter (SWA only)
# ---------------------------------------------------------------------------
#
# `DeepseekV4ModelNextN` hardcodes `compress_ratio_override=0`, so DSV4
# production EAGLE draft-extend is always SWA-only. The
# `DeepseekV4AttnBackend` instantiated via `DraftBackendFactory._create_dsv4_prefill_backend`
# handles the draft-extend forward + CG capture/replay paths (`init_forward_metadata_draft_extend`
# at `deepseek_v4_backend.py:636-663` forces `need_compress=False`).


class _DSAEagleDraftForward:
    """Minimal DSA draft model forward.

    Synthesizes `topk_indices` from `batch.seq_lens` (production gets
    them from the DSA indexer module). For chain draft (topk=1) the
    indices for each request's query token are the trailing
    `DSA_SPARSE_INDEX_TOPK` positions, in [0, seq_len+1) token-position
    space — the backend translates them to pool slots via
    `transform_index_page_table_decode`.
    """

    def __init__(
        self,
        *,
        module,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.module = module
        self.token_embed = nn.Embedding(
            vocab_size, hidden_size, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype, device=device
        )

    def _synthesize_topk_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Trailing-topk indices per query in token-position space.

        At a draft decode step, each request has one query token and
        `seq_lens[req_idx] + 1` keys (the prefix + steps written so far
        + the just-written step). The trailing-topk window covers the
        most-recent `DSA_SPARSE_INDEX_TOPK` positions, with `-1` padding
        when `key_count < DSA_SPARSE_INDEX_TOPK`.

        Built entirely on-GPU (no CPU<->GPU copies) so the captured CUDA
        graph stays valid — the previous CPU-list construction tripped
        `cudaErrorOperationNotPermitted` during capture.
        """
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        device = seq_lens.device
        topk = DSA_SPARSE_INDEX_TOPK
        key_counts = seq_lens + 1
        key_starts = torch.clamp(key_counts - topk, min=0)
        offsets = torch.arange(topk, dtype=torch.int32, device=device)
        indices = key_starts[:, None] + offsets[None, :]
        mask = indices < key_counts[:, None]
        return torch.where(
            mask,
            indices,
            torch.full_like(indices, -1),
        )

    def __call__(self, forward_batch: ForwardBatch):
        assert (
            forward_batch.forward_metadata_ready
        ), "draft-loop forward reached the runner without a pre-planned batch"
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states
        if hidden_states is None:
            raise ValueError("EAGLE draft runner tests expect hidden-state drafts.")

        token_hidden = self.token_embed(forward_batch.input_ids)
        hidden_states = hidden_states + token_hidden

        # DSA projects to nope+rope separately and writes the new K to
        # cache through `module.attn` (which delegates to the backend's
        # `forward_decode`).
        q_nope, q_rope = self.module.project_q(hidden_states)
        k_nope, k_rope = self.module.project_k(hidden_states)

        topk_indices = self._synthesize_topk_indices(forward_batch)

        attn_output = self.module.attn(
            q_nope,
            k_nope,
            k_nope,  # MLA absorbs V into K
            forward_batch,
            k_rope=k_rope,
            q_rope=q_rope,
            topk_indices=topk_indices,
        )
        attn_output = attn_output.reshape(
            -1, self.module.num_heads * self.module.qk_nope_head_dim
        )
        hidden_states = self.module.o_proj(attn_output)
        # Map back to spec hidden_size dim. The o_proj output is
        # `hidden_size`; lm_head expects `hidden_size` too.
        logits = self.lm_head(hidden_states).float()
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=hidden_states,
            )
        )


def _make_dsa_model_forward(fixture, settings: EagleDraftRunnerSettings):
    return _DSAEagleDraftForward(
        module=fixture.actual_module,
        hidden_size=settings.hidden_size,
        vocab_size=settings.vocab_size,
        dtype=settings.dtype,
        device=settings.device,
    )


def _make_dsa_draft_inputs(
    case: DSAAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    with _seeded_rng(9380 + len(case.name), device=settings.device):
        topk_index = (
            torch.arange(
                case.batch_size * settings.topk,
                dtype=torch.int64,
                device=settings.device,
            ).view(case.batch_size, settings.topk)
            + 5
        ) % settings.vocab_size
        topk_p = torch.linspace(
            0.55,
            0.95,
            steps=case.batch_size * settings.topk,
            dtype=torch.float32,
            device=settings.device,
        ).view(case.batch_size, settings.topk)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        return {
            "hidden_states": torch.randn(
                case.batch_size,
                settings.hidden_size,
                dtype=settings.dtype,
                device=settings.device,
            ),
            "topk_p": topk_p,
            "topk_index": topk_index,
        }


def _prepare_dsa_draft_replay_state(
    fixture,
    case: DSAAttentionCase,
    _draft_inputs,
    settings: EagleDraftRunnerSettings,
) -> None:
    """Populate req_to_token mappings for prefix + draft steps.

    DSA chain decode writes one new token per step at the position
    `prefix_len + step`. The `_dsa_token_loc` helper assigns a unique
    pool slot per (req, position).
    """
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for pos in range(prefix_len):
            runner.req_to_token_pool.req_to_token[req_idx, pos] = _dsa_token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )
        for step in range(settings.speculative_num_steps):
            pos = prefix_len + step
            runner.req_to_token_pool.req_to_token[req_idx, pos] = _dsa_token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )


def _check_dsa_draft_cache_layout(
    case: DSAAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> None:
    if settings.topk != 1:
        raise ValueError(
            "DSA EAGLE draft runner coverage is chain-only (topk=1). Tree "
            "draft requires parent-indices plumbing through the "
            "topk_indices synthesis; deferred."
        )
    if case.page_size != DSA_PAGE_SIZE:
        raise ValueError(
            f"DSA backend requires page_size == {DSA_PAGE_SIZE} (got {case.page_size})."
        )


def _make_dsa_eagle_draft_forward_batch(
    case: DSAAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    out_cache_locs = []
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for step in range(settings.speculative_num_steps):
            out_cache_locs.append(
                _dsa_token_loc(
                    req_idx,
                    prefix_len + step,
                    page_size=case.page_size,
                    max_context_len=settings.max_context_len,
                )
            )

    spec_info = EagleDraftInput(
        topk_p=draft_inputs["topk_p"].clone(),
        topk_index=draft_inputs["topk_index"].clone(),
        hidden_states=draft_inputs["hidden_states"].clone(),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=settings.topk,
        num_tokens_for_logprob_per_req=settings.topk,
    )
    seq_lens = torch.tensor(
        case.prefix_lens,
        dtype=torch.int32,
        device=settings.device,
    )
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=case.batch_size,
        input_ids=torch.zeros(
            case.batch_size * settings.topk,
            dtype=torch.int32,
            device=settings.device,
        ),
        req_pool_indices=torch.arange(
            case.batch_size,
            dtype=torch.int32,
            device=settings.device,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(
            out_cache_locs,
            dtype=torch.int64,
            device=settings.device,
        ),
        seq_lens_sum=sum(case.prefix_lens),
        positions=seq_lens.repeat_interleave(settings.topk).to(torch.int64),
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def run_dsa_eagle_draft_cuda_graph_runner_case(
    testcase,
    case: DSAAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 2,
    speculative_num_draft_tokens: int = 2,
    cuda_graph_capture_batch_size: int = 2,
    # The DSA sparse fixture's `ProjectedDSASparseAttention.q_nope_proj`
    # expects an input dim equal to the fixture's `hidden_size`
    # (= DEFAULT_HIDDEN_SIZE = 64 from dense_attention). The EAGLE
    # draft's synthetic `token_embed` / `lm_head` operate at the same
    # `hidden_size` so the spec_info `hidden_states` shape lines up.
    hidden_size: int = 64,
    max_context_len: int = 256,
    vocab_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """DSA EAGLE draft CUDA-graph runner coverage. Chain-only (topk=1)
    for now; the DSA indexer-replacement synthesis here uses trailing
    topk in token-position space.

    Pads `hidden_size` via the existing DSA sparse fixture's
    `qk_nope + qk_rope` head_dim. `vocab_size` is intentionally small
    so the synthetic `token_embed`/`lm_head` stay cheap.
    """
    settings = EagleDraftRunnerSettings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        atol=DSA_SPARSE_ATOL,
        rtol=DSA_SPARSE_RTOL,
    )
    adapter = EagleDraftCudaGraphRunnerAdapter(
        build_fixture=build_dsa_sparse_attention_fixture,
        make_model_forward=_make_dsa_model_forward,
        make_draft_inputs=_make_dsa_draft_inputs,
        prepare_replay_state=_prepare_dsa_draft_replay_state,
        make_forward_batch=_make_dsa_eagle_draft_forward_batch,
        check_case=_check_dsa_draft_cache_layout,
    )
    run_eagle_draft_cuda_graph_runner_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# DSA EAGLE draft-extend CUDA-graph runner adapter
# ---------------------------------------------------------------------------
#
# Draft-extend differs from draft-decode in three ways for DSA:
#   1. Multi-query-per-request: `num_input_tokens = sum(input_lens)`.
#   2. Routes through `forward_extend` rather than `forward_decode`.
#      Production picks `dsa_decode_impl` (default `flashmla_kv`)
#      because `is_draft_extend_v2()` is in the
#      decode-impl branch (`dsa_backend.py:1352-1358`).
#   3. DraftBackendFactory returns a single `DeepseekSparseAttnBackend`
#      (not a multi-step wrapper) via `_create_dsa_prefill_backend`.
#
# The topk_indices synthesis uses `batch.positions` to compute the
# absolute key_count per query token; same trailing-topk shape as the
# draft-decode synthesis. Built entirely on-GPU.
