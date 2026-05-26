from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput, EagleDraftInput
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPContext,
    FrozenKVMTPDraftInput,
)
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import fast_topk

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
from ..attention_methods.dense_attention import (
    _make_forward_batch as _make_dense_forward_batch,
)
from ..attention_methods.dense_attention import _token_loc as _dense_token_loc
from ..attention_methods.dense_attention import (
    build_dense_attention_fixture,
    prepare_dense_runner_inputs,
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
from ..attention_methods.mla_attention import (
    _make_forward_batch as _make_mla_forward_batch,
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


def _assert_draft_extend_outputs_close(actual, expected, settings) -> None:
    torch.testing.assert_close(
        actual.next_token_logits,
        expected.next_token_logits,
        atol=settings.atol,
        rtol=settings.rtol,
    )
    torch.testing.assert_close(
        actual.hidden_states,
        expected.hidden_states,
        atol=settings.atol,
        rtol=settings.rtol,
    )
    torch.testing.assert_close(
        actual.topk_p,
        expected.topk_p,
        atol=settings.atol,
        rtol=settings.rtol,
    )
    torch.testing.assert_close(actual.topk_index, expected.topk_index)


@dataclass(frozen=True)
class EagleDraftExtendCudaGraphRunnerAdapter:
    build_fixture: Callable[..., Any]
    make_model_forward: Callable[[Any, EagleDraftRunnerSettings], nn.Module]
    make_draft_inputs: Callable[[Any, EagleDraftRunnerSettings], Any]
    prepare_replay_state: Callable[[Any, Any, Any, EagleDraftRunnerSettings], None]
    make_forward_batch: Callable[
        [Any, Any, Any, EagleDraftRunnerSettings], ForwardBatch
    ]
    check_case: Callable[[Any, EagleDraftRunnerSettings], None] = (
        lambda _case, _settings: None
    )
    assert_outputs_close: Callable[[Any, Any, EagleDraftRunnerSettings], None] = (
        _assert_draft_extend_outputs_close
    )


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
        self.draft_forward = MethodType(EAGLEWorker.draft_forward, self)

    @property
    def draft_model_runner(self):
        return self.model_runner


class _EagleDraftExtendWorkerHarness:
    def __init__(
        self,
        *,
        fixture,
        draft_extend_attn_backend,
        model_forward: nn.Module,
        settings: EagleDraftRunnerSettings,
    ):
        self.model_runner = fixture.runner
        self.target_worker = SimpleNamespace(model_runner=fixture.runner)
        self.draft_extend_attn_backend = draft_extend_attn_backend
        self.topk = settings.topk
        self.speculative_num_steps = settings.speculative_num_steps
        self.speculative_num_draft_tokens = settings.speculative_num_draft_tokens
        self.server_args = fixture.runner.server_args
        self.model_config = fixture.runner.model_config
        self.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        self.eagle_use_aux_hidden_state = False
        self.hot_token_id = None
        self.model_runner.model = model_forward

    @property
    def draft_model_runner(self):
        return self.model_runner


class _EagleDraftExtendV2WorkerHarness:
    def __init__(
        self,
        *,
        fixture,
        draft_extend_attn_backend,
        model_forward: nn.Module,
        settings: EagleDraftRunnerSettings,
    ):
        self.draft_runner = fixture.runner
        self.target_worker = SimpleNamespace(model_runner=fixture.runner)
        self.draft_extend_attn_backend = draft_extend_attn_backend
        self.topk = settings.topk
        self.speculative_num_steps = settings.speculative_num_steps
        self.speculative_num_draft_tokens = settings.speculative_num_draft_tokens
        self.server_args = fixture.runner.server_args
        self.model_config = fixture.runner.model_config
        self.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        self.eagle_use_aux_hidden_state = False
        self.hot_token_id = None
        self.draft_runner.model = model_forward


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
        self.draft_forward = MethodType(FrozenKVMTPWorker.draft_forward, self)
        self._frozen_kv_target_view = MethodType(
            FrozenKVMTPWorker._frozen_kv_target_view,
            self,
        )
        self._target_kv_pool_view = MethodType(
            FrozenKVMTPWorker._target_kv_pool_view,
            self,
        )
        self._set_positions = MethodType(FrozenKVMTPWorker._set_positions, self)
        self._init_frozen_kv_metadata = MethodType(
            FrozenKVMTPWorker._init_frozen_kv_metadata,
            self,
        )
        self._init_frozen_kv_metadata_capture_cuda_graph = MethodType(
            FrozenKVMTPWorker._init_frozen_kv_metadata_capture_cuda_graph,
            self,
        )
        self._init_frozen_kv_metadata_replay_cuda_graph = MethodType(
            FrozenKVMTPWorker._init_frozen_kv_metadata_replay_cuda_graph,
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
        "cuda_graph_bs": [settings.capture_batch_size],
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


def _build_eagle_draft_extend_fixture(
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
    _configure_runner_for_eagle_draft(
        fixture.runner,
        case,
        settings,
        speculative_attention_mode="prefill",
    )
    draft_extend_attn_backend = DraftBackendFactory(
        fixture.runner.server_args,
        fixture.runner,
        settings.topk,
        settings.speculative_num_steps,
    ).create_draft_extend_backend()
    if draft_extend_attn_backend is None:
        testcase.skipTest(f"{case.backend} draft-extend backend is not available")
    fixture.runner.draft_extend_attn_backend = draft_extend_attn_backend
    fixture.runner.attn_backend = draft_extend_attn_backend
    worker_cls = (
        _EagleDraftExtendV2WorkerHarness
        if case.forward_mode.is_draft_extend_v2()
        else _EagleDraftExtendWorkerHarness
    )
    worker = worker_cls(
        fixture=fixture,
        draft_extend_attn_backend=draft_extend_attn_backend,
        model_forward=adapter.make_model_forward(fixture, settings),
        settings=settings,
    )
    return fixture, worker, draft_extend_attn_backend


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
):
    worker.draft_attn_backend.init_forward_metadata(batch)
    return worker.draft_forward(batch)


def _run_frozen_kv_mtp_eager(
    worker: _FrozenKVMTPWorkerHarness,
    batch: ForwardBatch,
):
    return worker.draft_forward(batch, skip_attn_backend_init=False)


def _capture_eagle_draft_graph_runner(
    worker: _EagleDraftWorkerHarness,
    draft_attn_backend,
    settings: EagleDraftRunnerSettings,
) -> EAGLEDraftCudaGraphRunner:
    with (
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.graph_capture",
            _single_rank_graph_capture,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_tensor_model_parallel_rank",
            lambda: 0,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_available_gpu_memory",
            lambda *args, **kwargs: 0.0,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_attention_cp_size",
            lambda: 1,
        ),
    ):
        _reset_cuda_graph_test_buffers()
        return EAGLEDraftCudaGraphRunner(
            worker,
            draft_attn_backend=draft_attn_backend,
            speculative_num_steps=settings.speculative_num_steps,
        )


def _capture_eagle_draft_extend_graph_runner(
    worker: _EagleDraftExtendWorkerHarness,
    draft_extend_attn_backend,
    settings: EagleDraftRunnerSettings,
) -> EAGLEDraftExtendCudaGraphRunner:
    with (
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.graph_capture",
            _single_rank_graph_capture,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_tensor_model_parallel_rank",
            lambda: 0,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_available_gpu_memory",
            lambda *args, **kwargs: 0.0,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_attention_cp_size",
            lambda: 1,
        ),
    ):
        _reset_cuda_graph_test_buffers()
        return EAGLEDraftExtendCudaGraphRunner(
            worker,
            draft_extend_attn_backend=draft_extend_attn_backend,
            speculative_num_steps=settings.speculative_num_steps,
        )


def _capture_frozen_kv_mtp_graph_runner(
    worker: _FrozenKVMTPWorkerHarness,
) -> FrozenKVMTPCudaGraphRunner:
    with (
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.graph_capture",
            _single_rank_graph_capture,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_tensor_model_parallel_rank",
            lambda: 0,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_available_gpu_memory",
            lambda *args, **kwargs: 0.0,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_attention_cp_size",
            lambda: 1,
        ),
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
        expected = _run_eagle_draft_eager(eager_worker, eager_batch)

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

        testcase.assertTrue(graph_runner.can_run(graph_batch))
        actual = graph_runner.replay(graph_batch)
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

        testcase.assertTrue(graph_runner.can_run(graph_batch))
        actual = graph_runner.replay(graph_batch)
        adapter.assert_outputs_close(actual, expected, settings)
    finally:
        _reset_cuda_graph_test_buffers()


def _check_eagle_draft_extend_cuda_graph_runner_case(
    case,
    *,
    adapter: EagleDraftExtendCudaGraphRunnerAdapter,
    settings: EagleDraftRunnerSettings,
) -> None:
    if not case.forward_mode.is_draft_extend(include_v2=True):
        raise ValueError(
            "EAGLE draft-extend CUDA graph runner coverage expects DRAFT_EXTEND "
            "or DRAFT_EXTEND_V2 cases."
        )
    if case.batch_size > settings.capture_batch_size:
        raise ValueError("Capture batch size must cover the replay batch size.")
    if max(case.input_lens) > settings.speculative_num_steps + 1:
        raise ValueError("Accepted-token count exceeds the configured draft length.")
    adapter.check_case(case, settings)


def _run_eagle_draft_extend_eager(
    worker: _EagleDraftExtendWorkerHarness,
    batch: ForwardBatch,
    settings: EagleDraftRunnerSettings,
):
    model_runner = (
        worker.model_runner if hasattr(worker, "model_runner") else worker.draft_runner
    )
    with torch.no_grad(), forward_context(
        ForwardContext(attn_backend=worker.draft_extend_attn_backend)
    ):
        worker.draft_extend_attn_backend.init_forward_metadata(batch)
        ret = model_runner.model.forward(
            batch.input_ids,
            batch.positions,
            batch,
        )
    probs = torch.softmax(ret.next_token_logits, dim=-1)
    ret.topk_p, ret.topk_index = fast_topk(probs, settings.topk, dim=-1)
    return ret


def run_eagle_draft_extend_cuda_graph_runner_case(
    testcase,
    case,
    *,
    adapter: EagleDraftExtendCudaGraphRunnerAdapter,
    build_kwargs: dict,
    settings: EagleDraftRunnerSettings,
):
    try:
        _check_eagle_draft_extend_cuda_graph_runner_case(
            case,
            adapter=adapter,
            settings=settings,
        )
        draft_inputs = adapter.make_draft_inputs(case, settings)

        eager_fixture, eager_worker, _ = _build_eagle_draft_extend_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(eager_fixture, case, draft_inputs, settings)
        eager_batch = adapter.make_forward_batch(
            eager_fixture,
            case,
            draft_inputs,
            settings,
        )
        expected = _run_eagle_draft_extend_eager(eager_worker, eager_batch, settings)

        graph_fixture, graph_worker, graph_backend = _build_eagle_draft_extend_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        graph_batch = adapter.make_forward_batch(
            graph_fixture,
            case,
            draft_inputs,
            settings,
        )
        graph_runner = _capture_eagle_draft_extend_graph_runner(
            graph_worker,
            graph_backend,
            settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)

        testcase.assertTrue(graph_runner.can_run(graph_batch))
        actual = graph_runner.replay(graph_batch)
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

    def __call__(self, forward_batch: ForwardBatch, *, skip_attn_backend_init: bool):
        del skip_attn_backend_init
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


class _EagleDraftExtendForward(nn.Module):
    def __init__(
        self,
        *,
        module,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.module = module
        self.token_embed = nn.Embedding(
            vocab_size, hidden_size, dtype=dtype, device=device
        )
        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=dtype, device=device
        )

    def _select_logits_positions(self, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.forward_mode.is_draft_extend_v2():
            return torch.arange(
                forward_batch.input_ids.shape[0],
                dtype=torch.int64,
                device=forward_batch.input_ids.device,
            )

        extend_lens = forward_batch.extend_seq_lens.to(torch.int64)
        starts = torch.zeros_like(extend_lens)
        if extend_lens.numel() > 1:
            starts[1:] = torch.cumsum(extend_lens[:-1], dim=0)
        return starts + forward_batch.spec_info.num_accept_tokens.to(torch.int64) - 1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        del positions
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states
        if hidden_states is None:
            raise ValueError("EAGLE draft-extend runner tests expect hidden states.")

        hidden_states = hidden_states + self.token_embed(input_ids)
        hidden_states = self.module(hidden_states, forward_batch)
        logits = self.lm_head(hidden_states).float()
        select_index = self._select_logits_positions(forward_batch)
        return LogitsProcessorOutput(
            next_token_logits=logits[select_index],
            hidden_states=hidden_states[select_index],
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

    def __call__(self, forward_batch: ForwardBatch, *, skip_attn_backend_init: bool):
        del skip_attn_backend_init
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


def _make_dense_draft_extend_model_forward(
    fixture,
    settings: EagleDraftRunnerSettings,
):
    return _EagleDraftExtendForward(
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
    return {
        "hidden_states": draft_inputs["hidden_states"],
        "topk_p": draft_inputs["topk_p"],
        "topk_index": draft_inputs["topk_index"],
    }


def _make_dense_draft_extend_inputs(
    case: DenseAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    with _seeded_rng(6080 + len(case.name), device=settings.device):
        return {
            "hidden_states": torch.randn(
                case.num_input_tokens,
                settings.hidden_size,
                dtype=settings.dtype,
                device=settings.device,
            ),
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


def _prepare_dense_draft_extend_replay_state(
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


def _make_eagle_draft_extend_input(
    case,
    batch: ForwardBatch,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> EagleDraftExtendInput:
    num_accept_tokens = torch.tensor(
        case.input_lens,
        dtype=torch.int32,
        device=settings.device,
    )
    num_tokens_per_req = settings.speculative_num_steps + 1
    spec_info = EagleDraftExtendInput(
        hidden_states=draft_inputs["hidden_states"].clone(),
        num_correct_drafts=num_accept_tokens - 1,
        num_accept_tokens=num_accept_tokens,
        num_accept_tokens_cpu=list(case.input_lens),
        input_ids=batch.input_ids,
        seq_lens=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        req_pool_indices=batch.req_pool_indices,
        positions=batch.positions,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=num_tokens_per_req,
        num_tokens_for_logprob_per_req=(
            num_tokens_per_req if case.forward_mode.is_draft_extend_v2() else 1
        ),
    )
    if case.forward_mode.is_draft_extend_v2():
        spec_info.extend_seq_lens_tensor = batch.extend_seq_lens
        spec_info.extend_seq_lens_cpu = list(case.input_lens)
    return spec_info


def _set_draft_extend_v2_prefix_lens(
    batch: ForwardBatch,
    case,
    *,
    device: str,
) -> None:
    batch.seq_lens = torch.tensor(case.prefix_lens, dtype=torch.int32, device=device)
    batch.seq_lens_cpu = torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_sum = sum(case.prefix_lens)


def _make_dense_eagle_draft_extend_forward_batch(
    fixture,
    case: DenseAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    batch = _make_dense_forward_batch(
        case,
        fixture.runner,
        max_context_len=settings.max_context_len,
        device=settings.device,
    )
    batch.spec_info = _make_eagle_draft_extend_input(
        case,
        batch,
        draft_inputs,
        settings,
    )
    return batch


def _make_dense_eagle_draft_extend_v2_forward_batch(
    fixture,
    case: DenseAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    batch = _make_dense_forward_batch(
        case,
        fixture.runner,
        max_context_len=settings.max_context_len,
        device=settings.device,
    )
    _set_draft_extend_v2_prefix_lens(batch, case, device=settings.device)
    batch.spec_info = _make_eagle_draft_extend_input(
        case,
        batch,
        draft_inputs,
        settings,
    )
    return batch


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


def run_dense_eagle_draft_extend_cuda_graph_runner_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int = 4,
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
    adapter = EagleDraftExtendCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_draft_extend_model_forward,
        make_draft_inputs=_make_dense_draft_extend_inputs,
        prepare_replay_state=_prepare_dense_draft_extend_replay_state,
        make_forward_batch=_make_dense_eagle_draft_extend_forward_batch,
    )
    run_eagle_draft_extend_cuda_graph_runner_case(
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


def run_dense_eagle_draft_extend_v2_cuda_graph_runner_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int | None = None,
    speculative_num_draft_tokens: int | None = None,
    cuda_graph_capture_batch_size: int = 4,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    if not case.forward_mode.is_draft_extend_v2():
        raise ValueError(
            "EAGLE draft-extend-v2 CUDA graph runner coverage expects "
            "DRAFT_EXTEND_V2 cases."
        )
    if len(set(case.input_lens)) != 1:
        raise ValueError("DRAFT_EXTEND_V2 runner coverage uses fixed token counts.")

    num_tokens_per_req = case.input_lens[0]
    if speculative_num_steps is None:
        speculative_num_steps = num_tokens_per_req - 1
    if speculative_num_draft_tokens is None:
        speculative_num_draft_tokens = num_tokens_per_req

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
    adapter = EagleDraftExtendCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_draft_extend_model_forward,
        make_draft_inputs=_make_dense_draft_extend_inputs,
        prepare_replay_state=_prepare_dense_draft_extend_replay_state,
        make_forward_batch=_make_dense_eagle_draft_extend_v2_forward_batch,
    )
    run_eagle_draft_extend_cuda_graph_runner_case(
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

    def __call__(self, forward_batch: ForwardBatch, *, skip_attn_backend_init: bool):
        del skip_attn_backend_init
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


def _make_mla_draft_extend_model_forward(
    fixture,
    settings: EagleDraftRunnerSettings,
):
    return _EagleDraftExtendForward(
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


def _make_mla_draft_extend_inputs(
    case: MLAAttentionCase,
    settings: EagleDraftRunnerSettings,
) -> dict[str, torch.Tensor]:
    with _seeded_rng(7080 + len(case.name), device=settings.device):
        return {
            "hidden_states": torch.randn(
                case.num_input_tokens,
                settings.hidden_size,
                dtype=settings.dtype,
                device=settings.device,
            ),
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


def _prepare_mla_draft_extend_replay_state(
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


def _make_mla_eagle_draft_extend_forward_batch(
    fixture,
    case: MLAAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    batch = _make_mla_forward_batch(
        case,
        fixture.runner,
        max_context_len=settings.max_context_len,
        device=settings.device,
    )
    batch.spec_info = _make_eagle_draft_extend_input(
        case,
        batch,
        draft_inputs,
        settings,
    )
    return batch


def _make_mla_eagle_draft_extend_v2_forward_batch(
    fixture,
    case: MLAAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    settings: EagleDraftRunnerSettings,
) -> ForwardBatch:
    batch = _make_mla_forward_batch(
        case,
        fixture.runner,
        max_context_len=settings.max_context_len,
        device=settings.device,
    )
    _set_draft_extend_v2_prefix_lens(batch, case, device=settings.device)
    batch.spec_info = _make_eagle_draft_extend_input(
        case,
        batch,
        draft_inputs,
        settings,
    )
    return batch


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


def run_mla_eagle_draft_extend_cuda_graph_runner_case(
    testcase,
    case: MLAAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int = 4,
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
    adapter = EagleDraftExtendCudaGraphRunnerAdapter(
        build_fixture=build_mla_attention_fixture,
        make_model_forward=_make_mla_draft_extend_model_forward,
        make_draft_inputs=_make_mla_draft_extend_inputs,
        prepare_replay_state=_prepare_mla_draft_extend_replay_state,
        make_forward_batch=_make_mla_eagle_draft_extend_forward_batch,
    )
    run_eagle_draft_extend_cuda_graph_runner_case(
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


def run_mla_eagle_draft_extend_v2_cuda_graph_runner_case(
    testcase,
    case: MLAAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int | None = None,
    speculative_num_draft_tokens: int | None = None,
    cuda_graph_capture_batch_size: int = 4,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
):
    if not case.forward_mode.is_draft_extend_v2():
        raise ValueError(
            "EAGLE draft-extend-v2 CUDA graph runner coverage expects "
            "DRAFT_EXTEND_V2 cases."
        )
    if len(set(case.input_lens)) != 1:
        raise ValueError("DRAFT_EXTEND_V2 runner coverage uses fixed token counts.")

    num_tokens_per_req = case.input_lens[0]
    if speculative_num_steps is None:
        speculative_num_steps = num_tokens_per_req - 1
    if speculative_num_draft_tokens is None:
        speculative_num_draft_tokens = num_tokens_per_req

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
    adapter = EagleDraftExtendCudaGraphRunnerAdapter(
        build_fixture=build_mla_attention_fixture,
        make_model_forward=_make_mla_draft_extend_model_forward,
        make_draft_inputs=_make_mla_draft_extend_inputs,
        prepare_replay_state=_prepare_mla_draft_extend_replay_state,
        make_forward_batch=_make_mla_eagle_draft_extend_v2_forward_batch,
    )
    run_eagle_draft_extend_cuda_graph_runner_case(
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
