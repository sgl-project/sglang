# --- Imports added for the EAGLEDraftExtendCudaGraphRunner production
# --- runner integration that lives below.
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import fast_topk

from ..attention_methods.dense_attention import DEFAULT_DEVICE
from ..attention_methods.dense_attention import DEFAULT_DEVICE as DENSE_DEFAULT_DEVICE
from ..attention_methods.dense_attention import DEFAULT_DTYPE
from ..attention_methods.dense_attention import DEFAULT_DTYPE as DENSE_DEFAULT_DTYPE
from ..attention_methods.dense_attention import (
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
)
from ..attention_methods.dense_attention import DEFAULT_MAX_CONTEXT_LEN
from ..attention_methods.dense_attention import (
    DEFAULT_MAX_CONTEXT_LEN as DENSE_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
)
from ..attention_methods.dense_attention import (
    _make_forward_batch as _make_dense_forward_batch,
)
from ..attention_methods.dense_attention import (
    build_dense_attention_fixture,
    dense_fixture_inputs,
    expected_dense_output_from_inputs,
    make_dense_padded_replay_inputs,
    make_dense_random_inputs,
    prepare_dense_runner_inputs,
    run_dense_forward,
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
from ..attention_methods.mla_attention import (
    build_mla_attention_fixture,
    expected_mla_output_from_inputs,
    make_mla_case_with_prefix_lens,
    make_mla_padded_replay_inputs,
    make_mla_random_inputs,
    mla_fixture_inputs,
    prepare_mla_runner_inputs,
    run_mla_forward,
)
from .speculative_cuda_graph_runner import (
    SpeculativeCudaGraphAdapter,
    run_speculative_cuda_graph_case,
)
from .speculative_draft_runner import (
    EagleDraftRunnerSettings,
    _configure_runner_for_eagle_draft,
    _reset_cuda_graph_test_buffers,
    _seeded_rng,
    _single_rank_graph_capture,
)


def _make_dense_spec_case_with_lens(
    case: DenseAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
    input_lens: tuple[int, ...],
) -> DenseAttentionCase:
    return DenseAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=input_lens,
        sliding_window_size=case.sliding_window_size,
    )


def _make_eagle_draft_extend_input(case, batch, *, device: str):
    num_accept_tokens = torch.tensor(
        case.input_lens,
        dtype=torch.int32,
        device=device,
    )
    return EagleDraftExtendInput(
        hidden_states=None,
        num_correct_drafts=num_accept_tokens - 1,
        num_accept_tokens=num_accept_tokens,
        num_accept_tokens_cpu=list(case.input_lens),
        input_ids=batch.input_ids,
        seq_lens=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        req_pool_indices=batch.req_pool_indices,
        positions=batch.positions,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=max(case.input_lens),
        num_tokens_for_logprob_per_req=1,
    )


def _make_eagle_draft_extend_v2_input(case, batch, *, device: str):
    draft_extend_input = _make_eagle_draft_extend_input(case, batch, device=device)
    draft_extend_input.extend_seq_lens_tensor = torch.tensor(
        case.input_lens,
        dtype=torch.int32,
        device=device,
    )
    draft_extend_input.extend_seq_lens_cpu = list(case.input_lens)
    return draft_extend_input


def _set_draft_extend_v2_prefix_lens(batch, case, *, device: str):
    # Production sets seq_lens = prefix + extend before init_forward_metadata
    # (eagle_info_v2.py bumps seq_lens by num_draft_tokens). Match that here.
    seq_lens = tuple(p + e for p, e in zip(case.prefix_lens, case.input_lens))
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_sum = sum(seq_lens)


def _prepare_eagle_draft_extend_v2_batch(case, batch, *, device: str) -> None:
    _set_draft_extend_v2_prefix_lens(batch, case, device=device)
    batch.spec_info = _make_eagle_draft_extend_v2_input(
        case,
        batch,
        device=device,
    )


def _run_draft_extend_cuda_graph_case(
    testcase,
    case,
    *,
    build_fixture,
    make_capture_case,
    make_replay_case,
    make_forward_batch,
    fixture_inputs,
    make_capture_inputs,
    make_replay_inputs,
    prepare_batch,
    prepare_inputs,
    run_forward,
    expected_output,
    build_kwargs: dict,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
    capture_batch_size: int,
    atol: float,
    rtol: float,
    max_num_tokens=None,
    run_graph_eager: bool = True,
    compare_replay_to_graph_eager: bool = True,
    pad_style: str = "small_real",
    pad_num_tokens_per_bs: int | None = None,
):
    adapter = SpeculativeCudaGraphAdapter(
        build_fixture=build_fixture,
        make_capture_case=make_capture_case,
        make_replay_case=make_replay_case,
        make_forward_batch=make_forward_batch,
        fixture_inputs=fixture_inputs,
        make_capture_inputs=make_capture_inputs,
        make_replay_inputs=make_replay_inputs,
        prepare_batch=prepare_batch,
        prepare_inputs=prepare_inputs,
        run_forward=run_forward,
        expected_output=lambda fixture, draft_case, inputs, _state: expected_output(
            fixture,
            draft_case,
            inputs,
            None,
        ),
        max_num_tokens=max_num_tokens,
        run_graph_eager=run_graph_eager,
        compare_replay_to_graph_eager=compare_replay_to_graph_eager,
        atol=atol,
        rtol=rtol,
        pad_style=pad_style,
        pad_num_tokens_per_bs=pad_num_tokens_per_bs,
    )
    run_speculative_cuda_graph_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=build_kwargs,
        capture_batch_size=capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def run_dense_draft_extend_v2_cuda_graph_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
    pad_style: str = "small_real",
):
    if not case.forward_mode.is_draft_extend_v2():
        raise ValueError("Draft-extend-v2 CUDA graph coverage expects DRAFT_EXTEND_V2.")
    if len(set(case.input_lens)) != 1:
        raise ValueError(
            "Draft-extend-v2 CUDA graph coverage uses a fixed token count per request."
        )

    num_tokens_per_req = case.input_lens[0]
    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_dense_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: (
            _make_dense_spec_case_with_lens(
                base,
                name,
                (prefix_len,) * bs,
                (num_tokens_per_req,) * bs,
            )
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: (
            _make_dense_spec_case_with_lens(
                base,
                name,
                base.prefix_lens + pad_prefix_lens,
                base.input_lens + (num_tokens_per_req,) * len(pad_prefix_lens),
            )
        ),
        make_forward_batch=_make_dense_forward_batch,
        fixture_inputs=dense_fixture_inputs,
        make_capture_inputs=make_dense_random_inputs,
        make_replay_inputs=make_dense_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_eagle_draft_extend_v2_batch(
            draft_case,
            batch,
            device=device,
        ),
        prepare_inputs=prepare_dense_runner_inputs,
        run_forward=run_dense_forward,
        expected_output=expected_dense_output_from_inputs,
        build_kwargs=dict(
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
        run_graph_eager=False,
        compare_replay_to_graph_eager=False,
        pad_style=pad_style,
        pad_num_tokens_per_bs=num_tokens_per_req,
    )


def run_mla_draft_extend_v2_cuda_graph_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
    pad_style: str = "small_real",
):
    if not case.forward_mode.is_draft_extend_v2():
        raise ValueError("Draft-extend-v2 CUDA graph coverage expects DRAFT_EXTEND_V2.")
    if len(set(case.input_lens)) != 1:
        raise ValueError(
            "Draft-extend-v2 CUDA graph coverage uses a fixed token count per request."
        )
    num_tokens_per_req = case.input_lens[0]

    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_mla_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: (
            make_mla_case_with_prefix_lens(base, name, (prefix_len,) * bs)
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: (
            make_mla_case_with_prefix_lens(
                base,
                name,
                base.prefix_lens + pad_prefix_lens,
            )
        ),
        make_forward_batch=_make_mla_forward_batch,
        fixture_inputs=mla_fixture_inputs,
        make_capture_inputs=make_mla_random_inputs,
        make_replay_inputs=make_mla_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_eagle_draft_extend_v2_batch(
            draft_case,
            batch,
            device=device,
        ),
        prepare_inputs=prepare_mla_runner_inputs,
        run_forward=run_mla_forward,
        expected_output=expected_mla_output_from_inputs,
        build_kwargs=dict(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
        run_graph_eager=False,
        compare_replay_to_graph_eager=False,
        pad_style=pad_style,
        pad_num_tokens_per_bs=num_tokens_per_req,
    )


# ---------------------------------------------------------------------------
# Production EAGLE draft-extend CUDA-graph runner integration
# ---------------------------------------------------------------------------
#
# The shared `EagleDraftExtendCudaGraphRunnerAdapter` lifecycle wires
# `EAGLEDraftExtendCudaGraphRunner` / `EAGLEDraftExtendV2CudaGraphRunner`
# against per-backend fixtures. Per-backend wrappers below provide the
# method-specific callbacks (model forward, draft-input synthesis,
# replay-state setup, forward-batch factory, layout check).
#
# Decode-side production-runner code (EAGLEDraftCudaGraphRunner /
# FrozenKVMTPCudaGraphRunner) lives in `speculative_draft_runner.py`.
# Shared infrastructure (`EagleDraftRunnerSettings`, `_DummyTpGroup`,
# `_TinyDraftModel`, `_seeded_rng`, `_configure_runner_for_eagle_draft`,
# etc.) is imported from there.


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


def _assert_draft_extend_v2_outputs_close(actual, expected, settings) -> None:
    # DRAFT_EXTEND_V2 graph runner only anchors the full-row
    # `next_token_logits` / `hidden_states`; the selected-row `topk_p` /
    # `topk_index` are owned by EAGLEWorkerV2 and computed *after* replay (see
    # `eagle_worker_v2._draft_extend_for_decode` and the early-return in
    # `EAGLEDraftExtendCudaGraphRunner.replay` for DRAFT_EXTEND_V2). The V2
    # production runner output therefore carries no topk fields, so the
    # runner-mode reference must only compare what the graph actually anchors.
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


def _capture_eagle_draft_extend_graph_runner(
    worker: _EagleDraftExtendWorkerHarness,
    draft_extend_attn_backend,
    settings: EagleDraftRunnerSettings,
) -> EAGLEDraftExtendCudaGraphRunner:
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
        patch(
            "sglang.srt.model_executor.runner.base_cuda_graph_runner.get_attention_cp_size",
            lambda: 1,
        ),
    ):
        _reset_cuda_graph_test_buffers()
        return EAGLEDraftExtendCudaGraphRunner(
            worker,
            draft_extend_attn_backend=draft_extend_attn_backend,
            speculative_num_steps=settings.speculative_num_steps,
        )


def _check_eagle_draft_extend_cuda_graph_runner_case(
    case,
    *,
    adapter: EagleDraftExtendCudaGraphRunnerAdapter,
    settings: EagleDraftRunnerSettings,
) -> None:
    if not case.forward_mode.is_draft_extend_v2():
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
    # Mirror the production fast path from
    # EAGLEDraftExtendCudaGraphRunner.replay (#26397): when topk == 1
    # production skips the full-vocab softmax and returns
    # `topk_p = ones_like(topk_index)` (the value is unused downstream).
    # The eager reference must match this for assert_outputs_close.
    from sglang.srt.utils import is_hip

    if settings.topk == 1 and not is_hip():
        ret.topk_index = torch.argmax(ret.next_token_logits, dim=-1, keepdim=True)
        ret.topk_p = torch.ones_like(ret.topk_index, dtype=torch.float32)
    else:
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

        testcase.assertTrue(graph_runner.can_run_graph(graph_batch))
        actual = graph_runner.execute(graph_batch)
        adapter.assert_outputs_close(actual, expected, settings)
    finally:
        _reset_cuda_graph_test_buffers()


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


def _make_eagle_draft_extend_input_for_production_runner(
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
    # Production sets seq_lens = prefix + extend before init_forward_metadata
    # (eagle_info_v2.py bumps seq_lens by num_draft_tokens). Match that here.
    seq_lens = tuple(p + e for p, e in zip(case.prefix_lens, case.input_lens))
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_sum = sum(seq_lens)


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
    batch.spec_info = _make_eagle_draft_extend_input_for_production_runner(
        case,
        batch,
        draft_inputs,
        settings,
    )
    return batch


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
        assert_outputs_close=_assert_draft_extend_v2_outputs_close,
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
    batch.spec_info = _make_eagle_draft_extend_input_for_production_runner(
        case,
        batch,
        draft_inputs,
        settings,
    )
    return batch


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
        assert_outputs_close=_assert_draft_extend_v2_outputs_close,
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
