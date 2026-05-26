from contextlib import contextmanager
from types import MethodType, SimpleNamespace
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
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker import EAGLEWorker
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


class _DummyTpGroup:
    ca_comm = None

    def barrier(self) -> None:
        return None


class _TinyDraftModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("EAGLEDraftCudaGraphRunner should call worker.draft_forward")


class _TinyEagleDraftForward:
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


class _TinyEagleWorker:
    def __init__(
        self,
        *,
        fixture,
        draft_attn_backend,
        topk: int,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.model_runner = fixture.runner
        self.draft_attn_backend = draft_attn_backend
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.server_args = fixture.runner.server_args
        self.model_config = fixture.runner.model_config
        self.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        self.hot_token_id = None
        self.model_runner.forward = _TinyEagleDraftForward(
            module=fixture.actual_module,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype=dtype,
            device=device,
        )
        self.draft_forward = MethodType(EAGLEWorker.draft_forward, self)

    @property
    def draft_model_runner(self):
        return self.model_runner


@contextmanager
def _single_rank_graph_capture():
    stream = torch.cuda.Stream()
    yield SimpleNamespace(stream=stream)


def _configure_runner_for_eagle_draft(
    runner,
    *,
    backend: str,
    topk: int,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    capture_batch_size: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    server_args = runner.server_args
    updates = {
        "attention_backend": backend,
        "cuda_graph_bs": [capture_batch_size],
        "debug_cuda_graph": False,
        "decode_attention_backend": backend,
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
        "prefill_attention_backend": backend,
        "speculative_algorithm": "EAGLE",
        "speculative_attention_mode": "decode",
        "speculative_draft_attention_backend": None,
        "speculative_eagle_topk": topk,
        "speculative_num_draft_tokens": speculative_num_draft_tokens,
        "speculative_num_steps": speculative_num_steps,
        "torch_compile_max_bs": 0,
        "use_mla_backend": False,
    }
    for key, value in updates.items():
        setattr(server_args, key, value)

    runner.spec_algorithm = SpeculativeAlgorithm.EAGLE
    runner.is_draft_worker = True
    runner.model = _TinyDraftModel()
    runner.tp_group = _DummyTpGroup()
    runner.device_timer = None
    runner.model_config.spec_hidden_size = hidden_size
    runner.model_config.dtype = runner.dtype
    runner.model_config.vocab_size = vocab_size
    runner.model_config.hf_config.vocab_size = vocab_size
    set_global_server_args_for_scheduler(server_args)


def _build_eagle_draft_fixture(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    capture_batch_size: int,
    head_dim: int,
    hidden_size: int,
    max_context_len: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: str,
):
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=capture_batch_size,
    )
    _configure_runner_for_eagle_draft(
        fixture.runner,
        backend=case.backend,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    draft_attn_backend = DraftBackendFactory(
        fixture.runner.server_args,
        fixture.runner,
        topk,
        speculative_num_steps,
    ).create_decode_backend()
    fixture.runner.draft_attn_backend = draft_attn_backend
    worker = _TinyEagleWorker(
        fixture=fixture,
        draft_attn_backend=draft_attn_backend,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
    )
    return fixture, worker, draft_attn_backend


def _make_draft_inputs(
    case: DenseAttentionCase,
    *,
    topk: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(4080 + len(case.name) + topk)
    torch.cuda.manual_seed_all(4080 + len(case.name) + topk)
    topk_index = (
        torch.arange(case.batch_size * topk, dtype=torch.int64, device=device).view(
            case.batch_size, topk
        )
        + 3
    ) % vocab_size
    topk_p = torch.linspace(
        0.6,
        0.9,
        steps=case.batch_size * topk,
        dtype=torch.float32,
        device=device,
    ).view(case.batch_size, topk)
    topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
    return {
        "hidden_states": torch.randn(
            case.batch_size,
            hidden_size,
            dtype=dtype,
            device=device,
        ),
        "topk_p": topk_p,
        "topk_index": topk_index,
    }


def _prepare_draft_prefix_state(
    fixture,
    case: DenseAttentionCase,
    *,
    topk: int,
    speculative_num_steps: int,
    max_context_len: int,
) -> None:
    prepare_dense_runner_inputs(
        fixture,
        case,
        fixture.forward_batch,
        {"prefix_hidden": fixture.prefix_hidden},
        max_context_len=max_context_len,
    )
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for branch in range(topk):
            for step in range(speculative_num_steps):
                position = prefix_len + branch * speculative_num_steps + step
                fixture.runner.req_to_token_pool.req_to_token[
                    req_idx,
                    position,
                ] = _dense_token_loc(
                    req_idx,
                    position,
                    page_size=case.page_size,
                    max_context_len=max_context_len,
                )


def _draft_cache_position(
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


def _check_draft_cache_layout(
    case: DenseAttentionCase,
    *,
    topk: int,
    speculative_num_steps: int,
    max_context_len: int,
) -> None:
    if topk > 1 and case.page_size != 1:
        raise ValueError(
            "The initial EAGLE draft runner fixture covers tree draft with "
            "page_size=1, where branch cache slots are laid out linearly."
        )
    if topk > 1:
        for prefix_len in case.prefix_lens:
            if prefix_len + topk * speculative_num_steps > max_context_len:
                raise ValueError(
                    "Draft cache layout exceeds the configured context len."
                )


def _make_eagle_draft_forward_batch(
    case: DenseAttentionCase,
    draft_inputs: dict[str, torch.Tensor],
    *,
    topk: int,
    speculative_num_steps: int,
    device: str,
    max_context_len: int,
) -> ForwardBatch:
    out_cache_locs = []
    for req_idx, prefix_len in enumerate(case.prefix_lens):
        for branch in range(topk):
            for step in range(speculative_num_steps):
                position = _draft_cache_position(
                    prefix_len=prefix_len,
                    branch=branch,
                    step=step,
                    topk=topk,
                    speculative_num_steps=speculative_num_steps,
                )
                out_cache_locs.append(
                    _dense_token_loc(
                        req_idx,
                        position,
                        page_size=case.page_size,
                        max_context_len=max_context_len,
                    )
                )

    spec_info = EagleDraftInput(
        topk_p=draft_inputs["topk_p"].clone(),
        topk_index=draft_inputs["topk_index"].clone(),
        hidden_states=draft_inputs["hidden_states"].clone(),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=topk,
        num_tokens_for_logprob_per_req=topk,
    )
    seq_lens = torch.tensor(case.prefix_lens, dtype=torch.int32, device=device)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=case.batch_size,
        input_ids=None,
        req_pool_indices=torch.arange(
            case.batch_size, dtype=torch.int32, device=device
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(out_cache_locs, dtype=torch.int64, device=device),
        seq_lens_sum=sum(case.prefix_lens),
        positions=seq_lens.repeat_interleave(topk).to(torch.int64),
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def _run_eagle_draft_eager(
    worker: _TinyEagleWorker,
    batch: ForwardBatch,
):
    worker.draft_attn_backend.init_forward_metadata(batch)
    return worker.draft_forward(batch)


def _assert_draft_outputs_close(actual, expected, *, atol: float, rtol: float) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)


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
    if not case.forward_mode.is_decode():
        raise ValueError("EAGLE draft CUDA graph runner coverage expects DECODE cases.")
    if case.batch_size > cuda_graph_capture_batch_size:
        raise ValueError("Capture batch size must cover the replay batch size.")
    _check_draft_cache_layout(
        case,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        max_context_len=max_context_len,
    )

    draft_inputs = _make_draft_inputs(
        case,
        topk=topk,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
    )

    eager_fixture, eager_worker, _ = _build_eagle_draft_fixture(
        testcase,
        case,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
    )
    _prepare_draft_prefix_state(
        eager_fixture,
        case,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        max_context_len=max_context_len,
    )
    eager_batch = _make_eagle_draft_forward_batch(
        case,
        draft_inputs,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        device=device,
        max_context_len=max_context_len,
    )
    expected = _run_eagle_draft_eager(eager_worker, eager_batch)

    graph_fixture, graph_worker, graph_backend = _build_eagle_draft_fixture(
        testcase,
        case,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
    )
    _prepare_draft_prefix_state(
        graph_fixture,
        case,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        max_context_len=max_context_len,
    )
    graph_batch = _make_eagle_draft_forward_batch(
        case,
        draft_inputs,
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        device=device,
        max_context_len=max_context_len,
    )

    with (
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.graph_capture",
            _single_rank_graph_capture,
        ),
        patch(
            "sglang.srt.model_executor.cuda_graph_runner.get_tensor_model_parallel_rank",
            lambda: 1,
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
        set_global_graph_memory_pool(None)
        graph_runner = EAGLEDraftCudaGraphRunner(
            graph_worker,
            draft_attn_backend=graph_backend,
            speculative_num_steps=speculative_num_steps,
        )

    testcase.assertTrue(graph_runner.can_run(graph_batch))
    actual = graph_runner.replay(graph_batch)
    _assert_draft_outputs_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)
