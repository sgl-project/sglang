from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

from ..attention_methods.dense_attention import DEFAULT_DEVICE as DENSE_DEFAULT_DEVICE
from ..attention_methods.dense_attention import DEFAULT_DTYPE as DENSE_DEFAULT_DTYPE
from ..attention_methods.dense_attention import (
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
)
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
    make_dense_case_with_prefix_lens,
    make_dense_padded_replay_inputs,
    make_dense_random_inputs,
    prepare_dense_runner_inputs,
    run_dense_fixture_eager,
    run_dense_forward,
)
from ..attention_methods.gdn_attention import DEFAULT_DEVICE as GDN_DEFAULT_DEVICE
from ..attention_methods.gdn_attention import DEFAULT_DTYPE as GDN_DEFAULT_DTYPE
from ..attention_methods.gdn_attention import (
    DEFAULT_HEAD_K_DIM,
    DEFAULT_HEAD_V_DIM,
)
from ..attention_methods.gdn_attention import (
    DEFAULT_MAX_CONTEXT_LEN as GDN_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.gdn_attention import (
    GDN_ATOL,
    GDN_RTOL,
    GDNAttentionCase,
    _clone_gdn_cache,
)
from ..attention_methods.gdn_attention import (
    _make_forward_batch as _make_gdn_forward_batch,
)
from ..attention_methods.gdn_attention import (
    _restore_gdn_cache,
    build_gdn_attention_fixture,
    expected_gdn_output_from_inputs,
    gdn_fixture_inputs,
    make_gdn_case_with_prefix_lens,
    make_gdn_random_inputs,
    make_gdn_replay_inputs,
    prepare_gdn_runner_inputs,
    run_gdn_fixture_eager,
    run_gdn_forward,
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
    run_mla_fixture_eager,
    run_mla_forward,
)

DENSE_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4
MLA_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4
GDN_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 3


@dataclass(frozen=True)
class CudaGraphDecodeAdapter:
    build_fixture: Callable[..., Any]
    make_case: Callable[[Any, str, tuple[int, ...]], Any]
    make_forward_batch: Callable[..., Any]
    fixture_inputs: Callable[[Any], dict[str, Any]]
    make_capture_inputs: Callable[..., dict[str, Any]]
    make_replay_inputs: Callable[..., dict[str, Any]]
    prepare_inputs: Callable[..., None]
    run_eager: Callable[[Any], torch.Tensor]
    run_forward: Callable[[Any, Any, dict[str, Any]], torch.Tensor]
    expected_output: Callable[[Any, Any, dict[str, Any], Any], torch.Tensor]
    clone_state: Callable[[Any], Any] = lambda _: None
    restore_state: Callable[[Any, Any], None] = lambda _fixture, _state: None
    allow_padding: bool = True
    atol: float = 0.0
    rtol: float = 0.0


def _check_decode_cuda_graph_case(case, capture_batch_size: int, *, allow_padding=True):
    if not case.forward_mode.is_decode():
        raise ValueError(
            "CUDA graph runner integration currently expects decode cases."
        )
    if allow_padding:
        if case.batch_size > capture_batch_size:
            raise ValueError(
                "CUDA graph capture batch size must be at least the replay batch size."
            )
    elif case.batch_size != capture_batch_size:
        raise ValueError(
            "This CUDA graph coverage uses an unpadded replay batch; choose a case "
            "whose batch size matches the capture batch size."
        )


def _init_cuda_graph_capture_metadata(backend, capture_batch_size: int, batch):
    backend.init_cuda_graph_state(
        max_bs=capture_batch_size,
        max_num_tokens=batch.input_ids.numel(),
    )
    backend.init_forward_metadata_capture_cuda_graph(
        bs=capture_batch_size,
        num_tokens=batch.input_ids.numel(),
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        encoder_lens=batch.encoder_lens,
        forward_mode=batch.forward_mode,
        spec_info=batch.spec_info,
    )


def _init_cuda_graph_replay_metadata(backend, capture_batch_size: int, batch):
    backend.init_forward_metadata_replay_cuda_graph(
        bs=capture_batch_size,
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        seq_lens_sum=batch.seq_lens_sum,
        encoder_lens=batch.encoder_lens,
        forward_mode=batch.forward_mode,
        spec_info=batch.spec_info,
        seq_lens_cpu=batch.seq_lens_cpu,
    )


def _run_cuda_graph_decode_case(
    testcase,
    case,
    *,
    adapter: CudaGraphDecodeAdapter,
    build_kwargs: dict,
    capture_batch_size: int,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
):
    _check_decode_cuda_graph_case(
        case,
        capture_batch_size,
        allow_padding=adapter.allow_padding,
    )

    eager_fixture = adapter.build_fixture(testcase, case, **build_kwargs)
    eager_inputs = adapter.fixture_inputs(eager_fixture)
    eager_initial_state = adapter.clone_state(eager_fixture)
    eager_actual = adapter.run_eager(eager_fixture)
    eager_expected = adapter.expected_output(
        eager_fixture,
        case,
        eager_inputs,
        eager_initial_state,
    )
    torch.testing.assert_close(
        eager_actual,
        eager_expected,
        atol=adapter.atol,
        rtol=adapter.rtol,
    )

    graph_fixture = adapter.build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=capture_batch_size,
    )
    backend = graph_fixture.backend
    graph_replay_inputs = adapter.fixture_inputs(graph_fixture)
    graph_initial_state = adapter.clone_state(graph_fixture)
    capture_prefix_len = max(0, backend.get_cuda_graph_seq_len_fill_value() - 1)

    capture_case = adapter.make_case(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * capture_batch_size,
    )
    capture_inputs = adapter.make_capture_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = adapter.make_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    adapter.prepare_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(backend, capture_batch_size, capture_batch)
        capture_actual = adapter.run_forward(
            graph_fixture,
            capture_batch,
            capture_inputs,
        )
        backend.on_after_cuda_graph_warmup()
        capture_expected = adapter.expected_output(
            graph_fixture,
            capture_case,
            capture_inputs,
            graph_initial_state,
        )

        adapter.restore_state(graph_fixture, graph_initial_state)
        replay_pad_prefix_lens = (capture_prefix_len,) * (
            capture_batch_size - case.batch_size
        )
        replay_case = adapter.make_case(
            case,
            f"{case.name}_cuda_graph_replay",
            case.prefix_lens + replay_pad_prefix_lens,
        )
        replay_inputs = adapter.make_replay_inputs(
            replay_case,
            graph_fixture,
            replay_pad_prefix_lens,
            graph_replay_inputs,
            dtype=dtype,
            device=device,
        )
        replay_batch = adapter.make_forward_batch(
            replay_case,
            graph_fixture.runner,
            max_context_len=max_context_len,
            device=device,
        )
        adapter.prepare_inputs(
            graph_fixture,
            replay_case,
            replay_batch,
            replay_inputs,
            max_context_len=max_context_len,
        )
        _init_cuda_graph_replay_metadata(backend, capture_batch_size, replay_batch)
        replay_actual = adapter.run_forward(
            graph_fixture,
            replay_batch,
            replay_inputs,
        )

    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=adapter.atol,
        rtol=adapter.rtol,
    )
    replay_expected = adapter.expected_output(
        graph_fixture,
        replay_case,
        replay_inputs,
        graph_initial_state,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=adapter.atol,
        rtol=adapter.rtol,
    )
    torch.testing.assert_close(
        replay_actual[: case.num_input_tokens],
        eager_actual,
        atol=adapter.atol,
        rtol=adapter.rtol,
    )


def run_dense_cuda_graph_decode_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = DENSE_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_dense_attention_fixture,
        make_case=make_dense_case_with_prefix_lens,
        make_forward_batch=_make_dense_forward_batch,
        fixture_inputs=dense_fixture_inputs,
        make_capture_inputs=make_dense_random_inputs,
        make_replay_inputs=make_dense_padded_replay_inputs,
        prepare_inputs=prepare_dense_runner_inputs,
        run_eager=run_dense_fixture_eager,
        run_forward=run_dense_forward,
        expected_output=expected_dense_output_from_inputs,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    _run_cuda_graph_decode_case(
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
        capture_batch_size=cuda_graph_capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def run_mla_cuda_graph_decode_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = MLA_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_mla_attention_fixture,
        make_case=make_mla_case_with_prefix_lens,
        make_forward_batch=_make_mla_forward_batch,
        fixture_inputs=mla_fixture_inputs,
        make_capture_inputs=make_mla_random_inputs,
        make_replay_inputs=make_mla_padded_replay_inputs,
        prepare_inputs=prepare_mla_runner_inputs,
        run_eager=run_mla_fixture_eager,
        run_forward=run_mla_forward,
        expected_output=expected_mla_output_from_inputs,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
    _run_cuda_graph_decode_case(
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
        capture_batch_size=cuda_graph_capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def run_gdn_cuda_graph_decode_case(
    testcase,
    case: GDNAttentionCase,
    *,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
    max_context_len: int = GDN_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = GDN_DEFAULT_DTYPE,
    device: str = GDN_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = GDN_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_gdn_attention_fixture,
        make_case=make_gdn_case_with_prefix_lens,
        make_forward_batch=_make_gdn_forward_batch,
        fixture_inputs=gdn_fixture_inputs,
        make_capture_inputs=make_gdn_random_inputs,
        make_replay_inputs=make_gdn_replay_inputs,
        prepare_inputs=prepare_gdn_runner_inputs,
        run_eager=run_gdn_fixture_eager,
        run_forward=run_gdn_forward,
        expected_output=expected_gdn_output_from_inputs,
        clone_state=_clone_gdn_cache,
        restore_state=_restore_gdn_cache,
        allow_padding=False,
        atol=GDN_ATOL,
        rtol=GDN_RTOL,
    )
    _run_cuda_graph_decode_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        capture_batch_size=cuda_graph_capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
