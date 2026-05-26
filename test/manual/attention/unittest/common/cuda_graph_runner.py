import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

from .dense_attention import DEFAULT_DEVICE as DENSE_DEFAULT_DEVICE
from .dense_attention import DEFAULT_DTYPE as DENSE_DEFAULT_DTYPE
from .dense_attention import (
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
)
from .dense_attention import DEFAULT_MAX_CONTEXT_LEN as DENSE_DEFAULT_MAX_CONTEXT_LEN
from .dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    _dense_attention_reference,
)
from .dense_attention import _make_forward_batch as _make_dense_forward_batch
from .dense_attention import _populate_prefix_kv as _populate_dense_prefix_kv
from .dense_attention import (
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    run_dense_fixture_eager,
)
from .gdn_attention import DEFAULT_DEVICE as GDN_DEFAULT_DEVICE
from .gdn_attention import DEFAULT_DTYPE as GDN_DEFAULT_DTYPE
from .gdn_attention import (
    DEFAULT_HEAD_K_DIM,
    DEFAULT_HEAD_V_DIM,
)
from .gdn_attention import DEFAULT_MAX_CONTEXT_LEN as GDN_DEFAULT_MAX_CONTEXT_LEN
from .gdn_attention import (
    GDN_ATOL,
    GDN_RTOL,
    GDNAttentionCase,
    _clone_gdn_cache,
)
from .gdn_attention import _make_forward_batch as _make_gdn_forward_batch
from .gdn_attention import (
    _pure_torch_gdn_reference,
    _restore_gdn_cache,
    build_gdn_attention_fixture,
    run_gdn_fixture_eager,
)
from .mla_attention import DEFAULT_DEVICE as MLA_DEFAULT_DEVICE
from .mla_attention import DEFAULT_DTYPE as MLA_DEFAULT_DTYPE
from .mla_attention import DEFAULT_HIDDEN_SIZE as MLA_DEFAULT_HIDDEN_SIZE
from .mla_attention import (
    DEFAULT_KV_LORA_RANK,
)
from .mla_attention import DEFAULT_MAX_CONTEXT_LEN as MLA_DEFAULT_MAX_CONTEXT_LEN
from .mla_attention import (
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
)
from .mla_attention import _make_forward_batch as _make_mla_forward_batch
from .mla_attention import (
    _mla_attention_reference,
)
from .mla_attention import _populate_prefix_kv as _populate_mla_prefix_kv
from .mla_attention import (
    build_mla_attention_fixture,
    expected_mla_fixture_output,
    run_mla_fixture_eager,
)

DENSE_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4
MLA_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4
GDN_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 3


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


def _random_hidden_by_lens(
    lens: tuple[int, ...],
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
) -> list[torch.Tensor]:
    return [
        torch.randn(length, hidden_size, dtype=dtype, device=device) for length in lens
    ]


def _run_projected_cuda_graph_decode_case(
    testcase,
    case,
    *,
    build_fixture,
    build_kwargs: dict,
    capture_batch_size: int,
    hidden_size: int,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
    make_case,
    make_forward_batch,
    populate_prefix_kv,
    run_eager,
    expected_eager,
    reference,
    atol: float,
    rtol: float,
):
    _check_decode_cuda_graph_case(case, capture_batch_size)

    eager_fixture = build_fixture(testcase, case, **build_kwargs)
    eager_actual = run_eager(eager_fixture)
    expected = expected_eager(eager_fixture)
    torch.testing.assert_close(eager_actual, expected, atol=atol, rtol=rtol)

    graph_fixture = build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=capture_batch_size,
    )
    backend = graph_fixture.backend
    capture_prefix_len = max(0, backend.get_cuda_graph_seq_len_fill_value() - 1)
    capture_case = make_case(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * capture_batch_size,
    )
    capture_prefix_hidden = _random_hidden_by_lens(
        capture_case.prefix_lens,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
    capture_input_hidden = torch.randn(
        capture_case.num_input_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    capture_batch = make_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    populate_prefix_kv(
        graph_fixture,
        capture_case,
        capture_prefix_hidden,
        max_context_len=max_context_len,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(backend, capture_batch_size, capture_batch)
        capture_actual = graph_fixture.actual_module(
            capture_input_hidden,
            capture_batch,
        )
        backend.on_after_cuda_graph_warmup()
        capture_expected = reference(
            graph_fixture.reference_module,
            capture_case,
            capture_prefix_hidden,
            capture_input_hidden,
        )

        replay_pad_prefix_lens = (capture_prefix_len,) * (
            capture_batch_size - case.batch_size
        )
        replay_case = make_case(
            case,
            f"{case.name}_cuda_graph_replay",
            case.prefix_lens + replay_pad_prefix_lens,
        )
        replay_pad_prefix_hidden = _random_hidden_by_lens(
            replay_pad_prefix_lens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
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
        replay_batch = make_forward_batch(
            replay_case,
            graph_fixture.runner,
            max_context_len=max_context_len,
            device=device,
        )
        populate_prefix_kv(
            graph_fixture,
            replay_case,
            replay_prefix_hidden,
            max_context_len=max_context_len,
        )
        _init_cuda_graph_replay_metadata(backend, capture_batch_size, replay_batch)
        replay_actual = graph_fixture.actual_module(replay_input_hidden, replay_batch)

    torch.testing.assert_close(capture_actual, capture_expected, atol=atol, rtol=rtol)
    replay_expected = reference(
        graph_fixture.reference_module,
        replay_case,
        replay_prefix_hidden,
        replay_input_hidden,
    )
    torch.testing.assert_close(replay_actual, replay_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        replay_actual[: case.num_input_tokens],
        eager_actual,
        atol=atol,
        rtol=rtol,
    )


def _make_dense_graph_case(
    case: DenseAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> DenseAttentionCase:
    return DenseAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        sliding_window_size=case.sliding_window_size,
    )


def _populate_dense_graph_prefix_kv(
    fixture,
    case: DenseAttentionCase,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
):
    _populate_dense_prefix_kv(
        fixture.actual_module,
        case,
        fixture.runner,
        prefix_hidden,
        max_context_len=max_context_len,
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
    _run_projected_cuda_graph_decode_case(
        testcase,
        case,
        build_fixture=build_dense_attention_fixture,
        build_kwargs=dict(
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        make_case=_make_dense_graph_case,
        make_forward_batch=_make_dense_forward_batch,
        populate_prefix_kv=_populate_dense_graph_prefix_kv,
        run_eager=run_dense_fixture_eager,
        expected_eager=expected_dense_fixture_output,
        reference=_dense_attention_reference,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )


def _make_mla_graph_case(
    case: MLAAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> MLAAttentionCase:
    return MLAAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
    )


def _populate_mla_graph_prefix_kv(
    fixture,
    case: MLAAttentionCase,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
):
    _populate_mla_prefix_kv(
        fixture.actual_module,
        case,
        fixture.runner,
        fixture.backend,
        prefix_hidden,
        max_context_len=max_context_len,
    )


def run_mla_cuda_graph_decode_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = MLA_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    _run_projected_cuda_graph_decode_case(
        testcase,
        case,
        build_fixture=build_mla_attention_fixture,
        build_kwargs=dict(
            kv_lora_rank=kv_lora_rank,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        make_case=_make_mla_graph_case,
        make_forward_batch=_make_mla_forward_batch,
        populate_prefix_kv=_populate_mla_graph_prefix_kv,
        run_eager=run_mla_fixture_eager,
        expected_eager=expected_mla_fixture_output,
        reference=_mla_attention_reference,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
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
    _check_decode_cuda_graph_case(
        case,
        cuda_graph_capture_batch_size,
        allow_padding=False,
    )

    build_kwargs = dict(
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    eager_fixture = build_gdn_attention_fixture(testcase, case, **build_kwargs)
    eager_initial_cache = _clone_gdn_cache(eager_fixture)
    eager_actual = run_gdn_fixture_eager(eager_fixture)
    eager_expected = _pure_torch_gdn_reference(
        eager_fixture,
        eager_initial_cache[1],
    )
    torch.testing.assert_close(
        eager_actual, eager_expected.output, atol=GDN_ATOL, rtol=GDN_RTOL
    )

    graph_fixture = build_gdn_attention_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    graph_initial_cache = _clone_gdn_cache(graph_fixture)
    replay_mixed_qkv = graph_fixture.mixed_qkv
    replay_a = graph_fixture.a
    replay_b = graph_fixture.b
    backend = graph_fixture.backend
    capture_prefix_len = max(0, backend.get_cuda_graph_seq_len_fill_value() - 1)
    capture_case = GDNAttentionCase(
        name=f"{case.name}_cuda_graph_capture",
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_k_heads=case.num_k_heads,
        num_v_heads=case.num_v_heads,
        page_size=case.page_size,
        prefix_lens=(capture_prefix_len,) * cuda_graph_capture_batch_size,
    )
    capture_batch = _make_gdn_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    capture_mixed_qkv = torch.randn(
        capture_case.num_input_tokens,
        graph_fixture.actual_module.mixed_qkv_dim,
        dtype=dtype,
        device=device,
    )
    capture_a = torch.randn(
        capture_case.num_input_tokens,
        capture_case.num_v_heads,
        dtype=dtype,
        device=device,
    )
    capture_b = torch.randn(
        capture_case.num_input_tokens,
        capture_case.num_v_heads,
        dtype=dtype,
        device=device,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            cuda_graph_capture_batch_size,
            capture_batch,
        )
        graph_fixture.case = capture_case
        graph_fixture.forward_batch = capture_batch
        graph_fixture.mixed_qkv = capture_mixed_qkv
        graph_fixture.a = capture_a
        graph_fixture.b = capture_b
        capture_actual = graph_fixture.actual_module(
            capture_batch,
            capture_mixed_qkv,
            capture_a,
            capture_b,
        )
        backend.on_after_cuda_graph_warmup()
        capture_expected = _pure_torch_gdn_reference(
            graph_fixture,
            graph_initial_cache[1],
        )

        _restore_gdn_cache(graph_fixture, graph_initial_cache)
        graph_fixture.case = case
        graph_fixture.mixed_qkv = replay_mixed_qkv
        graph_fixture.a = replay_a
        graph_fixture.b = replay_b
        replay_batch = _make_gdn_forward_batch(
            case,
            graph_fixture.runner,
            max_context_len=max_context_len,
            device=device,
        )
        _init_cuda_graph_replay_metadata(
            backend,
            cuda_graph_capture_batch_size,
            replay_batch,
        )
        graph_fixture.forward_batch = replay_batch
        replay_actual = graph_fixture.actual_module(
            replay_batch,
            graph_fixture.mixed_qkv,
            graph_fixture.a,
            graph_fixture.b,
        )

    torch.testing.assert_close(
        capture_actual, capture_expected.output, atol=GDN_ATOL, rtol=GDN_RTOL
    )
    torch.testing.assert_close(
        replay_actual, eager_actual, atol=GDN_ATOL, rtol=GDN_RTOL
    )
