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
from ..attention_methods.dsa_attention import (
    DSA_PAGE_SIZE,
    DSA_SPARSE_ATOL,
    DSA_SPARSE_RTOL,
    DSAAttentionCase,
    _clone_dsa_sparse_cache,
)
from ..attention_methods.dsa_attention import (
    _make_forward_batch as _make_dsa_forward_batch,
)
from ..attention_methods.dsa_attention import (
    _restore_dsa_sparse_cache,
    build_dsa_sparse_attention_fixture,
    dsa_sparse_fixture_inputs,
    expected_dsa_sparse_output_from_inputs,
    make_dsa_sparse_case_with_prefix_lens,
    make_dsa_sparse_random_inputs,
    make_dsa_sparse_replay_inputs,
    prepare_dsa_sparse_runner_inputs,
    run_dsa_sparse_forward,
)
from ..attention_methods.dsv4_attention import (
    DSV4_ATOL,
    DSV4_RTOL,
    DSV4AttentionCase,
)
from ..attention_methods.dsv4_attention import (
    _make_forward_batch as _make_dsv4_forward_batch,
)
from ..attention_methods.dsv4_attention import (
    build_dsv4_attention_fixture,
    dsv4_fixture_inputs,
    expected_dsv4_output_from_inputs,
    make_dsv4_case_with_prefix_lens,
    make_dsv4_padded_replay_inputs,
    make_dsv4_random_inputs,
    prepare_dsv4_runner_inputs,
    run_dsv4_fixture_eager,
    run_dsv4_forward,
)
from ..attention_methods.dual_chunk_attention import (
    DualChunkAttentionCase,
    _clone_dual_chunk_cache,
    _restore_dual_chunk_cache,
    build_dual_chunk_attention_fixture,
    dual_chunk_fixture_inputs,
    expected_dual_chunk_output_from_inputs,
    make_dual_chunk_case_with_prefix_lens,
    make_dual_chunk_random_inputs,
    make_dual_chunk_replay_inputs,
    prepare_dual_chunk_runner_inputs,
    run_dual_chunk_fixture_eager,
    run_dual_chunk_forward,
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
from ..attention_methods.kda_attention import DEFAULT_DEVICE as KDA_DEFAULT_DEVICE
from ..attention_methods.kda_attention import DEFAULT_DTYPE as KDA_DEFAULT_DTYPE
from ..attention_methods.kda_attention import (
    DEFAULT_HEAD_K_DIM as KDA_DEFAULT_HEAD_K_DIM,
)
from ..attention_methods.kda_attention import (
    DEFAULT_HEAD_V_DIM as KDA_DEFAULT_HEAD_V_DIM,
)
from ..attention_methods.kda_attention import (
    DEFAULT_MAX_CONTEXT_LEN as KDA_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.kda_attention import (
    KDA_GRAPH_ATOL,
    KDA_GRAPH_RTOL,
    KDAAttentionCase,
    _clone_kda_cache,
)
from ..attention_methods.kda_attention import (
    _make_forward_batch as _make_kda_forward_batch,
)
from ..attention_methods.kda_attention import (
    _restore_kda_cache,
    build_kda_attention_fixture,
    expected_kda_output_from_inputs,
    kda_fixture_inputs,
    make_kda_case_with_prefix_lens,
    make_kda_random_inputs,
    make_kda_replay_inputs,
    prepare_kda_runner_inputs,
    run_kda_fixture_eager,
    run_kda_forward,
)
from ..attention_methods.lightning_attention import (
    DEFAULT_DEVICE as LIGHTNING_DEFAULT_DEVICE,
)
from ..attention_methods.lightning_attention import (
    DEFAULT_DTYPE as LIGHTNING_DEFAULT_DTYPE,
)
from ..attention_methods.lightning_attention import (
    DEFAULT_HEAD_DIM as LIGHTNING_DEFAULT_HEAD_DIM,
)
from ..attention_methods.lightning_attention import (
    DEFAULT_MAX_CONTEXT_LEN as LIGHTNING_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.lightning_attention import (
    LIGHTNING_GRAPH_ATOL,
    LIGHTNING_GRAPH_RTOL,
    LightningAttentionCase,
    _clone_lightning_cache,
)
from ..attention_methods.lightning_attention import (
    _make_forward_batch as _make_lightning_forward_batch,
)
from ..attention_methods.lightning_attention import (
    _restore_lightning_cache,
    build_lightning_attention_fixture,
    expected_lightning_output_from_inputs,
    lightning_fixture_inputs,
    make_lightning_case_with_prefix_lens,
    make_lightning_random_inputs,
    make_lightning_replay_inputs,
    prepare_lightning_runner_inputs,
    run_lightning_fixture_eager,
    run_lightning_forward,
)
from ..attention_methods.mamba2_attention import DEFAULT_DEVICE as MAMBA2_DEFAULT_DEVICE
from ..attention_methods.mamba2_attention import DEFAULT_DTYPE as MAMBA2_DEFAULT_DTYPE
from ..attention_methods.mamba2_attention import (
    DEFAULT_MAX_CONTEXT_LEN as MAMBA2_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.mamba2_attention import (
    MAMBA2_GRAPH_ATOL,
    MAMBA2_GRAPH_RTOL,
    Mamba2AttentionCase,
    _clone_mamba2_cache,
)
from ..attention_methods.mamba2_attention import (
    _make_forward_batch as _make_mamba2_forward_batch,
)
from ..attention_methods.mamba2_attention import (
    _restore_mamba2_cache,
    build_mamba2_attention_fixture,
    expected_mamba2_output_from_inputs,
    make_mamba2_case_with_prefix_lens,
    make_mamba2_random_inputs,
    make_mamba2_replay_inputs,
    mamba2_fixture_inputs,
    prepare_mamba2_runner_inputs,
    run_mamba2_fixture_eager,
    run_mamba2_forward,
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
from .metadata_invariants import assert_cg_metadata_well_formed

DENSE_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4
MLA_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 4
GDN_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 3
DSV4_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 2
KDA_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 3
LIGHTNING_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 3
MAMBA2_CUDA_GRAPH_CAPTURE_BATCH_SIZE = 3


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
    if (
        backend.supports_tree_mask_scratch
        and batch.forward_mode.is_target_verify()
        and batch.spec_info is not None
    ):
        # Mirror DecodeCudaGraphRunner: the runner provisions the tree-mask
        # scratch before init_cuda_graph_state; the kit drives backends
        # directly, so it must follow the same protocol. Only target-verify
        # capture consumes the scratch (draft-extend spec inputs carry no
        # draft_token_num).
        backend.init_tree_mask_scratch(
            max_num_tokens=batch.input_ids.numel(),
            max_context_len=backend.max_context_len,
            num_draft_tokens=batch.spec_info.draft_token_num,
            device=batch.input_ids.device,
        )
    backend.init_cuda_graph_state(
        max_bs=capture_batch_size,
        max_num_tokens=batch.input_ids.numel(),
    )
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    backend.init_forward_metadata_in_graph(batch)


def _init_cuda_graph_replay_metadata(backend, capture_batch_size: int, batch):
    from types import SimpleNamespace

    fb_view = SimpleNamespace(
        batch_size=capture_batch_size,
        forward_mode=batch.forward_mode,
        actual_forward_mode=batch.forward_mode,
        input_ids=batch.input_ids,
        positions=getattr(batch, "positions", None),
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        seq_lens_sum=batch.seq_lens_sum,
        seq_lens_cpu=batch.seq_lens_cpu,
        encoder_lens=batch.encoder_lens,
        out_cache_loc=getattr(batch, "out_cache_loc", None),
        spec_info=batch.spec_info,
    )
    backend.init_forward_metadata_out_graph(fb_view)
    # No real cuda graph here, so run the in-graph step explicitly to produce
    # the Full metadata the forward path expects (no-op for non-DSV4).
    backend.init_forward_metadata_in_graph(fb_view)
    # Best-effort metadata-shape sanity check — catches negative kv_lens and
    # non-monotonic indptr that would otherwise leave real-row output correct
    # but corrupt padded-row scratch state. See `metadata_invariants.py`.
    assert_cg_metadata_well_formed(backend, bs=capture_batch_size)


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
    # NOTE: `capture_prefix_len`-vs-replay assertion happens below once the
    # graph fixture is built (we need `backend.get_cuda_graph_seq_len_fill_value`).

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
    if any(p < capture_prefix_len for p in case.prefix_lens):
        raise AssertionError(
            f"replay prefix_lens must each be >= capture_prefix_len="
            f"{capture_prefix_len} so capture-time random KV does not leak "
            f"into replay; got prefix_lens={case.prefix_lens}"
        )

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
        # Capture forward is a JIT warmup that mirrors production: the
        # captured CUDA graph records kernel launches against buffers
        # that *will* be populated by `init_forward_metadata_replay_cuda_graph`
        # at replay. The capture-time output itself is discarded in
        # production — and we discard it here too. Backends like FA3/FA4
        # legitimately assign-but-don't-populate metadata buffers at
        # capture, which makes the capture-time output undefined; only
        # the replay output is contractually required to match the
        # reference.
        adapter.run_forward(graph_fixture, capture_batch, capture_inputs)
        backend.on_after_cuda_graph_warmup()

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


def run_dsv4_cuda_graph_decode_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    swa_size: int = 1024,
    max_context_len: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cuda_graph_capture_batch_size: int = DSV4_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_dsv4_attention_fixture,
        make_case=make_dsv4_case_with_prefix_lens,
        make_forward_batch=_make_dsv4_forward_batch,
        fixture_inputs=dsv4_fixture_inputs,
        make_capture_inputs=make_dsv4_random_inputs,
        make_replay_inputs=make_dsv4_padded_replay_inputs,
        prepare_inputs=prepare_dsv4_runner_inputs,
        run_eager=run_dsv4_fixture_eager,
        run_forward=run_dsv4_forward,
        expected_output=expected_dsv4_output_from_inputs,
        atol=DSV4_ATOL,
        rtol=DSV4_RTOL,
    )
    _run_cuda_graph_decode_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            swa_size=swa_size,
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


def run_kda_cuda_graph_decode_case(
    testcase,
    case: KDAAttentionCase,
    *,
    head_k_dim: int = KDA_DEFAULT_HEAD_K_DIM,
    head_v_dim: int = KDA_DEFAULT_HEAD_V_DIM,
    max_context_len: int = KDA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = KDA_DEFAULT_DTYPE,
    device: str = KDA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = KDA_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    """KDA CUDA-graph decode replay. Mirrors `run_gdn_cuda_graph_decode_case`:
    KDA inherits the same `MambaAttnBackendBase` capture/replay path through
    `HybridLinearAttnBackend`, so the adapter wiring is identical to GDN.
    Only DECODE / TARGET_VERIFY are reachable here (the underlying
    `_replay_metadata` rejects other modes — see kda/README.md).
    """
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_kda_attention_fixture,
        make_case=make_kda_case_with_prefix_lens,
        make_forward_batch=_make_kda_forward_batch,
        fixture_inputs=kda_fixture_inputs,
        make_capture_inputs=make_kda_random_inputs,
        make_replay_inputs=make_kda_replay_inputs,
        prepare_inputs=prepare_kda_runner_inputs,
        run_eager=run_kda_fixture_eager,
        run_forward=run_kda_forward,
        expected_output=expected_kda_output_from_inputs,
        clone_state=_clone_kda_cache,
        restore_state=_restore_kda_cache,
        allow_padding=False,
        atol=KDA_GRAPH_ATOL,
        rtol=KDA_GRAPH_RTOL,
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


def run_lightning_cuda_graph_decode_case(
    testcase,
    case: LightningAttentionCase,
    *,
    head_dim: int = LIGHTNING_DEFAULT_HEAD_DIM,
    max_context_len: int = LIGHTNING_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = LIGHTNING_DEFAULT_DTYPE,
    device: str = LIGHTNING_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = LIGHTNING_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    """Lightning (Bailing seg_la) CUDA-graph decode replay. Mirrors GDN/KDA;
    Lightning uses `LightningAttentionBackend` (installed directly via
    ForwardContext rather than through `HybridLinearAttnBackend`), but the
    capture/replay contract is the same shape because the backend also
    inherits from `MambaAttnBackendBase`. Loose tolerance to absorb seg_la
    Triton kernel CG-replay drift; eager tolerance preserved for non-graph
    cases."""
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_lightning_attention_fixture,
        make_case=make_lightning_case_with_prefix_lens,
        make_forward_batch=_make_lightning_forward_batch,
        fixture_inputs=lightning_fixture_inputs,
        make_capture_inputs=make_lightning_random_inputs,
        make_replay_inputs=make_lightning_replay_inputs,
        prepare_inputs=prepare_lightning_runner_inputs,
        run_eager=run_lightning_fixture_eager,
        run_forward=run_lightning_forward,
        expected_output=expected_lightning_output_from_inputs,
        clone_state=_clone_lightning_cache,
        restore_state=_restore_lightning_cache,
        allow_padding=False,
        atol=LIGHTNING_GRAPH_ATOL,
        rtol=LIGHTNING_GRAPH_RTOL,
    )
    _run_cuda_graph_decode_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            head_dim=head_dim,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        capture_batch_size=cuda_graph_capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def run_mamba2_cuda_graph_decode_case(
    testcase,
    case: Mamba2AttentionCase,
    *,
    max_context_len: int = MAMBA2_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MAMBA2_DEFAULT_DTYPE,
    device: str = MAMBA2_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = MAMBA2_CUDA_GRAPH_CAPTURE_BATCH_SIZE,
):
    """Mamba2 CUDA-graph decode replay. The fixture's
    `initialize_mamba_selective_state_update_backend` call makes
    `MambaMixer2.forward_decode` reachable; this adapter then drives the
    capture/replay lifecycle the same way as GDN/KDA/Lightning, snapshotting
    both SSM and conv state between capture and replay so the recurrent
    backend output is reproducible.

    Loose `MAMBA2_GRAPH_ATOL=1e-1` absorbs CG-replay drift; eager
    `MAMBA2_ATOL=5e-2` is kept for non-graph cases.
    """
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_mamba2_attention_fixture,
        make_case=make_mamba2_case_with_prefix_lens,
        make_forward_batch=_make_mamba2_forward_batch,
        fixture_inputs=mamba2_fixture_inputs,
        make_capture_inputs=make_mamba2_random_inputs,
        make_replay_inputs=make_mamba2_replay_inputs,
        prepare_inputs=prepare_mamba2_runner_inputs,
        run_eager=run_mamba2_fixture_eager,
        run_forward=run_mamba2_forward,
        expected_output=expected_mamba2_output_from_inputs,
        clone_state=_clone_mamba2_cache,
        restore_state=_restore_mamba2_cache,
        allow_padding=False,
        atol=MAMBA2_GRAPH_ATOL,
        rtol=MAMBA2_GRAPH_RTOL,
    )
    _run_cuda_graph_decode_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        capture_batch_size=cuda_graph_capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def _run_dsa_sparse_eager_for_cg(fixture):
    """Eager wrapper for the DSA sparse CG decode adapter — wraps a
    `forward_context` around `run_dsa_sparse_forward` so `module.attn`
    sees the active backend (the existing
    `run_dsa_sparse_fixture_eager` has its own context but takes an
    extra `testcase` arg for `skipTest`, which doesn't fit the
    adapter's `run_eager(fixture)` signature)."""
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return run_dsa_sparse_forward(
            fixture, fixture.forward_batch, dsa_sparse_fixture_inputs(fixture)
        )


def run_dsa_sparse_cuda_graph_decode_case(
    testcase,
    case: DSAAttentionCase,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DENSE_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int | None = None,
    dsa_decode_backend: str = "flashmla_kv",
    fp8_kv_cache: bool = False,
):
    """DSA sparse-topk CUDA-graph decode replay (`flashmla_kv` path).
    Sparse decode uses cached MLA latent KV (written by
    `_populate_dsa_sparse_prefix_kv` at fixture build), so the
    capture/replay K-cache boundary is compatible with piecewise CG —
    unlike the dense-fallback MHA_ONE_SHOT path which passes prefix+
    extend K inline."""
    if not case.forward_mode.is_decode():
        raise ValueError(
            "run_dsa_sparse_cuda_graph_decode_case expects a DECODE case "
            "(the sparse `flashmla_kv` path is the natural CG decode target)."
        )
    capture_batch_size = cuda_graph_capture_batch_size or case.batch_size
    if max_context_len is None:
        max_context_len = max(case.seq_lens) if case.seq_lens else DSA_PAGE_SIZE
        # Round up to page_size multiple.
        if max_context_len % case.page_size:
            max_context_len = (
                (max_context_len + case.page_size - 1) // case.page_size
            ) * case.page_size
    from ..attention_methods.dsa_attention import (
        DSA_SPARSE_FP8_ATOL,
        DSA_SPARSE_FP8_RTOL,
    )

    if fp8_kv_cache:
        atol, rtol = DSA_SPARSE_FP8_ATOL, DSA_SPARSE_FP8_RTOL
    else:
        atol, rtol = DSA_SPARSE_ATOL, DSA_SPARSE_RTOL
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_dsa_sparse_attention_fixture,
        make_case=make_dsa_sparse_case_with_prefix_lens,
        make_forward_batch=_make_dsa_forward_batch,
        fixture_inputs=dsa_sparse_fixture_inputs,
        make_capture_inputs=make_dsa_sparse_random_inputs,
        make_replay_inputs=make_dsa_sparse_replay_inputs,
        prepare_inputs=prepare_dsa_sparse_runner_inputs,
        run_eager=_run_dsa_sparse_eager_for_cg,
        run_forward=run_dsa_sparse_forward,
        expected_output=expected_dsa_sparse_output_from_inputs,
        clone_state=_clone_dsa_sparse_cache,
        restore_state=_restore_dsa_sparse_cache,
        allow_padding=False,
        atol=atol,
        rtol=rtol,
    )
    _run_cuda_graph_decode_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
            dsa_decode_backend=dsa_decode_backend,
            fp8_kv_cache=fp8_kv_cache,
        ),
        capture_batch_size=capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def run_dual_chunk_cuda_graph_decode_case(
    testcase,
    case: DualChunkAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int | None = None,
):
    """Dual-chunk CUDA-graph decode replay. Decode reads cached K/V (set
    by `set_kv_buffer` inside `forward_decode`) so the capture/replay
    contract is the same shape as dense attention. The
    `_clone_dual_chunk_cache` / `_restore_dual_chunk_cache` hooks snapshot
    both K and V buffers so the capture forward's writes don't bleed into
    replay state."""
    if not case.forward_mode.is_decode():
        raise ValueError("run_dual_chunk_cuda_graph_decode_case expects a DECODE case.")
    capture_batch_size = cuda_graph_capture_batch_size or case.batch_size
    adapter = CudaGraphDecodeAdapter(
        build_fixture=build_dual_chunk_attention_fixture,
        make_case=make_dual_chunk_case_with_prefix_lens,
        make_forward_batch=_make_dense_forward_batch,
        fixture_inputs=dual_chunk_fixture_inputs,
        make_capture_inputs=make_dual_chunk_random_inputs,
        make_replay_inputs=make_dual_chunk_replay_inputs,
        prepare_inputs=prepare_dual_chunk_runner_inputs,
        run_eager=run_dual_chunk_fixture_eager,
        run_forward=run_dual_chunk_forward,
        expected_output=expected_dual_chunk_output_from_inputs,
        clone_state=_clone_dual_chunk_cache,
        restore_state=_restore_dual_chunk_cache,
        allow_padding=True,
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
        capture_batch_size=capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
