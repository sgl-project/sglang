from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
)
from sglang.srt.compilation.piecewise_context_manager import (
    set_forward_context as piecewise_forward_context,
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
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
    build_dense_attention_fixture,
    dense_attention_layers,
    dense_fixture_inputs,
    expected_dense_output_from_inputs,
    make_dense_token_padded_inputs,
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
    _restore_gdn_cache,
    build_gdn_attention_fixture,
    expected_gdn_output_from_inputs,
    gdn_attention_layers,
    gdn_fixture_inputs,
    make_gdn_token_padded_inputs,
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
    KDA_ATOL,
    KDA_RTOL,
    KDAAttentionCase,
    _clone_kda_cache,
    _restore_kda_cache,
    build_kda_attention_fixture,
    expected_kda_output_from_inputs,
    kda_attention_layers,
    kda_fixture_inputs,
    make_kda_token_padded_inputs,
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
    LIGHTNING_ATOL,
    LIGHTNING_RTOL,
    LightningAttentionCase,
    _clone_lightning_cache,
    _restore_lightning_cache,
    build_lightning_attention_fixture,
    expected_lightning_split_op_output_from_inputs,
    lightning_attention_layers,
    lightning_fixture_inputs,
    make_lightning_token_padded_inputs,
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
    MAMBA2_ATOL,
    MAMBA2_RTOL,
    Mamba2AttentionCase,
    _clone_mamba2_cache,
    _restore_mamba2_cache,
    build_mamba2_attention_fixture,
    expected_mamba2_output_from_inputs,
    make_mamba2_token_padded_inputs,
    mamba2_attention_layers,
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
    build_mla_attention_fixture,
    expected_mla_output_from_inputs,
    make_mla_token_padded_inputs,
    mla_attention_layers,
    mla_fixture_inputs,
    prepare_mla_runner_inputs,
    run_mla_fixture_eager,
    run_mla_forward,
)


@dataclass(frozen=True)
class SplitOpAdapter:
    build_fixture: Callable[..., Any]
    fixture_inputs: Callable[[Any], dict[str, Any]]
    make_token_padded_inputs: Callable[..., dict[str, Any]]
    prepare_inputs: Callable[..., None]
    run_eager: Callable[[Any], torch.Tensor]
    run_forward: Callable[[Any, Any, dict[str, Any]], torch.Tensor]
    expected_output: Callable[[Any, Any, dict[str, Any], Any], torch.Tensor]
    attention_layers: Callable[[Any], list[Any]]
    clone_state: Callable[[Any], Any] = lambda _: None
    restore_state: Callable[[Any, Any], None] = lambda _fixture, _state: None
    atol: float = 0.0
    rtol: float = 0.0


def _check_extend_split_op_case(case) -> None:
    if not case.forward_mode.is_extend_without_speculative():
        raise ValueError("PCG/BCG split-op coverage expects non-spec extend cases.")


def _split_op_context(*, breakable: bool):
    if breakable:
        return enable_breakable_cuda_graph()
    return enable_piecewise_cuda_graph()


def _make_static_forward_batch(raw_batch, static_num_tokens: int, device: str):
    raw_num_tokens = raw_batch.input_ids.numel()
    if static_num_tokens < raw_num_tokens:
        raise ValueError("static_num_tokens must cover the live input token count.")
    if static_num_tokens == raw_num_tokens:
        input_ids = raw_batch.input_ids
        positions = raw_batch.positions
        out_cache_loc = raw_batch.out_cache_loc
    else:
        pad_tokens = static_num_tokens - raw_num_tokens
        input_ids = torch.cat(
            [
                raw_batch.input_ids,
                torch.zeros(pad_tokens, dtype=raw_batch.input_ids.dtype, device=device),
            ],
            dim=0,
        )
        positions = torch.cat(
            [
                raw_batch.positions,
                torch.zeros(pad_tokens, dtype=raw_batch.positions.dtype, device=device),
            ],
            dim=0,
        )
        out_cache_loc = torch.cat(
            [
                raw_batch.out_cache_loc,
                torch.zeros(
                    pad_tokens,
                    dtype=raw_batch.out_cache_loc.dtype,
                    device=device,
                ),
            ],
            dim=0,
        )

    raw_batch.num_token_non_padded_cpu = raw_num_tokens
    return replace(
        raw_batch,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        padded_static_len=static_num_tokens,
        num_token_non_padded_cpu=raw_num_tokens,
    )


def _slice_live_tokens(output: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if output.dim() >= 2 and output.shape[0] == 1:
        return output[:, :num_tokens]
    return output[:num_tokens]


def _run_split_op_extend_case(
    testcase,
    case,
    *,
    adapter: SplitOpAdapter,
    build_kwargs: dict[str, Any],
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
    breakable: bool,
    static_num_tokens: int | None,
):
    _check_extend_split_op_case(case)

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

    split_fixture = adapter.build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_piecewise_cuda_graph=False,
    )
    split_inputs = adapter.fixture_inputs(split_fixture)
    split_initial_state = adapter.clone_state(split_fixture)
    expected = adapter.expected_output(
        split_fixture,
        case,
        split_inputs,
        split_initial_state,
    )
    raw_batch = split_fixture.forward_batch
    raw_num_tokens = case.num_input_tokens
    static_num_tokens = static_num_tokens or raw_num_tokens
    static_batch = _make_static_forward_batch(raw_batch, static_num_tokens, device)
    static_inputs = adapter.make_token_padded_inputs(
        case,
        split_fixture,
        static_num_tokens,
        split_inputs,
        dtype=dtype,
        device=device,
    )
    adapter.prepare_inputs(
        split_fixture,
        case,
        raw_batch,
        split_inputs,
        max_context_len=max_context_len,
    )

    with (
        torch.no_grad(),
        _split_op_context(breakable=breakable),
        forward_context(ForwardContext(attn_backend=split_fixture.backend)),
        piecewise_forward_context(
            static_batch,
            adapter.attention_layers(split_fixture),
            None,
            [],
            [],
        ),
    ):
        split_fixture.backend.init_forward_metadata(raw_batch)
        actual = adapter.run_forward(split_fixture, static_batch, static_inputs)

    actual = _slice_live_tokens(actual, raw_num_tokens)
    torch.testing.assert_close(actual, expected, atol=adapter.atol, rtol=adapter.rtol)
    torch.testing.assert_close(
        actual,
        eager_actual,
        atol=adapter.atol,
        rtol=adapter.rtol,
    )
    adapter.restore_state(split_fixture, split_initial_state)


def run_dense_split_op_extend_case(
    testcase,
    case: DenseAttentionCase,
    *,
    breakable: bool,
    static_num_tokens: int | None = None,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
):
    adapter = SplitOpAdapter(
        build_fixture=build_dense_attention_fixture,
        fixture_inputs=dense_fixture_inputs,
        make_token_padded_inputs=make_dense_token_padded_inputs,
        prepare_inputs=prepare_dense_runner_inputs,
        run_eager=run_dense_fixture_eager,
        run_forward=run_dense_forward,
        expected_output=expected_dense_output_from_inputs,
        attention_layers=dense_attention_layers,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    _run_split_op_extend_case(
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
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        breakable=breakable,
        static_num_tokens=static_num_tokens,
    )


def run_mla_split_op_extend_case(
    testcase,
    case: MLAAttentionCase,
    *,
    breakable: bool,
    static_num_tokens: int | None = None,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
):
    adapter = SplitOpAdapter(
        build_fixture=build_mla_attention_fixture,
        fixture_inputs=mla_fixture_inputs,
        make_token_padded_inputs=make_mla_token_padded_inputs,
        prepare_inputs=prepare_mla_runner_inputs,
        run_eager=run_mla_fixture_eager,
        run_forward=run_mla_forward,
        expected_output=expected_mla_output_from_inputs,
        attention_layers=mla_attention_layers,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
    _run_split_op_extend_case(
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
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        breakable=breakable,
        static_num_tokens=static_num_tokens,
    )


def run_gdn_split_op_extend_case(
    testcase,
    case: GDNAttentionCase,
    *,
    breakable: bool,
    static_num_tokens: int | None = None,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
    max_context_len: int = GDN_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = GDN_DEFAULT_DTYPE,
    device: str = GDN_DEFAULT_DEVICE,
):
    adapter = SplitOpAdapter(
        build_fixture=build_gdn_attention_fixture,
        fixture_inputs=gdn_fixture_inputs,
        make_token_padded_inputs=make_gdn_token_padded_inputs,
        prepare_inputs=prepare_gdn_runner_inputs,
        run_eager=run_gdn_fixture_eager,
        run_forward=run_gdn_forward,
        expected_output=expected_gdn_output_from_inputs,
        attention_layers=gdn_attention_layers,
        clone_state=_clone_gdn_cache,
        restore_state=_restore_gdn_cache,
        atol=GDN_ATOL,
        rtol=GDN_RTOL,
    )
    _run_split_op_extend_case(
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
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        breakable=breakable,
        static_num_tokens=static_num_tokens,
    )


def run_kda_split_op_extend_case(
    testcase,
    case: KDAAttentionCase,
    *,
    breakable: bool,
    static_num_tokens: int | None = None,
    head_k_dim: int = KDA_DEFAULT_HEAD_K_DIM,
    head_v_dim: int = KDA_DEFAULT_HEAD_V_DIM,
    max_context_len: int = KDA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = KDA_DEFAULT_DTYPE,
    device: str = KDA_DEFAULT_DEVICE,
):
    """KDA PCG/BCG split-op extend. Verifies the live-token slicing contract
    with a larger static token buffer, mirroring GDN's split_op coverage."""
    adapter = SplitOpAdapter(
        build_fixture=build_kda_attention_fixture,
        fixture_inputs=kda_fixture_inputs,
        make_token_padded_inputs=make_kda_token_padded_inputs,
        prepare_inputs=prepare_kda_runner_inputs,
        run_eager=run_kda_fixture_eager,
        run_forward=run_kda_forward,
        expected_output=expected_kda_output_from_inputs,
        attention_layers=kda_attention_layers,
        clone_state=_clone_kda_cache,
        restore_state=_restore_kda_cache,
        atol=KDA_ATOL,
        rtol=KDA_RTOL,
    )
    _run_split_op_extend_case(
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
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        breakable=breakable,
        static_num_tokens=static_num_tokens,
    )


def run_lightning_split_op_extend_case(
    testcase,
    case: LightningAttentionCase,
    *,
    breakable: bool,
    static_num_tokens: int | None = None,
    head_dim: int = LIGHTNING_DEFAULT_HEAD_DIM,
    max_context_len: int = LIGHTNING_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = LIGHTNING_DEFAULT_DTYPE,
    device: str = LIGHTNING_DEFAULT_DEVICE,
):
    """Lightning PCG/BCG split-op extend. Same pattern as KDA/GDN."""
    adapter = SplitOpAdapter(
        build_fixture=build_lightning_attention_fixture,
        fixture_inputs=lightning_fixture_inputs,
        make_token_padded_inputs=make_lightning_token_padded_inputs,
        prepare_inputs=prepare_lightning_runner_inputs,
        run_eager=run_lightning_fixture_eager,
        run_forward=run_lightning_forward,
        expected_output=expected_lightning_split_op_output_from_inputs,
        attention_layers=lightning_attention_layers,
        clone_state=_clone_lightning_cache,
        restore_state=_restore_lightning_cache,
        atol=LIGHTNING_ATOL,
        rtol=LIGHTNING_RTOL,
    )
    _run_split_op_extend_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            head_dim=head_dim,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        breakable=breakable,
        static_num_tokens=static_num_tokens,
    )


def run_mamba2_split_op_extend_case(
    testcase,
    case: Mamba2AttentionCase,
    *,
    breakable: bool,
    static_num_tokens: int | None = None,
    max_context_len: int = MAMBA2_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MAMBA2_DEFAULT_DTYPE,
    device: str = MAMBA2_DEFAULT_DEVICE,
):
    """Mamba2 PCG/BCG split-op extend. Same pattern as KDA. Mamba2's
    forward writes through an `empty_like(hidden_states)` buffer that
    short-circuits the RadixAttention dispatch path, so the per-head-vs-flat
    shape mismatch that blocks Lightning split-op doesn't apply."""
    adapter = SplitOpAdapter(
        build_fixture=build_mamba2_attention_fixture,
        fixture_inputs=mamba2_fixture_inputs,
        make_token_padded_inputs=make_mamba2_token_padded_inputs,
        prepare_inputs=prepare_mamba2_runner_inputs,
        run_eager=run_mamba2_fixture_eager,
        run_forward=run_mamba2_forward,
        expected_output=expected_mamba2_output_from_inputs,
        attention_layers=mamba2_attention_layers,
        clone_state=_clone_mamba2_cache,
        restore_state=_restore_mamba2_cache,
        atol=MAMBA2_ATOL,
        rtol=MAMBA2_RTOL,
    )
    _run_split_op_extend_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=dict(
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        breakable=breakable,
        static_num_tokens=static_num_tokens,
    )
