"""gfx950 KDA prefill (TLX) and decode against an fp32 token-by-token recurrence."""

import os
from itertools import accumulate, pairwise
from types import SimpleNamespace

import pytest
import torch
import triton

from sglang.kernels.ops.attention.tlx import kda as tlx_kda
from sglang.kernels.ops.attention.tlx.kda import decode_available, prefill_available
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not decode_available(), reason="gfx950 required")
requires_prefill = pytest.mark.skipif(
    not prefill_available(),
    reason="TLX-capable Triton with async copy required",
)

DEVICE = "cuda"

# (total_tokens, sequences, heads, key_dim, value_dim, dtype)
PREFILL_SHAPES = [(64, 1, 4, 128, 128, "bf16")] + [
    (tokens, sequences, heads, 128, 128, "bf16")
    for heads in (4, 12)
    for tokens, sequences in ((4096, 1), (4096, 4), (131072, 1), (131072, 8))
]
# (batch, heads, key_dim, value_dim, dtype)
DECODE_SHAPES = [
    (batch, heads, 128, 128, "bf16")
    for heads in (4, 12)
    for batch in (1, 2, 4, 8, 16, 32)
]


def test_only_plugin_tlx_needs_async_copy():
    triton_dir = os.path.dirname(os.path.abspath(triton.__file__))
    native = SimpleNamespace(
        __file__=os.path.join(triton_dir, "language", "extra", "tlx", "__init__.py")
    )
    plugin = SimpleNamespace(
        __file__=os.path.join(os.path.dirname(triton_dir), "utlx_plugin", "__init__.py")
    )
    assert tlx_kda._tlx_is_native(native)
    assert not tlx_kda._tlx_is_native(plugin)
    assert not tlx_kda._tlx_is_native(SimpleNamespace())


@requires_prefill
def test_plugin_tlx_prefill_is_unavailable_without_async_copy(monkeypatch):
    from triton.language.extra import tlx

    if tlx_kda._tlx_is_native(tlx):
        pytest.skip("native TLX compiles the prefill without async copy")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    assert not prefill_available()


def kda_paged_prefill(*args, **kwargs):
    return get_kernel("attention.kda_paged_prefill", KernelBackend.TLX)(*args, **kwargs)


def kda_recurrent_decode(*args, **kwargs):
    return get_kernel("attention.kda_recurrent_decode", KernelBackend.TRITON)(
        *args, **kwargs
    )


def _kda_recurrent_reference(q, k, v, g, beta, state, scale):
    """FP32 recurrence for prepared inputs and a physical [H, V, K] state."""
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    state = state.float().clone()
    outputs = []
    for token in range(q.shape[0]):
        state *= g[token].exp()[:, None, :]
        prediction = torch.einsum("hvk,hk->hv", state, k[token])
        delta = beta[token, :, None] * (v[token] - prediction)
        state += torch.einsum("hv,hk->hvk", delta, k[token])
        outputs.append(scale * torch.einsum("hvk,hk->hv", state, q[token]))
    if not outputs:
        return v.new_empty((0, *v.shape[1:]), dtype=torch.float32), state
    return torch.stack(outputs), state


def _normalized_kda_input(shape):
    value = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    return torch.nn.functional.normalize(value, dim=-1).to(torch.bfloat16)


def _check_kda_prefill(
    lengths, heads, scale, cu_on_cpu=False, gate_scale=1.0, state_atol=8e-3
):
    torch.manual_seed(31)
    dim = 128
    total_tokens = sum(lengths)
    shape = (1, total_tokens, heads, dim)
    q = _normalized_kda_input(shape)
    k = _normalized_kda_input(shape)
    v = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    g = -gate_scale * torch.nn.functional.softplus(
        torch.randn(shape, device=DEVICE, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, total_tokens, heads, device=DEVICE, dtype=torch.float32)
    )
    initial_state = 0.1 * torch.randn(
        len(lengths), heads, dim, dim, device=DEVICE, dtype=torch.float32
    )
    cu_device = "cpu" if cu_on_cpu else DEVICE
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device=cu_device, dtype=torch.int64
    )

    expected_outputs = []
    expected_states = []
    for sequence, (begin, end) in enumerate(pairwise(cu_seqlens.tolist())):
        expected_output, expected_state = _kda_recurrent_reference(
            q[0, begin:end],
            k[0, begin:end],
            v[0, begin:end],
            g[0, begin:end],
            beta[0, begin:end],
            initial_state[sequence],
            scale,
        )
        expected_outputs.append(expected_output)
        expected_states.append(expected_state)

    actual_output, actual_state = kda_paged_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(
        actual_output[0].float(), torch.cat(expected_outputs), atol=8e-3, rtol=3e-2
    )
    torch.testing.assert_close(
        actual_state, torch.stack(expected_states), atol=state_atol, rtol=3e-2
    )
    assert actual_state.dtype == torch.float32
    assert actual_state.shape == (len(lengths), heads, dim, dim)
    assert actual_state.stride()[-2:] == (dim, 1)


@requires_prefill
@pytest.mark.parametrize(
    "scale", [1.0, 128**-0.5], ids=["unit-scale", "attention-scale"]
)
def test_kda_paged_prefill_chunk_boundaries(scale):
    _check_kda_prefill(
        [0, 1, 15, 16, 17, 63, 64, 65], heads=12, scale=scale, cu_on_cpu=True
    )


@requires_prefill
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [
        pytest.param([64], 12, id="single-chunk-h12"),
        pytest.param([256, 255, 128, 17], 4, id="ragged-multichunk-h4"),
    ],
)
def test_kda_paged_prefill_smoke_shapes(lengths, heads):
    _check_kda_prefill(lengths, heads=heads, scale=128**-0.5)


@requires_prefill
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [
        # Enough heads that the unsplit walk takes all 128 value rows per program.
        pytest.param([256] * 16, 12, id="walk-h12"),
        # Long enough that the split's second pass also takes full-width rows.
        pytest.param([8192], 12, id="time-split-h12"),
        # Its 17-token sequence leaves most of its groups empty.
        pytest.param([17, 4000, 6000], 4, id="time-split-ragged-h4"),
        # The last chunk ends 37 rows short: its loads must stay inside the inputs.
        pytest.param([1638, 1228, 819, 411], 12, id="ragged-tail-h12"),
    ],
)
def test_kda_paged_prefill_walk_shapes(lengths, heads):
    # A wrong group-to-group hand-off only reaches the output if the state
    # remembers across a time group (about 400 tokens here), which takes a
    # gate this slow. It also lets bf16 rounding build up in the state.
    _check_kda_prefill(
        lengths, heads=heads, scale=128**-0.5, gate_scale=0.001, state_atol=2e-2
    )


def _raw_kda_inputs(lengths, heads):
    """Split views of a packed bf16 qkv, a raw gate and a raw beta, as a model layer passes them."""
    dim = 128
    total_tokens = sum(lengths)
    width = heads * dim
    qkv = torch.randn(1, total_tokens, 3 * width, device=DEVICE, dtype=torch.bfloat16)
    q, k, v = (x.unflatten(-1, (heads, dim)) for x in qkv.split(width, dim=-1))
    g = torch.randn(1, total_tokens, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    beta = torch.randn(1, total_tokens, heads, device=DEVICE, dtype=torch.bfloat16)
    A_log = -4.0 + 0.1 * torch.randn(heads, device=DEVICE)
    dt_bias = 0.1 * torch.randn(width, device=DEVICE)
    return q, k, v, g, beta, A_log, dt_bias


def _activate_kda_inputs(q, k, g, beta, A_log, dt_bias, lower_bound):
    """The fp32 prepared inputs that raw inputs stand for."""

    def l2norm(x):
        x = x.float()
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    heads, dim = g.shape[-2:]
    rate = A_log.exp().view(heads, 1)
    x = g.float() + dt_bias.view(heads, dim)
    if lower_bound is not None:
        g = lower_bound * torch.sigmoid(rate * x)
    else:
        g = -rate * torch.nn.functional.softplus(x)
    return l2norm(q), l2norm(k), g, beta.float().sigmoid()


@requires_prefill
@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["lower-bound", "softplus"])
def test_kda_paged_prefill_raw_inputs(lower_bound):
    torch.manual_seed(47)
    lengths, heads, scale = [17, 64, 300], 4, 128**-0.5
    q, k, v, g, beta, A_log, dt_bias = _raw_kda_inputs(lengths, heads)
    initial_state = 0.1 * torch.randn(
        len(lengths), heads, 128, 128, device=DEVICE, dtype=torch.float32
    )
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device=DEVICE, dtype=torch.int64
    )
    actual_output, actual_state = kda_paged_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=True,
    )

    prepared = _activate_kda_inputs(q, k, g, beta, A_log, dt_bias, lower_bound)
    expected_outputs = []
    expected_states = []
    for sequence, (begin, end) in enumerate(pairwise(cu_seqlens.tolist())):
        qn, kn, gn, bn = (x[0, begin:end] for x in prepared)
        expected_output, expected_state = _kda_recurrent_reference(
            qn, kn, v[0, begin:end], gn, bn, initial_state[sequence], scale
        )
        expected_outputs.append(expected_output)
        expected_states.append(expected_state)
    torch.testing.assert_close(
        actual_output[0].float(), torch.cat(expected_outputs), atol=8e-3, rtol=3e-2
    )
    torch.testing.assert_close(
        actual_state, torch.stack(expected_states), atol=8e-3, rtol=3e-2
    )


@requires_prefill
@pytest.mark.parametrize(
    "lengths", [[17, 100, 300], [17, 4000, 6000]], ids=["walk", "time-split"]
)
def test_kda_paged_prefill_state_pool(lengths):
    torch.manual_seed(59)
    heads, slots = 4, 9
    q, k, v, g, beta, A_log, dt_bias = _raw_kda_inputs(lengths, heads)
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device=DEVICE, dtype=torch.int64
    )
    kwargs = dict(
        scale=128**-0.5,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        beta_is_raw=True,
    )
    # Slots one apart in a doubled buffer: the pool's slot stride is not H * V * K.
    backing = 0.1 * torch.randn(
        slots, 2, heads, 128, 128, device=DEVICE, dtype=torch.float32
    )
    pool = backing[:, 1]
    indices = torch.randperm(slots, device=DEVICE)[: len(lengths)].int()
    original_pool = pool.clone()
    other_half = backing[:, 0].clone()

    expected_output, expected_state = kda_paged_prefill(
        q, k, v, g, beta, initial_state=pool[indices.long()].clone(), **kwargs
    )
    actual_output, actual_state = kda_paged_prefill(
        q, k, v, g, beta, state_pool=pool, state_indices=indices, **kwargs
    )
    assert actual_state is pool
    torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(pool[indices.long()], expected_state, atol=0, rtol=0)
    untouched = torch.ones(slots, dtype=torch.bool, device=DEVICE)
    untouched[indices.long()] = False
    torch.testing.assert_close(
        pool[untouched], original_pool[untouched], atol=0, rtol=0
    )
    torch.testing.assert_close(backing[:, 0], other_half, atol=0, rtol=0)


@requires_prefill
@pytest.mark.parametrize(
    "total_tokens,sequences,heads,key_dim,value_dim,dtype_name", PREFILL_SHAPES
)
def test_kda_paged_prefill_shape_suites_run(
    total_tokens, sequences, heads, key_dim, value_dim, dtype_name
):
    assert dtype_name == "bf16"
    torch.manual_seed(37)
    shape = (1, total_tokens, heads, key_dim)
    q = _normalized_kda_input(shape)
    k = _normalized_kda_input(shape)
    v = torch.randn(
        1, total_tokens, heads, value_dim, device=DEVICE, dtype=torch.bfloat16
    )
    g = -torch.nn.functional.softplus(
        torch.randn(shape, device=DEVICE, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, total_tokens, heads, device=DEVICE, dtype=torch.float32)
    )
    initial_state = torch.zeros(
        sequences, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32
    )
    base, remainder = divmod(total_tokens, sequences)
    lengths = [base + (index < remainder) for index in range(sequences)]
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device=DEVICE, dtype=torch.int64
    )

    output, final_state = kda_paged_prefill(
        q, k, v, g, beta, scale=1.0, initial_state=initial_state, cu_seqlens=cu_seqlens
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def _make_kda_decode_inputs(batch, heads, key_dim, value_dim, strided):
    q_shape = (1, batch, heads, key_dim)
    if strided:
        packed = torch.empty(
            1,
            batch,
            heads * (2 * key_dim + value_dim) + 7,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        q_end = heads * key_dim
        k_end = 2 * q_end
        v_end = k_end + heads * value_dim
        q = packed[..., :q_end].view(q_shape)
        k = packed[..., q_end:k_end].view(q_shape)
        v = packed[..., k_end:v_end].view(1, batch, heads, value_dim)
        q.copy_(_normalized_kda_input(q_shape))
        k.copy_(_normalized_kda_input(q_shape))
        v.normal_()
        gate_storage = torch.empty(
            1, batch, heads * key_dim + 5, device=DEVICE, dtype=torch.float32
        )
        g = gate_storage[..., :q_end].view(q_shape)
        g.copy_(-torch.nn.functional.softplus(torch.randn_like(g)))
        beta_storage = torch.empty(
            1, batch, heads + 3, device=DEVICE, dtype=torch.float32
        )
        beta = beta_storage[..., :heads]
        beta.copy_(torch.sigmoid(torch.randn_like(beta)))
    else:
        q = _normalized_kda_input(q_shape)
        k = _normalized_kda_input(q_shape)
        v = torch.randn(1, batch, heads, value_dim, device=DEVICE, dtype=torch.bfloat16)
        g = -torch.nn.functional.softplus(
            torch.randn(q_shape, device=DEVICE, dtype=torch.float32)
        )
        beta = torch.sigmoid(
            torch.randn(1, batch, heads, device=DEVICE, dtype=torch.float32)
        )
    return q, k, v, g, beta


@pytest.mark.parametrize("batch,heads,key_dim,value_dim,dtype_name", DECODE_SHAPES)
def test_kda_recurrent_decode_shape_suites(
    batch, heads, key_dim, value_dim, dtype_name
):
    assert dtype_name == "bf16"
    torch.manual_seed(43)
    q, k, v, g, beta = _make_kda_decode_inputs(
        batch, heads, key_dim, value_dim, strided=False
    )
    state_pool = torch.randn(
        2 * batch, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32
    )
    original_pool = state_pool.clone()
    read_indices = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    write_indices = read_indices + batch
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int64)

    expected = []
    expected_pool = state_pool.clone()
    for row in range(batch):
        row_output, row_state = _kda_recurrent_reference(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            g[0, row : row + 1],
            beta[0, row : row + 1],
            original_pool[row],
            1.0,
        )
        expected.append(row_output[0])
        expected_pool[batch + row] = row_state

    actual = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(
        actual.float(), torch.stack(expected).unsqueeze(0), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    ("heads", "key_dim", "value_dim", "strided_inputs", "scale_mode"),
    [
        pytest.param(2, 1, 1, False, "unit", id="minimal"),
        pytest.param(2, 3, 5, True, "attention", id="masked-strided"),
        pytest.param(2, 8, 5, False, "unit", id="mixed-dim"),
        pytest.param(12, 128, 128, True, "attention", id="production"),
    ],
)
def test_kda_recurrent_decode_indexed_state(
    heads, key_dim, value_dim, strided_inputs, scale_mode
):
    torch.manual_seed(13)
    batch = 3
    scale = key_dim**-0.5 if scale_mode == "attention" else 1.0
    q, k, v, g, beta = _make_kda_decode_inputs(
        batch, heads, key_dim, value_dim, strided_inputs
    )
    state_pool = torch.randn(
        7, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32
    )
    original_pool = state_pool.clone()
    expected_pool = state_pool.clone()
    read_indices = torch.tensor([0, 1, 1], device=DEVICE, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, 4], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int64)

    expected_output = []
    for row in range(batch):
        output, final_state = _kda_recurrent_reference(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            g[0, row : row + 1],
            beta[0, row : row + 1],
            original_pool[read_indices[row].long()],
            scale,
        )
        expected_output.append(output[0])
        expected_pool[write_indices[row].long()] = final_state
    expected_output = torch.stack(expected_output).unsqueeze(0)
    actual_output = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        actual_output.float(), expected_output, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(state_pool[0], original_pool[0], atol=0, rtol=0)
    torch.testing.assert_close(state_pool[1], original_pool[1], atol=0, rtol=0)


@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["lower-bound", "softplus"])
def test_kda_recurrent_decode_raw_inputs(lower_bound):
    torch.manual_seed(61)
    batch, heads, slots, scale = 5, 12, 9, 128**-0.5
    q, k, v, g, beta, A_log, dt_bias = _raw_kda_inputs([1] * batch, heads)
    state_pool = 0.1 * torch.randn(
        slots, heads, 128, 128, device=DEVICE, dtype=torch.float32
    )
    original_pool = state_pool.clone()
    indices = torch.randperm(slots, device=DEVICE)[:batch].int()
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int32)
    actual = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        state_pool=state_pool,
        read_indices=indices,
        write_indices=indices,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=True,
    )

    qn, kn, gn, bn = _activate_kda_inputs(q, k, g, beta, A_log, dt_bias, lower_bound)
    expected_pool = original_pool.clone()
    expected = []
    for row in range(batch):
        slot = indices[row].long()
        output, expected_pool[slot] = _kda_recurrent_reference(
            qn[0, row : row + 1],
            kn[0, row : row + 1],
            v[0, row : row + 1],
            gn[0, row : row + 1],
            bn[0, row : row + 1],
            original_pool[slot],
            scale,
        )
        expected.append(output[0])
    torch.testing.assert_close(
        actual.float(), torch.stack(expected).unsqueeze(0), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=1e-5, rtol=1e-5)


def test_kda_recurrent_decode_graph_padding_and_slot_stride():
    torch.manual_seed(23)
    batch, active, heads, key_dim, value_dim = 4, 2, 2, 8, 5
    state_elements = heads * key_dim * value_dim
    raw_pool = torch.randn(7, state_elements + 11, device=DEVICE, dtype=torch.float32)
    padding_before = raw_pool[:, state_elements:].clone()
    state_pool = raw_pool[:, :state_elements].view(7, heads, value_dim, key_dim)
    original_pool = state_pool.clone()
    q, k, v, g, beta = _make_kda_decode_inputs(
        batch, heads, key_dim, value_dim, strided=True
    )
    read_indices = torch.tensor([1, 2, -1, -1], device=DEVICE, dtype=torch.int32)
    write_indices = torch.tensor([3, 4, -1, -1], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device=DEVICE, dtype=torch.int32)
    kwargs = dict(
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
    )
    kda_recurrent_decode(q, k, v, g, beta, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_recurrent_decode(q, k, v, g, beta, **kwargs)
    graph.replay()
    torch.cuda.synchronize()

    expected_pool = original_pool.clone()
    expected_output = []
    for row in range(active):
        row_output, final_state = _kda_recurrent_reference(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            g[0, row : row + 1],
            beta[0, row : row + 1],
            original_pool[read_indices[row].long()],
            1.0,
        )
        expected_output.append(row_output[0])
        expected_pool[write_indices[row].long()] = final_state
    torch.testing.assert_close(
        captured[:, :active].float(),
        torch.stack(expected_output).unsqueeze(0),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        captured[:, active:], torch.zeros_like(captured[:, active:])
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        raw_pool[:, state_elements:], padding_before, atol=0, rtol=0
    )
    for untouched in (0, 1, 2, 5, 6):
        torch.testing.assert_close(
            state_pool[untouched], original_pool[untouched], atol=0, rtol=0
        )


@pytest.mark.parametrize(
    ("invalid_index", "scale"),
    [
        pytest.param(-1, 1.0, id="negative-unit-scale"),
        pytest.param(7, 8**-0.5, id="out-of-range-attention-scale"),
    ],
)
def test_kda_recurrent_decode_invalid_indices(scale, invalid_index):
    torch.manual_seed(41)
    batch, heads, key_dim, value_dim = 2, 2, 8, 5
    q, k, v, g, beta = _make_kda_decode_inputs(
        batch, heads, key_dim, value_dim, strided=False
    )
    pool = torch.randn(7, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32)
    original_pool = pool.clone()
    reads = torch.tensor([invalid_index, 0], device=DEVICE, dtype=torch.int32)
    writes = torch.tensor([5, invalid_index], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int32)

    expected_output = [torch.zeros_like(v[0, 0], dtype=torch.float32)]
    output1, _ = _kda_recurrent_reference(
        q[0, 1:2],
        k[0, 1:2],
        v[0, 1:2],
        g[0, 1:2],
        beta[0, 1:2],
        original_pool[0],
        scale,
    )
    expected_output.append(output1[0])
    actual = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        state_pool=pool,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(
        actual.float(), torch.stack(expected_output).unsqueeze(0), atol=2e-2, rtol=2e-2
    )
    for untouched in range(pool.shape[0]):
        torch.testing.assert_close(
            pool[untouched], original_pool[untouched], atol=0, rtol=0
        )


def test_kda_recurrent_decode_cpu_metadata_and_malformed_rows():
    torch.manual_seed(53)
    batch, heads, key_dim, value_dim = 3, 2, 8, 5
    q, k, v, g, beta = _make_kda_decode_inputs(
        batch, heads, key_dim, value_dim, strided=False
    )
    pool = torch.randn(6, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32)
    original_pool = pool.clone()
    expected_pool = pool.clone()
    reads = torch.tensor([0, 1, 2], dtype=torch.int32)
    writes = torch.tensor([3, 4, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3, 3], dtype=torch.int64)

    expected_output, expected_state = _kda_recurrent_reference(
        q[0, 2:3], k[0, 2:3], v[0, 2:3], g[0, 2:3], beta[0, 2:3], original_pool[1], 1.0
    )
    expected_pool[4] = expected_state
    actual = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        state_pool=pool,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]))
    torch.testing.assert_close(
        actual[:, 1].float(), expected_output, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(actual[:, 2], torch.zeros_like(actual[:, 2]))
    torch.testing.assert_close(pool, expected_pool, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
