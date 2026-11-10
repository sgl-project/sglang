"""SM70 KDA recurrence (kda_sm70_recurrent), the fp16 linear-attention path on V100.

Prefix-cache resume, chunked prefill and MTP verify rollback all treat the fp32
state the kernel leaves in the pool as exact: a sequence split into two calls
must give bitwise the same outputs and state as one call, and the per-step
verify states must equal the state of the matching prefix run.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The SM70 KDA kernel requires an NVIDIA V100",
)

Q_HEADS = 4
V_HEADS = 8
SCALE = 128**-0.5


def _inputs(tokens: int, seed: int):
    torch.manual_seed(seed)
    half = dict(device="cuda", dtype=torch.float16)
    return dict(
        q=torch.randn(tokens, Q_HEADS, 128, **half),
        k=torch.randn(tokens, Q_HEADS, 128, **half),
        v=torch.randn(tokens, V_HEADS, 128, **half),
        a=torch.randn(tokens, V_HEADS, 128, **half),
        b=torch.randn(tokens, V_HEADS, **half),
        A_log=torch.randn(V_HEADS, device="cuda") * 0.5,
        dt_bias=torch.randn(V_HEADS, 128, device="cuda") * 0.1,
    )


def _slice(inputs, start: int, end: int):
    per_token = ("q", "k", "v", "a", "b")
    return {
        name: t[start:end] if name in per_token else t for name, t in inputs.items()
    }


def _i32(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _run(inputs, state, indices, cu, **kwargs):
    from sglang.kernels.ops.attention.kda_sm70 import kda_sm70_recurrent

    return kda_sm70_recurrent(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        state,
        _i32(indices),
        _i32(cu),
        scale=SCALE,
        **kwargs,
    )[0]


def _reference(inputs, state, indices, cu, lower_bound):
    """fp64 KDA over packed sequences; returns (out [T, HV, 128], final state)."""
    q, k, v, a, b = (inputs[n].double() for n in ("q", "k", "v", "a", "b"))
    A_log, dt_bias = inputs["A_log"].double(), inputs["dt_bias"].double()
    out_state = state.double().clone()
    out = torch.zeros(q.shape[0], V_HEADS, 128, dtype=torch.float64, device="cuda")
    group = V_HEADS // Q_HEADS
    for seq, slot in enumerate(indices):
        h = out_state[slot].clone() if slot >= 0 else torch.zeros_like(out_state[0])
        for tok in range(cu[seq], cu[seq + 1]):
            k_heads = torch.arange(V_HEADS, device="cuda") // group
            qn = torch.nn.functional.normalize(q[tok, k_heads], dim=-1, eps=0) * SCALE
            kn = torch.nn.functional.normalize(k[tok, k_heads], dim=-1, eps=0)
            x = a[tok] + dt_bias
            exp_a = torch.exp(A_log)[:, None]
            if lower_bound is None:
                gate = -exp_a * torch.nn.functional.softplus(x)
            else:
                gate = lower_bound * torch.sigmoid(exp_a * x)
            h = h * torch.exp(gate)[:, None, :]
            v_new = (v[tok] - torch.einsum("hvk,hk->hv", h, kn)) * torch.sigmoid(
                b[tok]
            )[:, None]
            h = h + v_new[:, :, None] * kn[:, None, :]
            out[tok] = torch.einsum("hvk,hk->hv", h, qn)
        if slot >= 0:
            out_state[slot] = h
    return out, out_state


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_packed_batch_matches_reference(lower_bound):
    """An empty sequence, a padded slot and a 64-token snapshot in one launch."""
    inputs = _inputs(74, seed=0)
    state = torch.randn(4, V_HEADS, 128, 128, device="cuda") * 0.05
    indices = [2, 3, -1, 0]
    cu = [0, 3, 3, 4, 74]
    ref_out, ref_state = _reference(inputs, state, indices, cu, lower_bound)
    _, ref_chunk = _reference(_slice(inputs, 4, 68), state, [0], [0, 64], lower_bound)

    # The pool is a view behind a guard slot, so a write through index -1 shows.
    ran_pool = torch.cat([torch.full_like(state[:1], 7.0), state])
    ran = ran_pool[1:]
    track = torch.zeros(4, V_HEADS, 128, 128, device="cuda")
    out = _run(
        inputs,
        ran,
        indices,
        cu,
        lower_bound=lower_bound,
        track_state=track,
        track_chunk_idx=_i32([0, 0, 0, 1]),
    )

    tol = dict(atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out[:3].double(), ref_out[:3], **tol)
    torch.testing.assert_close(out[4:].double(), ref_out[4:], **tol)
    for slot in (0, 2):
        torch.testing.assert_close(ran[slot].double(), ref_state[slot], **tol)
    # The empty sequence keeps its slot; a chunk-0 snapshot is the input state.
    assert torch.equal(ran[3], state[3])
    assert torch.equal(track[0], state[2])
    torch.testing.assert_close(track[3].double(), ref_chunk[0], **tol)
    # The padded sequence never touches the pool or another sequence's slot.
    assert torch.equal(ran[1], state[1])
    assert (ran_pool[0] == 7.0).all()


def test_fused_projection_slices_match_contiguous():
    """q/k/v/a/b arrive as column slices of one projection output, no copy."""
    inputs = _inputs(70, seed=4)
    names = ("q", "k", "v", "a", "b")
    widths = [inputs[n][0].numel() for n in names]
    fused = torch.cat([inputs[n].reshape(70, -1) for n in names], dim=1)
    strided = dict(inputs)
    for name, piece in zip(names, fused.split(widths, dim=1)):
        strided[name] = piece.view(inputs[name].shape)
    assert not strided["v"].is_contiguous()

    state = torch.randn(2, V_HEADS, 128, 128, device="cuda") * 0.05
    dense_state, strided_state = state.clone(), state.clone()
    out_dense = _run(inputs, dense_state, [1, 0], [0, 30, 70])
    out_strided = _run(strided, strided_state, [1, 0], [0, 30, 70])
    assert torch.equal(out_strided, out_dense)
    assert torch.equal(strided_state, dense_state)


def test_split_run_is_bitwise_continuous():
    inputs = _inputs(130, seed=1)
    state = torch.randn(3, V_HEADS, 128, 128, device="cuda") * 0.05

    whole = state.clone()
    out_whole = _run(inputs, whole, [1], [0, 130])

    split = state.clone()
    out_head = _run(_slice(inputs, 0, 64), split, [1], [0, 64])
    out_tail = _run(_slice(inputs, 64, 130), split, [1], [0, 66])
    assert torch.equal(torch.cat([out_head, out_tail]), out_whole)
    assert torch.equal(split, whole)

    # A neighbour in the same launch changes neither output nor state.
    packed = state.clone()
    neighbour = _inputs(130, seed=2)
    both = {
        name: torch.cat([inputs[name], neighbour[name]])
        if name in ("q", "k", "v", "a", "b")
        else inputs[name]
        for name in inputs
    }
    out_packed = _run(both, packed, [1, 2], [0, 130, 260])
    assert torch.equal(out_packed[:130], out_whole)
    assert torch.equal(packed[1], whole[1])


def test_verify_states_equal_prefix_runs():
    """MTP verify stores the state after each draft step and must not commit."""
    steps = 4
    inputs = _inputs(steps, seed=3)
    state = torch.randn(3, V_HEADS, 128, 128, device="cuda") * 0.05
    inter = torch.zeros(2, steps, V_HEADS, 128, 128, device="cuda")
    ran = state.clone()
    _run(
        inputs,
        ran,
        [2],
        [0, steps],
        commit_state=False,
        intermediate_states=inter,
        intermediate_state_indices=_i32([1]),
        cache_steps=steps,
    )
    assert torch.equal(ran, state)
    assert not inter[0].any()
    for step in range(steps):
        prefix = state.clone()
        _run(_slice(inputs, 0, step + 1), prefix, [2], [0, step + 1])
        assert torch.equal(inter[1, step], prefix[2]), f"verify step {step}"
