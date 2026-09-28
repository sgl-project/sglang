import math

import pytest
import torch

import sgl_kernel  # noqa: F401

CHUNK_SIZE = 64
NUM_HEADS = 16


def _activate_gate(
    raw_gate: torch.Tensor,
    *,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    if a_log is None:
        return raw_gate.float()
    gate_input = raw_gate.float() + dt_bias.float().view(1, 1, raw_gate.shape[2], raw_gate.shape[3])
    gate_scale = torch.exp(a_log.float()).view(1, 1, raw_gate.shape[2], 1)
    if lower_bound is None:
        return -gate_scale * torch.nn.functional.softplus(gate_input)
    return lower_bound * torch.sigmoid(gate_scale * gate_input)


def _make_case(
    *,
    lens: tuple[int, ...],
    dtype: torch.dtype,
    head_dim: int,
    lower_bound: float | None,
    beta_is_raw: bool,
    use_pre_activated_gate: bool,
):
    total = sum(lens)
    num_slots = len(lens) + 2
    generator = torch.Generator(device="cpu").manual_seed(
        1000 + total + head_dim + (17 if dtype is torch.float16 else 3)
    )

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale

    q = randn(1, total, NUM_HEADS, head_dim).to(dtype)
    k = randn(1, total, NUM_HEADS, head_dim).to(dtype)
    v = randn(1, total, NUM_HEADS, head_dim, scale=0.1).to(dtype)
    raw_gate = randn(1, total, NUM_HEADS, head_dim, scale=0.35).to(dtype)
    beta_raw = randn(1, total, NUM_HEADS, scale=0.25).to(dtype)
    beta = beta_raw if beta_is_raw else torch.sigmoid(beta_raw.float()).to(dtype)
    initial_state = randn(num_slots, NUM_HEADS, head_dim, head_dim, scale=0.05).to(torch.float32)
    cu = torch.tensor([0, *torch.tensor(lens, dtype=torch.int32).cumsum(0).tolist()], dtype=torch.int32)
    state_indices = torch.tensor(list(range(1, len(lens) + 1)), dtype=torch.int32)
    track_chunk_idx = torch.tensor(
        [0 if seq_len < CHUNK_SIZE else 1 if seq_len > CHUNK_SIZE else -1 for seq_len in lens],
        dtype=torch.int32,
    )
    track_state = torch.full(
        (len(lens), NUM_HEADS, head_dim, head_dim), float("nan"), dtype=torch.float32
    )

    a_log = randn(NUM_HEADS, scale=0.2) - 1.2
    dt_bias = randn(NUM_HEADS * head_dim, scale=0.1)
    if use_pre_activated_gate:
        gate = _activate_gate(
            raw_gate, a_log=a_log, dt_bias=dt_bias, lower_bound=lower_bound
        ).to(dtype)
        a_log = None
        dt_bias = None
    else:
        gate = raw_gate

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": gate,
        "raw_gate": raw_gate,
        "beta": beta,
        "beta_raw": beta_raw,
        "initial_state": initial_state,
        "cu_seqlens": cu,
        "initial_state_indices": state_indices,
        "track_state": track_state,
        "track_chunk_idx": track_chunk_idx,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "lens": lens,
        "head_dim": head_dim,
        "lower_bound": lower_bound,
        "beta_is_raw": beta_is_raw,
    }


def _reference(case: dict):
    q = case["q"].float().clone()
    k = case["k"].float().clone()
    v = case["v"].float().clone()
    beta = case["beta_raw"].float().sigmoid() if case["beta_is_raw"] else case["beta"].float()
    gate = _activate_gate(
        case["raw_gate"] if case["A_log"] is not None else case["g"],
        a_log=case["A_log"],
        dt_bias=case["dt_bias"],
        lower_bound=case["lower_bound"],
    )

    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q = q * case.get("scale", case["head_dim"] ** -0.5)

    state = case["initial_state"].clone()
    output = torch.empty_like(v)
    num_chunks = sum((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE for seq_len in case["lens"])
    h = torch.empty(
        1,
        num_chunks,
        NUM_HEADS,
        case["head_dim"],
        case["head_dim"],
        dtype=case["q"].dtype,
    )
    track = torch.full_like(case["track_state"], float("nan"))

    token_start = 0
    chunk_row = 0
    for seq_idx, seq_len in enumerate(case["lens"]):
        slot = int(case["initial_state_indices"][seq_idx])
        seq_state = state[slot].clone()
        for chunk_idx in range((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE):
            h[0, chunk_row] = seq_state.to(case["q"].dtype)
            if int(case["track_chunk_idx"][seq_idx]) == chunk_idx:
                track[seq_idx] = seq_state
            chunk_row += 1
            for offset in range(chunk_idx * CHUNK_SIZE, min(seq_len, (chunk_idx + 1) * CHUNK_SIZE)):
                token = token_start + offset
                for head in range(NUM_HEADS):
                    seq_state[head] = seq_state[head] * gate[0, token, head].exp().unsqueeze(0)
                    residual = v[0, token, head] - torch.sum(
                        seq_state[head] * k[0, token, head].unsqueeze(0), dim=1
                    )
                    residual = residual * beta[0, token, head]
                    seq_state[head] = seq_state[head] + residual.unsqueeze(1) * k[0, token, head].unsqueeze(0)
                    output[0, token, head] = torch.sum(
                        seq_state[head] * q[0, token, head].unsqueeze(0), dim=1
                    )
        state[slot] = seq_state
        token_start += seq_len

    return output.to(case["q"].dtype), state, h, track


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "lens,head_dim,lower_bound,beta_is_raw,use_pre_activated_gate",
    [
        ((63,), 128, None, False, False),
        ((64,), 128, -5.0, False, False),
        ((65,), 128, None, True, False),
        ((257,), 128, None, False, False),
        ((128, 384, 512), 128, -5.0, False, False),
        ((128,), 64, None, False, True),
    ],
)
def test_chunk_kda_cpu_matches_reference(
    dtype, lens, head_dim, lower_bound, beta_is_raw, use_pre_activated_gate
):
    case = _make_case(
        lens=lens,
        dtype=dtype,
        head_dim=head_dim,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        use_pre_activated_gate=use_pre_activated_gate,
    )
    initial_state_before = case["initial_state"].clone()
    expected_out, expected_state, expected_h, expected_track = _reference(case)

    output, final_state, h = torch.ops.sgl_kernel.chunk_kda_cpu(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["initial_state"],
        case["cu_seqlens"],
        case["initial_state_indices"],
        case["A_log"],
        case["dt_bias"],
        case["lower_bound"],
        case["beta_is_raw"],
        True,
        True,
        case["track_state"],
        case["track_chunk_idx"],
    )

    atol = 3e-2 if dtype is torch.float16 else 2e-2
    rtol = 3e-2 if dtype is torch.float16 else 2e-2
    torch.testing.assert_close(output.float(), expected_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(final_state, expected_state, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(case["initial_state"], expected_state, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(h.float(), expected_h.float(), atol=atol, rtol=rtol)

    tracked_mask = case["track_chunk_idx"] >= 0
    if tracked_mask.any():
        torch.testing.assert_close(
            case["track_state"][tracked_mask],
            expected_track[tracked_mask],
            atol=4e-2,
            rtol=4e-2,
        )
    if (~tracked_mask).any():
        assert torch.isnan(case["track_state"][~tracked_mask]).all()

    touched = set(case["initial_state_indices"].tolist())
    untouched = [idx for idx in range(initial_state_before.shape[0]) if idx not in touched]
    if untouched:
        assert torch.equal(case["initial_state"][untouched], initial_state_before[untouched])

    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(final_state).all()
    assert h.shape == expected_h.shape


def test_chunk_kda_cpu_wrapper_matches_extend_contract():
    from sgl_kernel.mamba import chunk_kda_cpu

    case = _make_case(
        lens=(65, 128),
        dtype=torch.bfloat16,
        head_dim=128,
        lower_bound=-5.0,
        beta_is_raw=False,
        use_pre_activated_gate=False,
    )
    case["scale"] = 0.25
    result = chunk_kda_cpu(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        scale=case["scale"],
        initial_state=case["initial_state"],
        initial_state_indices=case["initial_state_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=case["cu_seqlens"],
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        lower_bound=case["lower_bound"],
        output_intermediate_states=True,
        track_state=case["track_state"],
        track_chunk_idx=case["track_chunk_idx"],
    )
    output, h = result
    expected_out, _, _, _ = _reference(case)
    torch.testing.assert_close(output.float(), expected_out.float(), atol=2e-2, rtol=2e-2)
    assert output.shape == case["v"].shape
    assert h.ndim == 5
    assert h.shape[1] == sum(math.ceil(seq_len / CHUNK_SIZE) for seq_len in case["lens"])
