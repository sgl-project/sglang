"""Correctness tests for bounded-gate KDA ReplaySSM pre-decay decode."""

from array import array

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
    fused_recurrent_linear_replayssm_decode,
)
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    _advance_replayssm_predecay_cursor,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_predecay_cursor_snapshot_advance_and_padding():
    persistent_wp = torch.tensor([0, 2, 3, 0], device="cuda", dtype=torch.int32)
    persistent_base = torch.tensor([0, 1, 2, 0], device="cuda", dtype=torch.int32)
    static_wp = torch.zeros(3, device="cuda", dtype=torch.int32)
    static_base = torch.zeros_like(static_wp)
    slots = torch.tensor([1, 2, -1], device="cuda", dtype=torch.int32)
    force_flush = torch.tensor([1, 0, 0], device="cuda", dtype=torch.int32)
    _advance_replayssm_predecay_cursor(
        persistent_write_pos=persistent_wp,
        persistent_cache_base=persistent_base,
        static_write_pos=static_wp,
        static_cache_base=static_base,
        state_indices=slots,
        cache_len=4,
        force_flush=force_flush,
    )
    torch.cuda.synchronize()
    assert static_wp.tolist() == [2, 3, 0]
    assert static_base.tolist() == [1, 2, 0]
    assert persistent_wp.tolist() == [0, 0, 1, 0]
    assert persistent_base.tolist() == [0, 0, 1, 0]


def test_predecay_fresh_slot_resets_independent_cursor_fields():
    shape = KimiLinearStateShape.create(
        tp_world_size=1,
        num_heads=4,
        head_dim=8,
        num_k_heads=4,
        head_k_dim=8,
        gate_lower_bound=-5.0,
    )
    cache_params = KimiLinearCacheParams(
        shape=shape,
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )
    pool = HybridReqToTokenPool(
        size=2,
        mamba_size=2,
        mamba_spec_state_size=2,
        max_context_len=16,
        device="cuda",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=False,
        enable_linear_replayssm=True,
        enable_kda_replayssm_predecay=True,
        linear_replayssm_cache_len=4,
    )
    assert pool.mamba_pool.replayssm_cache_base is not None
    assert pool.mamba_pool.replayssm_is_flush is None
    req = Req(
        rid="predecay-slot",
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )

    pool.alloc([req])
    slot = req.mamba_pool_idx
    assert pool.mamba_pool.replayssm_write_pos[slot].item() == 0
    assert pool.mamba_pool.replayssm_cache_base[slot].item() == 0
    pool.mamba_pool.replayssm_write_pos[slot] = 2
    pool.mamba_pool.replayssm_cache_base[slot] = 3
    pool.mamba_pool.clear_slots(torch.tensor([slot], device="cuda"))
    assert pool.mamba_pool.replayssm_write_pos[slot].item() == 0
    assert pool.mamba_pool.replayssm_cache_base[slot].item() == 0


def test_predecay_accepts_page_major_slot_stride():
    batch, heads, dim, cache_len = 2, 4, 32, 4
    replay = _allocate(batch, heads, dim, cache_len, torch.bfloat16)
    backing = torch.zeros(batch, 2, heads, dim, dim, device="cuda", dtype=torch.float32)
    replay["state"] = backing[:, 0]
    assert not replay["state"].is_contiguous()
    assert replay["state"].stride()[1:] == (dim * dim, dim, 1)
    mixed, a, b = _inputs(batch, heads, dim, torch.bfloat16, 7000)
    out = _decode(
        replay,
        mixed,
        a,
        b,
        torch.zeros(heads, device="cuda"),
        torch.zeros(heads, dim, device="cuda"),
        torch.zeros(batch, device="cuda", dtype=torch.int32),
        torch.zeros(batch, device="cuda", dtype=torch.int32),
    )
    assert torch.isfinite(out).all()


def _inputs(batch, heads, dim, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, heads, dim, generator=generator, device="cuda", dtype=dtype)
    k = torch.randn(batch, heads, dim, generator=generator, device="cuda", dtype=dtype)
    v = torch.randn(batch, heads, dim, generator=generator, device="cuda", dtype=dtype)
    a = (
        torch.randn(batch, heads, dim, generator=generator, device="cuda", dtype=dtype)
        * 0.25
    ).contiguous()
    b = torch.randn(batch, heads, generator=generator, device="cuda", dtype=dtype)
    return torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1), a, b


def _reference_step(mixed_qkv, a, b, a_log, dt_bias, state, lower_bound):
    batch, heads, value_dim, key_dim = state.shape
    width = heads * key_dim
    q, k, v = mixed_qkv.split((width, width, heads * value_dim), dim=-1)
    q = q.reshape(batch, heads, key_dim).float()
    k = k.reshape(batch, heads, key_dim).float()
    v = v.reshape(batch, heads, value_dim).float()
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q = q * (key_dim**-0.5)
    gate = lower_bound * torch.sigmoid(
        torch.exp(a_log)[None, :, None] * (a.float() + dt_bias[None])
    )
    alpha = torch.exp(gate)
    beta = torch.sigmoid(b.float())
    decayed = state * alpha.unsqueeze(-2)
    delta = beta.unsqueeze(-1) * (v - torch.einsum("bhvk,bhk->bhv", decayed, k))
    state.copy_(decayed + delta.unsqueeze(-1) * k.unsqueeze(-2))
    return torch.einsum("bhvk,bhk->bhv", state, q)


def _allocate(batch, heads, dim, cache_len, ring_dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260821)
    return {
        "state": torch.randn(
            batch,
            heads,
            dim,
            dim,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.03,
        "d": torch.zeros(batch, heads, cache_len, dim, device="cuda", dtype=ring_dtype),
        "k": torch.zeros(batch, heads, cache_len, dim, device="cuda", dtype=ring_dtype),
        "g": torch.zeros(
            batch, heads, cache_len, dim, device="cuda", dtype=torch.float32
        ),
        "base": torch.zeros(batch, device="cuda", dtype=torch.int32),
    }


def _decode(replay, mixed_qkv, a, b, a_log, dt_bias, write_pos, force_flush):
    batch, heads, dim = a.shape
    out = mixed_qkv.new_empty(batch, 1, heads, dim)
    fused_recurrent_linear_replayssm_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=dim**-0.5,
        initial_state=replay["state"],
        d_cache=replay["d"],
        k_cache=replay["k"],
        g_cache=replay["g"],
        out=out,
        ssm_state_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
        write_pos=write_pos,
        cache_base=replay["base"],
        force_flush=force_flush,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        block_v=16 if batch == 1 else 32,
        num_warps=2 if batch == 1 else 1,
        num_stages=2,
        nk=1,
        circular_replay=True,
        prefix_gate_cache=True,
        predecayed_k_cache=True,
    )
    return out


def _advance(replay, write_pos, force_flush):
    natural = write_pos == replay["d"].shape[2] - 1
    forced = force_flush.bool()
    history_flush = natural & ~forced
    replay["base"].copy_(
        torch.where(
            forced,
            torch.zeros_like(replay["base"]),
            torch.where(
                history_flush,
                (replay["base"] + write_pos) % replay["d"].shape[2],
                replay["base"],
            ),
        )
    )
    return torch.where(
        forced,
        torch.zeros_like(write_pos),
        torch.where(history_flush, torch.ones_like(write_pos), write_pos + 1),
    )


@pytest.mark.parametrize("cache_len", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("ring_dtype", [torch.bfloat16, torch.float32])
def test_predecay_matches_recurrence_across_wraps(cache_len, ring_dtype):
    batch, heads, dim = 3, 4, 32
    replay = _allocate(batch, heads, dim, cache_len, ring_dtype)
    reference_state = replay["state"].clone()
    cursor = torch.zeros(batch, device="cuda", dtype=torch.int32)
    a_log = (torch.randn(heads, device="cuda") * 0.2).float().contiguous()
    dt_bias = (torch.randn(heads, dim, device="cuda") * 0.1).float().contiguous()

    for step in range(3 * cache_len + 5):
        mixed_qkv, a, b = _inputs(batch, heads, dim, torch.bfloat16, 1000 + step)
        expected = _reference_step(
            mixed_qkv, a, b, a_log, dt_bias, reference_state, -5.0
        )
        write_pos = cursor.clone()
        force_flush = torch.zeros_like(write_pos)
        if step in (2, cache_len + 2):
            force_flush[1] = 1
        actual = _decode(
            replay, mixed_qkv, a, b, a_log, dt_bias, write_pos, force_flush
        )
        torch.testing.assert_close(
            actual[:, 0].float(),
            expected.to(torch.bfloat16).float(),
            atol=3e-2,
            rtol=3e-2,
        )
        if force_flush.any():
            mask = force_flush.bool()
            torch.testing.assert_close(
                replay["state"][mask],
                reference_state[mask],
                atol=5e-2,
                rtol=3e-2,
            )
        cursor = _advance(replay, write_pos, force_flush)


def test_predecay_cuda_graph_replay():
    batch, heads, dim, cache_len = 2, 4, 32, 4
    replay = _allocate(batch, heads, dim, cache_len, torch.bfloat16)
    initial_state = replay["state"].clone()
    reference_state = initial_state.clone()
    a_log = torch.zeros(heads, device="cuda")
    dt_bias = torch.zeros(heads, dim, device="cuda")
    mixed, a, b = _inputs(batch, heads, dim, torch.bfloat16, 8000)
    write_pos = torch.zeros(batch, device="cuda", dtype=torch.int32)
    force_flush = torch.zeros_like(write_pos)

    _decode(replay, mixed, a, b, a_log, dt_bias, write_pos, force_flush)
    replay["state"].copy_(initial_state)
    replay["d"].zero_()
    replay["k"].zero_()
    replay["g"].zero_()
    replay["base"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _decode(replay, mixed, a, b, a_log, dt_bias, write_pos, force_flush)
    replay["state"].copy_(initial_state)
    replay["d"].zero_()
    replay["k"].zero_()
    replay["g"].zero_()
    replay["base"].zero_()

    cursor = torch.zeros_like(write_pos)
    for step in range(9):
        new_mixed, new_a, new_b = _inputs(
            batch, heads, dim, torch.bfloat16, 8100 + step
        )
        expected = _reference_step(
            new_mixed, new_a, new_b, a_log, dt_bias, reference_state, -5.0
        )
        mixed.copy_(new_mixed)
        a.copy_(new_a)
        b.copy_(new_b)
        write_pos.copy_(cursor)
        force_flush.zero_()
        if step == 2:
            force_flush[0] = 1
        graph.replay()
        torch.testing.assert_close(
            out[:, 0].float(),
            expected.to(torch.bfloat16).float(),
            atol=3e-2,
            rtol=3e-2,
        )
        cursor = _advance(replay, cursor, force_flush)


@pytest.mark.parametrize(
    ("heads", "value_heads", "lower_bound", "match"),
    [
        (2, 4, -5.0, "one K head"),
        (4, 4, None, "finite negative"),
        (4, 4, -6.0, "inverse-prefix scaling is unsafe"),
    ],
)
def test_predecay_rejects_unsafe_configuration(heads, value_heads, lower_bound, match):
    batch, dim, cache_len = 2, 32, 16
    mixed, a, b = _inputs(batch, value_heads, dim, torch.bfloat16, 9000)
    if heads != value_heads:
        width = heads * dim
        mixed = torch.cat(
            (
                mixed[:, :width],
                mixed[:, value_heads * dim : value_heads * dim + width],
                mixed[:, 2 * value_heads * dim :],
            ),
            dim=-1,
        ).contiguous()
    state = torch.zeros(batch, value_heads, dim, dim, device="cuda")
    with pytest.raises(ValueError, match=match):
        fused_recurrent_linear_replayssm_decode(
            mixed_qkv=mixed,
            a=a,
            b=b,
            A_log=torch.zeros(value_heads, device="cuda"),
            dt_bias=torch.zeros(value_heads, dim, device="cuda"),
            scale=dim**-0.5,
            initial_state=state,
            d_cache=torch.zeros(
                batch, value_heads, cache_len, dim, device="cuda", dtype=torch.bfloat16
            ),
            k_cache=torch.zeros(
                batch, heads, cache_len, dim, device="cuda", dtype=torch.bfloat16
            ),
            g_cache=torch.zeros(batch, value_heads, cache_len, dim, device="cuda"),
            out=torch.empty(
                batch, 1, value_heads, dim, device="cuda", dtype=torch.bfloat16
            ),
            ssm_state_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
            write_pos=torch.zeros(batch, device="cuda", dtype=torch.int32),
            cache_base=torch.zeros(batch, device="cuda", dtype=torch.int32),
            lower_bound=lower_bound,
            is_kda=True,
            circular_replay=True,
            prefix_gate_cache=True,
            predecayed_k_cache=True,
        )
