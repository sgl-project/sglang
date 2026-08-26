"""Correctness tests for the FlashInfer SM100 KDA decode + MTP backend.

Compares ``FlashInferKDAKernel`` with the Triton KDA reference for decode output,
state updates, and topk=1 target_verify checkpoints. ``recurrent_kda`` is
SM100-only and requires a FlashInfer build that exposes it.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# SM100 suite, same slot as the CuteDSL KDA prefill test. Disabled until the
# pinned public FlashInfer dependency contains both KDA APIs exercised here.
register_cuda_ci(
    est_time=60,
    stage="base-b",
    runner_config="4-gpu-b200",
    disabled=(
        "recurrent_kda and packed_kda_decode are not both in the pinned public "
        "FlashInfer build"
    ),
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip(
        "FlashInfer KDA (recurrent_kda) requires CUDA SM10x (Blackwell).",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (  # noqa: E402
    CakeKDAKernel,
    FlashInferKDAKernel,
    _get_flashinfer_kda_kernel,
    _get_flashinfer_packed_kda_kernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)
from sglang.srt.mem_cache.allocator.mamba import (  # noqa: E402
    MambaSlotAllocator,
    _issue_state_index_contract,
)

_available, _ = _get_flashinfer_kda_kernel()
if not _available:
    pytest.skip(
        "FlashInfer build does not expose recurrent_kda (KDA decode).",
        allow_module_level=True,
    )

# KDA: head_k_dim == head_v_dim == 128; single q/v head group (HV == H) here.
H, HV, K, V = 16, 16, 128, 128
CAKE_ARCH_SUPPORTED = torch.cuda.get_device_capability() in ((10, 0), (10, 3))
CAKE_PACKED_AVAILABLE, _ = _get_flashinfer_packed_kda_kernel()


# ---------------------------------------------------------------------------
# Inputs (matched to the sglang KDA decode/verify contract: raw per-K gate `a`,
# beta logit `b`, SSM pool [N, HV, V, K], decode cu_seqlens = query_start_loc).
# ---------------------------------------------------------------------------
def _make_decode_inputs(
    batch_size,
    device="cuda",
    dtype=torch.bfloat16,
    num_heads=H,
    num_value_heads=HV,
):
    B, pool = batch_size, batch_size + 16
    return dict(
        B=B,
        q=(
            torch.randn(1, B, num_heads, K, device=device, dtype=dtype) * 0.5
        ).contiguous(),
        k=(
            torch.randn(1, B, num_heads, K, device=device, dtype=dtype) * 0.5
        ).contiguous(),
        v=(
            torch.randn(1, B, num_value_heads, V, device=device, dtype=dtype) * 0.5
        ).contiguous(),
        a=(
            torch.randn(B, num_value_heads * K, device=device, dtype=dtype) * 0.5 - 1.0
        ).contiguous(),
        b=(
            torch.randn(B, num_value_heads, device=device, dtype=dtype) * 0.5
        ).contiguous(),
        A_log=torch.randn(num_value_heads, device=device, dtype=torch.float32) * 0.2,
        dt_bias=(
            torch.randn(num_value_heads * K, device=device, dtype=torch.float32) * 0.1
        ),
        ssm=(
            torch.randn(pool, num_value_heads, V, K, device=device, dtype=dtype) * 0.01
        ).contiguous(),
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        qsl=torch.arange(B + 1, device=device, dtype=torch.int32),
    )


def _make_packed_decode_inputs(
    batch_size,
    device="cuda",
    dtype=torch.bfloat16,
    num_heads=12,
    num_value_heads=12,
):
    """Build the post-convolution serving layout consumed by packed decode."""
    data = _make_decode_inputs(
        batch_size,
        device=device,
        dtype=dtype,
        num_heads=num_heads,
        num_value_heads=num_value_heads,
    )
    q_width = num_heads * K
    k_width = num_heads * K
    v_width = num_value_heads * V
    packed_width = q_width + k_width + v_width
    mixed_storage = torch.randn(
        batch_size,
        packed_width + 64,
        device=device,
        dtype=dtype,
    )
    data["mixed_qkv"] = mixed_storage[:, :packed_width]

    # Production Kimi-K3 stores beta inside a 144-column projection backing
    # allocation.  Keep the interior offset and row pitch even for B=1, where
    # reshape/view would otherwise canonicalize the pitch to the logical width.
    beta_storage = (
        torch.randn(batch_size, 144, device=device, dtype=dtype) * 0.5
    ).contiguous()
    data["beta_storage"] = beta_storage
    data["b"] = beta_storage[:, 128 : 128 + num_value_heads]

    # Exercise the production pool contract: compact [HV, V, K] slots with a
    # legal non-compact outer stride supplied by the serving cache envelope.
    pool_size = data["ssm"].shape[0]
    compact_slot = num_value_heads * V * K
    slot_stride = compact_slot + 256
    state_storage = torch.randn(
        pool_size * slot_stride,
        device=device,
        dtype=dtype,
    )
    data["ssm"] = torch.as_strided(
        state_storage,
        (pool_size, num_value_heads, V, K),
        (slot_stride, V * K, K, 1),
    )
    data["index_allocator"] = MambaSlotAllocator(
        size=pool_size - 1,
        device=device,
    )
    data["allocated_cache_indices"] = (
        data["index_allocator"].alloc(pool_size - 1).to(torch.int32)
    )
    data["cache_indices"] = data["allocated_cache_indices"][:batch_size].clone()
    return data


def _clone_strided_state(state):
    clone = torch.empty_strided(
        state.shape,
        state.stride(),
        device=state.device,
        dtype=state.dtype,
    )
    clone.copy_(state)
    return clone


def _make_verify_inputs(
    batch_size,
    cache_steps,
    allocated_steps=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    B, T = batch_size, cache_steps
    S = allocated_steps or T
    assert S >= T
    seq, pool = B * T, B + 16
    return dict(
        B=B,
        T=T,
        allocated_steps=S,
        seq=seq,
        q=(torch.randn(1, seq, H, K, device=device, dtype=dtype) * 0.5).contiguous(),
        k=(torch.randn(1, seq, H, K, device=device, dtype=dtype) * 0.5).contiguous(),
        v=(torch.randn(1, seq, HV, V, device=device, dtype=dtype) * 0.5).contiguous(),
        a=(
            torch.randn(seq, HV * K, device=device, dtype=dtype) * 0.5 - 1.0
        ).contiguous(),
        b=(torch.randn(seq, HV, device=device, dtype=dtype) * 0.5).contiguous(),
        A_log=torch.randn(HV, device=device, dtype=torch.float32) * 0.2,
        dt_bias=torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1,
        ssm=(
            torch.randn(pool, HV, V, K, device=device, dtype=dtype) * 0.01
        ).contiguous(),
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        qsl=torch.arange(0, seq + 1, T, device=device, dtype=torch.int32),
        intermediate_states=torch.zeros(
            B, S, HV, V, K, device=device, dtype=dtype
        ).contiguous(),
        intermediate_indices=torch.arange(B, device=device, dtype=torch.int32),
    )


def _decode(kern, d, ssm, lower_bound=None):
    # `ssm` is updated in place (committed-pool decode step); pass a fresh clone.
    return kern.decode(
        d["q"],
        d["k"],
        d["v"],
        d["a"],
        d["b"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        ssm_states=ssm,
        cache_indices=d["cache_indices"],
        query_start_loc=d["qsl"],
        lower_bound=lower_bound,
    ).reshape(d["B"], d["v"].shape[2], d["v"].shape[3])


def _packed_decode(kern, d, ssm, lower_bound=-5.0, **kwargs):
    if isinstance(kern, CakeKDAKernel) and "cache_indices_cpu" not in kwargs:
        # This direct kernel harness has no MambaAttnBackendBase metadata
        # producer, so bind the same private attestation to allocator-produced
        # IDs explicitly. Production issuance occurs only in that producer.
        active_prefix = d.get("cache_index_active_prefix", d["B"])
        kwargs["cache_index_contract"] = _issue_state_index_contract(
            d["index_allocator"],
            d["cache_indices"],
            active_prefix=active_prefix,
            state_slots=ssm.shape[0],
            active_request_ids=tuple(
                f"direct-kernel-request-{i}" for i in range(active_prefix)
            ),
        )
    if getattr(kern, "supports_cake_route_telemetry", False):
        kwargs["layer_id"] = 7
    return kern.packed_decode(
        d["mixed_qkv"],
        d["a"].unsqueeze(1),
        d["b"].unsqueeze(0),
        A_log=d["A_log"].view(1, 1, -1, 1),
        dt_bias=d["dt_bias"],
        scale=K**-0.5,
        ssm_states=ssm,
        cache_indices=d["cache_indices"],
        num_v_heads=d["ssm"].shape[1],
        head_v_dim=d["ssm"].shape[2],
        lower_bound=lower_bound,
        **kwargs,
    ).reshape(d["B"], d["ssm"].shape[1], d["ssm"].shape[2])


def _verify(kern, d, ssm, intermediate_states):
    return kern.target_verify(
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        q=d["q"],
        k=d["k"],
        v=d["v"],
        a=d["a"],
        b=d["b"],
        ssm_states=ssm,
        cache_indices=d["cache_indices"],
        query_start_loc=d["qsl"],
        intermediate_states_buffer=intermediate_states,
        intermediate_state_indices=d["intermediate_indices"],
        cache_steps=d["T"],
        retrieve_parent_token=None,
    ).reshape(d["seq"], HV, V)


def _sequential_decode_states(kern, d):
    """Ground truth for verify checkpoints: single-token decode over each step."""
    B, T = d["B"], d["T"]
    st = d["ssm"].clone()  # committed pool [pool, HV, V, K], updated in place by decode
    ci = d["cache_indices"].long()
    qsl_dec = torch.arange(B + 1, device=st.device, dtype=torch.int32)
    ref = torch.zeros(B, T, HV, V, K, device=st.device, dtype=st.dtype)
    for t in range(T):
        pos = torch.arange(B, device=st.device) * T + t  # token t of each request
        kern.decode(
            d["q"][:, pos].contiguous(),
            d["k"][:, pos].contiguous(),
            d["v"][:, pos].contiguous(),
            d["a"][pos].contiguous(),
            d["b"][pos].contiguous(),
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            ssm_states=st,
            cache_indices=d["cache_indices"],
            query_start_loc=qsl_dec,
        )
        ref[:, t] = st[ci]  # post-token-t state for each request
    return ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 8, 64, 128])
def test_kda_decode_flashinfer_matches_triton(batch_size):
    """FlashInfer decode output + committed-pool state update match the Triton
    KDA decode reference."""
    torch.manual_seed(batch_size)
    d = _make_decode_inputs(batch_size)
    fi, tri = FlashInferKDAKernel(), TritonKDAKernel()

    st_ref = d["ssm"].clone()
    ref_out = _decode(tri, d, st_ref).float()
    st_fi = d["ssm"].clone()
    out = _decode(fi, d, st_fi).float()
    torch.cuda.synchronize()

    assert torch.isfinite(out).all(), "FlashInfer decode output has non-finite values"
    assert torch.isfinite(st_fi).all(), "FlashInfer decode state has non-finite values"

    o_err = (out - ref_out).abs()
    # bf16 recurrent step; B200 kernel-unit measured out max-abs-diff ~1e-4.
    assert o_err.max().item() < 1e-2, f"decode out max diff {o_err.max().item():.2e}"
    assert o_err.mean().item() < 1e-3, f"decode out mean diff {o_err.mean().item():.2e}"

    # Updated committed-pool slots (SSM state [HV, V, K]) must match too.
    idx = d["cache_indices"].long()
    s_err = (st_fi[idx].float() - st_ref[idx].float()).abs()
    assert s_err.max().item() < 1e-1, f"decode state max diff {s_err.max().item():.2e}"
    assert (
        s_err.mean().item() < 1e-2
    ), f"decode state mean diff {s_err.mean().item():.2e}"


@pytest.mark.parametrize("batch_size", [1, 8, 31, 32, 64, 128])
@pytest.mark.skipif(
    not CAKE_PACKED_AVAILABLE,
    reason="CAKE packed KDA decode requires its FlashInfer export on SM100/SM103.",
)
def test_kda_decode_cake_matches_triton_kimi_k3_h12(batch_size):
    """Exercise the exact packed Kimi-K3 TP8 serving contract."""
    torch.manual_seed(12000 + batch_size)
    d = _make_packed_decode_inputs(batch_size)
    if batch_size > 1:
        assert not d["mixed_qkv"].is_contiguous()
    assert d["mixed_qkv"].stride(1) == 1
    assert d["beta_storage"].shape == (batch_size, 144)
    assert d["b"].storage_offset() == 128
    assert d["b"].stride() == (144, 1)
    assert d["ssm"].stride(0) > 12 * V * K
    pool_size = d["ssm"].shape[0]
    d["cache_indices"] = d["allocated_cache_indices"][
        torch.randperm(pool_size - 1, device="cuda")[:batch_size]
    ].contiguous()

    cake, tri = CakeKDAKernel(), TritonKDAKernel()
    cake_calls = []
    run_cake = cake._packed_kda_decode

    def track_cake_call(**kwargs):
        cake_calls.append(kwargs)
        return run_cake(**kwargs)

    cake._packed_kda_decode = track_cake_call

    st_ref = _clone_strided_state(d["ssm"])
    ref_out = _packed_decode(tri, d, st_ref).float()
    st_cake = _clone_strided_state(d["ssm"])
    st_cake_before = _clone_strided_state(st_cake)
    out = _packed_decode(cake, d, st_cake).float()
    torch.cuda.synchronize()

    assert len(cake_calls) == 1
    assert cake_calls[0]["mixed_qkv"].data_ptr() == d["mixed_qkv"].data_ptr()
    assert cake_calls[0]["raw_beta"].data_ptr() == d["b"].data_ptr()
    assert cake_calls[0]["raw_beta"].storage_offset() == 128
    assert cake_calls[0]["raw_beta"].stride() == (144, 1)
    assert cake_calls[0]["state"].data_ptr() == st_cake.data_ptr()
    assert cake_calls[0]["state_indices"].data_ptr() == d["cache_indices"].data_ptr()
    assert cake_calls[0]["output"].shape == (batch_size, 1, 12, V)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    idx = d["cache_indices"].long()
    torch.testing.assert_close(
        st_cake[idx].float(), st_ref[idx].float(), atol=1e-2, rtol=1e-2
    )
    selected = torch.zeros(pool_size, dtype=torch.bool, device="cuda")
    selected[idx] = True
    torch.testing.assert_close(
        st_cake[~selected], st_cake_before[~selected], atol=0, rtol=0
    )


@pytest.mark.skipif(
    not CAKE_PACKED_AVAILABLE,
    reason="CAKE packed KDA decode requires its FlashInfer export on SM100/SM103.",
)
def test_kda_decode_cake_masks_negative_state_indices():
    """CUDA-graph padding rows must neither touch slot zero nor emit output."""
    batch_size = 8
    torch.manual_seed(12200)
    d = _make_packed_decode_inputs(batch_size)
    d["cache_indices"] = torch.tensor(
        [11, -1, 3, 7, -1, 1, 15, 4],
        device="cuda",
        dtype=torch.int32,
    )
    d["cache_indices_cpu"] = d["cache_indices"].cpu()
    active_rows = d["cache_indices"] >= 0

    # The Triton oracle only receives active rows, avoiding any dependency on
    # its own negative-index padding convention.
    active = dict(d)
    active["mixed_qkv"] = d["mixed_qkv"][active_rows].contiguous()
    for name in ("a", "b"):
        active[name] = d[name][active_rows].contiguous()
    active["B"] = int(active_rows.sum().item())
    active["cache_indices"] = d["cache_indices"][active_rows].contiguous()
    active["qsl"] = torch.arange(
        active["B"] + 1,
        device="cuda",
        dtype=torch.int32,
    )

    cake, tri = CakeKDAKernel(), TritonKDAKernel()
    state_ref = _clone_strided_state(d["ssm"])
    output_ref = _packed_decode(tri, active, state_ref).float()
    state_cake = _clone_strided_state(d["ssm"])
    state_before = _clone_strided_state(state_cake)
    output_cake = _packed_decode(
        cake,
        d,
        state_cake,
        cache_indices_cpu=d["cache_indices_cpu"],
    ).float()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output_cake[active_rows], output_ref, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        output_cake[~active_rows],
        torch.zeros_like(output_cake[~active_rows]),
        atol=0,
        rtol=0,
    )
    active_indices = d["cache_indices"][active_rows].long()
    torch.testing.assert_close(
        state_cake[active_indices].float(),
        state_ref[active_indices].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    selected = torch.zeros(state_cake.shape[0], dtype=torch.bool, device="cuda")
    selected[active_indices] = True
    torch.testing.assert_close(
        state_cake[~selected], state_before[~selected], atol=0, rtol=0
    )


@pytest.mark.skipif(
    not CAKE_PACKED_AVAILABLE,
    reason="CAKE packed KDA decode requires its FlashInfer export on SM100/SM103.",
)
@pytest.mark.parametrize("batch_size", [1, 8, 31, 32, 64, 128])
def test_kda_decode_cake_indexed_state_cuda_graph_replay(batch_size):
    """Replay must read changed beta/indices from stable captured buffers."""
    torch.manual_seed(12300 + batch_size)
    d = _make_packed_decode_inputs(batch_size)
    assert d["beta_storage"].shape == (batch_size, 144)
    assert d["b"].storage_offset() == 128
    assert d["b"].stride() == (144, 1)
    pool_size = d["ssm"].shape[0]
    initial_state = _clone_strided_state(d["ssm"])
    graph_state = _clone_strided_state(initial_state)
    graph_indices = d["cache_indices"]
    cake, tri = CakeKDAKernel(), TritonKDAKernel()
    cake_calls = []
    run_cake = cake._packed_kda_decode

    def track_cake_call(**kwargs):
        cake_calls.append(kwargs)
        return run_cake(**kwargs)

    cake._packed_kda_decode = track_cake_call

    def assert_replay_index_contract(active_prefix):
        contract = _issue_state_index_contract(
            d["index_allocator"],
            graph_indices,
            active_prefix=active_prefix,
            state_slots=graph_state.shape[0],
            active_request_ids=tuple(
                f"graph-replay-request-{i}" for i in range(active_prefix)
            ),
        )
        assert contract.matches(
            graph_indices,
            batch_size=batch_size,
            state_slots=graph_state.shape[0],
        )
        assert (
            CakeKDAKernel._cake_cache_index_source_admission(
                graph_indices,
                cache_indices_cpu=None,
                cache_index_contract=contract,
                batch_size=batch_size,
                state_slots=graph_state.shape[0],
            )
            is None
        )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        _packed_decode(cake, d, graph_state)
        graph_state.copy_(initial_state)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output = _packed_decode(cake, d, graph_state)
    capture_stream.synchronize()

    # Python runs for warmup/capture only.  Both calls must bind the interior
    # beta view directly; graph replay below reuses the captured pointer.
    assert len(cake_calls) == 2
    for call in cake_calls:
        assert call["raw_beta"].data_ptr() == d["b"].data_ptr()
        assert call["raw_beta"].storage_offset() == 128
        assert call["raw_beta"].stride() == (144, 1)

    # First replay changes beta only.  This isolates the captured beta read
    # from qkv/gate/index changes and catches a hidden contiguous copy made at
    # capture time (especially the B=1 row-pitch canonicalization case).
    captured_initial_output = captured_output.clone()
    changed_beta = torch.full_like(d["b"], 7.0)
    beta_case = dict(d)
    beta_case["b"] = changed_beta
    beta_state_ref = _clone_strided_state(initial_state)
    beta_output_ref = _packed_decode(tri, beta_case, beta_state_ref).float()

    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        graph_state.copy_(initial_state)
        d["b"].copy_(changed_beta)
    capture_stream.synchronize()
    assert_replay_index_contract(batch_size)
    with torch.cuda.stream(capture_stream):
        graph.replay()
    capture_stream.synchronize()

    torch.testing.assert_close(
        captured_output.float(), beta_output_ref, atol=1e-2, rtol=1e-2
    )
    beta_indices = d["cache_indices"].long()
    torch.testing.assert_close(
        graph_state[beta_indices].float(),
        beta_state_ref[beta_indices].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    assert not torch.equal(captured_output, captured_initial_output)

    permutations = [
        d["allocated_cache_indices"][
            torch.randperm(pool_size - 1, device="cuda")[:batch_size]
        ].contiguous(),
    ]
    num_padding = min(8, batch_size - 1)
    if num_padding:
        permutations.append(
            torch.cat(
                (
                    d["allocated_cache_indices"][
                        torch.randperm(pool_size - 1, device="cuda")[
                            : batch_size - num_padding
                        ]
                    ].contiguous(),
                    torch.full(
                        (num_padding,),
                        -1,
                        device="cuda",
                        dtype=torch.int32,
                    ),
                )
            )
        )
    for indices in permutations:
        next_mixed_qkv = torch.randn_like(d["mixed_qkv"])
        next_a = torch.randn_like(d["a"])
        next_b = torch.randn_like(d["b"])
        active_rows = indices >= 0
        active_indices = indices[active_rows].long()
        active = dict(d)
        active["mixed_qkv"] = next_mixed_qkv[active_rows].contiguous()
        active["a"] = next_a[active_rows].contiguous()
        active["b"] = next_b[active_rows].contiguous()
        active["B"] = int(active_rows.sum().item())
        active["cache_indices"] = indices[active_rows].contiguous()
        active["qsl"] = torch.arange(
            active["B"] + 1,
            device="cuda",
            dtype=torch.int32,
        )
        state_ref = _clone_strided_state(initial_state)
        output_ref = _packed_decode(tri, active, state_ref).float()

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph_state.copy_(initial_state)
            graph_indices.copy_(indices)
            d["mixed_qkv"].copy_(next_mixed_qkv)
            d["a"].copy_(next_a)
            d["b"].copy_(next_b)
        capture_stream.synchronize()
        assert_replay_index_contract(int(active_rows.sum().item()))
        with torch.cuda.stream(capture_stream):
            graph.replay()
        capture_stream.synchronize()

        torch.testing.assert_close(
            captured_output[active_rows].float(),
            output_ref,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            captured_output[~active_rows],
            torch.zeros_like(captured_output[~active_rows]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            graph_state[active_indices].float(),
            state_ref[active_indices].float(),
            atol=1e-2,
            rtol=1e-2,
        )
        selected = torch.zeros(pool_size, dtype=torch.bool, device="cuda")
        selected[active_indices] = True
        torch.testing.assert_close(
            graph_state[~selected], initial_state[~selected], atol=0, rtol=0
        )


@pytest.mark.skipif(
    not CAKE_ARCH_SUPPORTED,
    reason="CAKE KDA decode requires SM100 or SM103.",
)
def test_kda_decode_cake_falls_back_for_gqa():
    torch.manual_seed(4800)
    data = _make_packed_decode_inputs(4, num_heads=4, num_value_heads=8)
    cake, triton = CakeKDAKernel(), TritonKDAKernel()

    state_ref = _clone_strided_state(data["ssm"])
    output_ref = _packed_decode(triton, data, state_ref).float()
    state_cake = _clone_strided_state(data["ssm"])
    output_cake = _packed_decode(cake, data, state_cake).float()
    torch.cuda.synchronize()

    torch.testing.assert_close(output_cake, output_ref)
    torch.testing.assert_close(state_cake, state_ref)


@pytest.mark.parametrize("batch_size,num_spec", [(1, 7), (8, 7), (32, 3)])
def test_kda_target_verify_flashinfer_matches_triton(batch_size, num_spec):
    """FlashInfer MTP / target_verify (topk=1) per-draft-token output matches the
    Triton KDA verify reference over T = 1 + num_spec draft tokens per sequence."""
    torch.manual_seed(batch_size + num_spec)
    d = _make_verify_inputs(batch_size, 1 + num_spec)
    fi, tri = FlashInferKDAKernel(), TritonKDAKernel()

    ref_out = _verify(
        tri, d, d["ssm"].clone(), d["intermediate_states"].clone()
    ).float()
    out = _verify(fi, d, d["ssm"].clone(), d["intermediate_states"].clone()).float()
    torch.cuda.synchronize()

    assert torch.isfinite(out).all(), "FlashInfer verify output has non-finite values"
    o_err = (out - ref_out).abs()
    # B200 kernel-unit measured verify out max-abs-diff ~2e-4.
    assert o_err.max().item() < 1e-2, f"verify out max diff {o_err.max().item():.2e}"
    assert o_err.mean().item() < 1e-3, f"verify out mean diff {o_err.mean().item():.2e}"


@pytest.mark.parametrize(
    "batch_size,num_spec,extra_steps",
    [(1, 7, 0), (8, 7, 0), (32, 3, 2)],
)
def test_kda_target_verify_flashinfer_checkpoint_states(
    batch_size, num_spec, extra_steps
):
    """Checkpoint states must match true sequential decode states."""
    torch.manual_seed(1000 + batch_size + num_spec)
    cache_steps = 1 + num_spec
    d = _make_verify_inputs(
        batch_size,
        cache_steps,
        allocated_steps=cache_steps + extra_steps,
    )
    fi = FlashInferKDAKernel()

    ref_states = _sequential_decode_states(fi, d).float()

    intermediate_states = d["intermediate_states"].clone()
    _verify(
        fi, d, d["ssm"].clone(), intermediate_states
    )  # fills intermediate_states[n, t] in place
    torch.cuda.synchronize()

    got = intermediate_states[:, : d["T"]].float()  # [B, T, HV, V, K] checkpoint states
    assert torch.isfinite(got).all(), "verify checkpoint states have non-finite values"
    s_err = (got - ref_states).abs()
    # bf16 recurrent state; same tolerance as the decode committed-state check.
    assert (
        s_err.max().item() < 1e-1
    ), f"checkpoint state max diff {s_err.max().item():.2e}"
    assert (
        s_err.mean().item() < 1e-2
    ), f"checkpoint state mean diff {s_err.mean().item():.2e}"


def test_kda_target_verify_flashinfer_rejects_tree_spec():
    """Tree speculation (retrieve_parent_token != None) is unsupported (topk=1
    linear chain only) and must raise, not silently miscompute."""
    d = _make_verify_inputs(2, 4)
    parent = torch.zeros(d["seq"], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="topk=1"):
        FlashInferKDAKernel().target_verify(
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            q=d["q"],
            k=d["k"],
            v=d["v"],
            a=d["a"],
            b=d["b"],
            ssm_states=d["ssm"].clone(),
            cache_indices=d["cache_indices"],
            query_start_loc=d["qsl"],
            intermediate_states_buffer=d["intermediate_states"].clone(),
            intermediate_state_indices=d["intermediate_indices"],
            cache_steps=d["T"],
            retrieve_parent_token=parent,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
