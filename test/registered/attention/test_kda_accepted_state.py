"""Pending state maps preserve request ownership and all-layer materialization."""

import sys

import pytest
import torch

from sglang.srt.mem_cache.kda_accepted_state import KDAAcceptedState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="accepted-state verify is currently supported only on SM90",
)


def test_pending_state_materialize_track_restore_and_graph():
    # Physical state slots outnumber active-request scratch rows. Outer pitch
    # includes unrelated storage, which materialization must leave unchanged.
    storage = torch.full((2, 32, 2, 4, 16, 16), 123.0, device="cuda")
    temporal = storage[:, :, 0]
    scratch = torch.randn(2, 5, 2, 4, 16, 16, device="cuda")
    state = KDAAcceptedState(temporal, scratch)
    slots = torch.tensor([19, 7, -1], device="cuda", dtype=torch.int32)
    rows = torch.tensor([3, 1, -1], device="cuda", dtype=torch.int32)
    steps = torch.tensor([1, 0, -1], device="cuda", dtype=torch.int32)
    targets = torch.tensor([21, 22, -1], device="cuda", dtype=torch.int32)
    state.record(slots, rows, steps)
    assert (temporal == 123).all()
    state.track(targets, rows, steps)
    torch.testing.assert_close(temporal[:, 21], scratch[:, 3, 1], rtol=0, atol=0)
    torch.testing.assert_close(temporal[:, 22], scratch[:, 1, 0], rtol=0, atol=0)
    state.materialize(slots[:1])
    torch.testing.assert_close(temporal[:, 19], scratch[:, 3, 1], rtol=0, atol=0)
    assert state.steps[19] == -1 and state.steps[7] == 0
    assert (temporal[:, 7] == 123).all()
    # Recycling scratch row 3 must not change the materialized old request.
    checkpoint = temporal[:, 19].clone()
    scratch[:, 3].fill_(999)
    state.materialize(slots[:1])
    torch.testing.assert_close(temporal[:, 19], checkpoint, rtol=0, atol=0)
    # A restored/cleared physical slot invalidates its old pending source.
    temporal[:, 7].fill_(-42)
    state.invalidate(slots[1:2])
    state.materialize()
    assert (temporal[:, 7] == -42).all()
    assert (storage[:, :, 1] == 123).all()

    def run():
        state.record(slots, rows, steps)
        state.track(targets, rows, steps)
        state.materialize(slots)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    rows[:2] = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    steps[:2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    scratch.normal_()
    graph.replay()
    torch.testing.assert_close(temporal[:, 19], scratch[:, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(temporal[:, 7], scratch[:, 4, 1], rtol=0, atol=0)
    torch.testing.assert_close(temporal[:, 21], scratch[:, 0, 0], rtol=0, atol=0)
    assert (state.steps == -1).all()
    assert (storage[:, :, 1] == 123).all()


def test_verify_fresh_pending_reordered_requests_and_graph():
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )

    torch.manual_seed(2703)
    n, h, d, t = 3, 64, 128, 2
    storage = torch.randn(1, 12, 2, h, d, d, device="cuda") * 0.05
    temporal = storage[:, :, 0]
    reference = temporal.clone()
    scratch = torch.full((1, 5, t, h, d, d), float("nan"), device="cuda")
    base_scratch = torch.empty(1, n, t, h, d, d, device="cuda")
    state = KDAAcceptedState(temporal, scratch)
    slots = torch.tensor([9, 7, -1], device="cuda", dtype=torch.int32)
    rows = torch.tensor([3, 1, -1], device="cuda", dtype=torch.int32)
    steps = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32)
    batch_rows = torch.arange(n, device="cuda", dtype=torch.int32)
    q, k, v, a = [
        (torch.randn(1, n * t, h, d, device="cuda") * 0.5).bfloat16() for _ in range(4)
    ]
    common = dict(
        A_log=torch.randn(h, device="cuda") * 0.1,
        dt_bias=torch.randn(h * d, device="cuda") * 0.1,
        a=a,
        q=q,
        k=k,
        v=v,
        b=torch.randn(1, n * t, h, device="cuda").bfloat16(),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        initial_state_indices=slots,
        cu_seqlens=torch.arange(0, (n + 1) * t, t, device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
        disable_state_update=True,
        cache_steps=t,
    )

    def base():
        out = fused_sigmoid_gating_delta_rule_update(
            **common,
            initial_state_source=reference[0],
            intermediate_states_buffer=base_scratch[0],
            intermediate_state_indices=batch_rows,
        )
        fused_mamba_state_scatter_with_mask(reference, base_scratch, slots, steps)
        return out

    def direct():
        out = fused_sigmoid_gating_delta_rule_update(
            **common,
            initial_state_source=temporal[0],
            intermediate_states_buffer=scratch[0],
            intermediate_state_indices=rows,
            accepted_rows=state.rows,
            accepted_steps=state.steps,
        )
        state.record(slots, rows, steps)
        return out

    for round_id in range(8):
        if round_id % 2:
            slots[:2] = slots[:2].flip(0)
            rows[:2] = rows[:2].flip(0)
        steps[:2] = (steps[:2] + 1) % t
        expected, got = base(), direct()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            scratch[0, rows[:2].long(), steps[:2].long()],
            reference[0, slots[:2].long()],
            rtol=0,
            atol=0,
        )
        if round_id == 3:
            # Prefix copy / offload boundary; following verify seeds temporal.
            state.materialize(slots)
            torch.testing.assert_close(temporal, reference, rtol=0, atol=0)
            scratch.fill_(float("nan"))

    # Warm up and capture independent candidate/reference rounds. Restore the
    # same logical state before replay; captured metadata remains device-read.
    state.materialize(slots)
    checkpoint = reference.clone()
    graphs, outputs = {}, {}
    for name, fn in [("base", base), ("direct", direct)]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = fn()
        graphs[name] = graph
    temporal.copy_(checkpoint)
    reference.copy_(checkpoint)
    state.invalidate()
    scratch.fill_(float("nan"))
    for _ in range(6):
        slots[:2] = slots[:2].flip(0)
        rows[:2] = rows[:2].flip(0)
        steps[:2] = (steps[:2] + 1) % t
        graphs["base"].replay()
        graphs["direct"].replay()
        torch.testing.assert_close(outputs["direct"], outputs["base"], rtol=0, atol=0)
    state.materialize(slots)
    torch.testing.assert_close(temporal, reference, rtol=0, atol=0)


def test_real_pool_backend_commit_tracking_and_lifecycle():
    from types import SimpleNamespace as NS

    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
    )
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.kda_backend import (
        KDAAttnBackend,
        KDAKernelDispatcher,
    )
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    torch.manual_seed(3891)
    h, d, n = 64, 128, 2
    layers = [0, 2]
    shape = KimiLinearStateShape.create(tp_world_size=1, num_heads=h, head_dim=d)
    params = KimiLinearCacheParams(shape=shape, layers=layers)
    pools = [
        HybridReqToTokenPool(
            size=5,
            mamba_size=12,
            mamba_spec_state_size=5,
            max_context_len=32,
            device="cuda",
            enable_memory_saver=False,
            cache_params=params,
            mamba_layer_ids=layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=2,
            speculative_eagle_topk=1,
        )
        for _ in range(2)
    ]
    base_pool, pool = pools
    caches = [p.get_speculative_mamba2_params_all_layers() for p in pools]
    base_cache, cache = caches
    base_cache.temporal.normal_(std=0.05)
    base_cache.conv[0].normal_(std=0.05)
    cache.temporal.copy_(base_cache.temporal)
    cache.conv[0].copy_(base_cache.conv[0])
    accepted = KDAAcceptedState(cache.temporal, cache.intermediate_ssm)
    pool.mamba_pool.kda_accepted_state = accepted
    slots = torch.tensor([9, 7], device="cuda", dtype=torch.int32)
    rows = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    steps = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    track = torch.tensor([10, 11], device="cuda", dtype=torch.int32)
    track_steps = torch.tensor([0, -1], device="cuda", dtype=torch.int32)
    qsl = torch.arange(0, (n + 1) * 2, 2, device="cuda", dtype=torch.int32)
    batch = NS(
        spec_info=NS(draft_token_num=2, ragged_verify_layout=None),
        req_pool_indices=rows,
    )
    backends, hybrids = [], []
    for p, state in [(base_pool, None), (pool, accepted)]:
        backend = object.__new__(KDAAttnBackend)
        backend.req_to_token_pool = p
        backend.accepted_state = state
        backend.accept_lens_pool = None
        backend._fused_chain_verify_fn = None
        triton_backend = LinearAttnKernelBackend.TRITON
        backend.kernel_dispatcher = KDAKernelDispatcher(
            triton_backend, triton_backend, triton_backend
        )
        backend.verify_intermediate_state_indices = torch.arange(
            5, device="cuda", dtype=torch.int32
        )
        backend.forward_metadata = ForwardMetadata(
            query_start_loc=qsl, mamba_cache_indices=slots
        )
        hybrid = object.__new__(HybridLinearAttnBackend)
        hybrid.linear_attn_backend = backend
        backends.append(backend)
        hybrids.append(hybrid)
    layer_inputs = []
    for lid in layers:
        layer = NS(
            layer_id=lid,
            q_dim=h * d,
            k_dim=h * d,
            v_dim=h * d,
            head_q_dim=d,
            head_k_dim=d,
            head_v_dim=d,
            conv_weights=(torch.randn(3 * h * d, 4, device="cuda") * 0.1).bfloat16(),
            bias=None,
            A_log=torch.randn(h, device="cuda") * 0.1,
            dt_bias=torch.randn(h * d, device="cuda") * 0.1,
            lower_bound=-5.0,
        )
        mixed = (torch.randn(n * 2, 3 * h * d, device="cuda") * 0.5).bfloat16()
        a = (torch.randn(1, n * 2, h * d, device="cuda") * 0.5).bfloat16()
        b = torch.randn(1, n * 2, h, device="cuda").bfloat16()
        layer_inputs.append((layer, mixed, a, b))

    def run(index):
        outputs = [
            backends[index]._forward_target_verify(layer, batch, mixed, a, b)
            for layer, mixed, a, b in layer_inputs
        ]
        hybrids[index].update_mamba_state_after_mtp_verify(
            steps, track, track_steps, None, rows
        )
        return outputs

    for round_id in range(5):
        if round_id % 2:
            slots.copy_(slots.flip(0))
            rows.copy_(rows.flip(0))
        steps.copy_((steps + 1) % 2)
        expected, got = run(0), run(1)
        for x, y in zip(expected, got):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(cache.conv[0], base_cache.conv[0], rtol=0, atol=0)
        torch.testing.assert_close(
            cache.temporal[:, 10:], base_cache.temporal[:, 10:], rtol=0, atol=0
        )
    # Non-verify continuation flushes every layer before any committed-state read.
    backends[1]._materialize_accepted_state(layers[0])
    torch.testing.assert_close(cache.temporal, base_cache.temporal, rtol=0, atol=0)
    run(0)
    run(1)
    # Prefix/COW copying is an actual pool operation, including conv.
    dst = torch.tensor([5, 6], device="cuda", dtype=torch.int32)
    base_pool.mamba_pool.copy_from(slots, dst)
    pool.mamba_pool.copy_from(slots, dst)
    torch.testing.assert_close(cache.temporal, base_cache.temporal, rtol=0, atol=0)
    run(0)
    run(1)
    exported = pool.mamba_pool.get_cpu_copy(slots)
    expected_export = base_pool.mamba_pool.get_cpu_copy(slots)
    torch.testing.assert_close(exported[1], expected_export[1], rtol=0, atol=0)
    run(1)
    pool.mamba_pool.load_cpu_copy(exported, slots)
    accepted.materialize(slots)
    torch.testing.assert_close(
        cache.temporal[:, slots.long()].cpu(), exported[1], rtol=0, atol=0
    )
    run(1)
    pool.req_index_to_mamba_index_mapping[rows.long()] = slots
    pool.free_rows(rows.cpu().tolist())
    assert (accepted.steps[slots.long()] == -1).all()
    run(1)
    pool.mamba_pool.clear_slots(slots)
    accepted.materialize(slots)
    assert (cache.temporal[:, slots.long()] == 0).all()
    assert (cache.conv[0][:, slots.long()] == 0).all()

    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid_module
    from sglang.srt.layers.attention.linear import kda_backend as kda_module

    runner = NS(
        device="cuda",
        server_args=NS(enable_mis=False),
        is_draft_worker=False,
        req_to_token_pool=pool,
        token_to_kv_pool=None,
        linear_attn_backends=NS(
            decode=triton_backend, prefill=triton_backend, verify=triton_backend
        ),
    )
    spec = NS(
        speculative_eagle_topk=1,
        speculative_algorithm="EAGLE",
        speculative_num_draft_tokens=2,
    )
    with (
        envs.SGLANG_OPT_KDA_ACCEPTED_STATE.override(True),
        patch("sglang.srt.utils.common.require_mlp_sync", return_value=False),
        patch.object(kda_module, "get_spec", return_value=spec),
        patch.object(hybrid_module, "get_spec", return_value=spec),
        patch.object(
            hybrid_module, "get_memory", return_value=NS(enable_unified_memory=False)
        ),
        patch.object(
            kda_module, "get_disagg", return_value=NS(disaggregation_mode="decode")
        ),
        patch.object(
            kda_module,
            "get_exec",
            return_value=NS(mamba=NS(enable_linear_replayssm_spec=False)),
        ),
    ):
        constructed = KDAAttnBackend(runner)
        assert constructed.accepted_state is accepted
        with patch.object(pool.mamba_pool, "kda_accepted_state", None):
            fresh = KDAAttnBackend(runner)
            assert fresh.accepted_state.temporal is cache.temporal
            assert fresh.accepted_state.scratch is cache.intermediate_ssm
        with (
            patch.object(pool.mamba_pool, "kda_accepted_state", None),
            patch.object(
                pool,
                "get_speculative_mamba2_params_all_layers",
                return_value=NS(
                    temporal=cache.temporal,
                    intermediate_ssm=cache.intermediate_ssm[:, :1],
                ),
            ),
        ):
            with pytest.raises(ValueError, match="every request row"):
                KDAAttnBackend(runner)
        for name, value in (
            ("speculative_num_draft_tokens", 3),
            ("speculative_eagle_topk", 2),
            ("speculative_algorithm", "NGRAM"),
        ):
            with patch.object(spec, name, value):
                with pytest.raises(ValueError, match="T2 EAGLE"):
                    KDAAttnBackend(runner)
        with patch.object(pool.mamba_pool, "enable_linear_replayssm", True):
            with pytest.raises(ValueError, match="no ReplaySSM"):
                KDAAttnBackend(runner)
        with envs.SGLANG_OPT_KDA_ACCEPTED_STATE.override(False):
            assert KDAAttnBackend(runner).accepted_state is None
        runner.is_draft_worker = True
        assert KDAAttnBackend(runner).accepted_state is None


@pytest.mark.parametrize(
    "boundary", ["free_rows", "free_mamba_cache", "clear", "decode_clear"]
)
def test_allocator_release_does_not_retain_pending_scratch(boundary):
    from types import SimpleNamespace as NS

    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = HybridReqToTokenPool(
        size=1,
        mamba_size=1,
        mamba_spec_state_size=1,
        max_context_len=8,
        device="cuda",
        enable_memory_saver=False,
        cache_params=KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=1, num_heads=1, head_dim=16
            ),
            layers=[0],
        ),
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=2,
        speculative_eagle_topk=1,
    )
    rows = pool.alloc_rows(1)
    slots = pool.mamba_allocator.alloc(1)
    pool.req_index_to_mamba_index_mapping[rows] = slots
    cache = pool.get_speculative_mamba2_params_all_layers()
    state = KDAAcceptedState(cache.temporal, cache.intermediate_ssm)
    pool.mamba_pool.kda_accepted_state = state
    cache.intermediate_ssm.normal_()
    expected = cache.intermediate_ssm[:, rows[0], 1].clone()
    state.record(
        slots,
        torch.tensor(rows, device="cuda", dtype=torch.int32),
        torch.ones_like(slots),
    )
    if boundary == "free_rows":
        pool.free_rows(rows)
        assert pool.alloc_rows(1) == rows
    elif boundary == "free_mamba_cache":
        req = NS(kv=NS(mamba_pool_idx=slots[0]))
        pool.free_mamba_cache(req)
        assert req.kv.mamba_pool_idx is None
        torch.testing.assert_close(pool.mamba_allocator.alloc(1), slots)
    else:
        if boundary == "decode_clear":
            from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool

            # PD overrides clear(), so exercise that implementation separately
            # with the same real physical pool and pending checkpoint.
            HybridMambaDecodeReqToTokenPool.clear(pool)
        else:
            pool.clear()
        cache.temporal.fill_(17)

    # The released request's scratch can now be overwritten by another request.
    cache.intermediate_ssm.fill_(float("nan"))
    state.materialize()
    assert (state.steps == -1).all()
    if boundary in ("clear", "decode_clear"):
        assert (cache.temporal == 17).all()
    else:
        torch.testing.assert_close(
            cache.temporal[:, slots[0]], expected, rtol=0, atol=0
        )


def test_unified_cache_donation_materializes_before_request_row_reuse():
    from types import SimpleNamespace as NS

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
        MambaComponent,
    )

    temporal = torch.full((2, 8, 1, 16, 16), -1.0, device="cuda")
    scratch = torch.randn(2, 3, 2, 1, 16, 16, device="cuda")
    state = KDAAcceptedState(temporal, scratch)
    slot = torch.tensor([5], dtype=torch.int32, device="cuda")
    row = torch.tensor([2], dtype=torch.int32, device="cuda")
    step = torch.tensor([1], dtype=torch.int32, device="cuda")
    expected = scratch[:, 2, 1].clone()
    state.record(slot, row, step)

    # Exercise the real finish/donation hook. A donated slot is retained by the
    # radix tree, so this path need not call free_mamba_cache or copy_from.
    component = object.__new__(MambaComponent)
    component.cache = NS(
        enable_mamba_extra_buffer=False,
        req_to_token_pool=NS(
            mamba_pool=NS(kda_accepted_state=state, replayssm_write_pos=None),
            mamba_ckpt_pool=None,
        ),
    )
    req = NS(kv=NS(mamba_pool_idx=slot[0]))
    params = InsertParams()
    assert component.prepare_for_caching_req(req, params, 7, True) == 7
    torch.testing.assert_close(params.mamba_value, slot, rtol=0, atol=0)
    assert state.steps[5] == -1
    scratch[:, 2].fill_(float("nan"))
    state.materialize()
    torch.testing.assert_close(temporal[:, 5], expected, rtol=0, atol=0)


@pytest.mark.parametrize("which", ["temporal", "scratch"])
def test_reject_non_fp32_checkpoints(which):
    temporal = torch.empty(1, 2, 1, 16, 16, device="cuda")
    scratch = torch.empty(1, 2, 2, 1, 16, 16, device="cuda")
    if which == "temporal":
        temporal = temporal.bfloat16()
    else:
        scratch = scratch.bfloat16()
    with pytest.raises(ValueError, match="FP32 committed and scratch"):
        KDAAcceptedState(temporal, scratch)


def test_materialize_layer_stride_exceeds_int32():
    # One allocation holds disjoint committed/scratch entries in each layer.
    # Only the small entries are touched; the envelope pitch crosses 2^31 FP32
    # elements and catches int32 program_id * layer_stride overflow.
    pitch = (1 << 31) + 1024
    storage = torch.empty(pitch + 1024, device="cuda")
    temporal = storage.as_strided((2, 1, 1, 16, 16), (pitch, 256, 256, 16, 1))
    scratch = storage.as_strided(
        (2, 1, 1, 1, 16, 16), (pitch, 256, 256, 256, 16, 1), 512
    )
    temporal.fill_(99)
    scratch[0].fill_(3)
    scratch[1].fill_(7)
    state = KDAAcceptedState(temporal, scratch)
    index = torch.zeros(1, device="cuda", dtype=torch.int32)
    state.record(index, index, index)
    state.materialize()
    assert (temporal[0] == 3).all()
    assert (temporal[1] == 7).all()
    assert (state.steps == -1).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
