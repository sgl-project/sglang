"""PPU ReplaySSM verify/commit integration, using the serving backend and pools.

CUDA_VISIBLE_DEVICES=0 SGLANG_SAIL_PLA_CUDA=1 \
    python -m pytest -q test/ppu/test_kda_pla_replayssm.py

Requires PLA's rebuilt kda_mtp_sglang strided verify/commit extension. H=8/12
are GLM-5.3-Flash/Kimi-K3 at attention TP8. Kimi H=6/24/48/96 cover
attention TP16/4/2/1; T=2..8 cover the dense verify dispatch range.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires PPU")


def make_case(heads, steps):
    from sglang.srt.layers.attention.linear.kda_backend import (
        KDAAttnBackend,
        KDAKernelDispatcher,
    )
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend

    torch.manual_seed(20260916)
    batch, slots, layers, dim = 4, 8, 2, heads * 128

    def rand(*shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device="cuda") * 0.1).to(dtype)

    state = SimpleNamespace(
        temporal=rand(layers, slots, heads, 128, 128, dtype=torch.float32),
        conv=[rand(layers, slots, 3, 3 * dim)],
        intermediate_conv_window=[
            torch.full(
                (layers, slots, steps + 2, 3, 3 * dim),
                7,
                device="cuda",
                dtype=torch.bfloat16,
            )
        ],
        intermediate_ssm=None,
        replayssm_rawv=rand(layers, slots, heads, steps, 128),
        replayssm_rawk=rand(layers, slots, heads, steps, 128),
        replayssm_g=rand(layers, slots, heads, steps, 128, dtype=torch.float32),
        replayssm_beta=rand(layers, slots, heads, steps, dtype=torch.float32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=heads,
        head_q_dim=128,
        head_k_dim=128,
        head_v_dim=128,
        q_dim=dim,
        k_dim=dim,
        v_dim=dim,
        bias=None,
        lower_bound=-5.0,
        conv_weights=rand(3 * dim, 4, dtype=torch.float32),
        A_log=rand(heads, dtype=torch.float32),
        dt_bias=rand(dim, dtype=torch.float32),
    )
    # Fused model projections leave gaps between tokens in QKV, gate and beta.
    projection = rand(batch * steps, 4 * dim + heads)
    qkv = projection[:, : 3 * dim]
    beta = projection[:, 3 * dim : 3 * dim + heads].unsqueeze(0)
    gate = rand(1, batch * steps, 2 * dim)[..., :dim]
    if heads in (6, 12, 24, 48, 96):  # Kimi offers [1, T, H, K].
        gate = gate.unflatten(-1, (heads, 128))
    inputs = (qkv, gate, beta)
    indices = torch.tensor([2, 1, 3, -1], device="cuda", dtype=torch.int32)
    backend = KDAAttnBackend.__new__(KDAAttnBackend)
    backend.forward_metadata = SimpleNamespace(
        mamba_cache_indices=indices,
        query_start_loc=torch.arange(
            0, (batch + 1) * steps, steps, device="cuda", dtype=torch.int32
        ),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    backend.verify_intermediate_state_indices = torch.arange(
        slots, device="cuda", dtype=torch.int32
    )
    backend.kernel_dispatcher = KDAKernelDispatcher(
        *(LinearAttnKernelBackend.TRITON,) * 3
    )
    forward = SimpleNamespace(
        spec_info=SimpleNamespace(draft_token_num=steps, ragged_verify_layout=None)
    )
    return backend, state, layer, inputs, forward


def bind_state(backend, state):
    def layer_cache(i):
        return SimpleNamespace(
            **{
                name: (
                    [t[i] for t in value]
                    if isinstance(value, list)
                    else None if value is None else value[i]
                )
                for name, value in vars(state).items()
            }
        )

    backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=layer_cache)


def verify(backend, state, layer, inputs, forward):
    bind_state(backend, state)
    outputs = []
    for i in range(state.temporal.shape[0]):
        layer.layer_id = i
        outputs.append(backend._forward_target_verify(layer, forward, *inputs))
    return torch.stack(outputs)


@pytest.mark.parametrize(
    "heads,steps",
    [(8, 6), (4, 2), (6, 8)]
    + [(heads, steps) for heads in (12, 24, 48, 96) for steps in range(2, 9)],
)
def test_verify_commit_and_graph(heads, steps):
    from pla.decode import kda_mtp_sglang as pla

    from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
        commit_kda_replayssm_after_verify as commit,
    )
    from sglang.srt.environ import envs

    backend, initial, layer, inputs, forward = make_case(heads, steps)
    reference, candidate = copy.deepcopy(initial), copy.deepcopy(initial)
    commit_args = dict(
        state_batch_indices=backend.forward_metadata.mamba_cache_indices,
        accept_lens=torch.tensor(
            [1, steps, steps // 2, 0], device="cuda", dtype=torch.int32
        ),
        last_correct_step_indices=torch.tensor(
            [0, steps - 1, steps // 2 - 1, -1], device="cuda", dtype=torch.int32
        ),
        mamba_track_indices=torch.tensor(
            [5, -1, 6, -1], device="cuda", dtype=torch.int32
        ),
        mamba_steps_to_track=torch.tensor(
            [0, -1, 0, -1], device="cuda", dtype=torch.int32
        ),
    )
    # Consecutive iterations consume the committed state and expose rollback bugs.
    for _ in range(2):
        before = candidate.temporal.clone()
        conv_before = candidate.conv[0].clone()
        with envs.SGLANG_SAIL_PLA_CUDA.override(False):
            expected = verify(backend, reference, layer, inputs, forward)
            commit(spec_state=reference, **commit_args)
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            with patch.object(
                pla,
                "fused_kda_decode_mtp_dspark",
                wraps=pla.fused_kda_decode_mtp_dspark,
            ) as hit:
                actual = verify(backend, candidate, layer, inputs, forward)
                assert hit.call_count == 2
                kwargs = hit.call_args.kwargs
                assert (
                    kwargs["x_q"].untyped_storage().data_ptr()
                    == inputs[0].untyped_storage().data_ptr()
                )
                assert not kwargs["cs_q"].is_contiguous()
                assert not kwargs["intermediate_conv_q"].is_contiguous()
            torch.testing.assert_close(candidate.temporal, before, rtol=0, atol=0)
            torch.testing.assert_close(candidate.conv[0], conv_before, rtol=0, atol=0)
            # Isolate FP32 fold accuracy from BF16 verify/conv rounding across
            # iterations: both commit paths consume exactly the same rings.
            same_rings = copy.deepcopy(candidate)
            with envs.SGLANG_SAIL_PLA_CUDA.override(False):
                commit(spec_state=same_rings, **commit_args)
            with patch.object(
                pla,
                "commit_kda_replayssm_after_verify",
                wraps=pla.commit_kda_replayssm_after_verify,
            ) as hit:
                commit(spec_state=candidate, **commit_args)
                assert hit.call_count == 1
                assert hit.call_args.kwargs["spec_state"] is candidate
            torch.testing.assert_close(
                candidate.temporal, same_rings.temporal, rtol=1e-5, atol=1e-7
            )
            torch.testing.assert_close(
                candidate.conv[0], same_rings.conv[0], rtol=0, atol=0
            )
        # The old Triton padding output is undefined; PLA explicitly zeroes it.
        torch.testing.assert_close(
            actual[:, :, : 3 * steps], expected[:, :, : 3 * steps], rtol=0.02, atol=2e-3
        )
        assert torch.count_nonzero(actual[:, :, 3 * steps :]) == 0
        # BF16 verify intermediates can differ by one rounding step on a later
        # iteration. Keep the tighter same-ring FP32 fold check above.
        torch.testing.assert_close(
            candidate.temporal, reference.temporal, rtol=1e-3, atol=1e-4
        )
        torch.testing.assert_close(candidate.conv[0], reference.conv[0], rtol=0, atol=0)
        for name in (
            "replayssm_rawv",
            "replayssm_rawk",
            "replayssm_g",
            "replayssm_beta",
        ):
            torch.testing.assert_close(
                getattr(candidate, name), getattr(reference, name), rtol=0.02, atol=2e-3
            )
        torch.testing.assert_close(
            candidate.intermediate_conv_window[0],
            reference.intermediate_conv_window[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.temporal[:, [0, 4, 7]],
            initial.temporal[:, [0, 4, 7]],
            rtol=0,
            atol=0,
        )

    # Capture uses exactly the same serving buffers. Reset outside the graph.
    graph_state = copy.deepcopy(initial)
    eager_state = copy.deepcopy(initial)
    with envs.SGLANG_SAIL_PLA_CUDA.override(True):
        eager_out = verify(backend, eager_state, layer, inputs, forward)
        commit(spec_state=eager_state, **commit_args)
        verify(backend, graph_state, layer, inputs, forward)  # warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = verify(backend, graph_state, layer, inputs, forward)
            commit(spec_state=graph_state, **commit_args)
        for _ in range(2):
            graph_state.temporal.copy_(initial.temporal)
            graph_state.conv[0].copy_(initial.conv[0])
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)
            torch.testing.assert_close(
                graph_state.temporal, eager_state.temporal, rtol=0, atol=0
            )
            torch.testing.assert_close(
                graph_state.conv[0], eager_state.conv[0], rtol=0, atol=0
            )


def test_ragged_verify_fallback_then_pla_commit():
    from pla.decode import kda_mtp_sglang as pla

    from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
        commit_kda_replayssm_after_verify as commit,
    )
    from sglang.srt.environ import envs
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    backend, initial, layer, inputs, forward = make_case(12, 8)
    reference, candidate = copy.deepcopy(initial), copy.deepcopy(initial)
    dense_qsl = backend.forward_metadata.query_start_loc
    qsl = torch.tensor([0, 1, 9, 13, 13], device="cuda", dtype=torch.int32)
    # Three real requests verify 1/8/4 tokens; the final graph slot is empty.
    # Three extra graph-tier tokens must map to the discarded ghost row.
    selected = torch.tensor(
        [0, *range(8, 16), *range(16, 20), 24, 25, 26], device="cuda"
    )
    packed = (inputs[0][selected], inputs[1][:, selected], inputs[2][:, selected])
    ragged = RaggedVerifyLayout(
        verify_lens=torch.tensor([1, 8, 4, 0], device="cuda", dtype=torch.int32),
        graph_num_tokens=16,
        extend_start_loc=qsl[:-1],
        qo_indptr_device=qsl,
        cap=8,
    )
    kwargs = dict(
        state_batch_indices=backend.forward_metadata.mamba_cache_indices,
        accept_lens=torch.tensor([1, 8, 2, 0], device="cuda", dtype=torch.int32),
        last_correct_step_indices=torch.tensor(
            [0, 7, 1, -1], device="cuda", dtype=torch.int32
        ),
    )
    for _ in range(2):
        backend.forward_metadata.query_start_loc = dense_qsl
        forward.spec_info.ragged_verify_layout = None
        with envs.SGLANG_SAIL_PLA_CUDA.override(False):
            expected = verify(backend, reference, layer, inputs, forward)
            commit(spec_state=reference, **kwargs)
        backend.forward_metadata.query_start_loc = qsl
        forward.spec_info.ragged_verify_layout = ragged
        before = candidate.temporal.clone()
        with envs.SGLANG_SAIL_PLA_CUDA.override(True):
            with patch.object(
                pla, "fused_kda_decode_mtp_dspark", side_effect=AssertionError("ragged")
            ) as fused:
                actual = verify(backend, candidate, layer, packed, forward)
                fused.assert_not_called()
            torch.testing.assert_close(candidate.temporal, before, rtol=0, atol=0)
            # The existing Triton conv path updates conv state during verify;
            # the accepted-window rollback below must restore the right state.
            with patch.object(
                pla,
                "commit_kda_replayssm_after_verify",
                wraps=pla.commit_kda_replayssm_after_verify,
            ) as fold:
                commit(spec_state=candidate, **kwargs)
                fold.assert_called_once()
        torch.testing.assert_close(
            actual[:, :, :13], expected[:, :, selected[:13]], rtol=0.02, atol=2e-3
        )
        assert torch.count_nonzero(actual[:, :, 13:]) == 0
        torch.testing.assert_close(
            candidate.temporal, reference.temporal, rtol=1e-3, atol=3e-5
        )
        torch.testing.assert_close(candidate.conv[0], reference.conv[0], rtol=0, atol=0)


def test_dispatch_guards():
    from sglang.srt.environ import envs

    backend, state, layer, inputs, _ = make_case(8, 6)
    bind_state(backend, state)
    args = dict(
        layer=layer,
        mixed_qkv=inputs[0],
        a=inputs[1],
        b=inputs[2],
        draft_token_num=6,
        ragged_layout=None,
        conv_states=state.conv[0][0],
        ssm_states=state.temporal[0],
        intermediate_state_cache=None,
        intermediate_conv_window_cache=state.intermediate_conv_window[0][0],
        retrieve_parent_token=None,
        replayssm_rawv=state.replayssm_rawv[0],
    )
    with envs.SGLANG_SAIL_PLA_CUDA.override(True):
        assert backend._can_run_dspark_cutedsl_mtp(**args)
        for changes in (
            {"ragged_layout": object()},
            {"retrieve_parent_token": inputs[0]},
            {"intermediate_state_cache": state.temporal},
            {"draft_token_num": 9},
            {"a": inputs[1].float()},
        ):
            assert not backend._can_run_dspark_cutedsl_mtp(**(args | changes))
