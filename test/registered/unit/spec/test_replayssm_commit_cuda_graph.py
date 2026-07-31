"""CUDA-graphed ReplaySSM commit == the eager commit, bitwise.

The graph path fuses the index math into one Triton kernel and replays
[indices, fold, conv scatters] as one CUDA graph against staging buffers. Same
kernels, same inputs => the committed temporal/conv state must equal the eager
path exactly, including: capture round vs replay round, track crossing rows,
track disabled (None), and a second batch size.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_after_verify,
)
from sglang.srt.speculative.replayssm_commit_cuda_graph import (
    ReplayssmCommitGraphRunner,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

L, SLOTS, HV, H, K, V = 3, 16, 8, 4, 64, 64
DRAFT, WIDTH, DIM, KM1 = 4, 4, 32, 3
TRACK_INTERVAL = 8
DEVICE = "cuda"


def _make_state(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    return SimpleNamespace(
        temporal=rnd(L, SLOTS, HV, V, K),
        replayssm_rawv=rnd(L, SLOTS, HV, DRAFT, V),
        replayssm_rawk=rnd(L, SLOTS, H, DRAFT, K),
        replayssm_g=(
            -torch.rand(L, SLOTS, HV, DRAFT, device=DEVICE, generator=gen) * 0.3
        ),
        replayssm_beta=torch.rand(L, SLOTS, HV, DRAFT, device=DEVICE, generator=gen),
        conv=[rnd(L, SLOTS, DIM, KM1)],
        intermediate_conv_window=[rnd(L, SLOTS, DRAFT, DIM, KM1)],
    )


def _clone_state(st):
    return SimpleNamespace(
        temporal=st.temporal.clone(),
        replayssm_rawv=st.replayssm_rawv.clone(),
        replayssm_rawk=st.replayssm_rawk.clone(),
        replayssm_g=st.replayssm_g.clone(),
        replayssm_beta=st.replayssm_beta.clone(),
        conv=[c.clone() for c in st.conv],
        intermediate_conv_window=[w.clone() for w in st.intermediate_conv_window],
    )


def _eager_reference(
    st, mamba_map, req_idx, accept_lens, accept_index, seq_lens, track
):
    bs = accept_lens.shape[0]
    state_batch_indices = mamba_map[req_idx]
    offsets = torch.arange(
        0, bs * DRAFT, step=DRAFT, dtype=accept_lens.dtype, device=DEVICE
    )
    rows = torch.arange(bs, dtype=torch.int64, device=DEVICE)
    last_correct = accept_index[rows, (accept_lens - 1).to(torch.int64)] - offsets
    steps_to_track = None
    if track is not None:
        pre, post = seq_lens, seq_lens + accept_lens
        crossed = pre // TRACK_INTERVAL != post // TRACK_INTERVAL
        point = post // TRACK_INTERVAL * TRACK_INTERVAL
        ith = torch.clamp(point - pre - 1, min=0).to(torch.int64)
        cand = accept_index[rows, ith] - offsets
        steps_to_track = torch.where(crossed, cand, torch.full_like(cand, -1))
    commit_gdn_replayssm_fold_after_verify(
        spec_state=st,
        state_batch_indices=state_batch_indices,
        accept_lens=accept_lens,
        last_correct_step_indices=last_correct,
        mamba_track_indices=track,
        mamba_steps_to_track=steps_to_track,
        null_block_id=-1,
    )


class TestReplayssmCommitCudaGraph(CustomTestCase):
    def _run_case(self, bs, with_track, runner, st_graph, st_eager, mamba_map, seed):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        req_idx = torch.randperm(SLOTS, device=DEVICE, generator=gen)[:bs]
        accept_lens = torch.randint(
            1, DRAFT + 1, (bs,), device=DEVICE, dtype=torch.int32, generator=gen
        )
        base = torch.arange(bs, device=DEVICE).unsqueeze(1) * DRAFT
        accept_index = (base + torch.arange(WIDTH, device=DEVICE)).to(torch.int64)
        seq_lens = torch.randint(
            1, 40, (bs,), device=DEVICE, dtype=torch.int64, generator=gen
        )
        track = (
            torch.randperm(SLOTS, device=DEVICE, generator=gen)[:bs]
            if with_track
            else None
        )

        _eager_reference(
            st_eager, mamba_map, req_idx, accept_lens, accept_index, seq_lens, track
        )
        ok = runner.commit(
            req_pool_indices=req_idx,
            accept_lens=accept_lens,
            accept_index=accept_index,
            seq_lens=seq_lens,
            mamba_track_indices=track,
        )
        self.assertTrue(ok)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(st_graph.temporal, st_eager.temporal))
        for cg, ce in zip(st_graph.conv, st_eager.conv):
            self.assertTrue(torch.equal(cg, ce))

    def test_graph_matches_eager(self):
        st = _make_state(3)
        st_graph = _clone_state(st)
        st_eager = _clone_state(st)
        mamba_map = torch.randperm(SLOTS, device=DEVICE, dtype=torch.int64)
        req_pool = SimpleNamespace(
            get_speculative_mamba2_params_all_layers=lambda: st_graph,
            req_index_to_mamba_index_mapping=mamba_map,
        )
        runner = ReplayssmCommitGraphRunner(
            req_pool=req_pool,
            draft_token_num=DRAFT,
            accept_width=WIDTH,
            track_interval=TRACK_INTERVAL,
            max_bs=SLOTS,
            device=torch.device(DEVICE),
        )
        self._run_case(3, True, runner, st_graph, st_eager, mamba_map, seed=11)
        self._run_case(3, True, runner, st_graph, st_eager, mamba_map, seed=12)
        self._run_case(3, False, runner, st_graph, st_eager, mamba_map, seed=13)
        self._run_case(5, True, runner, st_graph, st_eager, mamba_map, seed=14)
        self._run_case(5, True, runner, st_graph, st_eager, mamba_map, seed=15)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
