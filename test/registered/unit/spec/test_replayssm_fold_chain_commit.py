"""Chain fold commit (draft-extend-graph body) == the generic eager commit.

The chain variant drops the accept_index table (linear-chain acceptance is a
row prefix) and writes through caller-owned buffers so the body is
allocation-free and capturable. Pins end-state equality (temporal + conv)
against the generic accept_index-based commit, including: graph padding rows
(raw_bs < launched rows -> slot -1, untouched state), track disabled, track
crossing on/off, and the +draft seq_lens offset used by the graph buffers.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_after_verify,
    commit_replayssm_fold_chain,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")

L, SLOTS, HV, H, K, V = 2, 16, 8, 4, 64, 64
DRAFT, DIM, KM1 = 4, 32, 3
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


def _clone(st):
    return SimpleNamespace(
        temporal=st.temporal.clone(),
        replayssm_rawv=st.replayssm_rawv,
        replayssm_rawk=st.replayssm_rawk,
        replayssm_g=st.replayssm_g,
        replayssm_beta=st.replayssm_beta,
        conv=[c.clone() for c in st.conv],
        intermediate_conv_window=st.intermediate_conv_window,
    )


def _generic_reference(st, mamba_map, req_idx, accept_lens, seq_lens, track):
    bs = accept_lens.shape[0]
    slots = mamba_map[req_idx]
    last_correct = (accept_lens - 1).to(torch.int64)
    steps = None
    if track is not None:
        pre, post = seq_lens, seq_lens + accept_lens
        crossed = pre // TRACK_INTERVAL != post // TRACK_INTERVAL
        point = post // TRACK_INTERVAL * TRACK_INTERVAL
        ith = torch.clamp(point - pre - 1, min=0).to(torch.int64)
        steps = torch.where(crossed, ith, torch.full_like(ith, -1))
    commit_gdn_replayssm_fold_after_verify(
        spec_state=st,
        state_batch_indices=slots,
        accept_lens=accept_lens,
        last_correct_step_indices=last_correct,
        mamba_track_indices=track,
        mamba_steps_to_track=steps,
        null_block_id=-1,
    )


class TestReplayssmFoldChainCommit(CustomTestCase):
    def _run(self, raw_bs, launched_bs, with_track, offset, seed):
        st = _make_state(seed)
        st_chain, st_ref = _clone(st), _clone(st)
        gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
        mamba_map = torch.randperm(SLOTS, device=DEVICE, dtype=torch.int64)
        req_idx = torch.randperm(SLOTS, device=DEVICE, generator=gen)[:launched_bs]
        accept_lens = torch.randint(
            1,
            DRAFT + 1,
            (launched_bs,),
            device=DEVICE,
            dtype=torch.int32,
            generator=gen,
        )
        seq_lens = torch.cat(
            [
                torch.tensor([TRACK_INTERVAL - 2, TRACK_INTERVAL, 3], device=DEVICE),
                torch.randint(1, 40, (launched_bs,), device=DEVICE, generator=gen),
            ]
        )[:launched_bs].to(torch.int64)
        track = (
            torch.randperm(SLOTS, device=DEVICE, generator=gen)[:launched_bs]
            if with_track
            else None
        )

        commit_replayssm_fold_chain(
            spec_state=st_chain,
            accept_lens=accept_lens,
            seq_lens=seq_lens + offset,
            req_pool_indices=req_idx,
            mamba_track_indices=track,
            mamba_map=mamba_map,
            raw_bs_tensor=torch.full((1,), raw_bs, dtype=torch.int32, device=DEVICE),
            seq_lens_offset=offset,
            track_interval=TRACK_INTERVAL,
            out_slots=torch.empty(launched_bs, dtype=torch.int64, device=DEVICE),
            out_last_correct=torch.empty(launched_bs, dtype=torch.int64, device=DEVICE),
            out_track_indices=torch.empty(
                launched_bs, dtype=torch.int64, device=DEVICE
            ),
            out_track_steps=torch.empty(launched_bs, dtype=torch.int64, device=DEVICE),
        )
        _generic_reference(
            st_ref,
            mamba_map,
            req_idx[:raw_bs],
            accept_lens[:raw_bs],
            seq_lens[:raw_bs],
            track[:raw_bs] if track is not None else None,
        )
        self.assertTrue(
            torch.equal(st_chain.temporal, st_ref.temporal),
            f"{raw_bs=} {launched_bs=} {with_track=} {offset=}",
        )
        for cc, cr in zip(st_chain.conv, st_ref.conv):
            self.assertTrue(torch.equal(cc, cr))

    def test_exact_bs(self):
        self._run(3, 3, True, 0, seed=5)
        self._run(3, 3, False, 0, seed=6)

    def test_padded_rows_are_skipped(self):
        self._run(3, 6, True, 0, seed=7)
        self._run(1, 4, False, 0, seed=8)

    def test_graph_seq_lens_offset(self):
        self._run(4, 4, True, DRAFT, seed=9)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
