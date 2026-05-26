import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.mamba2_attention import (
    DEFAULT_CONV_KERNEL,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAMBA_CHUNK_SIZE,
    DEFAULT_N_GROUPS,
    DEFAULT_NUM_HEADS,
    DEFAULT_STATE_SIZE,
    Mamba2AttentionCase,
    build_mamba2_attention_fixture,
    run_mamba2_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonMamba2BackendCorrectness(CustomTestCase):
    CASES = (
        Mamba2AttentionCase(
            name="mamba2_extend_zero_prefix_exact_page",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=DEFAULT_NUM_HEADS,
            head_dim=DEFAULT_HEAD_DIM,
            state_size=DEFAULT_STATE_SIZE,
            n_groups=DEFAULT_N_GROUPS,
            conv_kernel=DEFAULT_CONV_KERNEL,
            mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
        ),
    )
    # Decode case for the replay metadata assertion below. It is
    # intentionally constructed so that the replay seq_lens_cpu contains
    # the cuda-graph fill value (1) in some rows, so a
    # `seq_lens_cpu - 1` mutation in
    # `MambaAttnBackendBase._replay_metadata` (M21) changes the
    # computed `num_padding` and therefore the trailing `-1`
    # bookkeeping in `state_indices_list[bs - 1]`.
    REPLAY_METADATA_CASE = Mamba2AttentionCase(
        name="mamba2_decode_replay_metadata_padding",
        backend="triton",
        forward_mode=ForwardMode.DECODE,
        num_heads=DEFAULT_NUM_HEADS,
        head_dim=DEFAULT_HEAD_DIM,
        state_size=DEFAULT_STATE_SIZE,
        n_groups=DEFAULT_N_GROUPS,
        conv_kernel=DEFAULT_CONV_KERNEL,
        mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        page_size=16,
        prefix_lens=(4, 0, 0),
    )

    def test_projected_mamba2_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_mamba2_attention_case(self, case)

    def test_mamba2_replay_metadata_padding_indices(self):
        # The wrapper-level `seq_lens_cpu` mutation in
        # `init_forward_metadata_replay_cuda_graph` (M21) is
        # invisible to forward output as long as the replay seq_lens
        # never include the cuda-graph fill value (1). It also can't
        # be exercised by the GDN cuda-graph runner because that
        # adapter uses `allow_padding=False`. We therefore drive the
        # `Mamba2AttnBackend.init_forward_metadata_replay_cuda_graph`
        # entrypoint directly with a hand-rolled replay batch where
        # two of the three rows are at the fill value, and assert
        # that exactly those two trailing rows in
        # `state_indices_list[bs - 1]` are set to `-1`.
        #
        # Under M21, `count_nonzero(seq_lens_cpu - 1 == 1)` ==
        # `count_nonzero(seq_lens_cpu == 2)`, which for our replay
        # `seq_lens_cpu = [5, 1, 1]` returns 0 instead of the
        # expected 2, so the assertion fires.
        case = self.REPLAY_METADATA_CASE
        fixture = build_mamba2_attention_fixture(
            self,
            case,
            disable_cuda_graph=False,
            runner_batch_size=case.batch_size,
        )
        backend = fixture.backend
        bs = case.batch_size

        # Allocate the cuda-graph state buffers. We need
        # `state_indices_list[bs - 1]` to have length `bs` so the
        # padding-trailer slice is non-empty.
        backend.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs)

        # Seed the per-bs state indices buffer with a sentinel so we
        # can tell the difference between "never written" and
        # "written and then overwritten with -1".
        backend.state_indices_list[bs - 1].fill_(99)

        # Construct a replay batch where the last two rows are at the
        # cuda-graph fill value (1). The mamba index for req 0 is
        # whatever the pool returns; the trailing rows must be -1.
        device = fixture.runner.device
        req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
        # First request gets a non-fill seq_len; remaining two are
        # the fill value.
        seq_lens_cpu = torch.tensor([5, 1, 1], dtype=torch.int32, device="cpu")
        seq_lens = seq_lens_cpu.to(device=device)

        # Map req 0 to mamba slot 7 so we can verify it stays in the
        # state_indices_list (i.e. did NOT get overwritten by the
        # padding -1).
        fixture.runner.req_to_token_pool.req_index_to_mamba_index_mapping[
            req_pool_indices
        ] = torch.tensor([7, 0, 0], dtype=torch.int32, device=device)

        backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=int(seq_lens_cpu.sum().item()),
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=seq_lens_cpu,
        )

        state_indices = backend.state_indices_list[bs - 1].cpu().tolist()
        self.assertEqual(
            state_indices,
            [7, -1, -1],
            "`MambaAttnBackendBase._replay_metadata` must use the "
            "unmutated `seq_lens_cpu` to count cuda-graph padding rows "
            "(== fill value 1). With `seq_lens_cpu - 1` (M21) the "
            "padding count for `[5, 1, 1]` drops from 2 to 0, leaving "
            "the trailing rows holding the real mamba indices instead "
            "of -1.",
        )


if __name__ == "__main__":
    unittest.main()
