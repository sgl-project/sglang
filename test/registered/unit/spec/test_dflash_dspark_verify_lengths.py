import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import (
    build_ragged_verify_token_buckets,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import _copy_prefix_seq_lens_cpu
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _spec_args(*, draft_tokens: int = 8, algorithm: str = "DSPARK") -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.speculative_num_draft_tokens = draft_tokens
    args.speculative_num_steps = 1
    args.speculative_eagle_topk = 1
    args.page_size = 1
    return args


class _FakeBatch:
    def __init__(self, *, committed_lens, allocated_lens, row_width: int):
        self.device = torch.device("cpu")
        self.reqs = [
            SimpleNamespace(
                kv_committed_len=committed_len,
                kv_allocated_len=allocated_len,
                sampling_params=SimpleNamespace(top_k=1),
            )
            for committed_len, allocated_len in zip(committed_lens, allocated_lens)
        ]
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.empty((len(self.reqs), row_width), dtype=torch.int32)
        )

    def batch_size(self):
        return len(self.reqs)


class _FakeTargetWorker:
    def __init__(self):
        self.model_runner = SimpleNamespace(attn_backend=SimpleNamespace())

    def forward_batch_generation(self, **kwargs):
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.empty(0)),
            can_run_cuda_graph=False,
        )


class TestDFlashDSparkVerifyLengths(CustomTestCase):
    def test_ragged_verify_token_buckets_default_to_full_blocks(self):
        with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.override(0):
            self.assertEqual(
                build_ragged_verify_token_buckets(
                    capture_bs=[1, 2, 4], num_tokens_per_req=8
                ),
                [8, 16, 32],
            )

    def test_ragged_verify_token_buckets_can_add_fine_grained_tiers(self):
        with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.override(8):
            self.assertEqual(
                build_ragged_verify_token_buckets(
                    capture_bs=[1, 2, 4], num_tokens_per_req=8
                ),
                [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
            )

    def test_prepare_for_decode_keeps_committed_and_reserved_lengths_separate(self):
        args = _spec_args(draft_tokens=4)
        set_global_server_args_for_scheduler(args)
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        batch = _FakeBatch(
            committed_lens=[10, 20],
            allocated_lens=[18, 28],
            row_width=64,
        )

        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(False):
            draft_input.prepare_for_decode(batch)

        self.assertEqual(batch.seq_lens_cpu.tolist(), [10, 20])
        self.assertEqual(batch.seq_lens_sum, 30)
        self.assertEqual(draft_input.reserved_seq_lens_cpu.tolist(), [18, 28])
        self.assertEqual(draft_input.reserved_seq_lens_sum, 46)

    def test_prepare_for_decode_fails_before_req_to_token_oob(self):
        args = _spec_args(draft_tokens=8)
        set_global_server_args_for_scheduler(args)
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        batch = _FakeBatch(committed_lens=[8], allocated_lens=[24], row_width=23)

        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(False):
            with self.assertRaisesRegex(AssertionError, "over-allocation"):
                draft_input.prepare_for_decode(batch)

    def test_dflash_draft_block_uses_prefix_seq_lens_cpu(self):
        dst = torch.empty((2,), dtype=torch.int32)
        prefix_lens = torch.tensor([10, 20], dtype=torch.int32)
        host_lens = torch.tensor([10, 20], dtype=torch.int64)

        seq_lens_sum = _copy_prefix_seq_lens_cpu(dst, prefix_lens, host_lens)
        self.assertEqual(dst.tolist(), [10, 20])
        self.assertEqual(seq_lens_sum, 30)

        dst.fill_(-1)
        seq_lens_sum = _copy_prefix_seq_lens_cpu(dst, prefix_lens, None)
        self.assertEqual(dst.tolist(), [10, 20])
        self.assertEqual(seq_lens_sum, 30)

    def test_dspark_draft_proposer_passes_prefix_seq_lens_cpu(self):
        seen = {}
        gamma = 4
        bs = 2
        draft_width = gamma + 1

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["seq_lens"] = forward_batch.seq_lens.clone()
                seen["seq_lens_cpu"] = forward_batch.seq_lens_cpu.clone()
                seen["seq_lens_sum"] = forward_batch.seq_lens_sum
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=torch.empty((bs * draft_width, 16))
                    ),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_sum=30,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
            verify_cache_loc_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
        )

        proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, 16),
        )

        self.assertEqual(seen["seq_lens"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)

    def test_dspark_draft_proposer_derives_cpu_lens_from_gpu_only_batch(self):
        seen = {}
        gamma = 4
        bs = 2
        draft_width = gamma + 1

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["seq_lens"] = forward_batch.seq_lens.clone()
                seen["seq_lens_cpu"] = forward_batch.seq_lens_cpu.clone()
                seen["seq_lens_sum"] = forward_batch.seq_lens_sum
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=torch.empty((bs * draft_width, 16))
                    ),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
            reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
            reserved_seq_lens_sum=46,
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
            verify_cache_loc_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
        )

        proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, 16),
        )

        self.assertEqual(seen["seq_lens"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)

    def test_dspark_target_verify_passes_prefix_seq_lens_cpu_not_reserved(self):
        target_worker = _FakeTargetWorker()
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            gamma=3,
            verify_num_draft_tokens=4,
            model_runner=target_worker.model_runner,
            kv_injector=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=None,
        )
        draft_input = SimpleNamespace(
            reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
            reserved_seq_lens_sum=46,
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(8, dtype=torch.int64).view(2, 4),
            verify_cache_loc=torch.arange(8, dtype=torch.int64),
        )
        seen = {}

        def capture_prepare(_verify_input, verify_batch, _target_worker):
            seen["seq_lens_cpu"] = verify_batch.seq_lens_cpu.clone()
            seen["seq_lens_sum"] = verify_batch.seq_lens_sum
            return SimpleNamespace(), False

        with patch.object(DFlashVerifyInput, "prepare_for_verify", new=capture_prepare):
            executor.run_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=torch.ones((2, 4), dtype=torch.int64),
                verify_window=verify_window,
                sampling_info=None,
            )

        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)

    def test_dspark_ragged_verify_passes_prefix_seq_lens_cpu(self):
        target_worker = _FakeTargetWorker()
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            gamma=3,
            verify_num_draft_tokens=4,
            model_runner=target_worker.model_runner,
            kv_injector=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_sum=30,
            out_cache_loc=None,
        )
        layout = SimpleNamespace(
            verify_lens_cpu=[1, 3],
            verify_lens=torch.tensor([1, 3], dtype=torch.int32),
        )
        ragged_window = SimpleNamespace(
            verify_ids=torch.arange(4, dtype=torch.int64),
            positions=torch.arange(4, dtype=torch.int64),
            verify_cache_loc=torch.arange(4, dtype=torch.int64),
        )
        seen = {}

        def capture_prepare(_verify_input, verify_batch, _target_worker):
            seen["seq_lens_cpu"] = verify_batch.seq_lens_cpu.clone()
            seen["seq_lens_sum"] = verify_batch.seq_lens_sum
            return SimpleNamespace(), False

        with patch.object(DFlashVerifyInput, "prepare_for_verify", new=capture_prepare):
            executor._run_ragged(
                batch=batch,
                layout=layout,
                ragged_window=ragged_window,
                sampling_info=None,
            )

        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [10, 20])
        self.assertEqual(batch.seq_lens_sum, 30)

    def test_dflash_family_dummy_verify_input_uses_caller_width(self):
        args = _spec_args(draft_tokens=8)
        spec_algorithm = SpeculativeAlgorithm.DSPARK

        spec_info = create_dummy_verify_input(
            spec_algorithm=spec_algorithm,
            server_args=args,
            custom_mask=torch.empty(0, dtype=torch.bool),
            num_tokens_per_req=5,
            is_draft_worker=True,
        )

        self.assertEqual(spec_info.draft_token_num, 5)


if __name__ == "__main__":
    unittest.main()
