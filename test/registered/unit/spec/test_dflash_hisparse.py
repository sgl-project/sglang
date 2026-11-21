"""DFLASH verify must preserve accepted KV through the HiSparse commit lifecycle."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseSpecSwapManager,
)
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDFlashHiSparse(CustomTestCase):
    def test_staged_bonus_tokens_follow_filtered_and_merged_requests(self):
        """Staged requests carry relay indices, not empty idle-input token rows."""
        future_map = FutureMap.__new__(FutureMap)
        future_map.spec_algo = SpeculativeAlgorithm.DFLASH
        future_map.device = torch.device("cpu")
        future_map._deferred_relay = None
        future_map.output_tokens_buf = torch.full((10,), -1, dtype=torch.int64)
        future_map.output_tokens_buf[torch.tensor([2, 5, 8])] = torch.tensor(
            [22, 55, 88]
        )
        staged = future_map.make_staged_spec_input(torch.tensor([5, 2, 8]))
        staged.filter_batch(torch.tensor([2, 0]))
        staged.merge_batch(future_map.make_staged_spec_input(torch.tensor([2])))
        future_map._resolve_spec_extras(SimpleNamespace(spec_info=staged))

        self.assertIsInstance(staged, DFlashDraftInputV2)
        self.assertEqual(staged.bonus_tokens.tolist(), [88, 55, 22])
        self.assertEqual(staged.bonus_tokens.dtype, torch.int64)

    def test_verify_preserves_host_kv_beyond_the_hot_buffer(self):
        """A second commit reads cleared mappings and overwrites accepted host KV.

        Run the real worker and coordinator with tensor-backed transfer boundaries;
        mixed acceptance lengths also expose flattened-index mistakes across rows.
        """
        for seq_len, commit_lens in ((0, [3]), (10, [3]), (10, [3, 2]), (10, [1])):
            with self.subTest(seq_len=seq_len, commit_lens=commit_lens):
                self._run_verify(seq_len, commit_lens)

    def _run_verify(self, seq_len, commit_lens):
        bs, block_size = len(commit_lens), 4
        device_kv = torch.full((128, 2), -999, dtype=torch.int64)
        host_kv = torch.full((bs * 64, 2), -777, dtype=torch.int64)
        verify_locs = torch.arange(100, 100 + bs * block_size)
        req_to_token = torch.zeros((bs, 64), dtype=torch.int64)
        for row in range(bs):
            req_to_token[row, seq_len : seq_len + block_size] = verify_locs.view(
                bs, block_size
            )[row]

        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device_buffer_size = 4
        coordinator.page_size = 8
        coordinator.req_to_device_buffer = torch.arange(20, 20 + bs * 12).view(bs, 12)
        coordinator.req_device_buffer_tokens = torch.full(
            (1, bs, 12), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full_like(
            coordinator.req_device_buffer_tokens, -1
        )
        coordinator.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        coordinator.req_to_host_pool = torch.arange(bs * 64).view(bs, 64)
        coordinator.req_to_host_pool_allocated_len = torch.full((bs,), 64)
        mapping = torch.zeros(256, dtype=torch.int64)
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_hisparse_device_index_mapping=mapping
        )
        coordinator._skip_first_backup = [False] * bs

        def transfer(dst_indices, src_indices):
            device_kv[dst_indices] = device_kv[src_indices].clone()

        def backup(host_locs, device_locs):
            host_kv[host_locs] = device_kv[device_locs].clone()

        coordinator.mem_pool_device = SimpleNamespace(
            transfer_values_on_device=transfer
        )
        coordinator.spec_swap = object.__new__(HiSparseSpecSwapManager)
        coordinator.spec_swap._coordinator = coordinator
        coordinator.spec_swap.enabled = True
        coordinator.spec_swap.num_draft_tokens = block_size
        coordinator.spec_swap._backup_device_locs_to_host = backup

        # Model forwards and transfers are the external boundaries. Acceptance,
        # verify-slot binding, and host/device mapping transitions stay real.
        expected_kv = torch.arange(bs * block_size * 2).view(-1, 2) + 1000
        predictions = torch.tensor([[2, 3, 4, 9], [6, 7, 8, 9]])[:bs].clone()
        for row, length in enumerate(commit_lens):
            predictions[row, length - 1 :] = 9
        logits = torch.full((bs * block_size, 16), -100.0)
        logits.scatter_(1, predictions.reshape(-1, 1), 100.0)

        def target_forward(**kwargs):
            device_kv[mapping[verify_locs]] = expected_kv
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=logits,
                    hidden_states=torch.zeros(bs * block_size, 4),
                ),
                can_run_cuda_graph=False,
                routed_experts_output=None,
                indexer_topk_output=None,
                kv_weight_version_record=None,
            )

        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = block_size
        worker.draft_model = SimpleNamespace()
        embedding = torch.nn.Embedding(16, 4)
        target_model = SimpleNamespace(
            get_input_embeddings=lambda: embedding, lm_head=torch.nn.Linear(4, 16)
        )
        iteration_ids = SimpleNamespace(
            bind=Mock(), write_ids=lambda *args, **kw: verify_locs
        )
        worker.model_runner = SimpleNamespace(
            model=target_model,
            req_to_token_pool=coordinator.req_to_token_pool,
            runner_ids=SimpleNamespace(plan=lambda **kw: iteration_ids),
        )
        worker._target_worker = SimpleNamespace(
            model_runner=worker.model_runner, forward_batch_generation=target_forward
        )
        worker.draft_model_runner = SimpleNamespace(
            runner_ids=SimpleNamespace(),
            forward=lambda batch: SimpleNamespace(
                can_run_graph=True, logits_output=None
            ),
        )
        worker._draft_block_ids_buf = None
        worker._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=block_size, device="cpu"
        )
        worker._block_pos_offsets = torch.arange(block_size)
        worker._use_triton_prepare_block = False
        worker._mask_token_id = 0
        worker._full_embed_gpu = None
        worker._noise_embed_scale = 1.0
        worker.use_compact_draft_cache = False
        worker.draft_kv_ring = None
        worker.selector = worker.lilicorr = None
        worker._is_domino = False
        worker._draft_sampler = SimpleNamespace(
            out=torch.tensor([2, 3, 4, 6, 7, 8])[: bs * 3]
        )
        worker._selector_sample = None
        worker._target_logits_processor = None
        worker._use_triton_accept_bonus = False
        worker._tp_sync = SimpleNamespace(sync=lambda *args: None)
        worker._need_mamba_verify_commit = False
        worker._append_target_hidden_to_draft_kv_by_loc = Mock()

        draft_input = DFlashDraftInputV2.create_idle_input(device="cpu")
        draft_input.bonus_tokens = torch.tensor([1, 5])[:bs]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            sampling_info=None,
            spec_info=draft_input,
            seq_lens=torch.full((bs,), seq_len),
            seq_lens_cpu=torch.full((bs,), seq_len),
            seq_lens_sum=bs * seq_len,
            req_pool_indices=torch.arange(bs),
            reqs=[
                SimpleNamespace(kv=SimpleNamespace(req_pool_idx=r, kv_allocated_len=64))
                for r in range(bs)
            ],
            hisparse_coordinator=coordinator,
            can_run_decode_cuda_graph=False,
            has_grammar=False,
            return_logprob=False,
        )
        with (
            get_parallel().override(attn_dp_enabled=False),
            patch.object(
                dflash_worker_v2, "draft_tp_context", return_value=nullcontext()
            ),
            patch.object(torch.Tensor, "record_stream"),
            patch.object(torch, "get_device_module", return_value=Mock()),
            patch.object(
                dflash_worker_v2,
                "assign_extend_cache_locs_func",
                return_value=verify_locs,
            ),
            patch.object(
                dflash_worker_v2, "enable_num_token_non_padded", return_value=False
            ),
            patch.object(
                dflash_worker_v2.DFlashVerifyInput,
                "prepare_for_verify",
                return_value=(SimpleNamespace(), False),
            ),
        ):
            result = worker.forward_batch_generation(batch)

        self.assertEqual(result.accept_lens.tolist(), commit_lens)
        self.assertEqual(
            result.new_seq_lens.tolist(), [seq_len + n for n in commit_lens]
        )
        for row, length in enumerate(commit_lens):
            host_locs = coordinator.req_to_host_pool[row, seq_len : seq_len + length]
            torch.testing.assert_close(
                host_kv[host_locs],
                expected_kv[row * block_size : row * block_size + length],
            )


if __name__ == "__main__":
    unittest.main()
