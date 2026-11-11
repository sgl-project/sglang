"""Rejected draft graphs must not write bounded sampling buffers."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _SelectorDraftSampler,
)
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashDraftGraphEligibility(CustomTestCase):
    def test_sampler_staging_follows_graph_admission(self):
        """Oversized or rejected batches used to stage before graph admission."""
        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = 2
        worker.draft_model = SimpleNamespace(
            candidate_selector=SimpleNamespace(top_k=2)
        )
        embedding = torch.nn.Embedding(16, 4)
        target_model = SimpleNamespace(
            get_input_embeddings=lambda: embedding, lm_head=torch.nn.Linear(4, 16)
        )
        iteration_ids = SimpleNamespace(bind=Mock())
        target_translator = SimpleNamespace(plan=Mock(return_value=iteration_ids))
        draft_translator = SimpleNamespace()
        worker.model_runner = SimpleNamespace(
            runner_ids=target_translator,
            model=target_model,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(16).view(1, 16)
            ),
        )
        worker._target_worker = SimpleNamespace(model_runner=worker.model_runner)
        # Stop at the actual runner boundary, before target verification.
        worker.draft_model_runner = SimpleNamespace(
            runner_ids=draft_translator,
            forward=Mock(side_effect=StopIteration),
        )
        worker.draft_owns_attention = False
        worker._draft_block_ids_buf = None
        worker._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=2, device="cpu"
        )
        worker._block_pos_offsets = torch.arange(2)
        worker._use_triton_prepare_block = False
        worker._mask_token_id = 2
        worker._full_embed_gpu = None
        worker._noise_embed_scale = 1.0
        worker.use_compact_draft_cache = False
        worker.selector = worker.draft_model.candidate_selector
        worker._selector_sampling_enabled = True
        worker.lilicorr = None
        worker._draft_sampler_max_bs = 3
        sampler = worker._draft_sampler = _SelectorDraftSampler(
            draft_model=worker.draft_model,
            block_size=2,
            max_bs=3,
            device="cpu",
            sampling_enabled=True,
        )

        draft_input = DFlashDraftInputV2.create_idle_input(device="cpu")
        draft_input.bonus_tokens = torch.tensor([1])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            sampling_info=None,
            spec_info=draft_input,
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            req_pool_indices=torch.tensor([0]),
        )
        for bs, allowed, runner_device, has_runner in (
            (3, True, "cuda", True),  # Exactly at capacity.
            (4, True, "cuda", True),  # Eager fallback beyond capacity.
            (3, False, "cuda", True),  # Scheduler / variant rejection.
            (3, True, "cpu", True),  # TARGET_VERIFY is not a CPU graph mode.
            (3, True, "cuda", False),
        ):
            with self.subTest(
                bs=bs, allowed=allowed, device=runner_device, has_runner=has_runner
            ):
                draft_input.bonus_tokens = torch.ones(bs, dtype=torch.int64)
                batch.seq_lens = batch.seq_lens_cpu = torch.full((bs,), 3)
                batch.req_pool_indices = torch.arange(bs)
                batch.sampling_info = SimpleNamespace(
                    temperatures=torch.full((bs, 1), 0.7),
                    top_ks=torch.arange(bs) + 1,
                    is_all_greedy=False,
                )
                batch.can_run_decode_cuda_graph = allowed
                worker.draft_model_runner.device = runner_device
                worker.draft_model_runner.decode_cuda_graph_runner = (
                    SimpleNamespace(
                        can_run_graph=lambda fb: (
                            fb.batch_size <= 3 and fb.can_run_decode_cuda_graph
                        )
                    )
                    if has_runner
                    else None
                )
                sampler.temperatures.fill_(1.0)
                sampler.greedy_mask.fill_(True)
                temperature_ptr = sampler.temperatures.data_ptr()
                mask_ptr = sampler.greedy_mask.data_ptr()
                with (
                    patch.object(
                        dflash_worker_v2, "draft_tp_context", return_value=nullcontext()
                    ),
                    patch.object(torch.Tensor, "record_stream"),
                    patch.object(torch, "get_device_module", return_value=Mock()),
                    patch.object(
                        dflash_worker_v2,
                        "assign_extend_cache_locs_func",
                        return_value=torch.tensor([3, 4]).repeat(bs),
                    ),
                    patch.object(
                        dflash_worker_v2,
                        "enable_num_token_non_padded",
                        return_value=False,
                    ),
                    self.assertRaises(StopIteration),
                ):
                    worker.forward_batch_generation(batch)
                forward_batch = worker.draft_model_runner.forward.call_args.args[0]
                iteration_ids.bind.assert_called_with(forward_batch, draft_translator)
                self.assertIs(forward_batch.can_run_decode_cuda_graph, allowed)
                self.assertEqual(forward_batch.forward_mode, ForwardMode.TARGET_VERIFY)
                torch.testing.assert_close(
                    forward_batch.input_ids, torch.tensor([1, 2]).repeat(bs)
                )
                staged = bs <= 3 and allowed and runner_device == "cuda" and has_runner
                torch.testing.assert_close(
                    sampler.temperatures, torch.full((3,), 0.7 if staged else 1.0)
                )
                torch.testing.assert_close(
                    sampler.greedy_mask,
                    torch.tensor([True, False, False])
                    if staged
                    else torch.ones(3, dtype=torch.bool),
                )
                self.assertEqual(sampler.temperatures.data_ptr(), temperature_ptr)
                self.assertEqual(sampler.greedy_mask.data_ptr(), mask_ptr)

    def test_sampler_capacity_uses_resolved_buckets(self):
        """The request pool can clamp raw graph buckets and add a new maximum."""
        worker = object.__new__(DFlashWorkerV2)
        worker.block_size = 2
        worker.device = "cpu"
        worker._target_tp_rank = 0
        worker._selector_sampling_enabled = True
        worker.selector = SimpleNamespace(top_k=2)
        worker.draft_model = SimpleNamespace(candidate_selector=worker.selector)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(lm_head=torch.nn.Linear(4, 16))
            )
        )
        runner = worker.draft_model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(size=3),
            server_args=SimpleNamespace(),
            decode_num_tokens_per_req=Mock(return_value=2),
            decode_graph_gather_requirements=Mock(return_value=(False, False)),
        )
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(bs=[1, 2, 4, 8])
                )
            )
        )
        with (
            patch.object(
                dflash_worker_v2.envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER,
                "get",
                return_value=False,
            ),
            patch.object(dflash_worker_v2, "get_exec", return_value=execution),
            patch.object(
                dflash_worker_v2,
                "get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=2),
            ),
            patch.object(
                dflash_worker_v2,
                "get_batch_sizes_to_capture",
                return_value=([1, 2, 3], []),
            ) as resolve_buckets,
        ):
            sampler = worker._maybe_build_draft_sampler()
        self.assertEqual(sampler.temperatures.numel(), runner.req_to_token_pool.size)
        self.assertEqual(worker._draft_sampler_max_bs, runner.req_to_token_pool.size)
        self.assertEqual(sampler.out.numel(), 3 * (worker.block_size - 1))
        resolve_buckets.assert_called_once_with(
            runner, 2, gathered_buffer_required=False
        )
        runner.decode_num_tokens_per_req.assert_called_once_with(num_draft_tokens=2)
        runner.decode_graph_gather_requirements.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
