"""DFlash graph admission and KV pool bindings reach the draft runner."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.kv.memory.views import MHAKVView
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashDraftGraphEligibility(unittest.TestCase):
    def test_scheduler_admission_reaches_the_draft_forward_batch(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = 2
        worker.draft_model = SimpleNamespace()
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
        worker.draft_kv_ring = None
        worker.selector = None
        worker.lilicorr = None

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
        for allowed in (False, True):
            batch.can_run_decode_cuda_graph = allowed
            with (
                self.subTest(allowed=allowed),
                patch.object(
                    dflash_worker_v2, "draft_tp_context", return_value=nullcontext()
                ),
                patch.object(torch.Tensor, "record_stream"),
                patch.object(torch, "get_device_module", return_value=Mock()),
                patch.object(
                    dflash_worker_v2,
                    "assign_extend_cache_locs_func",
                    return_value=torch.tensor([3, 4]),
                ),
                patch.object(
                    dflash_worker_v2, "enable_num_token_non_padded", return_value=False
                ),
                self.assertRaises(StopIteration),
            ):
                worker.forward_batch_generation(batch)
            forward_batch = worker.draft_model_runner.forward.call_args.args[0]
            iteration_ids.bind.assert_called_with(forward_batch, draft_translator)
            self.assertIs(forward_batch.can_run_decode_cuda_graph, allowed)
            self.assertEqual(forward_batch.forward_mode, ForwardMode.TARGET_VERIFY)
            torch.testing.assert_close(forward_batch.input_ids, torch.tensor([1, 2]))


class TestDFlashBatchedKvAllocation(CustomTestCase):
    def test_batched_write_uses_each_subpools_physical_locations(self):
        """A window layer must never scatter at the full pool's physical ids."""
        worker = object.__new__(DFlashWorkerV2)
        worker._batched_kv_layers = None
        worker._batched_kv_buffers = None
        worker._fused_kv_helper = SimpleNamespace(
            flat_kv_weight_t=torch.empty(1, dtype=torch.bfloat16)
        )
        buffers = [torch.zeros(32, 2, 4, dtype=torch.bfloat16) for _ in range(4)]
        views = [MHAKVView() for _ in range(2)]
        for i, view in enumerate(views):
            view.get_key_buffer = Mock(return_value=buffers[2 * i])
            view.get_value_buffer = Mock(return_value=buffers[2 * i + 1])
        pool = SimpleNamespace()
        pool.layer_router = KVLayerRouter.from_layer_maps(
            pool, layer_maps=[("full", views[0], {0: 0}), ("window", views[1], {1: 0})]
        )
        worker.draft_model_runner = SimpleNamespace(token_to_kv_pool=pool)
        worker.draft_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(layer_id=i), num_kv_heads=2, head_dim=4
                    )
                )
                for i in range(2)
            ]
        )
        worker._init_batched_kv_write()
        full_loc, window_loc = torch.tensor([3, 4]), torch.tensor([17, 18])
        k = torch.ones(2, 2, 2, 4, dtype=torch.bfloat16)
        v = 2 * k

        def scatter(k, v, kc, vc, locs, **kwargs):
            for i, loc in enumerate(locs):
                kc[i][loc] = k[i]
                vc[i][loc] = v[i]

        with patch(
            "sglang.kernels.ops.kvcache.batched_store_cache.batched_store_cache",
            side_effect=scatter,
        ):
            writer = worker._make_stacked_kv_writer(full_loc, None, None, window_loc)
            writer(k, v)
        for index, loc in ((0, full_loc), (2, window_loc)):
            torch.testing.assert_close(buffers[index][loc], k[0])
            torch.testing.assert_close(buffers[index + 1][loc], v[0])
        self.assertEqual(buffers[2][full_loc].count_nonzero().item(), 0)

    def test_pool_replacement_rebuilds_batched_write_buffers(self):
        """A pool rebuild must discard the previous buffers before binding writes."""
        worker = object.__new__(DFlashWorkerV2)
        worker.draft_parallel_layout = None
        worker.use_compact_draft_cache = False
        worker._fused_kv_helper = object()
        worker._batched_kv_layers = ("old",)
        worker._batched_kv_buffers = ("old",)
        generation = []

        def allocate(**kwargs):
            generation.append(object())

        def bind():
            self.assertIsNone(worker._batched_kv_layers)
            self.assertIsNone(worker._batched_kv_buffers)
            worker._batched_kv_layers = (generation[-1],)
            worker._batched_kv_buffers = (generation[-1],)

        worker._draft_worker = SimpleNamespace(alloc_memory_pool=allocate)
        worker._init_batched_kv_write = bind
        with (
            patch.object(
                dflash_worker_v2, "draft_tp_context", return_value=nullcontext()
            ),
            patch.object(dflash_worker_v2, "is_cuda", return_value=True),
            patch.object(
                dflash_worker_v2,
                "get_spec",
                return_value=SimpleNamespace(speculative_dflash_draft_kv_ring=False),
            ),
            patch.object(
                dflash_worker_v2.envs.SGLANG_ENABLE_ASYNC_ASSERT,
                "get",
                return_value=False,
            ),
        ):
            for _ in range(2):
                worker.alloc_memory_pool()
                self.assertIs(worker._batched_kv_buffers[0], generation[-1])


if __name__ == "__main__":
    unittest.main()
