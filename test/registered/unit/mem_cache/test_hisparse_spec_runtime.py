"""CPU tests of eager runtime calls; only GPU compute/copy/event leaves are fake."""

import contextlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_spec_coordinator import HiSparseSpecCoordinator
from sglang.srt.mem_cache.hisparse_spec_runtime import (
    EagerHiSparseBatch,
    eager_hisparse_worker_boundary,
    restore_staging_drafts,
    save_staging_draft,
    validate_eager_layout,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput


class Event:
    def __init__(self):
        self.done = False

    def record(self, stream):
        pass

    def query(self):
        return self.done


class TestEagerRuntime(unittest.TestCase):
    def setUp(self):
        self.events = []

        def event():
            value = Event()
            self.events.append(value)
            return value

        self.dm = SimpleNamespace(
            Event=event,
            current_stream=lambda: None,
            stream=lambda _: contextlib.nullcontext(),
            synchronize=self.synchronize,
        )
        self.allocator = HiSparseTokenToKVPoolAllocator(
            256, 64, torch.float32, "cpu", MagicMock(), False, 16
        )
        hot = self.allocator.hisparse_attn_allocator.alloc(64)
        self.host = MagicMock(page_size=64)
        self.host.alloc_page.return_value = torch.arange(256, 320)
        mapping = torch.full((1, 256), -1, dtype=torch.int64)
        mapping[0, :64] = torch.arange(128, 192)
        self.c = SimpleNamespace(
            is_dsv4_hisparse=False,
            is_m3_hisparse=False,
            compress_ratio=1,
            mem_pool_device=SimpleNamespace(page_size=64, layer_num=1),
            mem_pool_host=self.host,
            token_to_kv_pool_allocator=self.allocator,
            device="cpu",
            _prepare_speculative_stream=MagicMock(),
            _copy_speculative_union=MagicMock(),
            device_buffer_size=4,
            req_device_buffer_size=torch.tensor([64]),
            req_device_buffer_tokens=torch.full((2, 1, 4), -1),
            req_device_buffer_token_locs=hot[:4].view(1, 1, 4).repeat(2, 1, 1),
            lru_slots=torch.arange(4).view(1, 1, 4).repeat(2, 1, 1),
            req_to_host_pool=mapping,
            req_to_host_pool_allocated_len=torch.tensor([64]),
            decode_backup_stream=MagicMock(),
            decode_producer_stream=None,
        )
        self.adapter = HiSparseSpecCoordinator(self.c, self.dm)
        self.c.speculative_verifier = lambda: self.adapter
        self.req = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=0), hisparse_staging=False
        )
        self.batch = SimpleNamespace(
            reqs=[self.req],
            out_cache_loc=torch.arange(2047, 2051),
            seq_lens=torch.tensor([63]),
            hisparse_coordinator=self.c,
            is_extend_in_batch=False,
            forward_mode=SimpleNamespace(is_extend=lambda: False),
        )
        self.forward = SimpleNamespace()
        self.transaction = EagerHiSparseBatch(self.c)

    def synchronize(self):
        for event in self.events:
            event.done = True

    def prepare(self):
        self.transaction.prepare(self.batch, self.forward, 4, 8)

    def add_second_request(self):
        hot = self.allocator.hisparse_attn_allocator.alloc(64)
        self.c.req_device_buffer_size = torch.tensor([64, 64])
        self.c.req_device_buffer_tokens = self.c.req_device_buffer_tokens.repeat(
            1, 2, 1
        )
        self.c.req_device_buffer_token_locs = torch.cat(
            (
                self.c.req_device_buffer_token_locs,
                hot[:4].view(1, 1, 4).repeat(2, 1, 1),
            ),
            dim=1,
        )
        self.c.lru_slots = self.c.lru_slots.repeat(1, 2, 1)
        second_host = torch.full((1, 256), -1, dtype=torch.int64)
        second_host[0, :64] = torch.arange(512, 576)
        self.c.req_to_host_pool = torch.cat((self.c.req_to_host_pool, second_host))
        self.c.req_to_host_pool_allocated_len = torch.tensor([64, 64])
        second = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=1), hisparse_staging=False
        )
        self.batch.reqs.append(second)
        self.batch.seq_lens = torch.tensor([63, 3])
        self.batch.out_cache_loc = torch.tensor(
            [2047, 2048, 2049, 2050, 3071, 3072, 3073, 3074]
        )
        return second

    def test_mixed_lengths_and_different_unions_preserve_request_row_order(self):
        self.add_second_request()
        self.prepare()
        table = self.transaction.stage_layer(
            0,
            torch.tensor(
                [
                    [0, 63, -1],
                    [0, 64, -1],
                    [0, 65, -1],
                    [0, 66, -1],
                    [0, 1, 3],
                    [2, 4, -1],
                    [0, 5, -1],
                    [1, 6, -1],
                ]
            ),
            9,
        )
        copies = self.c._copy_speculative_union.call_args_list
        self.assertEqual([int(call.args[3][0]) for call in copies], [1, 3])
        self.assertEqual(table[-1].tolist(), [-1, -1, -1])
        self.assertEqual(
            int(table[0, 1]),
            int(self.allocator.full_to_hisparse_device_index_mapping[2047]),
        )
        self.assertEqual(
            int(table[4, 2]),
            int(self.allocator.full_to_hisparse_device_index_mapping[3071]),
        )
        self.transaction.target_done()
        self.transaction.commit(
            torch.tensor([1, 3]), torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        )
        self.transaction.close(failed=False)
        self.assertEqual(self.adapter.backend.valid_lengths, {0: (0, 64), 1: (0, 6)})

    def test_mid_prepare_failure_releases_first_arena_and_armed_readers(self):
        second = self.add_second_request()
        second.hisparse_staging = True
        physical = self.allocator.hisparse_attn_allocator.available_size()
        logical = self.allocator.logical_attn_allocator.available_size()

        def forward(worker, batch, on_publish=None):
            batch.hisparse_spec_transaction.prepare(batch, self.forward, 4, 8)

        with (
            envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True),
            self.assertRaisesRegex(ValueError, "staging"),
        ):
            eager_hisparse_worker_boundary(forward)(None, self.batch)
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(), physical
        )
        self.assertEqual(
            self.allocator.logical_attn_allocator.available_size(), logical
        )
        self.assertTrue(
            (self.allocator.full_to_hisparse_device_index_mapping[2047:2051] == 0).all()
        )
        self.assertIsNone(self.adapter.owners[0].key)

    def test_second_commit_failure_cancels_first_without_host_publication(self):
        self.add_second_request()
        self.prepare()
        self.transaction.stage_layer(0, torch.tensor([[0]] * 8), 8)
        self.transaction.target_done()
        self.host.backup_from_device_all_layer.side_effect = [
            None,
            RuntimeError("copy failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.transaction.commit(
                torch.tensor([2, 2]), torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
            )
        self.transaction.close(failed=True)
        self.assertEqual(self.adapter.backend.valid_lengths, {0: (0, 63), 1: (0, 3)})
        self.assertTrue(
            all(owner.key is None for owner in self.adapter.owners.values())
        )
        self.host.free.assert_called_once()

    def test_actual_request_teardown_then_scheduler_free_owns_each_pool_once(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        self.allocator.logical_attn_allocator.alloc(30 * 64)
        request_ids = self.allocator.logical_attn_allocator.alloc(128)
        self.assertEqual(int(request_ids[63]), 2047)
        logical_before = self.allocator.logical_attn_allocator.available_size()
        self.prepare()
        self.transaction.close(failed=True)
        self.assertEqual(
            self.allocator.logical_attn_allocator.available_size(), logical_before
        )
        self.req.kv.kv_allocated_len = 128
        self.c.req_to_token_pool = SimpleNamespace(req_to_token=request_ids.view(1, -1))
        self.c.req_to_device_buffer = torch.arange(64, 128).view(1, -1)
        self.c.mem_pool_device.translate_loc_from_full_to_compressed = lambda ids: ids
        self.c.mem_pool_device.full_to_hisparse_device_index_mapping = (
            self.allocator.full_to_hisparse_device_index_mapping
        )
        self.c._speculative_verifier = self.adapter
        self.c._speculative_teardown = lambda req: (
            HiSparseCoordinator._speculative_teardown(self.c, req)
        )
        self.c.wait_for_pending_backup = MagicMock()
        self.c._lru_init = torch.arange(4)
        self.c._skip_first_backup = torch.tensor([False])
        self.host.allocated_host_indices.return_value = torch.arange(128, 192)
        HiSparseCoordinator.request_finished(self.c, self.req)
        self.allocator.get_kvcache()._translate_loc_to_hisparse_device.side_effect = (
            lambda ids: self.allocator.full_to_hisparse_device_index_mapping[ids]
        )
        self.allocator.free(request_ids)
        self.assertEqual(
            self.allocator.logical_attn_allocator.available_size(), logical_before + 128
        )
        self.assertEqual(self.allocator.hisparse_attn_allocator.available_size(), 256)
        free = self.allocator.hisparse_attn_allocator.get_all_free_pages()
        self.assertEqual(free.numel(), free.unique().numel())
        self.assertNotIn(0, self.adapter.owners)

    def test_reject_all_partial_and_accept_all_publish_only_input_prefix(self):
        for accepted in (1, 2, 4):
            with self.subTest(accepted=accepted):
                self.setUp()
                self.prepare()
                self.transaction.stage_layer(
                    0, torch.tensor([[0, 63], [0, 64], [0, 65], [0, 66]]), 4
                )
                self.transaction.target_done()
                self.transaction.commit(
                    torch.tensor([accepted]), torch.tensor([[0, 1, 2, 3]])
                )
                self.assertEqual(self.adapter.backend.valid_lengths[0][1], 63)
                self.assertTrue(
                    (
                        self.allocator.full_to_hisparse_device_index_mapping[2047:2051]
                        > 0
                    ).all()
                )
                self.transaction.close(failed=False)
                self.assertEqual(
                    self.adapter.backend.valid_lengths[0][1], 63 + accepted
                )
                backup = self.host.backup_from_device_all_layer.call_args
                self.assertEqual(backup.args[1].numel(), accepted)
                self.assertEqual(backup.args[2].numel(), accepted)
                self.assertTrue(
                    (
                        self.allocator.full_to_hisparse_device_index_mapping[2047:2051]
                        == 0
                    ).all()
                )
                self.assertIsNone(self.adapter.owners[0].key)

    def test_failure_before_target_and_after_commit_cancel_without_publication(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                self.setUp()
                self.prepare()
                if committed:
                    self.transaction.stage_layer(
                        0, torch.tensor([[0], [0], [0], [0]]), 4
                    )
                    self.transaction.target_done()
                    self.transaction.commit(
                        torch.tensor([2]), torch.tensor([[0, 1, 2, 3]])
                    )
                self.transaction.close(failed=True)
                self.assertEqual(self.adapter.backend.valid_lengths[0][1], 63)
                self.assertIsNone(self.adapter.owners[0].key)
                self.assertTrue(
                    (
                        self.allocator.full_to_hisparse_device_index_mapping[2047:2051]
                        == 0
                    ).all()
                )

    def test_missing_layer_and_nonlinear_acceptance_cannot_publish(self):
        self.prepare()
        self.transaction.target_done()
        with self.assertRaisesRegex(ValueError, "every verifier attention layer"):
            self.transaction.commit(torch.tensor([1]), torch.tensor([[0, 1, 2, 3]]))
        self.transaction.close(failed=True)
        self.setUp()
        self.prepare()
        self.transaction.stage_layer(0, torch.tensor([[0], [0], [0], [0]]), 4)
        self.transaction.target_done()
        with self.assertRaisesRegex(ValueError, "nonlinear"):
            self.transaction.commit(torch.tensor([2]), torch.tensor([[0, 2, 1, 3]]))
        self.transaction.close(failed=True)
        self.host.backup_from_device_all_layer.assert_not_called()

    def test_stale_owner_refuses_retirement_and_preserves_mapping(self):
        self.prepare()
        original = self.adapter.owners[0].req
        self.adapter.owners[0].req = object()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.transaction.close(failed=True)
        self.assertTrue(
            (self.allocator.full_to_hisparse_device_index_mapping[2047:2051] > 0).all()
        )
        self.adapter.owners[0].req = original
        self.adapter.teardown(self.req)

    def test_worker_boundary_defers_publication_until_real_retirement(self):
        def forward(worker, batch, on_publish=None):
            self.assertIsNone(on_publish)
            txn = batch.hisparse_spec_transaction
            txn.prepare(batch, self.forward, 4, 8)
            txn.stage_layer(0, torch.tensor([[0], [0], [0], [0]]), 4)
            txn.target_done()
            txn.commit(torch.tensor([1]), torch.tensor([[0, 1, 2, 3]]))
            return SimpleNamespace(new_seq_lens=torch.tensor([64]))

        published = []

        def publish(lengths):
            self.assertIsNone(self.adapter.owners[0].key)
            published.append(lengths.tolist())

        with envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True):
            eager_hisparse_worker_boundary(forward)(
                None, self.batch, on_publish=publish
            )
        self.assertEqual(published, [[64]])

    def test_worker_exception_cancels_and_never_publishes(self):
        def forward(worker, batch, on_publish=None):
            batch.hisparse_spec_transaction.prepare(batch, self.forward, 4, 8)
            raise RuntimeError("target failed")

        publish = MagicMock()
        with (
            envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True),
            self.assertRaisesRegex(RuntimeError, "target failed"),
        ):
            eager_hisparse_worker_boundary(forward)(
                None, self.batch, on_publish=publish
            )
        publish.assert_not_called()
        self.assertIsNone(self.adapter.owners[0].key)

    def test_draft_exception_after_commit_cancels_without_publication(self):
        def forward(worker, batch, on_publish=None):
            txn = batch.hisparse_spec_transaction
            txn.prepare(batch, self.forward, 4, 8)
            txn.stage_layer(0, torch.tensor([[0]] * 4), 4)
            txn.target_done()
            txn.commit(torch.tensor([2]), torch.tensor([[0, 1, 2, 3]]))
            raise RuntimeError("draft failed")

        publish = MagicMock()
        with (
            envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True),
            self.assertRaisesRegex(RuntimeError, "draft failed"),
        ):
            eager_hisparse_worker_boundary(forward)(
                None, self.batch, on_publish=publish
            )
        publish.assert_not_called()
        self.assertEqual(self.adapter.backend.valid_lengths[0][1], 63)
        self.assertIsNone(self.adapter.owners[0].key)
        self.host.free.assert_called_once()

    def test_staging_restores_native_draft_state_in_ready_order(self):
        draft = EagleDraftInput(
            topk_p=torch.tensor([[0.2], [0.7]]),
            topk_index=torch.tensor([[11], [22]]),
            hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            bonus_tokens=torch.tensor([31, 32]),
        )
        requests = [SimpleNamespace(), SimpleNamespace()]
        for i, req in enumerate(requests):
            save_staging_draft(req, draft, i)
        siblings = [req.hisparse_staging_draft for req in requests]
        restored = restore_staging_drafts(requests[::-1])
        self.assertEqual(restored.bonus_tokens.tolist(), [32, 31])
        self.assertEqual(restored.topk_index.tolist(), [[22], [11]])
        self.assertEqual(restored.hidden_states.tolist(), [[3.0, 4.0], [1.0, 2.0]])
        self.assertEqual(draft.bonus_tokens.tolist(), [31, 32])
        self.assertFalse(hasattr(requests[0], "hisparse_staging_draft"))
        restored.hidden_states[0, 0] = 999
        self.assertEqual(siblings[1].hidden_states.tolist(), [[3.0, 4.0]])
        self.assertEqual(siblings[0].hidden_states.tolist(), [[1.0, 2.0]])
        self.assertEqual(draft.hidden_states.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_actual_scheduler_staging_transition_restores_native_state(self):
        from sglang.srt.managers import scheduler

        self.req.origin_input_ids = list(range(63))
        self.req.output_ids = [31]
        draft = EagleDraftInput(
            topk_p=torch.tensor([[0.2]]),
            topk_index=torch.tensor([[11]]),
            hidden_states=torch.tensor([[1.0, 2.0]]),
            bonus_tokens=torch.tensor([31]),
        )
        save_staging_draft(self.req, draft, 0)
        owner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=object(),
            model_config=SimpleNamespace(vocab_size=64),
            enable_overlap=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            future_map=MagicMock(),
        )
        with (
            patch.object(
                scheduler.ScheduleBatch,
                "init_new",
                return_value=SimpleNamespace(return_logprob=False),
            ),
            patch.object(
                scheduler.SamplingBatchInfo,
                "from_schedule_batch",
                return_value=object(),
            ),
        ):
            batch = scheduler.Scheduler._build_hisparse_decode_batch(owner, [self.req])
        self.assertEqual(batch.spec_info.bonus_tokens.tolist(), [31])
        self.assertEqual(batch.spec_info.hidden_states.tolist(), [[1.0, 2.0]])
        self.assertEqual(batch.seq_lens.tolist(), [63])
        owner.future_map.stash.assert_not_called()

    def test_actual_spec_allocation_reserves_logical_pages_only(self):
        from sglang.srt.mem_cache import allocation
        from sglang.srt.mem_cache.allocator import paged

        self.allocator.logical_attn_allocator.alloc(31 * 64)
        pool = SimpleNamespace(req_to_token=torch.zeros((1, 256), dtype=torch.int64))
        self.req.kv.kv_allocated_len = 0
        physical_before = self.allocator.hisparse_attn_allocator.available_size()

        def cpu_kernel(prefix, seq, last, free, output, block, page):
            paged.alloc_extend_naive(prefix, seq, last, free, output, page, "cpu")

        kernel = MagicMock()
        kernel.__getitem__.return_value = cpu_kernel

        def assign(indices, table, prefix, seq, locations, count):
            table[0, int(prefix[0]) : int(seq[0])] = locations

        with (
            envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True),
            patch.object(paged, "alloc_extend_kernel", kernel),
            patch.object(allocation, "get_last_loc", return_value=torch.tensor([-1])),
            patch.object(
                allocation, "assign_req_to_token_pool_func", side_effect=assign
            ),
        ):
            allocation.alloc_for_spec_decode(
                SimpleNamespace(token_to_kv_pool_allocator=self.allocator),
                pool,
                reqs=[self.req],
                req_pool_indices=torch.tensor([0]),
                cur_kv_lens=torch.tensor([0]),
                cur_kv_lens_cpu=torch.tensor([0]),
                nxt_kv_lens=torch.tensor([64]),
                nxt_kv_lens_cpu=torch.tensor([64]),
                num_needed_tokens=64,
                batch=SimpleNamespace(device=torch.device("cpu")),
            )
        self.assertEqual(pool.req_to_token[0, :64].tolist(), list(range(2048, 2112)))
        self.assertEqual(self.req.kv.kv_allocated_len, 64)
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(), physical_before
        )
        self.assertTrue(
            (self.allocator.full_to_hisparse_device_index_mapping[2048:2112] == 0).all()
        )

    def test_actual_verify_forward_exception_records_reader_then_cancels(self):
        from sglang.srt.speculative import eagle_worker_common as common

        self.batch.spec_info = MagicMock()
        self.batch.input_ids = torch.arange(4)
        self.batch.forward_mode.is_idle = lambda: False
        self.batch.has_grammar = False
        worker = SimpleNamespace(
            forward_batch_generation=MagicMock(
                side_effect=RuntimeError("forward failed")
            )
        )

        def forward(unused, batch, on_publish=None):
            return common.run_eagle_verify(
                batch,
                target_worker=worker,
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=self.allocator,
                plan_stream=None,
                plan_stream_ctx=contextlib.nullcontext(),
                topk=1,
                num_draft_tokens=4,
                device="cpu",
                metadata_ready_pre_pad=False,
                finalize_tree_path=True,
            )

        with (
            envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True),
            patch.object(common, "record_stream_for_v2_verify"),
            patch.object(common, "record_stream_each"),
            patch.object(
                common, "eagle_prepare_for_verify", return_value=(self.forward, False)
            ),
            patch(
                "sglang.srt.mem_cache.allocation_sizing.get_alloc_reserve_per_decode",
                return_value=8,
            ),
            self.assertRaisesRegex(RuntimeError, "forward failed"),
        ):
            eager_hisparse_worker_boundary(forward)(None, self.batch)
        self.assertTrue(self.forward.hisparse_spec_transaction.target_recorded)
        self.assertIsNone(self.adapter.owners[0].key)
        self.host.backup_from_device_all_layer.assert_not_called()

    def test_actual_native_verify_prepares_and_commits_transaction(self):
        from sglang.srt.speculative import eagle_worker_common as common

        self.allocator.get_kvcache().clear_unaccepted_c128_draft_states = None
        self.batch.spec_info = MagicMock()
        self.batch.input_ids = torch.arange(4)
        self.batch.forward_mode.is_idle = lambda: False
        self.batch.has_grammar = False
        self.batch.return_logprob = False
        self.batch.hisparse_spec_transaction = self.transaction
        logits = SimpleNamespace(next_token_logits=torch.zeros(4, 8))

        def target_forward(**kwargs):
            self.assertIs(
                kwargs["forward_batch"].hisparse_spec_transaction, self.transaction
            )
            self.assertTrue(
                (
                    self.allocator.full_to_hisparse_device_index_mapping[2047:2051] > 0
                ).all()
            )
            self.transaction.stage_layer(0, torch.tensor([[0], [0], [0], [0]]), 4)
            return SimpleNamespace(
                logits_output=logits,
                routed_experts_output=None,
                indexer_topk_output=None,
            )

        worker = SimpleNamespace(forward_batch_generation=target_forward)
        with (
            patch.object(common, "record_stream_for_v2_verify"),
            patch.object(common, "record_stream_each"),
            patch.object(
                common, "eagle_prepare_for_verify", return_value=(self.forward, False)
            ),
            patch.object(
                common,
                "eagle_sample",
                return_value=(
                    torch.arange(4),
                    torch.tensor([2]),
                    torch.tensor([[0, 1, -1, -1]]),
                ),
            ),
            patch.object(common, "maybe_detect_nan"),
            patch.object(common, "maybe_detect_inf"),
            patch.object(common, "commit_mamba_states_after_verify"),
            patch.object(common, "fill_bonus_tokens_func"),
            patch(
                "sglang.srt.mem_cache.allocation_sizing.get_alloc_reserve_per_decode",
                return_value=8,
            ),
        ):
            result = common.run_eagle_verify(
                self.batch,
                target_worker=worker,
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=self.allocator,
                plan_stream=None,
                plan_stream_ctx=contextlib.nullcontext(),
                topk=1,
                num_draft_tokens=4,
                device="cpu",
                metadata_ready_pre_pad=False,
                finalize_tree_path=True,
            )
        self.assertEqual(result.new_seq_lens.tolist(), [65])
        self.assertTrue(self.transaction.target_recorded)
        self.assertTrue(self.transaction.committed)
        self.transaction.close(failed=False)
        self.assertEqual(self.adapter.backend.valid_lengths[0][1], 65)

    def test_actual_dsa_attention_uses_union_table_without_logical_retranslation(self):
        from sglang.srt.layers.attention import dsa_backend as dsa

        self.prepare()
        self.forward.forward_mode = SimpleNamespace(
            is_target_verify=lambda: True, is_draft_extend_v2=lambda: False
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(),
            dsa_decode_impl="tilelang",
            dsa_prefill_impl="tilelang",
            _resolve_kpool_tail_backend=lambda indices, impl: impl,
            _check_kpool_tail_backend=MagicMock(),
            use_mha=False,
            token_to_kv_pool=MagicMock(),
            hisparse_coordinator=self.c,
            use_fused_topk=True,
            get_topk_transform_method=lambda mode: dsa.TopkTransformMethod.PAGED,
            _forward_tilelang=MagicMock(return_value=torch.zeros(4, 1, 2)),
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=1,
            v_head_dim=2,
            head_dim=4,
            scaling=1.0,
        )
        with patch.object(
            dsa,
            "concat_mla_absorb_q_general",
            side_effect=lambda a, b: torch.cat((a, b), dim=-1),
        ):
            dsa.DeepseekSparseAttnBackend.forward_extend(
                backend,
                torch.zeros(4, 1, 2),
                None,
                None,
                layer,
                self.forward,
                q_rope=torch.zeros(4, 1, 2),
                topk_indices=torch.tensor([[0, 63], [0, 64], [0, 65], [0, 66]]),
            )
        backend.token_to_kv_pool.translate_loc_to_hisparse_device.assert_not_called()
        table = backend._forward_tilelang.call_args.kwargs["page_table_1"]
        self.assertEqual(tuple(table.shape), (4, 2))
        self.assertTrue((table > 0).all())
        self.transaction.close(failed=True)


class TestEagerLayout(unittest.TestCase):
    def test_native_glm_eagle3_layout_and_unsupported_modes(self):
        cfg = SimpleNamespace(
            enable_hisparse=True,
            speculative_algorithm="EAGLE3",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_adaptive=False,
            enable_multi_layer_eagle=False,
            disable_overlap_schedule=True,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="disabled"),
                prefill=SimpleNamespace(backend="disabled"),
            ),
            disaggregation_mode="null",
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            enable_dp_attention=False,
            page_size=64,
            kv_cache_dtype="bfloat16",
        )
        hf = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
        with envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(True):
            validate_eager_layout(cfg, hf)
            for field, value in [
                (
                    "cuda_graph_config",
                    SimpleNamespace(
                        decode=SimpleNamespace(backend="full"),
                        prefill=SimpleNamespace(backend="disabled"),
                    ),
                ),
                ("disable_overlap_schedule", False),
                ("disaggregation_mode", "decode"),
                ("speculative_eagle_topk", 2),
            ]:
                old = getattr(cfg, field)
                setattr(cfg, field, value)
                with self.assertRaises(ValueError):
                    validate_eager_layout(cfg, hf)
                setattr(cfg, field, old)
        with (
            envs.SGLANG_HISPARSE_SPECULATIVE_EAGER.override(False),
            self.assertRaises(ValueError),
        ):
            validate_eager_layout(cfg, hf)


if __name__ == "__main__":
    unittest.main()
