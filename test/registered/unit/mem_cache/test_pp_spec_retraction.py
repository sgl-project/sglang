import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import decode_schedule_batch_mixin
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components import request_retirer
from sglang.srt.mem_cache import common
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CopyPool:
    cpu_copy_carries_mamba = True

    def __init__(self, offset):
        # A pool copy may include layout-specific buffers in addition to KV.
        self.buffers = {
            "kv": torch.arange(24).reshape(12, 2) + offset,
            "scale": torch.arange(12) + offset,
        }

    def get_cpu_copy(self, indices, **kwargs):
        return {name: value[indices].clone() for name, value in self.buffers.items()}

    def load_cpu_copy(self, values, indices, **kwargs):
        for name, value in values.items():
            self.buffers[name][indices] = value


def _request():
    req = object.__new__(Req)
    req.rid = "r"
    req.kv = ReqKvInfo(req_pool_idx=0)
    req.origin_input_ids = [10, 20]
    req.output_ids = [30, 40]
    pool = SimpleNamespace(req_to_token=torch.tensor([[1, 3, 5, 7]]))
    return req, pool


def _relay():
    return PPSpecRelayInput(
        rids=["r", "s"],
        tokens=torch.tensor([[40, 41, 42], [70, 71, 72]]),
        parents=torch.tensor([[-1, 0], [-1, 0]]),
        top_scores=torch.tensor([[0, 1], [0, 1]]),
    )


class TestPPSpecRetraction(CustomTestCase):
    def test_target_draft_and_proposal_survive_slot_reuse(self):
        req, pool = _request()
        target, draft = _CopyPool(0), _CopyPool(100)
        relay = _relay()
        indices = pool.req_to_token[0, :3]
        target_values = target.get_cpu_copy(indices)
        draft_values = draft.get_cpu_copy(indices)

        cache = SimpleNamespace(
            req_to_token_pool=pool, token_to_kv_pool_allocator=target
        )
        retirer = release_only_retirer(cache)
        retirer.draft_token_to_kv_pool = draft
        with (
            patch.object(
                request_retirer,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup="cpu_tensor",
                ),
            ),
            patch.object(common, "dsv4_needs_rebootstrap", return_value=False),
            patch.object(request_retirer, "release_kv_cache"),
        ):
            self.assertTrue(
                retirer.retract(
                    req, offload_kv=True, pp_spec_relay=relay.reindex([req.rid])
                )
            )
        relay.tokens.zero_()
        relay.parents.zero_()
        relay.top_scores.zero_()
        for values in (target.buffers, draft.buffers):
            for value in values.values():
                value.fill_(-1)
        pool.req_to_token[0, :3] = torch.tensor([6, 8, 10])

        common.restore_kv_cache(
            req, cache, pool, target, "cpu_tensor", draft_token_to_kv_pool=draft
        )

        indices = pool.req_to_token[0, :3]
        for name in target_values:
            torch.testing.assert_close(
                target.buffers[name][indices], target_values[name]
            )
            torch.testing.assert_close(draft.buffers[name][indices], draft_values[name])
        backup = req.kv.retraction_backup
        self.assertEqual(backup.pp_spec_seq_len, 3)
        self.assertIsNone(backup.cpu_tensors)
        self.assertIsNone(backup.draft_cpu_tensors)
        torch.testing.assert_close(
            backup.pp_spec_relay.tokens, torch.tensor([[40, 41, 42]])
        )
        torch.testing.assert_close(
            backup.pp_spec_relay.parents, torch.tensor([[-1, 0]])
        )
        torch.testing.assert_close(
            backup.pp_spec_relay.top_scores, torch.tensor([[0, 1]])
        )

    def test_non_tail_stage_retains_proposal_without_draft_pool(self):
        req, pool = _request()
        target = _CopyPool(0)
        req.offload_kv_cache(pool, target, pp_spec_relay=_relay().reindex([req.rid]))
        self.assertIsNone(req.kv.retraction_backup.draft_cpu_tensors)
        req.load_kv_cache(pool, target)
        self.assertEqual(req.kv.retraction_backup.pp_spec_relay.rids, [req.rid])

    def test_ordinary_retraction_still_clears_backup(self):
        req, pool = _request()
        target = _CopyPool(0)
        with patch.object(common, "dsv4_needs_rebootstrap", return_value=False):
            self.assertTrue(
                common.backup_kv_cache(req, None, pool, target, "cpu_tensor")
            )
        self.assertIsNone(req.kv.retraction_backup.pp_spec_relay)
        common.restore_kv_cache(req, None, pool, target, "cpu_tensor")
        self.assertIsNone(req.kv.retraction_backup)

    def test_restore_rejects_missing_draft_pool_or_stale_proposal(self):
        req, pool = _request()
        target, draft = _CopyPool(0), _CopyPool(100)
        req.offload_kv_cache(
            pool,
            target,
            draft_token_to_kv_pool=draft,
            pp_spec_relay=_relay().reindex([req.rid]),
        )
        with patch.object(target, "load_cpu_copy") as load:
            with self.assertRaisesRegex(AssertionError, "Missing draft pool"):
                req.load_kv_cache(pool, target)
            req.output_ids[-1] = 99
            with self.assertRaises(AssertionError):
                req.load_kv_cache(pool, target, draft_token_to_kv_pool=draft)
            load.assert_not_called()

    def test_batch_retraction_owns_only_the_selected_proposal(self):
        req, _ = _request()
        relay = _relay()
        batch = SimpleNamespace(reqs=[req], spec_info=relay, tree_cache=object())
        with (
            patch.object(
                schedule_batch,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ),
            patch.object(schedule_batch, "release_req", return_value=True) as release,
        ):
            self.assertTrue(ScheduleBatch.release_req(batch, 0, 1, object()))
        saved = release.call_args.kwargs["pp_spec_relay"]
        relay.tokens.zero_()
        self.assertEqual(saved.rids, [req.rid])
        torch.testing.assert_close(saved.tokens, torch.tensor([[40, 41, 42]]))

    def test_host_pool_cannot_silently_drop_proposal(self):
        with patch.object(common, "dsv4_needs_rebootstrap", return_value=False):
            with self.assertRaisesRegex(ValueError, "cpu_tensor"):
                common.backup_kv_cache(
                    Mock(), Mock(), Mock(), Mock(), "host_pool", pp_spec_relay=_relay()
                )

    def test_prebuilt_consumes_restored_proposals_in_request_order(self):
        relay = _relay()
        reqs = [
            SimpleNamespace(
                rid=rid,
                output_ids=[token],
                seqlen=4,
                swa_branching_seqlen=None,
                grammar=None,
                kv=SimpleNamespace(
                    retraction_backup=common.RetractionBackup(
                        pp_spec_relay=relay.reindex([rid]), pp_spec_seq_len=3
                    )
                ),
            )
            for rid, token in (("s", 70), ("r", 40))
        ]
        batch = SimpleNamespace(
            reqs=reqs, tree_cache=object(), device="cpu", spec_algorithm=Mock()
        )
        with (
            patch.object(decode_schedule_batch_mixin, "checkpoint_kv_cache"),
            patch.object(
                decode_schedule_batch_mixin, "async_h2d", side_effect=torch.tensor
            ),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch, None, pp_spec_decode=True
            )
        self.assertEqual(batch.spec_info.rids, ["s", "r"])
        torch.testing.assert_close(batch.spec_info.tokens, relay.tokens[[1, 0]])
        self.assertTrue(all(req.kv.retraction_backup is None for req in reqs))
        batch.spec_algorithm.build_disagg_draft_input.assert_not_called()

    def test_ordinary_prebuilt_does_not_inspect_pp_backup(self):
        # Ordinary PD requests do not need the new backup fields.
        req = SimpleNamespace(output_ids=[40], swa_branching_seqlen=None, grammar=None)
        batch = SimpleNamespace(
            reqs=[req], tree_cache=object(), device="cpu", spec_algorithm=Mock()
        )
        with (
            patch.object(decode_schedule_batch_mixin, "checkpoint_kv_cache"),
            patch.object(
                decode_schedule_batch_mixin, "async_h2d", side_effect=torch.tensor
            ),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, None)
        self.assertIs(
            batch.spec_info, batch.spec_algorithm.build_disagg_draft_input.return_value
        )


if __name__ == "__main__":
    unittest.main()
