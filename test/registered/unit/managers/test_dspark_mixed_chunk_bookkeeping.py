import types
import unittest

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsparkMixedChunkBookkeeping(unittest.TestCase):
    def test_running_row_commits_one_target_token(self):
        req = types.SimpleNamespace(kv_committed_len=17)
        batch = types.SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)

        SchedulerBatchResultProcessor._commit_dspark_mixed_target_token(
            batch=batch, req=req, decoding_req_ids={id(req)}
        )

        self.assertEqual(req.kv_committed_len, 18)

    def test_new_prefill_row_is_not_double_committed(self):
        req = types.SimpleNamespace(kv_committed_len=17)
        batch = types.SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)

        SchedulerBatchResultProcessor._commit_dspark_mixed_target_token(
            batch=batch, req=req, decoding_req_ids=set()
        )

        self.assertEqual(req.kv_committed_len, 17)

    def test_non_dspark_mixed_row_keeps_existing_owner(self):
        req = types.SimpleNamespace(kv_committed_len=17)
        batch = types.SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)

        SchedulerBatchResultProcessor._commit_dspark_mixed_target_token(
            batch=batch, req=req, decoding_req_ids={id(req)}
        )

        self.assertEqual(req.kv_committed_len, 17)


if __name__ == "__main__":
    unittest.main()
