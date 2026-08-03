import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

FORBIDDEN_TOKENS = ("self.running_batch", "self.last_batch", "self.cur_batch")

DECISION_METHODS = (
    Scheduler.get_next_batch_to_run,
    Scheduler.get_new_batch_prefill,
    Scheduler._get_new_batch_prefill_raw,
    Scheduler._abort_on_running_timeout,
    Scheduler.is_disable_overlap_for_batch,
    SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run,
    SchedulerDisaggregationPrefillMixin.process_prefill_chunk,
    SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch,
    SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run,
)


class TestDecisionMethodsHaveNoHiddenBatchChannel(unittest.TestCase):
    def test_decision_methods_take_batches_as_params_not_self(self):
        """The batch decision tree must receive running/last batch as params, never via self.*."""
        for method in DECISION_METHODS:
            source = inspect.getsource(inspect.unwrap(method))
            self.assertIn(
                f"def {method.__name__}",
                source,
                msg=f"failed to read the real source of {method.__qualname__}",
            )
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    source,
                    msg=(
                        f"{method.__qualname__} references {token}; pass the batch "
                        "explicitly and return it via NextBatchPlan instead."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
