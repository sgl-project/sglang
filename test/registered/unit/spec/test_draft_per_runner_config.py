"""What differs for a draft worker is resolved per runner, not on a config copy.

The v2 spec workers used to write the draft's `context_length` onto the
`ServerArgs` they share with the target, and the scheduler the draft's
`load_format`; then both moved to a published copy of the config. Neither is a
process-wide config change: the draft's context length is the target model's,
its load format is `--speculative-draft-load-format`, and both are consumed by
one constructor each — so they travel as arguments to the runner that owns them.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    build_load_config,
)
from sglang.srt.runtime_context import get_context, get_model
from sglang.srt.speculative.draft_worker_common import draft_server_args_overrides
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopConstruction(Exception):
    """Cuts the draft worker off once its ServerArgs is captured."""


class TestDraftPerRunnerConfig(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    # -- the draft load format is the draft runner's own resolved value --------

    def _load_format_of(self, *, is_draft_worker: bool):
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = is_draft_worker
        return runner._resolve_draft_load_format()

    def test_the_draft_load_format_applies_to_the_draft_runner_only(self):
        self._seed(load_format="auto", speculative_draft_load_format="dummy")
        self.assertEqual(self._load_format_of(is_draft_worker=True), "dummy")
        self.assertIsNone(self._load_format_of(is_draft_worker=False))

    def test_an_unset_draft_load_format_leaves_the_load_config_alone(self):
        self._seed(load_format="auto")
        self.assertIsNone(self._load_format_of(is_draft_worker=True))

    def test_the_draft_format_is_published_while_the_draft_loads(self):
        """Model code reads the load format off the bag as it builds."""
        self._seed(load_format="auto", speculative_draft_load_format="dummy")
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = True

        with runner._load_format_scope(runner._resolve_draft_load_format()):
            self.assertEqual(get_model().load_format, "dummy")
        self.assertEqual(get_model().load_format, "auto")

    def test_the_target_load_never_shifts_the_published_format(self):
        self._seed(load_format="auto", speculative_draft_load_format="dummy")
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = False

        with runner._load_format_scope(runner._resolve_draft_load_format()):
            self.assertEqual(get_model().load_format, "auto")

    def test_the_transfer_engine_gate_answers_for_the_runner(self):
        """The engine is initialized at the top of initialize(), long before the
        weights load, so the gate has to see the draft's format."""
        server_args = self._seed(
            load_format="auto",
            speculative_draft_load_format="remote_instance",
            remote_instance_weight_loader_backend="transfer_engine",
        )
        self.assertFalse(
            server_args.remote_instance_weight_loader_use_transfer_engine()
        )
        self.assertTrue(
            server_args.remote_instance_weight_loader_use_transfer_engine(
                load_format="remote_instance"
            )
        )

    def test_the_load_config_takes_the_per_runner_format_when_given(self):
        server_args = self._seed(load_format="auto")
        common = dict(
            server_args=server_args,
            tp_rank=0,
            remote_instance_weight_transporter_engine=None,
            remote_instance_weight_transporter_session_id=None,
            draft_model_idx=None,
            weight_cache_mode="disable",
            weight_cache_socket=None,
        )
        self.assertEqual(build_load_config(**common).load_format, "auto")
        self.assertEqual(
            build_load_config(load_format="dummy", **common).load_format, "dummy"
        )

    # -- the variant left in the dflash / dspark path carries backends only ----

    def test_the_draft_variant_carries_the_backend_family_only(self):
        self._seed(disable_chunked_prefix_cache=False)
        fields = draft_server_args_overrides("triton")

        self.assertEqual(fields["attention_backend"], "triton")
        self.assertEqual(fields["speculative_draft_attention_backend"], "triton")
        self.assertIsNone(fields["prefill_attention_backend"])
        self.assertIsNone(fields["decode_attention_backend"])
        self.assertNotIn("context_length", fields)
        self.assertNotIn("load_format", fields)
        self.assertNotIn("skip_tokenizer_init", fields)

    def test_the_variant_carries_the_targets_resolved_gate(self):
        """Publishing the variant re-projects the bags, so the gate travels."""
        self._seed(disable_chunked_prefix_cache=False)
        get_context().override("test.gate", disable_chunked_prefix_cache=True)
        self.assertTrue(
            draft_server_args_overrides("triton")["disable_chunked_prefix_cache"]
        )

    # -- the scheduler hands over the process's own config ---------------------

    def _scheduler(self, server_args, create_worker):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = server_args
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=SimpleNamespace(context_len=4096))
        )
        scheduler.ps = SimpleNamespace(gpu_id=0)
        scheduler.nccl_port = 0
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_ngram=lambda: False,
            create_worker=create_worker,
        )
        return scheduler

    def test_the_draft_worker_and_its_factory_get_the_published_config(self):
        server_args = self._seed(speculative_algorithm="EAGLE", load_format="auto")
        seen = {}

        def worker_class(**kwargs):
            seen["worker"] = kwargs["server_args"]
            seen["published_while_building"] = get_model().load_format
            raise _StopConstruction

        def create_worker(factory_server_args):
            seen["factory"] = factory_server_args
            return worker_class

        with self.assertRaises(_StopConstruction):
            self._scheduler(server_args, create_worker).maybe_init_draft_worker()

        # No copy, and no publish switch: a registered algorithm picking its
        # worker class from the config it is handed sees the same object the
        # worker does, and the bags stay the target's throughout.
        self.assertIs(seen["factory"], server_args)
        self.assertIs(seen["worker"], server_args)
        self.assertEqual(seen["published_while_building"], "auto")
        self.assertIs(get_context().server_args, server_args)


if __name__ == "__main__":
    unittest.main()
