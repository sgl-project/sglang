import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)


class _RecordingExecutor:
    """Records the residency manager visible when an execute entry point runs.

    ``PipelineExecutor.__init__`` seeds ``component_residency_manager = None``,
    and every ``_execute_stages`` run dereferences it unguarded via
    ``_component_residency_request``. Capturing the value at execute time is
    therefore what distinguishes an installed manager from the ``None`` that
    raises ``AttributeError`` inside the real executor.
    """

    def __init__(self):
        self.component_residency_manager = None
        self.seen_at_execute = "not-called"

    def execute_with_profiling(self, stages, batch, server_args):
        self.seen_at_execute = self.component_residency_manager

    def execute_group_with_profiling(self, stages, batches, server_args):
        self.seen_at_execute = self.component_residency_manager

    def execute_group_sequentially_with_profiling(self, stages, batches, server_args):
        self.seen_at_execute = self.component_residency_manager
        yield None


def _make_req():
    return SimpleNamespace(
        is_warmup=False,
        suppress_logs=True,
        num_outputs_per_prompt=2,
    )


def _make_server_args():
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            task_type=SimpleNamespace(is_action_gen=lambda: False),
            supports_sequential_multi_output_inference=lambda: True,
        )
    )


class TestPipelineResidencyManagerInstall(unittest.TestCase):
    """Guards the grouped forward paths against a ``None`` residency manager.

    ``forward_batch`` used to reach ``execute_group_with_profiling`` without
    installing the manager that ``forward`` installs, so a request expanded to
    ``num_outputs_per_prompt > 1`` died with
    ``AttributeError: 'NoneType' object has no attribute 'begin_request'`` on the
    first grouped forward. Deleting these cases restores that silent asymmetry
    between the three entry points.
    """

    def setUp(self):
        self.pipeline = object.__new__(ComposedPipelineBase)
        self.pipeline.executor = _RecordingExecutor()
        self.pipeline._stage_name_mapping = {}
        self.pipeline.stages = []
        self.pipeline.is_lora_set = lambda: False
        self.pipeline.is_lora_effective = lambda: False

        self.manager = object()
        patcher = patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base"
            ".get_global_component_residency_manager",
            return_value=self.manager,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_forward_installs_manager_before_execute(self):
        self.pipeline.forward(_make_req(), _make_server_args())

        self.assertIs(self.pipeline.executor.seen_at_execute, self.manager)

    def test_forward_batch_installs_manager_before_grouped_execute(self):
        self.pipeline.forward_batch([_make_req(), _make_req()], _make_server_args())

        self.assertIs(self.pipeline.executor.seen_at_execute, self.manager)

    def test_forward_batch_sequentially_installs_manager_before_execute(self):
        list(
            self.pipeline.forward_batch_sequentially(
                [_make_req(), _make_req()], _make_server_args()
            )
        )

        self.assertIs(self.pipeline.executor.seen_at_execute, self.manager)

    def test_executor_dereferences_manager_without_a_none_guard(self):
        """Pins why installing matters: the executor has no ``None`` fallback.

        If a future diff makes ``begin_component_residency_request`` tolerate a
        missing manager, the asymmetry above stops being fatal and these cases
        can be revisited. Until then a skipped install is a hard crash.
        """
        executor = object.__new__(PipelineExecutor)
        executor.component_residency_manager = None

        with self.assertRaises(AttributeError):
            executor.begin_component_residency_request([], _make_req(), None)


if __name__ == "__main__":
    unittest.main()
