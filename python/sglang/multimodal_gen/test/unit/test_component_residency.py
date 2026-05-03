from types import SimpleNamespace
import unittest

import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.component_resident_strategies import (
    ComponentResidencyStrategy,
)


class _CountingStrategy(ComponentResidencyStrategy):
    def __init__(self):
        self.finish_use_count = 0
        self.finish_request_count = 0

    def finish_use(self, module, use, state):
        self.finish_use_count += 1

    def finish_request(self, module, use, state, *, preferred):
        self.finish_request_count += 1
        super().finish_request(module, use, state, preferred=preferred)


class _FakeStage:
    def __init__(self, uses):
        self._uses = uses

    def component_uses(self, _server_args, stage_name=None):
        return self._uses


class _FakePipeline:
    def __init__(self, module, strategy):
        self.modules = {"vae": module}
        self._stage_name_mapping = {}
        self.component_residency_strategies = {"vae": strategy}


class ComponentResidencyTests(unittest.TestCase):
    def _make_manager(self, use):
        strategy = _CountingStrategy()
        manager = ComponentResidencyManager(
            _FakePipeline(nn.Linear(1, 1), strategy),
            SimpleNamespace(),
        )
        manager.begin_request(
            [_FakeStage([use])],
            SimpleNamespace(is_warmup=False),
            SimpleNamespace(),
        )
        return manager, strategy

    def test_finish_after_response_defers_until_task_runs(self):
        use = ComponentUse("DecodingStage", "vae", finish_after_response=True)
        manager, strategy = self._make_manager(use)

        with manager.use_component(use):
            pass

        self.assertEqual(strategy.finish_use_count, 0)

        tasks = manager.finish_request()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(strategy.finish_request_count, 0)
        self.assertEqual(strategy.finish_use_count, 0)

        tasks[0][1]()

        self.assertEqual(strategy.finish_request_count, 1)
        self.assertEqual(strategy.finish_use_count, 1)

    def test_regular_use_finishes_before_response(self):
        use = ComponentUse("DecodingStage", "vae")
        manager, strategy = self._make_manager(use)

        with manager.use_component(use):
            pass

        self.assertEqual(strategy.finish_use_count, 1)
        self.assertEqual(manager.finish_request(), [])


if __name__ == "__main__":
    unittest.main()
