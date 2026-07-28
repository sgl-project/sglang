import unittest
from types import SimpleNamespace

import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
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


def _server_args():
    return SimpleNamespace(enable_layerwise_nvtx_marker=False)


class ComponentResidencyPostResponseTests(unittest.TestCase):
    def _make_manager(self, use, batch=None):
        strategy = _CountingStrategy()
        manager = ComponentResidencyManager(
            _FakePipeline(nn.Linear(1, 1), strategy),
            _server_args(),
        )
        manager.begin_request(
            [_FakeStage([use])],
            batch or SimpleNamespace(is_warmup=False),
            _server_args(),
        )
        return manager, strategy

    def test_terminal_finish_after_response_defers_even_when_preferred(self):
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

    def test_warmup_keeps_ready_without_post_response_task(self):
        use = ComponentUse(
            "DecodingStage",
            "vae",
            keep_ready_after_warmup=True,
            finish_after_response=True,
        )
        manager, strategy = self._make_manager(use, SimpleNamespace(is_warmup=True))

        with manager.use_component(use):
            pass

        self.assertEqual(manager.finish_request(), [])
        self.assertEqual(strategy.finish_request_count, 0)
        self.assertEqual(strategy.finish_use_count, 0)

    def test_grouped_batch_warmup_state(self):
        use = ComponentUse("DecodingStage", "vae", finish_after_response=True)
        manager, _strategy = self._make_manager(
            use,
            [
                SimpleNamespace(is_warmup=True),
                SimpleNamespace(is_warmup=True),
            ],
        )

        self.assertTrue(manager.state.batch_is_warmup)

        manager, _strategy = self._make_manager(
            use,
            [
                SimpleNamespace(is_warmup=True),
                SimpleNamespace(is_warmup=False),
            ],
        )

        self.assertFalse(manager.state.batch_is_warmup)


if __name__ == "__main__":
    unittest.main()
