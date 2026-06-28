from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import torch

import sglang.srt.layers.moe.hash_topk as hash_topk_module
import sglang.srt.server_args as server_args_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.state_capturer import routed_experts as routed_experts_module
from sglang.srt.state_capturer.routed_experts import (
    disable_routed_experts_capture_for_draft,
    set_global_experts_capturer,
)
from sglang.test.test_utils import CustomTestCase


class FakeCapturer:
    def __init__(self):
        self.calls = []

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        self.calls.append((layer_id, topk_indices.detach().clone()))


class TestHashTopKRoutedExpertsCapture(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._old_server_args = server_args_module._global_server_args
        self._old_capturer = routed_experts_module.get_global_experts_capturer()
        server_args_module.set_global_server_args_for_scheduler(
            SimpleNamespace(enable_deepep_waterfill=False)
        )
        self.capturer = FakeCapturer()
        set_global_experts_capturer(self.capturer)

    def tearDown(self):
        set_global_experts_capturer(self._old_capturer)
        server_args_module.set_global_server_args_for_scheduler(self._old_server_args)
        super().tearDown()

    def _make_hash_topk(self, layer_id=3):
        return HashTopK(
            topk=2,
            num_experts=5,
            num_fused_shared_experts=0,
            vocab_size=8,
            scoring_func="softmax",
            layer_id=layer_id,
        )

    def _run_hash_topk(self, hash_topk, hidden_states, router_logits, input_ids):
        with hash_topk_module.envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False):
            return hash_topk(hidden_states, router_logits, input_ids)

    def test_hash_topk_captures_selected_experts(self):
        hash_topk = self._make_hash_topk()
        hidden_states = torch.zeros((3, 4), dtype=torch.float32)
        router_logits = torch.ones((3, 5), dtype=torch.float32)
        input_ids = torch.tensor([1, 2, 4], dtype=torch.long)

        output = self._run_hash_topk(hash_topk, hidden_states, router_logits, input_ids)

        self.assertEqual(len(self.capturer.calls), 1)
        layer_id, captured = self.capturer.calls[0]
        self.assertEqual(layer_id, 3)
        self.assertTrue(torch.equal(captured, output.topk_ids))
        expected = torch.tensor([[1, 2], [2, 3], [4, 0]], dtype=torch.int32)
        self.assertTrue(torch.equal(captured, expected))

    def test_draft_opt_out_disables_hash_topk_capture(self):
        model = torch.nn.Module()
        model.hash_topk = self._make_hash_topk()

        disable_routed_experts_capture_for_draft(model)
        self._run_hash_topk(
            model.hash_topk,
            torch.zeros((1, 4), dtype=torch.float32),
            torch.ones((1, 5), dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )

        self.assertEqual(self.capturer.calls, [])

    def test_missing_layer_id_raises_when_capture_is_active(self):
        hash_topk = self._make_hash_topk(layer_id=None)

        with self.assertRaisesRegex(RuntimeError, "requires layer_id"):
            self._run_hash_topk(
                hash_topk,
                torch.zeros((1, 4), dtype=torch.float32),
                torch.ones((1, 5), dtype=torch.float32),
                torch.tensor([1], dtype=torch.long),
            )

        self.assertEqual(self.capturer.calls, [])
