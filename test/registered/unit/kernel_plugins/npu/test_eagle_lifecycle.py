import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.kernel_plugins.npu import eagle
from sglang.srt.speculative.eagle_megakernel import init_eagle_draft_runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestEagleLifecycle(unittest.TestCase):
    def setUp(self):
        self.binding = NS(
            role="nextn",
            mode="fused_extend",
            hit=Mock(),
            supports_extend=Mock(return_value=True),
        )
        worker = NS(
            speculative_num_steps=7,
            hot_token_id=None,
            draft_extend_attn_backend=object(),
            draft_runner=NS(
                ascend_binding=self.binding, model_config=NS(hidden_size=4)
            ),
        )
        self.runtime = init_eagle_draft_runtime(draft_worker=worker)
        self.runner = Mock()
        self.runtime.bind_executors(decode=self.binding, extend=self.runner)
        self.mode = NS(is_idle=lambda: False)
        self.req = NS(kv=NS(req_pool_idx=3))
        self.batch = NS(reqs=[self.req], forward_mode=self.mode)

        def presence(present, valid=True):
            if not valid:
                raise RuntimeError("stale lease")
            return present

        def admit(local, *, outstanding_chain=False):
            if outstanding_chain:
                raise RuntimeError("Outstanding chain")
            return bool(local)

        for key, effect in (
            ("candidate_present", presence),
            ("all_admit", admit),
            ("forward_context", lambda *a, **kw: nullcontext()),
        ):
            mock = patch.object(eagle, key, side_effect=effect)
            mock.start()
            self.addCleanup(mock.stop)

    def test_borrowed_outputs_have_one_consumer(self):
        for accepted in (1, 3, 8):
            tokens = torch.arange(7).reshape(1, 7)
            hidden = torch.ones(1, 4)
            self.runner.execute_extend.return_value = NS(
                token_ids=tokens, first_hidden_states=hidden
            )
            seed = NS()
            forward = NS(positions=torch.arange(124, 132))
            next_forward = NS(
                positions=torch.tensor([124 + accepted]), forward_mode=self.mode
            )
            self.assertIsNone(self.runtime.propose_chain(self.batch, next_forward))
            run = self.runtime.plan_fused_extend(self.batch, object())
            run(self.batch, forward, torch.tensor([accepted - 1]), seed)
            with self.assertRaises(RuntimeError):
                self.runtime.plan_fused_extend(self.batch, object())
            tokens.zero_()
            hidden.zero_()
            self.assertEqual(seed.hidden_states.tolist(), [[1] * 4])
            self.req.kv.req_pool_idx += 1
            with self.assertRaises(RuntimeError):
                self.runtime.propose_chain(self.batch, next_forward)
            self.req.kv.req_pool_idx -= 1
            chain = self.runtime.propose_chain(self.batch, next_forward)
            self.assertEqual(chain.tolist(), [list(range(7))])
            self.assertIsNone(self.runtime.propose_chain(self.batch, next_forward))
        self.assertEqual(self.runner.execute_extend.call_count, 3)

    def test_decline_before_execution_and_propagate_failure(self):
        self.binding.supports_extend.return_value = False
        self.assertIsNone(self.runtime.plan_fused_extend(self.batch, object()))
        self.runner.execute_extend.assert_not_called()
        self.binding.supports_extend.return_value = True
        self.runner.covers.return_value = False
        self.assertIsNone(self.runtime.plan_fused_extend(self.batch, object()))
        self.runner.execute_extend.assert_not_called()
        self.runner.covers.return_value = True
        self.runner.execute_extend.side_effect = RuntimeError("device replay failed")
        run = self.runtime.plan_fused_extend(self.batch, object())
        with self.assertRaisesRegex(RuntimeError, "device replay failed"):
            run(self.batch, object(), object(), NS())
        self.runner.execute_extend.assert_called_once()
        self.assertFalse(hasattr(self.req, "ascend_candidate"))
