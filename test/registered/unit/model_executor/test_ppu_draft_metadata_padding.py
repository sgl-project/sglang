import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import eager_runner as er
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPPUDraftMetadataPadding(unittest.TestCase):
    def run_decode(self, *, ppu=True, draft=True, replan=True):
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.zeros(1, dtype=torch.long),
            seq_lens=torch.ones(1, dtype=torch.long),
            out_cache_loc=torch.zeros(1, dtype=torch.long),
            seq_lens_sum=1,
        )
        fb.mark_forward_metadata_ready(replan_equivalent=replan)
        fb.batch_size = 8
        fb.input_ids = torch.zeros(8, dtype=torch.long)
        fb.positions = torch.zeros(8, dtype=torch.long)
        fb.req_pool_indices = torch.zeros(8, dtype=torch.long)
        fb.seq_lens = torch.ones(8, dtype=torch.long)
        active = SimpleNamespace(init_forward_metadata=Mock())
        default = SimpleNamespace(init_forward_metadata=Mock())
        output = object()
        model = SimpleNamespace(forward=Mock(return_value=output))
        mr = SimpleNamespace(
            is_draft_worker=draft,
            model=model,
            _pp_kwargs=lambda _: {},
            device_timer=None,
        )
        runner = SimpleNamespace(
            model_runner=mr,
            enable_pdmux=False,
            load_batch=lambda batch, _: batch,
            _resolve_decode_pdmux=lambda: (default, contextlib.nullcontext()),
        )
        with (
            patch.object(er, "_is_ppu", ppu, create=True),
            patch.object(er, "device_timer_ctx", lambda *_: contextlib.nullcontext()),
            forward_context(ForwardContext(attn_backend=active)),
        ):
            result = er.EagerRunner._execute_decode(runner, fb)
        self.assertIs(result, output)
        return fb, active.init_forward_metadata, default.init_forward_metadata

    def test_ppu_draft_replans_active_step_after_padding(self):
        fb, active, default = self.run_decode()
        active.assert_called_once_with(fb)
        self.assertEqual(active.call_args.args[0].seq_lens.shape[0], 8)
        default.assert_not_called()

    def test_non_ppu_keeps_default_backend(self):
        fb, active, default = self.run_decode(ppu=False)
        default.assert_called_once_with(fb)
        active.assert_not_called()

    def test_target_keeps_default_backend(self):
        fb, active, default = self.run_decode(draft=False)
        default.assert_called_once_with(fb)
        active.assert_not_called()

    def test_non_equivalent_wrapper_plan_is_preserved(self):
        _, active, default = self.run_decode(replan=False)
        active.assert_not_called()
        default.assert_not_called()


if __name__ == "__main__":
    unittest.main()
