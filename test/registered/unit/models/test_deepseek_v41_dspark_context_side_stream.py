"""DSpark draft context KV on a side stream (--enable-dsv41-low-latency-decode).

The switch replaces the static verify epilogue; its side stream projects the draft
context KV beside the accept only inside a CUDA graph capture, and the KV store then
uses that projection instead of recomputing it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.dsv41_low_latency.dspark.static_epilogue import (
    ContextKvSideStream,
)
from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue
from sglang.test.test_utils import CustomTestCase

STRIDE = 6


def _forbidden(**kwargs):
    raise AssertionError("projected the context KV")


class TestContextKvSideStream(CustomTestCase):
    def test_no_fork_outside_capture(self):
        # Eager steps keep the serial order even with the stream configured.
        side = object.__new__(ContextKvSideStream)
        model = SimpleNamespace(project_context_kv=_forbidden)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertIsNone(
                side.fork(draft_model=model, pool=None, hidden=torch.zeros(6, 8))
            )

    def test_capture_hook_takes_the_override_only_when_set(self):
        """Without the switch the static verify epilogue is main's; with it, the
        switch's epilogue (side stream, Engram commit) replaces it."""
        epilogue = DsparkVerifyEpilogue(
            max_bs=2,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
        )
        self.assertIsNone(epilogue.static_epilogue_override)
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(is_draft_worker=False),
            ragged_verify_mode=False,
        )
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(STRIDE, 4), hidden_states=torch.zeros(6, 8)
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        calls = []
        with patch.object(
            DsparkVerifyEpilogue,
            "_static_epilogue",
            lambda self, out, fb: calls.append("stock"),
        ):
            epilogue.capture_hook(runner, out, batch, STRIDE)
            epilogue.static_epilogue_override = lambda out, fb: calls.append("switch")
            epilogue.capture_hook(runner, out, batch, STRIDE)
        self.assertEqual(calls, ["stock", "switch"])

    def test_store_uses_the_precomputed_context_kv(self):
        stored = []
        pool = SimpleNamespace(
            set_swa_key_buffer_radix_fused_norm_rope=lambda **kw: stored.append(
                kw["kv"]
            ),
            uniform_fp8=False,
        )
        stages = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    layer_id=i,
                    kv_norm=SimpleNamespace(weight=SimpleNamespace(data=None)),
                    eps=1e-6,
                    freqs_cis=None,
                )
            )
            for i in range(3)
        ]
        model = SimpleNamespace(stages=stages, project_context_kv=_forbidden)
        kvs = [torch.full((6, 4), float(i)) for i in range(3)]
        DeepseekV4ForCausalLMDSpark.write_target_hidden_kv(
            model,
            main_hidden=torch.zeros(6, 8),
            swa_loc=torch.zeros(6, dtype=torch.int64),
            positions=torch.zeros(6, dtype=torch.int64),
            pool=pool,
            context_kv=kvs,
        )
        self.assertEqual(stored, kvs)


if __name__ == "__main__":
    unittest.main()
