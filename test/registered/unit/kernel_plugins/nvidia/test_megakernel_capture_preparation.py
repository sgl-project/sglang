# SPDX-License-Identifier: Apache-2.0
"""CPU selection/admission tests; native runtime and collectives are separate."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

ARCH = ["KimiK3ForConditionalGeneration"]
DEFAULT = tuple(range(6, 65, 6))
FLAG = mb.envs.SGLANG_KIMI_K3_MEGAKERNEL_CAPTURE_ONLY


class TestCapturePreparation(unittest.TestCase):
    def setUp(self):
        self.graph = NS(decode=NS(backend=mb.Backend.FULL, bs=[1, 2]))
        self.config = NS(speculative_algorithm="DSPARK", speculative_adaptive=False)
        self.parallel = NS(dcp_size=1)
        for name, value in (
            ("get_exec", NS(graph=NS(cuda_graph_config=self.graph))),
            ("get_spec", self.config),
            ("get_parallel", self.parallel),
        ):
            context = patch.object(mb, name, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(
            mb, "_gather_tp", side_effect=lambda value, **kw: [value] * 8
        )
        self.gather = context.start()
        self.addCleanup(context.stop)
        context = FLAG.override(True)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def select(self, q=6, arch=ARCH, draft=False):
        return mb._kimi_build_widths(64, q, arch, is_draft=draft)

    def test_disabled_preserves_dense_context_and_agrees(self):
        with FLAG.override(False):
            self.assertEqual(self.select(), DEFAULT)
            self.assertEqual(self.select(q=1), tuple(range(1, 65)))
        self.assertEqual(self.gather.call_count, 2)

    def test_request_to_token_conversion_baseline_and_capacity(self):
        for bs, expected in (
            ([1, 2], (6, 12)),
            ([4, 2, 2], (6, 12, 24)),
            ([1, 8, 16], (6, 48)),
            ([16], (6,)),
        ):
            with self.subTest(bs=bs):
                self.graph.decode.bs = bs
                self.assertEqual(self.select(), expected)
                self.assertEqual(self.graph.decode.bs, bs)

    def test_other_models_and_draft_are_unchanged(self):
        self.assertEqual(self.select(arch=["Qwen3_5ForConditionalGeneration"]), DEFAULT)
        self.assertEqual(self.select(draft=True), DEFAULT)
        self.gather.assert_not_called()

    def test_invalid_or_out_of_scope_settings_fail(self):
        for q in (1, 4, 8):
            with (
                self.subTest(q=q),
                self.assertRaisesRegex(ValueError, "fixed DSpark q6"),
            ):
                self.select(q=q)
        for owner, key, values in (
            (self.config, "speculative_algorithm", [None, "EAGLE"]),
            (self.config, "speculative_adaptive", [True]),
            (self.parallel, "dcp_size", [8]),
            (self.graph.decode, "backend", [mb.Backend.DISABLED]),
            (self.graph.decode, "bs", [[], None, [0], [-1], [True], [1.5]]),
        ):
            original = getattr(owner, key)
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    setattr(owner, key, value)
                    self.select()
            setattr(owner, key, original)

    def test_rank_disagreement_includes_flag_widths_and_errors(self):
        for peer in (
            (False, DEFAULT, None),
            (True, (6, 24), None),
            (True, DEFAULT, "bad graph"),
        ):
            with (
                self.subTest(peer=peer),
                self.assertRaisesRegex(RuntimeError, "ranks.*differ"),
            ):
                self.gather.side_effect = lambda value, **kw: [value] * 7 + [peer]
                self.select()


if __name__ == "__main__":
    unittest.main()
