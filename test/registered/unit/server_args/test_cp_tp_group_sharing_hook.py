"""Kimi-K3.1 CP-TP group sharing gates: PD prefill with DFLASH only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import cp_tp_group_sharing_hook
from sglang.srt.arg_groups.cp_tp_group_sharing_hook import resolve_cp_tp_group_sharing
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_K31 = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
)


def _args(**kwargs):
    base = dict(
        model_path="dummy",
        tp_size=2,
        attn_cp_size=2,
        enable_prefill_cp=True,
        cp_strategy="interleave",
    )
    return ServerArgs(**(base | kwargs))


class TestKimiK31CpTpGroupSharing(CustomTestCase):
    def _resolve(self, args):
        hook = cp_tp_group_sharing_hook
        with (
            patch.object(
                hook,
                "mambaish_config",
                return_value=SimpleNamespace(linear_attn_config={"num_heads": 64}),
            ),
            patch.object(hook, "with_phase"),
            patch.object(hook, "declare_resolution") as declare,
        ):
            resolve_cp_tp_group_sharing(args, _K31)
        return declare

    def test_pd_prefill_with_dflash(self):
        declare = self._resolve(
            _args(disaggregation_mode="prefill", speculative_algorithm="DFLASH")
        )
        self.assertTrue(declare.call_args.kwargs["enable_cp_tp_group_sharing"])

    def test_rejects_decode_node(self):
        with self.assertRaisesRegex(ValueError, "disaggregation-mode decode"):
            self._resolve(_args(disaggregation_mode="decode"))

    def test_rejects_dflash_without_pd_prefill(self):
        with self.assertRaisesRegex(ValueError, "DFLASH on a PD prefill node"):
            self._resolve(_args(speculative_algorithm="DFLASH"))

    def test_rejects_other_speculative_algorithms(self):
        with self.assertRaisesRegex(ValueError, "DFLASH on a PD prefill node"):
            self._resolve(
                _args(disaggregation_mode="prefill", speculative_algorithm="EAGLE")
            )


if __name__ == "__main__":
    unittest.main()
