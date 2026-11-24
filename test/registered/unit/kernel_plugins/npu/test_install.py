import logging
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.kernels.plugin_kernels.npu.spec import Decline, MegakernelSpec
from sglang.srt.kernel_plugins.npu.binding import install_megakernels
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

_LOG = "sglang.srt.kernel_plugins.npu.binding"


class TestInstallMegakernels(unittest.TestCase):
    def _runner(self, name="GlmMoeDsaForCausalLM", draft=False):
        return NS(
            server_args=NS(speculative_num_steps=3, speculative_num_draft_tokens=4),
            model=type(name, (), {})(),
            is_draft_worker=draft,
            runner_ids=NS(is_translating=False),
            token_to_kv_pool=object(),
            token_to_kv_pool_allocator=NS(hands_out_virtual_ids=False),
        )

    def _install(self, runner, prepare, *, role="target", peers=None, world=1):
        spec = MegakernelSpec(role, prepare)

        def gather(ranks, signature, group=None):
            ranks[:] = [signature] if peers is None else peers(signature)

        with (
            patch(
                "sglang.srt.kernel_plugins.npu.binding.find_spec",
                return_value=spec,
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.resolved_view",
                side_effect=lambda args: args,
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.get_parallel",
                return_value=NS(tp_group=NS(cpu_group="cpu")),
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.dist.get_world_size",
                return_value=world,
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.dist.all_gather_object",
                side_effect=gather,
            ),
        ):
            return install_megakernels(runner)

    def test_agreed_binding_is_built(self):
        built = Mock()
        binding = NS(role="target", mode="decode_loop", build=built)
        runner = self._runner()
        result = self._install(runner, lambda resources, cfg: binding)
        self.assertIs(result, binding)
        built.assert_called_once_with(runner.model)

    def test_expected_decline_logs_info_without_reading_the_text(self):
        declined = Decline("nextn: fused draft is off", expected=True)
        with self.assertLogs(_LOG, level="INFO") as logs:
            result = self._install(
                self._runner(name="GlmMoeDsaForCausalLMNextN", draft=True),
                lambda resources, cfg: declined,
                role="nextn",
            )
        self.assertIsNone(result)
        self.assertEqual(logs.records[0].levelno, logging.INFO)
        self.assertIn("fused draft is off", logs.output[0])
        self.assertNotIn("draft mode is native", logs.output[0])

    def test_phrase_does_not_make_a_refusal_expected(self):
        declined = Decline("target: draft mode is native", expected=False)
        with self.assertLogs(_LOG, level="WARNING") as logs:
            result = self._install(self._runner(), lambda resources, cfg: declined)
        self.assertIsNone(result)
        self.assertEqual(logs.records[0].levelno, logging.WARNING)

    def test_missing_spec_and_role_mismatch_warn(self):
        with (
            patch(
                "sglang.srt.kernel_plugins.npu.binding.find_spec",
                return_value=None,
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.resolved_view",
                side_effect=lambda args: args,
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.get_parallel",
                return_value=NS(tp_group=NS(cpu_group="cpu")),
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.dist.get_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.kernel_plugins.npu.binding.dist.all_gather_object",
                side_effect=lambda ranks, signature, group=None: ranks.__setitem__(
                    slice(None), [signature]
                ),
            ),
            self.assertLogs(_LOG, level="WARNING") as logs,
        ):
            self.assertIsNone(install_megakernels(self._runner()))
        self.assertIn("no megakernel spec", logs.output[0])

        with self.assertLogs(_LOG, level="WARNING") as logs:
            self.assertIsNone(
                self._install(self._runner(), lambda resources, cfg: None, role="nextn")
            )
        self.assertIn("does not match", logs.output[0])

    def test_prepare_exception_fails_preflight(self):
        def prepare(resources, cfg):
            raise ValueError("pool layout")

        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            self._install(self._runner(), prepare)

    def test_peer_disagreement_fails_preflight(self):
        binding = NS(role="target", mode="decode_loop", build=Mock())

        def peers(signature):
            other = (signature[0], "nextn", *signature[2:])
            return [signature, other]

        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            self._install(
                self._runner(),
                lambda resources, cfg: binding,
                peers=peers,
                world=2,
            )

    def test_peer_error_fails_preflight(self):
        binding = NS(role="target", mode="decode_loop", build=Mock())

        def peers(signature):
            bad = (*signature[:5], "RuntimeError: boom", signature[6])
            return [signature, bad]

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self._install(
                self._runner(),
                lambda resources, cfg: binding,
                peers=peers,
                world=2,
            )


if __name__ == "__main__":
    unittest.main()
