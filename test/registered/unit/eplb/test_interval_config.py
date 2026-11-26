import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
from pydantic import TypeAdapter

from sglang.srt.arg_groups import parallel_hook
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIntervalConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def resolve(self, role, *args):
        argv = [
            "--model-path",
            "unused",
            "--device",
            "npu",
            "--enable-eplb",
            "--tp-size",
            "16",
            "--ep-size",
            "16",
            "--attn-dp-size",
            "16",
            "--ep-num-redundant-experts",
            "16",
            "--quantization",
            "modelslim",
            "--moe-a2a-backend",
            "deepep",
            *(("--disaggregation-mode", role) if role is not None else ()),
            *args,
        ]
        inputs = ServerArgs.from_cli_args(self.parser.parse_args(argv))
        hf = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], n_routed_experts=256
        )
        with (
            patch.object(
                parallel_hook,
                "model_config_of",
                return_value=SimpleNamespace(hf_config=hf),
            ),
            patch(
                "sglang.srt.utils.device.get_device_name", return_value="Ascend910_9391"
            ),
        ):
            parallel_hook.handle_eplb_and_dispatch(inputs)
            parallel_hook.handle_expert_distribution_metrics(inputs)
        cfg = resolving_view(inputs)
        return (
            cfg.eplb_interval_policy,
            cfg.eplb_rebalance_num_iterations,
            cfg.expert_distribution_recorder_buffer_size,
        )

    def test_role_profiles_and_explicit_values(self):
        self.assertEqual(self.resolve("decode"), ("fixed", 1000, 1000))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.resolve("decode", "--eplb-rebalance-num-iterations", "0")
        self.assertEqual(
            self.resolve("prefill", "--eplb-interval-policy", "auto"), ("auto", 50, 50)
        )
        self.assertEqual(
            self.resolve("decode", "--eplb-interval-policy", "recommend"),
            ("recommend", 4000, 1000),
        )
        self.assertEqual(
            self.resolve(
                "decode",
                "--eplb-interval-policy",
                "recommend",
                "--eplb-rebalance-num-iterations",
                "1000",
            ),
            ("recommend", 1000, 1000),
        )
        self.assertEqual(
            self.resolve(
                "prefill",
                "--eplb-interval-policy",
                "auto",
                "--expert-distribution-recorder-buffer-size",
                "128",
            ),
            ("auto", 50, 128),
        )
        self.assertEqual(
            self.resolve(
                "decode",
                "--eplb-interval-policy",
                "auto",
                "--ep-dispatch-algorithm",
                "dynamic",
            ),
            ("fixed", 1000, 1000),
        )
        with self.assertRaisesRegex(ValueError, "recorder window"):
            self.resolve(
                "prefill",
                "--eplb-interval-policy",
                "auto",
                "--eplb-rebalance-num-iterations",
                "2",
                "--expert-distribution-recorder-buffer-size",
                "0",
            )

    def test_colocated_interval_uses_one_neutral_starting_value(self):
        for policy in ("auto", "recommend", "fixed"):
            with self.subTest(policy=policy):
                self.assertEqual(
                    self.resolve(None, "--eplb-interval-policy", policy),
                    (policy, 1000, 1000),
                )
        self.assertEqual(
            self.resolve(
                "null",
                "--eplb-interval-policy",
                "auto",
                "--eplb-rebalance-num-iterations",
                "25",
                "--expert-distribution-recorder-buffer-size",
                "50",
            ),
            ("auto", 25, 50),
        )
        self.assertEqual(
            self.resolve(
                None, "--eplb-interval-policy", "auto", "--enable-pd-role-switch"
            ),
            ("fixed", 1000, 1000),
        )

    def test_admin_schema_wire_and_handler(self):
        request = TypeAdapter(SetInternalStateReq).validate_json(
            '{"eplb_interval":{"policy":"fixed","interval":4000}}'
        )
        request = msgspec.msgpack.decode(
            msgspec.msgpack.encode(request), type=SetInternalStateReq
        )
        controller = Mock()
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    eplb_manager=SimpleNamespace(interval_controller=controller)
                )
            )
        )
        handler = SchedulerInternalStateController(scheduler, False)
        self.assertTrue(handler.set_internal_state(request).updated)
        controller.request.assert_called_once_with("fixed", 4000)
        request.server_args["pp_max_micro_batch_size"] = 1
        self.assertFalse(handler.set_internal_state(request).updated)
        controller.request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
