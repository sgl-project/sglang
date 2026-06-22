import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_flip_state_machine():
    path = REPO_ROOT / "python/sglang/srt/disaggregation/flip_state_machine.py"
    spec = importlib.util.spec_from_file_location("flip_state_machine_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPDRuntimeRoleSwitch(unittest.TestCase):
    def test_server_args_declares_runtime_role_switch_flags(self):
        source = (REPO_ROOT / "python/sglang/srt/server_args.py").read_text()

        self.assertIn("enable_pd_runtime_role_switch: bool = False", source)
        self.assertIn(
            'pd_runtime_initial_role: Optional[Literal["prefill", "decode"]] = None',
            source,
        )
        self.assertIn("--enable-pd-runtime-role-switch", source)
        self.assertIn("--pd-runtime-initial-role", source)

    def test_d_to_p_status_declares_hot_switch_and_migration(self):
        fsm = load_flip_state_machine()
        machine = fsm.FlipStateMachine(
            fsm.SLOThresholdFlipEvaluator(slo_threshold=0.9),
            min_window_seconds=0,
        )

        event = machine.tick(
            fsm.ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.5,
                decode_slo_attainment=1.0,
            )
        )

        status = machine.status()
        self.assertEqual(event.to_state, fsm.FlipState.PREPARING)
        self.assertEqual(status["direction"], fsm.FlipDirection.D_TO_P.value)
        self.assertTrue(status["requires_active_request_migration"])
        self.assertTrue(status["can_hot_switch_in_process"])
        self.assertFalse(status["requires_process_restart"])


if __name__ == "__main__":
    unittest.main()
