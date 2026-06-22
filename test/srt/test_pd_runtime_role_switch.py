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

    def test_scheduler_declares_hybrid_queue_initialization_helpers(self):
        source = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()

        self.assertIn("enable_pd_runtime_role_switch", source)
        self.assertIn("def _init_decode_disaggregation", source)
        self.assertIn("def _init_prefill_disaggregation", source)
        self.assertIn("def pd_runtime_role", source)
        self.assertIn("def pd_runtime_role_switch_enabled", source)

    def test_load_inquirer_reads_runtime_role_dynamically(self):
        source = (
            REPO_ROOT
            / "python/sglang/srt/managers/scheduler_components/load_inquirer.py"
        ).read_text()

        self.assertIn("get_disaggregation_mode: Callable", source)
        self.assertIn("def _disaggregation_mode", source)
        self.assertNotIn("disaggregation_mode: DisaggregationMode", source)

    def test_worker_runtime_role_control_surface_is_declared(self):
        io_struct = (REPO_ROOT / "python/sglang/srt/managers/io_struct.py").read_text()
        tokenizer_control = (
            REPO_ROOT / "python/sglang/srt/managers/tokenizer_control_mixin.py"
        ).read_text()
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
        http_server = (
            REPO_ROOT / "python/sglang/srt/entrypoints/http_server.py"
        ).read_text()

        self.assertIn("class PDRuntimeRoleSetReq", io_struct)
        self.assertIn("class PDRuntimeRoleStatusReq", io_struct)
        self.assertIn("class PDRuntimeRoleAdmissionReq", io_struct)
        self.assertIn("class PDRuntimeRoleReqOutput", io_struct)

        self.assertIn("(\"pd_runtime_role\", PDRuntimeRoleReqOutput)", tokenizer_control)
        self.assertIn("async def set_pd_runtime_role", tokenizer_control)
        self.assertIn("async def get_pd_runtime_role_status", tokenizer_control)
        self.assertIn("async def set_pd_runtime_admission", tokenizer_control)

        self.assertIn("(PDRuntimeRoleSetReq, self.set_pd_runtime_role)", scheduler)
        self.assertIn(
            "(PDRuntimeRoleStatusReq, self.get_pd_runtime_role_status)", scheduler
        )
        self.assertIn(
            "(PDRuntimeRoleAdmissionReq, self.set_pd_runtime_admission)", scheduler
        )
        self.assertIn("def set_pd_runtime_role", scheduler)
        self.assertIn("def get_pd_runtime_role_status", scheduler)
        self.assertIn('"event_loop_dynamic": True', scheduler)

        self.assertIn("/pd_flip/runtime_role/status", http_server)
        self.assertIn("/pd_flip/runtime_role/set", http_server)
        self.assertIn("/pd_flip/runtime_role/admission", http_server)

    def test_pd_flip_commit_mutates_runtime_role_when_hot_switch_enabled(self):
        scheduler = (REPO_ROOT / "python/sglang/srt/managers/scheduler.py").read_text()

        self.assertIn(
            "if self.pd_runtime_role_switch_enabled():\n            return True",
            scheduler,
        )
        self.assertIn(
            "if self.pd_runtime_role_switch_enabled():\n            target = (",
            scheduler,
        )
        self.assertIn(
            '"prefill"\n                if decision.direction == FlipDirection.D_TO_P',
            scheduler,
        )
        self.assertIn(
            "out = self.set_pd_runtime_role(PDRuntimeRoleSetReq(role=target))",
            scheduler,
        )
        self.assertIn("return out.success", scheduler)


if __name__ == "__main__":
    unittest.main()
