"""Four-GPU PP2/EP2 recorder terminals, failure recovery and real migration."""

import math
import re
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import requests
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")


class _PPEPLBServer(CustomTestCase):
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MLA
    automatic = False

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dumps = cls.root / "dumps"
        cls.dumps.mkdir()
        cls.log = (cls.root / "server.log").open("w")
        cls.base_url = DEFAULT_URL_FOR_TEST
        args = [
            "--trust-remote-code",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--ep-size",
            "2",
            "--max-running-requests",
            "1",
            "--context-length",
            "4096",
            "--mem-fraction-static",
            "0.6",
            "--ep-num-redundant-experts",
            "8",
            "--expert-distribution-recorder-mode",
            "stat",
            "--expert-distribution-recorder-buffer-size",
            "8",
            "--eplb-rebalance-num-iterations",
            "8",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
        if cls.automatic:
            args.append("--enable-eplb")
        with ExitStack() as stack:
            for field, value in (
                (envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR, str(cls.dumps)),
                (envs.SGLANG_EXPERT_LOCATION_UPDATER_CANARY, True),
                (envs.SGLANG_LOG_EXPERT_LOCATION_METADATA, True),
            ):
                stack.enter_context(field.override(value))
            cls.process = popen_launch_server(
                try_cached_model(cls.model_name),
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=args,
                return_stdout_stderr=(cls.log, cls.log),
            )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            terminate_and_kill_process_tree(cls.process)
        if getattr(cls, "log", None):
            cls.log.close()
        if getattr(cls, "temp", None):
            cls.temp.cleanup()

    def control(self, action, *, status=200):
        response = requests.post(
            f"{self.base_url}/{action}_expert_distribution_record", timeout=120
        )
        self.assertEqual(response.status_code, status, response.text)
        return response

    def generate(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Explain why the sky appears blue in clear weather.",
                "sampling_params": {
                    "temperature": 0,
                    "sampling_seed": 42,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["meta_info"]["completion_tokens"], 32)
        probabilities = result["meta_info"]["output_token_logprobs"]
        self.assertTrue(all(math.isfinite(p[0]) for p in probabilities))
        return result


class TestPPRecorderControl(_PPEPLBServer):
    def test_stage_dumps_and_failed_dump_recovery(self):
        self.control("start")
        self.generate()
        self.control("stop")
        self.control("stop")
        self.control("dump")
        records = [
            torch.load(p, map_location="cpu", weights_only=False)
            for p in self.dumps.glob("*.pt")
        ]
        self.assertEqual(len(records), 2)
        self.assertEqual({r["pp_rank"] for r in records}, {0, 1})
        self.assertEqual(len({r["operation_id"] for r in records}), 1)
        owned = []
        for record in records:
            self.assertTrue(record["operation_id"])
            self.assertTrue(record["layout_hash"])
            self.assertGreater(record["logical_count"].sum().item(), 0)
            layer_ids = set(record["owned_layer_ids"])
            nonzero = set(
                record["logical_count"].sum((0, 2)).nonzero().flatten().tolist()
            )
            self.assertLessEqual(nonzero, layer_ids)
            owned.append(layer_ids)
        self.assertTrue(owned[0].isdisjoint(owned[1]))

        self.control("start")
        self.generate()
        self.control("stop")
        backup = self.root / "successful-dumps"
        self.dumps.rename(backup)
        self.dumps.write_text("task-owned failure injection")
        try:
            response = self.control("dump", status=500)
            self.assertIn("PP0", response.text)
            self.assertIn("PP1", response.text)
        finally:
            self.dumps.unlink()
            backup.rename(self.dumps)
        self.control("start")
        self.generate()
        self.control("stop")
        self.control("dump")
        self.generate()


class TestPPAutomaticEPLB(_PPEPLBServer):
    automatic = True

    def test_both_stages_rebalance_and_continue(self):
        self.generate()
        self.generate()
        self.log.flush()
        log = (self.root / "server.log").read_text()
        for pp in (0, 1):
            self.assertRegex(log, rf"PP{pp}[^\n]*\[EPLBManager\] rebalance end")
        changes = re.findall(r"changed_physical_slots=(\d+)/", log)
        self.assertTrue(any(int(value) > 0 for value in changes), log[-4000:])
        self.assertNotIn("Scheduler hit an exception", log)
        self.generate()


if __name__ == "__main__":
    unittest.main()
