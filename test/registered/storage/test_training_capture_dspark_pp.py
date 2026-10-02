"""Real PP target-KV draft serving and exact Mooncake snapshot readback."""

import json
import os
import sys
import time
import unittest
from unittest.mock import patch

import requests
import torch
from sglang.test import test_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.dspark_capture_pressure import exercise_dspark_capture_pressure
from sglang.test.dspark_target_kv_runtime import export_synthetic_kv_draft
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase, free_port
from sglang.test.training_capture_utils import exercise_capture_abort, read_snapshot

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestDSparkPipelineCapture(PDCaptureRuntimeBase):
    tp_size = 1

    def launch_colocated(self, root, *, replay, draft=None):
        root.mkdir()
        config = {
            "dataset_id": "dspark-pp",
            "model_id": "Qwen/Qwen3-0.6B",
            "producer_revision": "checkout-under-test",
            "selected_layer_ids": [0, 14, 27],
            "catalog_endpoint": self.catalog.endpoint,
            "journal_directory": str(root / "journal"),
            "store": self.store_setup,
            "sample_ratio": 1.0,
            "max_sample_tokens": 256,
            "max_inflight_samples": 4,
            "max_host_bytes": 64 << 20,
            "kv_d2h_batch_tokens": 16,
            "max_device_bytes": 8 << 20,
            "storage_chunk_tokens": 64,
        }
        path = root / "capture.json"
        path.write_text(json.dumps(config))
        url = f"http://127.0.0.1:{free_port()}"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            module = (
                "sglang.test.dspark_target_kv_server"
                if draft
                else "sglang.test.dspark_capture_server"
            )
            return launch([sys.executable, "-m", module] + command[2:], *args)

        with patch.object(test_utils, "_launch_server_process", observed_server):
            server = test_utils.popen_launch_server(
                self.model,
                url,
                timeout=240,
                env={
                    **os.environ,
                    "SGLANG_RAGGED_VERIFY_MODE": "static",
                    "SGLANG_TEST_RETRACT": "0",
                },
                other_args=[
                    "--tp-size",
                    str(self.tp_size),
                    "--pp-size",
                    "2",
                    "--pp-max-micro-batch-size",
                    "4",
                    "--training-capture-config",
                    str(path),
                    "--disable-overlap-schedule",
                    "--skip-server-warmup",
                    "--enable-metrics",
                    "--attention-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.25",
                    "--max-total-tokens",
                    "512",
                    "--schedule-conservativeness",
                    "0.05",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "128",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-backend-decode",
                    "full" if replay else "disabled",
                    "--cuda-graph-bs-decode",
                    "1",
                    "2",
                    "4",
                    *(
                        [
                            "--speculative-algorithm",
                            "DSPARK",
                            "--speculative-draft-model-path",
                            str(draft),
                            "--speculative-draft-attention-backend",
                            "triton",
                        ]
                        if draft
                        else []
                    ),
                ],
            )
        self.addCleanup(self.stop_process, server)
        return server, url

    def wait_available(self, url, count=1):
        deadline = time.monotonic() + 30
        while True:
            state = requests.get(url + "/server_info", timeout=10).json()[
                "internal_states"
            ][0]["training_capture"]
            if state["states"].get("available", 0) >= count:
                return state
            self.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.05)

    def read_sample(self, publication):
        return read_snapshot(self.reader, publication)

    def generate_colocated(self, url, prompt, **params):
        self.wait_available(url)
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                    **params,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def exercise_pipeline(self, replay):
        root = self.root / f"tp{self.tp_size}-pp2-{replay}"
        root.mkdir()
        first = len(self.catalog.publications)
        server, url = self.launch_colocated(root / "ar", replay=False)
        prompt = [100, 200, 300, 400] * 40
        try:
            baseline = self.generate_colocated(url, prompt)
            publication = self.catalog.wait_publications(first + 1, timeout=30)[-1]
            seed = read_snapshot(self.reader, publication)
        finally:
            self.stop_process(server)
        draft = root / "draft"
        export_synthetic_kv_draft(self.model, draft, *seed)
        server, url = self.launch_colocated(root / "spec", replay=replay, draft=draft)
        start = len(self.catalog.publications)
        results = []
        try:
            results.append(self.generate_colocated(url, prompt))
            self.assertEqual(results[-1]["output_ids"], baseline["output_ids"])
            results.append(
                self.generate_colocated(url, prompt, logit_bias={"100": 100.0})
            )
            self.assertEqual(results[-1]["output_ids"], [100] * 8)
            self.assertGreater(results[-1]["meta_info"]["cached_tokens"], 0)
            self.wait_available(url, 2)
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": [prompt, prompt],
                    "sampling_params": [
                        {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
                        {
                            "temperature": 0,
                            "max_new_tokens": 5,
                            "ignore_eos": True,
                            "logit_bias": {"100": 100.0},
                        },
                    ],
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            results.extend(response.json())
            self.assertEqual(results[-2]["output_ids"], baseline["output_ids"])
            self.assertEqual(results[-1]["output_ids"], [100] * 5)
            results.append(
                self.generate_colocated(
                    url,
                    prompt,
                    temperature=0.8,
                    regex="[0-9]{12}",
                    max_new_tokens=16,
                )
            )
            self.assertRegex(results[-1]["text"], r"^[0-9]{12}$")
            results.append(
                self.generate_colocated(
                    url,
                    prompt,
                    temperature=0.8,
                    top_k=16,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    frequency_penalty=0.15,
                )
            )
            results.append(
                self.generate_colocated(
                    url,
                    prompt,
                    min_new_tokens=1,
                    ignore_eos=False,
                    stop_token_ids=[100],
                    logit_bias={"100": 100.0},
                )
            )
            self.assertEqual(len(results[-1]["output_ids"]), 2)
            self.assertEqual(results[-1]["output_ids"][-1], 100)
            self.assertEqual(results[-1]["meta_info"]["finish_reason"]["type"], "stop")
            publications = self.catalog.wait_publications(
                start + len(results), timeout=30
            )[start:]
            references = [
                torch.load(path, weights_only=True) for path in draft.rglob("*.pt")
            ]
            ranks = {(tp, pp) for tp in range(self.tp_size) for pp in range(2)}
            self.assertEqual(
                {(row["tp_rank"], row["pp_rank"]) for row in references}, ranks
            )
            self.assertTrue(any(row["batch_size"] == 2 for row in references))
            self.assertEqual(any(row["cuda_graph"] for row in references), replay)
            samples = [read_snapshot(self.reader, item) for item in publications]
            for manifest, tensors in samples:
                self.assertEqual(
                    (manifest.topology.tp_size, manifest.topology.pp_size),
                    (self.tp_size, 2),
                )
                check_capture_snapshot(self, manifest, tensors, references)
            self.assertCountEqual(
                [
                    tensors["token_ids"][manifest.sequence.prompt_length :].tolist()
                    for manifest, tensors in samples
                ],
                [result["output_ids"] for result in results],
            )
            observations = [
                json.loads(line)
                for path in draft.glob("observations*.jsonl")
                for line in path.read_text().splitlines()
            ]
            self.assertEqual(
                {
                    (row["tp_rank"], row["pp_rank"])
                    for row in observations
                    if row["kind"] == "projection"
                },
                ranks,
            )
            commits = [row for row in observations if row["kind"] == "verify"]
            self.assertTrue(any(row["num_reject"] > 0 for row in commits))
            self.assertTrue(any(max(row["num_commit"]) > 1 for row in commits))
            self.wait_available(url)
            exercise_capture_abort(
                self,
                url=url,
                rid=f"pp-abort-{replay}",
                prompt=prompt[:8],
                max_new_tokens=248,
                distributed=True,
            )
            self.assertEqual(len(self.catalog.publications), start + len(results))
            publications += exercise_dspark_capture_pressure(
                self,
                url=url,
                directory=draft,
                cuda_graph=replay,
                enable_overlap=False,
                tp_size=self.tp_size,
                pp_size=2,
            )
            print(
                json.dumps(
                    {
                        "tp_size": self.tp_size,
                        "pp_size": 2,
                        "cuda_graph": replay,
                        "checked_snapshots": len(publications),
                        "aborted": True,
                    }
                ),
                flush=True,
            )
        finally:
            self.stop_process(server)
        for item in publications:
            read_snapshot(self.reader, item)

    def test_eager(self):
        self.exercise_pipeline(False)

    def test_cuda_graph(self):
        self.exercise_pipeline(True)


if __name__ == "__main__":
    unittest.main()
