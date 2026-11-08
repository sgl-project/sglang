"""Keep a live token stream across CPU offload and a staged weight refit."""

import json
import math
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from transformers import AutoModelForCausalLM

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


class TestPartialKvMemoryRelease(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    tp_size = 1

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-memory-saver",
                "--enable-weights-cpu-backup",
                "--enable-kv-cache-cpu-backup",
                "--skip-tokenizer-init",
                "--enable-prefill-weight-versions",
                "--attention-backend",
                "triton",
                "--tp-size",
                str(cls.tp_size),
                "--weight-version",
                "before",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "2",
                "--cuda-graph-max-bs-decode",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def _control(self, endpoint, payload):
        response = requests.post(self.base_url + endpoint, json=payload, timeout=180)
        response.raise_for_status()
        return response.json() if response.content else None

    def _train_checkpoint(self, directory):
        actor = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to("cuda:0")
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, foreach=False)
        inputs = torch.tensor([[1, 2, 3, 4] * 4], device="cuda:0")
        loss = actor(input_ids=inputs, labels=inputs).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        actor.to("cpu").save_pretrained(directory)
        del optimizer, actor, inputs, loss
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def test_tokens_and_prefill_survive_offload_refit_restore(self):
        payload = dict(
            input_ids=[1, 2, 3, 4] * 16,
            sampling_params=dict(temperature=0, max_new_tokens=768, ignore_eos=True),
            return_logprob=True,
        )
        baseline = self._control(
            "/generate", payload | dict(cache_salt=uuid.uuid4().hex)
        )
        offloaded = False
        prefix_ids = None
        last = None
        with requests.post(
            self.base_url + "/generate",
            json=payload | dict(stream=True, cache_salt=uuid.uuid4().hex),
            stream=True,
            timeout=180,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                last = json.loads(line[6:])
                if not offloaded and len(last["output_ids"]) >= 16:
                    prefix_ids = last["output_ids"][:]
                    self.assertLess(len(prefix_ids), 768)
                    self._control("/pause_generation", dict(mode="in_place"))
                    self._control("/release_memory_occupation", {})
                    self._control("/resume_memory_occupation", dict(tags=["weights"]))
                    result = self._control(
                        "/update_weights_from_disk",
                        dict(
                            model_path=self.model,
                            weight_version="after",
                            flush_cache=False,
                            abort_all_requests=False,
                        ),
                    )
                    self.assertTrue(result["success"], result)
                    self._control(
                        "/resume_memory_occupation",
                        dict(tags=["cuda_graph", "kv_cache"]),
                    )
                    self._control("/continue_generation", {})
                    offloaded = True

        self.assertTrue(offloaded, "The handoff must interrupt an unfinished request")
        self.assertEqual(last["output_ids"][: len(prefix_ids)], prefix_ids)
        self.assertEqual(last["output_ids"], baseline["output_ids"])
        meta = last["meta_info"]
        self.assertEqual(meta["completion_tokens"], 768)
        self.assertEqual(
            [span["version"] for span in meta["weight_versions"]], ["before", "after"]
        )
        self.assertEqual(
            {span["version"] for span in meta["prefill_weight_versions"]}, {"before"}
        )
        for actual, expected in zip(
            meta["output_token_logprobs"],
            baseline["meta_info"]["output_token_logprobs"],
            strict=True,
        ):
            self.assertTrue(math.isfinite(actual[0]))
            self.assertEqual(actual[1], expected[1])
            self.assertAlmostEqual(actual[0], expected[0], delta=1e-3)

    def test_z_changed_weights_and_single_abort_preserve_the_other_stream(self):
        """Train on the yielded GPU and cancel one of two retained requests."""
        self._control(
            "/update_weight_version",
            dict(new_version="train-before", abort_all_requests=False),
        )
        before = self._control("/weights_checker", dict(action="checksum"))["ranks"]
        before = [rank["per_gpu_checksum"] for rank in before]
        self.assertEqual(len(before), self.tp_size)
        request_ids = [uuid.uuid4().hex for _ in range(2)]
        ready = [threading.Event() for _ in request_ids]
        latest = {}

        def consume(index):
            with requests.post(
                self.base_url + "/generate",
                json=dict(
                    rid=request_ids[index],
                    cache_salt=request_ids[index],
                    input_ids=[1, 2, 3, 4] * 16,
                    stream=True,
                    return_logprob=True,
                    sampling_params=dict(
                        temperature=0, max_new_tokens=768, ignore_eos=True
                    ),
                ),
                stream=True,
                timeout=180,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if line.startswith(b"data: ") and line != b"data: [DONE]":
                        latest[index] = json.loads(line[6:])
                        if len(latest[index]["output_ids"]) >= 16:
                            ready[index].set()
            return latest[index]

        completed = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(consume, i) for i in range(2)]
            try:
                self.assertTrue(all(event.wait(30) for event in ready))
                self._control("/pause_generation", dict(mode="in_place"))
                prefix = latest[0]["output_ids"][:]
                prefix_logprobs = latest[0]["meta_info"]["output_token_logprobs"][:]
                self.assertLess(len(prefix), 768)
                free_before = torch.cuda.mem_get_info()[0]
                self._control("/release_memory_occupation", {})
                self.assertGreater(torch.cuda.mem_get_info()[0], free_before)
                self._control("/abort_request", dict(rid=request_ids[1]))
                with tempfile.TemporaryDirectory() as directory:
                    self._train_checkpoint(directory)
                    self._control("/resume_memory_occupation", dict(tags=["weights"]))
                    result = self._control(
                        "/update_weights_from_disk",
                        dict(
                            model_path=directory,
                            weight_version="train-after",
                            flush_cache=False,
                            abort_all_requests=False,
                        ),
                    )
                    self.assertTrue(result["success"], result)
                after = self._control("/weights_checker", dict(action="checksum"))[
                    "ranks"
                ]
                after = [rank["per_gpu_checksum"] for rank in after]
                self.assertEqual(len(after), self.tp_size)
                self.assertTrue(
                    all(old != new for old, new in zip(before, after, strict=True))
                )
                self._control(
                    "/resume_memory_occupation", dict(tags=["cuda_graph", "kv_cache"])
                )
                self._control("/continue_generation", {})
                survivor, cancelled = [future.result(timeout=120) for future in futures]
                completed = True
            finally:
                if not completed:
                    terminate_and_kill_process_tree(self.process)

        self.assertEqual(survivor["output_ids"][: len(prefix)], prefix)
        self.assertEqual(
            survivor["meta_info"]["output_token_logprobs"][: len(prefix_logprobs)],
            prefix_logprobs,
        )
        self.assertEqual(survivor["meta_info"]["completion_tokens"], 768)
        self.assertEqual(cancelled["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(
            [s["version"] for s in survivor["meta_info"]["weight_versions"]],
            ["train-before", "train-after"],
        )
        self.assertEqual(
            {s["version"] for s in survivor["meta_info"]["prefill_weight_versions"]},
            {"train-before"},
        )
        self.assertTrue(
            all(
                math.isfinite(p[0])
                for p in survivor["meta_info"]["output_token_logprobs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
