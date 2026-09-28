"""Four-engine PP prefill / DCP decode correctness test with Mooncake TCP.

Run on six free GPUs (the two PP prefill engines share the first two):
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PYTHONPATH=python:. python -m pytest \
    test/manual/hicache/test_pp_prefill_dcp_decode_mooncake.py -q -s
Uses the model, Mooncake tools, and DCP_MOONCAKE_OUTPUT_DIR/PORT settings
documented in test_dcp_mooncake.py. Leave DCP_MOONCAKE_LAYOUT=page_first.
"""

import json
import os
import re
import secrets
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from test_dcp_mooncake import MODEL, MooncakeTestBase
from transformers import AutoConfig

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.test_utils import (
    popen_launch_pd_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready


class TestPPPrefillDCPDecodeMooncake(MooncakeTestBase):
    def setUp(self):
        super().setUp()
        self.assertGreaterEqual(len(self.devices), 6)
        self.assertEqual(self.layout, "page_first")
        config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        self.layers = config.num_hidden_layers
        self.layer_page_bytes = 64 * (config.kv_lora_rank + config.qk_rope_head_dim) * 2
        namespace = "pd-" + uuid.uuid4().hex
        self.tags = {role: f"{namespace}-{role}" for role in ("prefill", "decode")}
        self.expected_objects = {}
        self.object_counts = []
        self.results = []
        self.engines = {}
        for index, (name, role, devices) in enumerate(
            (
                ("P1", "prefill", self.devices[:2]),
                ("P2", "prefill", self.devices[:2]),
                ("D1", "decode", self.devices[2:4]),
                ("D2", "decode", self.devices[4:6]),
            )
        ):
            self.engines[name] = self.start_engine(name, role, devices, index)
        for engine in self.engines.values():
            wait_for_http_ready(
                url=engine["url"] + "/health", timeout=300, process=engine["process"]
            )
            response = requests.get(engine["url"] + "/server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            (self.output / f"{engine['name']}-config.json").write_text(
                json.dumps(info, indent=2)
            )
            prefill = engine["role"] == "prefill"
            for field, expected in dict(
                tp_size=1 if prefill else 2,
                pp_size=2 if prefill else 1,
                dcp_size=1 if prefill else 2,
                hicache_mem_layout="page_first",
                hicache_storage_backend="mooncake",
                enable_hierarchical_cache=True,
                disaggregation_mode=engine["role"],
            ).items():
                self.assertEqual(info[field], expected, (engine["name"], field))
            if not prefill:
                self.assertTrue(info["disaggregation_decode_enable_radix_cache"])
                self.assertFalse(info["disable_radix_cache"])

    def start_engine(self, name, role, devices, index):
        prefill = role == "prefill"
        url = f"http://127.0.0.1:{self.port + 20 + index}"
        bootstrap_port = self.port + 30 + index
        extra = dict(self.extra, extra_backend_tag=self.tags[role])
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            role,
            "--disaggregation-transfer-backend",
            "mooncake_tcp",
            "--disaggregation-bootstrap-port",
            str(bootstrap_port),
            "--nccl-port",
            str(self.port + 40 + index),
            "--tp-size",
            "1" if prefill else "2",
            "--pp-size",
            "2" if prefill else "1",
            "--dcp-size",
            "1" if prefill else "2",
            "--attention-backend",
            "flashinfer",
            "--dcp-comm-backend",
            "ag_rs",
            "--dtype",
            "bfloat16",
            "--page-size",
            "64",
            "--context-length",
            "8192",
            "--chunked-prefill-size",
            "2048",
            "--max-running-requests",
            "4",
            "--max-total-tokens",
            "8192",
            "--mem-fraction-static",
            "0.25" if prefill else "0.4",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "4",
            "--hicache-mem-layout",
            "page_first",
            "--hicache-io-backend",
            "kernel",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            "mooncake",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps(extra),
            "--enable-cache-report",
            "--log-level",
            "debug",
            "--random-seed",
            "42",
        ]
        if prefill:
            args.append("--disable-overlap-schedule")
        else:
            args.extend(
                [
                    "--disaggregation-decode-enable-radix-cache",
                    "--disaggregation-decode-retraction-backup",
                    "cpu_tensor",
                ]
            )
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(devices),
            MOONCAKE_PROTOCOL="tcp",
            MC_FORCE_TCP="1",
            MC_TCP_ENABLE_CONNECTION_POOL="true",
        )
        (self.output / f"{name}-launch.json").write_text(
            json.dumps(
                dict(
                    model=MODEL,
                    url=url,
                    args=args,
                    env={
                        key: env[key]
                        for key in (
                            "CUDA_VISIBLE_DEVICES",
                            "MOONCAKE_PROTOCOL",
                            "MC_FORCE_TCP",
                            "MC_TCP_ENABLE_CONNECTION_POOL",
                        )
                    },
                ),
                indent=2,
            )
        )
        log_path = self.output / f"{name}.log"
        log = log_path.open("w")
        self.addCleanup(log.close)
        process = popen_launch_pd_server(
            MODEL,
            url,
            timeout=300,
            other_args=args,
            env=env,
            return_stdout_stderr=(log, log),
        )
        self.addCleanup(terminate_and_kill_process_tree, process)
        return dict(
            name=name,
            role=role,
            url=url,
            bootstrap_port=bootstrap_port,
            process=process,
            log=log_path,
        )

    @staticmethod
    def storage_tokens(response):
        details = response["meta_info"]["cached_tokens_details"]
        return details["storage"] if details is not None else 0

    def pair(self, label, prefill, decode, tokens, output_len=8):
        p, d = self.engines[prefill], self.engines[decode]
        starts = {
            name: self.engines[name]["log"].stat().st_size for name in (prefill, decode)
        }
        payload = dict(
            input_ids=tokens,
            rid=uuid.uuid4().hex,
            bootstrap_host="127.0.0.1",
            bootstrap_port=p["bootstrap_port"],
            bootstrap_room=secrets.randbits(63),
            sampling_params=dict(
                temperature=0, max_new_tokens=output_len, ignore_eos=True
            ),
        )
        with ThreadPoolExecutor(2) as executor:
            futures = [
                executor.submit(
                    requests.post, e["url"] + "/generate", json=payload, timeout=180
                )
                for e in (p, d)
            ]
            responses = [future.result() for future in futures]
        for response in responses:
            self.assertTrue(
                response.ok, f"{label}: {response.status_code} {response.text}"
            )
        record = dict(
            label=label,
            prefill=prefill,
            decode=decode,
            request=payload,
            prefill_response=responses[0].json(),
            decode_response=responses[1].json(),
        )
        record["log_fragments"] = {}
        for name, start in starts.items():
            with self.engines[name]["log"].open() as log:
                log.seek(start)
                record["log_fragments"][name] = log.read()
        (self.output / f"{label}.json").write_text(json.dumps(record, indent=2))
        return record

    def object_sizes(self, tokens, role):
        logical_page = 64 if role == "prefill" else 128
        full_tokens = tokens[: len(tokens) // logical_page * logical_page]
        prefix = f"{self.tags[role]}_deepseek-ai-DeepSeek-V2-Lite-Chat"
        if role == "decode":
            prefix += "_tp2_dcp2_page128_pp1_cp1"
        result = {}
        for page_hash in get_storage_hash_str(
            full_tokens, None, page_size=logical_page
        ):
            for rank in range(2):
                if role == "prefill":
                    first, end = get_pp_indices(self.layers, rank, 2)
                    suffix, size = str(rank), (end - first) * self.layer_page_bytes
                else:
                    suffix, size = f"dcp{rank}_cp0", self.page_bytes
                result[f"{prefix}_{page_hash}_{suffix}_k"] = size
        return result

    def wait_objects(self, objects):
        keys = list(objects)
        deadline = time.monotonic() + 60
        while self.store.batch_is_exist(keys) != [1] * len(keys):
            self.assertLess(time.monotonic(), deadline, "L3 writes did not complete")
            time.sleep(0.1)
        self.assertEqual({key: self.store.get_size(key) for key in keys}, objects)

    def record(
        self, result, baseline=None, prefill_l3=None, decode_l3=None, cold=False
    ):
        p, d = result["prefill_response"], result["decode_response"]
        self.assertEqual(
            len(d["output_ids"]), result["request"]["sampling_params"]["max_new_tokens"]
        )
        self.assertEqual(d["meta_info"]["num_retractions"], 0)
        if cold:
            self.assertEqual(p["meta_info"]["cached_tokens"], 0)
            self.assertEqual(d["meta_info"]["cached_tokens"], 0)
        if baseline is not None:
            self.assertEqual(d["output_ids"], baseline["decode_response"]["output_ids"])
        rank_counts = {}
        for role, response, expected in (
            ("prefill", p, prefill_l3),
            ("decode", d, decode_l3),
        ):
            # PD copies prefill cache statistics into the decode response.
            # Decode-side L3 reads must be verified on the decode workers.
            if expected is not None and role == "prefill":
                self.assertEqual(
                    self.storage_tokens(response), expected, (result["label"], role)
                )
            log = result["log_fragments"][result[role]]
            pattern = (
                r"PP(\d+)[^\n]*HiCache prefetch success[^\n]*completed=(\d+)"
                if role == "prefill"
                else r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)"
            )
            counts = {int(rank): int(count) for rank, count in re.findall(pattern, log)}
            rank_counts[role] = counts
            if expected:
                self.assertEqual(counts, {0: expected, 1: expected}, result["label"])
            elif expected == 0:
                self.assertTrue(all(count == 0 for count in counts.values()), counts)
        tokens = result["request"]["input_ids"]
        for role, stored_tokens in (
            ("prefill", tokens),
            ("decode", tokens + d["output_ids"][:-1]),
        ):
            objects = self.object_sizes(stored_tokens, role)
            self.wait_objects(objects)
            self.expected_objects.update(objects)
        row = dict(
            label=result["label"],
            pair=f"{result['prefill']}->{result['decode']}",
            prefill_l3=self.storage_tokens(p),
            decode_l3=rank_counts["decode"].get(0, 0),
            rank_counts=rank_counts,
            output_ids=d["output_ids"],
        )
        self.results.append(row)
        (self.output / "pd-results.json").write_text(json.dumps(self.results, indent=2))
        print("PD_MOONCAKE=" + json.dumps(row), flush=True)

    def flush(self):
        for engine in self.engines.values():
            response = requests.post(
                engine["url"] + "/flush_cache", params={"timeout": 30}, timeout=40
            )
            self.assertTrue(response.ok, (engine["name"], response.text))

    def remove_object(self, key):
        self.assertEqual(
            self.store.remove_by_regex(f"^{re.escape(key)}$", force=True), 1
        )
        self.assertEqual(self.store.is_exist(key), 0)

    def check_and_clear_objects(self, phase):
        self.wait_objects(self.expected_objects)
        (self.output / f"{phase}-objects.json").write_text(
            json.dumps(self.expected_objects, indent=2)
        )
        for role in ("prefill", "decode"):
            sizes = [
                size
                for key, size in self.expected_objects.items()
                if key.startswith(self.tags[role] + "_")
            ]
            actual = self.store.remove_by_regex(f"^{self.tags[role]}_.*", force=True)
            self.assertEqual(actual, len(sizes))
            self.assertEqual(actual % 2, 0)
            self.object_counts.append(
                dict(
                    phase=phase,
                    role=role,
                    objects=actual,
                    logical_pages=actual // 2,
                    total_bytes=sum(sizes),
                    object_byte_sizes=sorted(set(sizes)),
                )
            )
        (self.output / "pd-object-counts.json").write_text(
            json.dumps(self.object_counts, indent=2)
        )
        self.expected_objects.clear()

    def test_four_engines(self):
        a, b = self.prompts
        cold_a = self.pair("cold-a", "P1", "D1", a)
        self.record(cold_a, prefill_l3=0, decode_l3=0, cold=True)
        cold_b = self.pair("cold-b", "P2", "D2", b)
        self.record(cold_b, prefill_l3=0, decode_l3=0, cold=True)
        for label, p, d, tokens, baseline, p_l3, d_l3 in (
            ("prefill-a", "P2", "D1", a, cold_a, 1024, 0),
            ("prefill-b", "P1", "D2", b, cold_b, 1024, 0),
            ("decode-a", "P1", "D2", a, cold_a, 0, 1024),
            ("decode-b", "P2", "D1", b, cold_b, 0, 1024),
        ):
            self.record(self.pair(label, p, d, tokens), baseline, p_l3, d_l3)

        # A new history isolates the generated-token test from the basic prompts.
        c = a[:1] + b[10:12] + a[3:]
        self.assertNotIn(c[:64], (a[:64], b[:64]))
        generated = self.pair("generated-cold", "P1", "D1", c, output_len=256)
        self.record(generated, prefill_l3=0, decode_l3=0, cold=True)
        history = c + generated["decode_response"]["output_ids"] + b[1:65]
        self.flush()
        continuation = self.pair("generated-reuse", "P2", "D2", history)
        self.record(continuation, prefill_l3=1024, decode_l3=1280)
        # A clean L3 namespace and empty local caches provide a cold reference
        # for the longer history without modifying the saved response.
        self.flush()
        self.check_and_clear_objects("before-reference")
        cold_continuation = self.pair("generated-reference", "P1", "D1", history)
        self.record(
            cold_continuation, continuation, prefill_l3=0, decode_l3=0, cold=True
        )

        # Rebuild prompt A before testing an incomplete page in each namespace.
        self.flush()
        rebuilt = self.pair("rebuild-a", "P1", "D1", a)
        self.record(rebuilt, cold_a)
        self.flush()
        self.remove_object(list(self.object_sizes(a, "prefill"))[5])
        partial_p = self.pair("missing-pp-stage", "P2", "D2", a)
        self.record(partial_p, cold_a, prefill_l3=128, decode_l3=1024)
        self.flush()
        self.remove_object(list(self.object_sizes(a, "decode"))[5])
        partial_d = self.pair("missing-dcp-shard", "P1", "D1", a)
        self.record(partial_d, cold_a, prefill_l3=1024, decode_l3=256)

        # Exercise every P/D pairing with two requests in flight.
        for round_index, pairs in enumerate(
            ((("P1", "D1"), ("P2", "D2")), (("P1", "D2"), ("P2", "D1")))
        ):
            with ThreadPoolExecutor(2) as executor:
                futures = [
                    executor.submit(
                        self.pair,
                        f"concurrent-{round_index}-{index}",
                        p,
                        d,
                        (a, b)[index],
                    )
                    for index, (p, d) in enumerate(pairs)
                ]
                results = [future.result() for future in futures]
            for result, baseline in zip(results, (cold_a, cold_b)):
                self.record(result, baseline)

        self.check_and_clear_objects("final")
        for engine in self.engines.values():
            self.assertIsNone(engine["process"].poll(), engine["name"])
            response = requests.get(engine["url"] + "/health", timeout=10)
            self.assertTrue(response.ok, engine["name"])


if __name__ == "__main__":
    unittest.main()
