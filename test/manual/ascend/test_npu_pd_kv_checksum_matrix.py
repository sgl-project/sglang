"""Extended Ascend PD checksum tests with exact input-token lengths."""
import json
import random
import time
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import test_npu_pd_kv_checksum as base


class TestNpuPdKvChecksumMatrix(base.TestNpuPdKvChecksum):
    PAGE_SIZE = 128
    matrix_records = []
    _record_lock = threading.Lock()
    COLD_LENGTHS = (
        1, 2, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193, 10000,
        16383, 16384, 16385, 24577, 32768,
    )

    @classmethod
    def launch_all(cls):
        for args in (cls.extra_prefill_args, cls.extra_decode_args):
            args.extend(["--page-size", str(cls.PAGE_SIZE)])
            if "--disable-overlap-schedule" in args:
                raise RuntimeError("This regression suite requires overlap enabled")
        super().launch_all()

    def token_ids(self, length, seed):
        # Direct IDs prevent decode/re-tokenize from changing boundary lengths.
        pool = self.tokenizer.encode(
            "The quick brown fox checks the shared key value cache. "
            "Numbers 0123456789 and different words exercise token pages. "
            "\u8fd9\u662f\u7528\u4e8e\u9a8c\u8bc1\u7f13\u5b58\u4f20\u8f93\u6821\u9a8c\u7684\u6d4b\u8bd5\u6587\u672c\u3002",
            add_special_tokens=False,
        )
        special = set(self.tokenizer.all_special_ids)
        pool = [token for token in pool if token not in special]
        self.assertTrue(pool)
        return random.Random(seed).choices(pool, k=length)

    def flush(self, *, prefill=True, decode=True):
        urls = ([self.prefill_url] if prefill else []) + ([self.decode_url] if decode else [])
        for url in urls:
            deadline = time.monotonic() + 30
            while True:
                response = requests.get(f"{url}/flush_cache", timeout=10)
                if response.status_code == 200:
                    break
                self.assertEqual(response.status_code, 400, response.text)
                if time.monotonic() >= deadline:
                    self.fail(f"Cache could not be flushed at {url}: {response.text}")
                time.sleep(0.25)

    def request_ids(self, ids, label, *, output_tokens=4, corrupt=False):
        rid = f"matrix-{label}-{uuid.uuid4().hex}"
        if corrupt:
            rid = base.CORRUPT_RID_PREFIX + rid
        start = time.monotonic()
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "input_ids": ids,
                "rid": rid,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        record = {
            "label": label, "rid": rid, "input_tokens": len(ids),
            "requested_output_tokens": output_tokens,
            "expected_corruption": corrupt, "http_status": response.status_code,
            "elapsed_s": round(time.monotonic() - start, 4),
        }
        data = None
        if response.status_code == 200:
            data = response.json()
            meta = data["meta_info"]
            record.update(
                prompt_tokens=meta["prompt_tokens"],
                completion_tokens=meta["completion_tokens"],
                cached_tokens=meta.get("cached_tokens", 0),
            )
        else:
            record["error"] = response.text[:1000]
        with self._record_lock:
            self.matrix_records.append(record)
            print("MATRIX_ITEM " + json.dumps(record, sort_keys=True), flush=True)
        if corrupt:
            self.assertEqual(response.status_code, 500, response.text)
            self.assertIn("KV checksum mismatch", response.text)
            return None
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(data["meta_info"]["prompt_tokens"], len(ids), record)
        self.assertEqual(data["meta_info"]["completion_tokens"], output_tokens, record)
        return data

    def test_matrix_cold_lengths(self):
        for length in self.COLD_LENGTHS:
            with self.subTest(input_tokens=length):
                self.flush()
                data = self.request_ids(self.token_ids(length, length), f"cold-{length}")
                self.assertEqual(data["meta_info"].get("cached_tokens", 0), 0)

    def test_matrix_chunk_repetition_and_page_reuse(self):
        # Alternate long/short inputs so released KV pages are reused.
        for repeat in range(3):
            self.flush()
            for length in (8193, 129, 16385, 257):
                with self.subTest(repeat=repeat, input_tokens=length):
                    self.request_ids(
                        self.token_ids(length, 100000 + repeat * 100 + length),
                        f"reuse-{repeat}-{length}",
                    )

    def test_matrix_cached_prefix_boundaries(self):
        for prefix_len, suffix_len in (
            (128, 1), (129, 129), (256, 257),
            (257, 8193), (8192, 1), (8193, 129),
        ):
            with self.subTest(prefix_tokens=prefix_len, suffix_tokens=suffix_len):
                self.flush()
                prefix = self.token_ids(prefix_len, 200000 + prefix_len)
                self.request_ids(
                    prefix + self.token_ids(37, 300000 + prefix_len),
                    f"prefix-fill-{prefix_len}",
                )
                # Retain prefill's prefix, force decode to receive it.
                self.flush(prefill=False)
                data = self.request_ids(
                    prefix + self.token_ids(suffix_len, 400000 + prefix_len),
                    f"prefix-probe-{prefix_len}-{suffix_len}",
                )
                expected_hit = prefix_len // self.PAGE_SIZE * self.PAGE_SIZE
                self.assertGreaterEqual(data["meta_info"]["cached_tokens"], expected_hit)

    def concurrent_group(self, requests_to_send, min_cached_tokens=0):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self.request_ids, ids, label): label
                for ids, label in requests_to_send
            }
            for future in as_completed(futures):
                with self.subTest(label=futures[future]):
                    data = future.result()
                    if min_cached_tokens:
                        self.assertGreaterEqual(
                            data["meta_info"]["cached_tokens"], min_cached_tokens
                        )

    def test_matrix_concurrent_mixed_lengths(self):
        for repeat, lengths in enumerate(
            ((129, 8193, 16385, 32768), (128, 8192, 8193, 16384))
        ):
            self.flush()
            self.concurrent_group([
                (self.token_ids(length, 500000 + repeat * 100 + i), f"mixed-{repeat}-{length}")
                for i, length in enumerate(lengths)
            ])

    def test_matrix_concurrent_shared_prefix(self):
        self.flush()
        prefix = self.token_ids(1024, 600000)
        self.request_ids(prefix + self.token_ids(37, 600001), "shared-prefix-fill")
        for repeat in range(3):
            # Flush only between fully completed concurrent batches.
            self.flush(prefill=False)
            self.concurrent_group([
                (
                    prefix + self.token_ids(length, 610000 + repeat * 100 + i),
                    f"shared-{repeat}-{length}",
                )
                for i, length in enumerate((1, 129, 8193, 16385))
            ], min_cached_tokens=1024)

    def test_matrix_output_lengths(self):
        for output_tokens in (1, 32, 128):
            with self.subTest(output_tokens=output_tokens):
                self.flush()
                self.request_ids(
                    self.token_ids(1025, 700000 + output_tokens),
                    f"output-{output_tokens}", output_tokens=output_tokens,
                )

    def test_matrix_corruption_and_recovery(self):
        for length in (129, 8193, 16385):
            with self.subTest(corrupted_input_tokens=length):
                self.flush()
                self.request_ids(
                    self.token_ids(length, 800000 + length),
                    f"corrupt-{length}", corrupt=True,
                )
                self.request_ids(
                    self.token_ids(129, 900000 + length), f"recovery-{length}"
                )


if __name__ == "__main__":
    unittest.main()
