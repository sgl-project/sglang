"""A prefix-cache hit on a hybrid model's recurrent checkpoint must score the
continuation like a cache miss, also when the cached prompt finished its
prefill inside a mixed chunked-prefill forward."""

import shutil
import tempfile
import threading
import unittest

import requests
import zmq
from transformers import AutoTokenizer

from sglang.srt.observability.forward_pass_metrics import decode as decode_fpm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=140, stage="base-b", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen3.5-0.8B"

NUM_BACKGROUND = 4
NUM_DONORS = 8
# A multiple of the 64-token checkpoint grid.
DONOR_LEN = 320
# One extend covers a whole donor, so its checkpoint is taken inside the forward
# that mixes it with decodes; an extend shorter than the grid takes none.
CHUNK_SIZE = 1024
FILLER = (
    "The history of the printing press begins in the fifteenth century, when "
    "movable type transformed how books were made and shared across Europe. "
) * 40
FOLLOW_UP = " In summary, the most important consequence of this invention was"
# Total absolute logprob difference over the scored suffix. Measured: a correct
# checkpoint stays within 0.5 nat of a cache miss; one never written, 7 to 17 nats.
MAX_SCORE_DELTA = 3.0


class _PrefillForwardCounter:
    """Counts forward passes with prefill rows, split by whether they also carried
    decode rows (MIXED), read from the forward-pass-metrics ZMQ publisher."""

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._stop = threading.Event()
        self.mixed = 0
        self.prefill_only = 0
        # Drain continuously; a SUB socket drops messages past its high-water mark.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(self._endpoint)
        while not self._stop.is_set():
            if not socket.poll(timeout=50):
                continue
            _topic, _seq, payload = socket.recv_multipart()
            scheduled = decode_fpm(payload).scheduled_requests
            if scheduled.num_prefill_requests and scheduled.num_decode_requests:
                self.mixed += 1
            elif scheduled.num_prefill_requests:
                self.prefill_only += 1
        socket.close(linger=0)

    def stop(self) -> tuple[int, int]:
        self._stop.set()
        self._thread.join()
        return self.mixed, self.prefill_only


class TestMambaMixedChunkCheckpoint(CustomTestCase):
    mamba_radix_cache_strategy = "extra_buffer"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.fpm_dir = tempfile.mkdtemp(prefix="sglang-fpm-")
        cls.fpm_base = f"ipc://{cls.fpm_dir}/fpm"
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--random-seed",
                "1",
                "--enable-mixed-chunk",
                "--chunked-prefill-size",
                str(CHUNK_SIZE),
                "--linear-attn-backend",
                "triton",
                "--mamba-backend",
                "triton",
                "--mamba-radix-cache-strategy",
                cls.mamba_radix_cache_strategy,
                "--max-mamba-cache-size",
                "64",
                "--max-total-tokens",
                "32768",
                "--mem-fraction-static",
                "0.3",
                "--enable-forward-pass-metrics",
                "--forward-pass-metrics-ipc-name",
                cls.fpm_base,
            ],
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        cls.filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
        cls.follow_up_ids = tokenizer.encode(FOLLOW_UP, add_special_tokens=False)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "fpm_dir"):
            shutil.rmtree(cls.fpm_dir, ignore_errors=True)

    def _generate(self, body):
        body.setdefault("sampling_params", {}).setdefault("temperature", 0)
        response = requests.post(self.base_url + "/generate", json=body, timeout=300)
        response.raise_for_status()
        return response.json()

    def _flush_cache(self):
        # A generation response can arrive before the scheduler becomes idle.
        requests.post(
            self.base_url + "/flush_cache", params={"timeout": 30}, timeout=60
        ).raise_for_status()

    def _donor_prompt(self, k):
        # Distinct leading token so donors share no prefix with each other.
        return [self.filler_ids[k]] + self.filler_ids[
            NUM_DONORS : NUM_DONORS + DONOR_LEN - 1
        ]

    def _prefill_donors_while_decoding(self) -> tuple[int, int]:
        """Prefill the donors (no logprobs, so they may mix) while background
        requests keep the running batch non-empty; returns the forward counts."""
        background = [
            threading.Thread(
                target=self._generate,
                args=(
                    {
                        "text": f"Write a long story about planet number {k}. ",
                        "sampling_params": {"max_new_tokens": 512, "ignore_eos": True},
                    },
                ),
            )
            for k in range(NUM_BACKGROUND)
        ]
        for thread in background:
            thread.start()
        # Let the background requests reach decode before the donors arrive;
        # from here on, only donor prefills can join a mixed forward.
        self._generate({"text": "Hi.", "sampling_params": {"max_new_tokens": 8}})
        counter = _PrefillForwardCounter(f"{self.fpm_base}.0")
        donors = [
            threading.Thread(
                target=self._generate,
                args=(
                    {
                        "input_ids": self._donor_prompt(k),
                        "sampling_params": {"max_new_tokens": 1},
                    },
                ),
            )
            for k in range(NUM_DONORS)
        ]
        for thread in donors:
            thread.start()
        for thread in donors + background:
            thread.join()
        return counter.stop()

    def _score_suffix(self, input_ids, start):
        result = self._generate(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": start,
            }
        )
        logprobs = [
            logprob
            for logprob, _token_id, _text in result["meta_info"]["input_token_logprobs"]
            if logprob is not None
        ]
        return logprobs, result["meta_info"]["cached_tokens"]

    def test_checkpoint_from_mixed_forward_matches_cache_miss(self):
        self._flush_cache()
        mixed, prefill_only = self._prefill_donors_while_decoding()
        # Every donor checkpoint must come from a mixed forward; one taken in a
        # plain extend batch would pass the comparison below without the bug.
        self.assertGreater(mixed, 0)
        self.assertEqual(prefill_only, 0)

        follow_ups = [
            self._donor_prompt(k) + self.follow_up_ids for k in range(NUM_DONORS)
        ]
        hits = [self._score_suffix(ids, start=DONOR_LEN) for ids in follow_ups]
        self._flush_cache()
        misses = [self._score_suffix(ids, start=DONOR_LEN) for ids in follow_ups]

        for k, ((hit, cached_tokens), (miss, _)) in enumerate(zip(hits, misses)):
            with self.subTest(donor=k, cached_tokens=cached_tokens):
                self.assertGreater(cached_tokens, 0)
                self.assertEqual(len(hit), len(miss))
                delta = sum(abs(a - b) for a, b in zip(hit, miss))
                self.assertLess(delta, MAX_SCORE_DELTA)


class TestMambaMixedChunkCheckpointLazy(TestMambaMixedChunkCheckpoint):
    mamba_radix_cache_strategy = "extra_buffer_lazy"


if __name__ == "__main__":
    unittest.main()
