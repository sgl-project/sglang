"""
Test --enable-streaming-session on NPU: functional correctness of
streaming session KV inheritance, abort recovery, and session lifecycle.

Scope:
  - KV cache inheritance across turns (P0 correctness)
  - Nth-turn mid-decode abort → rollback to last successful turn (P1)
  - First-turn mid-decode abort → no inherited context (P1)
  - Pre-abort (offset=1) preserves session slot (P1)
  - Session close / reopen lifecycle (P2)

[Test Category] Parameter
[Test Target] --enable-streaming-session
"""

import threading
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)


# ------------------------------------------------------------------
# Test data
# ------------------------------------------------------------------

CHUNKS = [
    "Let me tell you something about France.",
    "The capital of France is",
    "The population of the city is",
    "A brief history about that city is",
]

SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 12,
    "no_stop_trim": True,
    "skip_special_tokens": False,
}

LONG_SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 100000,  # Will be aborted mid-decode
    "no_stop_trim": True,
    "skip_special_tokens": False,
}

# ------------------------------------------------------------------
# Test class
# ------------------------------------------------------------------


class TestNpuEnableStreamingSession(CustomTestCase):
    """Test --enable-streaming-session parameter: functional correctness of
    streaming session KV cache inheritance, abort recovery, and lifecycle.

    Business scenario: Multi-turn conversational AI (chatbots, assistants)
    that need session-based KV cache persistence across turns.  Streaming
    sessions allow efficient KV reuse within a session while streaming
    tokens to the client.

    [Test Category] Parameter
    [Test Target] --enable-streaming-session
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _launch_server(cls, extra_args=None):
        """Launch a server with NPU fixture args + extra_args.

        Returns the subprocess handle. Caller is responsible for
        ``kill_process_tree(process.pid)`` in a finally block.
        """
        all_args = ["--attention-backend", "ascend"] + (extra_args or [])
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(True):
                return popen_launch_server(
                    cls.model,
                    cls.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=all_args,
                )

    @classmethod
    def _get_tokenizer(cls):
        """Lazy-load tokenizer for self.model."""
        return get_tokenizer(cls.model)

    @staticmethod
    def _open_session(streaming=None, session_id=None, capacity=128):
        """POST /open_session and return the requests.Response.

        streaming: True / False / None (field omitted when None)
        session_id: optional pre-set session id
        """
        body = {"capacity_of_str_len": capacity}
        if streaming is not None:
            body["streaming"] = streaming
        if session_id is not None:
            body["session_id"] = session_id
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/open_session", json=body, timeout=10
        )

    @staticmethod
    def _close_session(session_id):
        """POST /close_session and return the requests.Response."""
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/close_session",
            json={"session_id": session_id},
            timeout=10,
        )

    @staticmethod
    def _generate(payload, timeout=30):
        """POST /generate and return the requests.Response."""
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json=payload,
            timeout=timeout,
        )

    @staticmethod
    def _abort_all():
        """POST /abort_request with abort_all=True."""
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/abort_request",
            json={"rid": "", "abort_all": True},
            timeout=10,
        )

    def _encode_chunks(self, tokenizer):
        """Encode CHUNKS, stripping BOS from non-first chunks."""
        chunk_ids = [tokenizer.encode(x) for x in CHUNKS]
        for i in range(1, len(chunk_ids)):
            if chunk_ids[i][0] == tokenizer.bos_token_id:
                chunk_ids[i] = chunk_ids[i][1:]
        return chunk_ids

    # ==================================================================
    # Part B: KV cache inheritance (P0 correctness)
    # ==================================================================

    # ------------------------------------------------------------------
    # TC-SS-03: KV cache inheritance across turns
    #   Ref: StreamingSessionKitMixin.test_kv_cache_inheritance
    #   Covers: inherit_kv_states for streaming sessions
    # ------------------------------------------------------------------

    def test_kv_cache_inheritance(self):
        """Each turn's cached_tokens must equal the previous turn's
        prompt+completion (KV inherited via inherit_kv_states).

        Multi-turn conversation flow:
          Turn 1: [chunk0] → clean start, no cache hit, builds KV
          Turn 2: [chunk0→KV + chunk1] → cached_tokens == Turn1 KV len
          Turn 3: [chunk0→KV + chunk1→KV + chunk2] → inherits Turn2 KV
          Turn N: inherits (N-1)th turn's full KV (prompt + completion)
        """
        process = self._launch_server(["--enable-streaming-session"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            tokenizer = self._get_tokenizer()
            chunk_ids = self._encode_chunks(tokenizer)
            # prompt_tokens will grow each turn because the session
            # concatenates all previous chunks (they are KV-cached,
            # not re-computed, but still counted as prompt_tokens)

            requests.post(self.base_url + "/flush_cache")

            resp = self._open_session(streaming=True, capacity=1000)
            self.assertEqual(resp.status_code, 200)
            session_id = resp.json()

            # rid: response id to identify which turn the next request
            #      extends; None for the first turn
            rid = None
            # prev_kv_len: total tokens (prompt + completion) of the
            #              previous turn, i.e. the KV cache written
            prev_kv_len = 0

            for turn_idx, ch_ids in enumerate(chunk_ids):
                response = self._generate(
                    {
                        "input_ids": ch_ids,
                        "session_params": {"id": session_id, "rid": rid},
                        "sampling_params": SAMPLING_PARAMS,
                    }
                )
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                rid = data["meta_info"]["id"]
                cached = data["meta_info"]["cached_tokens"]
                prompt_tokens = data["meta_info"]["prompt_tokens"]
                completion_tokens = data["meta_info"]["completion_tokens"]

                turn_total = prompt_tokens + completion_tokens
                print(
                    f"[Turn {turn_idx + 1}] "
                    f"prompt_tokens={prompt_tokens}, "
                    f"completion_tokens={completion_tokens}, "
                    f"turn_total={turn_total}, "
                    f"cached_tokens={cached}, "
                    f"prev_kv_len={prev_kv_len}"
                )

                if turn_idx == 0:
                    # First turn: no prior KV to inherit
                    self.assertEqual(cached, 0, "Turn 1: clean start, no cache hit")
                else:
                    # Turns 2+: cached_tokens must equal previous
                    # turn's total tokens, proving full KV reuse
                    self.assertEqual(
                        cached,
                        prev_kv_len,
                        f"Turn {turn_idx + 1}: inherited {cached} != "
                        f"prev total {prev_kv_len}",
                    )
                # Record this turn's total for the next iteration
                prev_kv_len = turn_total

            close_resp = self._close_session(session_id)
            self.assertEqual(
                close_resp.status_code, 200, "Closing session should return 200"
            )
        finally:
            kill_process_tree(process.pid)

    # ==================================================================
    # Part C: Abort recovery (P1)
    # ==================================================================

    # ------------------------------------------------------------------
    # TC-SS-04: Nth-turn mid-decode abort recovery
    #   Ref: StreamingSessionKitMixin.test_nth_mid_abort_recovery
    #   Covers: abort mid-decode rolls back to last successful turn
    #
    #   Scenario: In a multi-turn streaming session, abort the Nth turn
    #   (N>1) mid-decode. The session state must roll back to the last
    #   successfully completed turn.
    #
    #   Turn 1 → complete (establish KV cache baseline)
    #   Turn 2 → start then abort (verify KV cache inheritance → abort → rollback)
    #   Turn 3 → recovery (verify prompt_tokens == Turn1 history + own input
    #             only; no stale abort context from Turn 2;
    #             verify full 8-token output)
    # ------------------------------------------------------------------

    def test_nth_mid_abort_recovery(self):
        """Abort an Nth-turn request mid-decode; session must roll back
        to the last successful turn.

        Assertions:
        - Turn 1 completes normally → record turn_1_total
        - Turn 2 cached_tokens > 0 (inherits Turn 1 KV cache)
        - Turn 2 finish_reason.type == "abort"
        - Turn 3 prompt_tokens == turn_1_total + len(ids_3) - bos
          (no stale abort context from Turn 2)
        - Turn 3 completion_tokens == 8 (full output)"""
        process = self._launch_server(["--enable-streaming-session"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            tokenizer = self._get_tokenizer()

            requests.post(self.base_url + "/flush_cache")

            resp = self._open_session(streaming=True, capacity=50000)
            self.assertEqual(resp.status_code, 200)
            session_id = resp.json()

            try:
                # Turn 1: normal generate to create session slot
                ids_1 = tokenizer.encode("Tell me a very long story about a wizard.")
                resp_1 = self._generate(
                    {
                        "input_ids": ids_1,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                self.assertEqual(resp_1.status_code, 200, resp_1.text)
                data_1 = resp_1.json()
                turn_1_total = (
                    data_1["meta_info"]["prompt_tokens"]
                    + data_1["meta_info"]["completion_tokens"]
                )

                # Turn 2: long generate, abort mid-decode
                ids_2 = tokenizer.encode(" Continue the story in great detail.")

                result = [None]

                def do_generate():
                    r = self._generate(
                        {
                            "input_ids": ids_2,
                            "sampling_params": LONG_SAMPLING_PARAMS,
                            "session_params": {"id": session_id, "rid": None},
                        },
                        timeout=60,
                    )
                    result[0] = r

                t = threading.Thread(target=do_generate)
                t.start()
                time.sleep(0.5)
                abort_resp = self._abort_all()
                self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
                t.join(timeout=30)

                self.assertIsNotNone(result[0], "Turn 2 should have returned")
                data_2 = result[0].json()
                self.assertEqual(
                    data_2["meta_info"]["finish_reason"]["type"],
                    "abort",
                    "Turn 2 should be aborted, not finished normally",
                )
                # Turn 2 must inherit KV cache from Turn 1
                self.assertGreater(
                    data_2["meta_info"]["cached_tokens"],
                    0,
                    "Turn 2 should inherit KV cache from Turn 1",
                )

                # Turn 3: recovery — rolls back to turn 1 state
                ids_3 = tokenizer.encode(" What happens next?")
                for attempt in range(20):
                    resp_3 = self._generate(
                        {
                            "input_ids": ids_3,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8,
                                "no_stop_trim": True,
                                "skip_special_tokens": False,
                            },
                            "session_params": {"id": session_id, "rid": None},
                        },
                        timeout=30,
                    )
                    if resp_3.status_code == 200:
                        break
                    time.sleep(0.5)
                self.assertEqual(resp_3.status_code, 200, resp_3.text)
                data_3 = resp_3.json()
                bos = 1 if ids_3[0] == tokenizer.bos_token_id else 0
                expected_prompt = turn_1_total + len(ids_3) - bos
                self.assertEqual(
                    data_3["meta_info"]["prompt_tokens"],
                    expected_prompt,
                    "prompt_tokens must equal turn_1_total + append "
                    "(no stale abort context)",
                )
                self.assertEqual(
                    data_3["meta_info"]["completion_tokens"],
                    8,
                    "Turn 3 should generate full 8 tokens after recovery",
                )
            finally:
                close_resp = self._close_session(session_id)
                self.assertEqual(close_resp.status_code, 200)

            health = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(health.status_code, 200)
        finally:
            kill_process_tree(process.pid)

    # ------------------------------------------------------------------
    # TC-SS-05: First-turn mid-decode abort recovery
    #   Ref: StreamingSessionKitMixin.test_first_mid_abort_recovery
    #   Covers: abort on first turn (no slot yet), recovery works
    #
    #   Scenario: Abort the very first request mid-decode. No persistent
    #   slot exists yet — the framework creates an ephemeral slot for the
    #   request and must nuke it on abort. Turn 2 recovery must start
    #   from a completely clean slate with zero inherited context.
    #
    #   Turn 1 → start then abort (ephemeral slot created → nuked)
    #   Turn 2 → recovery (verify cached_tokens == 0, prompt_tokens == own
    #             input only, full 8-token output)
    # ------------------------------------------------------------------

    def test_first_mid_abort_recovery(self):
        """Abort the very first request mid-decode (no slot yet; ephemeral
        slot is created and nuked). Session must still be usable.

        Assertions:
        - Turn 1 finish_reason.type == "abort"
        - Turn 2 cached_tokens == 0 (no inherited context from nuked slot)
        - Turn 2 prompt_tokens == len(ids_2) (zero-history baseline)
        - Turn 2 completion_tokens == 8 (full output)"""
        process = self._launch_server(["--enable-streaming-session"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            tokenizer = self._get_tokenizer()

            requests.post(self.base_url + "/flush_cache")

            resp = self._open_session(streaming=True, capacity=50000)
            self.assertEqual(resp.status_code, 200)
            session_id = resp.json()

            try:
                # Turn 1: long generate, abort mid-decode (no slot yet)
                ids_1 = tokenizer.encode("Tell me a very long story about a wizard.")

                result = [None]

                def do_generate():
                    r = self._generate(
                        {
                            "input_ids": ids_1,
                            "sampling_params": LONG_SAMPLING_PARAMS,
                            "session_params": {"id": session_id, "rid": None},
                        },
                        timeout=60,
                    )
                    result[0] = r

                t = threading.Thread(target=do_generate)
                t.start()
                time.sleep(0.5)
                abort_resp = self._abort_all()
                self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
                t.join(timeout=30)

                self.assertIsNotNone(result[0], "Turn 1 should have returned")
                data_1 = result[0].json()
                self.assertEqual(
                    data_1["meta_info"]["finish_reason"]["type"],
                    "abort",
                    "Turn 1 should be aborted, not finished normally",
                )

                # Turn 2: recovery — no inherited context
                ids_2 = tokenizer.encode("Tell me a short joke.")
                for attempt in range(20):
                    resp_2 = self._generate(
                        {
                            "input_ids": ids_2,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8,
                                "no_stop_trim": True,
                                "skip_special_tokens": False,
                            },
                            "session_params": {"id": session_id, "rid": None},
                        },
                        timeout=30,
                    )
                    if resp_2.status_code == 200:
                        break
                    time.sleep(0.5)
                self.assertEqual(resp_2.status_code, 200, resp_2.text)
                data_2 = resp_2.json()
                self.assertEqual(
                    data_2["meta_info"]["cached_tokens"],
                    0,
                    "Turn 2 must have no cache hit (ephemeral slot was nuked)",
                )
                self.assertEqual(
                    data_2["meta_info"]["prompt_tokens"],
                    len(ids_2),
                    "prompt_tokens must equal turn 2 input only "
                    "(no inherited context)",
                )
                self.assertEqual(
                    data_2["meta_info"]["completion_tokens"],
                    8,
                    "Turn 2 should generate full 8 tokens after recovery",
                )
            finally:
                close_resp = self._close_session(session_id)
                self.assertEqual(close_resp.status_code, 200)

            health = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(health.status_code, 200)
        finally:
            kill_process_tree(process.pid)

    # ------------------------------------------------------------------
    # TC-SS-06: Pre-abort recovery
    #   Ref: StreamingSessionKitMixin.test_preabort_recovery
    #   Covers: pre-abort (rejected by create_req) preserves slot
    #
    #   Scenario: Turn 2 is rejected at prefill stage (offset=1 triggers
    #   create_req rejection before any decode). Unlike mid-abort, the
    #   request never enters the session — the persistent slot from
    #   Turn 1 must remain intact and reusable.
    #
    #   Turn 1 → complete (establish persistent slot)
    #   Turn 2 → pre-abort via offset=1 (create_req rejects, no decode)
    #   Turn 3 → recovery (verify cached_tokens > 0 inherited from Turn 1,
    #             prompt_tokens == Turn1 history + own input, full 8-token output)
    # ------------------------------------------------------------------

    def test_preabort_recovery(self):
        """Pre-abort (rejected by create_req via offset=1) preserves the
        session slot; next turn inherits correct KV from turn 1.

        Assertions:
        - Turn 1 completes normally → record turn_1_total
        - Turn 2 status in (200, 400) (rejected by create_req)
        - Turn 3 cached_tokens > 0 (slot intact, KV inherited from Turn 1)
        - Turn 3 prompt_tokens == turn_1_total + len(ids_3) - bos
          (no corruption from pre-aborted Turn 2)
        - Turn 3 completion_tokens == 8 (full output)"""
        process = self._launch_server(["--enable-streaming-session"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            tokenizer = self._get_tokenizer()

            requests.post(self.base_url + "/flush_cache")

            resp = self._open_session(streaming=True, capacity=50000)
            self.assertEqual(resp.status_code, 200)
            session_id = resp.json()

            try:
                # Turn 1: normal generate to create slot
                ids_1 = tokenizer.encode("Tell me a very long story about a wizard.")
                resp_1 = self._generate(
                    {
                        "input_ids": ids_1,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                self.assertEqual(resp_1.status_code, 200, resp_1.text)
                data_1 = resp_1.json()
                turn_1_total = (
                    data_1["meta_info"]["prompt_tokens"]
                    + data_1["meta_info"]["completion_tokens"]
                )

                # Turn 2: pre-aborted via unsupported offset parameter
                ids_2 = tokenizer.encode(" This should be rejected.")
                resp_2 = self._generate(
                    {
                        "input_ids": ids_2,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                        "session_params": {
                            "id": session_id,
                            "rid": None,
                            "offset": 1,
                        },
                    },
                    timeout=30,
                )
                self.assertIn(resp_2.status_code, (200, 400), resp_2.text)

                # Turn 3: normal append. Slot should be intact from turn 1.
                ids_3 = tokenizer.encode(" What happens next?")
                resp_3 = self._generate(
                    {
                        "input_ids": ids_3,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                self.assertEqual(resp_3.status_code, 200, resp_3.text)
                data_3 = resp_3.json()
                bos = 1 if ids_3[0] == tokenizer.bos_token_id else 0
                expected_prompt = turn_1_total + len(ids_3) - bos
                self.assertEqual(
                    data_3["meta_info"]["prompt_tokens"],
                    expected_prompt,
                    "prompt_tokens must equal turn_1_total + append "
                    "(slot preserved after pre-abort)",
                )
                # Turn 3 must inherit KV cache from Turn 1 (slot intact)
                self.assertGreater(
                    data_3["meta_info"]["cached_tokens"],
                    0,
                    "Turn 3 should inherit KV cache from Turn 1 "
                    "(slot must not be corrupted by pre-abort)",
                )
                self.assertEqual(
                    data_3["meta_info"]["completion_tokens"],
                    8,
                    "Turn 3 should generate full 8 tokens after recovery",
                )
            finally:
                close_resp = self._close_session(session_id)
                self.assertEqual(close_resp.status_code, 200)

            health = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(health.status_code, 200)
        finally:
            kill_process_tree(process.pid)

    # ==================================================================
    # Part D: Session lifecycle (P2)
    # ==================================================================

    # ------------------------------------------------------------------
    # TC-SS-07: Session reopen
    #   Ref: test_session_control.py test_session_control
    #   Covers: close → reopen → reuse same session_id
    #
    #   Scenario: Close a session and reopen with the same session_id.
    #   The reopened session must be a fresh instance with no KV cache
    #   inherited from the previous session — close must fully release
    #   all resources (KV pages, slot state).
    #
    #   Open → Generate → Close → Reopen(same_id) → Generate
    #   (verify reopen succeeds, cached_tokens == 0, full output)
    # ------------------------------------------------------------------

    def test_session_reopen(self):
        """Close a session and reopen with the same session_id; should
        start fresh with no cached KV.

        Assertions:
        - Reopen with same session_id succeeds (returns the id)
        - Reopened session cached_tokens == 0 (no leaked KV)
        - Reopened session completion_tokens == 12 (full output)"""
        process = self._launch_server(["--enable-streaming-session"])
        try:
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            tokenizer = self._get_tokenizer()

            requests.post(self.base_url + "/flush_cache")

            # Open, use, close
            resp = self._open_session(
                streaming=True, capacity=1000, session_id="reopen-test"
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), "reopen-test")

            ids = tokenizer.encode("Hello world.")
            resp = self._generate(
                {
                    "input_ids": ids,
                    "session_params": {"id": "reopen-test", "rid": None},
                    "sampling_params": SAMPLING_PARAMS,
                }
            )
            self.assertEqual(resp.status_code, 200)

            close_resp = self._close_session("reopen-test")
            self.assertEqual(close_resp.status_code, 200)

            # Reopen same id — should succeed (fresh session)
            resp2 = self._open_session(
                streaming=True, capacity=1000, session_id="reopen-test"
            )
            self.assertEqual(
                resp2.status_code, 200, "Reopen after close should succeed"
            )
            self.assertEqual(resp2.json(), "reopen-test")

            # Fresh session — no KV to inherit
            ids2 = tokenizer.encode("Tell me a joke.")
            resp3 = self._generate(
                {
                    "input_ids": ids2,
                    "session_params": {"id": "reopen-test", "rid": None},
                    "sampling_params": SAMPLING_PARAMS,
                }
            )
            self.assertEqual(resp3.status_code, 200)
            data_3 = resp3.json()
            self.assertEqual(
                data_3["meta_info"]["cached_tokens"],
                0,
                "Reopened session should have no cached KV",
            )
            self.assertEqual(
                data_3["meta_info"]["completion_tokens"],
                SAMPLING_PARAMS["max_new_tokens"],
                "Reopened session should generate full output",
            )

            close_resp = self._close_session("reopen-test")
            self.assertEqual(close_resp.status_code, 200)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
