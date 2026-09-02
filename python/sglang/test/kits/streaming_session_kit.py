"""Streaming-session test method mixins.

Pair these with `StreamingSessionServerBase` (from sglang.test.server_fixtures.streaming_session_fixture)
to assemble a concrete test class. Per the sglang fixture/kit split:
the fixture only launches the server; the kit owns the `test_*` methods.

- `StreamingSessionKitMixin`: KV-inheritance + chunked-prefill + abort-recovery
  + concurrent-logprob/stress test methods.
- `AbortLeakReproKitMixin`: single test method for abort-heavy chunked-prefill leak repro.
"""

import asyncio
import json
import time

import aiohttp
import requests

from sglang.test.server_fixtures.streaming_session_fixture import (
    _abort_repro_run_all,
    _concurrent_logprob_run,
    _make_token_sized_ids,
    _stress_run_all,
)


class StreamingSessionKitMixin:
    """Streaming-session KV-inheritance + retract/abort-recovery suite."""

    # Allowed inherited-cache offsets vs the previous turn's total. Non-overlap
    # spec decode can be off by 1: the bonus token's KV is only computed by the
    # next forward, which sync skips at finish (overlap drains it, so it's 0).
    kv_inherit_offsets = (0,)

    def test_kv_cache_inheritance(self, gen_len=12):
        """Each turn's cached_tokens must equal previous turn's prompt+completion
        (modulo kv_inherit_offsets)."""
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
        ]
        chunks_ids = [self.tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == self.tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # === Part 1: streaming session — check KV inheritance ===
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "streaming": True},
        ).json()
        rid = None

        prev_kv_len = 0
        for turn_idx, chunk_ids in enumerate(chunks_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {"id": session_id, "rid": rid},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": gen_len,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            rid = response["meta_info"]["id"]
            cached = response["meta_info"]["cached_tokens"]
            prompt_tokens = response["meta_info"]["prompt_tokens"]
            completion_tokens = response["meta_info"]["completion_tokens"]

            if turn_idx == 0:
                # Turn 1: cache flushed, no hit.
                self.assertEqual(cached, 0, "Turn 1: clean start, no cache hit")
            else:
                # Turns 2+: cached_tokens reflects KV inherited from previous turn
                # (via inherit_kv_states, not radix tree matching).
                allowed = {prev_kv_len + off for off in self.kv_inherit_offsets}
                self.assertIn(
                    cached,
                    allowed,
                    f"Turn {turn_idx + 1}: inherited {cached} not in {sorted(allowed)}",
                )
            prev_kv_len = prompt_tokens + completion_tokens

        # Close the session.
        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

    def test_leak_logprob_concurrent(self) -> None:
        """Concurrent multi-session × 3 logprob modes (output / input / none),
        watch for KV leak."""
        requests.post(self.base_url + "/flush_cache")
        # Output logprob
        asyncio.run(
            _concurrent_logprob_run(self.base_url, self.tokenizer, return_logprob=True)
        )
        # Input logprob (logprob_start_len=0)
        asyncio.run(
            _concurrent_logprob_run(
                self.base_url,
                self.tokenizer,
                return_logprob=True,
                logprob_start_len=0,
            )
        )
        # No logprob
        asyncio.run(_concurrent_logprob_run(self.base_url, self.tokenizer))
        time.sleep(3)
        assert (
            requests.get(self.base_url + "/health").status_code == 200
        ), "Server unhealthy after concurrent logprob sessions."

    def test_stress_concurrent_sessions(self) -> None:
        """High concurrency streaming + non-streaming with retract pressure;
        scheduler must roll back streaming KV without leaking."""
        requests.post(self.base_url + "/flush_cache")
        asyncio.run(_stress_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-stress cleanup {i}.")
            requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
            )

        time.sleep(5)
        health = requests.get(self.base_url + "/health")
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after concurrent stress test — "
            "likely a token leak from retract/mixed-chunk + streaming session.",
        )

    def test_nth_mid_abort_recovery(self) -> None:
        """Abort an Nth-turn request mid-decode; session rolls back to last
        successful turn."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot.
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
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

            # Turn 2: long generate, then abort mid-decode.
            ids_2 = self.tokenizer.encode(" Continue the story in great detail.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 100000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 2 should have returned")
            data_2 = result[0].json()
            self.assertEqual(
                data_2["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 2 should be aborted, not finished normally",
            )

            # Turn 3: recovery. Rolls back to turn 1.
            ids_3 = self.tokenizer.encode(" What happens next?")
            for attempt in range(20):
                resp_3 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_3,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                if resp_3.status_code == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            # prompt_tokens = turn_1_total + append (BOS stripped).
            bos = 1 if ids_3[0] == self.tokenizer.bos_token_id else 0
            expected_prompt_3 = turn_1_total + len(ids_3) - bos
            self.assertEqual(
                data_3["meta_info"]["prompt_tokens"],
                expected_prompt_3,
                "prompt_tokens must equal turn_1_total + append (no stale abort context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_first_mid_abort_recovery(self) -> None:
        """Abort the very first request mid-decode (no slot yet; ephemeral
        slot is created and nuked). Session must still be usable."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_1,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 100000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 1 should have returned")
            data_1 = result[0].json()
            self.assertEqual(
                data_1["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 1 should be aborted, not finished normally",
            )

            # Turn 2: recovery. No inherited context (req_nodes empty).
            ids_2 = self.tokenizer.encode("Tell me a short joke.")
            for attempt in range(20):
                resp_2 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
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
                data_2["meta_info"]["prompt_tokens"],
                len(ids_2),
                "prompt_tokens must equal turn 2 input only (no inherited context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_preabort_recovery(self) -> None:
        """Pre-abort (rejected by create_req) preserves the slot; next turn
        inherits correctly."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot.
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
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

            # Turn 2: pre-aborted via unsupported offset parameter.
            ids_2 = self.tokenizer.encode(" This should be rejected.")
            resp_2 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_2,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
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
            ids_3 = self.tokenizer.encode(" What happens next?")
            resp_3 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_3,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            bos = 1 if ids_3[0] == self.tokenizer.bos_token_id else 0
            expected_prompt_3 = turn_1_total + len(ids_3) - bos
            self.assertEqual(
                data_3["meta_info"]["prompt_tokens"],
                expected_prompt_3,
                "prompt_tokens must equal turn_1_total + append (slot preserved)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)


class AbortLeakReproKitMixin:
    """Abort-heavy chunked-prefill leak repro."""

    def test_abort_heavy_chunked_prefill_does_not_leak(self) -> None:
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_abort_repro_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-session cleanup request {i}.")
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)

        time.sleep(5)
        self.assertIsNone(
            self.process.poll(),
            "Server crashed during abort-heavy streaming session repro.",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after abort-heavy streaming session cleanup.",
        )


async def _client_disconnect_turn(
    http: aiohttp.ClientSession,
    base_url: str,
    input_ids: list,
    session_id: str,
    gen_len: int,
    disconnect_after_chunks: int = 0,
) -> dict:
    """Stream one session turn; optionally drop the connection mid-stream.

    Returns the last received meta_info."""
    payload = {
        "input_ids": input_ids,
        "session_params": {"id": session_id, "rid": None},
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": gen_len,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
        "stream": True,
    }
    last_meta: dict = {}
    chunks = 0
    async with http.post(base_url + "/generate", json=payload) as resp:
        assert resp.status == 200, await resp.text()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                continue
            item = json.loads(data)
            if item.get("meta_info") is not None:
                last_meta = item["meta_info"]
            chunks += 1
            if disconnect_after_chunks and chunks >= disconnect_after_chunks:
                # Exiting the context with the body unread closes the
                # connection: a mid-turn client disconnect.
                return last_meta
    return last_meta


async def _client_disconnect_run(base_url: str, tokenizer) -> None:
    """Repro for #36475: disconnect mid-turn, immediately send the next turn.

    The next turn must be admitted against the last completed turn (rollback
    semantics, like an explicit /abort_request), not silently corrupted."""
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        async with http.post(
            base_url + "/open_session",
            json={"capacity_of_str_len": 10_000_000, "streaming": True},
        ) as resp:
            assert resp.status == 200, await resp.text()
            session_id = await resp.json()

        turn_ids = [
            _make_token_sized_ids(
                tokenizer, prefix=f"[disconnect repro turn{i}]", min_tokens=n
            )
            for i, n in enumerate((512, 64, 64))
        ]
        # The session controller strips a leading BOS from appended turns.
        bos = getattr(tokenizer, "bos_token_id", None)
        for ids in turn_ids:
            if bos is not None and ids and ids[0] == bos:
                ids.pop(0)
        turn0_ids, turn1_ids, turn2_ids = turn_ids
        gen_len = 48

        turn0_meta = await _client_disconnect_turn(
            http, base_url, turn0_ids, session_id, gen_len
        )
        turn0_total = turn0_meta["prompt_tokens"] + turn0_meta["completion_tokens"]

        # Turn 1: disconnect after the first streamed chunk.
        turn1_meta = await _client_disconnect_turn(
            http, base_url, turn1_ids, session_id, gen_len, disconnect_after_chunks=1
        )
        assert turn1_meta["prompt_tokens"] == turn0_total + len(
            turn1_ids
        ), f"turn1 lost inherited context: {turn1_meta}"

        # Turn 2 must run against the rolled-back session (last completed
        # turn = turn 0), not be pre-aborted with a one-token context.
        turn2_meta = await _client_disconnect_turn(
            http, base_url, turn2_ids, session_id, gen_len
        )
        expected = turn0_total + len(turn2_ids)
        assert turn2_meta.get("prompt_tokens") == expected, (
            "turn2 after mid-turn client disconnect lost session context: "
            f"expected prompt_tokens={expected}, got {turn2_meta}"
        )
        finish_type = (turn2_meta.get("finish_reason") or {}).get("type")
        assert finish_type != "abort", (
            "turn2 after mid-turn client disconnect was pre-aborted: " f"{turn2_meta}"
        )


class ClientDisconnectKitMixin:
    """Client disconnects mid-turn (issue #36475)."""

    def test_client_disconnect_next_turn_rolls_back(self) -> None:
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_client_disconnect_run(self.base_url, self.tokenizer))

        time.sleep(5)
        self.assertIsNone(
            self.process.poll(),
            "Server crashed during client-disconnect streaming session repro.",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after client-disconnect streaming session repro.",
        )
