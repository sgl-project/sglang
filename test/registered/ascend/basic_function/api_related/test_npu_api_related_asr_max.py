import base64
import json
import logging
import os
import threading
import time
import unittest
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import torchaudio
import websocket
from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_ASR_WEIGHTS_PATH, WAV_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)

parsed = urlparse(DEFAULT_URL_FOR_TEST)
host = parsed.hostname
port = parsed.port
WS_URL = f"ws://{host}:{port}/v1/realtime"
TARGET_SR = 16000
CHUNK_SEC = 0.5


class TestAsrMaxTranscription(CustomTestCase):
    """Testcase: The maximum number of concurrent real-time ASR WebSocket sessions handled and
    the maximum duration of PCM audio data that can be accumulated before a session closes.

    [Test Category] Functional
    [Test Target] api related on NPU
    --asr-max-concurrent-sessions;--asr-max-buffer-seconds;
    """

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["SGLANG_SET_CPU_AFFINITY"] = "1"
        env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        env["STREAMS_PER_DEVICE"] = "32"
        env["HCCL_BUFFSIZE"] = "1536"
        env["HCCL_OP_EXPANSION_MODE"] = "AIV"
        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "32"
        env["SGLANG_DEEPEP_BF16_DISPATCH"] = "1"
        env["ENABLE_ASCEND_MOE_NZ"] = "1"
        cls.model = QWEN3_0_6B_ASR_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                2,
                "--log-requests-level",
                2,
                "--mem-fraction-static",
                "0.85",
                "--chunked-prefill-size",
                -1,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--dtype",
                "bfloat16",
                "--disable-radix-cache",
                "--asr-max-concurrent-sessions",
                2,
                "--asr-max-buffer-seconds",
                5,
                "--served-model-name",
                "qwen3-asr",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def load_wav_to_pcm16(self, file_path, target_sr):
        # Read WAV file, convert to 16kHz PCM byte stream.
        data, orig_sr = sf.read(file_path, dtype="float32")
        # Dual-channel to single-channel conversion
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if orig_sr != target_sr:
            audio_t = torch.from_numpy(data).unsqueeze(0)
            audio_t = torchaudio.functional.resample(
                audio_t, orig_freq=orig_sr, new_freq=target_sr
            )
            data = audio_t.squeeze(0).numpy()

        data = np.clip(data, -1.0, 1.0)
        data_int16 = (data * 32767).astype(np.int16)
        return data_int16.tobytes()

    def test_max_buffer(self):
        # Load a local WAV file and convert it to a PCM16 byte stream.
        full_pcm = self.load_wav_to_pcm16(WAV_PATH, TARGET_SR)
        samples_per_chunk = int(CHUNK_SEC * TARGET_SR)
        bytes_per_sample = 2
        chunk_bytes_len = samples_per_chunk * bytes_per_sample

        ws = websocket.create_connection(WS_URL, timeout=10)
        error = None
        transcript_text = None
        try:
            # Handle the initial `session.created` event and add exception handling.
            try:
                init_raw = ws.recv()
                init_msg = json.loads(init_raw)
                logging.warning("Initial service message:", init_msg)
            except WebSocketTimeoutException:
                self.fail(
                    "Service unresponsive: Initial `session.created` event not received within 10 seconds of connecting to WebSocket."
                )

            # Update session configuration
            session_update_msg = json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": TARGET_SR},
                                "transcription": {"model": "qwen3-asr"},
                            }
                        },
                    },
                }
            )
            ws.send(session_update_msg)
            # wait session.updated
            ws.settimeout(10)
            session_ok = False
            for _ in range(20):
                try:
                    evt_raw = ws.recv()
                    evt = json.loads(evt_raw)
                    if evt["type"] == "session.updated":
                        session_ok = True
                        break
                    if evt["type"] == "error":
                        error = evt
                        break
                except (WebSocketTimeoutException, WebSocketConnectionClosedException):
                    continue
            self.assertTrue(
                session_ok,
                "Failed to wait for the session.updated configuration timeout.",
            )

            offset = 0
            total_sec = 0.0
            ws.settimeout(1.0)

            # Cyclically shard and transmit the complete audio.
            while offset < len(full_pcm):
                end = offset + chunk_bytes_len
                chunk = full_pcm[offset:end]
                offset = end
                send_msg = json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                ws.send(send_msg)

                total_sec += CHUNK_SEC
                logging.warning(f"Total duration of sent audio: {total_sec:.1f}s")

                # Catching Server-Side Errors
                try:
                    evt_raw = ws.recv()
                    evt = json.loads(evt_raw)
                    if evt["type"] == "error":
                        error = evt
                        logging.warning("Received a server-side error:", error)
                        self.assertIn(
                            "Accumulated audio exceeded", error["error"]["message"]
                        )
                        break
                except (WebSocketTimeoutException, WebSocketConnectionClosedException):
                    continue

            # No errors occurred; sending a commit triggers full inference.
            if not error:
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                ws.settimeout(30)
                max_wait_loop = 60
                finish_flag = False
                for _ in range(max_wait_loop):
                    try:
                        resp_raw = ws.recv()
                        resp = json.loads(resp_raw)
                        if (
                            resp["type"]
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            transcript_text = resp["transcript"]
                            logging.warning("Final version:", transcript_text)
                            finish_flag = True
                            break
                        if resp["type"] == "error":
                            error = resp
                            logging.warning(
                                "Service error during the transcription stage:", error
                            )
                            break
                    except (
                        WebSocketTimeoutException,
                        WebSocketConnectionClosedException,
                    ):
                        logging.warning("Timed out waiting for transcription event.")
                        continue
                self.assertTrue(
                    finish_flag,
                    f"Transcription completion event not received within 30 seconds.",
                )
                self.assertGreater(
                    len(transcript_text.strip()), 0, "Transcription result is empty."
                )

        finally:
            ws.close()

    def create_active_session(self, idx, result_list, lock):
        res = {"idx": idx, "ok": False, "msg": ""}
        try:
            ws = websocket.create_connection(WS_URL, timeout=3)

            # Receive the first message
            first_raw = ws.recv()
            first_msg = json.loads(first_raw)
            ws.close()

            if first_msg.get("type") == "error":
                err = first_msg.get("error", {})
                res["msg"] = (
                    f"Service rejected | code={err.get('code')} | {err.get('message')}"
                )

            elif first_msg.get("type") != "session.created":
                res["msg"] = (
                    f"Session creation event not received; actual type is: {first_msg.get('type')}"
                )

            else:
                time.sleep(2)
                res["ok"] = True
                res["msg"] = "Active session successfully created."
        except Exception as e:
            res["msg"] = f"Connection error: {str(e)}"

        # Thread-safe writing of results
        with lock:
            result_list.append(res)

    def test_concurrent(self):
        # Concurrently create 3 sessions to verify the --asr-max-concurrent-sessions=2 configuration.
        thread_list = []
        result_list = []
        lock = threading.Lock()

        for i in range(1, 4):
            t = threading.Thread(
                target=self.create_active_session, args=(i, result_list, lock)
            )
            thread_list.append(t)
            t.start()

        for t in thread_list:
            t.join()
        success_count = 0
        for item in result_list:
            idx = item["idx"]
            ok = item["ok"]
            msg = item["msg"]
            status = "success" if ok else "fail"
            logging.warning(f"session{idx}: {status} {msg}")
            if ok:
                success_count += 1
        self.assertEqual(
            success_count,
            2,
            f"The restrictions did not take effect as expected. In reality, it is {success_count}",
        )


if __name__ == "__main__":
    unittest.main()
