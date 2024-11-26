"""
Usage:
python3 -m unittest test_session_control.TestSessionControl.test_session_control
python3 -m unittest test_session_control.TestSessionControl.test_session_control_vlm
"""

import unittest

import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestSessionControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid, include_self=True)

    def test_session_control(self):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "A brief history about that city is",
            "To plan a travel, the budget is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]

        # 1. using session control
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        rid = None

        first_rid = None
        outputs_from_session = []
        for i, chunk_ids in enumerate(chunks_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session": [session_id, rid],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            16 if i > 0 else 0
                        ),  # prefill only for the first chunk
                    },
                },
            ).json()
            rid = response["meta_info"]["id"]
            if i == 0:
                first_rid = rid
            if i > 0:
                outputs_from_session.append(response["text"])

        # backtrack to the first request and regenerate
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session": [session_id, first_rid],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
        ).json()
        outputs_from_session.append(response["text"])

        # query with a non-existing rid (the last one should be disappeared becuase of backtrack), should see abort
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session": [session_id, rid],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
        ).json()
        assert response["meta_info"]["finish_reason"]["type"] == "abort"

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        assert ret.status_code == 200

        # send a request to a closed session, should see abort
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session": [session_id, first_rid],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
        ).json()
        assert response["meta_info"]["finish_reason"]["type"] == "abort"

        # 2. not use session control
        input_ids_first_req = None
        input_ids = []
        outputs_normal = []
        for i, chunk_ids in enumerate(chunks_ids):
            input_ids += chunk_ids
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            16 if i > 0 else 0
                        ),  # prefill only for the first chunk
                    },
                },
            ).json()
            if i > 0:
                input_ids += tokenizer.encode(response["text"])[
                    1:
                ]  # drop the bos token
                outputs_normal.append(response["text"])
            if i == 0:
                input_ids_first_req = input_ids.copy()

        input_ids_first_req += chunks_ids[-1]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids_first_req,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
        ).json()
        outputs_normal.append(response["text"])

        print("outputs from chunked queries with session control:")
        print(outputs_from_session)
        print("outputs from normal queries:")
        print(outputs_normal)
        assert outputs_from_session == outputs_normal


if __name__ == "__main__":
    unittest.main()
