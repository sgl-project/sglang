"""
Usage:
python3 -m unittest test_session_control.TestSessionControl.test_session_control
python3 -m unittest test_session_control.TestSessionControl.test_session_control_with_branching
python3 -m unittest test_session_control.TestSessionControl.test_session_control_backtrack_with_abort
python3 -m unittest test_session_control.TestSessionControlVision.test_session_control
"""

import asyncio
import json
import unittest

import aiohttp
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


class TestSessionControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "flashinfer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_session_control(self, gen_len=12):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
            "A brief history about that city is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # 1. using session control
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        rid = None

        # open an existing session, should get session_id as None
        ret = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "session_id": session_id},
        )
        self.assertNotEqual(ret.status_code, 200)

        first_rid = None
        outputs_from_session = []
        logprobs_from_session = []
        cur_logprob_start_len = 0
        for i, chunk_ids in enumerate(chunks_ids):
            max_new_tokens = gen_len if i > 0 else 1  # prefill only for the first chunk
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {
                        "id": session_id,
                        "rid": rid,
                        "offset": -1,
                        "replace": True,
                    },
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                    "return_logprob": True,
                    "logprob_start_len": cur_logprob_start_len - 1,
                },
            ).json()
            rid = response["meta_info"]["id"]
            if i == 0:
                first_rid = rid
            if i > 0:
                outputs_from_session.append(response["text"])
                logprobs_from_session.extend(
                    [
                        round(sublist[0], 2)
                        for sublist in response["meta_info"]["output_token_logprobs"]
                    ]
                )
            cur_logprob_start_len += len(chunk_ids) + max_new_tokens

        # query with a logprob_start_len longer than the request, should see error
        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunk_ids,
                "session_params": {
                    "id": session_id,
                    "rid": rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": cur_logprob_start_len + len(chunk_ids),
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        # backtrack to the first request and regenerate
        cur_logprob_start_len = 0
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": cur_logprob_start_len,
            },
        ).json()
        outputs_from_session.append(response["text"])
        logprobs_from_session.extend(
            [
                round(sublist[0], 2)
                for sublist in response["meta_info"]["output_token_logprobs"]
            ]
        )

        # query with a non-existing rid (the last one should be disappeared because of backtrack), should see abort
        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

        # send a request to a closed session, should see abort
        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        # 2. not use session control
        requests.post(self.base_url + "/flush_cache")

        input_ids_first_req = None
        input_ids = []
        outputs_normal = []
        logprobs_normal = []
        for i, chunk_ids in enumerate(chunks_ids):
            input_ids += chunk_ids
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            gen_len if i > 0 else 1
                        ),  # prefill only for the first chunk
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                    "return_logprob": True,
                },
            ).json()
            if i > 0:
                output_ids = tokenizer.encode(response["text"])
                if output_ids[0] == tokenizer.bos_token_id:
                    output_ids = output_ids[1:]
                input_ids += output_ids[:-1]
                outputs_normal.append(response["text"])
                logprobs_normal.extend(
                    [
                        round(sublist[0], 2)
                        for sublist in response["meta_info"]["output_token_logprobs"]
                    ]
                )
            if i == 0:
                input_ids_first_req = input_ids.copy()

        input_ids_first_req += chunks_ids[-1]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids_first_req,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
            },
        ).json()
        outputs_normal.append(response["text"])
        logprobs_normal.extend(
            [
                round(sublist[0], 2)
                for sublist in response["meta_info"]["output_token_logprobs"]
            ]
        )

        print("outputs from chunked queries with session control:")
        print(outputs_from_session)
        print("outputs from normal queries:")
        print(outputs_normal)
        self.assertEqual(outputs_from_session, outputs_normal)
        print("logprobs from chunked queries with session control:")
        print(logprobs_from_session)
        print("logprobs from normal queries:")
        print(logprobs_normal)
        assert len(logprobs_from_session) == len(
            logprobs_normal
        ), "logprobs must have equal length"
        for a, b in zip(logprobs_from_session, logprobs_normal):
            assert abs(a - b) <= 0.15, f"logprobs {a} and {b} differ by more than 0.15"

    async def async_generate(self, payload):
        url = self.base_url + "/generate"
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload) as response:
                assert response.status == 200
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if chunk == "[DONE]":
                        yield "", None, ""
                    else:
                        data = json.loads(chunk)
                        finish_reason = (
                            data["meta_info"]["finish_reason"]["type"]
                            if data["meta_info"]["finish_reason"]
                            else ""
                        )
                        yield data["text"], data["meta_info"]["id"], finish_reason

    async def run_session_control_backtrack_with_abort(self, replace):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # 1. using session control
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        rid = None

        payload = {
            "input_ids": chunks_ids[0],
            "session_params": {
                "id": session_id,
                "rid": rid,
                "offset": -1,
                "replace": True,
            },
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
                "no_stop_trim": True,
                "skip_special_tokens": False,
                "ignore_eos": True,
            },
            "stream": True,
        }
        gen_so_far = ""
        finish_reason = ""
        second_output = ""
        async for chunk, rid, finish_reason_chunk in self.async_generate(payload):
            gen_so_far += chunk
            if finish_reason == "":
                finish_reason = finish_reason_chunk
            if len(gen_so_far) > 50 and second_output == "":
                payload2 = {
                    "input_ids": chunks_ids[1],
                    "session_params": {
                        "id": session_id,
                        "rid": rid,
                        "offset": 50,
                        "replace": replace,
                    },
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                    "stream": False,
                    "stream_output": True,
                }
                response = requests.post(
                    url=self.base_url + "/generate", json=payload2
                ).json()
                second_output = response["text"]
        if replace:
            assert finish_reason == "abort"
        print("first request output:")
        print(gen_so_far)
        print("second request output:")
        print(second_output)

        # close the session
        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        assert ret.status_code == 200

        if not replace:
            assert response["meta_info"]["finish_reason"]["type"] == "abort"
        else:
            # 2. not using session control
            requests.post(self.base_url + "/flush_cache")
            output_ids = tokenizer.encode(gen_so_far)
            if output_ids[0] == tokenizer.bos_token_id:
                output_ids = output_ids[1:]
            input_ids = chunks_ids[0] + output_ids
            input_ids = input_ids[:50] + chunks_ids[1]
            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "stream": False,
                "stream_output": True,
            }
            response = requests.post(
                url=self.base_url + "/generate", json=payload
            ).json()
            output_no_session = response["text"]
            print("second request output without session:")
            print(output_no_session)
            assert (
                second_output == output_no_session
            ), f"second_output: {second_output}, output_no_session: {output_no_session}"

    @unittest.skip("broken")
    def test_session_control_backtrack_with_abort(self):
        asyncio.run(self.run_session_control_backtrack_with_abort(replace=True))
        asyncio.run(self.run_session_control_backtrack_with_abort(replace=False))

    def run_session_control_with_branching(
        self, root_prompt, chunks_per_step, gen_len=16
    ):
        for x in chunks_per_step:
            assert len(x) == len(chunks_per_step[0])

        # 1. using session control
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()

        outputs_from_session = []
        # send the root prompt
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": root_prompt,
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "offset": 0,
                    "replace": False,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        rid_per_branch = [response["meta_info"]["id"]] * len(chunks_per_step[0])
        outputs_from_session.append(response["text"])

        # send the prompts in branches
        for chunks_for_branches in chunks_per_step:
            for j, chunk in enumerate(chunks_for_branches):
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": chunk,
                        "session_params": {
                            "id": session_id,
                            "rid": rid_per_branch[j],
                            "offset": 0,
                            "replace": False,
                        },
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": gen_len,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                    },
                ).json()
                rid = response["meta_info"]["id"]
                rid_per_branch[j] = rid
                outputs_from_session.append(response["text"])

        # close the session
        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        assert ret.status_code == 200

        # 2. not use session control
        requests.post(self.base_url + "/flush_cache")

        outputs_normal = []
        input_texts = [root_prompt] * len(chunks_per_step[0])
        # send the root prompt
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": root_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        outputs_normal.append(response["text"])
        input_texts = [x + response["text"] for x in input_texts]

        # send the prompts in branches
        for chunks_for_branches in chunks_per_step:
            for j, chunk in enumerate(chunks_for_branches):
                input_texts[j] += chunk
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": input_texts[j],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": gen_len,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                    },
                ).json()
                outputs_normal.append(response["text"])
                input_texts[j] += response["text"]

        print("====== outputs from chunked queries with session control: =======")
        print(outputs_from_session)
        print("====== outputs from normal queries: =======")
        print(outputs_normal)
        assert (
            outputs_from_session == outputs_normal
        ), f"outputs_from_session: {outputs_from_session}, outputs_normal: {outputs_normal}"

    def test_session_control_with_branching(self):
        root_prompt = "First, let me explain in one sentence about AI"
        chunks_per_step = [
            [
                "Then, briefly, the positive side of AI is",
                "But, briefly, AI could be harmful to human",
            ],
            ["For example", "For example"],
        ]
        self.run_session_control_with_branching(
            root_prompt=root_prompt, chunks_per_step=chunks_per_step, gen_len=8
        )

        root_prompt = "I have three apples."
        chunks_per_step = [
            ["I then give one apple to my friend", "My friend give me another apple."],
            ["I still have", "I now have"],
        ]
        self.run_session_control_with_branching(
            root_prompt=root_prompt, chunks_per_step=chunks_per_step, gen_len=8
        )


@unittest.skip("broken")
class TestSessionControlVision(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmms-lab/llava-onevision-qwen2-7b-ov"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # other_args={"--disable-radix"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_session_control(self):
        text_chunks = [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>user\n<image>\nDescribe this image in a very short sentence.<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\n<image>\nIs this image same with one of the previous images?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\n<image>\nIs this image same with one of the previous images?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nDescribe this image in a very short sentence.<|im_end|>\nassistant:",
        ]
        image_chunks = [
            "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png",
        ]

        self.assertEqual(
            len(text_chunks), len(image_chunks) + 2
        )  # the first and the last prompt does not contain images
        tokenizer = get_tokenizer(self.model)
        text_input_ids = [tokenizer.encode(x) for x in text_chunks]
        for i in range(1, len(text_input_ids)):
            if text_input_ids[i][0] == tokenizer.bos_token_id:
                text_input_ids[i] = text_input_ids[i][1:]
        gen_len = 32

        # 1. using session control
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        rid = None

        # open an existing session, should get session_id as None
        ret = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "session_id": session_id},
        )
        self.assertNotEqual(ret.status_code, 200)

        first_rid = None
        outputs_from_session = []
        for i in range(len(text_input_ids[:-1])):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": text_input_ids[i],
                    "image_data": image_chunks[i - 1] if i > 0 else None,
                    "modalities": ["multi-images"],
                    "session_params": {
                        "id": session_id,
                        "rid": rid,
                        "offset": 0,
                        "replace": True,
                    },
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            gen_len if i > 0 else 0
                        ),  # prefill only for the first chunk
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
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
                "input_ids": text_input_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": 0,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        outputs_from_session.append(response["text"])

        # query with a non-existing rid (the last one should be disappeared because of backtrack), should see abort
        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": text_input_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": rid,
                    "offset": 0,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

        # send a request to a closed session, should see abort
        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": text_input_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": 0,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        # 2. not use session control
        requests.post(self.base_url + "/flush_cache")

        input_ids_first_req = None
        input_ids = []
        outputs_normal = []
        for i in range(len(text_input_ids[:-1])):
            input_ids += text_input_ids[i]
            image_data = image_chunks[:i] if i > 0 else None
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "image_data": image_data,
                    "modalities": ["multi-images"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            gen_len if i > 0 else 0
                        ),  # prefill only for the first chunk
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            if i > 0:
                output_ids = tokenizer.encode(response["text"])
                if output_ids[0] == tokenizer.bos_token_id:
                    output_ids = output_ids[1:]
                input_ids += output_ids
                outputs_normal.append(response["text"])
            if i == 0:
                input_ids_first_req = input_ids.copy()

        input_ids_first_req += text_input_ids[-1]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids_first_req,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        outputs_normal.append(response["text"])

        print("outputs from chunked queries with session control:")
        print(outputs_from_session)
        print("outputs from normal queries:")
        print(outputs_normal)
        assert (
            outputs_from_session == outputs_normal
        ), f"outputs_from_session: {outputs_from_session}, outputs_normal: {outputs_normal}"


if __name__ == "__main__":
    unittest.main()
