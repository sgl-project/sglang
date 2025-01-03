"""
python3 -m unittest test_openai_server.TestOpenAIServer.test_batch
python3 -m unittest test_openai_server.TestOpenAIServer.test_completion

"""

import json
import re
import time
import unittest

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestOpenAIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_completion(
        self, echo, logprobs, use_list_input, parallel_sample_num, token_input
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
            num_prompt_tokens *= 2
        else:
            prompt_arg = prompt_input
            num_choices = 1

        response = client.completions.create(
            model=self.model,
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
            n=parallel_sample_num,
        )

        assert len(response.choices) == num_choices * parallel_sample_num

        if echo:
            text = response.choices[0].text
            assert text.startswith(prompt)

        if logprobs:
            assert response.choices[0].logprobs
            assert isinstance(response.choices[0].logprobs.tokens[0], str)
            assert isinstance(response.choices[0].logprobs.top_logprobs[1], dict)
            ret_num_top_logprobs = len(response.choices[0].logprobs.top_logprobs[1])

            # FIXME: Sometimes, some top_logprobs are missing in the return value. The reason is that some output id maps to the same output token and duplicate in the map
            # assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"
            assert ret_num_top_logprobs > 0

            # when echo=True and request.logprobs>0, logprob_start_len is 0, so the first token's logprob would be None.
            if not echo:
                assert response.choices[0].logprobs.token_logprobs[0]

        assert response.id
        assert response.created
        assert (
            response.usage.prompt_tokens == num_prompt_tokens
        ), f"{response.usage.prompt_tokens} vs {num_prompt_tokens}"
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_completion_stream(
        self, echo, logprobs, use_list_input, parallel_sample_num, token_input
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
            num_prompt_tokens *= 2
        else:
            prompt_arg = prompt_input
            num_choices = 1

        generator = client.completions.create(
            model=self.model,
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0
                assert usage.completion_tokens > 0
                assert usage.total_tokens > 0
                continue

            index = response.choices[0].index
            is_first = is_firsts.get(index, True)

            if logprobs:
                assert response.choices[0].logprobs
                assert isinstance(response.choices[0].logprobs.tokens[0], str)
                if not (is_first and echo):
                    assert isinstance(
                        response.choices[0].logprobs.top_logprobs[0], dict
                    )
                    ret_num_top_logprobs = len(
                        response.choices[0].logprobs.top_logprobs[0]
                    )
                    # FIXME: Sometimes, some top_logprobs are missing in the return value. The reason is that some output id maps to the same output token and duplicate in the map
                    # assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"
                    assert ret_num_top_logprobs > 0

            if is_first:
                if echo:
                    assert response.choices[0].text.startswith(
                        prompt
                    ), f"{response.choices[0].text} and all args {echo} {logprobs} {token_input} {is_first}"
                is_firsts[index] = False
            assert response.id
            assert response.created

        for index in [i for i in range(parallel_sample_num * num_choices)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

    def run_chat_completion(self, logprobs, parallel_sample_num):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            n=parallel_sample_num,
        )

        if logprobs:
            assert isinstance(
                response.choices[0].logprobs.content[0].top_logprobs[0].token, str
            )

            ret_num_top_logprobs = len(
                response.choices[0].logprobs.content[0].top_logprobs
            )
            assert (
                ret_num_top_logprobs == logprobs
            ), f"{ret_num_top_logprobs} vs {logprobs}"

        assert len(response.choices) == parallel_sample_num
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_chat_completion_stream(self, logprobs, parallel_sample_num=1):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        generator = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0
                assert usage.completion_tokens > 0
                assert usage.total_tokens > 0
                continue

            index = response.choices[0].index
            data = response.choices[0].delta

            if is_firsts.get(index, True):
                assert data.role == "assistant"
                is_firsts[index] = False
                continue

            if logprobs:
                assert response.choices[0].logprobs
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs[0].token, str
                )
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs, list
                )
                ret_num_top_logprobs = len(
                    response.choices[0].logprobs.content[0].top_logprobs
                )
                assert (
                    ret_num_top_logprobs == logprobs
                ), f"{ret_num_top_logprobs} vs {logprobs}"

            assert isinstance(data.content, str)
            assert response.id
            assert response.created

        for index in [i for i in range(parallel_sample_num)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

    def _create_batch(self, mode, client):
        if mode == "completion":
            input_file_path = "complete_input.jsonl"
            # write content to input file
            content = [
                {
                    "custom_id": "request-1",
                    "method": "POST",
                    "url": "/v1/completions",
                    "body": {
                        "model": "gpt-3.5-turbo-instruct",
                        "prompt": "List 3 names of famous soccer player: ",
                        "max_tokens": 20,
                    },
                },
                {
                    "custom_id": "request-2",
                    "method": "POST",
                    "url": "/v1/completions",
                    "body": {
                        "model": "gpt-3.5-turbo-instruct",
                        "prompt": "List 6 names of famous basketball player:  ",
                        "max_tokens": 40,
                    },
                },
                {
                    "custom_id": "request-3",
                    "method": "POST",
                    "url": "/v1/completions",
                    "body": {
                        "model": "gpt-3.5-turbo-instruct",
                        "prompt": "List 6 names of famous tenniss player:  ",
                        "max_tokens": 40,
                    },
                },
            ]

        else:
            input_file_path = "chat_input.jsonl"
            content = [
                {
                    "custom_id": "request-1",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-3.5-turbo-0125",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {
                                "role": "user",
                                "content": "Hello! List 3 NBA players and tell a story",
                            },
                        ],
                        "max_tokens": 30,
                    },
                },
                {
                    "custom_id": "request-2",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-3.5-turbo-0125",
                        "messages": [
                            {"role": "system", "content": "You are an assistant. "},
                            {
                                "role": "user",
                                "content": "Hello! List three capital and tell a story",
                            },
                        ],
                        "max_tokens": 50,
                    },
                },
            ]

        with open(input_file_path, "w") as file:
            for line in content:
                file.write(json.dumps(line) + "\n")

        with open(input_file_path, "rb") as file:
            uploaded_file = client.files.create(file=file, purpose="batch")
        if mode == "completion":
            endpoint = "/v1/completions"
        elif mode == "chat":
            endpoint = "/v1/chat/completions"
        completion_window = "24h"
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=endpoint,
            completion_window=completion_window,
        )

        return batch_job, content, uploaded_file

    def run_batch(self, mode):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        batch_job, content, uploaded_file = self._create_batch(mode=mode, client=client)

        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)
            print(
                f"Batch job status: {batch_job.status}...trying again in 3 seconds..."
            )
            batch_job = client.batches.retrieve(batch_job.id)
        assert (
            batch_job.status == "completed"
        ), f"Batch job status is not completed: {batch_job.status}"
        assert batch_job.request_counts.completed == len(content)
        assert batch_job.request_counts.failed == 0
        assert batch_job.request_counts.total == len(content)

        result_file_id = batch_job.output_file_id
        file_response = client.files.content(result_file_id)
        result_content = file_response.read().decode("utf-8")  # Decode bytes to string
        results = [
            json.loads(line)
            for line in result_content.split("\n")
            if line.strip() != ""
        ]
        assert len(results) == len(content)
        for delete_fid in [uploaded_file.id, result_file_id]:
            del_pesponse = client.files.delete(delete_fid)
            assert del_pesponse.deleted

    def run_cancel_batch(self, mode):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        batch_job, _, uploaded_file = self._create_batch(mode=mode, client=client)

        assert batch_job.status not in ["cancelling", "cancelled"]

        batch_job = client.batches.cancel(batch_id=batch_job.id)
        assert batch_job.status == "cancelling"

        while batch_job.status not in ["failed", "cancelled"]:
            batch_job = client.batches.retrieve(batch_job.id)
            print(
                f"Batch job status: {batch_job.status}...trying again in 3 seconds..."
            )
            time.sleep(3)

        assert batch_job.status == "cancelled"
        del_response = client.files.delete(uploaded_file.id)
        assert del_response.deleted

    def test_completion(self):
        for echo in [False, True]:
            for logprobs in [None, 5]:
                for use_list_input in [True, False]:
                    for parallel_sample_num in [1, 2]:
                        for token_input in [False, True]:
                            self.run_completion(
                                echo,
                                logprobs,
                                use_list_input,
                                parallel_sample_num,
                                token_input,
                            )

    def test_completion_stream(self):
        # parallel sampling adn list input are not supported in streaming mode
        for echo in [False, True]:
            for logprobs in [None, 5]:
                for use_list_input in [True, False]:
                    for parallel_sample_num in [1, 2]:
                        for token_input in [False, True]:
                            self.run_completion_stream(
                                echo,
                                logprobs,
                                use_list_input,
                                parallel_sample_num,
                                token_input,
                            )

    def test_chat_completion(self):
        for logprobs in [None, 5]:
            for parallel_sample_num in [1, 2]:
                self.run_chat_completion(logprobs, parallel_sample_num)

    def test_chat_completion_stream(self):
        for logprobs in [None, 5]:
            for parallel_sample_num in [1, 2]:
                self.run_chat_completion_stream(logprobs, parallel_sample_num)

    def test_batch(self):
        for mode in ["completion", "chat"]:
            self.run_batch(mode)

    def test_cancel_batch(self):
        for mode in ["completion", "chat"]:
            self.run_cancel_batch(mode)

    def test_regex(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        regex = (
            r"""\{\n"""
            + r"""   "name": "[\w]+",\n"""
            + r"""   "population": [\d]+\n"""
            + r"""\}"""
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=128,
            extra_body={"regex": regex},
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_penalty(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=32,
            frequency_penalty=1.0,
        )
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_response_prefill(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": """
Extract the name, size, price, and color from this product description as a JSON object:

<description>
The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99. At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via voice or app—no matter where you place it in your home. This affordable little hub brings convenient hands-free control to your smart devices.
</description>
""",
                },
                {
                    "role": "assistant",
                    "content": "{\n",
                },
            ],
            temperature=0,
        )

        assert (
            response.choices[0]
            .message.content.strip()
            .startswith('"name": "SmartHome Mini",')
        )


# -------------------------------------------------------------------------
#    EBNF Test Class: TestOpenAIServerEBNF
#    Launches the server with xgrammar, has only EBNF tests
# -------------------------------------------------------------------------
class TestOpenAIServerEBNF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # passing xgrammar specifically
        other_args = ["--grammar-backend", "xgrammar"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ebnf(self):
        """
        Ensure we can pass `ebnf` to the local openai server
        and that it enforces the grammar.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        ebnf_grammar = r"""
        root ::= "Hello" | "Hi" | "Hey"
        """
        pattern = re.compile(r"^(Hello|Hi|Hey)[.!?]*\s*$")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful EBNF test bot."},
                {"role": "user", "content": "Say a greeting (Hello, Hi, or Hey)."},
            ],
            temperature=0,
            max_tokens=32,
            extra_body={"ebnf": ebnf_grammar},
        )
        text = response.choices[0].message.content.strip()
        print("EBNF test output:", repr(text))
        self.assertTrue(len(text) > 0, "Got empty text from EBNF generation")
        self.assertRegex(text, pattern, f"Text '{text}' doesn't match EBNF choices")

    def test_ebnf_strict_json(self):
        """
        A stricter EBNF that produces exactly {"name":"Alice"} format
        with no trailing punctuation or extra fields.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        ebnf_grammar = r"""
        root    ::= "{" pair "}"
        pair    ::= "\"name\"" ":" string
        string  ::= "\"" [A-Za-z]+ "\""
        """
        pattern = re.compile(r'^\{"name":"[A-Za-z]+"\}$')

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "EBNF mini-JSON generator."},
                {
                    "role": "user",
                    "content": "Generate single key JSON with only letters.",
                },
            ],
            temperature=0,
            max_tokens=64,
            extra_body={"ebnf": ebnf_grammar},
        )
        text = response.choices[0].message.content.strip()
        print("EBNF strict JSON test output:", repr(text))
        self.assertTrue(len(text) > 0, "Got empty text from EBNF strict JSON test")
        self.assertRegex(
            text, pattern, f"Text '{text}' not matching the EBNF strict JSON shape"
        )

    def test_function_calling_format(self):

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Compute the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "int",
                                "description": "A number",
                            },
                            "b": {
                                "type": "int",
                                "description": "A number",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "Compute (3+5)"}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls

        assert (
            content is None
        ), "When tools provided by the response, content should be None"
        assert (
            isinstance(tool_calls, list) and len(tool_calls) > 0
        ), "Format not matched, tool_calls should be a list"

        function_name = tool_calls[0].function.name
        assert (
            function_name == "add"
        ), "Function name should be add for the above response"


class TestOpenAIEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Configure embedding-specific args
        other_args = ["--is-embedding", "--enable-metrics"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_embedding_single(self):
        """Test single embedding request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input="Hello world")
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_embedding_batch(self):
        """Test batch embedding request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model, input=["Hello world", "Test text"]
        )
        self.assertEqual(len(response.data), 2)
        self.assertTrue(len(response.data[0].embedding) > 0)
        self.assertTrue(len(response.data[1].embedding) > 0)


def test_function_calling_streaming_simple(self):
    """
    Test a simple Function Calling scenario in streaming mode.
    Verify if the function name is correctly returned and if the response contains multiple chunks.
    """
    client = openai.Client(api_key=self.api_key, base_url=self.base_url)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What is the temperature in Paris?"}]

    # Enable streaming mode
    response_stream = client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0.8,
        top_p=0.8,
        stream=True,
        tools=tools,
    )

    chunks = []
    for chunk in response_stream:
        chunks.append(chunk)

    self.assertTrue(len(chunks) > 0, "Streaming should return at least one chunk")
    # Attempt to find function call information from the streaming chunks
    found_function_name = False
    for chunk in chunks:
        choice = chunk.choices[0]
        if choice.delta.tool_calls:
            tool_call = choice.delta.tool_calls[0]
            if tool_call.function.name:
                self.assertEqual(tool_call.function.name, "get_current_weather")
                found_function_name = True
                break

    self.assertTrue(
        found_function_name,
        "Target function name 'get_current_weather' not found in streaming chunks",
    )


def test_function_calling_streaming_args_parsing(self):
    """
    Test the ability of streaming responses to handle argument assembly for Function Calling:
    - The user's request requires multiple arguments
    - AI may return these arguments in chunks, requiring us to assemble them
    """
    client = openai.Client(api_key=self.api_key, base_url=self.base_url)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Compute the sum of two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "int",
                            "description": "First integer",
                        },
                        "b": {
                            "type": "int",
                            "description": "Second integer",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "Please sum 5 and 7, just call the function."}
    ]

    # Enable streaming API
    response_stream = client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0.9,
        top_p=0.9,
        stream=True,
        tools=tools,
    )

    argument_fragments = []
    function_name = None
    for chunk in response_stream:
        choice = chunk.choices[0]
        # If the chunk contains function call information
        if choice.delta.tool_calls:
            tool_call = choice.delta.tool_calls[0]
            function_name = tool_call.function.name or function_name
            # Arguments may be returned in chunks
            if tool_call.function.arguments:
                argument_fragments.append(tool_call.function.arguments)

    self.assertEqual(function_name, "add", "Function name should be 'add'")
    # Combine all argument fragments
    joined_args = "".join(argument_fragments)
    self.assertTrue(len(joined_args) > 0, "No argument content found in function calls")
    # Test if it can be parsed as JSON
    import json

    try:
        args_obj = json.loads(joined_args)
    except json.JSONDecodeError:
        self.fail("Arguments returned from streaming cannot be parsed as valid JSON")

    self.assertIn("a", args_obj, "Parameter 'a' is missing from JSON")
    self.assertIn("b", args_obj, "Parameter 'b' is missing from JSON")
    self.assertEqual(args_obj["a"], "5", "Value of 'a' should be '5'")
    self.assertEqual(args_obj["b"], "7", "Value of 'b' should be '7'")


if __name__ == "__main__":
    unittest.main()
