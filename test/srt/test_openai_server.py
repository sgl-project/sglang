import subprocess
import time
import unittest

import openai
import requests

from sglang.srt.utils import kill_child_process


class TestOpenAIServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        port = 30000
        timeout = 300

        command = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "localhost",
            "--port", str(port),
        ]
        cls.process = subprocess.Popen(command, stdout=None, stderr=None)
        cls.base_url = f"http://localhost:{port}/v1"
        cls.model = model

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.base_url}/models")
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(10)
        raise TimeoutError("Server failed to start within the timeout period.")

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_completion(self, echo, logprobs):
        client = openai.Client(api_key="EMPTY", base_url=self.base_url)
        prompt = "The capital of France is"
        response = client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=0.1,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
        )
        text = response.choices[0].text
        if echo:
            assert text.startswith(prompt)
        if logprobs:
            assert response.choices[0].logprobs
            if echo:
                assert response.choices[0].logprobs.token_logprobs[0] == None
            else:
                assert response.choices[0].logprobs.token_logprobs[0] != None
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_completion(self):
        for echo in [True, False]:
            for logprobs in [None, 5]:
                self.run_completion(echo, logprobs)


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # from sglang.global_config import global_config

    # global_config.verbosity = 2
    # t = TestSRTBackend()
    # t.setUpClass()
    # t.test_few_shot_qa()
    # t.tearDownClass()
