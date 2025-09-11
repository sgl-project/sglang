import json
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


###############################################################################
# Engine Mode Tests (Single-configuration)
###############################################################################
class TestEngineUpdateWeightsFromDisk(CustomTestCase):
    def setUp(self):
        self.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # Initialize the engine in offline (direct) mode.
        self.engine = sgl.Engine(model_path=self.model)

    def tearDown(self):
        self.engine.shutdown()

    def run_decode(self):
        prompts = ["The capital of France is"]
        sampling_params = {"temperature": 0, "max_new_tokens": 32}
        outputs = self.engine.generate(prompts, sampling_params)
        print("=" * 100)
        print(
            f"[Engine Mode] Prompt: {prompts[0]}\nGenerated text: {outputs[0]['text']}"
        )
        return outputs[0]["text"]

    def run_update_weights(self, model_path):
        ret = self.engine.update_weights_from_disk(model_path)
        print(json.dumps(ret))
        return ret

    def test_update_weights(self):
        origin_response = self.run_decode()
        # Update weights: use new model (remove "-Instruct")
        new_model_path = self.model.replace("-Instruct", "")
        ret = self.run_update_weights(new_model_path)
        self.assertTrue(ret[0])  # ret is a tuple; index 0 holds the success flag

        updated_response = self.run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])

        # Revert back to original weights
        ret = self.run_update_weights(self.model)
        self.assertTrue(ret[0])
        reverted_response = self.run_decode()
        self.assertEqual(origin_response[:32], reverted_response[:32])

    def test_update_weights_unexist_model(self):
        origin_response = self.run_decode()
        new_model_path = self.model.replace("-Instruct", "wrong")
        ret = self.run_update_weights(new_model_path)
        self.assertFalse(ret[0])
        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])


###############################################################################
# HTTP Server Mode Tests (Single-configuration)
###############################################################################
class TestServerUpdateWeightsFromDisk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        print("=" * 100)
        print(f"[Server Mode] Generated text: {response.json()['text']}")
        return response.json()["text"]

    def get_model_info(self):
        response = requests.get(self.base_url + "/get_model_info")
        model_path = response.json()["model_path"]
        print(json.dumps(response.json()))
        return model_path

    def run_update_weights(self, model_path):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": model_path},
        )
        ret = response.json()
        print(json.dumps(ret))
        return ret

    def test_update_weights(self):
        origin_model_path = self.get_model_info()
        print(f"[Server Mode] origin_model_path: {origin_model_path}")
        origin_response = self.run_decode()

        new_model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST.replace("-Instruct", "")
        ret = self.run_update_weights(new_model_path)
        self.assertTrue(ret["success"])

        updated_model_path = self.get_model_info()
        print(f"[Server Mode] updated_model_path: {updated_model_path}")
        self.assertEqual(updated_model_path, new_model_path)
        self.assertNotEqual(updated_model_path, origin_model_path)

        updated_response = self.run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])

        ret = self.run_update_weights(origin_model_path)
        self.assertTrue(ret["success"])
        updated_model_path = self.get_model_info()
        self.assertEqual(updated_model_path, origin_model_path)

        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])

    def test_update_weights_unexist_model(self):
        origin_model_path = self.get_model_info()
        print(f"[Server Mode] origin_model_path: {origin_model_path}")
        origin_response = self.run_decode()

        new_model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST.replace("-Instruct", "wrong")
        ret = self.run_update_weights(new_model_path)
        self.assertFalse(ret["success"])

        updated_model_path = self.get_model_info()
        print(f"[Server Mode] updated_model_path: {updated_model_path}")
        self.assertEqual(updated_model_path, origin_model_path)

        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])


class TestServerUpdateWeightsFromDiskAbortAllRequests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--max-running-requests", 8],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens=32):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def get_model_info(self):
        response = requests.get(self.base_url + "/get_model_info")
        model_path = response.json()["model_path"]
        print(json.dumps(response.json()))
        return model_path

    def run_update_weights(self, model_path, abort_all_requests=False):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={
                "model_path": model_path,
                "abort_all_requests": abort_all_requests,
            },
        )
        ret = response.json()
        print(json.dumps(ret))
        return ret

    def test_update_weights_abort_all_requests(self):
        origin_model_path = self.get_model_info()
        print(f"[Server Mode] origin_model_path: {origin_model_path}")

        num_requests = 32
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [
                executor.submit(self.run_decode, 16000) for _ in range(num_requests)
            ]

            # ensure the decode has been started
            time.sleep(2)

            new_model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST.replace("-Instruct", "")
            ret = self.run_update_weights(new_model_path, abort_all_requests=True)
            self.assertTrue(ret["success"])

            for future in as_completed(futures):
                self.assertEqual(
                    future.result()["meta_info"]["finish_reason"]["type"], "abort"
                )

        updated_model_path = self.get_model_info()
        print(f"[Server Mode] updated_model_path: {updated_model_path}")
        self.assertEqual(updated_model_path, new_model_path)
        self.assertNotEqual(updated_model_path, origin_model_path)


###############################################################################
# Parameterized Tests for update_weights_from_disk
# Test coverage is determined based on the value of is_in_ci:
# - In a CI environment: randomly select one mode (Engine or Server) and test only with tp=1, dp=1.
# - In a non-CI environment: test both Engine and Server modes, and enumerate all combinations
#   with tp and dp ranging from 1 to 2.
###############################################################################
class TestUpdateWeightsFromDiskParameterized(CustomTestCase):
    def run_common_test(self, mode, tp, dp):
        """
        Common test procedure for update_weights_from_disk.
        For Engine mode, we instantiate the engine with tp_size=tp.
        For Server mode, we launch the server with additional arguments for tp (dp is not used in server launch here).
        """
        if mode == "Engine":
            # Instantiate engine with additional parameter tp_size.
            print(f"[Parameterized Engine] Testing with tp={tp}, dp={dp}")
            engine = sgl.Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                random_seed=42,
                tp_size=tp,
                # dp parameter is not explicitly used in this API.
            )
            try:
                origin_response = self._engine_update_weights_test(engine)
            finally:
                engine.shutdown()
        elif mode == "Server":
            print(f"[Parameterized Server] Testing with tp={tp}, dp={dp}")
            # Pass additional arguments to launch the server.
            base_args = ["--tp-size", str(tp)]
            process = popen_launch_server(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=base_args,
            )
            try:
                origin_response = self._server_update_weights_test(DEFAULT_URL_FOR_TEST)
            finally:
                kill_process_tree(process.pid)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _engine_update_weights_test(self, engine):
        # Run the update weights test on the given engine instance.
        def run_decode():
            prompts = ["The capital of France is"]
            sampling_params = {"temperature": 0, "max_new_tokens": 32}
            outputs = engine.generate(prompts, sampling_params)
            print("=" * 100)
            print(
                f"[Parameterized Engine] Prompt: {prompts[0]}\nGenerated text: {outputs[0]['text']}"
            )
            return outputs[0]["text"]

        def run_update_weights(model_path):
            ret = engine.update_weights_from_disk(model_path)
            print(json.dumps(ret))
            return ret

        origin_response = run_decode()
        new_model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST.replace("-Instruct", "")
        ret = run_update_weights(new_model_path)
        self.assertTrue(ret[0])
        updated_response = run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])
        ret = run_update_weights(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        self.assertTrue(ret[0])
        reverted_response = run_decode()
        self.assertEqual(origin_response[:32], reverted_response[:32])
        return origin_response

    def _server_update_weights_test(self, base_url):
        def run_decode():
            response = requests.post(
                base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            print("=" * 100)
            print(f"[Parameterized Server] Generated text: {response.json()['text']}")
            return response.json()["text"]

        def get_model_info():
            response = requests.get(base_url + "/get_model_info")
            model_path = response.json()["model_path"]
            print(json.dumps(response.json()))
            return model_path

        def run_update_weights(model_path):
            response = requests.post(
                base_url + "/update_weights_from_disk",
                json={"model_path": model_path},
            )
            ret = response.json()
            print(json.dumps(ret))
            return ret

        origin_model_path = get_model_info()
        origin_response = run_decode()
        new_model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST.replace("-Instruct", "")
        ret = run_update_weights(new_model_path)
        self.assertTrue(ret["success"])
        updated_model_path = get_model_info()
        self.assertEqual(updated_model_path, new_model_path)
        self.assertNotEqual(updated_model_path, origin_model_path)
        updated_response = run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])
        ret = run_update_weights(origin_model_path)
        self.assertTrue(ret["success"])
        updated_model_path = get_model_info()
        self.assertEqual(updated_model_path, origin_model_path)
        reverted_response = run_decode()
        self.assertEqual(origin_response[:32], reverted_response[:32])
        return origin_response

    def test_parameterized_update_weights(self):
        if is_in_ci():
            # In CI, choose one random mode (Engine or Server) with tp=1, dp=1.
            mode = random.choice(["Engine", "Server"])
            test_suits = [(1, 1, mode)]
        else:
            # Otherwise, test both modes and enumerate tp,dp combinations from 1 to 2.
            test_suits = []
            for mode in ["Engine", "Server"]:
                for tp in [1, 2]:
                    for dp in [1, 2]:
                        test_suits.append((tp, dp, mode))
        for tp, dp, mode in test_suits:
            with self.subTest(mode=mode, tp=tp, dp=dp):
                self.run_common_test(mode, tp, dp)


if __name__ == "__main__":
    unittest.main()
