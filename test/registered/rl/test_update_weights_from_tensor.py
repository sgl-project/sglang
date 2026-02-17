from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=195, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=195, suite="stage-b-test-small-1-gpu-amd")

import functools
import gc
import json
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch

import sglang as sgl
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.srt.utils.weight_checksum import compute_weights_checksum
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Llama stacked params: HF splits -> SGLang merged (concat along dim=0)
_STACKED_PARAMS = [
    (".qkv_proj", [".q_proj", ".k_proj", ".v_proj"]),
    (".gate_up_proj", [".gate_proj", ".up_proj"]),
]

_PERTURB_PARAM_NAME = "model.layers.0.mlp.down_proj.weight"
_PERTURB_NUMEL = 128


def _merge_hf_to_sglang(hf_named_params):
    """Merge HF-format params to SGLang internal format (TP=1, concat along dim=0)."""
    merged = {}
    pending = {}

    for name, tensor in hf_named_params:
        matched = False
        for merged_suffix, shard_suffixes in _STACKED_PARAMS:
            for shard_suffix in shard_suffixes:
                if shard_suffix in name:
                    merged_name = name.replace(shard_suffix, merged_suffix)
                    pending.setdefault(merged_name, {})[shard_suffix] = tensor
                    matched = True
                    break
            if matched:
                break
        if not matched:
            merged[name] = tensor

    for name, parts in pending.items():
        for merged_suffix, shard_suffixes in _STACKED_PARAMS:
            if merged_suffix in name:
                merged[name] = torch.cat([parts[s] for s in shard_suffixes], dim=0)
                break

    return merged


@functools.lru_cache(maxsize=1)
def _load_hf_params():
    """Load and cache HF model params for cross-verification."""
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST, torch_dtype=torch.bfloat16
    )
    params = [(n, p.detach().clone()) for n, p in hf_model.named_parameters()]
    del hf_model
    return params


@functools.lru_cache(maxsize=1)
def _load_perturbed_hf_params():
    """Return HF params with a deterministic perturbation on one tensor."""
    perturbed = []
    found = False
    for name, tensor in _load_hf_params():
        cloned = tensor.clone()
        if name == _PERTURB_PARAM_NAME:
            numel = min(_PERTURB_NUMEL, cloned.numel())
            delta = torch.linspace(0.01, 0.02, steps=numel, dtype=torch.float32).to(
                cloned.dtype
            )
            cloned.view(-1)[:numel].add_(delta)
            found = True
        perturbed.append((name, cloned))

    assert found, f"Cannot find parameter to perturb: {_PERTURB_PARAM_NAME}"
    return perturbed


@functools.lru_cache(maxsize=1)
def _expected_checksum_after_perturbation():
    """Compute expected checksum from perturbed HF params merged to SGLang format."""
    merged = _merge_hf_to_sglang(_load_perturbed_hf_params())
    return compute_weights_checksum(merged.items())


class TestUpdateWeightsFromTensor(CustomTestCase):
    def test_update_weights_from_tensor(self):
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()

        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        checksum_before = engine.get_weights_checksum()
        hf_params = _load_perturbed_hf_params()
        engine.update_weights_from_tensor(list(hf_params))

        checksum_after = engine.get_weights_checksum()
        assert checksum_after == _expected_checksum_after_perturbation()
        assert checksum_after != checksum_before
        engine.shutdown()

        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        memory_after = torch.cuda.memory_allocated()
        assert (
            memory_after <= memory_before + 1024
        ), f"Memory leak detected: {memory_after - memory_before} bytes"

    def test_update_weights_from_tensor_load_format_direct(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        checksum_before = engine.get_weights_checksum()
        # Direct format bypasses merge; send already-merged params
        merged = _merge_hf_to_sglang(_load_perturbed_hf_params())
        engine.update_weights_from_tensor(list(merged.items()), load_format="direct")

        checksum_after = engine.get_weights_checksum()
        assert checksum_after == _expected_checksum_after_perturbation()
        assert checksum_after != checksum_before
        engine.shutdown()

    def test_update_weights_from_tensor_load_format_custom(self):
        custom_loader_name = (
            "sglang.srt.model_executor.model_runner._model_load_weights_direct"
        )
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            custom_weight_loader=[custom_loader_name],
        )

        checksum_before = engine.get_weights_checksum()
        merged = _merge_hf_to_sglang(_load_perturbed_hf_params())
        engine.update_weights_from_tensor(
            list(merged.items()), load_format=custom_loader_name
        )

        checksum_after = engine.get_weights_checksum()
        assert checksum_after == _expected_checksum_after_perturbation()
        assert checksum_after != checksum_before
        engine.shutdown()

    def test_update_weights_from_tensor_load_format_flattened_bucket(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        checksum_before = engine.get_weights_checksum()
        # Flattened bucket calls model.load_weights() internally, so use HF-format names
        hf_params = _load_perturbed_hf_params()
        named_tensors = [(n, t.cuda()) for n, t in hf_params]

        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)

        engine.update_weights_from_tensor(
            named_tensors=[serialized], load_format="flattened_bucket"
        )

        checksum_after = engine.get_weights_checksum()
        assert checksum_after == _expected_checksum_after_perturbation()
        assert checksum_after != checksum_before
        engine.shutdown()


class TestServerUpdateWeightsFromTensorNonBlocking(CustomTestCase):
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
                "text": f"Question: {random.randint(0, 100)},The capital of France is",
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

    def pause_generation(self, mode):
        response = requests.post(
            self.base_url + "/pause_generation",
            json={"mode": mode},
        )
        ret = response.json()
        return ret

    def continue_generation(self):
        response = requests.post(
            self.base_url + "/continue_generation",
            json={},
        )
        ret = response.json()
        return ret

    def run_update_weights(self, named_tensors, flush_cache=True):
        response = requests.post(
            self.base_url + "/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ],
                "flush_cache": flush_cache,
            },
        )
        ret = response.json()
        return ret

    def test_update_weights(self):
        pause_generation_modes = ["in_place", "retract"]
        for pause_generation_mode in pause_generation_modes:
            num_requests = 32
            with ThreadPoolExecutor(num_requests) as executor:
                futures = [
                    executor.submit(self.run_decode, 3000) for _ in range(num_requests)
                ]

                # ensure the decode has been started
                time.sleep(2)

                param_names = [
                    f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)
                ]
                new_tensor = torch.full((16384, 2048), 1.5, device="cuda")
                named_tensors = [(x, new_tensor) for x in param_names]

                ret = self.pause_generation(pause_generation_mode)
                ret = self.run_update_weights(
                    named_tensors, flush_cache=pause_generation_mode == "retract"
                )
                self.assertTrue(ret["success"])
                ret = self.continue_generation()

                for future in as_completed(futures):
                    self.assertNotEqual(
                        future.result()["meta_info"]["finish_reason"]["type"], "abort"
                    )

                for param_name in param_names[:3]:
                    response = requests.post(
                        self.base_url + "/get_weights_by_name",
                        json={"name": param_name},
                    )
                    actual_values = torch.tensor(response.json())[0, :5]
                    assert torch.allclose(
                        actual_values, torch.tensor([1.5] * 5), atol=0.002
                    ), f"{actual_values=}"


if __name__ == "__main__":
    unittest.main()
