"""Exercise RunAI shard selection with real tensor reads from a local S3 server."""

import json
import os
import tempfile
import threading
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import boto3
import torch
from moto.server import create_backend_app
from safetensors.torch import save
from werkzeug.serving import make_server

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.utils.runai_utils import ObjectStorageModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextmanager
def _s3_bucket():
    requests = []
    backend = create_backend_app("s3")

    def observed(environ, start_response):
        requests.append((environ["REQUEST_METHOD"], environ["PATH_INFO"]))
        return backend(environ, start_response)

    server = make_server("127.0.0.1", 0, observed, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        with (
            tempfile.TemporaryDirectory() as cache,
            patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "testing",
                    "AWS_SECRET_ACCESS_KEY": "testing",
                    "AWS_SESSION_TOKEN": "testing",
                    "AWS_DEFAULT_REGION": "us-east-1",
                    "AWS_EC2_METADATA_DISABLED": "true",
                    "AWS_ENDPOINT_URL": endpoint,
                    "RUNAI_STREAMER_S3_ENDPOINT": endpoint,
                    "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING": "0",
                    "SGLANG_CACHE_DIR": cache,
                },
            ),
            patch.object(loader_mod, "get_server_args", return_value=None),
        ):
            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
                aws_session_token="testing",
            )
            bucket = f"sglang-runai-selection-{uuid.uuid4().hex}"
            client.create_bucket(Bucket=bucket)
            yield client, bucket, requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class TestRunaiModelStreamerS3(CustomTestCase):
    def test_selected_shards_are_read_with_original_tensor_values(self):
        with _s3_bucket() as (client, bucket, requests):
            contents = {
                "main.safetensors": {"main.weight": torch.tensor([1.0])},
                "draft.safetensors": {"draft.weight": torch.tensor([2.0, 3.0])},
                "mixed.safetensors": {
                    "main.bias": torch.tensor([4.0]),
                    "draft.bias": torch.tensor([5.0]),
                },
                "mtp.safetensors": {"draft.extra": torch.tensor([6.0])},
            }
            for shard, tensors in contents.items():
                client.put_object(
                    Bucket=bucket, Key=f"model/{shard}", Body=save(tensors)
                )
            client.put_object(
                Bucket=bucket,
                Key="model/model.safetensors.index.json",
                Body=json.dumps(
                    {
                        "weight_map": {
                            "main.weight": "main.safetensors",
                            "draft.weight": "draft.safetensors",
                            "main.bias": "mixed.safetensors",
                            "draft.bias": "mixed.safetensors",
                        }
                    }
                ),
            )
            root = f"s3://{bucket}/model"
            metadata = ObjectStorageModel.download_and_get_path(root)
            self.assertTrue(
                os.path.isfile(os.path.join(metadata, "model.safetensors.index.json"))
            )
            self.assertFalse(
                any(name.endswith(".safetensors") for name in os.listdir(metadata))
            )
            config = SimpleNamespace(
                model_path=root,
                revision=None,
                hf_config=SimpleNamespace(
                    # The existing GLM packaging workaround restores this
                    # unindexed MTP file before model-owned shard selection.
                    architectures=["Glm4MoeForCausalLMNextN"],
                    num_nextn_predict_layers=1,
                ),
            )
            for has_hook in (True, False):
                with self.subTest(has_hook=has_hook):
                    model = SimpleNamespace()
                    if has_hook:
                        model.is_unused_checkpoint_weight = lambda name: (
                            not name.startswith("draft.")
                        )
                    loader = loader_mod.RunaiModelStreamerLoader(
                        LoadConfig(
                            load_format=LoadFormat.RUNAI_STREAMER,
                            model_loader_extra_config={"distributed": False},
                        )
                    )
                    loader.target_device_str = "cpu"
                    requests.clear()
                    # Streamer buffers may be reused after advancing the iterator.
                    loaded = {
                        name: tensor.clone()
                        for name, tensor in loader._get_all_weights(config, model)
                    }
                    expected = {
                        "draft.weight": torch.tensor([2.0, 3.0]),
                        "main.bias": torch.tensor([4.0]),
                        "draft.bias": torch.tensor([5.0]),
                        "draft.extra": torch.tensor([6.0]),
                    }
                    expected_shards = {
                        "draft.safetensors",
                        "mixed.safetensors",
                        "mtp.safetensors",
                    }
                    if not has_hook:
                        expected["main.weight"] = torch.tensor([1.0])
                        expected_shards.add("main.safetensors")
                    self.assertEqual(set(loaded), set(expected))
                    for name, value in expected.items():
                        torch.testing.assert_close(loaded[name], value)
                    self.assertEqual(
                        {
                            path.rsplit("/", 1)[-1]
                            for method, path in requests
                            if method == "GET" and path.endswith(".safetensors")
                        },
                        expected_shards,
                    )

    def test_qwen35_draft_loads_shared_embedding_without_target_shards(self):
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

        with _s3_bucket() as (client, bucket, requests):
            contents = {
                "target.safetensors": {
                    "model.layers.0.weight": torch.tensor([[1.0, 1.0]]),
                    "lm_head.weight": torch.tensor([[6.0], [7.0]]),
                },
                "draft.safetensors": {
                    "mtp.layers.0.weight": torch.tensor([[2.0, 3.0]]),
                },
                "embedding.safetensors": {
                    "model.language_model.embed_tokens.weight": torch.tensor(
                        [[4.0], [5.0]]
                    ),
                },
            }
            weight_map = {}
            for shard, tensors in contents.items():
                client.put_object(
                    Bucket=bucket, Key=f"model/{shard}", Body=save(tensors)
                )
                weight_map.update({name: shard for name in tensors})
            client.put_object(
                Bucket=bucket,
                Key="model/model.safetensors.index.json",
                Body=json.dumps({"weight_map": weight_map}),
            )
            root = f"s3://{bucket}/model"
            ObjectStorageModel.download_and_get_path(root)
            config = SimpleNamespace(
                model_path=root,
                revision=None,
                hf_config=SimpleNamespace(architectures=["Qwen3_5ForCausalLMMTP"]),
            )
            model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
            torch.nn.Module.__init__(model)
            model.config = SimpleNamespace()
            model.model = torch.nn.Module()
            model.model.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 1, bias=False)]
            )
            model.model.embed_tokens = torch.nn.Embedding(2, 1)
            loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={"distributed": False},
                )
            )
            loader.target_device_str = "cpu"
            requests.clear()
            loaded = model.load_weights(loader._get_all_weights(config, model))
            self.assertEqual(
                loaded, {"model.layers.0.weight", "model.embed_tokens.weight"}
            )
            torch.testing.assert_close(
                model.model.layers[0].weight, torch.tensor([[2.0, 3.0]])
            )
            torch.testing.assert_close(
                model.model.embed_tokens.weight, torch.tensor([[4.0], [5.0]])
            )
            self.assertEqual(
                {
                    path.rsplit("/", 1)[-1]
                    for method, path in requests
                    if method == "GET" and path.endswith(".safetensors")
                },
                {"draft.safetensors", "embedding.safetensors"},
            )


if __name__ == "__main__":
    unittest.main()
