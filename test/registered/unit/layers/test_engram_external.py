"""Focused contract checks for externally owned Engram tables."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers import engram_external
from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestExternalEngram(CustomTestCase):
    def test_external_requires_complete_identity_and_direct_lookup(self):
        valid = {
            "placement": "external",
            "shared_host_endpoint": "127.0.0.1:56000",
            "shared_host_token_file": "/tmp/token",
            "shared_host_generation": "generation",
        }
        self.assertEqual(EngramConfig.from_dict(valid).placement, "external")
        for changed in (
            {"shared_host_generation": ""},
            {"replication": "per_node"},
            {"lookup": "all_reduce"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                EngramConfig.from_dict({**valid, **changed})
        with self.assertRaises(ValueError):
            EngramConfig.from_dict({"shared_host_endpoint": "127.0.0.1:56000"})

    def test_skip_large_table_before_get_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            table_name = "layers.1.engram.embed.weight"
            scale_name = "layers.1.engram.embed.scale"
            projection_name = "layers.1.engram.wkv.weight"
            save_file(
                {
                    table_name: torch.arange(16, dtype=torch.float32),
                    scale_name: torch.arange(2, dtype=torch.float32),
                    projection_name: torch.arange(4, dtype=torch.float32),
                },
                path,
            )
            original = weight_utils.safetensors.safe_open
            materialized = []

            class SpyOpen:
                def __init__(self, *args, **kwargs):
                    self.inner = original(*args, **kwargs)

                def __enter__(self):
                    self.inner.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.inner.__exit__(*args)

                def keys(self):
                    return self.inner.keys()

                def get_tensor(self, name):
                    materialized.append(name)
                    return self.inner.get_tensor(name)

            config = EngramConfig.from_dict(
                {
                    "placement": "external",
                    "shared_host_endpoint": "127.0.0.1:56000",
                    "shared_host_token_file": "/tmp/token",
                    "shared_host_generation": "generation",
                }
            )
            source = DefaultModelLoader.Source(
                str(directory := path.parent),
                None,
                model_config=mock.Mock(hf_text_config=mock.Mock(engram_layer_ids=[1])),
            )
            resolved = DefaultModelLoader.ResolvedSource(
                source, str(directory), (str(path),), True
            )
            loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.SAFETENSORS))
            with (
                get_context().override_server_args(
                    model_path="dummy",
                    load_format="safetensors",
                    engram_config=config,
                ),
                mock.patch.object(weight_utils.safetensors, "safe_open", SpyOpen),
            ):
                weights = dict(
                    loader._get_weights_iterator(source, resolved_source=resolved)
                )
            self.assertEqual(materialized, [projection_name])
            self.assertEqual(list(weights), [projection_name])

            materialized.clear()
            with (
                get_context().override_server_args(
                    model_path="dummy",
                    load_format="safetensors",
                    engram_config=EngramConfig.from_dict({"placement": "cpu"}),
                ),
                mock.patch.object(weight_utils.safetensors, "safe_open", SpyOpen),
            ):
                ordinary = dict(
                    loader._get_weights_iterator(source, resolved_source=resolved)
                )
            self.assertEqual(
                set(materialized), {table_name, scale_name, projection_name}
            )
            self.assertEqual(set(ordinary), set(materialized))

    def test_detach_follows_unmap_and_failed_unmap_retains_lease(self):
        events = []

        class FakeMapping:
            address = 1000

            def __init__(self, fail=False):
                self.fail = fail

            def close(self):
                events.append("unmap")
                if self.fail:
                    raise RuntimeError("GPU work still active")

        class FakeConnection:
            def sendall(self, value):
                self.message = json.loads(value)
                events.append("detach")

            def close(self):
                events.append("socket_close")

        connection = FakeConnection()
        engram_external.ExternalEngramTable(
            connection, FakeMapping(), "lease-id", 64
        ).close()
        self.assertEqual(events, ["unmap", "detach", "socket_close"])
        self.assertEqual(connection.message, {"op": "detach", "lease": "lease-id"})
        events.clear()
        with self.assertRaisesRegex(RuntimeError, "GPU work still active"):
            engram_external.ExternalEngramTable(
                FakeConnection(), FakeMapping(fail=True), "lease-id", 64
            ).close()
        self.assertEqual(events, ["unmap"])

    def test_identity_rejection_releases_owner_lease_before_import(self):
        class FakeConnection:
            def __init__(self):
                self.reply = (
                    json.dumps(
                        {
                            "ok": True,
                            "lease": "lease-id",
                            "manifest": {"generation": "wrong-generation"},
                        }
                    ).encode()
                    + b"\n"
                )
                self.sent = []
                self.closed = False

            def sendall(self, value):
                self.sent.append(value)

            def recv(self, _size):
                reply, self.reply = self.reply, b""
                return reply

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}")
            (directory / "model.safetensors.index.json").write_text("{}")
            token = directory / "token"
            token.write_text("private-token")
            connection = FakeConnection()
            with (
                mock.patch.object(
                    engram_external.socket,
                    "create_connection",
                    return_value=connection,
                ),
                mock.patch.object(engram_external, "import_table") as imported,
                self.assertRaisesRegex(ValueError, "generation or checkpoint"),
            ):
                engram_external.attach_table(
                    endpoint="127.0.0.1:56000",
                    token_file=token,
                    generation="expected-generation",
                    checkpoint_dir=directory,
                    layer=1,
                    rows=3,
                    dim=64,
                    head_sizes=(3,),
                    device=0,
                )
            imported.assert_not_called()
            self.assertTrue(connection.closed)
            self.assertIn(b'"op": "detach"', connection.sent[-1])


if __name__ == "__main__":
    unittest.main()
