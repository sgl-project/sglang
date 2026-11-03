"""Focused contract checks for externally owned Engram tables."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.engram_share.vmm import external as engram_external
from sglang.srt.engram_share.vmm import mapping as engram_shared_vmm
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestExternalEngram(CustomTestCase):
    def test_online_weight_update_requires_new_owner_generation(self):
        config = EngramConfig.from_dict(
            {
                "placement": "external",
                "shared_host_endpoint": "127.0.0.1:56000",
                "shared_host_token_file": "/tmp/token",
                "shared_host_generation": "generation",
            }
        )
        with get_context().override_server_args(engram_config=config):
            self.assertEqual(
                _unsupported_derived_weight_cache_error(torch.nn.Linear(1, 1)),
                "Online weight updates require a new external Engram owner generation.",
            )

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

    def test_failed_import_cleanup_retains_owner_lease(self):
        class FakeConnection:
            def __init__(self, reply):
                self.reply = json.dumps(reply).encode() + b"\n"
                self.sent = []
                self.closed = False

            def sendall(self, value):
                self.sent.append(json.loads(value))

            def recv(self, _size):
                reply, self.reply = self.reply, b""
                return reply

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "config.json"
            index_path = directory / "model.safetensors.index.json"
            config_path.write_text("{}")
            index_path.write_text("{}")
            token = directory / "token"
            token.write_text("private-token")
            manifest = {
                "generation": "expected-generation",
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "schema": 1,
                "format": "FP8_E4M3FN_E8M0_BLOCK32",
                "tables": {
                    "1": {
                        "rows": 3,
                        "dim": 64,
                        "scale_width": 2,
                        "head_sizes": [3],
                        "table": {
                            "size_bytes": 198,
                            "padded_bytes": 256,
                            "chunk_bytes": 256,
                            "fabric_handles": ["00" * 64],
                        },
                    }
                },
            }
            connection = FakeConnection(
                {"ok": True, "lease": "lease-id", "manifest": manifest}
            )
            with (
                mock.patch.object(
                    engram_external.socket,
                    "create_connection",
                    return_value=connection,
                ),
                mock.patch.object(
                    engram_external,
                    "import_table",
                    side_effect=engram_shared_vmm.IncompleteTableImportCleanup(
                        "mapped pages remain"
                    ),
                ),
                self.assertRaisesRegex(
                    engram_shared_vmm.IncompleteTableImportCleanup,
                    "mapped pages remain",
                ),
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
            self.assertTrue(connection.closed)
            self.assertEqual(len(connection.sent), 1)
            self.assertEqual(connection.sent[0]["op"], "attach")

    def test_partial_vmm_import_reports_failed_unmap(self):
        class FakeDriver:
            class CUmemAllocationHandleType:
                CU_MEM_HANDLE_TYPE_FABRIC = 1

            class CUmemLocationType:
                CU_MEM_LOCATION_TYPE_DEVICE = 1

            class CUmemAccess_flags:
                CU_MEM_ACCESS_FLAGS_PROT_READ = 1

            class CUresult:
                CUDA_SUCCESS = 0

            def __init__(self):
                self.map_calls = 0
                self.unmap_calls = 0

            def CUmemAccessDesc(self):
                return mock.Mock(location=mock.Mock())

            def cuMemImportFromShareableHandle(self, *_args):
                return (0, 123)

            def cuMemAddressReserve(self, *_args):
                return (0, 4096)

            def cuMemMap(self, *_args):
                self.map_calls += 1
                return 0 if self.map_calls == 1 else 1

            def cuMemSetAccess(self, *_args):
                return 0

            def cuMemUnmap(self, *_args):
                self.unmap_calls += 1
                return 1

            def cuMemAddressFree(self, *_args):
                return 1

            def cuMemRelease(self, *_args):
                return 0

        def check(result, label):
            values = result if isinstance(result, tuple) else (result,)
            if values[0]:
                raise RuntimeError(f"{label} failed")
            return values[1] if len(values) > 1 else None

        driver = FakeDriver()
        table = engram_shared_vmm.TableHandles(
            size_bytes=128,
            padded_bytes=128,
            chunk_bytes=64,
            fabric_handles=(bytes(64), bytes(64)),
        )
        with (
            mock.patch.object(
                engram_shared_vmm, "_get_cuda_driver", return_value=driver
            ),
            mock.patch.object(engram_shared_vmm, "check_drv", side_effect=check),
            self.assertRaises(engram_shared_vmm.IncompleteTableImportCleanup),
        ):
            engram_shared_vmm.import_table(table, 0)
        self.assertEqual(driver.map_calls, 2)
        self.assertEqual(driver.unmap_calls, 1)


if __name__ == "__main__":
    unittest.main()
