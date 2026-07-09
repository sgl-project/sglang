import ctypes
import json
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save
except ModuleNotFoundError:
    torch = None
    save = None


class _Record:
    def __init__(self, path, size, chunks):
        self.path = path
        self.size = size
        self.chunks = chunks


class _Manifest:
    def __init__(self, status, files):
        self.status = status
        self.files = files


class _FakeStore:
    manifest = None
    objects = {}
    range_calls = 0
    registered_buffers = []
    unregistered_buffers = []
    fail_ranges = False
    range_key_batches = []

    def setup(self, *args, **kwargs):
        return 0

    def get(self, key):
        return self.objects.get(key)

    def register_buffer(self, ptr, size):
        self.registered_buffers.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        self.unregistered_buffers.append(ptr)
        return 0

    def get_into_ranges(
        self,
        buffer_ptrs,
        all_keys,
        all_dst_offsets,
        all_src_offsets,
        all_sizes,
    ):
        type(self).range_calls += 1
        self.range_key_batches.append([list(keys) for keys in all_keys])
        if self.fail_ranges:
            return [
                [[-1 for _ in key_sizes] for key_sizes in sizes] for sizes in all_sizes
            ]
        results = []
        for base_ptr, keys, dst_offsets, src_offsets, sizes in zip(
            buffer_ptrs,
            all_keys,
            all_dst_offsets,
            all_src_offsets,
            all_sizes,
        ):
            buffer_results = []
            for key, key_dst, key_src, key_sizes in zip(
                keys, dst_offsets, src_offsets, sizes
            ):
                data = self.objects[key]
                for dst, src, size in zip(key_dst, key_src, key_sizes):
                    ctypes.memmove(base_ptr + dst, data[src : src + size], size)
                buffer_results.append(list(key_sizes))
            results.append(buffer_results)
        return results


class _FakeModelFileCacheClient:
    def __init__(self, store, **kwargs):
        self.store = store

    def inspect_model(self, checkpoint_id):
        return self.store.manifest

    def materialize_file(self, checkpoint_id, path, output_path):
        for record in self.store.manifest.files:
            if record.path == path:
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(
                    b"".join(self.store.get(key) for key in record.chunks)
                )
                return
        raise KeyError(path)


def _install_fake_mooncake_modules():
    mooncake_mod = types.ModuleType("mooncake")
    store_mod = types.ModuleType("mooncake.store")
    weight_store_mod = types.ModuleType("mooncake.weight_store")

    store_mod.MooncakeDistributedStore = _FakeStore
    weight_store_mod.READY = "READY"
    weight_store_mod.ModelFileCacheClient = _FakeModelFileCacheClient

    sys.modules["mooncake"] = mooncake_mod
    sys.modules["mooncake.store"] = store_mod
    sys.modules["mooncake.weight_store"] = weight_store_mod


class MooncakeStoreFileConnectorTest(unittest.TestCase):
    def setUp(self):
        _install_fake_mooncake_modules()
        _FakeStore.range_calls = 0
        _FakeStore.registered_buffers = []
        _FakeStore.unregistered_buffers = []
        _FakeStore.fail_ranges = False
        _FakeStore.range_key_batches = []
        self.tmp = Path(__file__).parent / ".tmp_mooncake_connector_test"
        if self.tmp.exists():
            import shutil

            shutil.rmtree(self.tmp)
        self.tmp.mkdir()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        for name in ("mooncake", "mooncake.store", "mooncake.weight_store"):
            sys.modules.pop(name, None)

    def test_materializes_metadata_and_streams_weights(self):
        if torch is None or save is None:
            self.skipTest("torch and safetensors are required")

        from sglang.srt.connector import create_remote_connector

        config_path = self.tmp / "weight-store.json"
        config_path.write_text(
            json.dumps(
                {
                    "local_hostname": "test-client:12346",
                    "metadata_server": "http://127.0.0.1:8080/metadata",
                    "global_segment_size": "0",
                    "local_buffer_size": "64MB",
                    "protocol": "tcp",
                    "rdma_devices": "",
                    "master_server_addr": "127.0.0.1:50051",
                    "replica_num": 1,
                    "file_chunk_size": 32,
                }
            )
        )

        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        safetensors_payload = save({"linear.weight": tensor})
        chunk_size = 32
        weight_chunks = [
            safetensors_payload[offset : offset + chunk_size]
            for offset in range(0, len(safetensors_payload), chunk_size)
        ]
        weight_keys = [f"weight-chunk-{index}" for index in range(len(weight_chunks))]
        _FakeStore.objects = {
            "config-chunk": b'{"model_type":"test"}',
            **dict(zip(weight_keys, weight_chunks)),
        }
        _FakeStore.manifest = _Manifest(
            "READY",
            [
                _Record(
                    "config.json",
                    len(_FakeStore.objects["config-chunk"]),
                    ["config-chunk"],
                ),
                _Record(
                    "model-00001-of-00001.safetensors",
                    len(safetensors_payload),
                    weight_keys,
                ),
            ],
        )

        connector = create_remote_connector(
            "mooncake://MiniMax-M2.7",
            mooncake_weight_store_config=str(config_path),
            materialize_dir=str(self.tmp / "metadata"),
        )

        connector.pull_files(ignore_pattern=["*.safetensors", "*.bin", "*.pt"])
        self.assertTrue(
            (self.tmp / "metadata" / "MiniMax-M2.7" / "config.json").exists()
        )
        self.assertFalse(
            (
                self.tmp
                / "metadata"
                / "MiniMax-M2.7"
                / "model-00001-of-00001.safetensors"
            ).exists()
        )

        loaded = dict(connector.weight_iterator())
        self.assertTrue(torch.equal(loaded["linear.weight"], tensor))

    def test_tensor_stream_loads_tensor_across_chunks(self):
        if torch is None or save is None:
            self.skipTest("torch and safetensors are required")

        from sglang.srt.connector import create_remote_connector
        from sglang.srt.connector.mooncake_store_file import (
            MooncakeStoreFileConnector,
        )

        config_path = self.tmp / "weight-store.json"
        config_path.write_text(
            json.dumps(
                {
                    "local_hostname": "test-client:12346",
                    "metadata_server": "http://127.0.0.1:8080/metadata",
                    "global_segment_size": "0",
                    "local_buffer_size": "64MB",
                    "protocol": "tcp",
                    "master_server_addr": "127.0.0.1:50051",
                    "file_chunk_size": 32,
                    "tensor_batch_buffer_size": "1KB",
                }
            )
        )

        first_tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        second_tensor = torch.arange(64, 128, dtype=torch.float32).reshape(8, 8)
        payload = save(
            {
                "first.weight": first_tensor,
                "second.weight": second_tensor,
            }
        )
        chunk_size = 32
        chunks = [
            payload[offset : offset + chunk_size]
            for offset in range(0, len(payload), chunk_size)
        ]
        chunk_keys = [f"weight-chunk-{index}" for index in range(len(chunks))]
        _FakeStore.objects = dict(zip(chunk_keys, chunks))
        _FakeStore.manifest = _Manifest(
            "READY",
            [_Record("model.safetensors", len(payload), chunk_keys)],
        )

        connector = create_remote_connector(
            "mooncake://stream-test",
            mooncake_weight_store_config=str(config_path),
            materialize_dir=str(self.tmp / "metadata"),
        )

        self.assertIsInstance(connector, MooncakeStoreFileConnector)
        loaded = dict(connector.weight_iterator())
        self.assertTrue(torch.equal(loaded["first.weight"], first_tensor))
        self.assertTrue(torch.equal(loaded["second.weight"], second_tensor))
        self.assertEqual(_FakeStore.range_calls, 1)
        range_keys = _FakeStore.range_key_batches[0][0]
        self.assertTrue(
            all(left != right for left, right in zip(range_keys, range_keys[1:]))
        )
        self.assertEqual(len(_FakeStore.registered_buffers), 1)
        connector.close()
        self.assertEqual(
            _FakeStore.unregistered_buffers,
            [_FakeStore.registered_buffers[0][0]],
        )

        _FakeStore.fail_ranges = True
        fallback_connector = create_remote_connector(
            "mooncake://stream-test",
            mooncake_weight_store_config=str(config_path),
            materialize_dir=str(self.tmp / "metadata"),
        )
        fallback_loaded = dict(fallback_connector.weight_iterator())
        self.assertTrue(torch.equal(fallback_loaded["first.weight"], first_tensor))
        self.assertTrue(torch.equal(fallback_loaded["second.weight"], second_tensor))

    def test_tensor_stream_resolves_cuda_device_for_worker_threads(self):
        if torch is None or not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        from sglang.srt.connector import create_remote_connector

        config_path = self.tmp / "weight-store.json"
        config_path.write_text(
            json.dumps(
                {
                    "local_hostname": "test-client:12346",
                    "metadata_server": "http://127.0.0.1:8080/metadata",
                    "global_segment_size": "0",
                    "local_buffer_size": "64MB",
                    "protocol": "rdma",
                    "master_server_addr": "127.0.0.1:50051",
                    "file_chunk_size": 32,
                }
            )
        )
        _FakeStore.objects = {}
        _FakeStore.manifest = _Manifest("READY", [])

        connector = create_remote_connector(
            "mooncake://device-test",
            device="cuda",
            mooncake_weight_store_config=str(config_path),
            materialize_dir=str(self.tmp / "metadata"),
        )

        self.assertEqual(connector.tensor_batch_device.type, "cuda")
        self.assertEqual(
            connector.tensor_batch_device.index, torch.cuda.current_device()
        )
        connector.close()


if __name__ == "__main__":
    unittest.main()
