"""Unit tests for srt/managers/output_store.py and the detokenizer side of it."""

import concurrent.futures
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
import pybase64
import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402
from sglang.srt.managers.io_struct import BatchTokenIDOutput  # noqa: E402
from sglang.srt.managers.output_store import (  # noqa: E402
    MooncakeBundleWriter,
    OutputStoreConfig,
    TokenReplayStash,
    maybe_create_output_store,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BASE_CONFIG = {
    "master_server_address": "10.0.0.1:50051",
    "local_hostname": "10.0.0.2",
    "local_buffer_size": "2gb",
    "key_prefix": "miles-object-store",
}


def _parse(**overrides):
    return OutputStoreConfig.from_extra_config(
        json.dumps({**_BASE_CONFIG, **overrides})
    )


def _chunk(lengths, token_ids, logprobs):
    return SamplingMaskChunk(
        lengths=np.array(lengths, np.int32),
        token_ids=np.array(token_ids, np.int32),
        logprobs=np.array(logprobs, np.float32),
    )


class TestOutputStoreConfig(CustomTestCase):
    def test_defaults_match_mooncake_client_defaults(self):
        config = _parse()
        self.assertEqual(config.local_buffer_size, 2 * 1024**3)
        self.assertEqual(config.protocol, "rdma")
        self.assertEqual(config.metadata_server, "P2PHANDSHAKE")
        self.assertEqual(config.device_name, "")
        self.assertEqual((config.namespace, config.partition), ("default", "default"))
        self.assertEqual((config.replica_num, config.chunk_bytes), (1, None))

        config = _parse(
            protocol="tcp",
            partition="run-1",
            replica_num=2,
            chunk_bytes=64 * 1024**2,
        )
        self.assertEqual(config.protocol, "tcp")
        self.assertEqual((config.partition, config.replica_num), ("run-1", 2))
        self.assertEqual(config.chunk_bytes, 64 * 1024**2)

    def test_mooncake_env_does_not_configure_the_store(self):
        """MOONCAKE_* env vars belong to SGLang's other Mooncake users; the output
        store reads only its JSON config."""
        with envs.MOONCAKE_PROTOCOL.override("tcp"):
            config = _parse()
        self.assertEqual(config.protocol, "rdma")

    def test_rejects_configs_it_cannot_serve(self):
        cases = {
            "unknown key": json.dumps({**_BASE_CONFIG, "check_server": True}),
            "replica_num as text": json.dumps({**_BASE_CONFIG, "replica_num": "2"}),
        }
        for key in _BASE_CONFIG:
            cases[f"no {key}"] = json.dumps(
                {k: v for k, v in _BASE_CONFIG.items() if k != key}
            )
        for name, extra_config in cases.items():
            with (
                self.subTest(name),
                self.assertRaises((ValueError, msgspec.ValidationError)),
            ):
                OutputStoreConfig.from_extra_config(extra_config)

    def test_backend_none_needs_no_mooncake_and_pd_is_rejected(self):
        with patch.dict(sys.modules, {"mooncake": None}):
            self.assertIsNone(
                maybe_create_output_store(
                    backend="none", extra_config=None, disaggregation_mode="null"
                )
            )
            with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                maybe_create_output_store(
                    backend="mooncake",
                    extra_config=json.dumps(_BASE_CONFIG),
                    disaggregation_mode="prefill",
                )


class _FakeTransfer:
    def __init__(self, store, key_prefix):
        self.puts = []
        self.removed = []
        self.cleanup_error = None

    def put(self, data, **kwargs):
        self.puts.append((data, kwargs))
        return f"ref-{len(self.puts)}"

    def cleanup_dataproto(self, ref):
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.removed.append(ref)


def _fake_mooncake(setup_error=0):
    store_module = types.ModuleType("mooncake.store")
    store_module.setup_configs = []

    class MooncakeDistributedStore:
        def setup(self, config):
            store_module.setup_configs.append(config)
            return setup_error

    class ReplicateConfig:
        replica_num = 1
        with_hard_pin = False

    store_module.MooncakeDistributedStore = MooncakeDistributedStore
    store_module.ReplicateConfig = ReplicateConfig

    structured = types.ModuleType("mooncake.structured_object_store")
    structured.FieldSchema = lambda **kwargs: kwargs
    structured.MooncakeBundleTransfer = _FakeTransfer
    structured.export_ref = lambda ref: {"exported": ref}
    structured.import_ref = lambda handle: ("imported", handle["exported"])
    return {
        "mooncake": types.ModuleType("mooncake"),
        "mooncake.store": store_module,
        "mooncake.structured_object_store": structured,
    }


class TestMooncakeBundleWriter(CustomTestCase):
    def _store(self, setup_error=0, **overrides):
        modules = _fake_mooncake(setup_error)
        config = msgspec.convert(
            {
                **_BASE_CONFIG,
                "local_buffer_size": 1024,
                "protocol": "tcp",
                "metadata_server": "P2PHANDSHAKE",
                "device_name": "mlx5_0",
                **overrides,
            },
            type=OutputStoreConfig,
        )
        with patch.dict(sys.modules, modules):
            store = MooncakeBundleWriter(config)
        self.addCleanup(store._executor.shutdown, wait=True)
        return store, modules["mooncake.store"].setup_configs

    def test_setup_registers_no_segment(self):
        _, setup_configs = self._store()
        self.assertEqual(
            setup_configs,
            [
                {
                    "local_hostname": "10.0.0.2",
                    "metadata_server": "P2PHANDSHAKE",
                    "global_segment_size": 0,
                    "local_buffer_size": 1024,
                    "protocol": "tcp",
                    "rdma_devices": "mlx5_0",
                    "master_server_addr": "10.0.0.1:50051",
                }
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            self._store(setup_error=-1)

    def test_put_writes_each_field_as_one_tensor_row(self):
        """Readers rely on one tensor-batch row per field and on the packed
        sampling-mask arrays concatenating every streamed chunk in order, zero-row
        chunks included."""
        for replica_num in (1, 2):
            with self.subTest(replica_num=replica_num):
                store, _ = self._store(
                    replica_num=replica_num, partition="run-1", chunk_bytes=4096
                )
                stash = TokenReplayStash(
                    routed_experts=torch.arange(4, dtype=torch.int32).reshape(2, 1, 2),
                    indexer_topk=torch.zeros((2, 3, 4), dtype=torch.int32),
                )
                stash.add_sampling_mask(_chunk([2], [5, 6], [-0.5]))
                stash.add_sampling_mask(_chunk([], [], []))
                stash.add_sampling_mask(_chunk([1], [7], [-1.0]))

                output_store_ref = store.submit_put(stash).result()

                (data, kwargs) = store._transfer.puts[0]
                self.assertEqual(
                    output_store_ref,
                    {
                        "handle": {"exported": "ref-1"},
                        "fields": {
                            "routed_experts": {"dtype": "int32", "shape": [2, 1, 2]},
                            "indexer_topk": {"dtype": "int32", "shape": [2, 3, 4]},
                            "output_token_sampling_mask_lengths": {
                                "dtype": "int32",
                                "shape": [2],
                            },
                            "output_token_sampling_mask_token_ids": {
                                "dtype": "int32",
                                "shape": [3],
                            },
                            "output_token_sampling_logprobs": {
                                "dtype": "float32",
                                "shape": [2],
                            },
                        },
                    },
                )
                fields = output_store_ref["fields"]
                self.assertEqual(
                    {name: list(rows.shape) for name, rows in data.items()},
                    {name: [1, *field["shape"]] for name, field in fields.items()},
                )
                np.testing.assert_array_equal(
                    data["output_token_sampling_mask_token_ids"][0], [5, 6, 7]
                )
                np.testing.assert_array_equal(
                    data["output_token_sampling_logprobs"][0], [-0.5, -1.0]
                )
                self.assertEqual(
                    kwargs["field_schemas"],
                    dict.fromkeys(
                        fields,
                        {
                            "codec": "auto",
                            "nullable": False,
                            "metadata": {"section": "batch"},
                        },
                    ),
                )
                self.assertEqual(
                    (kwargs["type"], kwargs["partition"], kwargs["chunk_bytes"]),
                    ("dict", "run-1", 4096),
                )
                # Readers own the bundle's lifetime, so Mooncake must not evict it.
                self.assertEqual(
                    (kwargs["config"].replica_num, kwargs["config"].with_hard_pin),
                    (replica_num, True),
                )
                self.assertEqual(
                    stash.inline_meta_info(), {"output_token_sampling_mask_length": 2}
                )

    def test_put_keeps_dtypes_numpy_cannot_hold(self):
        """A stash may hand over bfloat16, e.g. diffusion latents, which numpy
        cannot hold; the ref spells dtypes without the torch prefix."""

        class _LatentStash:
            def is_empty(self):
                return False

            def to_bundle_fields(self):
                return {"latents": torch.ones((3, 4), dtype=torch.bfloat16)}

        store, _ = self._store()
        output_store_ref = store.submit_put(_LatentStash()).result()

        (data, _) = store._transfer.puts[0]
        self.assertEqual(data["latents"].dtype, torch.bfloat16)
        self.assertEqual(
            output_store_ref["fields"],
            {"latents": {"dtype": "bfloat16", "shape": [3, 4]}},
        )

    def test_undelivered_bundles_are_removed(self):
        store, _ = self._store()
        written = concurrent.futures.Future()
        failed = concurrent.futures.Future()
        store.cleanup_after(written)
        store.cleanup_after(failed)
        written.set_result({"handle": {"exported": "ref-a"}})
        failed.set_exception(RuntimeError("put failed"))
        store._executor.shutdown(wait=True)
        self.assertEqual(store._transfer.removed, [("imported", "ref-a")])

    def test_failed_cleanup_logs_the_handle(self):
        store, _ = self._store()
        store._transfer.cleanup_error = RuntimeError("master down")
        written = concurrent.futures.Future()
        with self.assertLogs("sglang.srt.managers.output_store", "ERROR") as logs:
            store.cleanup_after(written)
            written.set_result({"handle": {"exported": "r"}})
            store._executor.shutdown(wait=True)
        self.assertIn('{"exported": "r"}', "\n".join(logs.output))


class TestDetokenizerReplayOutputs(CustomTestCase):
    def _handle(self, *, output_store_enabled):
        manager = DetokenizerManager.__new__(DetokenizerManager)
        manager.output_store_enabled = output_store_enabled
        recv_obj = SimpleNamespace(
            **{f.name: None for f in msgspec.structs.fields(BatchTokenIDOutput)}
        )
        recv_obj.rids = []
        recv_obj.routed_experts = [torch.tensor([[1, 2]], dtype=torch.int32)]
        return manager.handle_batch_token_id_out(recv_obj), recv_obj.routed_experts

    def test_tensors_skip_base64_only_when_store_enabled(self):
        out, tensors = self._handle(output_store_enabled=True)
        self.assertIsNone(out.routed_experts)
        self.assertIs(out.routed_experts_raw, tensors)

        out, tensors = self._handle(output_store_enabled=False)
        self.assertEqual(
            out.routed_experts,
            [pybase64.b64encode(tensors[0].numpy().tobytes()).decode("utf-8")],
        )
        self.assertIsNone(out.routed_experts_raw)


if __name__ == "__main__":
    unittest.main()
