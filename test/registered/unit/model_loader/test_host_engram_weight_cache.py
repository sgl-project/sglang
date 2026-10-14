# SPDX-License-Identifier: Apache-2.0
"""Host Engram FD ownership, mapping, and loader recovery contracts."""

import array
import gc
import mmap
import os
import socket
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.srt.weight_cache.model_state import (
    STATE_ENTRY,
    export_model_state,
    host_table_exports,
)
from sglang.srt.weight_cache.protocol import recv_msg
from sglang.srt.weight_cache.transport import (
    TorchIpcTransportBackend,
    _recv_fd,
    _send_fd,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class HostLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.host_table = None
        self.weight = nn.Parameter(
            torch.empty(2, 32, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.empty(2, 1, dtype=torch.float8_e8m0fnu), requires_grad=False
        )


class HostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = HostLayer()


class TestHostEngramCache(CustomTestCase):
    def test_memfd_shards_do_not_run_anonymous_hugepage_compaction(self):
        from sglang.srt.layers.engram import _HostTable

        table = _HostTable.__new__(_HostTable)
        table.layout, table.fd, table.dirty = "per_rank", 123, True
        table.bytes = SimpleNamespace(data_ptr=lambda: 4096)
        with (
            patch("sglang.srt.layers.engram._huge_pages_backing", return_value=(64, 0)),
            patch("sglang.srt.layers.engram.drop_checkpoint_page_cache") as drop,
            patch.object(
                table, "_collapse", side_effect=AssertionError("anonymous-only")
            ),
        ):
            table.finish_load("test")
        drop.assert_not_called()
        self.assertFalse(table.dirty)

    def test_shared_backing_survives_owner_fd_close_and_repeated_client_mapping(self):

        fd = os.memfd_create("engram-unit", 0)
        os.ftruncate(fd, 66)
        backing = mmap.mmap(fd, 66, flags=mmap.MAP_SHARED)
        raw = torch.frombuffer(backing, dtype=torch.uint8)
        raw.fill_(127)
        source = HostModel()
        source.layer.weight = nn.Parameter(
            raw[:64].view(torch.float8_e4m3fn).view(2, 32), requires_grad=False
        )
        source.layer.scale = nn.Parameter(
            raw[64:].view(torch.float8_e8m0fnu).view(2, 1), requires_grad=False
        )
        source.layer.host_table = SimpleNamespace(
            fd=fd, bytes=raw, nbytes=66, layout="per_rank"
        )
        tensors = {name: (t, True) for name, t in source.state_dict().items()}
        host_entries, fds = host_table_exports(source, tensors)
        manifest = export_model_state(source, tensors)
        sender = TorchIpcTransportBackend()
        sender.configure_host_table_exports(host_entries, fds)
        entries = sender.prepare_export(tensors)
        entries[STATE_ENTRY] = {"model_state": manifest}
        register_calls = []
        cudart = SimpleNamespace(
            cudaHostRegister=lambda ptr, size, flags: (
                register_calls.append((ptr, size)) or 0
            ),
            cudaHostUnregister=lambda ptr: 0,
        )
        try:
            # Two clients receive independent FDs for the same physical backing.
            receivers = []
            for _ in range(2):
                receiver = TorchIpcTransportBackend()
                a, b = socket.socketpair()
                try:
                    sender.send_fetch_state_response(
                        a, config={}, entries=entries, pid=os.getpid()
                    )
                    result = receiver.recv_fetch_state_response(b, recv_msg(b))
                    self.assertEqual(
                        os.fstat(receiver._host_received_fds[0]).st_ino,
                        os.fstat(fd).st_ino,
                    )
                    self.assertFalse(os.get_inheritable(receiver._host_received_fds[0]))
                    receivers.append((receiver, result))
                finally:
                    a.close()
                    b.close()
            os.close(fd)
            fd = None
            for receiver, result in receivers:
                loader = IpcModelLoader(
                    LoadConfig(model_loader_extra_config={"dsv41_weight_cache": True})
                )
                loader._transport_backend = receiver
                cfg = SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41"),
                    dtype=torch.float32,
                )
                with (
                    patch(
                        "sglang.srt.weight_cache.ipc_loader._initialize_model",
                        side_effect=lambda *args: HostModel(),
                    ),
                    patch.object(torch.cuda, "cudart", return_value=cudart),
                    patch(
                        "sglang.srt.layers.engram._registered_device_ptr",
                        side_effect=lambda ptr: ptr + 4096,
                    ),
                ):
                    model = loader._load_zero_copy_mode(
                        cfg, None, result["entries"], None
                    )
                    self.assertEqual(model.layer.weight.device.type, "cpu")
                    self.assertNotEqual(model.layer.weight.data_ptr(), raw.data_ptr())
                    self.assertEqual(
                        model.layer._host_table_ptrs[0],
                        model.layer.host_table.device_ptr,
                    )
                    self.assertEqual(
                        model.layer._host_table_ptrs[1]
                        - model.layer._host_table_ptrs[0],
                        64,
                    )
                    raw[0] = 42
                    self.assertEqual(
                        model.layer.weight.view(torch.uint8)[0, 0].item(), 42
                    )
                    table = model.layer.host_table
                    del model
                    gc.collect()
                    receiver._host_tables.clear()
                    table.close_import()
            self.assertEqual(
                len(register_calls), 2
            )  # once per table/client, not per tensor
        finally:
            if fd is not None:
                os.close(fd)
            del tensors, source, raw
            gc.collect()
            backing.close()

    def test_meta_embedding_does_not_allocate_or_register_host_memory(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.engram import EngramEmbedding, _HostTable
        from sglang.srt.runtime_context import get_context, get_parallel

        with (
            get_context().override_server_args(model_path="dummy"),
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_tp_size=2,
                moe_tp_rank=1,
            ),
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("shared"),
            patch.object(
                _HostTable, "__init__", side_effect=AssertionError("must not allocate")
            ),
            torch.device("meta"),
        ):
            model = EngramEmbedding(8, 32, 1)
        self.assertTrue(model.weight.is_meta)
        self.assertEqual(tuple(model.weight.shape), (8, 32))
        self.assertIsNone(model.host_table)

    def test_fragmented_fd_header(self):
        fd = os.memfd_create("fragmented", 0)
        a, b = socket.socketpair()
        try:
            header = struct.pack("<Q", 7)
            a.sendmsg(
                [header[:2]],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))],
            )
            a.sendall(header[2:])
            index, imported = _recv_fd(b)
            self.assertEqual(index, 7)
            self.assertFalse(os.get_inheritable(imported))
            os.close(imported)
        finally:
            a.close()
            b.close()
            os.close(fd)

    def test_bad_fd_index_is_closed(self):
        fd = os.memfd_create("bad-index", 0)
        os.ftruncate(fd, 1)
        backend = TorchIpcTransportBackend()
        desc = {"index": 0, "module": "layer", "layout": "shared", "nbytes": 1}
        result = {
            "host_table_count": 1,
            "entries": {"layer.weight": {"host_table": desc}},
        }
        a, b = socket.socketpair()
        try:
            _send_fd(a, fd, 9)
            before = len(os.listdir("/proc/self/fd"))
            with self.assertRaisesRegex(ValueError, "index/size"):
                backend.recv_fetch_state_response(b, result)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        finally:
            a.close()
            b.close()
            os.close(fd)

    def test_registration_failure_closes_imported_mapping_and_fd(self):
        from sglang.srt.layers.engram import _HostTable

        fd = os.memfd_create("register-fail", 0)
        os.ftruncate(fd, 32)
        with patch.object(
            torch.cuda,
            "cudart",
            return_value=SimpleNamespace(cudaHostRegister=lambda *a: 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "cudaHostRegister"):
                _HostTable.from_fd("shared", 32, fd)
        with self.assertRaises(OSError):
            os.fstat(fd)


if __name__ == "__main__":
    unittest.main()
