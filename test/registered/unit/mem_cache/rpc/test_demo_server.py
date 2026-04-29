# SPDX-License-Identifier: Apache-2.0
"""Units for ``demo_server`` (gRPC UDS + separate mempool RegisterPool UDS)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import grpc
import pytest
import torch

from sglang.srt.mem_cache.storage.rpc import hicache_pb2_grpc

from sglang.srt.mem_cache.storage.rpc.demo_server import (
    parse_unix_socket_uri,
    start_demo_server,
)
from sglang.srt.mem_cache.storage.rpc.rpc_fd import (
    recv_pool_registration_reply_pool_id,
    send_pool_registration,
    unix_connect,
)

from sglang.srt.mem_cache.storage.rpc.hicache_pb2 import (
    BatchGetRequest,
    BatchPutRequest,
    ExistRequest,
    Iovec,
    Transfer,
)
from sglang.srt.mem_cache.storage.rpc.hicache_rpc_storage import (
    MemfdTensorAllocator,
)


def _register_pool_keepalive(mempool_path: str, pool_name: str, fd: int):
    s = unix_connect(mempool_path)
    send_pool_registration(s, pool_name, fd)
    pool_id = recv_pool_registration_reply_pool_id(s)
    return s, pool_id


@pytest.fixture
def demo_ctx(tmp_path: Path):
    data = tmp_path / "ldata"
    data.mkdir()
    mempool_sock = str(tmp_path / "mempool.sock")
    grpc_sock = str(tmp_path / "grpc.sock")
    ctx = start_demo_server(str(data), mempool_sock, grpc_sock)
    time.sleep(0.35)
    keepalives: list = []
    ctx["_test_mempool_keepalives"] = keepalives
    try:
        yield ctx, mempool_sock, grpc_sock, keepalives
    finally:
        for s in keepalives:
            try:
                s.close()
            except OSError:
                pass
        ctx["shutdown"].set()
        ctx["grpc_server"].stop(grace=2.0)


def _stub(grpc_sock_path: str):
    tgt = "unix://" + parse_unix_socket_uri(grpc_sock_path)
    chan = grpc.insecure_channel(tgt)
    grpc.channel_ready_future(chan).result(timeout=15)
    return hicache_pb2_grpc.StorageServerStub(chan)


def test_grpc_mux_registration_put_exist_get(demo_ctx):
    ctx, mempool_path, grpc_path, keepalives = demo_ctx

    fd = os.memfd_create("tdemo", getattr(os, "MFD_CLOEXEC", 0))
    os.ftruncate(fd, 4096)
    reg_sock, pool_id = _register_pool_keepalive(mempool_path, "kv", fd)
    keepalives.append(reg_sock)
    time.sleep(0.1)

    buf = ctx["pool_mmap"][pool_id]
    stub = _stub(grpc_path)

    buf[0:4] = b"\xde\xad\xbe\xef"
    key_id = b"page-a"
    put_resp = stub.BatchPut(
        BatchPutRequest(
            puts=[
                Transfer(
                    id=key_id,
                    iovec=[Iovec(offset=0, length=4)],
                    pool_id=pool_id,
                )
            ],
        ),
        timeout=10,
    )
    assert put_resp.results == [0]
    ex = stub.Exist(ExistRequest(id=[key_id]), timeout=10)
    assert list(ex.results) == [True]
    buf[0:4] = b"\x00\x00\x00\x00"
    get_resp = stub.BatchGet(
        BatchGetRequest(
            gets=[
                Transfer(
                    id=key_id,
                    iovec=[Iovec(offset=0, length=4)],
                    pool_id=pool_id,
                )
            ],
        ),
        timeout=10,
    )
    assert get_resp.results == [0]
    assert buf[0:4] == b"\xde\xad\xbe\xef"


def test_multi_iovec_put_and_get_roundtrip(demo_ctx):
    """Several disjoint mmap slices concatenate on Put and scatter back on Get."""
    ctx, mempool_path, grpc_path, keepalives = demo_ctx

    fd = os.memfd_create("tdemo_multi_iov", getattr(os, "MFD_CLOEXEC", 0))
    os.ftruncate(fd, 4096)
    reg_sock, pool_id = _register_pool_keepalive(mempool_path, "kv", fd)
    keepalives.append(reg_sock)
    time.sleep(0.1)

    buf = ctx["pool_mmap"][pool_id]
    stub = _stub(grpc_path)

    slice_a_off, slice_a_len = 64, 3
    slice_b_off, slice_b_len = 512, 5
    buf[slice_a_off : slice_a_off + slice_a_len] = b"foo"
    buf[slice_b_off : slice_b_off + slice_b_len] = b"hello"

    key_id = b"multi-iov-page"
    put_resp = stub.BatchPut(
        BatchPutRequest(
            puts=[
                Transfer(
                    id=key_id,
                    pool_id=pool_id,
                    iovec=[
                        Iovec(offset=slice_a_off, length=slice_a_len),
                        Iovec(offset=slice_b_off, length=slice_b_len),
                    ],
                )
            ],
        ),
        timeout=10,
    )
    assert put_resp.results == [0]

    buf[slice_a_off : slice_a_off + slice_a_len] = b"\x00" * slice_a_len
    buf[slice_b_off : slice_b_off + slice_b_len] = b"\x00" * slice_b_len

    get_resp = stub.BatchGet(
        BatchGetRequest(
            gets=[
                Transfer(
                    id=key_id,
                    pool_id=pool_id,
                    iovec=[
                        Iovec(offset=slice_a_off, length=slice_a_len),
                        Iovec(offset=slice_b_off, length=slice_b_len),
                    ],
                )
            ],
        ),
        timeout=10,
    )
    assert get_resp.results == [0]
    assert bytes(buf[slice_a_off : slice_a_off + slice_a_len]) == b"foo"
    assert bytes(buf[slice_b_off : slice_b_off + slice_b_len]) == b"hello"


def test_batch_put_and_batch_get_multiple_transfers(demo_ctx):
    """One BatchPut / BatchGet RPC carries several independent Transfer items."""
    ctx, mempool_path, grpc_path, keepalives = demo_ctx

    fd = os.memfd_create("tdemo_batch", getattr(os, "MFD_CLOEXEC", 0))
    os.ftruncate(fd, 4096)
    reg_sock, pool_id = _register_pool_keepalive(mempool_path, "kv", fd)
    keepalives.append(reg_sock)
    time.sleep(0.1)

    buf = ctx["pool_mmap"][pool_id]
    stub = _stub(grpc_path)

    off0, len0 = 0, 4
    off1, len1 = 256, 6
    buf[off0 : off0 + len0] = b"\xaa\xcc\xee\xff"
    buf[off1 : off1 + len1] = b"qwerty"

    put_resp = stub.BatchPut(
        BatchPutRequest(
            puts=[
                Transfer(
                    id=b"key-alpha",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=off0, length=len0)],
                ),
                Transfer(
                    id=b"key-beta",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=off1, length=len1)],
                ),
                Transfer(
                    id=b"key-gamma",
                    pool_id=pool_id,
                    iovec=[
                        Iovec(offset=off0, length=2),
                        Iovec(offset=off1 + 4, length=2),
                    ],
                ),
            ],
        ),
        timeout=10,
    )
    assert list(put_resp.results) == [0, 0, 0]

    ex = stub.Exist(
        ExistRequest(id=[b"key-alpha", b"key-beta", b"key-gamma"]),
        timeout=10,
    )
    assert list(ex.results) == [True, True, True]

    buf[off0 : off0 + len0] = b"\x00" * len0
    buf[off1 : off1 + len1] = b"\x00" * len1

    get_resp = stub.BatchGet(
        BatchGetRequest(
            gets=[
                Transfer(
                    id=b"key-beta",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=off1, length=len1)],
                ),
                Transfer(
                    id=b"key-alpha",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=off0, length=len0)],
                ),
                Transfer(
                    id=b"key-gamma",
                    pool_id=pool_id,
                    iovec=[
                        Iovec(offset=off0, length=2),
                        Iovec(offset=off1 + 4, length=2),
                    ],
                ),
            ],
        ),
        timeout=10,
    )
    assert list(get_resp.results) == [0, 0, 0]
    assert bytes(buf[off0 : off0 + len0]) == b"\xaa\xcc\xee\xff"
    assert bytes(buf[off1 : off1 + len1]) == b"qwerty"


def test_batch_get_multiple_mixed_existing_and_missing(demo_ctx):
    """Each slot in BatchGet.results is independent: hit vs miss."""
    ctx, mempool_path, grpc_path, keepalives = demo_ctx

    fd = os.memfd_create("tdemo_mix", getattr(os, "MFD_CLOEXEC", 0))
    os.ftruncate(fd, 4096)
    reg_sock, pool_id = _register_pool_keepalive(mempool_path, "kv", fd)
    keepalives.append(reg_sock)
    time.sleep(0.1)

    buf = ctx["pool_mmap"][pool_id]
    stub = _stub(grpc_path)

    buf[32:36] = b"zzzz"
    stub.BatchPut(
        BatchPutRequest(
            puts=[
                Transfer(
                    id=b"present",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=32, length=4)],
                )
            ],
        ),
        timeout=10,
    )

    buf[32:36] = b"\x00" * 4
    get_resp = stub.BatchGet(
        BatchGetRequest(
            gets=[
                Transfer(
                    id=b"present",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=32, length=4)],
                ),
                Transfer(
                    id=b"absent-never_put",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=64, length=4)],
                ),
                Transfer(
                    id=b"present",
                    pool_id=pool_id,
                    iovec=[Iovec(offset=32, length=4)],
                ),
            ],
        ),
        timeout=10,
    )
    assert get_resp.results[0] == 0
    assert get_resp.results[1] == grpc.StatusCode.NOT_FOUND.value[0]
    assert get_resp.results[2] == 0
    assert bytes(buf[32:36]) == b"zzzz"
    assert bytes(buf[64:68]) == b"\x00" * 4


def test_allocator_tensors_roundtrip_via_pool_mmap(demo_ctx):
    ctx, mempool_path, grpc_path, keepalives = demo_ctx
    allocator = MemfdTensorAllocator()
    t = allocator.allocate((8, 8), dtype=torch.float32, device="cpu")
    t.fill_(3.14159)
    reg_sock, pool_id = _register_pool_keepalive(
        mempool_path, "kv", allocator.fd
    )
    keepalives.append(reg_sock)
    time.sleep(0.1)

    stub = _stub(grpc_path)
    assert allocator.base_ptr is not None
    assert getattr(t, "memfd_offset", None) == 0
    off = int(t.memfd_offset)
    key_id = b"tensor-key"
    put_resp = stub.BatchPut(
        BatchPutRequest(
            puts=[
                Transfer(
                    id=key_id,
                    iovec=[
                        Iovec(
                            offset=off,
                            length=t.numel() * t.element_size(),
                        )
                    ],
                    pool_id=pool_id,
                )
            ],
        ),
        timeout=10,
    )
    assert put_resp.results == [0]
    t.zero_()
    get_resp = stub.BatchGet(
        BatchGetRequest(
            gets=[
                Transfer(
                    id=key_id,
                    iovec=[
                        Iovec(
                            offset=off,
                            length=t.numel() * t.element_size(),
                        )
                    ],
                    pool_id=pool_id,
                )
            ],
        ),
        timeout=10,
    )
    assert get_resp.results == [0]
    assert torch.allclose(
        t, torch.full_like(t, 3.14159), rtol=1e-5, atol=1e-5
    )


def test_mempool_disconnect_unmaps_pool_id(demo_ctx):
    ctx, mempool_path, grpc_path, keepalives = demo_ctx
    fd = os.memfd_create("tdemo_unmap", getattr(os, "MFD_CLOEXEC", 0))
    os.ftruncate(fd, 64)
    reg_sock, pool_id = _register_pool_keepalive(mempool_path, "kv", fd)
    stub = _stub(grpc_path)
    time.sleep(0.1)
    reg_sock.close()
    time.sleep(0.3)
    assert pool_id not in ctx["pool_mmap"]
    put_resp = stub.BatchPut(
        BatchPutRequest(
            puts=[
                Transfer(
                    id=b"x",
                    iovec=[Iovec(offset=0, length=1)],
                    pool_id=pool_id,
                )
            ],
        ),
        timeout=10,
    )
    assert put_resp.results[0] == grpc.StatusCode.FAILED_PRECONDITION.value[0]
