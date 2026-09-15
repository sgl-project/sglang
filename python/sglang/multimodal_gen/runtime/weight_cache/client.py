# SPDX-License-Identifier: Apache-2.0
"""Strict diffusion component admission on the existing SRT wire/IPC layers."""

import dataclasses
import json
import os
import socket
import stat
import struct
import time
import uuid

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weight_cache.adapters import dit_wan
from sglang.multimodal_gen.runtime.weight_cache.identity import (
    compatibility_plan,
    locate,
)
from sglang.multimodal_gen.runtime.weight_cache.plan import plan_diff
from sglang.srt.weight_cache.protocol import recv_msg, send_msg
from sglang.weight_cache_common.descriptors import CACHE_ABI, StateManifest
from sglang.weight_cache_common.liveness import ProcessIdentity
from sglang.weight_cache_common.transport import (
    CudaIpcImporter,
    ExportGeneration,
    IpcDelivery,
    StorageHandle,
)

logger = init_logger(__name__)
PROTOCOL = {"family": "diffusion", "protocol_version": 1, "cache_abi": CACHE_ABI}


def peer_identity(sock):
    pid, uid, _ = struct.unpack(
        "3i",
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")),
    )
    if uid != os.getuid():
        raise PermissionError("Weight-cache peer belongs to another user")
    return ProcessIdentity.read(pid)


def decode_generation(value):
    return ExportGeneration(
        **{**value, "producer": ProcessIdentity(**value["producer"])}
    )


def validate_response(response):
    if any(response.get(key) != value for key, value in PROTOCOL.items()):
        raise ValueError("Weight-cache protocol family/version/ABI mismatch")
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Weight-cache request rejected: {response.get('error', response)}"
        )


class WeightCacheClient:
    def __init__(self, plan, args):
        self.plan = plan
        self.path = locate(plan, args)
        self.timeout = args.weight_cache_timeout
        self.sock = None

    def __enter__(self):
        st = self.path.lstat()  # Missing is a strict error, never a disk fallback.
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise PermissionError(f"Not an owned weight-cache socket: {self.path}")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.settimeout(self.timeout)
            self.sock.connect(str(self.path))
            self.peer = peer_identity(self.sock)
        except BaseException:
            self.sock.close()
            raise
        return self

    def __exit__(self, *_):
        self.sock.close()

    def request(self, kind, **fields):
        send_msg(
            self.sock,
            {**PROTOCOL, "type": kind, "compatibility": self.plan.to_dict(), **fields},
        )
        response = recv_msg(self.sock)
        validate_response(response)
        return response

    def manifest(self, expected_generation=None):
        response = self.request("query_manifest")
        if response["compatibility"] != self.plan.to_dict():
            raise ValueError(
                f"Weight-cache compatibility mismatch: {plan_diff(self.plan.to_dict(), response['compatibility'])}"
            )
        generation = decode_generation(response["generation"])
        if generation.producer != self.peer or not self.peer.is_alive():
            raise ValueError(
                "Weight-cache producer differs from authenticated socket peer"
            )
        if expected_generation is not None and generation != expected_generation:
            raise ValueError("Weight-cache generation changed after launcher admission")
        manifest = StateManifest.from_dict(response["manifest"])
        if manifest.digest != generation.manifest_digest:
            raise ValueError("Weight-cache state manifest digest mismatch")
        return generation, manifest


def materialize_from_cache(prepared, args):
    start = time.perf_counter()
    plan = compatibility_plan(prepared, args)
    planned = time.perf_counter()
    admission = args._weight_cache_admission
    if admission is None or admission[0] != plan:
        raise RuntimeError("Strict cache client requires matching launcher admission")
    with WeightCacheClient(plan, args) as client:
        generation, manifest = client.manifest(admission[1])
        admitted = time.perf_counter()
        # Watchdog is live before even requesting any counted send references.
        importer = CudaIpcImporter(generation, manifest)
        guarded = time.perf_counter()
        model = dit_wan.build_meta(prepared.transformer)
        constructed = time.perf_counter()
        request_id = uuid.uuid4().hex
        response = client.request(
            "fetch_component",
            component="transformer",
            generation=dataclasses.asdict(generation),
            request_id=request_id,
        )
        delivery = IpcDelivery(
            decode_generation(response["generation"]),
            response["request_id"],
            tuple(StorageHandle(**handle) for handle in response["storages"]),
        )
        if delivery.request_id != request_id:
            raise ValueError("Weight-cache delivery request ID mismatch")
        fetched = time.perf_counter()
        importer.receive(delivery, model, request_id=request_id)
        mapped = time.perf_counter()
        model = dit_wan.finalize_after_import(model)
    finalized = time.perf_counter()
    elapsed = finalized - start
    logger.info(
        "[WeightCache] transformer imported in %.3fs (%d shared bytes)",
        elapsed,
        manifest.unique_storage_bytes,
    )
    logger.info(
        "[WeightCache] transformer import stages: %s",
        json.dumps(
            {
                "compatibility": planned - start,
                "manifest": admitted - planned,
                "guard": guarded - admitted,
                "meta": constructed - guarded,
                "fetch": fetched - constructed,
                "mapping": mapped - fetched,
                "finalize": finalized - mapped,
                "total": elapsed,
            },
            sort_keys=True,
        ),
    )
    pipeline = prepared.materialize(args, loaded_modules={"transformer": model})
    pipeline.memory_usages["transformer"] = manifest.unique_storage_bytes / (1024**3)
    pipeline._weight_cache_import_seconds = elapsed
    importer.check_alive()
    return pipeline
