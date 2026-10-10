"""Two-GPU UniFlow KV transfer tests on the real uniflow._core binding.

Prefill and decode managers run in spawned processes on GPU 0 and GPU 1 of one
host, as with --base-gpu-id, and exchange control messages over the ZMQ side
channel. The binding picks the transport. No model or server is launched: the
oracle is a digest of the KV, aux, and state bytes on each side.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from enum import IntEnum, auto
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.uniflow.conn import (
    KVTransferError,
    UniflowKVBootstrapServer,
    UniflowKVManager,
    UniflowKVReceiver,
    UniflowKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import is_in_amd_ci

register_amd_ci(
    est_time=60,
    suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd",
    disabled="uniflow._core is not in the CI image",
)

NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_DIM = 64
PAGE_SIZE = 16
NUM_PAGES = 8
AUX_BYTES_PER_SLOT = 64
NUM_AUX_SLOTS = 4
STATE_BYTES = 512
NUM_STATE_SLOTS = 4

LOCALHOST = "127.0.0.1"
PREFILL_GPU = 0
DECODE_GPU = 1
BOOTSTRAP_TIMEOUT_S = 5
POLL_INTERVAL = 0.02
HANDSHAKE_TIMEOUT_S = 30.0
TRANSFER_TIMEOUT_S = 60.0
STARTUP_TIMEOUT_S = 300.0
CALL_TIMEOUT_S = 120.0
TERMINAL = (KVPoll.Success, KVPoll.Failed)
SINGLE_RANK_PARALLEL = dict(
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dcp_size=1,
    attn_dcp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
    pp_group=None,
)

CTX = mp.get_context("spawn")


class Cmd(IntEnum):
    """Requests to a worker; each maps to the _Endpoint method of the same name."""

    FILL = auto()
    DIGEST = auto()
    HANDSHAKE = auto()
    START_SENDER = auto()
    SEND = auto()
    START_RECEIVER = auto()
    SEND_METADATA = auto()
    ABORT_RECEIVER = auto()
    WAIT = auto()
    WAIT_STATUS = auto()
    FAILURE = auto()
    WAIT_ABORT_ACK = auto()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCALHOST, 0))
        return int(sock.getsockname()[1])


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL)
    return True


def _poll_until(
    fetch: Callable[[], int], targets: Iterable[int], timeout: float
) -> int:
    target_set = set(int(target) for target in targets)
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        last = int(fetch())
        if last in target_set:
            return last
        time.sleep(POLL_INTERVAL)
    return last


def _make_buffers(device: torch.device) -> dict[str, Any]:
    kv_shape = (NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return {
        "kv_buffers": [
            torch.zeros(kv_shape, dtype=torch.bfloat16, device=device)
            for _ in range(NUM_LAYERS * 2)
        ],
        # Host memory, as in serving, so aux cannot use the GPU peer tier.
        "aux_buffer": torch.zeros(NUM_AUX_SLOTS, AUX_BYTES_PER_SLOT, dtype=torch.uint8),
        "state_buffer": torch.zeros(
            NUM_STATE_SLOTS, STATE_BYTES, dtype=torch.uint8, device=device
        ),
    }


def _ptrs_lens_items(
    bufs: list[torch.Tensor],
) -> tuple[list[int], list[int], list[int]]:
    ptrs = [int(b.data_ptr()) for b in bufs]
    lens = [int(b.numel() * b.element_size()) for b in bufs]
    items = [int(b[0].numel() * b.element_size()) for b in bufs]
    return ptrs, lens, items


def _fill_pattern(buffers: dict[str, Any], seed: int) -> None:
    # Raw random bytes, so bf16 NaN and denormal patterns are in the copy too.
    device = buffers["state_buffer"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    for tensor in [*buffers["kv_buffers"], buffers["state_buffer"]]:
        tensor.view(torch.uint8).random_(0, 256, generator=generator)
    cpu_generator = torch.Generator().manual_seed(seed)
    buffers["aux_buffer"].random_(0, 256, generator=cpu_generator)
    torch.cuda.synchronize(device)


def _hash_slots(
    buffers: dict[str, Any],
    pages: list[int],
    aux_slots: list[int],
    state_slots: list[int],
) -> dict[str, str]:
    kv = hashlib.sha256()
    for tensor in buffers["kv_buffers"]:
        kv.update(tensor[pages].cpu().view(torch.uint8).numpy().tobytes())
    aux = buffers["aux_buffer"][aux_slots].numpy().tobytes()
    state = buffers["state_buffer"][state_slots].cpu().numpy().tobytes()
    return {
        "kv": kv.hexdigest(),
        "aux": hashlib.sha256(aux).hexdigest(),
        "state": hashlib.sha256(state).hexdigest(),
    }


def _build_kv_args(buffers: dict[str, Any], gpu: int) -> KVArgs:
    args = KVArgs()
    args.kv_cache_dtype_str = "bfloat16"
    args.engine_rank = 0
    args.kv_data_ptrs, args.kv_data_lens, args.kv_item_lens = _ptrs_lens_items(
        buffers["kv_buffers"]
    )
    args.aux_data_ptrs, args.aux_data_lens, args.aux_item_lens = _ptrs_lens_items(
        [buffers["aux_buffer"]]
    )
    state_ptrs, state_lens, state_items = _ptrs_lens_items([buffers["state_buffer"]])
    args.state_data_ptrs = [state_ptrs]
    args.state_data_lens = [state_lens]
    args.state_item_lens = [state_items]
    args.state_types = ["mamba"]
    args.gpu_id = gpu
    args.kv_head_num = NUM_KV_HEADS
    args.total_kv_head_num = NUM_KV_HEADS
    args.page_size = PAGE_SIZE
    args.pp_rank = 0
    args.system_dp_rank = 0
    args.prefill_start_layer = 0
    args.prefill_end_layer = NUM_LAYERS
    args.mla_compression_ratios = None
    return args


class _Endpoint:
    """One manager and its GPU buffers, living in a spawned worker process."""

    def __init__(
        self,
        mode: DisaggregationMode,
        gpu: int,
        bootstrap_port: int,
        server_args: ServerArgs,
    ) -> None:
        torch.cuda.set_device(gpu)
        self.buffers = _make_buffers(torch.device("cuda", gpu))
        self.bootstrap_addr = f"{LOCALHOST}:{bootstrap_port}"
        self.mgr = UniflowKVManager(
            _build_kv_args(self.buffers, gpu), mode, server_args, is_mla_backend=False
        )
        self.senders: dict[int, UniflowKVSender] = {}
        self.receivers: dict[int, UniflowKVReceiver] = {}

    def fill(self, seed: int) -> None:
        _fill_pattern(self.buffers, seed)

    def digest(
        self, pages: list[int], aux_slots: list[int], state_slots: list[int]
    ) -> dict[str, str]:
        return _hash_slots(self.buffers, pages, aux_slots, state_slots)

    def handshake(self) -> bool:
        return _wait_until(
            lambda: self.mgr.try_ensure_parallel_info(self.bootstrap_addr),
            HANDSHAKE_TIMEOUT_S,
        )

    def start_sender(self, room: int, num_pages: int, aux_idx: int) -> None:
        sender = UniflowKVSender(self.mgr, self.bootstrap_addr, room)
        sender.init(num_pages, aux_index=aux_idx)
        self.senders[room] = sender

    def send(self, room: int, pages: list[int], state: list[list[int]] | None) -> None:
        self.senders[room].send(np.array(pages, dtype=np.int32), state)

    def wait_status(self, room: int, targets: Iterable[int], timeout: float) -> int:
        # The manager status, not sender.poll(), which also runs its own
        # bootstrap timeout and would race the transfer worker's.
        return _poll_until(lambda: self.mgr.check_status(room), targets, timeout)

    def start_receiver(self, room: int) -> None:
        receiver = UniflowKVReceiver(self.mgr, self.bootstrap_addr, room)
        receiver.init(0)
        self.receivers[room] = receiver

    def send_metadata(
        self, room: int, pages: list[int], aux_idx: int, state: list[list[int]]
    ) -> None:
        self.receivers[room].send_metadata(
            np.array(pages, dtype=np.int32), aux_index=aux_idx, state_indices=state
        )

    def abort_receiver(self, room: int) -> None:
        self.receivers[room].abort()

    def _room(self, room: int) -> UniflowKVSender | UniflowKVReceiver:
        return self.senders.get(room) or self.receivers[room]

    def wait(self, room: int, targets: Iterable[int]) -> int:
        return _poll_until(self._room(room).poll, targets, TRANSFER_TIMEOUT_S)

    def failure(self, room: int) -> str:
        try:
            self._room(room).failure_exception()
        except KVTransferError as e:
            return e.failure_reason
        raise AssertionError("failure_exception() returned without raising")

    def wait_abort_ack(self, room: int) -> bool:
        return _wait_until(
            lambda: self.mgr.is_abort_release_safe(room, 1), TRANSFER_TIMEOUT_S
        )


def _subprocess_main(
    mode: DisaggregationMode,
    gpu: int,
    bootstrap_port: int,
    cmd_q: mp.Queue,
    res_q: mp.Queue,
) -> None:
    # Read when the manager is built, so setting them here is enough.
    os.environ["SGLANG_HOST_IP"] = LOCALHOST
    os.environ["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = str(BOOTSTRAP_TIMEOUT_S)
    try:
        with contextlib.ExitStack() as stack:
            server_args = stack.enter_context(
                get_context().override_server_args(
                    host=LOCALHOST,
                    disaggregation_bootstrap_port=bootstrap_port,
                    tp_size=1,
                    kv_cache_dtype="bfloat16",
                )
            )
            stack.enter_context(get_parallel().override(**SINGLE_RANK_PARALLEL))
            endpoint = _Endpoint(mode, gpu, bootstrap_port, server_args)
            res_q.put(("ok", None))
            while (request := cmd_q.get()) is not None:
                cmd, args = request
                try:
                    res_q.put(("ok", getattr(endpoint, cmd.name.lower())(*args)))
                except Exception:
                    res_q.put(("err", traceback.format_exc()))
    except Exception:
        res_q.put(("err", traceback.format_exc()))


class _Worker:
    def __init__(self, mode: DisaggregationMode, gpu: int, bootstrap_port: int) -> None:
        self.role = mode.value
        self.cmd_q: mp.Queue = CTX.Queue()
        self.res_q: mp.Queue = CTX.Queue()
        self.proc = CTX.Process(
            target=_subprocess_main,
            args=(mode, gpu, bootstrap_port, self.cmd_q, self.res_q),
            daemon=True,
        )
        self.proc.start()

    def result(self, what: str, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            try:
                kind, payload = self.res_q.get(timeout=1.0)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(
                        f"{self.role} worker exited with {self.proc.exitcode} "
                        f"during {what}"
                    ) from None
                if time.monotonic() >= deadline:
                    # A late reply would be read as the next call's result.
                    self.proc.kill()
                    raise TimeoutError(f"{self.role} {what} took over {timeout}s")
        if kind == "err":
            raise RuntimeError(f"{self.role} {what} failed:\n{payload}")
        return payload

    def call(self, cmd: Cmd, *args: Any) -> Any:
        self.cmd_q.put((cmd, args))
        return self.result(cmd.name, CALL_TIMEOUT_S)

    def shutdown(self) -> None:
        self.cmd_q.put(None)
        self.proc.join(timeout=10.0)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5.0)


class _Pair(NamedTuple):
    prefill: _Worker
    decode: _Worker


def _init_pair(stack: contextlib.ExitStack, bootstrap_port: int) -> _Pair:
    workers = []
    for mode, gpu in (
        (DisaggregationMode.PREFILL, PREFILL_GPU),
        (DisaggregationMode.DECODE, DECODE_GPU),
    ):
        worker = _Worker(mode, gpu, bootstrap_port)
        stack.callback(worker.shutdown)
        workers.append(worker)
    for worker in workers:
        worker.result("startup", STARTUP_TIMEOUT_S)
    pair = _Pair(*workers)
    assert pair.decode.call(Cmd.HANDSHAKE)
    return pair


@pytest.fixture(scope="module")
def bootstrap_server() -> Iterator[int]:
    try:
        importlib.import_module("uniflow._core")
    except ModuleNotFoundError as e:
        # Skip only if uniflow._core cannot be found; a load failure is an error.
        if e.name not in ("uniflow", "uniflow._core") or is_in_amd_ci():
            raise
        pytest.skip(f"uniflow._core is not installed: {e}")
    if torch.cuda.device_count() < 2:
        reason = f"needs two GPUs, found {torch.cuda.device_count()}"
        if is_in_amd_ci():
            raise RuntimeError(reason)
        pytest.skip(reason)
    port = _free_port()
    server = UniflowKVBootstrapServer(LOCALHOST, port)
    try:
        yield port
    finally:
        server.close()


@pytest.fixture(scope="module")
def pd(bootstrap_server: int) -> Iterator[_Pair]:
    """A prefill and decode pair shared by every test; each test uses its own rooms."""
    with contextlib.ExitStack() as stack:
        yield _init_pair(stack, bootstrap_server)


def _transfer(
    pd: _Pair,
    room: int,
    chunks: list[list[int]],
    dst_pages: list[int],
    aux: tuple[int, int],
    state: tuple[int, int],
) -> None:
    """Runs one room to Success on both sides; state rides the last chunk."""
    src_pages = [page for chunk in chunks for page in chunk]
    pd.decode.call(Cmd.START_RECEIVER, room)
    pd.prefill.call(Cmd.START_SENDER, room, len(src_pages), aux[0])
    pd.decode.call(Cmd.SEND_METADATA, room, dst_pages, aux[1], [[state[1]]])
    ready = pd.prefill.call(Cmd.WAIT, room, (KVPoll.WaitingForInput, KVPoll.Failed))
    assert ready == KVPoll.WaitingForInput, pd.prefill.call(Cmd.FAILURE, room)
    for chunk in chunks[:-1]:
        pd.prefill.call(Cmd.SEND, room, chunk, None)
    pd.prefill.call(Cmd.SEND, room, chunks[-1], [[state[0]]])
    received = pd.decode.call(Cmd.WAIT, room, TERMINAL)
    assert received == KVPoll.Success, pd.decode.call(Cmd.FAILURE, room)
    sent = pd.prefill.call(Cmd.WAIT, room, TERMINAL)
    assert sent == KVPoll.Success, pd.prefill.call(Cmd.FAILURE, room)


def test_transfer_is_bit_exact_with_non_identity_pages(pd: _Pair) -> None:
    # One shifted multi-page block, then two pages in reverse order.
    chunks, dst_pages = [[0, 1, 2, 3], [5, 6]], [3, 4, 5, 6, 1, 0]
    src_pages = [page for chunk in chunks for page in chunk]
    aux, state = (2, 1), (0, 3)
    others = (
        [page for page in range(NUM_PAGES) if page not in dst_pages],
        [slot for slot in range(NUM_AUX_SLOTS) if slot != aux[1]],
        [slot for slot in range(NUM_STATE_SLOTS) if slot != state[1]],
    )
    pd.prefill.call(Cmd.FILL, 1)
    pd.decode.call(Cmd.FILL, 2)
    source = pd.prefill.call(Cmd.DIGEST, src_pages, [aux[0]], [state[0]])
    poisoned = pd.decode.call(Cmd.DIGEST, dst_pages, [aux[1]], [state[1]])
    untouched = pd.decode.call(Cmd.DIGEST, *others)
    assert all(poisoned[part] != source[part] for part in source)

    _transfer(pd, 0xC0FFEE_401, chunks, dst_pages, aux, state)

    assert pd.decode.call(Cmd.DIGEST, dst_pages, [aux[1]], [state[1]]) == source
    assert pd.prefill.call(Cmd.DIGEST, src_pages, [aux[0]], [state[0]]) == source
    assert pd.decode.call(Cmd.DIGEST, *others) == untouched


def test_sender_fails_when_metadata_never_arrives(pd: _Pair) -> None:
    room = 0xC0FFEE_402
    started = time.monotonic()
    pd.prefill.call(Cmd.START_SENDER, room, 1, 0)
    pd.prefill.call(Cmd.SEND, room, [0], [[0]])

    status = pd.prefill.call(Cmd.WAIT_STATUS, room, TERMINAL, BOOTSTRAP_TIMEOUT_S * 4)

    assert status == KVPoll.Failed
    assert time.monotonic() - started >= BOOTSTRAP_TIMEOUT_S
    assert pd.prefill.call(Cmd.WAIT, room, TERMINAL) == KVPoll.Failed
    reason = pd.prefill.call(Cmd.FAILURE, room)
    assert "did not arrive before bootstrap timeout" in reason, reason


def test_decode_abort_before_prefill_send_fails_sender(pd: _Pair) -> None:
    room = 0xC0FFEE_403
    pages, aux, state = [0, 1], 0, 0
    pd.prefill.call(Cmd.FILL, 3)
    pd.decode.call(Cmd.FILL, 4)
    held = pd.decode.call(Cmd.DIGEST, pages, [aux], [state])
    source = pd.prefill.call(Cmd.DIGEST, pages, [aux], [state])
    assert all(held[part] != source[part] for part in held)
    pd.decode.call(Cmd.START_RECEIVER, room)
    pd.prefill.call(Cmd.START_SENDER, room, len(pages), aux)
    pd.decode.call(Cmd.SEND_METADATA, room, pages, aux, [[state]])
    ready = pd.prefill.call(Cmd.WAIT, room, (KVPoll.WaitingForInput, KVPoll.Failed))
    assert ready == KVPoll.WaitingForInput

    # abort() arms the drain-ack tracker before it sends ABORT, so the ack
    # cannot arrive first and be dropped.
    pd.decode.call(Cmd.ABORT_RECEIVER, room)
    # Prefill queues the ack only after it marks the room Failed.
    assert pd.decode.call(Cmd.WAIT_ABORT_ACK, room)
    pd.prefill.call(Cmd.SEND, room, pages, [[state]])

    assert pd.prefill.call(Cmd.WAIT, room, TERMINAL) == KVPoll.Failed
    assert pd.prefill.call(Cmd.FAILURE, room) == "Aborted by AbortReq."
    # A later room drains the FIFO transfer worker past the dropped chunk and
    # shows the pair still transfers after an abort.
    _transfer(pd, room + 1, [[4]], [4], (1, 1), (1, 1))
    assert pd.decode.call(Cmd.DIGEST, pages, [aux], [state]) == held


def test_empty_last_chunk_still_transfers_aux_and_state(pd: _Pair) -> None:
    all_pages = list(range(NUM_PAGES))
    aux, state = (3, 2), (1, 2)
    pd.prefill.call(Cmd.FILL, 5)
    pd.decode.call(Cmd.FILL, 6)
    source = pd.prefill.call(Cmd.DIGEST, [], [aux[0]], [state[0]])
    poisoned = pd.decode.call(Cmd.DIGEST, all_pages, [aux[1]], [state[1]])
    assert poisoned["aux"] != source["aux"]
    assert poisoned["state"] != source["state"]

    _transfer(pd, 0xC0FFEE_405, [[]], [], aux, state)

    after = pd.decode.call(Cmd.DIGEST, all_pages, [aux[1]], [state[1]])
    assert after["kv"] == poisoned["kv"]
    assert (after["aux"], after["state"]) == (source["aux"], source["state"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
