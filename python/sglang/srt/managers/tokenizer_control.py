"""Instance control operations, independent of the tokenizer IPC transport."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import msgspec

from sglang.srt.constants import GPU_MEMORY_ALL_TYPES
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    AbortReq,
    BaseReq,
    ContinueGenerationReqInput,
    GetInternalStateReq,
    TokenizerControlAckReq,
    TokenizerControlBackendResultReq,
    TokenizerControlBroadcastReq,
    TokenizerControlReq,
    msgpack_decode,
    msgpack_encode,
)

# This return route is consumed by SenderWrapper, never by SocketMapping.
CONTROL_RETURN_PREFIX = "tokenizer-control:"
CONTROL_TIMEOUT_SECONDS = envs.SGLANG_TOKENIZER_CONTROL_TIMEOUT.get()


class _Replies(msgspec.Struct):
    expected: int
    future: asyncio.Future
    values: dict[str, object] = msgspec.field(default_factory=dict)


class ControlRpc:
    """Correlate backend replies without changing generation's wire format."""

    def __init__(self, send: Callable[[BaseReq], Awaitable[None]]):
        self.send = send
        self.pending: dict[str, _Replies] = {}

    async def call(self, request: BaseReq, fan_out: int) -> list[BaseReq]:
        if fan_out < 1:
            raise ValueError("Control RPC requires at least one backend participant")
        operation_id = uuid.uuid4().hex
        replies = _Replies(fan_out, asyncio.get_running_loop().create_future())
        self.pending[operation_id] = replies
        request = msgspec.structs.replace(
            request, http_worker_ipc=CONTROL_RETURN_PREFIX + operation_id
        )
        try:
            await self.send(request)
            return await replies.future
        finally:
            self.pending.pop(operation_id, None)

    def handle_recv(self, obj: TokenizerControlBackendResultReq) -> None:
        replies = self.pending.get(obj.operation_id)
        if replies is None or replies.future.done():
            return
        replies.values.setdefault(obj.worker_id, msgpack_decode(obj.payload))
        if len(replies.values) == replies.expected:
            replies.future.set_result(list(replies.values.values()))


class WorkerBroadcast:
    """Broadcast one action and wait for ACKs from a snapshot of the workers.

    Each call owns its pending replies, so abort can run while another action
    waits for requests to drain. Operation ordering belongs to the caller.
    """

    def __init__(self, workers: Callable[[], set[str]], send: Callable):
        self.workers = workers
        self.send = send
        self.pending: dict[str, tuple[set[str], asyncio.Future]] = {}

    async def broadcast_and_wait(self, action: TokenizerControlBroadcastReq) -> None:
        action = msgspec.structs.replace(action, broadcast_id=uuid.uuid4().hex)
        workers = set(self.workers())
        future = asyncio.get_running_loop().create_future()
        self.pending[action.broadcast_id] = (workers, future)
        try:
            for worker in workers.copy():
                self.send(worker, action)
            if not workers and not future.done():
                future.set_result(None)
            await asyncio.wait_for(future, CONTROL_TIMEOUT_SECONDS)
        finally:
            self.pending.pop(action.broadcast_id, None)

    def handle_recv(self, obj: TokenizerControlAckReq) -> None:
        pending = self.pending.get(obj.broadcast_id)
        if pending is None:
            return
        workers, future = pending
        if obj.worker_ipc_name not in workers or future.done():
            return
        workers.remove(obj.worker_ipc_name)
        if obj.error is not None:
            future.set_exception(RuntimeError(obj.error))
        elif not workers:
            future.set_result(None)


class ControlCoordinator:
    """Compose worker broadcasts and backend calls for instance mutations.

    Callers own the task independently of the HTTP connection. A failed operation
    fences admission: a backend timeout cannot establish whether weights changed.
    """

    def __init__(self, backend: Callable, broadcast_and_wait: Callable, fail: Callable):
        self.backend = backend
        self._broadcast = broadcast_and_wait
        self.fail = fail
        self.active: str | None = None
        self.error: str | None = None
        self.paused = False
        self.released_tags: set[str] = set()
        self.revision = 0
        self.stage = "worker broadcast"

    @property
    def admission_closed(self) -> bool:
        return self.paused or bool(self.released_tags)

    async def run(self, obj: TokenizerControlReq) -> list[BaseReq]:
        if self.error is not None:
            raise RuntimeError(
                f"Previous control operation failed; restart the server: {self.error}"
            )
        if self.active is not None:
            raise RuntimeError("Another control operation is in progress")
        if obj.kind == "continue" and self.released_tags:
            raise RuntimeError(
                f"Resume memory before generation: {sorted(self.released_tags)}"
            )
        self.active = obj.operation_id
        try:
            handler = {
                "pause": self._pause,
                "continue": self._continue,
                "weights": self._update,
                "version": self._update,
                "release": self._release_memory,
                "resume": self._resume_memory,
            }[obj.kind]
            return await asyncio.wait_for(
                handler(obj, msgpack_decode(obj.payload)), CONTROL_TIMEOUT_SECONDS
            )
        except (Exception, asyncio.CancelledError) as exc:
            self.error = (
                f"{obj.kind} timed out during {self.stage}"
                if isinstance(exc, asyncio.TimeoutError)
                else str(exc) or type(exc).__name__
            )
            self.paused = True
            self.fail(obj.operation_id, self.error)
            if isinstance(exc, asyncio.TimeoutError):
                raise TimeoutError(self.error) from exc
            raise
        finally:
            self.active = None

    async def broadcast_and_wait(self, action: TokenizerControlBroadcastReq) -> None:
        self.revision += 1
        action = msgspec.structs.replace(action, revision=self.revision)
        self.stage = f"tokenizer {action.action} broadcast"
        await self._broadcast(action)

    async def _call_backend(self, request: BaseReq, fan_out: int) -> list[BaseReq]:
        self.stage = "backend completion"
        return await self.backend(request, fan_out)

    async def _wait_for_idle(self, fan_out: int) -> None:
        self.stage = "scheduler idle barrier"
        while True:
            states = await self.backend(GetInternalStateReq(), fan_out)
            if states and all(
                state.internal_state["is_fully_idle"] for state in states
            ):
                return
            await asyncio.sleep(0.1)

    async def _quiesce(self, obj: TokenizerControlReq, *, abort: bool = False) -> None:
        if abort:
            await self.broadcast_and_wait(
                TokenizerControlBroadcastReq(action="pause", abort_all=True)
            )
            await self._call_backend(AbortReq(abort_all=True), obj.idle_fan_out)
            if self.paused:
                # Retained requests cannot finish while inference is paused.
                # Admission remains closed throughout this internal resume.
                await self._call_backend(
                    ContinueGenerationReqInput(torch_empty_cache=False),
                    obj.idle_fan_out,
                )
            await self.broadcast_and_wait(TokenizerControlBroadcastReq(action="drain"))
            await self._wait_for_idle(obj.idle_fan_out)
        elif not self.admission_closed:
            await self.broadcast_and_wait(
                TokenizerControlBroadcastReq(action="pause", wait_for_requests=True)
            )
            await self._wait_for_idle(obj.idle_fan_out)
        # A completed global pause already closed admission. Retained requests
        # hold reader locks until continue, so waiting for them here deadlocks.

    async def _pause(self, obj, request):
        if request.mode == "abort":
            await self._quiesce(obj, abort=True)
            results = []
        else:
            if not self.admission_closed:
                await self.broadcast_and_wait(
                    TokenizerControlBroadcastReq(action="pause")
                )
            results = await self._call_backend(request, obj.fan_out)
        self.paused = True
        return results

    async def _continue(self, obj, request):
        results = await self._call_backend(request, obj.fan_out)
        await self.broadcast_and_wait(
            TokenizerControlBroadcastReq(action="update_state", is_pause=False)
        )
        self.paused = False
        return results

    async def _update(self, obj, request):
        abort = getattr(request, "abort_all_requests", False)
        # Version-only updates also relabel active generation in the scheduler.
        if obj.kind == "weights" or abort:
            await self._quiesce(obj, abort=abort)
        results = await self._call_backend(request, obj.fan_out)
        if any(not getattr(result, "success", True) for result in results):
            # Even a reported failure may mean that some ranks changed weights.
            self.error = " | ".join(
                getattr(result, "message", "") for result in results
            )
            self.paused = True
            self.fail(
                obj.operation_id, self.error or "Backend control operation failed"
            )
            return results

        updates = {}
        version = (
            request.weight_version if obj.kind == "weights" else request.new_version
        )
        if version is not None:
            updates["weight_version"] = version
        if hasattr(request, "model_path"):
            updates.update(
                model_path=request.model_path, load_format=request.load_format
            )
        self.paused = self.paused or getattr(request, "keep_pause", False)
        await self.broadcast_and_wait(
            TokenizerControlBroadcastReq(
                action="update_state",
                is_pause=self.admission_closed,
                updates=updates,
                clear_mm_cache=obj.kind == "weights"
                and getattr(request, "flush_cache", False),
                weights_ready=obj.kind == "weights",
            )
        )
        return results

    async def _release_memory(self, obj, request):
        await self._quiesce(obj)
        results = await self._call_backend(request, obj.fan_out)
        self.released_tags.update(request.tags or GPU_MEMORY_ALL_TYPES)
        return results

    async def _resume_memory(self, obj, request):
        was_closed = self.admission_closed
        results = await self._call_backend(request, obj.fan_out)
        if not request.tags:
            self.released_tags.clear()
        else:
            self.released_tags.difference_update(request.tags)
        if was_closed and not self.admission_closed:
            await self.broadcast_and_wait(
                TokenizerControlBroadcastReq(action="update_state", is_pause=False)
            )
        return results


def make_control_request(
    request: BaseReq, kind: str, fan_out: int, idle_fan_out: int
) -> TokenizerControlReq:
    return TokenizerControlReq(
        operation_id=uuid.uuid4().hex,
        kind=kind,
        payload=msgpack_encode(request),
        fan_out=fan_out,
        idle_fan_out=idle_fan_out,
    )
