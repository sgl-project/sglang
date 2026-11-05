"""Bounded Mooncake codec pipelines, composed explicitly by the adapter.

These helpers import neither CUDA nor KVServe and do not patch global classes.
Blocking GPU/credit waits execute only in codec/TX/completion workers.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass


class SourceRoomGate:
    """Track queued and running readers until a terminal sender closes the room."""

    def __init__(self):
        self._lock = threading.Lock()
        self._closed = False
        self._readers = 0

    def acquire(self):
        with self._lock:
            if self._closed:
                return False
            self._readers += 1
            return True

    def release(self):
        with self._lock:
            if self._readers <= 0:
                raise RuntimeError("KVServe source reader released twice")
            self._readers -= 1

    def close_and_is_idle(self):
        # No GPU operation or wait under this lock.  Once closed, even a late
        # enqueue using this sender cannot acquire ownership of released pages.
        with self._lock:
            self._closed = True
            return self._readers == 0


@dataclass
class EncodedLease:
    index: int
    buffers: tuple
    state: str = "free"


class EncodedTransferPipeline:
    """Fixed ownership bank; failures quarantine only potentially active writes."""

    def __init__(self, buffers, write, failed, workers=2, timeout=60):
        if not buffers or not 1 <= workers <= len(buffers):
            raise ValueError("TX workers must fit the nonempty encoded bank")
        self.leases = [EncodedLease(i, tuple(value)) for i, value in enumerate(buffers)]
        self.available = list(self.leases)
        self.condition = threading.Condition()
        self.queue = queue.Queue(maxsize=len(self.leases))
        self.write, self.failed, self.timeout = write, failed, timeout
        self.error = None
        self.closing = False
        self._close_started = False
        self._close_finished = False
        self.threads = [
            threading.Thread(target=self._run, name=f"kvserve-tx-{i}", daemon=True)
            for i in range(workers)
        ]
        for thread in self.threads:
            thread.start()

    def acquire(self):
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.available or self.error or self.closing,
                timeout=self.timeout,
            ):
                raise TimeoutError("Encoded staging bank exhausted")
            if self.error or self.closing:
                raise RuntimeError(self.error or "Encoded transfer pipeline is closing")
            lease = self.available.pop(0)
            if lease.state != "free":
                raise RuntimeError("Encoded lease ownership corruption")
            lease.state = "encoding_copy"
            return lease

    def discard_unsubmitted(self, lease):
        with self.condition:
            if lease.state != "encoding_copy":
                raise RuntimeError("Only an unsubmitted lease can be discarded")
            self._release_locked(lease)

    def submit(self, lease, job):
        future = Future()
        with self.condition:
            if lease.state != "encoding_copy":
                raise RuntimeError("Encoded lease submitted twice")
            if self.error or self.closing:
                self._release_locked(lease)
                raise RuntimeError(self.error or "Encoded transfer pipeline is closing")
            lease.state = "queued"
            # Every queued/running item owns one bank: capacity cannot overflow.
            self.queue.put_nowait((lease, job, future))
        return future

    def fail(self, exc):
        with self.condition:
            first = self.error is None
            if first:
                self.error = str(exc)
            self.condition.notify_all()
        if first:
            self.failed(str(exc))

    def _release_locked(self, lease):
        lease.state = "free"
        self.available.append(lease)
        self.condition.notify_all()

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            lease, job, future = item
            try:
                with self.condition:
                    error = self.error
                    lease.state = "inflight" if error is None else "queued_cancelled"
                if error is not None:
                    with self.condition:
                        self._release_locked(lease)
                    raise RuntimeError(error)
                value = self.write(lease, job)
                # A successful callback means RDMA is complete, not merely enqueued.
                with self.condition:
                    self._release_locked(lease)
                future.set_result(value)
            except BaseException as exc:
                with self.condition:
                    # Callback failures can leave RDMA unsettled: never recycle.
                    if lease.state == "inflight":
                        lease.state = "quarantined"
                self.fail(exc)
                future.set_exception(exc)
            finally:
                self.queue.task_done()

    def close(self):
        if threading.current_thread() in self.threads:
            raise RuntimeError("TX worker cannot close its own queue")
        with self.condition:
            if self._close_finished:
                return
            if self._close_started:
                raise RuntimeError(
                    "TX close is already in progress; banks stay registered"
                )
            self._close_started = True
            self.closing = True
            self.condition.notify_all()
        self.queue.join()
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join(timeout=self.timeout)
        if any(thread.is_alive() for thread in self.threads):
            raise TimeoutError(
                "TX worker did not exit; registered banks must stay alive"
            )
        with self.condition:
            self._close_finished = True


class InlineNotifications:
    def __init__(self, send, failed, record, encode=None):
        if encode is None:
            import msgspec

            encode = msgspec.msgpack.encode
        self.send, self.failed, self.record, self.encode = send, failed, record, encode
        self.condition = threading.Condition()
        self.active = 0
        self.closed = False
        # Compatibility for callers that explicitly drain notifier.queue.join.
        self.queue = self

    def put(self, address, tag, value):
        with self.condition:
            if self.closed:
                raise RuntimeError("Inline KVServe notifier is closed")
            self.active += 1
        try:
            begin = time.perf_counter()
            payload = self.encode([value])
            # send is the existing _send_multipart_locked, not an unlocked socket.
            self.send(address, [tag, payload])
            self.record(
                dict(
                    event="control_batch",
                    tag=tag.decode(),
                    frames=1,
                    bytes=len(payload),
                    inline_control=True,
                    control_submit_ms=(time.perf_counter() - begin) * 1000,
                )
            )
        except Exception as exc:
            self.failed(str(exc))
            raise
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def join(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.active == 0, timeout=60):
                raise TimeoutError("Inline KVServe notification send did not drain")

    def close(self):
        with self.condition:
            self.closed = True
        self.join()


class RestorePipeline:
    def __init__(self, adapter, torch_module, failed_status):
        self.a, self.torch, self.failed_status = adapter, torch_module, failed_status
        self.count = len(adapter.runtime.slots)
        self.ready = queue.Queue(maxsize=self.count)
        self.completions = [queue.Queue(maxsize=1) for _ in range(self.count)]
        self.condition = threading.Condition()
        self.busy = {}
        self.pending_pages = {}
        self.cancelled_rooms = set()
        self.unsettled_rooms = set()
        self.closed = False
        self._close_finished = False
        self.submit_thread = threading.Thread(
            target=self._submit_loop, name="kvserve-restore-submit", daemon=True
        )
        self.waiters = [
            threading.Thread(
                target=self._complete_loop,
                args=(slot,),
                name=f"kvserve-restore-complete-{slot}",
                daemon=True,
            )
            for slot in range(self.count)
        ]
        self.submit_thread.start()
        for thread in self.waiters:
            thread.start()

    def accept(self, metas):
        # Reserve all slots atomically, before enqueue, so malformed duplicate
        # READY cannot grow an executor queue or reuse scratch behind the GPU.
        with self.condition:
            if self.closed:
                raise RuntimeError("KVServe receive pipeline is closed")
            for meta in metas:
                if not isinstance(meta, dict):
                    raise ValueError("Invalid KVServe READY metadata")
                int(meta["room"])
                for key in ("sequence", "epoch"):
                    if not isinstance(meta[key], str):
                        raise ValueError("Invalid KVServe READY identity")
            slots = [meta.get("slot_id") for meta in metas]
            if any(
                type(slot) is not int or not 0 <= slot < self.count for slot in slots
            ):
                raise ValueError("Invalid KVServe receive slot")
            if len(set(slots)) != len(slots) or any(
                slot in self.busy for slot in slots
            ):
                raise ValueError("Duplicate or unsettled KVServe receive slot")
            for slot, meta in zip(slots, metas):
                self.busy[slot] = meta
            for meta in metas:
                self.ready.put_nowait((meta, time.perf_counter()))

    def wait_room(self, room):
        # No CUDA spin/query/sleep loop; completion threads wake this condition.
        with self.condition:
            if not self.condition.wait_for(
                lambda: all(int(meta["room"]) != room for meta in self.busy.values()),
                timeout=60,
            ):
                raise TimeoutError("KVServe room still has unsettled GPU restoration")

    def cancel_room(self, room):
        """Prevent new native-cache writes without waiting on the scheduler."""
        with self.condition:
            if room in self.a.manager.request_status or any(
                int(m["room"]) == room for m in self.busy.values()
            ):
                self.cancelled_rooms.add(room)
            else:
                self.cancelled_rooms.discard(room)

    def room_is_idle(self, room):
        # Only CPU bookkeeping. Event/stream waits belong to completion workers.
        with self.condition:
            return room not in self.unsettled_rooms and all(
                int(meta["room"]) != room for meta in self.busy.values()
            )

    def _error(self, room, result, error):
        result["error"] = str(error)
        self.a.poisoned = True
        self.a.manager.record_failure(room, str(error))
        self.a.manager.update_status(room, self.failed_status)

    def _submit_loop(self):
        while True:
            item = self.ready.get()
            if item is None:
                self.ready.task_done()
                return
            meta, queued_at = item
            slot = meta["slot_id"]
            job = dict(
                meta=meta,
                slot=slot,
                result={"sequence": meta["sequence"], "epoch": meta["epoch"]},
                room=int(meta["room"]),
                queued_at=queued_at,
                submitted=False,
                locked=False,
                reservation=None,
            )
            try:
                self._submit(job)
            except Exception as error:
                self._error(job["room"], job["result"], error)
            # Exactly one completion item per reserved slot; independent waiters
            # avoid a long slot delaying another already-finished slot's ACK.
            self.completions[slot].put_nowait(job)
            self.ready.task_done()

    def _submit(self, job):
        a, meta, room, slot = self.a, job["meta"], job["room"], job["slot"]
        if not a.slot_locks[slot].acquire(blocking=False):
            raise RuntimeError("KVServe receive slot has another restore owner")
        job["locked"] = True
        pages, ids, part = a._validate_restore(meta, slot)
        with self.condition:
            cancelled = room in self.cancelled_rooms
        if (
            cancelled
            or room not in a.manager.request_status
            or a.manager.check_status(room) == self.failed_status
        ):
            job["result"].update(cancelled=True, restore_ms=0)
            return
        with a.lock:
            expected, restored = a.expected_rooms.get(room), a.restored_rooms.get(room)
            key = (room, part)
            pending = self.pending_pages.get(key, set())
            if (
                expected is None
                or restored is None
                or not set(pages) <= expected
                or set(pages) & (restored[part] | pending)
            ):
                raise ValueError("Unregistered or duplicate KVServe destination pages")
            pending.update(pages)
            self.pending_pages[key] = pending
            job["reservation"] = key, set(pages), restored
        if a.verify and meta.get("digest") != a.digest(
            [meta["codec_bytes"], meta["aux_bytes"]], slot
        ):
            raise ValueError("KVServe payload digest mismatch")
        job.update(begin=time.perf_counter(), ids=ids, pages=pages, part=part)
        job["queue_ms"] = (job["begin"] - job["queued_at"]) * 1000
        with self.torch.cuda.device(a.device):
            # Set BEFORE decode: a synchronous submission error may follow some
            # enqueued GPU work. Completion drains the stream before any error ACK.
            job["submitted"] = True
            event = a.runtime.decode(
                slot,
                [ids],
                part,
                meta["codec_bytes"],
                meta["u8_bytes"],
                valid_tokens_by_group={0: len(ids)},
            )
            job["event"] = event
            job["event_recorded"] = True

    def _complete_loop(self, slot):
        while True:
            job = self.completions[slot].get()
            if job is None:
                self.completions[slot].task_done()
                return
            try:
                self._complete(job)
            finally:
                self.completions[slot].task_done()

    def _complete(self, job):
        a, slot, room, result = self.a, job["slot"], job["room"], job["result"]
        meta = job["meta"]
        gpu_drained = not job["submitted"]
        try:
            if job["submitted"]:
                with self.torch.cuda.device(a.device):
                    if job.get("event_recorded"):
                        job["event"].synchronize()
                    else:
                        a.runtime.streams[slot].synchronize()
                    gpu_drained = True
                    if "error" not in result:
                        result["restore_ms"] = (
                            time.perf_counter() - job["begin"]
                        ) * 1000
                        result["restore_gpu_ms"] = None
            with self.condition:
                cancelled = room in self.cancelled_rooms
            if cancelled:
                result["cancelled"] = True
            if job["reservation"] and "error" not in result and not cancelled:
                key, pages, restored = job["reservation"]
                with a.lock:
                    if a.restored_rooms.get(room) is not restored:
                        if room not in a.manager.request_status:
                            result["cancelled"] = True
                        else:
                            raise RuntimeError(
                                "KVServe room changed before GPU restoration completed"
                            )
                    else:
                        restored[key[1]].update(pages)
        except Exception as error:
            # A failed CUDA wait does not prove that writes stopped. Keep this
            # room's native pages quarantined even after a remote abort timeout.
            if not gpu_drained:
                with self.condition:
                    self.unsettled_rooms.add(room)
            self._error(room, result, error)
        finally:
            if job["reservation"]:
                key, pages, _ = job["reservation"]
                with a.lock:
                    self.pending_pages[key].difference_update(pages)
                    if not self.pending_pages[key]:
                        del self.pending_pages[key]
            if job["locked"]:
                a.slot_locks[slot].release()
        # ACK is queued only AFTER completion and page accounting, BEFORE file
        # logging. No poll interval, no batching window, no premature credit.
        with self.condition:
            del self.busy[slot]
            if room not in a.manager.request_status and all(
                int(m["room"]) != room for m in self.busy.values()
            ):
                self.cancelled_rooms.discard(room)
            self.condition.notify_all()
        try:
            a.reply(meta, result)
        except Exception as error:
            a.control_failed(str(error))
        if result.get("cancelled"):
            self._record(dict(event="restore_cancelled", room=room, **result))
        elif "error" not in result:
            self._record(
                dict(
                    event="restore",
                    room=room,
                    part=job["part"],
                    slot_id=slot,
                    tokens=len(job["ids"]),
                    wire_bytes=meta["codec_bytes"] + meta["aux_bytes"],
                    digest_verified=a.verify,
                    pipelined_restore=True,
                    restore_queue_ms=job["queue_ms"],
                    **result,
                )
            )

    def _record(self, value):
        try:
            self.a.record(value)
        except Exception as error:
            self.a.control_failed(str(error))

    def close(self):
        if threading.current_thread() in [self.submit_thread, *self.waiters]:
            raise RuntimeError("Restore worker cannot close its own pipeline")
        with self.condition:
            if self._close_finished:
                if self.unsettled_rooms:
                    raise RuntimeError(
                        "Unsettled CUDA restores retain their registered arenas"
                    )
                return
            if self.closed:
                raise RuntimeError(
                    "Restore close is already in progress; arenas stay registered"
                )
            self.closed = True
        self.ready.join()
        for completion in self.completions:
            completion.join()
        self.ready.put(None)
        for completion in self.completions:
            completion.put(None)
        self.submit_thread.join()
        for thread in self.waiters:
            thread.join()
        with self.condition:
            self._close_finished = True
        if self.unsettled_rooms:
            raise RuntimeError("Unsettled CUDA restores retain their registered arenas")


class BoundedSourcePool:
    """A bounded shared FIFO; each worker exclusively owns one Runtime slot.

    Workers draw both K and V work from the same FIFO. Slot IDs are never
    inferred from part IDs, and a slot is not reused before the callback returns.
    The callback must wait for GPU encode completion and read its live metadata.
    """

    def __init__(self, count, timeout=60):
        if type(count) is not int or count < 1:
            raise ValueError("Codec slot count must be positive")
        self.timeout = timeout
        self.queue = queue.Queue(maxsize=count)
        self.condition = threading.Condition()
        self.closed = False
        self.submitters = 0
        self.active = set()
        self._shutdown_started = False
        self._shutdown_done = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._run,
                args=(slot,),
                name=f"kvserve-encode-{slot}",
                daemon=True,
            )
            for slot in range(count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, function, *args, **kwargs):
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError("KVServe source pool is closed")
            self.submitters += 1
        try:
            # Do not hold the state lock while applying backpressure. Workers
            # need it when they finish a slot and accept the next queued frame.
            self.queue.put((function, args, kwargs, future), timeout=self.timeout)
        finally:
            with self.condition:
                self.submitters -= 1
                self.condition.notify_all()
        return future

    def _run(self, slot):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            function, args, kwargs, future = item
            with self.condition:
                if slot in self.active:
                    raise RuntimeError("Duplicate Runtime source-slot owner")
                self.active.add(slot)
            try:
                # No caller cancels a source future. Even an aborted room must
                # drain GPU submission/completion before its native pages retire.
                result = function(slot, *args, **kwargs)
                future.set_result(result)
            except BaseException as error:
                future.set_exception(error)
            finally:
                with self.condition:
                    self.active.remove(slot)
                    self.condition.notify_all()
                self.queue.task_done()

    def shutdown(self, wait=True):
        if not wait:
            raise ValueError("KVServe source owners must be drained before close")
        if threading.current_thread() in self._threads:
            raise RuntimeError("Source worker cannot close its own pool")
        with self.condition:
            if self._shutdown_started:
                owner = False
            else:
                self._shutdown_started = True
                self.closed = True
                owner = True
            if owner and not self.condition.wait_for(
                lambda: self.submitters == 0, timeout=self.timeout
            ):
                raise TimeoutError("Source submissions did not retire")
        if not owner:
            if not self._shutdown_done.wait(self.timeout):
                raise TimeoutError("Source pool shutdown is still in progress")
            return
        self.queue.join()
        for _ in self._threads:
            self.queue.put(None)
        for thread in self._threads:
            thread.join(self.timeout)
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError("Source workers did not exit")
        self._shutdown_done.set()
