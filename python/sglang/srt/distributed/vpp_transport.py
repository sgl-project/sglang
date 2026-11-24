"""Scheduling FIFO and send credits using TP-sliced P2P activations."""

import threading
from contextlib import nullcontext

import torch


class P2PWorkGroup:
    """Retain send payloads until CPU waits and device work have completed."""

    def __init__(self, items):
        self.items = items
        self._error = None
        self._done = threading.Event()
        self._cpu = [
            item
            for item in items
            if isinstance(item.payload, torch.Tensor) and item.payload.is_cpu
        ]
        threading.Thread(target=self._wait_cpu, daemon=True).start()

    def _wait_cpu(self):
        try:
            for item in self._cpu:
                item.wait()
        except BaseException as error:
            self._error = error
        finally:
            self._done.set()

    def poll(self):
        if not self._done.is_set():
            return False
        if self._error is not None:
            raise self._error
        cpu_ids = {id(item) for item in self._cpu}
        device_items = [item for item in self.items if id(item) not in cpu_ids]
        if any(
            item.work is not None and not item.work.is_completed()
            for item in device_items
        ):
            return False
        for item in device_items:
            item.wait()
            if isinstance(item.payload, torch.Tensor) and item.payload.is_cuda:
                item.payload.record_stream(torch.cuda.current_stream())
        self.items.clear()
        return True

    def payload_bytes(self):
        return sum(
            item.payload.numel() * item.payload.element_size()
            for item in self.items
            if isinstance(item.payload, torch.Tensor)
        )


class VPPSchedulerActivationReceiver:
    """Receive a FIFO on each TP lane and reconstruct on a dedicated group."""

    def __init__(self, pp_group, tp_group, max_inflight, *, batch_p2p, tag):
        self.pp_group = pp_group
        self.tp_group = tp_group
        self.batch_p2p = batch_p2p
        self.tag = tag
        self._handle = None
        self._receive_ready = False
        self._stream = None
        self._module = None

    def _context(self):
        return (
            self._module.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        )

    def start(self):
        device = torch.device(self.pp_group.device)
        if device.type != "cpu":
            self._module = torch.get_device_module(device)
            self._stream = self._module.Stream(device=device)
        self._post_metadata()

    def _post_metadata(self):
        with self._context():
            self._handle = self.pp_group.recv_tensor_dict_async(
                tag=self.tag,
                all_gather_group=self.tp_group,
                batch_p2p=self.batch_p2p,
            )

    def progress(self):
        with torch.no_grad(), self._context():
            self._receive_ready = self._handle.poll_metadata_ready()

    def receive_count(self):
        return int(self._receive_ready)

    def take(self):
        with torch.no_grad(), self._context():
            result = self._handle.wait()
            if self._stream is not None:
                event = self._module.Event()
                event.record(self._stream)
                result["_vpp_local_recv_event"] = event
            self._receive_ready = False
            self._post_metadata()
            return result
