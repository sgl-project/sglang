"""Publishes the ids of deleted video jobs to the replica drivers, see managers/request_cancel.py."""

import os
from contextlib import suppress

import zmq


class CancelDeliveryError(RuntimeError):
    """The caller must retain the job and retry cancellation."""


class CancelPublisher:
    """Enqueue cancellations for all replica drivers, or reject for retry."""

    def __init__(self, endpoint: str, replicas: int = 1):
        self._endpoint = endpoint
        self._replicas = replicas
        self._subscribers = 0
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.XPUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        # cancellation is a control command: backpressure must not silently drop it
        self._socket.setsockopt(zmq.XPUB_NODROP, 1)
        self._socket.setsockopt(zmq.XPUB_VERBOSER, 1)
        try:
            self._socket.bind(endpoint)
        except zmq.ZMQError:
            self._context.destroy(linger=0)
            raise

    def publish(self, request_id: str) -> None:
        try:
            while self._socket.poll(0, zmq.POLLIN):
                subscribed = self._socket.recv(zmq.NOBLOCK)
                self._subscribers += 1 if subscribed[0] else -1
            if self._subscribers < self._replicas:
                raise CancelDeliveryError("Not all replica drivers are connected")
            self._socket.send(request_id.encode(), flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            raise CancelDeliveryError(
                "Cancellation channel is unavailable or full"
            ) from e

    def close(self) -> None:
        self._context.destroy(linger=0)
        with suppress(FileNotFoundError):
            os.unlink(self._endpoint.removeprefix("ipc://"))


_publisher: CancelPublisher | None = None


def start_cancel_publisher(endpoint: str, replicas: int = 1) -> None:
    global _publisher
    _publisher = CancelPublisher(endpoint, replicas)


def stop_cancel_publisher() -> None:
    global _publisher
    if _publisher is not None:
        _publisher.close()
        _publisher = None


def publish_cancel(request_id: str) -> None:
    if _publisher is not None:
        _publisher.publish(request_id)
