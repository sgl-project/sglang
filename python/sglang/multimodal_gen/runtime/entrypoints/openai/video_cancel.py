"""Publishes the ids of deleted video jobs to the replica drivers, see managers/request_cancel.py."""

import os
from contextlib import suppress

import zmq

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CancelPublisher:
    """Sends each cancelled request id to every connected replica driver.

    A driver that is not connected when an id is sent does not receive it, so
    its request runs to the end, as before cancellation was supported.
    """

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.bind(endpoint)
        except zmq.ZMQError:
            self._context.destroy(linger=0)
            raise

    def publish(self, request_id: str) -> None:
        try:
            self._socket.send(request_id.encode(), flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logger.warning("Failed to publish cancel of %s: %s", request_id, e)

    def close(self) -> None:
        self._context.destroy(linger=0)
        with suppress(FileNotFoundError):
            os.unlink(self._endpoint.removeprefix("ipc://"))


_publisher: CancelPublisher | None = None


def start_cancel_publisher(endpoint: str) -> None:
    global _publisher
    try:
        _publisher = CancelPublisher(endpoint)
    except zmq.ZMQError as e:
        logger.warning(
            "Failed to bind cancel endpoint %s, request cancellation disabled: %s",
            endpoint,
            e,
        )


def stop_cancel_publisher() -> None:
    global _publisher
    if _publisher is not None:
        _publisher.close()
        _publisher = None


def publish_cancel(request_id: str) -> None:
    if _publisher is not None:
        _publisher.publish(request_id)
