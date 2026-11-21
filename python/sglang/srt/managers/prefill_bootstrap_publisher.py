"""Publish selected P ranks while their scheduler is still running a batch."""

import http.client
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PrefillBootstrapPublisher:
    def __init__(self, port: int, max_pending: int = 64):
        self.port = port
        self._slots = threading.BoundedSemaphore(max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="prefill-bootstrap"
        )

    def publish(self, room, rank):
        if type(room) is not int or type(rank) is not int or rank < 0:
            return False
        if not self._slots.acquire(blocking=False):
            return False
        try:
            future = self._executor.submit(self._post, room, rank)
        except RuntimeError:
            self._slots.release()
            return False
        future.add_done_callback(lambda _: self._slots.release())
        return True

    def _post(self, room: int, rank: int):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            body = json.dumps(
                {"bootstrap_room": room, "dp_rank": rank, "advisory": True}
            )
            connection.request(
                "POST", "/register_dp_rank", body, {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            response.read()
            if response.status != 200:
                logger.debug(
                    "Early prefill registration returned HTTP %s", response.status
                )
        except (OSError, http.client.HTTPException):
            logger.debug("Early prefill registration failed", exc_info=True)
        finally:
            connection.close()
