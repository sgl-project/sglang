"""Utilities for SSL certificate hot-reloading."""

import asyncio
import logging
import ssl
from typing import Optional

from watchfiles import awatch

logger = logging.getLogger(__name__)


class SSLCertRefresher:
    """Monitors SSL certificate files and reloads them when changed.

    Uses ``watchfiles.awatch()`` for efficient inotify/kqueue-based
    file monitoring.  On change the referenced :class:`ssl.SSLContext`
    is updated in-place so that new TLS connections automatically pick
    up the fresh certificates.
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext,
        key_path: str,
        cert_path: str,
        ca_path: Optional[str] = None,
    ) -> None:
        self._ssl_context = ssl_context
        self._key_path = key_path
        self._cert_path = cert_path
        self._ca_path = ca_path
        self._tasks: list[asyncio.Task] = []

        loop = asyncio.get_running_loop()
        self._tasks.append(
            loop.create_task(self._watch_cert_key(), name="ssl-cert-key-watcher")
        )
        if self._ca_path:
            self._tasks.append(
                loop.create_task(self._watch_ca(), name="ssl-ca-watcher")
            )

    async def _watch_cert_key(self) -> None:
        """Watch cert and key files and reload on change."""
        try:
            async for _changes in awatch(self._cert_path, self._key_path):
                logger.info(
                    "SSL cert/key file change detected, reloading: " "cert=%s key=%s",
                    self._cert_path,
                    self._key_path,
                )
                try:
                    self._ssl_context.load_cert_chain(self._cert_path, self._key_path)
                    logger.info("SSL cert/key reloaded successfully.")
                except Exception:
                    logger.exception(
                        "Failed to reload SSL cert/key — continuing with "
                        "previous certificates."
                    )
        except asyncio.CancelledError:
            return

    async def _watch_ca(self) -> None:
        """Watch CA file and reload on change."""
        assert self._ca_path is not None
        try:
            async for _changes in awatch(self._ca_path):
                logger.info(
                    "SSL CA file change detected, reloading: ca=%s",
                    self._ca_path,
                )
                try:
                    self._ssl_context.load_verify_locations(self._ca_path)
                    logger.info("SSL CA certificates reloaded successfully.")
                except Exception:
                    logger.exception(
                        "Failed to reload SSL CA certificates — continuing "
                        "with previous CA bundle."
                    )
        except asyncio.CancelledError:
            return

    def stop(self) -> None:
        """Cancel all watching tasks."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
