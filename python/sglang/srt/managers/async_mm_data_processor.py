import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AsyncMMDataProcessor:
    """
    Async wrapper for a multimodal processor.

    Behavior:
      - If the underlying processor exposes `process_mm_data_async`, call/await it directly.
      - Otherwise, fall back to running a synchronous `process_mm_data` in a thread pool.
      - Optionally guard per-call concurrency via an asyncio.Semaphore.
      - Optionally enforce per-call timeout via asyncio.wait_for.
    """

    def __init__(
        self,
        mm_processor: Any,
        *,
        max_concurrent_calls: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        """
        Args:
            mm_processor: An object exposing either
                - async def process_mm_data_async(...): -> Dict[str, Any]
              or
                - def process_mm_data(...): -> Dict[str, Any]
            max_concurrent_calls: Optional concurrency cap for per-call execution.
            timeout_s: Optional timeout (seconds) for each `process()` call.
        """
        self.mm_processor = mm_processor
        self.timeout_s = timeout_s

        # Concurrency guard (None -> unlimited)
        self.semaphore = (
            asyncio.Semaphore(max_concurrent_calls) if max_concurrent_calls else None
        )

        # Detect async path; if missing, prepare a fallback executor for sync path
        self._proc_async = getattr(mm_processor, "process_mm_data_async", None)
        self.is_async = asyncio.iscoroutinefunction(self._proc_async)
        self.fallback_exec: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=max_concurrent_calls)
            if not self.is_async
            else None
        )

    async def process(
        self,
        *,
        image_data: Optional[List[Union[str, bytes]]] = None,
        audio_data: Optional[List[Union[str, bytes]]] = None,
        input_text_or_ids: Union[str, List[int], None] = None,
        request_obj: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Public entrypoint: process a single multimodal request without blocking the event loop.
        """

        async def _invoke() -> Dict[str, Any]:
            if self.is_async:
                # Native async implementation
                return await self._proc_async(
                    image_data=image_data,
                    audio_data=audio_data,
                    input_text=input_text_or_ids,
                    request_obj=request_obj,
                    **kwargs,
                )

            # Synchronous fallback
            sync_fn = getattr(self.mm_processor, "process_mm_data", None)
            if not callable(sync_fn):
                raise RuntimeError(
                    "mm_processor has neither 'process_mm_data_async' nor 'process_mm_data'."
                )
            loop = asyncio.get_running_loop()
            fn = partial(
                sync_fn,
                image_data=image_data,
                audio_data=audio_data,
                input_text=input_text_or_ids,
                request_obj=request_obj,
                **kwargs,
            )
            return await loop.run_in_executor(self.fallback_exec, fn)

        # Apply optional concurrency guard
        if self.semaphore is not None:
            async with self.semaphore:
                if self.timeout_s is not None:
                    return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
                return await _invoke()

        # No concurrency guard
        if self.timeout_s is not None:
            return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
        return await _invoke()

    def shutdown(self) -> None:
        """Gracefully shutdown resources owned by this wrapper."""
        try:
            if self.fallback_exec:
                self.fallback_exec.shutdown(wait=False)
        except Exception:
            logger.exception(
                "Error while shutting down fallback executor in AsyncMMDataProcessor"
            )

    def __del__(self):
        # Best-effort shutdown
        try:
            self.shutdown()
        except Exception:
            pass
