"""
Asynchronous dynamic batch tokenizer for SGLang.

This module provides an async tokenizer with dynamic batching capabilities
to reduce tokenization overhead when multiple requests arrive concurrently.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncDynamicbatchTokenizer:
    """Asynchronous tokenizer with dynamic batching for single string prompts.

    Dynamically batches pending encode requests from a queue to reduce overhead.
    Only handles single string prompts - regular batch processing of multiple
    strings per request should be handled at a higher level.
    A single-thread ThreadPoolExecutor is used so the event loop stays responsive.

    Note: Uses lazy initialization for asyncio components because this class
    is instantiated in TokenizerManager.__init__() before the event loop starts.
    """

    def __init__(
        self,
        tokenizer,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.002,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s

        # Single queue for all encode requests - initialized lazily
        self._queue: Optional[asyncio.Queue] = None
        self._batcher_task: Optional[asyncio.Task] = None

        # Single-thread executor for blocking tokenizer calls
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of event loop dependent components."""
        if not self._initialized:
            self._queue = asyncio.Queue()
            self._batcher_task = asyncio.create_task(self._dynamic_batch_loop())
            self._initialized = True

    async def __call__(self, prompt: str, **kwargs) -> Any:
        """Encode a single prompt."""
        return await self.encode(prompt, **kwargs)

    async def encode(self, prompt: str, **kwargs) -> Any:
        """Encode a single prompt."""
        self._ensure_initialized()
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((prompt, kwargs, result_future))
        return await result_future

    async def _dynamic_batch_loop(self):
        """Dynamically batch incoming encode requests for efficiency."""
        while True:
            try:
                # Get the first request
                prompt, kwargs, result_future = await self._queue.get()

                # Collect requests into dynamic batch
                prompts = [prompt]
                kwargs_list = [kwargs]
                result_futures = [result_future]

                # Check if there are more items immediately available in the queue
                # If queue is empty, process single item immediately without timeout
                if self._queue.empty():
                    # No other requests waiting, process immediately
                    pass
                else:
                    # There might be more requests, wait for dynamic batching opportunity
                    start_time = asyncio.get_running_loop().time()

                    # Collect more requests up to max_batch_size or batch_wait_timeout_s
                    while len(prompts) < self.max_batch_size:
                        elapsed = asyncio.get_running_loop().time() - start_time
                        if elapsed >= self.batch_wait_timeout_s:
                            break

                        remaining_time = self.batch_wait_timeout_s - elapsed
                        try:
                            prompt, kwargs, result_future = await asyncio.wait_for(
                                self._queue.get(), remaining_time
                            )
                            prompts.append(prompt)
                            kwargs_list.append(kwargs)
                            result_futures.append(result_future)
                        except asyncio.TimeoutError:
                            break

                # Log dynamic batch information
                logger.debug(
                    f"AsyncDynamicbatchTokenizer: Processing dynamic batch of size {len(prompts)}"
                )

                # Process the dynamic batch
                await self._process_dynamic_batch(prompts, kwargs_list, result_futures)

            except Exception as e:
                logger.error(f"Error in dynamic batch loop: {e}")
                # Continue the loop to handle other requests

    async def _process_dynamic_batch(
        self,
        prompts: List[str],
        kwargs_list: List[Dict],
        result_futures: List[asyncio.Future],
    ) -> None:
        """Process a dynamic batch of encode requests for single string prompts."""
        # Check if all kwargs are identical for efficient batch processing
        can_batch = len(set(str(sorted(kw.items())) for kw in kwargs_list)) == 1
        kwargs = kwargs_list[0] if can_batch else None

        try:
            # If every request uses identical kwargs we can run a single
            # batch tokenizer call for a big speed-up.
            if can_batch and len(prompts) > 1:
                encode_fn = partial(self.tokenizer, prompts, **kwargs)
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, encode_fn
                )

                for i, fut in enumerate(result_futures):
                    if not fut.done():
                        data = {k: v[i] for k, v in results.items()}
                        fut.set_result(data)
            else:
                # Process each request individually due to different kwargs
                if len(prompts) > 1 and not can_batch:
                    logger.warning(
                        f"AsyncDynamicbatchTokenizer: Dynamic batching disabled for batch of {len(prompts)} "
                        f"requests due to differing kwargs. This reduces performance benefits. "
                        f"Consider using consistent tokenization parameters across requests."
                    )

                encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                    self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs_list)
                ]
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, encode_fn
                )

                for fut, res in zip(result_futures, results):
                    if not fut.done():
                        fut.set_result(res)
        except Exception as e:
            logger.error(f"Error in dynamic batch processing: {e}")
            for fut in result_futures:
                if not fut.done():
                    fut.set_exception(e)

    def __del__(self):
        """Clean up background tasks."""
        if hasattr(self, "_batcher_task") and self._batcher_task:
            if not self._batcher_task.done():
                self._batcher_task.cancel()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
