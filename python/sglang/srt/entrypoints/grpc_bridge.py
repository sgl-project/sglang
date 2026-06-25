"""Python-side bridge between the Rust gRPC server and TokenizerManager.

The RuntimeHandle exposes synchronous methods that Rust can call via PyO3
(with a brief GIL acquisition). Response chunks are pushed into Rust-side
channels via callback objects while all async work stays on the
TokenizerManager's event loop.
"""

import asyncio
import dataclasses
import json
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import ValidationError

logger = logging.getLogger(__name__)


class _BadOpenAIRequest(ValueError):
    pass


class _CaseInsensitiveHeaders:
    __slots__ = ("_data",)

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self._data = {k.lower(): v for k, v in (headers or {}).items()}

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(name.lower(), default)


class _GrpcRequest:
    """Small FastAPI Request shim used by OpenAIServing* and TokenizerManager."""

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        self.headers = _CaseInsensitiveHeaders(headers)
        self.state = SimpleNamespace()
        self._is_disconnected_fn = is_disconnected_fn

    async def is_disconnected(self) -> bool:
        if self._is_disconnected_fn is None:
            return False
        return bool(self._is_disconnected_fn())


class RuntimeHandle:
    """Thin Python handle that the Rust gRPC server calls into.

    Provides synchronous ``submit_*``, ``abort``, and info methods.
    Each submit method receives a ``chunk_callback`` (a Rust-side PyO3 object)
    that it invokes with ``(chunk_dict, finished, error)`` for each response
    chunk produced by TokenizerManager.
    """

    def __init__(
        self,
        tokenizer_manager,
        template_manager,
        server_args,
        scheduler_info: Optional[Dict] = None,
    ):
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.server_args = server_args
        self.scheduler_info = scheduler_info or {}

        self._openai_serving_classes = None

        self.tokenizer_manager.auto_create_handle_loop()
        self._event_loop = self.tokenizer_manager.event_loop

    @property
    def _tm_loop(self):
        """Return the TokenizerManager loop used by communicator RPCs."""
        return self._event_loop

    def _safe_callback(self, chunk_callback, payload, **kwargs):
        """Invoke a Rust callback and return its ChunkSendStatus, if any."""
        try:
            return chunk_callback(payload, **kwargs)
        except Exception as e:
            logger.warning("gRPC chunk_callback failed: %s", e)
            return None

    def _send_native_error(self, chunk_callback, message: str):
        # ChunkCallback extracts the PyDict arg before reading error=.
        return self._safe_callback(chunk_callback, {}, finished=True, error=message)

    _BACKPRESSURE_TIMEOUT_S = 300.0

    @staticmethod
    def _is_pending_status(status) -> bool:
        return status is not None and status == type(status).Pending

    @staticmethod
    def _is_closed_status(status) -> bool:
        return status is not None and status == type(status).Closed

    def _abort_request_id(self, rid) -> None:
        if isinstance(rid, list):
            for single_rid in rid:
                self.tokenizer_manager.abort_request(rid=single_rid)
        else:
            self.tokenizer_manager.abort_request(rid=rid)

    async def _send_with_backpressure(
        self,
        chunk_callback,
        ready_event: Optional[asyncio.Event],
        payload,
        *,
        timeout_abort_rid=None,
        **kwargs,
    ) -> bool:
        status = self._safe_callback(chunk_callback, payload, **kwargs)
        if status is None or self._is_closed_status(status):
            return False
        if not self._is_pending_status(status):
            return True

        if kwargs.get("finished"):
            return True
        if ready_event is None:
            return True

        try:
            await asyncio.wait_for(
                ready_event.wait(), timeout=self._BACKPRESSURE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            if timeout_abort_rid is not None:
                self._abort_request_id(timeout_abort_rid)
                logger.warning(
                    "gRPC chunk backpressure wait timed out after %ss; aborted request",
                    self._BACKPRESSURE_TIMEOUT_S,
                )
            else:
                logger.warning(
                    "gRPC chunk backpressure wait timed out after %ss; closing stream",
                    self._BACKPRESSURE_TIMEOUT_S,
                )
            return False
        ready_event.clear()
        return True

    def _install_on_ready(self, chunk_callback) -> Optional[asyncio.Event]:
        set_on_ready = getattr(chunk_callback, "set_on_ready", None)
        if set_on_ready is None:
            return None
        ready_event = asyncio.Event()
        loop = self._tm_loop

        def _on_ready() -> None:
            loop.call_soon_threadsafe(ready_event.set)

        try:
            set_on_ready(_on_ready)
        except Exception as e:
            logger.warning("gRPC set_on_ready failed: %s", e)
            raise
        return ready_event

    @staticmethod
    def _uninstall_on_ready(chunk_callback) -> None:
        clear = getattr(chunk_callback, "clear_on_ready", None)
        if clear is None:
            return
        try:
            clear()
        except Exception as e:
            logger.warning("gRPC clear_on_ready failed: %s", e)

    def _submit_on_tm_loop(self, coro: Awaitable) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._tm_loop)
        future.add_done_callback(self._log_unhandled_future_exception)

    @staticmethod
    def _log_unhandled_future_exception(future) -> None:
        try:
            future.result()
        except Exception as e:
            logger.error(
                "gRPC scheduled coroutine raised unhandled exception: %s",
                e,
                exc_info=True,
            )

    def _submit_json_unary(
        self,
        op_name: str,
        payload_coro_factory: Callable[[], Awaitable[Any]],
        chunk_callback,
        *,
        error_payload_fn: Optional[Callable[[Exception], Any]] = None,
    ) -> None:
        error_fn = error_payload_fn or (lambda e: {"error": {"message": str(e)}})

        async def _run() -> None:
            try:
                payload = await payload_coro_factory()
                self._safe_callback(
                    chunk_callback,
                    json.dumps(payload, default=str).encode("utf-8"),
                    finished=True,
                )
            except Exception as e:
                logger.error("gRPC %s error: %s", op_name, e)
                self._safe_callback(
                    chunk_callback,
                    json.dumps(error_fn(e), default=str).encode("utf-8"),
                    finished=True,
                    error=str(e),
                )

        self._submit_on_tm_loop(_run())

    def _get_openai_serving(self):
        """Lazily initialize OpenAI serving classes."""
        if self._openai_serving_classes is not None:
            return self._openai_serving_classes

        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.entrypoints.openai.serving_classify import (
            OpenAIServingClassify,
        )
        from sglang.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )
        from sglang.srt.entrypoints.openai.serving_embedding import (
            OpenAIServingEmbedding,
        )
        from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
        from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore

        self._openai_serving_classes = {
            "chat": OpenAIServingChat(self.tokenizer_manager, self.template_manager),
            "completion": OpenAIServingCompletion(
                self.tokenizer_manager, self.template_manager
            ),
            "embedding": OpenAIServingEmbedding(
                self.tokenizer_manager, self.template_manager
            ),
            "classify": OpenAIServingClassify(
                self.tokenizer_manager, self.template_manager
            ),
            "score": OpenAIServingScore(self.tokenizer_manager),
            "rerank": OpenAIServingRerank(
                self.tokenizer_manager, self.template_manager
            ),
        }
        return self._openai_serving_classes

    def submit_request(
        self,
        *,
        req_type: str,
        req_dict: dict,
        chunk_callback,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        mock_request = (
            _GrpcRequest(is_disconnected_fn=is_disconnected_fn)
            if is_disconnected_fn is not None
            else None
        )
        if req_type == "generate":
            from sglang.srt.managers.io_struct import GenerateReqInput

            obj = GenerateReqInput(**req_dict)
            stream = req_dict.get("stream", False)
            self._submit_on_tm_loop(
                self._run_generate(obj, chunk_callback, stream, mock_request)
            )
        elif req_type == "embed":
            from sglang.srt.managers.io_struct import EmbeddingReqInput

            obj = EmbeddingReqInput(**req_dict)
            self._submit_on_tm_loop(self._run_embed(obj, chunk_callback, mock_request))
        else:
            raise ValueError(
                f"Unknown req_type: {req_type!r} (expected 'generate' or 'embed')"
            )

    async def _run_generate(self, obj, chunk_callback, stream: bool, request):
        ready_event = None
        gen = None
        try:
            ready_event = self._install_on_ready(chunk_callback) if stream else None
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            if stream:
                async for chunk in gen:
                    finished = (
                        chunk.get("meta_info", {}).get("finish_reason") is not None
                    )
                    keep_going = await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        chunk,
                        finished=finished,
                        timeout_abort_rid=obj.rid,
                    )
                    if finished or not keep_going:
                        return
                # Defensive: generator exited without a finish_reason chunk.
                self._safe_callback(chunk_callback, {}, finished=True)
            else:
                result = await gen.__anext__()
                self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            self._send_native_error(chunk_callback, str(e))
        finally:
            if gen is not None:
                await gen.aclose()
            if stream:
                self._uninstall_on_ready(chunk_callback)

    async def _run_embed(self, obj, chunk_callback, request):
        gen = None
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            result = await gen.__anext__()
            self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC embed error for rid=%s: %s", obj.rid, e)
            self._send_native_error(chunk_callback, str(e))
        finally:
            if gen is not None:
                await gen.aclose()

    # Bounded so a stuck TM loop can't deadlock the gRPC handler thread that
    # called abort. abort_request only enqueues a message on the ZMQ socket,
    # so a few seconds is generous; if we time out, log and drop — the client
    # will retry or give up.
    _ABORT_TIMEOUT_S = 5.0

    def abort(self, rid: str = "", abort_all: bool = False):
        """Abort a request by request ID or abort all active requests."""
        loop = self._tm_loop

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
            return

        future = asyncio.run_coroutine_threadsafe(
            self._abort_async(rid, abort_all),
            loop,
        )
        try:
            future.result(timeout=self._ABORT_TIMEOUT_S)
        except TimeoutError:
            future.cancel()
            logger.error(
                "gRPC abort timed out after %ss (rid=%r, abort_all=%s); "
                "tokenizer_manager loop appears stuck",
                self._ABORT_TIMEOUT_S,
                rid,
                abort_all,
            )

    async def _abort_async(self, rid: str, abort_all: bool) -> None:
        self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

    def get_model_info(self) -> str:
        model_config = self.tokenizer_manager.model_config
        result = {
            "model_path": self.tokenizer_manager.model_path,
            "tokenizer_path": self.server_args.tokenizer_path,
            "is_generation": self.tokenizer_manager.is_generation,
            "weight_version": self.server_args.weight_version,
            "model_type": getattr(model_config.hf_config, "model_type", None),
            "architectures": getattr(model_config.hf_config, "architectures", None),
        }
        return json.dumps(result, default=str)

    def get_server_info(self) -> str:
        result: Dict[str, Any] = dict(dataclasses.asdict(self.server_args))
        result.update(self.scheduler_info)
        return json.dumps(result, default=str)

    def health_check(self) -> bool:
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        if self.tokenizer_manager.gracefully_exit:
            return False
        return self.tokenizer_manager.server_status not in (
            ServerStatus.Starting,
            ServerStatus.UnHealthy,
        )

    def tokenize(self, text: str, add_special_tokens: bool = True) -> str:
        tokenizer = self.tokenizer_manager.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        result = {
            "tokens": tokens,
            "count": len(tokens),
            "max_model_len": self.tokenizer_manager.model_config.context_len,
            "input_text": text,
        }
        return json.dumps(result)

    def detokenize(self, tokens: List[int]) -> str:
        tokenizer = self.tokenizer_manager.tokenizer
        text = tokenizer.decode(tokens)
        return json.dumps({"text": text})

    def list_models(self) -> str:
        served_model_name = self.tokenizer_manager.served_model_name
        models = [
            {
                "id": served_model_name,
                "root": served_model_name,
                "max_model_len": self.tokenizer_manager.model_config.context_len,
            }
        ]
        if self.server_args.enable_lora and hasattr(
            self.tokenizer_manager, "lora_registry"
        ):
            lora_registry = self.tokenizer_manager.lora_registry
            for _, lora_ref in lora_registry.get_all_adapters().items():
                models.append(
                    {
                        "id": lora_ref.lora_name,
                        "root": lora_ref.lora_path,
                        "parent": served_model_name,
                    }
                )
        return json.dumps(models)

    def get_load(self, chunk_callback, dp_rank: Optional[int] = None) -> None:
        async def _payload():
            result = await self.tokenizer_manager.get_loads(dp_rank=dp_rank)
            return [r.to_dict() for r in result]

        self._submit_json_unary("get_load", _payload, chunk_callback)

    def flush_cache(self, chunk_callback) -> None:
        async def _payload():
            ret = await self.tokenizer_manager.flush_cache()
            return {"success": ret.success, "message": "Cache flushed."}

        self._submit_json_unary(
            "flush_cache",
            _payload,
            chunk_callback,
            error_payload_fn=lambda e: {"success": False, "message": str(e)},
        )

    def pause_generation(self, mode: str, chunk_callback) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import PauseGenerationReqInput

            await self.tokenizer_manager.pause_generation(
                PauseGenerationReqInput(mode=mode)
            )
            return {"message": f"Generation paused (mode={mode})."}

        self._submit_json_unary("pause_generation", _payload, chunk_callback)

    def continue_generation(self, chunk_callback) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput

            await self.tokenizer_manager.continue_generation(
                ContinueGenerationReqInput()
            )
            return {"message": "Generation continued."}

        self._submit_json_unary("continue_generation", _payload, chunk_callback)

    def start_profile(self, output_dir: Optional[str], chunk_callback) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import ProfileReq

            req = ProfileReq(output_dir=output_dir) if output_dir else ProfileReq()
            await self.tokenizer_manager.start_profile(req)
            return {"message": "Profiling started."}

        self._submit_json_unary("start_profile", _payload, chunk_callback)

    def stop_profile(self, chunk_callback) -> None:
        async def _payload():
            await self.tokenizer_manager.stop_profile()
            return {"message": "Profiling stopped."}

        self._submit_json_unary("stop_profile", _payload, chunk_callback)

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str], chunk_callback
    ) -> None:
        async def _payload():
            from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

            obj = UpdateWeightFromDiskReqInput(
                model_path=model_path, load_format=load_format
            )
            success, message, num_paused = (
                await self.tokenizer_manager.update_weights_from_disk(obj, request=None)
            )
            return {
                "success": success,
                "message": message,
                "num_paused_requests": num_paused,
            }

        self._submit_json_unary(
            "update_weights",
            _payload,
            chunk_callback,
            error_payload_fn=lambda e: {"success": False, "message": str(e)},
        )

    def _submit_openai(
        self,
        serving_key: str,
        streaming: bool,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]],
        is_disconnected_fn: Optional[Callable[[], bool]],
    ) -> None:
        self._submit_on_tm_loop(
            self._run_openai_request(
                serving_key,
                json_body,
                chunk_callback,
                streaming=streaming,
                trace_headers=trace_headers,
                is_disconnected_fn=is_disconnected_fn,
            )
        )

    def submit_openai_chat(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "chat", True, json_body, chunk_callback, trace_headers, is_disconnected_fn
        )

    def submit_openai_complete(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "completion",
            True,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def submit_openai_embed(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "embedding",
            False,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def submit_openai_classify(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "classify",
            False,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def submit_openai_score(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "score", False, json_body, chunk_callback, trace_headers, is_disconnected_fn
        )

    def submit_openai_rerank(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._submit_openai(
            "rerank",
            False,
            json_body,
            chunk_callback,
            trace_headers,
            is_disconnected_fn,
        )

    def _get_openai_request_class(self, serving_key: str):
        """Return the Pydantic request class for a given serving key."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ClassifyRequest,
            CompletionRequest,
            EmbeddingRequest,
            ScoringRequest,
            V1RerankReqInput,
        )

        return {
            "chat": ChatCompletionRequest,
            "completion": CompletionRequest,
            "embedding": EmbeddingRequest,
            "classify": ClassifyRequest,
            "score": ScoringRequest,
            "rerank": V1RerankReqInput,
        }[serving_key]

    async def _run_openai_request(
        self,
        serving_key: str,
        json_body: bytes,
        chunk_callback,
        streaming: bool,
        trace_headers: Optional[Dict[str, str]] = None,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        try:
            serving = self._get_openai_serving()[serving_key]

            try:
                request_dict = json.loads(json_body)
                if not isinstance(request_dict, dict):
                    raise _BadOpenAIRequest(
                        f"Request body must be a JSON object, got {type(request_dict).__name__}"
                    )
                request_cls = self._get_openai_request_class(serving_key)
                request_obj = request_cls(**request_dict)
            except (json.JSONDecodeError, ValidationError, _BadOpenAIRequest) as e:
                error_body = json.dumps(
                    {"error": {"message": str(e), "type": "BadRequest"}}
                ).encode("utf-8")
                if streaming:
                    self._safe_callback(
                        chunk_callback, error_body, finished=True, error=str(e)
                    )
                else:
                    self._safe_callback(
                        chunk_callback, error_body, finished=True, status_code=400
                    )
                return

            mock_request = _GrpcRequest(
                headers=trace_headers,
                is_disconnected_fn=is_disconnected_fn,
            )

            result = await serving.handle_request(request_obj, mock_request)

            if hasattr(result, "body_iterator"):
                ready_event = self._install_on_ready(chunk_callback)
                data_buf: List[str] = []
                stream_closed = False

                async def _flush_event() -> bool:
                    """Flush buffered SSE data lines as one chunk. Returns False if Rust closed."""
                    if not data_buf:
                        return True
                    body = "\n".join(data_buf)
                    data_buf.clear()
                    if body == "[DONE]" or not body:
                        return True
                    return await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        body.encode("utf-8"),
                        finished=False,
                    )

                try:
                    async for raw_chunk in result.body_iterator:
                        if isinstance(raw_chunk, bytes):
                            raw_chunk = raw_chunk.decode("utf-8", errors="replace")
                        for line in raw_chunk.split("\n"):
                            line = line.rstrip("\r")
                            if not line:
                                if not await _flush_event():
                                    stream_closed = True
                                    break
                            elif line.startswith(":"):
                                continue  # SSE comment / heartbeat
                            elif line.startswith("data:"):
                                value = line[5:]
                                if value.startswith(" "):
                                    value = value[1:]
                                data_buf.append(value)
                            # event:, id:, retry:, unknown fields: ignored
                        if stream_closed:
                            break

                    if not stream_closed:
                        await _flush_event()
                        self._safe_callback(chunk_callback, b"", finished=True)
                finally:
                    self._uninstall_on_ready(chunk_callback)
            else:
                if hasattr(result, "model_dump"):
                    resp_bytes = json.dumps(result.model_dump()).encode("utf-8")
                elif hasattr(result, "body"):
                    resp_bytes = result.body
                elif isinstance(result, (dict, list)):
                    resp_bytes = json.dumps(result).encode("utf-8")
                else:
                    resp_bytes = str(result).encode("utf-8")
                status_code = int(
                    getattr(result, "status_code", None)
                    or getattr(result, "code", None)
                    or 200
                )
                self._safe_callback(
                    chunk_callback,
                    resp_bytes,
                    finished=True,
                    status_code=status_code,
                )

        except Exception as e:
            logger.error("gRPC OpenAI %s error: %s", serving_key, e)
            error_body = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            if streaming:
                self._safe_callback(
                    chunk_callback, error_body, finished=True, error=str(e)
                )
            else:
                self._safe_callback(
                    chunk_callback,
                    error_body,
                    finished=True,
                    status_code=int(getattr(e, "status_code", 500)),
                )
