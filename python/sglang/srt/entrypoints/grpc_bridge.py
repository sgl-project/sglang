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


class _CaseInsensitiveHeaders:
    """Minimal read-only header view with case-insensitive lookup.

    Mirrors the only Starlette ``Headers`` surface the OpenAI serving classes
    use (``.get(name)``), without importing Starlette or FastAPI.
    """

    __slots__ = ("_data",)

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self._data = {k.lower(): v for k, v in (headers or {}).items()}

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(name.lower(), default)


class MockRequest:
    """Stand-in for ``fastapi.Request`` when serving classes are called from gRPC.

    Implements the three surfaces the OpenAI serving classes actually touch:
      * ``headers.get(name)`` — case-insensitive, matches HTTP semantics
      * ``state.<attr>`` — attribute namespace for downstream mutations
      * ``await is_disconnected()`` — client-cancellation probe

    ``is_disconnected_fn`` is an optional sync callable the Rust side can pass
    that reads its Tonic ``CancellationToken``. When omitted, the request is
    treated as always-connected (current Rust behaviour).
    """

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

        # Ensure the handle_loop task exists before any gRPC RPC is dispatched.
        # auto_create_handle_loop is idempotent (no-op if the loop is already
        # running) and uses get_or_create_event_loop on the calling thread, so
        # constructing RuntimeHandle from the launcher's main thread pins the
        # loop where handle_loop's signal handlers expect it.
        self.tokenizer_manager.auto_create_handle_loop()
        self._event_loop = self.tokenizer_manager.event_loop

    @property
    def _tm_loop(self):
        """Return the tokenizer_manager's event loop.

        Communicator-based async methods (flush_cache, get_load, etc.) use
        asyncio.Event internally, which only works within a single event
        loop. These must run on the same loop as handle_loop(), i.e. the
        tokenizer_manager's event_loop. Cached at construction time so RPCs
        never race the loop's creation.
        """
        return self._event_loop

    def _safe_callback(self, chunk_callback, payload, **kwargs) -> None:
        try:
            chunk_callback(payload, **kwargs)
        except Exception as e:
            # Most often: Rust receiver dropped (client disconnect, channel
            # closed). Log at warning so it's visible without spamming on
            # every normal cancellation.
            logger.warning("gRPC chunk_callback failed: %s", e)

    def _submit_on_tm_loop(self, coro: Awaitable) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._tm_loop)
        future.add_done_callback(self._log_unhandled_future_exception)

    @staticmethod
    def _log_unhandled_future_exception(future) -> None:
        # All RuntimeHandle coroutines wrap their bodies in try/except and
        # route errors through chunk_callback, so this is a defence in depth:
        # if anything ever escapes (or a new caller forgets the wrap), we
        # surface it instead of silently hanging the gRPC stream.
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
        """Schedule a unary op whose result is JSON-encoded and sent via chunk_callback.

        ``payload_coro_factory`` is a zero-arg callable returning a coroutine
        that awaits the operation and returns the success payload (dict/list).
        ``error_payload_fn`` maps a caught exception to the payload sent on
        the error path; defaults to ``{"error": {"message": str(e)}}``,
        matching the OpenAI passthrough error shape.
        """
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
        """Submit a generate or embed request from a pre-built dict.

        The Rust gRPC server builds ``req_dict`` directly from proto fields,
        mapping them to GenerateReqInput / EmbeddingReqInput field names.
        Python just does ``**dict`` unpacking - no JSON parsing needed.

        Args:
            req_type: "generate" or "embed" (classify uses "embed").
            req_dict: Dict matching the dataclass constructor kwargs.
            chunk_callback: Rust-side PyO3 callback object.
            is_disconnected_fn: Optional sync callable wrapping the Rust
                cancellation token; lets TokenizerManager abort the request
                when the gRPC client drops.
        """
        mock_request = (
            MockRequest(is_disconnected_fn=is_disconnected_fn)
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
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            if stream:
                async for chunk in gen:
                    finished = (
                        chunk.get("meta_info", {}).get("finish_reason") is not None
                    )
                    self._safe_callback(chunk_callback, chunk, finished=finished)
                    if finished:
                        return
                # Defensive: generator exited without a finish_reason chunk.
                # Send a terminal callback so the Rust side closes the stream.
                self._safe_callback(chunk_callback, {}, finished=True)
            else:
                result = await gen.__anext__()
                self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            self._safe_callback(
                chunk_callback,
                json.dumps({"error": {"message": str(e)}}).encode("utf-8"),
                finished=True,
                error=str(e),
            )

    async def _run_embed(self, obj, chunk_callback, request):
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            result = await gen.__anext__()
            self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC embed error for rid=%s: %s", obj.rid, e)
            self._safe_callback(
                chunk_callback,
                json.dumps({"error": {"message": str(e)}}).encode("utf-8"),
                finished=True,
                error=str(e),
            )

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
        """Return model info as a JSON string."""
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
        """Return server info as a JSON string."""
        # dataclasses.asdict only walks declared fields, so dynamic attrs like
        # ServerArgs.model_config (set lazily by get_model_config) are
        # excluded automatically.
        result: Dict[str, Any] = dict(dataclasses.asdict(self.server_args))
        result.update(self.scheduler_info)
        return json.dumps(result, default=str)

    def health_check(self) -> bool:
        """Return True if the server is healthy."""
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        if self.tokenizer_manager.gracefully_exit:
            return False
        return self.tokenizer_manager.server_status not in (
            ServerStatus.Starting,
            ServerStatus.UnHealthy,
        )

    def tokenize(self, text: str, add_special_tokens: bool = True) -> str:
        """Tokenize text and return result as JSON string."""
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
        """Detokenize token IDs and return result as JSON string."""
        tokenizer = self.tokenizer_manager.tokenizer
        text = tokenizer.decode(tokens)
        return json.dumps({"text": text})

    def list_models(self) -> str:
        """Return the list of served models as JSON string."""
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
        """Return load info via chunk_callback."""

        async def _payload():
            result = await self.tokenizer_manager.get_loads(dp_rank=dp_rank)
            return [dataclasses.asdict(r) for r in result]

        self._submit_json_unary("get_load", _payload, chunk_callback)

    def flush_cache(self, chunk_callback) -> None:
        """Flush the radix cache. Sends result through chunk_callback."""

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
        """Pause generation. Sends result through chunk_callback."""

        async def _payload():
            from sglang.srt.managers.io_struct import PauseGenerationReqInput

            await self.tokenizer_manager.pause_generation(
                PauseGenerationReqInput(mode=mode)
            )
            return {"message": f"Generation paused (mode={mode})."}

        self._submit_json_unary("pause_generation", _payload, chunk_callback)

    def continue_generation(self, chunk_callback) -> None:
        """Continue generation. Sends result through chunk_callback."""

        async def _payload():
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput

            await self.tokenizer_manager.continue_generation(
                ContinueGenerationReqInput()
            )
            return {"message": "Generation continued."}

        self._submit_json_unary("continue_generation", _payload, chunk_callback)

    def start_profile(self, output_dir: Optional[str], chunk_callback) -> None:
        """Start profiling. Sends result through chunk_callback."""

        async def _payload():
            kwargs = {"output_dir": output_dir} if output_dir else {}
            await self.tokenizer_manager.start_profile(**kwargs)
            return {"message": "Profiling started."}

        self._submit_json_unary("start_profile", _payload, chunk_callback)

    def stop_profile(self, chunk_callback) -> None:
        """Stop profiling. Sends result through chunk_callback."""

        async def _payload():
            await self.tokenizer_manager.stop_profile()
            return {"message": "Profiling stopped."}

        self._submit_json_unary("stop_profile", _payload, chunk_callback)

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str], chunk_callback
    ) -> None:
        """Update weights from disk. Sends result through chunk_callback."""

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
        """Schedule an OpenAI pass-through request on the tokenizer_manager loop."""
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
        """Submit OpenAI chat completion (JSON pass-through, streaming)."""
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
        """Submit OpenAI completion (JSON pass-through, streaming)."""
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
        """Submit OpenAI embedding (JSON pass-through, unary)."""
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
        """Submit OpenAI classify (JSON pass-through, unary)."""
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
        """Submit OpenAI score (JSON pass-through, unary)."""
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
        """Submit OpenAI rerank (JSON pass-through, unary)."""
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
        """Generic OpenAI pass-through handler.

        Delegates to the appropriate OpenAIServing* class, sending response
        data back through the Rust chunk_callback.
        """
        try:
            serving = self._get_openai_serving()[serving_key]

            # JSON parse + Pydantic validation are client errors. Surface
            # them with status_code=400 so the Rust side maps them to
            # INVALID_ARGUMENT instead of INTERNAL. They always happen
            # before any stream chunks are emitted, so it is safe to send a
            # single status-bearing chunk even on streaming endpoints.
            try:
                request_dict = json.loads(json_body)
                request_cls = self._get_openai_request_class(serving_key)
                request_obj = request_cls(**request_dict)
            except (json.JSONDecodeError, ValidationError) as e:
                error_body = json.dumps(
                    {"error": {"message": str(e), "type": "BadRequest"}}
                ).encode("utf-8")
                self._safe_callback(
                    chunk_callback, error_body, finished=True, status_code=400
                )
                return

            mock_request = MockRequest(
                headers=trace_headers,
                is_disconnected_fn=is_disconnected_fn,
            )

            result = await serving.handle_request(request_obj, mock_request)

            if hasattr(result, "body_iterator"):
                # Parse SSE events per WHATWG spec (data:/event:/id: fields,
                # `:` comments, blank-line event boundaries). OpenAIServing*
                # currently emits one `data: <json>\n\n` per yield, but a
                # naive `startswith("data: ")` filter silently drops anything
                # else — multi-line data, comments, future event/id usage.
                # See follow-up: replace SSE round-trip with a (dict,
                # finished) hook on OpenAIServing*.
                data_buf: List[str] = []

                def _flush_event() -> None:
                    if not data_buf:
                        return
                    body = "\n".join(data_buf)
                    data_buf.clear()
                    if body == "[DONE]" or not body:
                        return
                    chunk_callback(body.encode("utf-8"), finished=False)

                async for raw_chunk in result.body_iterator:
                    if isinstance(raw_chunk, bytes):
                        raw_chunk = raw_chunk.decode("utf-8", errors="replace")
                    for line in raw_chunk.split("\n"):
                        line = line.rstrip("\r")
                        if not line:
                            _flush_event()
                        elif line.startswith(":"):
                            continue  # SSE comment / heartbeat
                        elif line.startswith("data:"):
                            value = line[5:]
                            if value.startswith(" "):
                                value = value[1:]
                            data_buf.append(value)
                        # event:, id:, retry:, unknown fields: ignored

                # Defensive: emit a trailing event if the stream ended
                # without a final blank line.
                _flush_event()
                chunk_callback(b"", finished=True)
            else:
                if hasattr(result, "model_dump"):
                    resp_bytes = json.dumps(result.model_dump()).encode("utf-8")
                elif hasattr(result, "body"):
                    resp_bytes = result.body
                elif isinstance(result, (dict, list)):
                    resp_bytes = json.dumps(result).encode("utf-8")
                else:
                    resp_bytes = str(result).encode("utf-8")
                # ErrorResponse uses ``code``; Starlette/FastAPI Response
                # uses ``status_code``. Honour whichever is present so error
                # responses don't ship as HTTP 200 with an error body.
                status_code = int(
                    getattr(result, "status_code", None)
                    or getattr(result, "code", None)
                    or 200
                )
                chunk_callback(resp_bytes, finished=True, status_code=status_code)

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
