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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MockRequest:
    """Lightweight stand-in for fastapi.Request when called from gRPC.

    The OpenAI serving classes expect a fastapi.Request for disconnect
    detection and routing key extraction. This satisfies that interface
    without importing FastAPI.
    """

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.headers = headers or {}
        self.url = type("URL", (), {"path": "/grpc"})()
        self.state = type("State", (), {})()
        self.app = type("App", (), {"state": type("AppState", (), {})()})()

    async def is_disconnected(self) -> bool:
        return False


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

    @property
    def _tm_loop(self):
        """Return the tokenizer_manager's event loop (uvicorn loop).

        Communicator-based async methods (flush_cache, get_load, etc.) use
        asyncio.Event internally, which only works within a single event
        loop. These must run on the same loop as handle_loop(), i.e. the
        tokenizer_manager's event_loop.
        """
        loop = getattr(self.tokenizer_manager, "event_loop", None)
        if loop is None:
            raise RuntimeError(
                "TokenizerManager event loop not ready - server still starting"
            )
        return loop

    def _safe_callback(self, chunk_callback, payload, *, finished: bool, error=None):
        try:
            chunk_callback(payload, finished=finished, error=error)
        except Exception:
            pass

    def _submit_on_tm_loop(self, coro_factory, chunk_callback, *, empty_response):
        try:
            loop = self._tm_loop
        except RuntimeError as e:
            self._safe_callback(
                chunk_callback, empty_response, finished=True, error=str(e)
            )
            return
        asyncio.run_coroutine_threadsafe(coro_factory(), loop)

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

    # ------------------------------------------------------------------
    # Consolidated request submission (generate / embed / classify)
    # ------------------------------------------------------------------

    def submit_request(self, *, req_type: str, req_dict: dict, chunk_callback):
        """Submit a generate or embed request from a pre-built dict.

        The Rust gRPC server builds ``req_dict`` directly from proto fields,
        mapping them to GenerateReqInput / EmbeddingReqInput field names.
        Python just does ``**dict`` unpacking - no JSON parsing needed.

        Args:
            req_type: "generate" or "embed" (classify uses "embed").
            req_dict: Dict matching the dataclass constructor kwargs.
            chunk_callback: Rust-side PyO3 callback object.
        """
        if req_type == "generate":
            from sglang.srt.managers.io_struct import GenerateReqInput

            obj = GenerateReqInput(**req_dict)
            stream = req_dict.get("stream", False)
            self._submit_on_tm_loop(
                lambda: self._run_generate(obj, chunk_callback, stream),
                chunk_callback,
                empty_response={},
            )
        else:
            from sglang.srt.managers.io_struct import EmbeddingReqInput

            obj = EmbeddingReqInput(**req_dict)
            self._submit_on_tm_loop(
                lambda: self._run_embed(obj, chunk_callback),
                chunk_callback,
                empty_response={},
            )

    async def _run_generate(self, obj, chunk_callback, stream: bool):
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=None)
            if stream:
                async for chunk in gen:
                    finished = (
                        chunk.get("meta_info", {}).get("finish_reason") is not None
                    )
                    chunk_callback(chunk, finished=finished)
                    if finished:
                        return
            else:
                result = await gen.__anext__()
                chunk_callback(result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            self._safe_callback(chunk_callback, {}, finished=True, error=str(e))

    async def _run_embed(self, obj, chunk_callback):
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=None)
            result = await gen.__anext__()
            chunk_callback(result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC embed error for rid=%s: %s", obj.rid, e)
            self._safe_callback(chunk_callback, {}, finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def abort(self, rid: str = "", abort_all: bool = False):
        """Abort a request by request ID or abort all active requests."""
        try:
            loop = self._tm_loop
        except RuntimeError:
            self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
            return

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
        future.result()

    async def _abort_async(self, rid: str, abort_all: bool) -> None:
        self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

    # ------------------------------------------------------------------
    # Info RPCs (synchronous, small data)
    # ------------------------------------------------------------------

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
        result: Dict[str, Any] = {}
        try:
            sa = self.server_args
            if hasattr(sa, "model_config"):
                sa = dataclasses.replace(sa)
                if hasattr(sa, "model_config"):
                    delattr(sa, "model_config")
            result.update(dataclasses.asdict(sa))
        except Exception:
            pass
        result.update(self.scheduler_info)
        return json.dumps(result, default=str)

    def health_check(self) -> bool:
        """Return True if the server is healthy."""
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        if self.tokenizer_manager.gracefully_exit:
            return False
        if self.tokenizer_manager.server_status == ServerStatus.Starting:
            return False
        return True

    # ------------------------------------------------------------------
    # Tokenize / Detokenize (local ops, no inference)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Get load
    # ------------------------------------------------------------------

    def get_load(self, chunk_callback, dp_rank: Optional[int] = None) -> None:
        """Return load info via chunk_callback."""
        self._submit_on_tm_loop(
            lambda: self._get_load_async(chunk_callback, dp_rank),
            chunk_callback,
            empty_response=b"",
        )

    async def _get_load_async(
        self, chunk_callback, dp_rank: Optional[int] = None
    ) -> None:
        try:
            result = await self.tokenizer_manager.get_loads(dp_rank=dp_rank)
            data = json.dumps([dataclasses.asdict(r) for r in result], default=str)
            chunk_callback(data.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC get_load error: %s", e)
            data = json.dumps({"error": str(e)})
            chunk_callback(data.encode("utf-8"), finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Flush cache
    # ------------------------------------------------------------------

    def flush_cache(self, chunk_callback) -> None:
        """Flush the radix cache. Sends result through chunk_callback."""
        self._submit_on_tm_loop(
            lambda: self._flush_cache_async(chunk_callback),
            chunk_callback,
            empty_response=b"",
        )

    async def _flush_cache_async(self, chunk_callback) -> None:
        try:
            ret = await self.tokenizer_manager.flush_cache()
            result = json.dumps({"success": ret.success, "message": "Cache flushed."})
            chunk_callback(result.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC flush_cache error: %s", e)
            result = json.dumps({"success": False, "message": str(e)})
            chunk_callback(result.encode("utf-8"), finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Pause / Continue generation
    # ------------------------------------------------------------------

    def pause_generation(self, mode: str, chunk_callback) -> None:
        """Pause generation. Sends result through chunk_callback."""
        from sglang.srt.managers.io_struct import PauseGenerationReqInput

        obj = PauseGenerationReqInput(mode=mode)
        self._submit_on_tm_loop(
            lambda: self._pause_generation_async(obj, chunk_callback),
            chunk_callback,
            empty_response=b"",
        )

    async def _pause_generation_async(self, obj, chunk_callback) -> None:
        try:
            await self.tokenizer_manager.pause_generation(obj)
            result = json.dumps({"message": f"Generation paused (mode={obj.mode})."})
            chunk_callback(result.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC pause_generation error: %s", e)
            result = json.dumps({"message": str(e)})
            chunk_callback(result.encode("utf-8"), finished=True, error=str(e))

    def continue_generation(self, chunk_callback) -> None:
        """Continue generation. Sends result through chunk_callback."""
        from sglang.srt.managers.io_struct import ContinueGenerationReqInput

        obj = ContinueGenerationReqInput()
        self._submit_on_tm_loop(
            lambda: self._continue_generation_async(obj, chunk_callback),
            chunk_callback,
            empty_response=b"",
        )

    async def _continue_generation_async(self, obj, chunk_callback) -> None:
        try:
            await self.tokenizer_manager.continue_generation(obj)
            result = json.dumps({"message": "Generation continued."})
            chunk_callback(result.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC continue_generation error: %s", e)
            result = json.dumps({"message": str(e)})
            chunk_callback(result.encode("utf-8"), finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Profile (admin)
    # ------------------------------------------------------------------

    def start_profile(self, output_dir: Optional[str], chunk_callback) -> None:
        """Start profiling. Sends result through chunk_callback."""
        self._submit_on_tm_loop(
            lambda: self._start_profile_async(output_dir, chunk_callback),
            chunk_callback,
            empty_response=b"",
        )

    async def _start_profile_async(self, output_dir, chunk_callback) -> None:
        try:
            kwargs = {}
            if output_dir:
                kwargs["output_dir"] = output_dir
            await self.tokenizer_manager.start_profile(**kwargs)
            result = json.dumps({"message": "Profiling started."})
            chunk_callback(result.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC start_profile error: %s", e)
            result = json.dumps({"message": str(e)})
            chunk_callback(result.encode("utf-8"), finished=True, error=str(e))

    def stop_profile(self, chunk_callback) -> None:
        """Stop profiling. Sends result through chunk_callback."""
        self._submit_on_tm_loop(
            lambda: self._stop_profile_async(chunk_callback),
            chunk_callback,
            empty_response=b"",
        )

    async def _stop_profile_async(self, chunk_callback) -> None:
        try:
            await self.tokenizer_manager.stop_profile()
            result = json.dumps({"message": "Profiling stopped."})
            chunk_callback(result.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC stop_profile error: %s", e)
            result = json.dumps({"message": str(e)})
            chunk_callback(result.encode("utf-8"), finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Update weights from disk (admin)
    # ------------------------------------------------------------------

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str], chunk_callback
    ) -> None:
        """Update weights from disk. Sends result through chunk_callback."""
        self._submit_on_tm_loop(
            lambda: self._update_weights_async(model_path, load_format, chunk_callback),
            chunk_callback,
            empty_response=b"",
        )

    async def _update_weights_async(
        self, model_path: str, load_format: Optional[str], chunk_callback
    ) -> None:
        try:
            from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

            obj = UpdateWeightFromDiskReqInput(
                model_path=model_path,
                load_format=load_format,
            )
            success, message, num_paused = (
                await self.tokenizer_manager.update_weights_from_disk(obj, request=None)
            )
            result = json.dumps(
                {
                    "success": success,
                    "message": message,
                    "num_paused_requests": num_paused,
                }
            )
            chunk_callback(result.encode("utf-8"), finished=True)
        except Exception as e:
            logger.error("gRPC update_weights error: %s", e)
            result = json.dumps({"success": False, "message": str(e)})
            chunk_callback(result.encode("utf-8"), finished=True, error=str(e))

    # ------------------------------------------------------------------
    # OpenAI-compatible RPCs (JSON pass-through)
    # ------------------------------------------------------------------

    def submit_openai_chat(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit OpenAI chat completion (JSON pass-through)."""
        self._submit_on_tm_loop(
            lambda: self._run_openai_request(
                "chat",
                json_body,
                chunk_callback,
                streaming=True,
                trace_headers=trace_headers,
            ),
            chunk_callback,
            empty_response=b"",
        )

    def submit_openai_complete(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit OpenAI completion (JSON pass-through)."""
        self._submit_on_tm_loop(
            lambda: self._run_openai_request(
                "completion",
                json_body,
                chunk_callback,
                streaming=True,
                trace_headers=trace_headers,
            ),
            chunk_callback,
            empty_response=b"",
        )

    def submit_openai_embed(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit OpenAI embedding (JSON pass-through, unary)."""
        self._submit_on_tm_loop(
            lambda: self._run_openai_request(
                "embedding",
                json_body,
                chunk_callback,
                streaming=False,
                trace_headers=trace_headers,
            ),
            chunk_callback,
            empty_response=b"",
        )

    def submit_openai_classify(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit OpenAI classify (JSON pass-through, unary)."""
        self._submit_on_tm_loop(
            lambda: self._run_openai_request(
                "classify",
                json_body,
                chunk_callback,
                streaming=False,
                trace_headers=trace_headers,
            ),
            chunk_callback,
            empty_response=b"",
        )

    def submit_openai_score(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit OpenAI score (JSON pass-through, unary)."""
        self._submit_on_tm_loop(
            lambda: self._run_openai_request(
                "score",
                json_body,
                chunk_callback,
                streaming=False,
                trace_headers=trace_headers,
            ),
            chunk_callback,
            empty_response=b"",
        )

    def submit_openai_rerank(
        self,
        *,
        json_body: bytes,
        chunk_callback,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit OpenAI rerank (JSON pass-through, unary)."""
        self._submit_on_tm_loop(
            lambda: self._run_openai_request(
                "rerank",
                json_body,
                chunk_callback,
                streaming=False,
                trace_headers=trace_headers,
            ),
            chunk_callback,
            empty_response=b"",
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
    ):
        """Generic OpenAI pass-through handler.

        Delegates to the appropriate OpenAIServing* class, sending response
        data back through the Rust chunk_callback.
        """
        try:
            serving = self._get_openai_serving()[serving_key]
            request_dict = json.loads(json_body)
            mock_request = MockRequest(headers=trace_headers)

            request_cls = self._get_openai_request_class(serving_key)
            request_obj = request_cls(**request_dict)

            result = await serving.handle_request(request_obj, mock_request)

            if hasattr(result, "body_iterator"):
                async for raw_chunk in result.body_iterator:
                    if isinstance(raw_chunk, bytes):
                        raw_chunk = raw_chunk.decode("utf-8", errors="replace")
                    if (
                        raw_chunk.startswith("data: ")
                        and raw_chunk.strip() != "data: [DONE]"
                    ):
                        json_chunk = raw_chunk[len("data: ") :].strip()
                        if json_chunk:
                            chunk_callback(json_chunk.encode("utf-8"), finished=False)
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
                status_code = int(getattr(result, "status_code", 200))
                chunk_callback(resp_bytes, finished=True, status_code=status_code)

        except Exception as e:
            logger.error("gRPC OpenAI %s error: %s", serving_key, e)
            error_body = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            if streaming:
                self._safe_callback(
                    chunk_callback, error_body, finished=True, error=str(e)
                )
            else:
                try:
                    chunk_callback(
                        error_body,
                        finished=True,
                        status_code=int(getattr(e, "status_code", 500)),
                    )
                except Exception:
                    pass
