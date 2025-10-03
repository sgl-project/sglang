import asyncio
import uuid
import torch
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional, Union

import sglang as sgl
from fastapi import FastAPI, HTTPException, Request, Body, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hathora_config import DeploymentConfig
from sglang.srt.utils import add_prometheus_middleware, set_prometheus_multiproc_dir
 

# Optional HTTP client for enrollment callback
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

# Configure comprehensive logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("serve_hathora.log", mode="a")
    ]
)

logger = logging.getLogger(__name__)

# Environment and deployment configuration
HATHORA_REGION = os.environ.get("HATHORA_REGION", "unknown")


def _detect_gpu_names() -> List[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def _is_h100_only() -> bool:
    names = _detect_gpu_names()
    if not names:
        return False
    return all("H100" in name or "NVIDIA H100" in name for name in names)


def _load_deployment_config() -> DeploymentConfig:
    # Primary: DEPLOYMENT_CONFIG_JSON
    cfg_json = os.environ.get("DEPLOYMENT_CONFIG_JSON")
    if cfg_json:
        try:
            data = json.loads(cfg_json)
            return DeploymentConfig(**data)
        except Exception as e:
            logger.warning(f"Failed to parse DEPLOYMENT_CONFIG_JSON: {e}")

    # Fallback: individual env vars
    # Speculative decoding toggle: enable only if SPEC_DECODE is truthy
    _spec_decode_enabled = os.environ.get("SPEC_DECODE", "").lower() in ("1", "true", "yes")
    _spec_algo_default = None
    if _spec_decode_enabled:
        # If user specified the algorithm, honor it
        _spec_algo_env = os.environ.get("SPECULATIVE_ALGORITHM")
        if _spec_algo_env:
            _spec_algo_default = _spec_algo_env
        else:
            # Choose a no-draft default unless a draft is provided
            _draft_path = os.environ.get("SPECULATIVE_DRAFT_MODEL_PATH")
            _model_path_env = os.environ.get("MODEL_PATH")
            if _draft_path:
                _spec_algo_default = "EAGLE"  # two-model mode
            elif "eagle3" in _model_path_env:
                _spec_algo_default = "EAGLE3"  # single-model eagle3

    return DeploymentConfig(
        hf_token=os.environ.get("HF_TOKEN"),
        model_id=os.environ.get("MODEL_PATH"),
        revision=os.environ.get("REVISION"),
        dtype=os.environ.get("DTYPE", "auto"),
        quantization=os.environ.get("QUANTIZATION") or None,
        kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "auto"),
        tp_size=int(os.environ.get("TP_SIZE", "1")),
        max_total_tokens=int(os.environ.get("MAX_TOTAL_TOKENS", "4096")),
        mem_fraction_static=(
            float(os.environ["MEM_FRACTION_STATIC"]) if os.environ.get("MEM_FRACTION_STATIC") else None
        ),
        schedule_conservativeness=float(os.environ.get("SCHEDULE_CONSERVATIVENESS", "1.0")),
        max_queued_requests=(
            int(os.environ["MAX_QUEUED_REQUESTS"]) if os.environ.get("MAX_QUEUED_REQUESTS") else None
        ),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", os.environ.get("HATHORA_DEFAULT_PORT", "8000"))),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        enable_metrics=os.environ.get("ENABLE_METRICS", "true").lower() in ("1", "true", "yes"),
        log_requests=os.environ.get("LOG_REQUESTS", "true").lower() in ("1", "true", "yes"),
        enable_p2p_check=os.environ.get("ENABLE_P2P_CHECK", "false").lower() in ("1", "true", "yes"),
        enable_torch_compile=os.environ.get("ENABLE_TORCH_COMPILE", "false").lower() in ("1", "true", "yes"),
        # Allow L4 in prod by default; can still force H100 via env
        h100_only=os.environ.get("H100_ONLY", "false").lower() in ("1", "true", "yes"),
        auto_use_fp8_on_h100=os.environ.get("AUTO_USE_FP8_ON_H100", "true").lower() in ("1", "true", "yes"),
        autoscale_target_tokens_per_s=(
            float(os.environ["AUTOSCALE_TARGET_TOKENS_PER_S"]) if os.environ.get("AUTOSCALE_TARGET_TOKENS_PER_S") else None
        ),
        autoscale_target_queue_depth=(
            int(os.environ["AUTOSCALE_TARGET_QUEUE_DEPTH"]) if os.environ.get("AUTOSCALE_TARGET_QUEUE_DEPTH") else None
        ),
        # Speculative decoding configs (enabled only if SPEC_DECODE is set)
        speculative_algorithm=_spec_algo_default,
        speculative_draft_model_path=os.environ.get("SPECULATIVE_DRAFT_MODEL_PATH") or None,
        speculative_draft_model_revision=os.environ.get("SPECULATIVE_DRAFT_MODEL_REVISION") or None,
        speculative_num_steps=(int(os.environ["SPECULATIVE_NUM_STEPS"]) if os.environ.get("SPECULATIVE_NUM_STEPS") else 1),
        speculative_eagle_topk=(int(os.environ["SPECULATIVE_EAGLE_TOPK"]) if os.environ.get("SPECULATIVE_EAGLE_TOPK") else 1),
        speculative_num_draft_tokens=(int(os.environ["SPECULATIVE_NUM_DRAFT_TOKENS"]) if os.environ.get("SPECULATIVE_NUM_DRAFT_TOKENS") else 2),
        speculative_token_map=os.environ.get("SPECULATIVE_TOKEN_MAP") or None,
        speculative_attention_mode=os.environ.get("SPECULATIVE_ATTENTION_MODE") or "prefill",
    )


CONFIG = _load_deployment_config()

if CONFIG.h100_only and not _is_h100_only():
    logger.error("H100-only constraint violated: non-H100 GPU detected")
    raise SystemExit(1)

if CONFIG.hf_token:
    os.environ["HF_TOKEN"] = CONFIG.hf_token

if getattr(CONFIG, "enable_expandable_segments", False):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if CONFIG.enable_metrics:
    # Ensure prometheus multi-process directory is configured
    try:
        set_prometheus_multiproc_dir()
    except Exception as e:
        logger.warning(f"Failed setting PROMETHEUS_MULTIPROC_DIR: {e}")

logger.info("Deployment configuration:")
logger.info(f"  HATHORA_REGION: {HATHORA_REGION}")
logger.info(f"  model_id: {CONFIG.model_id}")
logger.info(f"  tp_size: {CONFIG.tp_size}")
logger.info(f"  dtype: {CONFIG.dtype}, quant: {CONFIG.quantization}, kv: {CONFIG.kv_cache_dtype}")
logger.info(f"  max_total_tokens: {CONFIG.max_total_tokens}")
logger.info(f"  enable_metrics: {CONFIG.enable_metrics}")
logger.info(f"  h100_only: {CONFIG.h100_only}")
logger.info(
    "  speculative: algo=%s, draft=%s, steps=%s, topk=%s, num_draft_tokens=%s, token_map=%s, attn_mode=%s",
    CONFIG.speculative_algorithm,
    CONFIG.speculative_draft_model_path,
    CONFIG.speculative_num_steps,
    CONFIG.speculative_eagle_topk,
    CONFIG.speculative_num_draft_tokens,
    CONFIG.speculative_token_map,
    (CONFIG.speculative_attention_mode or "prefill"),
)
logger.info(f"  namespace: {CONFIG.namespace}, deployment_id: {CONFIG.deployment_id}, customer_id: {CONFIG.customer_id}")

 

# Global engine instance
engine = None
# Enrollment state and chosen runtime params
_enrollment_sent = False
_chosen_dtype = None
_chosen_quant = None
_chosen_kv_dtype = None
# Enrollment helper
def _send_enrollment_if_needed():
    global _enrollment_sent
    if _enrollment_sent:
        return
    if CONFIG.enrollment_url and httpx is not None:
        try:
            payload = {
                "deployment_id": CONFIG.deployment_id,
                "namespace": CONFIG.namespace,
                "customer_id": CONFIG.customer_id,
                "model_id": CONFIG.model_id,
                "tp_size": CONFIG.tp_size,
                "dtype": _chosen_dtype or CONFIG.dtype,
                "quantization": _chosen_quant or CONFIG.quantization,
                "kv_cache_dtype": _chosen_kv_dtype or CONFIG.kv_cache_dtype,
                "region": HATHORA_REGION,
                "metrics_url": f"http://{CONFIG.host}:{CONFIG.port}/metrics" if CONFIG.enable_metrics else None,
                "logs_url": f"http://{CONFIG.host}:{CONFIG.port}/logs",
                "labels": CONFIG.labels or {},
            }
            with httpx.Client(timeout=5.0) as c:  # type: ignore[attr-defined]
                c.post(CONFIG.enrollment_url, json=payload)
            _enrollment_sent = True
            logger.info("Enrollment callback sent")
        except Exception as e:
            logger.warning(f"Enrollment callback failed: {e}")

# Global engine instance
engine = None

# --- Engine Management ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages SGLang engine lifecycle during FastAPI startup/shutdown."""
    global engine
    
    logger.info("Starting SGLang engine initialization...")
    startup_start_time = time.time()
    
    try:
        # Auto-prefer fp8 on H100 if not explicitly set
        dtype = CONFIG.dtype
        kv_dtype = CONFIG.kv_cache_dtype
        quant = CONFIG.quantization
        if _is_h100_only() and CONFIG.auto_use_fp8_on_h100:
            if dtype == "auto" and (not quant or quant == "auto"):
                # prefer bf16/auto unless explicitly opted into FP8 via env
                pass
            if kv_dtype == "auto":
                pass

        # NOTE: Qwen3 spec-dec temporary disable block (re-enabled to ignore eagle for qwen3 for now)
        spec_algo = CONFIG.speculative_algorithm
        try:
            if spec_algo is not None:
                model_id_lower = (CONFIG.model_id or "").lower()
                force_qwen3 = os.environ.get("SPEC_DECODE_QWEN3_ENABLE", "").lower() in ("1", "true", "yes")
                if ("qwen3" in model_id_lower) and not force_qwen3:
                    logger.warning(
                        "Speculative decoding disabled for Qwen3 models by default. Set SPEC_DECODE_QWEN3_ENABLE=1 to force enable."
                    )
                    spec_algo = None
        except Exception:
            pass

        _is_embedding_env = os.environ.get("IS_EMBEDDING", "").lower() in ("1", "true", "yes")
        _is_embedding_auto = False
        try:
            _model_id_lower = (CONFIG.model_id or "").lower()
            if "embedding" in _model_id_lower:
                _is_embedding_auto = True
        except Exception:
            pass

        # Tune scheduler polling: prefer 0 (busy/fast poll) for low-latency warmup
        try:
            _recv_interval = int(os.environ.get("SCHEDULER_RECV_INTERVAL", "0"))
        except Exception:
            _recv_interval = 0

        # Decide whether to skip tokenizer/processor initialization.
        _skip_tokenizer_init = os.environ.get("FORCE_SKIP_TOKENIZER_INIT", "").lower() in ("1", "true", "yes")

        engine = sgl.Engine(
            model_path=CONFIG.model_id,
            tokenizer_path=(os.environ.get("TOKENIZER_PATH") or CONFIG.model_id),
            revision=CONFIG.revision,
            dtype=dtype,
            quantization=quant,
            kv_cache_dtype=kv_dtype,
            tp_size=CONFIG.tp_size,
            is_embedding=_is_embedding_env or _is_embedding_auto,
            disable_radix_cache=(_is_embedding_env or _is_embedding_auto),
            disaggregation_decode_tp=os.environ.get("DISAGGREGATION_DECODE_TP") or os.environ.get("DISAGG_DECODE_TP"),
            disaggregation_prefill_pp=os.environ.get("DISAGGREGATION_PREFILL_PP"),
            max_total_tokens=CONFIG.max_total_tokens,
            enable_memory_saver=getattr(CONFIG, "enable_memory_saver", True),
            disable_custom_all_reduce=True,
            mem_fraction_static=CONFIG.mem_fraction_static,
            schedule_conservativeness=CONFIG.schedule_conservativeness,
            max_queued_requests=CONFIG.max_queued_requests,
            enable_metrics=CONFIG.enable_metrics,
            enable_p2p_check=CONFIG.enable_p2p_check,
            enable_torch_compile=CONFIG.enable_torch_compile,
            trust_remote_code=os.environ.get("TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"),
            skip_tokenizer_init=_skip_tokenizer_init,
            # Reduce server-side warmup to speed startup
            skip_server_warmup=True,
            # Faster scheduler polling for lower latency
            scheduler_recv_interval=_recv_interval,
            log_level="error",  # Reduce SGLang internal logging
            # Speculative decoding passthrough (possibly adjusted above)
            speculative_algorithm=spec_algo,
            speculative_draft_model_path=CONFIG.speculative_draft_model_path,
            speculative_draft_model_revision=CONFIG.speculative_draft_model_revision,
            speculative_num_steps=CONFIG.speculative_num_steps,
            speculative_eagle_topk=CONFIG.speculative_eagle_topk,
            speculative_num_draft_tokens=CONFIG.speculative_num_draft_tokens,
            speculative_token_map=CONFIG.speculative_token_map,
            speculative_attention_mode=(CONFIG.speculative_attention_mode or "prefill"),
        )
        # Optional warmup to avoid first-request latency
        try:
            warmup_on_start = os.environ.get("WARMUP_ON_START", "1").lower() in ("1", "true", "yes")
            if warmup_on_start:
                is_embed_mode = False
                try:
                    is_embed_mode = bool(getattr(engine.tokenizer_manager.server_args, "is_embedding", False))
                except Exception:
                    pass
                t0_ns = time.perf_counter_ns()
                if is_embed_mode:
                    # Optional: number of warmup samples
                    try:
                        _samples = int(os.environ.get("WARMUP_EMBED_SAMPLES", "1"))
                        _samples = max(1, min(_samples, 8))
                    except Exception:
                        _samples = 1
                    logger.info(f"Warmup: starting embedding warmup ({_samples} sample(s))")
                    print(f"[Warmup] Embedding warmup start: samples={_samples}")
                    sys.stdout.flush()
                    # Pre-warm cuBLAS with a tiny GEMM
                    try:
                        import torch
                        a = torch.ones((16, 16), device="cuda", dtype=torch.float16)
                        b = torch.ones((16, 16), device="cuda", dtype=torch.float16)
                        _ = a @ b
                        torch.cuda.synchronize()
                        logger.info("Warmup: cuBLAS initialized via tiny GEMM")
                        print("[Warmup] cuBLAS tiny GEMM done")
                        sys.stdout.flush()
                    except Exception as _g:
                        logger.debug(f"Warmup: GEMM prewarm skipped: {_g}")
                    w0_ns = time.perf_counter_ns()
                    fast_only = os.environ.get("WARMUP_FAST", "0").lower() in ("1", "true", "yes")
                    if not fast_only:
                        payload = ["warmup"] * _samples
                        await engine.async_encode(payload)
                    else:
                        logger.info("Warmup: FAST mode enabled, skipping encode warmup")
                        print("[Warmup] FAST mode: skipping encode")
                        sys.stdout.flush()
                    w1_ns = time.perf_counter_ns()
                    logger.info(f"Warmup: embedding encode completed in {(w1_ns - w0_ns)/1e6:.2f} ms")
                    print(f"[Warmup] Embedding warmup done in {(w1_ns - w0_ns)/1e6:.2f} ms")
                    sys.stdout.flush()
                else:
                    logger.info("Warmup: starting generation warmup (1 token)")
                    print("[Warmup] Generation warmup start: 1 token")
                    sys.stdout.flush()
                    w0_ns = time.perf_counter_ns()
                    await engine.async_generate(
                        "warmup",
                        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                    )
                    w1_ns = time.perf_counter_ns()
                    logger.info(f"Warmup: generation completed in {(w1_ns - w0_ns)/1e6:.2f} ms")
                    print(f"[Warmup] Generation warmup done in {(w1_ns - w0_ns)/1e6:.2f} ms")
                    sys.stdout.flush()
                t1_ns = time.perf_counter_ns()
                logger.info(f"Warmup: total warmup time {(t1_ns - t0_ns)/1e6:.2f} ms")
                print(f"[Warmup] Total warmup time {(t1_ns - t0_ns)/1e6:.2f} ms")
                sys.stdout.flush()
        except Exception as _e:
            logger.warning(f"Warmup skipped due to error: {_e}")
        # Record chosen dtypes; send enrollment if configured
        global _chosen_dtype, _chosen_quant, _chosen_kv_dtype
        _chosen_dtype, _chosen_quant, _chosen_kv_dtype = dtype, quant, kv_dtype
        _send_enrollment_if_needed()
        
        startup_duration = time.time() - startup_start_time
        logger.info(f"SGLang engine loaded successfully in {startup_duration:.2f}s")
        logger.info("Engine ready to serve requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize SGLang engine: {e}")
        raise
    
    yield
    
    logger.info("Shutting down SGLang engine...")
    try:
        if hasattr(engine, 'shutdown'):
            engine.shutdown()
        logger.info("SGLang engine shutdown completed")
    except Exception as e:
        logger.warning(f"Error during engine shutdown: {e}")


# --- Pydantic Models for OpenAI compatibility ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

    top_k: Optional[int] = None
    min_p: Optional[float] = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    logit_bias: Optional[Dict[str, float]] = None
    response_format: Optional[Dict] = None
    n: Optional[int] = 1
    no_stop_trim: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True

    seed: Optional[int] = None
    top_a: Optional[float] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    parallel_tool_calls: Optional[bool] = None
    verbosity: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    hathora_region: str = HATHORA_REGION
    time_to_first_token: Optional[float] = None
    total_inference_latency: float


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]
    hathora_region: str = HATHORA_REGION
    time_to_token: Optional[float] = None

# --- Utility Functions ---


def extract_prompt_from_messages(messages: List[ChatMessage]) -> str:
    """Extract a formatted prompt from chat messages.

    Prefer the HF tokenizer's built-in chat template when available; fall back to a
    simple plain-text template otherwise.
    """
    logger.debug(f"Extracting prompt from {len(messages)} messages")

    # Try to use the model's tokenizer chat template
    try:
        tok = getattr(getattr(engine, "tokenizer_manager", None), "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            conv = [{"role": m.role, "content": m.content} for m in messages]
            prompt = tok.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug("Applied HF chat template for prompt construction")
            return prompt
    except Exception as e:
        logger.warning(f"Chat template application failed, falling back: {e}")

    # Fallback: minimal plain-text formatting
    formatted_messages = []
    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted_messages.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted_messages.append(f"Assistant: {msg.content}")

    formatted_messages.append("Assistant:")
    prompt = "\n".join(formatted_messages)
    logger.debug(f"Formatted prompt (fallback): {prompt[:100]}...")
    return prompt


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"chatcmpl-{int(time.time() * 1000000)}"


async def generate_response_sglang(
    request: ChatCompletionRequest,
    prompt: str
) -> tuple[str, float, dict, float]:
    """Generate response using SGLang engine."""
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Starting generation for prompt length: {len(prompt)}")
    
    inference_start_ns = time.perf_counter_ns()
    
    try:
        # Allow 'seed' and 'top_a' for compatibility; currently ignored by the backend
        # Also accept 'logprobs/top_logprobs' and 'tools' related fields without effect

        sampling_params: Dict[str, Union[int, float, bool, str, List[str], List[int], Dict[str, float]]] = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "repetition_penalty": request.repetition_penalty,
            "min_p": request.min_p if request.min_p is not None else 0.0,
            "n": request.n or 1,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
        }
        if request.top_k is not None:
            sampling_params["top_k"] = request.top_k if request.top_k > 0 else -1
        if request.stop is not None:
            sampling_params["stop"] = request.stop
        # Sanitize stop_token_ids to integers only
        if request.stop_token_ids is not None:
            try:
                stop_ids = []
                for x in request.stop_token_ids:
                    try:
                        stop_ids.append(int(x))
                    except Exception:
                        continue
                if stop_ids:
                    sampling_params["stop_token_ids"] = stop_ids
            except Exception:
                logger.warning(f"[{request_id}] Ignoring invalid stop_token_ids: {request.stop_token_ids}")
        # Sanitize logit_bias keys to integer token ids
        if request.logit_bias is not None and isinstance(request.logit_bias, dict):
            try:
                cleaned_logit_bias = {}
                for k, v in request.logit_bias.items():
                    try:
                        token_id = int(k)
                        cleaned_logit_bias[str(token_id)] = float(v)
                    except Exception:
                        continue
                if cleaned_logit_bias:
                    sampling_params["logit_bias"] = cleaned_logit_bias
            except Exception:
                logger.warning(f"[{request_id}] Ignoring invalid logit_bias")

        if request.response_format and isinstance(request.response_format, dict):
            rf_type = request.response_format.get("type")
            if rf_type == "json_schema":
                schema = request.response_format.get("json_schema")
                try:
                    schema_str = json.dumps(schema) if not isinstance(schema, str) else schema
                    sampling_params["json_schema"] = schema_str
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid json_schema: {str(e)}")
            elif rf_type == "json_object":
                sampling_params["json_schema"] = "{\"type\": \"object\"}"
        
        logger.debug(f"[{request_id}] Sampling params: {sampling_params}")
        
        # Use SGLang async_generate
        eng_t0_ns = time.perf_counter_ns()
        result = await engine.async_generate(
            prompt,
            sampling_params=sampling_params
        )
        eng_ms = (time.perf_counter_ns() - eng_t0_ns) / 1e6
        
        total_inference_latency_ms = (time.perf_counter_ns() - inference_start_ns) / 1e6
        
        generated_text = result.get("text", "") if isinstance(result, dict) else str(result)

        if isinstance(result, dict) and isinstance(result.get("meta_info"), dict):
            response_text = generated_text.strip()
        else:
            response_text = generated_text[len(prompt):].strip()
        
        logger.info(f"[{request_id}] Engine generate time: {eng_ms:.2f}ms; request total inference: {total_inference_latency_ms:.2f}ms")
        logger.debug(f"[{request_id}] Response: {response_text[:100]}...")
        
        if isinstance(result, dict) and isinstance(result.get("meta_info"), dict):
            meta = result["meta_info"]
            prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
            completion_tokens = int(meta.get("completion_tokens", 0) or 0)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        else:
            tokenizer = getattr(getattr(engine, "tokenizer_manager", None), "tokenizer", None)
            if tokenizer is None:
                raise HTTPException(status_code=500, detail="Tokenizer not initialized")
            try:
                prompt_tokens = len(tokenizer.encode(prompt))
                completion_tokens = len(tokenizer.encode(response_text))
                usage_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Tokenizer error: {str(e)}")
        
        return response_text, total_inference_latency_ms, usage_info, eng_ms
        
    except Exception as e:
        logger.error(f"[{request_id}] Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def build_non_streaming_response(
    request: ChatCompletionRequest,
    response_text: str,
    total_inference_latency_ms: float,
    usage_info: dict
) -> ChatCompletionResponse:
    """Build a non-streaming chat completion response."""
    completion_message = ChatMessage(role="assistant", content=response_text)
    choice = ChatCompletionChoice(index=0, message=completion_message)
    usage = UsageInfo(**usage_info)
    
    response = ChatCompletionResponse(
        id=generate_request_id(),
        model=request.model,
        choices=[choice],
        usage=usage,
        total_inference_latency=total_inference_latency_ms,
    )
    
    logger.debug(f"Built non-streaming response with {len(response_text)} chars")
    return response


async def stream_response_sglang(
    request: ChatCompletionRequest,
    prompt: str
) -> AsyncGenerator[str, None]:
    """Stream response using SGLang engine."""
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Starting streaming generation for prompt length: {len(prompt)}")
    
    try:
        # Validate unsupported params (allow 'seed' and 'top_a' for compatibility; currently ignored)
        # Also accept 'logprobs/top_logprobs' and 'tools' related fields without effect

        sampling_params: Dict[str, Union[int, float, bool, str, List[str], List[int], Dict[str, float]]] = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "repetition_penalty": request.repetition_penalty,
            "min_p": request.min_p if request.min_p is not None else 0.0,
            "n": request.n or 1,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
        }
        if request.top_k is not None:
            sampling_params["top_k"] = request.top_k if request.top_k > 0 else -1
        if request.stop is not None:
            sampling_params["stop"] = request.stop
        # Sanitize stop_token_ids to integers only
        if request.stop_token_ids is not None:
            try:
                stop_ids = []
                for x in request.stop_token_ids:
                    try:
                        stop_ids.append(int(x))
                    except Exception:
                        continue
                if stop_ids:
                    sampling_params["stop_token_ids"] = stop_ids
            except Exception:
                logger.warning(f"[{request_id}] Ignoring invalid stop_token_ids: {request.stop_token_ids}")
        # Sanitize logit_bias keys to integer token ids
        if request.logit_bias is not None and isinstance(request.logit_bias, dict):
            try:
                cleaned_logit_bias = {}
                for k, v in request.logit_bias.items():
                    try:
                        token_id = int(k)
                        cleaned_logit_bias[str(token_id)] = float(v)
                    except Exception:
                        continue
                if cleaned_logit_bias:
                    sampling_params["logit_bias"] = cleaned_logit_bias
            except Exception:
                logger.warning(f"[{request_id}] Ignoring invalid logit_bias")
        if request.response_format and isinstance(request.response_format, dict):
            rf_type = request.response_format.get("type")
            if rf_type == "json_schema":
                schema = request.response_format.get("json_schema")
                try:
                    schema_str = json.dumps(schema) if not isinstance(schema, str) else schema
                    sampling_params["json_schema"] = schema_str
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid json_schema: {str(e)}")
            elif rf_type == "json_object":
                sampling_params["json_schema"] = "{\"type\": \"object\"}"
        
        logger.debug(f"[{request_id}] Streaming sampling params: {sampling_params}")
        
        start_time = time.time()
        
        # Use SGLang async_generate with streaming
        async_generator = await engine.async_generate(
            prompt,
            sampling_params=sampling_params,
            stream=True
        )
        
        token_count = 0

        # Track previously emitted content length from cumulative generated text
        last_emitted_len = 0

        # Send initial role chunk to follow typical streaming contract
        role_choice = ChatCompletionStreamChoice(
            index=0,
            delta={"role": "assistant"},
            finish_reason=None
        )
        role_response = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[role_choice]
        )
        yield f"data: {role_response.model_dump_json()}\n\n"
        
        # Stream the tokens
        async for chunk in async_generator:
            # Engine yields cumulative completion text (without the prompt)
            content = chunk.get("text", "")
            if not isinstance(content, str):
                content = str(content)

            if len(content) <= last_emitted_len:
                # No new delta yet
                await asyncio.sleep(0)
                continue

            new_content = content[last_emitted_len:]
            last_emitted_len = len(content)

            token_count += 1
            time_to_token = (time.time() - start_time) * 1000

            stream_choice = ChatCompletionStreamChoice(
                index=0,
                delta={"content": new_content},
                finish_reason=None
            )

            stream_response = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[stream_choice],
                time_to_token=time_to_token
            )

            yield f"data: {stream_response.model_dump_json()}\n\n"
            await asyncio.sleep(0)  # Yield control to event loop
        
        # Send final chunk with finish_reason
        final_choice = ChatCompletionStreamChoice(
            index=0,
            delta={},
            finish_reason="stop"
        )
        
        final_response = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[final_choice]
        )
        
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info(f"[{request_id}] Streaming completed with {token_count} tokens")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error during streaming: {e}")
        error_choice = ChatCompletionStreamChoice(
            index=0,
            delta={"content": f"Error: {str(e)}"},
            finish_reason="stop"
        )
        
        error_response = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[error_choice]
        )
        
        yield f"data: {error_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


# --- Token Counting Utilities and Endpoint ---

def _count_tokens_from_messages(messages: List[ChatMessage]) -> int:
    """Count prompt tokens by applying the same simple chat template used for generation.

    Strict: requires tokenizer; no approximations or fallbacks.
    """
    tokenizer = getattr(getattr(engine, "tokenizer_manager", None), "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized")
    prompt_text = extract_prompt_from_messages(messages)
    try:
        return len(tokenizer.encode(prompt_text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenizer error: {str(e)}")


class TokenCountResponse(BaseModel):
    prompt_tokens: int

# Simple request schema for /v1/tokens
class TokenCountSimpleRequest(BaseModel):
    model: Optional[str] = None
    input: str

# --- FastAPI App and Endpoints ---
app = FastAPI(lifespan=lifespan, title="SGLang Hathora Serve", version="1.0.0")

# Enable CORS (match root server behavior)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 

# Attach Prometheus metrics endpoint if enabled
if CONFIG.enable_metrics:
    try:
        add_prometheus_middleware(app)
        logger.info("Prometheus /metrics endpoint enabled")
    except Exception as e:
        logger.warning(f"Failed to add Prometheus middleware: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing information."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"[{client_ip}] {request.method} {request.url.path} - Request started")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"[{client_ip}] {request.method} {request.url.path} - "
            f"Status: {response.status_code}, Duration: {process_time:.2f}ms"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Hathora-Region"] = HATHORA_REGION
        
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"[{client_ip}] {request.method} {request.url.path} - "
            f"Error: {e}, Duration: {process_time:.2f}ms"
        )
        raise


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint with SGLang backend."""
    if not engine:
        logger.error("Engine not initialized - cannot process request")
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    logger.info(f"Chat completion request: model={request.model}, messages={len(request.messages)}, "
                f"max_tokens={request.max_tokens}, stream={request.stream}")
    
    try:
        # Extract and format prompt from messages
        prompt = extract_prompt_from_messages(request.messages)
        
        if request.stream:
            logger.debug("Processing streaming request")
            return StreamingResponse(
                stream_response_sglang(request, prompt),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Hathora-Region": HATHORA_REGION
                }
            )
        else:
            logger.debug("Processing non-streaming request")
            response_text, total_inference_latency_ms, usage_info, eng_ms = await generate_response_sglang(
                request, prompt
            )
            
            response = build_non_streaming_response(
                request, response_text, total_inference_latency_ms, usage_info
            )
            
            logger.info(f"Non-streaming response completed: {len(response_text)} chars generated")
            # Add engine timing to response headers via FastAPI Response in StreamingResponse path.
            # For non-streaming JSON response, we attach timing as a header in a JSONResponse-like fashion.
            from fastapi.responses import JSONResponse
            resp = JSONResponse(content=response.model_dump())
            resp.headers["X-Engine-Time-MS"] = f"{eng_ms:.2f}"
            resp.headers["X-Total-Inference-MS"] = f"{total_inference_latency_ms:.2f}"
            resp.headers["X-Hathora-Region"] = HATHORA_REGION
            return resp
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _count_tokens_from_text(text: str) -> int:
    tokenizer = getattr(getattr(engine, "tokenizer_manager", None), "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized")
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenizer error: {str(e)}")


@app.post("/v1/tokens", response_model=TokenCountResponse)
async def count_tokens_endpoint(req: TokenCountSimpleRequest) -> TokenCountResponse:
    """Return the number of tokens for a plain input string."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return TokenCountResponse(prompt_tokens=_count_tokens_from_text(req.input))


# --- Embedding Endpoints ---


class EmbedQueryRequest(BaseModel):
    input: Union[str, List[str]]


class EmbedDocumentsRequest(BaseModel):
    # New structure: only 'documents' is accepted
    documents: Union[str, List[str]]


class EmbeddingVector(BaseModel):
    embedding: List[float]


class EmbedResponse(BaseModel):
    data: List[EmbeddingVector]
    model: str


def _ensure_embedding_mode():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    # Best-effort guard: embedding requires is_embedding at startup
    if getattr(getattr(engine, "tokenizer_manager", None), "server_args", None):
        if not engine.tokenizer_manager.server_args.is_embedding:
            raise HTTPException(status_code=400, detail="Server not started in embedding mode")


@app.post("/v1/embedding/query", response_model=EmbedResponse)
async def embed_query(req: EmbedQueryRequest):
    _ensure_embedding_mode()
    texts = req.input if isinstance(req.input, list) else [req.input]
    eng_t0_ns = time.perf_counter_ns()
    out = await engine.async_encode(texts)
    eng_ms = (time.perf_counter_ns() - eng_t0_ns) / 1e6
    logger.info(f"[embedding.query] Engine encode {len(texts)} item(s) in {eng_ms:.2f}ms")
    # Expect shape: {"embedding": [...]} or batch style
    if isinstance(out, dict) and "embedding" in out:
        vecs = [out["embedding"]]
    elif isinstance(out, list):
        vecs = [o.get("embedding", []) for o in out]
    else:
        vecs = []
    # Attach timing header
    from fastapi.responses import JSONResponse
    payload = EmbedResponse(data=[EmbeddingVector(embedding=v) for v in vecs], model=CONFIG.model_id or "").model_dump()
    resp = JSONResponse(content=payload)
    resp.headers["X-Engine-Time-MS"] = f"{eng_ms:.2f}"
    resp.headers["X-Hathora-Region"] = HATHORA_REGION
    return resp


@app.post("/v1/embedding/documents", response_model=EmbedResponse)
async def embed_documents(req: EmbedDocumentsRequest):
    _ensure_embedding_mode()
    texts: List[str] = req.documents if isinstance(req.documents, list) else [req.documents]
    eng_t0_ns = time.perf_counter_ns()
    out = await engine.async_encode(texts)
    eng_ms = (time.perf_counter_ns() - eng_t0_ns) / 1e6
    logger.info(f"[embedding.documents] Engine encode {len(texts)} item(s) in {eng_ms:.2f}ms")
    if isinstance(out, dict) and "embedding" in out:
        vecs = [out["embedding"]]
    elif isinstance(out, list):
        vecs = [o.get("embedding", []) for o in out]
    else:
        vecs = []
    from fastapi.responses import JSONResponse
    payload = EmbedResponse(data=[EmbeddingVector(embedding=v) for v in vecs], model=CONFIG.model_id or "").model_dump()
    resp = JSONResponse(content=payload)
    resp.headers["X-Engine-Time-MS"] = f"{eng_ms:.2f}"
    resp.headers["X-Hathora-Region"] = HATHORA_REGION
    return resp


class SimilarityRequest(BaseModel):
    query: Union[str, List[float]]
    documents: Union[List[str], List[List[float]]]
    top_k: Optional[int] = None


class SimilarityScore(BaseModel):
    index: int
    score: float


class SimilarityResponse(BaseModel):
    scores: List[SimilarityScore]


def _to_tensor_1d(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32).view(-1)
    return torch.tensor(x, dtype=torch.float32, device=device).view(-1)


def _normalize_1d(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x) + 1e-12
    return x / norm


@app.post("/v1/embedding/similarity", response_model=SimilarityResponse)
async def embedding_similarity(req: SimilarityRequest):
    _ensure_embedding_mode()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare query vector
    if isinstance(req.query, list):
        q = _to_tensor_1d(req.query, device)
    else:
        out = await engine.async_encode(req.query)
        if isinstance(out, dict) and "embedding" in out:
            q = _to_tensor_1d(out["embedding"], device)
        else:
            q = _to_tensor_1d((out[0] or {}).get("embedding", []), device)
    q = _normalize_1d(q)

    # Prepare document vectors
    if len(req.documents) == 0:
        return SimilarityResponse(scores=[])
    doc_tensors: List[torch.Tensor] = []
    if isinstance(req.documents[0], (int, float)):
        doc_tensors = [_to_tensor_1d(req.documents, device)]
    elif isinstance(req.documents[0], list) and (
        len(req.documents[0]) == 0 or isinstance(req.documents[0][0], (int, float))
    ):
        doc_tensors = [_to_tensor_1d(v, device) for v in req.documents]  # type: ignore[arg-type]
    else:
        out = await engine.async_encode(req.documents)  # type: ignore[arg-type]
        if isinstance(out, dict) and "embedding" in out:
            doc_tensors = [_to_tensor_1d(out["embedding"], device)]
        else:
            doc_tensors = [_to_tensor_1d(o.get("embedding", []), device) for o in out]

    if not doc_tensors:
        return SimilarityResponse(scores=[])

    D = torch.stack([_normalize_1d(d) for d in doc_tensors], dim=0)
    sims = torch.matmul(D, q)  # cosine since normalized
    sims_list = sims.detach().cpu().tolist()

    scores = [SimilarityScore(index=i, score=float(sims_list[i])) for i in range(len(sims_list))]
    if req.top_k is not None and req.top_k > 0:
        scores = sorted(scores, key=lambda x: x.score, reverse=True)[: req.top_k]
    return SimilarityResponse(scores=scores)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    
    engine_status = "ready" if engine else "not_initialized"
    
    health_info = {
        "status": "ok",
        "engine_status": engine_status,
        "hathora_region": HATHORA_REGION,
        "model_path": CONFIG.model_id,
        "namespace": CONFIG.namespace,
        "deployment_id": CONFIG.deployment_id,
        "customer_id": CONFIG.customer_id,
        "timestamp": time.time()
    }
    
    logger.debug(f"Health check response: {health_info}")
    # Ensure enrollment sent even if no other endpoint touched
    _send_enrollment_if_needed()
    return health_info


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "SGLang Hathora Serve",
        "version": "1.0.0",
        "hathora_region": HATHORA_REGION,
        "model_path": CONFIG.model_id,
        "namespace": CONFIG.namespace,
        "deployment_id": CONFIG.deployment_id,
        "customer_id": CONFIG.customer_id,
        "endpoints": [
            "/v1/chat/completions",
            
            "/v1/embedding/query",
            "/v1/embedding/documents",
            "/v1/embedding/similarity",
            "/health",
            "/docs"
        ]
    }


@app.get("/logs")
async def stream_logs():
    async def log_stream():
        log_path = "serve_hathora.log"
        try:
            with open(log_path, "r") as f:
                # Seek to end
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.5)
                        continue
                    yield line
        except FileNotFoundError:
            yield "logs unavailable\n"

    return StreamingResponse(log_stream(), media_type="text/plain")


 


# --- Signal Handlers for Graceful Shutdown ---


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SGLang Hathora server...")
    logger.info(
        f"Configuration: MODEL_ID={CONFIG.model_id}, TP_SIZE={CONFIG.tp_size}, DTYPE={CONFIG.dtype}, "
        f"QUANT={CONFIG.quantization}, KV={CONFIG.kv_cache_dtype}"
    )
    
    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )