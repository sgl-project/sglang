import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import sglang as sgl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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

# Environment configuration
HATHORA_REGION = os.environ.get("HATHORA_REGION", "unknown")
MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", "4096"))

logger.info("Environment configuration:")
logger.info(f"  HATHORA_REGION: {HATHORA_REGION}")
logger.info(f"  MODEL_PATH: {MODEL_PATH}")
logger.info(f"  TP_SIZE: {TP_SIZE}")
logger.info(f"  MAX_TOTAL_TOKENS: {MAX_TOTAL_TOKENS}")
logger.info(f"  LOG_LEVEL: {LOG_LEVEL}")

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
        engine = sgl.Engine(
            model_path=MODEL_PATH,
            tp_size=TP_SIZE,
            max_total_tokens=MAX_TOTAL_TOKENS,
            log_level="error"  # Reduce SGLang internal logging
        )
        
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
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = False


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
    """Extract a formatted prompt from chat messages."""
    logger.debug(f"Extracting prompt from {len(messages)} messages")
    
    # For simplicity, we'll use a basic chat template
    # In production, you'd want to use the model's specific chat template
    formatted_messages = []
    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted_messages.append(f"Human: {msg.content}")
        elif msg.role == "assistant":
            formatted_messages.append(f"Assistant: {msg.content}")
    
    # Add the Assistant: prefix for the response
    formatted_messages.append("Assistant:")
    prompt = "\n".join(formatted_messages)
    
    logger.debug(f"Formatted prompt: {prompt[:100]}...")
    return prompt


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"chatcmpl-{int(time.time() * 1000000)}"


async def generate_response_sglang(
    request: ChatCompletionRequest,
    prompt: str
) -> tuple[str, float, dict]:
    """Generate response using SGLang engine."""
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Starting generation for prompt length: {len(prompt)}")
    
    inference_start_time = time.time()
    
    try:
        sampling_params = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        
        logger.debug(f"[{request_id}] Sampling params: {sampling_params}")
        
        # Use SGLang async_generate
        result = await engine.async_generate(
            prompt,
            sampling_params=sampling_params
        )
        
        inference_end_time = time.time()
        total_inference_latency_ms = (inference_end_time - inference_start_time) * 1000
        
        generated_text = result["text"]
        # Remove the original prompt from the response
        response_text = generated_text[len(prompt):].strip()
        
        logger.info(f"[{request_id}] Generation completed in {total_inference_latency_ms:.2f}ms")
        logger.debug(f"[{request_id}] Response: {response_text[:100]}...")
        
        # Create usage info (approximation for now)
        usage_info = {
            "prompt_tokens": len(prompt.split()),  # Rough approximation
            "completion_tokens": len(response_text.split()),  # Rough approximation
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
        
        return response_text, total_inference_latency_ms, usage_info
        
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
        sampling_params = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        
        logger.debug(f"[{request_id}] Streaming sampling params: {sampling_params}")
        
        start_time = time.time()
        
        # Use SGLang async_generate with streaming
        async_generator = await engine.async_generate(
            prompt,
            sampling_params=sampling_params,
            stream=True
        )
        
        token_count = 0
        
        # Stream the tokens
        async for chunk in async_generator:
            token_count += 1
            time_to_token = (time.time() - start_time) * 1000
            
            # Extract the new content from the chunk
            # The chunk should contain the incremental text
            content = chunk.get("text", "")
            if content and len(content) > len(prompt):
                new_content = content[len(prompt):]
                if token_count == 1:
                    # First token, remove any previous content
                    new_content = new_content
                else:
                    # Get only the new part since last chunk
                    # This is a simplified approach; in practice, you'd want more sophisticated diff
                    pass
            else:
                new_content = ""
            
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

# --- FastAPI App and Endpoints ---
app = FastAPI(lifespan=lifespan, title="SGLang Hathora Serve", version="1.0.0")


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
            response_text, total_inference_latency_ms, usage_info = await generate_response_sglang(
                request, prompt
            )
            
            response = build_non_streaming_response(
                request, response_text, total_inference_latency_ms, usage_info
            )
            
            logger.info(f"Non-streaming response completed: {len(response_text)} chars generated")
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    
    engine_status = "ready" if engine else "not_initialized"
    
    health_info = {
        "status": "ok",
        "engine_status": engine_status,
        "hathora_region": HATHORA_REGION,
        "model_path": MODEL_PATH,
        "timestamp": time.time()
    }
    
    logger.debug(f"Health check response: {health_info}")
    return health_info


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "SGLang Hathora Serve",
        "version": "1.0.0",
        "hathora_region": HATHORA_REGION,
        "model_path": MODEL_PATH,
        "endpoints": [
            "/v1/chat/completions",
            "/health",
            "/docs"
        ]
    }


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
    logger.info(f"Configuration: MODEL_PATH={MODEL_PATH}, TP_SIZE={TP_SIZE}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )