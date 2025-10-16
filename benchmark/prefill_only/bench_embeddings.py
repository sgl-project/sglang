"""
SGLang Embeddings Benchmark Script

This script benchmarks SGLang's /v1/embeddings API performance using HTTP requests.

Features:
- HTTP-only implementation
- Uses /v1/embeddings API endpoint directly
- Configurable RPS, duration, and batch sizes
- Progress tracking and detailed metrics
- Poisson and constant request distributions

Usage:
- Update configuration variables at the top of the file
- Ensure SGLang server is running on the configured HTTP_URL
- Run: python bench_embeddings.py
"""

import asyncio
import logging

from transformers import AutoTokenizer
from util import (
    BenchmarkConfig,
    generate_text_with_token_count,
    run_benchmark_main,
    run_generic_benchmark,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################################################################
# CONFIG
###############################################################################
# Create benchmark configuration
config = BenchmarkConfig()
config.rps_values = [500]
config.duration_secs_values = [60]
config.num_unique_requests = 100
config.distribution = "POISSON"
config.profile = False
config.freeze_gc = True  # Enable GC freeze functionality
# Profiler output directory - by default uses present working directory (pwd)
# Uncomment and customize the line below to override the default location:
# config.profiler_dir = "/sglang-oss-trace"

# HTTP Configuration
HTTP_URL = "http://localhost:30000/v1/embeddings"

# Embeddings API Config
EMBEDDINGS_MODEL_PATH = "/Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = [1]  # Number of items per request (batch size)

# Configurable input token length
EMBEDDINGS_INPUT_TOKENS = 500  # Default token length

# Load tokenizer once for embeddings text generation
print("Loading tokenizer for embeddings input generation...")
embeddings_tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS_MODEL_PATH)

# Generate input text with the specified token length using pre-loaded tokenizer
EMBEDDINGS_INPUT_TEXT = generate_text_with_token_count(
    EMBEDDINGS_MODEL_PATH,
    EMBEDDINGS_INPUT_TOKENS,
    config.special_replicated_token,
    tokenizer=embeddings_tokenizer,
)


###############################################################################
# REQUEST GENERATION (in parallel)
###############################################################################
def build_embeddings_request(index: int, item_count: int) -> tuple:
    """Build a single embeddings request."""
    try:
        # For embeddings, input can be a string or list of strings
        if item_count == 1:
            input_data = EMBEDDINGS_INPUT_TEXT
        else:
            input_data = [EMBEDDINGS_INPUT_TEXT for _ in range(item_count)]
        req = {
            "input": input_data,
            "model": EMBEDDINGS_MODEL_PATH,
        }
        return (index, req)
    except Exception as e:
        logger.error(f"Error building request {index}: {e}")
        return (index, None)


def validate_embeddings_response(response_data: dict) -> bool:
    """Validate embeddings API response."""
    return "data" in response_data


def build_warmup_embeddings_request() -> dict:
    """Build a warmup request for the embeddings API."""
    return {
        "input": EMBEDDINGS_INPUT_TEXT,
        "model": EMBEDDINGS_MODEL_PATH,
    }


###############################################################################
# MAIN
###############################################################################
async def run_benchmark(rps, duration_secs, item_count):
    """Run a single embeddings benchmark with the given RPS value."""
    return await run_generic_benchmark(
        rps=rps,
        duration_secs=duration_secs,
        item_count=item_count,
        config=config,
        http_url=HTTP_URL,
        build_request_func=build_embeddings_request,
        response_validator=validate_embeddings_response,
        api_name="EMBEDDINGS",
        request_description="embeddings requests",
    )


async def main():
    additional_info = {
        "Input text length": f"{EMBEDDINGS_INPUT_TOKENS} tokens",
        "Input text preview": (
            EMBEDDINGS_INPUT_TEXT[:100] + "..."
            if len(EMBEDDINGS_INPUT_TEXT) > 100
            else EMBEDDINGS_INPUT_TEXT
        ),
    }

    await run_benchmark_main(
        config,
        run_benchmark,
        "EMBEDDINGS",
        HTTP_URL,
        BATCH_SIZE,
        additional_info,
        build_warmup_embeddings_request,
    )


if __name__ == "__main__":
    asyncio.run(main())
