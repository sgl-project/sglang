"""
SGLang Scoring Benchmark Script

This script benchmarks SGLang's scoring API performance using HTTP requests.

Current Features:
- HTTP-only implementation (open source compatible)
- Uses /v1/score API endpoint directly
- Single item scoring with batching support
- Configurable RPS, duration, and batch sizes
- Progress tracking and detailed metrics
- Poisson and constant request distributions

Usage:
- Update configuration variables at the top of the file
- Ensure SGLang server is running on the configured HTTP_URL
- Run: python bench_score.py
- Each request will contain ITEM_COUNT_VALUES items for batch scoring

"""

import asyncio

from transformers import AutoTokenizer
from util import (
    BenchmarkConfig,
    generate_text_with_token_count,
    run_benchmark_main,
    run_generic_benchmark,
)

###############################################################################
# CONFIG
###############################################################################
# Create benchmark configuration
config = BenchmarkConfig()
config.rps_values = [160]
config.duration_secs_values = [60]
config.num_unique_requests = 100
config.distribution = "POISSON"
config.profile = False
config.freeze_gc = True  # Enable GC freeze functionality
# Profiler output directory - by default uses present working directory (pwd)
# Uncomment and customize the line below to override the default location:
# config.profiler_dir = "/sglang-oss-trace"

# HTTP Configuration
HTTP_URL = "http://localhost:30000/v1/score"  # Use score API directly

# Score API Config
# ITEM_COUNT_VALUES determines number of items per score request (batch size)
SCORE_QUERY_TOKENS = 120
SCORE_ITEM_TOKENS = 180
SCORE_MODEL_PATH = "Qwen/Qwen3-0.6B"
SCORE_LABEL_TOKEN_IDS = [9454, 2753]  # Yes/No token IDs
ITEM_COUNT_VALUES = [10]  # Number of items per request

# Special token to replicate for precise token counting
SPECIAL_REPLICATED_TOKEN = "<|im_start|>"


###############################################################################
# REQUEST GENERATION (in parallel)
###############################################################################
def create_score_request_builder():
    """Create a score request builder function with shared tokenizer."""
    # Load tokenizer once here to verify special token and get precise counts
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SCORE_MODEL_PATH)

    # Verify that our special token produces exactly 1 token
    special_token_count = len(
        tokenizer.encode(config.special_replicated_token, add_special_tokens=False)
    )
    print(
        f"Special token '{config.special_replicated_token}' produces "
        f"{special_token_count} token(s)"
    )

    def generate_text_with_token_count_local(num_toks):
        """Generate text with precise token count using replicated token."""
        return generate_text_with_token_count(
            SCORE_MODEL_PATH,
            num_toks,
            config.special_replicated_token,
            tokenizer=tokenizer,
        )

    def build_score_request(index: int, item_count: int) -> tuple:
        """Build a single score request."""
        try:
            # Generate query and items for score API
            query = generate_text_with_token_count_local(SCORE_QUERY_TOKENS)
            items = [
                generate_text_with_token_count_local(SCORE_ITEM_TOKENS)
                for _ in range(item_count)
            ]

            # Return as dict for score API format
            score_data = {
                "query": query,
                "items": items,
                "label_token_ids": SCORE_LABEL_TOKEN_IDS,
                "model": SCORE_MODEL_PATH,
            }
            return (index, score_data)

        except Exception as e:
            print(f"Error building request {index}: {e}")
            return (index, None)

    return build_score_request


def validate_score_response(response_data: dict) -> bool:
    """Validate score API response."""
    return "scores" in response_data or "logprobs" in response_data


def build_warmup_score_request() -> dict:
    """Build a warmup request for the score API."""
    # Load tokenizer once for warmup generation
    tokenizer = AutoTokenizer.from_pretrained(SCORE_MODEL_PATH)

    warmup_query = generate_text_with_token_count(
        SCORE_MODEL_PATH,
        SCORE_QUERY_TOKENS,
        config.special_replicated_token,
        tokenizer=tokenizer,
    )
    warmup_items = [
        generate_text_with_token_count(
            SCORE_MODEL_PATH,
            SCORE_ITEM_TOKENS,
            config.special_replicated_token,
            tokenizer=tokenizer,
        )
        for _ in range(3)
    ]

    return {
        "query": warmup_query,
        "items": warmup_items,
        "label_token_ids": SCORE_LABEL_TOKEN_IDS,
        "model": SCORE_MODEL_PATH,
        # Add missing parameters for consistency with the original warmup
        "apply_softmax": True,
        "item_first": False,
    }


###############################################################################
# MAIN
###############################################################################
async def run_benchmark(rps, duration_secs, item_count):
    """Run a single benchmark with the given RPS value."""
    # Create the request builder function with shared tokenizer
    build_request_func = create_score_request_builder()

    return await run_generic_benchmark(
        rps=rps,
        duration_secs=duration_secs,
        item_count=item_count,
        config=config,
        http_url=HTTP_URL,
        build_request_func=build_request_func,
        response_validator=validate_score_response,
        api_name="SINGLE_ITEM_SCORING",
        request_description="score requests",
    )


async def main():
    """Main function that runs benchmarks for all RPS values."""
    additional_info = {
        "Query tokens per request": SCORE_QUERY_TOKENS,
        "Item tokens per item": SCORE_ITEM_TOKENS,
    }

    await run_benchmark_main(
        config,
        run_benchmark,
        "SINGLE_ITEM_SCORING",
        HTTP_URL,
        ITEM_COUNT_VALUES,
        additional_info,
        build_warmup_score_request,
    )


if __name__ == "__main__":
    asyncio.run(main())
