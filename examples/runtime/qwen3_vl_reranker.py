"""
Example usage of Qwen3-VL-Reranker with SGLang.

This example demonstrates how to use the Qwen3-VL-Reranker model for multimodal
reranking tasks, supporting text, images, and videos.

Server Launch:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-VL-Reranker-2B \
        --served-model-name Qwen3-VL-Reranker-2B \
        --trust-remote-code \
        --disable-radix-cache \
        --chat-template examples/chat_template/qwen3_vl_reranker.jinja

Client Usage:
    python examples/runtime/qwen3_vl_reranker.py
"""

import requests

# Server URL
BASE_URL = "http://localhost:30000"


def rerank_text_only():
    """Example: Text-only reranking (backward compatible)."""
    print("=" * 60)
    print("Text-only reranking example")
    print("=" * 60)

    request_data = {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
            "The weather in Paris is usually mild with occasional rain.",
            "Deep learning is a subset of machine learning using neural networks with many layers.",
        ],
        "instruct": "Retrieve passages that answer the question.",
        "return_documents": True,
    }

    response = requests.post(f"{BASE_URL}/v1/rerank", json=request_data)
    results = response.json()

    print("Results (sorted by relevance):")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f} - {result['document'][:60]}...")
    print()


def rerank_with_images():
    """Example: Query is text, documents contain images."""
    print("=" * 60)
    print("Image reranking example")
    print("=" * 60)

    request_data = {
        "query": "A woman playing with her dog on a beach at sunset.",
        "documents": [
            # Document 1: Text description
            "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
            # Document 2: Image URL
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                }
            ],
            # Document 3: Text + Image (mixed)
            [
                {
                    "type": "text",
                    "text": "A joyful scene at the beach:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                },
            ],
        ],
        "instruct": "Retrieve images or text relevant to the user's query.",
        "return_documents": False,
    }

    response = requests.post(f"{BASE_URL}/v1/rerank", json=request_data)
    results = response.json()

    # Debug: print raw response if it's an error
    if isinstance(results, dict) and "message" in results:
        print(f"Error: {results['message']}")
        return
    if isinstance(results, str):
        print(f"Error: {results}")
        return

    print("Results (sorted by relevance):")
    for i, result in enumerate(results):
        print(f"  {i+1}. Index: {result['index']}, Score: {result['score']:.4f}")
    print()


def rerank_multimodal_query():
    """Example: Query contains both text and image."""
    print("=" * 60)
    print("Multimodal query reranking example")
    print("=" * 60)

    request_data = {
        # Query with text and image
        "query": [
            {"type": "text", "text": "Find similar images to this:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                },
            },
        ],
        "documents": [
            "A cat sleeping on a couch.",
            "A woman and her dog enjoying the sunset at the beach.",
            "A busy city street with cars and pedestrians.",
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                }
            ],
        ],
        "instruct": "Find images or descriptions similar to the query image.",
    }

    response = requests.post(f"{BASE_URL}/v1/rerank", json=request_data)
    results = response.json()

    # Debug: print raw response if it's an error
    if isinstance(results, dict) and "message" in results:
        print(f"Error: {results['message']}")
        return
    if isinstance(results, str):
        print(f"Error: {results}")
        return

    print("Results (sorted by relevance):")
    for i, result in enumerate(results):
        print(f"  {i+1}. Index: {result['index']}, Score: {result['score']:.4f}")
    print()


def main():
    """Run all examples."""
    print("\nQwen3-VL-Reranker Examples")
    print("Make sure the server is running with the correct model and template.\n")

    # Check if server is available
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"Server health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to server at {BASE_URL}")
        print("Please start the server first with:")
        print("  python -m sglang.launch_server \\")
        print("      --model-path Qwen/Qwen3-VL-Reranker-2B \\")
        print("      --served-model-name Qwen3-VL-Reranker-2B \\")
        print("      --trust-remote-code \\")
        print("      --disable-radix-cache \\")
        print("      --chat-template examples/chat_template/qwen3_vl_reranker.jinja")
        return

    # Run examples
    rerank_text_only()
    rerank_with_images()
    rerank_multimodal_query()


if __name__ == "__main__":
    main()
