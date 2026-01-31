"""
Token-in-Token-out LLM Engine Example

This example demonstrates how to use SGLang Engine with skip_tokenizer_init=True
to perform token-in-token-out inference for LLMs. This is particularly useful for:

1. RLHF (Reinforcement Learning from Human Feedback) workflows
2. Custom tokenization pipelines
3. Performance optimization when you already have tokenized inputs
4. Integration with external tokenization systems

The example shows:
- Basic token-in-token-out generation
- Batch processing with multiple prompts
- Streaming vs non-streaming generation
- Logprob extraction
- Multiple sampling (n > 1)
- EOS token handling

Usage:
    python token_in_token_out_llm_engine.py [--model-path MODEL_PATH]
"""

import argparse
import logging
from typing import Optional

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Default model path
DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def basic_token_in_token_out_example(model_path):
    """Basic example of token-in-token-out generation."""
    print("=" * 60)
    print("Basic Token-in-Token-out Example")
    print("=" * 60)

    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Get tokenizer separately (since we skip tokenizer init in engine)
    tokenizer = get_tokenizer(model_path)

    # Tokenize inputs manually
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]

    print(f"Original prompts: {prompts}")
    print(f"Tokenized inputs: {token_ids_list}")

    # Create engine with skip_tokenizer_init=True
    # This tells the engine to expect token IDs as input instead of text
    with sgl.Engine(model_path=model_path, skip_tokenizer_init=True) as llm:
        # Generate with token IDs as input
        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 32}
        outputs = llm.generate(
            input_ids=token_ids_list, sampling_params=sampling_params
        )

        # Print results
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            decoded_output = tokenizer.decode(
                output["output_ids"], skip_special_tokens=True
            )
            print(f"\nExample {i+1}:")
            print(f"  Original prompt: {prompt}")
            print(f"  Input token IDs: {token_ids_list[i]}")
            print(f"  Output token IDs: {output['output_ids']}")
            print(f"  Decoded output: {decoded_output}")
            print(
                f"  Finish reason: {output.get('meta_info', {}).get('finish_reason', 'N/A')}"
            )


def logprob_example(model_path):
    """Example showing how to extract logprobs with token-in-token-out."""
    print("\n" + "=" * 60)
    print("Logprob Extraction Example")
    print("=" * 60)

    tokenizer = get_tokenizer(model_path)

    with sgl.Engine(model_path=model_path, skip_tokenizer_init=True) as llm:
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt)

        print(f"Prompt: {prompt}")
        print(f"Input token IDs: {input_ids}")

        # Generate with logprobs
        sampling_params = {
            "temperature": 0.0,  # Use temperature 0 for deterministic output
            "max_new_tokens": 16,
        }

        output = llm.generate(
            input_ids=[input_ids],
            sampling_params=sampling_params,
            return_logprob=True,
            top_logprobs_num=3,  # Get top 3 logprobs for each token
            logprob_start_len=0,  # Start logprob calculation from beginning
        )[0]

        print(f"\nOutput token IDs: {output['output_ids']}")
        print(
            f"Decoded output: {tokenizer.decode(output['output_ids'], skip_special_tokens=True)}"
        )

        # Print logprob information
        meta_info = output.get("meta_info", {})
        if "input_token_logprobs" in meta_info:
            print(
                f"\nInput token logprobs length: {len(meta_info['input_token_logprobs'])}"
            )
        if "output_token_logprobs" in meta_info:
            print(
                f"Output token logprobs length: {len(meta_info['output_token_logprobs'])}"
            )
            print("First few output token logprobs:")
            for i, logprob_info in enumerate(meta_info["output_token_logprobs"][:3]):
                token_id = output["output_ids"][i]
                token_text = tokenizer.decode([token_id])
                print(
                    f"  Token {i}: '{token_text}' (ID: {token_id}) - logprob: {logprob_info}"
                )


def multiple_sampling_example(model_path):
    """Example showing multiple sampling (n > 1) with token-in-token-out."""
    print("\n" + "=" * 60)
    print("Multiple Sampling Example (n=3)")
    print("=" * 60)

    tokenizer = get_tokenizer(model_path)

    with sgl.Engine(model_path=model_path, skip_tokenizer_init=True) as llm:
        prompt = "Once upon a time"
        input_ids = tokenizer.encode(prompt)

        print(f"Prompt: {prompt}")
        print(f"Input token IDs: {input_ids}")

        # Generate multiple samples
        sampling_params = {
            "temperature": 0.8,
            "max_new_tokens": 20,
            "n": 3,  # Generate 3 different completions
        }

        outputs = llm.generate(
            input_ids=[input_ids],
            sampling_params=sampling_params,
        )

        print(f"\nGenerated {len(outputs)} completions:")
        for i, output in enumerate(outputs):
            decoded_output = tokenizer.decode(
                output["output_ids"], skip_special_tokens=True
            )
            print(f"\nCompletion {i+1}:")
            print(f"  Output token IDs: {output['output_ids']}")
            print(f"  Decoded output: {decoded_output}")
            print(
                f"  Finish reason: {output.get('meta_info', {}).get('finish_reason', 'N/A')}"
            )


def eos_token_handling_example(model_path):
    """Example showing EOS token handling with custom stop tokens."""
    print("\n" + "=" * 60)
    print("EOS Token Handling Example")
    print("=" * 60)

    tokenizer = get_tokenizer(model_path)

    with sgl.Engine(model_path=model_path, skip_tokenizer_init=True) as llm:
        prompt = "List three colors: 1."
        input_ids = tokenizer.encode(prompt)

        print(f"Prompt: {prompt}")
        print(f"Input token IDs: {input_ids}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")

        # Generate with custom stop tokens
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 50,
            "stop_token_ids": (
                [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
            ),  # Stop on EOS token if available
        }

        output = llm.generate(
            input_ids=[input_ids],
            sampling_params=sampling_params,
        )[0]

        decoded_output = tokenizer.decode(
            output["output_ids"], skip_special_tokens=True
        )
        print(f"\nOutput token IDs: {output['output_ids']}")
        print(f"Decoded output: {decoded_output}")

        # Check finish reason
        meta_info = output.get("meta_info", {})
        finish_reason = meta_info.get("finish_reason", {})
        print(f"Finish reason: {finish_reason}")

        if finish_reason.get("type") == "stop":
            print(f"Stopped due to stop token: {finish_reason.get('matched')}")
        elif finish_reason.get("type") == "length":
            print("Stopped due to max length")


def streaming_example(model_path):
    """Example showing streaming generation with token-in-token-out."""
    print("\n" + "=" * 60)
    print("Streaming Generation Example")
    print("=" * 60)

    tokenizer = get_tokenizer(model_path)

    with sgl.Engine(model_path=model_path, skip_tokenizer_init=True) as llm:
        prompt = "Write a short story about a robot:"
        input_ids = tokenizer.encode(prompt)

        print(f"Prompt: {prompt}")
        print(f"Input token IDs: {input_ids}")

        # Generate with streaming
        sampling_params = {
            "temperature": 0.7,
            "max_new_tokens": 50,
        }

        print(f"\nStreaming generation:")
        print("Generated tokens: ", end="", flush=True)

        # Use actual streaming with stream=True
        all_output_tokens = []
        for output in llm.generate(
            input_ids=[input_ids],
            sampling_params=sampling_params,
            stream=True,
        ):
            # Process each streaming chunk
            if hasattr(output, "output_ids") and output.output_ids:
                # SGLang's streaming API provides delta tokens
                new_tokens = output.output_ids
                all_output_tokens.extend(new_tokens)

                # Print new tokens as they arrive
                for token_id in new_tokens:
                    print(f"[{token_id}]", end=" ", flush=True)

                # New line every 10 tokens for readability
                if len(all_output_tokens) % 10 == 0:
                    print()

        print(f"\n\nComplete output token IDs: {all_output_tokens}")
        decoded_output = tokenizer.decode(all_output_tokens, skip_special_tokens=True)
        print(f"Complete decoded output: {decoded_output}")


def main(model_path: Optional[str] = None) -> None:
    """Run all examples."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    print("Token-in-Token-out LLM Engine Examples")
    print("This example demonstrates skip_tokenizer_init=True functionality")
    print(f"Using model: {model_path}")

    # Run all examples with error handling
    try:
        basic_token_in_token_out_example(model_path)
        logprob_example(model_path)
        multiple_sampling_example(model_path)
        eos_token_handling_example(model_path)
        streaming_example(model_path)

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        logging.exception("Model file not found")
        print(f"\n" + "=" * 60)
        print("Model File Not Found Error:")
        print(f"Error: {e}")
        print("=" * 60)
        print("\nSolutions:")
        print(f"1. Check if the model path '{model_path}' is correct")
        print("2. Ensure the model is downloaded and accessible")
        print("3. Try using a different model with --model-path")
        raise
    except ImportError as e:
        logging.exception("Import error - missing dependencies")
        print(f"\n" + "=" * 60)
        print("Missing Dependencies Error:")
        print(f"Error: {e}")
        print("=" * 60)
        print("\nSolutions:")
        print("1. Install required dependencies: pip install sglang")
        print("2. Check your Python environment")
        raise
    except RuntimeError as e:
        logging.exception("Runtime error - likely GPU/memory issue")
        print(f"\n" + "=" * 60)
        print("Runtime Error (likely GPU/Memory):")
        print(f"Error: {e}")
        print("=" * 60)
        print("\nSolutions:")
        print("1. Ensure you have sufficient GPU memory")
        print("2. Try with a smaller model")
        print("3. Check CUDA installation if using GPU")
        raise
    except Exception as e:
        logging.exception("Unexpected error occurred")
        print(f"\n" + "=" * 60)
        print("Unexpected Error:")
        print(f"Error: {e}")
        print("=" * 60)
        print("\nGeneral solutions:")
        print("1. Make sure the model path is correct and accessible")
        print("2. Ensure you have sufficient GPU memory")
        print("3. Check that all required dependencies are installed")
        print("4. Try with a smaller model if you're running into memory issues")
        print(f"5. Verify the model '{model_path}' is available")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Token-in-Token-out Example")
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH, help="Path to the LLM model"
    )

    args = parser.parse_args()
    main(args.model_path)
