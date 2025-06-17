"""
Token-in-Token-out VLM Engine Example

This example demonstrates how to use SGLang Engine with skip_tokenizer_init=True
for Vision-Language Models (VLMs). This is particularly useful for:

1. Custom multimodal tokenization pipelines
2. RLHF workflows with vision-language models
3. Performance optimization when you already have tokenized inputs
4. Integration with external multimodal processing systems

The example shows:
- Basic token-in-token-out generation with images
- Image URL processing (local image files can be used by providing file paths)
- Custom chat template handling
- Multimodal input processing with skip_tokenizer_init

Usage:
    python token_in_token_out_vlm_engine.py
"""

import argparse
import dataclasses
import logging
import os
from typing import List, Optional, Tuple

from transformers import AutoProcessor

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.server_args import ServerArgs

# Set up logging
logging.basicConfig(level=logging.INFO)

# Default model and image for testing
DEFAULT_MODEL_PATH = "Qwen/Qwen2-VL-2B"
DEFAULT_IMAGE_URL = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"


def validate_image_path(image_path):
    """Validate if image path exists (for local files) or is a valid URL."""
    if image_path.startswith(("http://", "https://")):
        # It's a URL, assume it's valid (could add URL validation if needed)
        return True
    else:
        # It's a local file path, check if it exists
        if os.path.exists(image_path):
            return True
        else:
            print(f"Warning: Local image file '{image_path}' not found.")
            print("Please ensure the file exists or use a valid URL.")
            return False


def get_input_ids_and_image_data(
    model_path: str,
    image_url: str = None,
    custom_text: str = None,
    trust_remote_code: bool = True,
) -> Tuple[List[int], List[str]]:
    """
    Prepare input token IDs and image data for VLM inference.

    Args:
        model_path: Path to the VLM model
        image_url: URL of the image to process
        custom_text: Custom text prompt (if None, uses default)
        trust_remote_code: Whether to trust remote code for model loading

    Returns:
        Tuple of (input_ids, image_data)
    """
    # Get the chat template for this model
    chat_template = get_chat_template_by_model_path(model_path)

    # Use default image if none provided
    if image_url is None:
        image_url = DEFAULT_IMAGE_URL

    # Use default text if none provided
    if custom_text is None:
        text = f"{chat_template.image_token}What is in this picture?"
    else:
        text = f"{chat_template.image_token}{custom_text}"

    # Note: image_data can accept both URLs and local file paths
    # For local images, use: image_data = ["/path/to/local/image.jpg"]
    # For URLs, use: image_data = ["https://example.com/image.jpg"]
    image_data = [image_url]

    print(f"Text prompt: {text}")
    print(f"Image URL: {image_url}")
    print(f"Image token: {chat_template.image_token}")

    # Load the processor to tokenize the text
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        print(f"Error loading processor for model '{model_path}': {e}")
        print("Please check:")
        print("1. Model path is correct")
        print("2. Model supports AutoProcessor")
        print("3. Required dependencies are installed")
        raise

    # Tokenize the text (note: we don't process the image here since
    # the engine will handle image processing when skip_tokenizer_init=True)
    input_ids = (
        processor.tokenizer(
            text=[text],
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )

    print(f"Input token IDs: {input_ids}")
    print(f"Number of input tokens: {len(input_ids)}")

    return input_ids, image_data


def basic_vlm_token_in_token_out_example(model_path, image_url=None):
    """Basic example of VLM token-in-token-out generation."""
    print("=" * 60)
    print("Basic VLM Token-in-Token-out Example")
    print("=" * 60)

    # Prepare input data
    input_ids, image_data = get_input_ids_and_image_data(
        model_path, image_url=image_url
    )

    # Create server args with skip_tokenizer_init=True
    server_args = ServerArgs(
        model_path=model_path,
        skip_tokenizer_init=True,
        trust_remote_code=True,
    )

    # Create engine
    with Engine(**dataclasses.asdict(server_args)) as engine:
        # Generate response
        sampling_params = {
            "temperature": 0.8,
            "max_new_tokens": 32,
        }

        output = engine.generate(
            input_ids=input_ids,
            image_data=image_data,
            sampling_params=sampling_params,
        )

        print(f"\nGeneration Results:")
        print(f"Output token IDs: {output['output_ids']}")
        print(f"Number of output tokens: {len(output['output_ids'])}")

        # Note: Since we used skip_tokenizer_init=True, we get token IDs as output
        # To decode them, we would need to use the processor separately


def multiple_images_example(model_path):
    """Example with multiple different images and prompts."""
    print("\n" + "=" * 60)
    print("Multiple Images Example")
    print("=" * 60)

    # Different images and prompts (mix of URLs and local file example)
    test_cases = [
        {"image_url": DEFAULT_IMAGE_URL, "text": "Describe this image in detail."},
        {
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "text": "What type of landscape is shown?",
        },
        # Example of how to use local image files:
        # {"image_url": "/path/to/your/local/image.jpg", "text": "Analyze this local image."},
    ]

    server_args = ServerArgs(
        model_path=model_path,
        skip_tokenizer_init=True,
        trust_remote_code=True,
    )

    with Engine(**dataclasses.asdict(server_args)) as engine:
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")

            # Validate image path before processing
            if not validate_image_path(test_case["image_url"]):
                print(f"Skipping test case {i+1} due to invalid image path.")
                continue

            input_ids, image_data = get_input_ids_and_image_data(
                model_path, test_case["image_url"], test_case["text"]
            )

            sampling_params = {
                "temperature": 0.7,
                "max_new_tokens": 50,
            }

            output = engine.generate(
                input_ids=input_ids,
                image_data=image_data,
                sampling_params=sampling_params,
            )

            print(f"Output token IDs: {output['output_ids']}")
            print(f"Number of output tokens: {len(output['output_ids'])}")

            # Check finish reason
            meta_info = output.get("meta_info", {})
            finish_reason = meta_info.get("finish_reason", "N/A")
            print(f"Finish reason: {finish_reason}")


def custom_sampling_vlm_example(model_path, image_url=None):
    """Example showing different sampling strategies with VLM."""
    print("\n" + "=" * 60)
    print("Custom Sampling VLM Example")
    print("=" * 60)

    input_ids, image_data = get_input_ids_and_image_data(
        model_path,
        image_url=image_url,
        custom_text="What objects can you identify in this image? List them.",
    )

    server_args = ServerArgs(
        model_path=model_path,
        skip_tokenizer_init=True,
        trust_remote_code=True,
    )

    with Engine(**dataclasses.asdict(server_args)) as engine:
        # Test different sampling strategies
        sampling_strategies = [
            {
                "name": "Greedy (temperature=0)",
                "params": {"temperature": 0.0, "max_new_tokens": 40},
            },
            {
                "name": "Creative (temperature=1.0)",
                "params": {"temperature": 1.0, "max_new_tokens": 40},
            },
            {
                "name": "Balanced (temperature=0.7, top_p=0.9)",
                "params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 40},
            },
        ]

        for strategy in sampling_strategies:
            print(f"\n--- {strategy['name']} ---")

            output = engine.generate(
                input_ids=input_ids,
                image_data=image_data,
                sampling_params=strategy["params"],
            )

            print(f"Sampling params: {strategy['params']}")
            print(f"Output token IDs: {output['output_ids']}")
            print(f"Number of output tokens: {len(output['output_ids'])}")

            # Check token statistics
            meta_info = output.get("meta_info", {})
            if "prompt_tokens" in meta_info:
                print(f"Prompt tokens: {meta_info['prompt_tokens']}")
            if "completion_tokens" in meta_info:
                print(f"Completion tokens: {meta_info['completion_tokens']}")


def main(model_path: Optional[str] = None, image_url: Optional[str] = None) -> None:
    """Run all VLM examples."""
    # Use defaults if not provided
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if image_url is None:
        image_url = DEFAULT_IMAGE_URL

    print("Token-in-Token-out VLM Engine Examples")
    print("This example demonstrates skip_tokenizer_init=True functionality for VLMs")
    print(f"Using model: {model_path}")
    print(f"Default image: {image_url}")

    print("\nNote: This example demonstrates token-level input/output.")
    print("To see decoded text, you would need to use the processor separately.")
    print("The focus here is on the token-in-token-out workflow.\n")

    # Run all examples
    try:
        basic_vlm_token_in_token_out_example(model_path, image_url)
        multiple_images_example(model_path)
        custom_sampling_vlm_example(model_path, image_url)

        print("\n" + "=" * 60)
        print("All VLM examples completed!")
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
        print("1. Install required dependencies: pip install sglang transformers")
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
    # Allow command line arguments for customization
    parser = argparse.ArgumentParser(description="VLM Token-in-Token-out Example")
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH, help="Path to the VLM model"
    )
    parser.add_argument(
        "--image-url", default=DEFAULT_IMAGE_URL, help="URL of the image to process"
    )

    args = parser.parse_args()

    # Pass arguments to main function instead of modifying globals
    main(args.model_path, args.image_url)
