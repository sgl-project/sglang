"""
Profile model kernel shapes using DIRECT model inference (no Engine wrapper).

This approach loads the model directly and runs inference in the current process,
allowing the ShapeLogger to properly capture all tensor operations.

Usage:
    python profile_model_direct.py \
        --model-path Qwen/Qwen2.5-14B-Instruct \
        --tp-size 8 \
        --num-prompts 3 \
        --max-tokens 50 \
        --output-file shapes.jsonl
"""

import argparse
import json
import os
import sys
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add profiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_shape_logger import CompactShapeLogger, ShapeLogger, analyze_shape_log


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    print("(This may take a few minutes...)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model in bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    """Generate text for given prompts."""
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "generated": generated_text,
        })
        
        print(f"  ✓ Generated {len(generated_text)} characters")
    
    return results


def main(
    model_path: str,
    output_file: str,
    num_prompts: int = 3,
    max_tokens: int = 50,
    verbose: bool = False,
    compact: bool = True,
    analyze_after: bool = True,
):
    """Main profiling function."""
    print("=" * 80)
    print("Direct Model Kernel Shape Profiler")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output file: {output_file}")
    print(f"Compact mode: {compact}")
    print(f"Verbose mode: {verbose}")
    print(f"Prompts: {num_prompts}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 80)
    print()
    
    # Sample prompts
    all_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Explain quantum computing in simple terms:",
    ]
    prompts = all_prompts[:num_prompts]
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Choose logger
    LoggerClass = CompactShapeLogger if compact else ShapeLogger
    
    print("\n" + "=" * 80)
    print("Starting inference with shape logging...")
    print("=" * 80)
    
    # Run inference with shape logging
    logger = LoggerClass(output_file=output_file, verbose=verbose)
    try:
        with logger:
            results = generate_text(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_tokens,
            )
        
        print("\n" + "=" * 80)
        print("Generation Results")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n[{i}/{len(results)}]")
            print(f"Prompt: {result['prompt']}")
            generated = result['generated']
            display = generated[:150] + '...' if len(generated) > 150 else generated
            print(f"Generated: {display}")
        
        # Print summary
        summary = logger.get_summary()
        print("\n" + "=" * 80)
        print("Shape Logging Summary")
        print("=" * 80)
        print(f"Total operations captured: {summary['total_operations']:,}")
        print(f"Unique operations: {summary['unique_operations']}")
        
        if summary['operation_counts']:
            print(f"\nTop 10 most frequent operations:")
            sorted_ops = sorted(
                summary['operation_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for op_name, count in sorted_ops[:10]:
                display_name = op_name if len(op_name) <= 60 else op_name[:57] + "..."
                print(f"  {count:8,d} : {display_name}")
        
        # Analyze if requested
        if analyze_after and summary['total_operations'] > 0:
            print("\n" + "=" * 80)
            print("Detailed Analysis")
            print("=" * 80)
            analyze_shape_log(output_file)
        
        # Check file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"\n✓ Log file created: {output_file} ({file_size:,} bytes)")
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)
    print("Clearing CUDA cache...")
    del model
    torch.cuda.empty_cache()
    print("✓ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile kernel shapes using direct model inference"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="model_shapes.jsonl",
        help="Output JSONL file for shape logs",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print shapes to console",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Disable compact logging",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip automatic analysis",
    )
    
    args = parser.parse_args()
    
    # Set default output file
    output_file = args.output_file
    if output_file == "model_shapes.jsonl":
        model_name = args.model_path.split('/')[-1].replace('-', '_').lower()
        output_file = f"{model_name}_direct_shapes.jsonl"
    
    main(
        model_path=args.model_path,
        output_file=output_file,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        compact=not args.no_compact,
        analyze_after=not args.no_analyze,
    )
