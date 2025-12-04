# SPDX-License-Identifier: Apache-2.0
"""
cache-dit integration test script.

This script tests cache-dit acceleration with different configurations
using environment variables.

Usage:
    # List available configurations and models
    python test_cache_dit.py --list-configs

    # Basic test with preset model (cache-dit disabled, baseline)
    python test_cache_dit.py --model qwen_image

    # Test with custom model path
    python test_cache_dit.py --model-path Qwen/Qwen-Image
    python test_cache_dit.py --model-path /path/to/local/model

    # Test with custom model and parameters
    python test_cache_dit.py --model-path Qwen/Qwen-Image --width 1024 --height 1024

    # Test with cache-dit enabled via environment
    SGLANG_CACHE_DIT_ENABLED=true python test_cache_dit.py --model-path Qwen/Qwen-Image

    # Test with cache-dit + SCM
    SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_SCM_PRESET=medium python test_cache_dit.py

    # Run specific cache-dit config
    python test_cache_dit.py --model-path Qwen/Qwen-Image --config cache_dit_scm_medium

    # Run all configurations (baseline + all cache-dit variants)
    python test_cache_dit.py --model-path Qwen/Qwen-Image --all
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestConfig:
    """Test configuration with environment variables."""
    name: str
    env_vars: dict[str, str]
    description: str


# Test configurations
TEST_CONFIGS = {
    "baseline": TestConfig(
        name="baseline",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "false",
        },
        description="Baseline without cache-dit",
    ),
    "cache_dit_basic": TestConfig(
        name="cache_dit_basic",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "true",
        },
        description="cache-dit with default parameters",
    ),
    "cache_dit_taylorseer": TestConfig(
        name="cache_dit_taylorseer",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "true",
            "SGLANG_CACHE_DIT_TAYLORSEER": "true",
            "SGLANG_CACHE_DIT_TS_ORDER": "1",
        },
        description="cache-dit with TaylorSeer order=1",
    ),
    "cache_dit_scm_slow": TestConfig(
        name="cache_dit_scm_slow",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "true",
            "SGLANG_CACHE_DIT_SCM_PRESET": "slow",
        },
        description="cache-dit with SCM preset=slow",
    ),
    "cache_dit_scm_medium": TestConfig(
        name="cache_dit_scm_medium",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "true",
            "SGLANG_CACHE_DIT_SCM_PRESET": "medium",
        },
        description="cache-dit with SCM preset=medium",
    ),
    "cache_dit_scm_fast": TestConfig(
        name="cache_dit_scm_fast",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "true",
            "SGLANG_CACHE_DIT_SCM_PRESET": "fast",
        },
        description="cache-dit with SCM preset=fast",
    ),
    "cache_dit_custom": TestConfig(
        name="cache_dit_custom",
        env_vars={
            "SGLANG_CACHE_DIT_ENABLED": "true",
            "SGLANG_CACHE_DIT_FN": "2",
            "SGLANG_CACHE_DIT_BN": "1",
            "SGLANG_CACHE_DIT_WARMUP": "4",
            "SGLANG_CACHE_DIT_RDT": "0.4",
            "SGLANG_CACHE_DIT_MC": "4",
        },
        description="cache-dit with custom DBCache parameters",
    ),
}

# Model configurations for testing
MODEL_CONFIGS = {
    "qwen_image": {
        "model_path": "Qwen/Qwen-Image",
        "modality": "image",
        "width": 512,
        "height": 512,
        "prompt": "A futuristic cityscape at sunset with flying cars",
    },
    "flux": {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "modality": "image",
        "width": 512,
        "height": 512,
        "prompt": "A curious raccoon in a forest",
    },
}


def run_generate_command(
    model_config: dict,
    test_config: TestConfig,
    output_dir: str = "cache_dit_test_outputs",
) -> tuple[Optional[float], bool]:
    """Run sglang generate command with specified configuration.

    Returns:
        Tuple of (duration_seconds, success)
    """
    output_name = f"{test_config.name}_{model_config['model_path'].split('/')[-1]}"

    command = [
        "sglang",
        "generate",
        f"--model-path={model_config['model_path']}",
        f"--prompt={model_config['prompt']}",
        f"--width={model_config['width']}",
        f"--height={model_config['height']}",
        f"--output-path={output_dir}",
        f"--output-file-name={output_name}",
        "--save-output",
        "--log-level=info",
    ]

    # Prepare environment with test config
    env = os.environ.copy()
    env.update(test_config.env_vars)

    print(f"\n{'='*60}")
    print(f"Running: {test_config.name}")
    print(f"Description: {test_config.description}")
    print(f"Environment variables:")
    for k, v in test_config.env_vars.items():
        print(f"  {k}={v}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    duration = None

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        ) as process:
            for line in process.stdout:
                sys.stdout.write(line)
                # Parse duration from output if available
                if "Pixel data generated" in line:
                    words = line.split(" ")
                    try:
                        duration = float(words[-2])
                    except (ValueError, IndexError):
                        pass

        success = process.returncode == 0
        if duration is None:
            duration = time.time() - start_time

    except Exception as e:
        print(f"Error running command: {e}")
        return None, False

    return duration, success


def verify_output(
    model_config: dict,
    test_config: TestConfig,
    output_dir: str = "cache_dit_test_outputs",
) -> bool:
    """Verify the output file exists and is valid."""
    output_name = f"{test_config.name}_{model_config['model_path'].split('/')[-1]}"

    if model_config["modality"] == "image":
        ext = "jpg"
    else:
        ext = "mp4"

    output_path = Path(output_dir) / f"{output_name}.{ext}"

    if not output_path.exists():
        print(f"Output file not found: {output_path}")
        return False

    file_size = output_path.stat().st_size
    if file_size < 1000:  # At least 1KB
        print(f"Output file too small: {file_size} bytes")
        return False

    print(f"Output verified: {output_path} ({file_size} bytes)")
    return True


def run_single_test(
    model_name: str,
    config_name: str,
    output_dir: str = "cache_dit_test_outputs",
) -> dict:
    """Run a single test with specified model and configuration."""
    model_config = MODEL_CONFIGS.get(model_name)
    test_config = TEST_CONFIGS.get(config_name)

    if not model_config:
        print(f"Unknown model: {model_name}")
        return {"success": False, "error": f"Unknown model: {model_name}"}

    if not test_config:
        print(f"Unknown config: {config_name}")
        return {"success": False, "error": f"Unknown config: {config_name}"}

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    duration, success = run_generate_command(model_config, test_config, output_dir)

    if success:
        verified = verify_output(model_config, test_config, output_dir)
    else:
        verified = False

    return {
        "model": model_name,
        "config": config_name,
        "duration": duration,
        "success": success,
        "verified": verified,
    }


def run_all_tests(
    model_name: str = "qwen_image",
    output_dir: str = "cache_dit_test_outputs",
) -> list[dict]:
    """Run all test configurations for a model."""
    results = []

    for config_name in TEST_CONFIGS:
        result = run_single_test(model_name, config_name, output_dir)
        results.append(result)

    return results


def print_results(results: list[dict]):
    """Print test results as a markdown table."""
    print("\n" + "="*80)
    print("## cache-dit Test Results")
    print("="*80 + "\n")

    print("| Config | Model | Duration (s) | Success | Verified |")
    print("|--------|-------|--------------|---------|----------|")

    for r in results:
        duration_str = f"{r['duration']:.2f}" if r['duration'] else "N/A"
        success_str = "Yes" if r['success'] else "No"
        verified_str = "Yes" if r['verified'] else "No"
        print(f"| {r['config']:<20} | {r['model']:<10} | {duration_str:<12} | {success_str:<7} | {verified_str:<8} |")

    print()

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r['success'] and r['verified'])
    print(f"**Summary**: {passed}/{total} tests passed")

    # Performance comparison
    baseline = next((r for r in results if r['config'] == 'baseline'), None)
    if baseline and baseline['duration']:
        print("\n### Speedup vs Baseline")
        print("| Config | Speedup |")
        print("|--------|---------|")
        for r in results:
            if r['duration'] and r['config'] != 'baseline':
                speedup = baseline['duration'] / r['duration']
                print(f"| {r['config']:<20} | {speedup:.2f}x |")


def main():
    parser = argparse.ArgumentParser(description="cache-dit integration test")
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Preset model to test (use --model-path for custom models)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Custom model path (e.g., Qwen/Qwen-Image or /path/to/local/model)",
    )
    parser.add_argument(
        "--modality",
        default="image",
        choices=["image", "video"],
        help="Output modality (default: image)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width (default: 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height (default: 512)",
    )
    parser.add_argument(
        "--prompt",
        default="A futuristic cityscape at sunset with flying cars",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--config",
        default=None,
        choices=list(TEST_CONFIGS.keys()),
        help="Specific config to test (default: use env vars or baseline)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test configurations",
    )
    parser.add_argument(
        "--output-dir",
        default="cache_dit_test_outputs",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available test configurations",
    )

    args = parser.parse_args()

    if args.list_configs:
        print("Available test configurations:")
        print("-" * 60)
        for name, config in TEST_CONFIGS.items():
            print(f"\n{name}:")
            print(f"  Description: {config.description}")
            print(f"  Environment variables:")
            for k, v in config.env_vars.items():
                print(f"    {k}={v}")
        print("\nAvailable preset models:")
        print("-" * 60)
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name}: {config['model_path']}")
        return

    # Determine model config
    if args.model_path:
        # Custom model path
        model_name = args.model_path.split("/")[-1]
        model_config = {
            "model_path": args.model_path,
            "modality": args.modality,
            "width": args.width,
            "height": args.height,
            "prompt": args.prompt,
        }
        # Add to MODEL_CONFIGS temporarily
        MODEL_CONFIGS["custom"] = model_config
        model_key = "custom"
    elif args.model:
        model_key = args.model
    else:
        # Default to qwen_image
        model_key = "qwen_image"

    if args.all:
        results = run_all_tests(model_key, args.output_dir)
        print_results(results)
    elif args.config:
        result = run_single_test(model_key, args.config, args.output_dir)
        print_results([result])
    else:
        # Check if cache-dit is enabled via environment
        if os.environ.get("SGLANG_CACHE_DIT_ENABLED", "").lower() == "true":
            # Create config from current environment
            env_config = TestConfig(
                name="env_config",
                env_vars={
                    k: v for k, v in os.environ.items()
                    if k.startswith("SGLANG_CACHE_DIT_")
                },
                description="Configuration from environment variables",
            )
            TEST_CONFIGS["env_config"] = env_config
            result = run_single_test(model_key, "env_config", args.output_dir)
        else:
            # Run baseline
            result = run_single_test(model_key, "baseline", args.output_dir)
        print_results([result])


if __name__ == "__main__":
    main()
