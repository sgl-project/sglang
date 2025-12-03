#!/usr/bin/env python3
"""
Shape Profiler Helper - Shows all available commands and examples.

Usage:
    python profiler_helper.py
    python profiler_helper.py --command profile
    python profiler_helper.py --command analyze
"""

import argparse
import os
import sys


COMMANDS = {
    "profile": {
        "description": "Profile model inference and capture kernel shapes",
        "script": "profile_deepseek_shapes.py",
        "examples": [
            {
                "title": "DeepSeek V3 with TP=8",
                "command": "python profile_deepseek_shapes.py --model-path deepseek-ai/DeepSeek-V3 --tp-size 8",
            },
            {
                "title": "Qwen 14B with TP=2",
                "command": "python profile_deepseek_shapes.py --model-path Qwen/Qwen2.5-14B-Instruct --tp-size 2",
            },
            {
                "title": "Quick test with small model",
                "command": "python profile_deepseek_shapes.py --model-path Qwen/Qwen2.5-7B-Instruct --tp-size 1 --num-prompts 2 --max-tokens 20",
            },
        ],
    },
    "analyze": {
        "description": "Analyze shape logs and generate statistics",
        "script": "analyze_shapes.py",
        "examples": [
            {
                "title": "Basic analysis",
                "command": "python analyze_shapes.py deepseek_v3_tp8_shapes.jsonl",
            },
            {
                "title": "Show detailed shapes",
                "command": "python analyze_shapes.py deepseek_v3_tp8_shapes.jsonl --show-shapes --top 20",
            },
            {
                "title": "Filter by operation type",
                "command": "python analyze_shapes.py deepseek_v3_tp8_shapes.jsonl --filter-op attention",
            },
            {
                "title": "Find large tensor operations",
                "command": "python analyze_shapes.py deepseek_v3_tp8_shapes.jsonl --min-elements 1000000",
            },
        ],
    },
    "compare": {
        "description": "Compare shapes between different runs",
        "script": "compare_shapes.py",
        "examples": [
            {
                "title": "Compare different TP sizes",
                "command": "python compare_shapes.py tp4_shapes.jsonl tp8_shapes.jsonl --labels 'TP=4' 'TP=8'",
            },
            {
                "title": "Compare prefill vs decode",
                "command": "python compare_shapes.py prefill.jsonl decode.jsonl --labels 'Prefill' 'Decode'",
            },
        ],
    },
    "test": {
        "description": "Test the shape logger setup",
        "script": "test_shape_logger.py",
        "examples": [
            {
                "title": "Run all tests",
                "command": "python test_shape_logger.py",
            },
        ],
    },
}


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_section(title):
    """Print a section title."""
    print(f"\n{title}")
    print("-" * len(title))


def show_all_commands():
    """Show all available commands."""
    print_header("SGLang Kernel Shape Profiler - Command Reference")
    
    print("\nAvailable Commands:")
    print()
    
    for cmd_name, cmd_info in COMMANDS.items():
        print(f"  {cmd_name:12s} - {cmd_info['description']}")
    
    print("\n\nUsage:")
    print("  python profiler_helper.py                    # Show this help")
    print("  python profiler_helper.py --command profile  # Show profile examples")
    print("  python profiler_helper.py --command analyze  # Show analyze examples")
    
    print("\n\nQuick Start:")
    print("  1. Test the setup:")
    print("     python test_shape_logger.py")
    print()
    print("  2. Profile DeepSeek with TP=8:")
    print("     python profile_deepseek_shapes.py \\")
    print("         --model-path deepseek-ai/DeepSeek-V3 \\")
    print("         --tp-size 8 \\")
    print("         --num-prompts 3")
    print()
    print("  3. Analyze the results:")
    print("     python analyze_shapes.py deepseek_v3_tp8_shapes.jsonl")
    
    print_header("Documentation")
    print("\nFor detailed documentation, see:")
    print("  - README.md       : Full documentation")
    print("  - QUICKREF.md     : Quick reference guide")


def show_command_examples(command):
    """Show examples for a specific command."""
    if command not in COMMANDS:
        print(f"Error: Unknown command '{command}'")
        print(f"Available commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)
    
    cmd_info = COMMANDS[command]
    
    print_header(f"Command: {command}")
    print(f"\n{cmd_info['description']}")
    print(f"Script: {cmd_info['script']}")
    
    print_section("Examples")
    
    for i, example in enumerate(cmd_info['examples'], 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['command']}")
    
    print("\n\nFor more options, run:")
    print(f"  python {cmd_info['script']} --help")


def check_files_exist():
    """Check if required files exist."""
    required_files = [
        "torch_shape_logger.py",
        "profile_deepseek_shapes.py",
        "analyze_shapes.py",
        "compare_shapes.py",
        "test_shape_logger.py",
    ]
    
    missing = []
    for filename in required_files:
        if not os.path.exists(filename):
            missing.append(filename)
    
    if missing:
        print("\nWarning: Some required files are missing:")
        for filename in missing:
            print(f"  - {filename}")
        print("\nMake sure you're in the correct directory:")
        print("  cd examples/profiler/")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Shape Profiler Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--command",
        choices=list(COMMANDS.keys()),
        help="Show examples for a specific command",
    )
    
    args = parser.parse_args()
    
    # Check files
    if not check_files_exist():
        sys.exit(1)
    
    if args.command:
        show_command_examples(args.command)
    else:
        show_all_commands()


if __name__ == "__main__":
    main()
