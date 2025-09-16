#!/usr/bin/env python3
"""
Test script to verify SGLang config file integration.
"""

import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add the python directory to the path
sys.path.insert(0, '/Users/kahmadian/Documents/sglang/python')

from sglang.srt.config_parser import ConfigArgumentMerger
from sglang.srt.server_args import prepare_server_args


def test_config_parser():
    """Test the config parser functionality."""
    print("Testing config parser...")

    # Create a temporary config file
    config_data = {
        'model_path': 'microsoft/DialoGPT-medium',
        'host': '0.0.0.0',
        'port': 30000,
        'tensor_parallel_size': 2,
        'enable_flashinfer': True,
        'trust_remote_code': False
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Test config parser directly
        merger = ConfigArgumentMerger()
        config_args = merger._parse_yaml_config(config_file)
        print(f"Config args: {config_args}")

        # Test merging with CLI args
        cli_args = ['--config', config_file, '--max_num_seqs', '128']
        merged_args = merger.merge_config_with_args(cli_args)
        print(f"Merged args: {merged_args}")

        # Verify the merged args contain both config and CLI values
        assert '--model_path' in merged_args
        assert 'microsoft/DialoGPT-medium' in merged_args
        assert '--host' in merged_args
        assert '0.0.0.0' in merged_args
        assert '--port' in merged_args
        assert '30000' in merged_args
        assert '--tensor_parallel_size' in merged_args
        assert '2' in merged_args
        assert '--enable_flashinfer' in merged_args
        assert '--max_num_seqs' in merged_args
        assert '128' in merged_args

        print("Config parser test passed!")

    finally:
        os.unlink(config_file)


def test_server_args_integration():
    """Test the integration with server args."""
    print("\nTesting server args integration...")

    # Create a temporary config file
    config_data = {
        'model_path': 'microsoft/DialoGPT-medium',
        'host': '0.0.0.0',
        'port': 30000,
        'tensor_parallel_size': 1,
        'max_num_seqs': 256,
        'enable_flashinfer': True
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Test with config file
        argv = ['--config', config_file]
        server_args = prepare_server_args(argv)

        # Verify that config values were loaded
        assert server_args.model_path == 'microsoft/DialoGPT-medium'
        assert server_args.host == '0.0.0.0'
        assert server_args.port == 30000
        assert server_args.tensor_parallel_size == 1
        assert server_args.max_num_seqs == 256
        assert server_args.enable_flashinfer == True

        print("Server args integration test passed!")

    finally:
        os.unlink(config_file)


def test_cli_override():
    """Test that CLI arguments override config file values."""
    print("\nTesting CLI override...")

    # Create a temporary config file
    config_data = {
        'model_path': 'microsoft/DialoGPT-medium',
        'port': 30000,
        'tensor_parallel_size': 1
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Test CLI override (CLI should take precedence)
        argv = ['--config', config_file, '--port', '40000', '--tensor_parallel_size', '2']
        server_args = prepare_server_args(argv)

        # Verify that CLI values override config values
        assert server_args.model_path == 'microsoft/DialoGPT-medium'  # From config
        assert server_args.port == 40000  # From CLI (overrides config)
        assert server_args.tensor_parallel_size == 2  # From CLI (overrides config)

        print("CLI override test passed!")

    finally:
        os.unlink(config_file)


def test_error_handling():
    """Test error handling for invalid config files."""
    print("\nTesting error handling...")

    # Test non-existent config file
    try:
        argv = ['--config', 'non_existent.yaml']
        prepare_server_args(argv)
        assert False, "Should have raised an error for non-existent file"
    except ValueError as e:
        assert "Config file not found" in str(e)
            print("Non-existent file error handling passed!")

    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        invalid_yaml_file = f.name

    try:
        try:
            argv = ['--config', invalid_yaml_file]
            prepare_server_args(argv)
            assert False, "Should have raised an error for invalid YAML"
        except Exception as e:
            print(f"Invalid YAML error handling passed! Error: {e}")
    finally:
        os.unlink(invalid_yaml_file)


if __name__ == "__main__":
    print("Running SGLang config file integration tests...\n")

    try:
        test_config_parser()
        test_server_args_integration()
        test_cli_override()
        test_error_handling()

        print("\nAll tests passed! Config file integration is working correctly.")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
