#!/usr/bin/env python3
"""
Test script for memory utilities and OOM handling improvements.
"""

import logging
import sys
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_memory_utils():
    """Test the memory utilities without requiring actual GPU."""
    try:
        from sglang.srt.utils.memory_utils import (
            suggest_memory_optimizations,
            create_oom_error_message,
            get_memory_pressure_level,
        )
        
        # Test suggestion function
        config = {
            'mem_fraction_static': 0.9,
            'max_running_requests': 128,
            'chunked_prefill_size': None,
            'max_prefill_tokens': 16384,
        }
        
        suggestions = suggest_memory_optimizations(config, "high", "prefill")
        logger.info(f"Suggestions for high memory pressure during prefill: {suggestions}")
        
        # Test error message creation
        error_msg = create_oom_error_message(
            error_context="Prefill",
            tokens_requested=8192,
            available_tokens=4096,
            current_config=config
        )
        logger.info("Generated OOM error message:")
        print(error_msg)
        
        # Test with mock GPU memory
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.memory_allocated', return_value=int(8 * 1024**3)), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_device = Mock()
            mock_device.total_memory = int(10 * 1024**3)  # 10GB total
            mock_props.return_value = mock_device
            
            pressure = get_memory_pressure_level()
            logger.info(f"Memory pressure level with 8GB/10GB usage: {pressure}")
        
        logger.info("✓ Memory utilities test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory utilities test failed: {e}")
        return False

def test_import_fixes():
    """Test that the imports work correctly."""
    try:
        # Test schedule_batch imports
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        logger.info("✓ Schedule batch imports work")
        
        # Test memory pool imports  
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        logger.info("✓ Memory pool imports work")
        
        # Test server args imports
        from sglang.srt.server_args import ServerArgs
        logger.info("✓ Server args imports work")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False

def test_decorator():
    """Test the OOM decorator."""
    try:
        from sglang.srt.utils.memory_utils import handle_cuda_oom_gracefully
        
        @handle_cuda_oom_gracefully
        def dummy_function():
            return "success"
        
        result = dummy_function()
        assert result == "success"
        logger.info("✓ OOM decorator test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ OOM decorator test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Testing OOM handling improvements...")
    
    tests = [
        ("Import fixes", test_import_fixes),
        ("Memory utilities", test_memory_utils),
        ("OOM decorator", test_decorator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())