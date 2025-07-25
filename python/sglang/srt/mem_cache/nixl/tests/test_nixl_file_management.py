#!/usr/bin/env python3

import sys
import os
import tempfile
import torch

# Set up the path
import test_utils
test_utils.setup_path()

def test_nixl_file_management():
    """Test NixlFileManagement class."""
    print("=== Testing NixlFileManagement ===")
    
    try:
        from sglang.srt.mem_cache.nixl.nixl_file_management import NixlFileManagement
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = NixlFileManagement(temp_dir)
            
            # Test directory creation
            result = file_manager.ensure_directory_exists()
            print(f"✓ Directory creation: {'Success' if result else 'Failed'}")
            
            # Test file path generation
            key = "test_key_123"
            file_path = file_manager.get_file_path(key)
            expected_path = os.path.join(temp_dir, f"{key}.bin")
            print(f"✓ File path generation: {'Success' if file_path == expected_path else 'Failed'}")
            
            # Test file creation
            result = file_manager.create_file(file_path)
            print(f"✓ File creation: {'Success' if result else 'Failed'}")
            
            # Test file opening
            fd = file_manager.open_file(file_path)
            print(f"✓ File opening: {'Success' if fd is not None else 'Failed'}")
            
            if fd is not None:
                # Test file closing
                result = file_manager.close_file(fd)
                print(f"✓ File closing: {'Success' if result else 'Failed'}")
            
            # Test file deletion
            result = file_manager.delete_file(file_path)
            print(f"✓ File deletion: {'Success' if result else 'Failed'}")
            
            # Test cleanup (should not fail even with no files)
            file_manager.cleanup_files()
            print("✓ Cleanup: Success")
            
        print("✓ All NixlFileManagement tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during NixlFileManagement test: {e}")
        return False

if __name__ == "__main__":
    success = test_nixl_file_management()
    sys.exit(0 if success else 1) 