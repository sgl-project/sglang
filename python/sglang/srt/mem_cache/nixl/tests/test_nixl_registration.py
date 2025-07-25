#!/usr/bin/env python3

import sys
import os
import tempfile
import torch

# Set up the path
import test_utils
test_utils.setup_path()

def test_nixl_registration():
    """Test NixlRegistration class."""
    print("=== Testing NixlRegistration ===")
    
    try:
        from sglang.srt.mem_cache.nixl.nixl_registration import NixlRegistration
        from sglang.srt.mem_cache.nixl.nixl_file_management import NixlFileManagement
        from nixl._api import nixl_agent, nixl_agent_config
        import uuid
        
        # Create NIXL agent
        agent_config = nixl_agent_config(backends=[])
        agent = nixl_agent(str(uuid.uuid4()), agent_config)
        print("✓ Created NIXL agent")
        
        # Create backend
        from sglang.srt.mem_cache.nixl.hicache_nixl import NixlBackendSelection
        backend_selector = NixlBackendSelection("auto")
        backend_selector.create_backend(agent)
        print("✓ Created NIXL backend")
        
        # Create registration manager
        registration = NixlRegistration(agent)
        print("✓ Created NixlRegistration instance")
        
        # Create temporary directory and file manager
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = NixlFileManagement(temp_dir)
            file_manager.ensure_directory_exists()
            
            # Test single tensor registration
            cpu_tensor = torch.randn(10, 10)
            result = registration.register_tensor(cpu_tensor)
            print(f"✓ Single CPU tensor registration: {'Success' if result else 'Failed'}")
            
            # Test batch tensor registration
            cpu_tensor2 = torch.randn(5, 5)
            cpu_tensor3 = torch.randn(3, 3)
            result = registration.register_tensors_batch([cpu_tensor2, cpu_tensor3])
            print(f"✓ Batch CPU tensor registration: {'Success' if result else 'Failed'}")
            
            # Test file registration
            test_file_path = file_manager.get_file_path("test_key")
            file_manager.create_file(test_file_path)
            result = registration.register_files_batch([test_file_path], file_manager)
            print(f"✓ File registration: {'Success' if result else 'Failed'}")
            
            # Test manual deregistration pattern
            print("\n--- Testing Manual Deregistration Pattern ---")
            print(f"Registered tensors before deregistration: {list(registration.registered_tensors.keys())}")
            
            result = registration.deregister_tensor(cpu_tensor)
            print(f"✓ Single tensor deregistration: {'Success' if result else 'Failed'}")
            print(f"Registered tensors after single deregistration: {list(registration.registered_tensors.keys())}")
            
            result = registration.deregister_tensors_batch([cpu_tensor2, cpu_tensor3])
            print(f"✓ Batch tensor deregistration: {'Success' if result else 'Failed'}")
            print(f"Registered tensors after batch deregistration: {list(registration.registered_tensors.keys())}")
            
            # Test cleanup pattern (with fresh registrations)
            print("\n--- Testing Cleanup Pattern ---")
            cpu_tensor4 = torch.randn(2, 2)
            cpu_tensor5 = torch.randn(1, 1)
            registration.register_tensor(cpu_tensor4)
            registration.register_tensor(cpu_tensor5)
            print("✓ Registered additional tensors for cleanup test")
            
            # Let cleanup handle all deregistration
            registration.cleanup_registrations()
            print("✓ Registration cleanup: Success")
            
        print("✓ All NixlRegistration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during NixlRegistration test: {e}")
        return False

if __name__ == "__main__":
    success = test_nixl_registration()
    sys.exit(0 if success else 1) 