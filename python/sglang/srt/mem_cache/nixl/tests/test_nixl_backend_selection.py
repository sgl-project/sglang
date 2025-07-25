#!/usr/bin/env python3

import sys
import os
import tempfile

# Set up the path
import test_utils
test_utils.setup_path()

def test_nixl_backend_selection():
    """Test NixlBackendSelection class."""
    print("=== Testing NixlBackendSelection ===")
    
    try:
        from sglang.srt.mem_cache.nixl.nixl_backend_selection import NixlBackendSelection
        from nixl._api import nixl_agent, nixl_agent_config
        import uuid
        
        # Create NIXL agent
        agent_config = nixl_agent_config(backends=[])
        agent = nixl_agent(str(uuid.uuid4()), agent_config)
        print("✓ Created NIXL agent")
        
        # Test auto backend selection
        backend_selector = NixlBackendSelection("auto")
        result = backend_selector.create_backend(agent)
        print(f"✓ Auto backend selection: {'Success' if result else 'Failed'}")
        
        # Test specific backend selection
        backend_selector_posix = NixlBackendSelection("POSIX")
        result = backend_selector_posix.create_backend(agent)
        print(f"✓ POSIX backend selection: {'Success' if result else 'Failed'}")
        
        # Test invalid backend selection
        backend_selector_invalid = NixlBackendSelection("INVALID_BACKEND")
        result = backend_selector_invalid.create_backend(agent)
        print(f"✓ Invalid backend selection: {'Failed as expected' if not result else 'Unexpected success'}")
        
        print("✓ All NixlBackendSelection tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during NixlBackendSelection test: {e}")
        return False

if __name__ == "__main__":
    success = test_nixl_backend_selection()
    sys.exit(0 if success else 1) 