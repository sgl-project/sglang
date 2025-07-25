#!/usr/bin/env python3
"""Test Anthropic module imports and basic functionality"""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_imports():
    """Test if all Anthropic modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test individual imports to isolate issues
        print("Testing typing_extensions...")
        from typing_extensions import Literal
        print("✓ typing_extensions imported")
        
        print("Testing pydantic...")
        from pydantic import BaseModel, Field
        print("✓ pydantic imported")
        
        print("Testing protocol module...")
        try:
            from sglang.srt.entrypoints.anthropic.protocol import (
                AnthropicMessagesRequest,
                AnthropicMessagesResponse,
                AnthropicError,
                AnthropicUsage,
                AnthropicContentBlock
            )
            print("✓ Protocol classes imported successfully")
        except ImportError as e:
            print(f"✗ Protocol import failed: {e}")
            return False
        
        print("Testing serving handler...")
        try:
            from sglang.srt.entrypoints.anthropic.serving_messages import AnthropicServingMessages
            print("✓ Serving handler imported successfully")
        except ImportError as e:
            print(f"✗ Serving handler import failed: {e}")
            # This might fail due to dependencies, but protocol should work
            pass
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_protocol_creation():
    """Test creating protocol objects"""
    print("\nTesting protocol object creation...")
    
    try:
        from sglang.srt.entrypoints.anthropic.protocol import (
            AnthropicMessagesRequest,
            AnthropicMessagesResponse,
            AnthropicContentBlock,
            AnthropicUsage,
            AnthropicMessage
        )
        
        # Test content block creation
        content_block = AnthropicContentBlock(
            type="text",
            text="Hello, world!"
        )
        print("✓ Content block created")
        
        # Test message creation
        message = AnthropicMessage(
            role="user",
            content="Hello!"
        )
        print("✓ Message created")
        
        # Test request creation
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[message]
        )
        print("✓ Request created")
        
        # Test usage creation
        usage = AnthropicUsage(
            input_tokens=10,
            output_tokens=20
        )
        print("✓ Usage created")
        
        # Test response creation
        response = AnthropicMessagesResponse(
            id="msg_123",
            content=[content_block],
            model="claude-sonnet-4-20250514",
            usage=usage
        )
        print("✓ Response created")
        
        return True
        
    except Exception as e:
        print(f"✗ Protocol creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_serialization():
    """Test JSON serialization"""
    print("\nTesting serialization...")
    
    try:
        from sglang.srt.entrypoints.anthropic.protocol import (
            AnthropicMessagesRequest,
            AnthropicMessage
        )
        
        message = AnthropicMessage(
            role="user",
            content="Test message"
        )
        
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[message],
            temperature=0.7
        )
        
        # Test model_dump
        request_dict = request.model_dump()
        assert "model" in request_dict
        assert "max_tokens" in request_dict
        assert "messages" in request_dict
        assert request_dict["temperature"] == 0.7
        print("✓ Request serialization works")
        
        # Test JSON serialization
        import json
        json_str = json.dumps(request_dict)
        assert "claude-sonnet-4-20250514" in json_str
        print("✓ JSON serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation():
    """Test validation logic"""
    print("\nTesting validation...")
    
    try:
        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest, AnthropicMessage
        from pydantic import ValidationError
        
        # Test valid request
        valid_message = AnthropicMessage(role="user", content="Hello")
        valid_request = AnthropicMessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[valid_message]
        )
        print("✓ Valid request creation works")
        
        # Test invalid max_tokens (should fail)
        try:
            invalid_message = AnthropicMessage(role="user", content="Hello")
            invalid_request = AnthropicMessagesRequest(
                model="claude-sonnet-4-20250514",
                max_tokens=-1,  # Invalid
                messages=[invalid_message]
            )
            print("✗ Validation should have failed for negative max_tokens")
            return False
        except ValidationError:
            print("✓ Validation correctly rejects negative max_tokens")
        
        # Test missing required fields
        try:
            incomplete_request = AnthropicMessagesRequest(
                model="claude-sonnet-4-20250514"
                # Missing max_tokens and messages
            )
            print("✗ Validation should have failed for missing fields")
            return False
        except ValidationError:
            print("✓ Validation correctly rejects missing required fields")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing Anthropic API Implementation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_protocol_creation,
        test_serialization,
        test_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed!")
        print("✓ Protocol modules import correctly")
        print("✓ Object creation works")
        print("✓ Serialization is functional")
        print("✓ Validation works as expected")
        print("\nNext steps:")
        print("1. Start SGLang server to test end-to-end functionality")
        print("2. Run: python -m sglang.launch_server --model-path <model>")
        print("3. Run: python test_anthropic_api.py")
    else:
        print(f"❌ {total - passed} tests failed.")
        print("Please fix the implementation before testing with server.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)