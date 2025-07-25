#!/usr/bin/env python3
"""Test script for Anthropic API compatibility in SGLang"""

import asyncio
import json
import requests
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:30000"
ANTHROPIC_ENDPOINT = f"{BASE_URL}/v1/messages"

def test_anthropic_simple_message():
    """Test basic Anthropic Messages API call"""
    print("Testing simple Anthropic message...")
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Message ID: {data.get('id')}")
            print(f"Content: {data.get('content', [{}])[0].get('text', 'No text')}")
            print(f"Usage: {data.get('usage')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_anthropic_system_message():
    """Test Anthropic Messages API with system prompt"""
    print("\nTesting Anthropic message with system prompt...")
    
    payload = {
        "model": "claude-sonnet-4-20250514", 
        "max_tokens": 150,
        "system": "You are a helpful assistant specialized in mathematics.",
        "messages": [
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Content: {data.get('content', [{}])[0].get('text', 'No text')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_anthropic_streaming():
    """Test Anthropic Messages API with streaming"""
    print("\nTesting Anthropic streaming...")
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "stream": True,
        "messages": [
            {
                "role": "user", 
                "content": "Count from 1 to 5"
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers, stream=True)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str.strip() != '[DONE]':
                            try:
                                data = json.loads(data_str)
                                event_type = data.get('type')
                                print(f"Event: {event_type}")
                                if event_type == 'content_block_delta':
                                    delta = data.get('delta', {})
                                    if 'text' in delta:
                                        print(f"Text: {delta['text']}", end='', flush=True)
                            except json.JSONDecodeError:
                                pass
            print("\nStreaming completed")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_anthropic_complex_content():
    """Test Anthropic Messages API with complex content blocks"""
    print("\nTesting Anthropic complex content...")
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this request and respond appropriately."
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Content: {data.get('content', [{}])[0].get('text', 'No text')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_anthropic_error_handling():
    """Test Anthropic API error handling"""
    print("\nTesting Anthropic error handling...")
    
    # Test missing required field
    payload = {
        "model": "claude-sonnet-4-20250514",
        # Missing max_tokens and messages
    }
    
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response = requests.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"Error type: {error_data.get('error', {}).get('type')}")
            print(f"Error message: {error_data.get('error', {}).get('message')}")
            return True
        else:
            print(f"Unexpected response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting Anthropic API compatibility tests...")
    print("=" * 50)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code != 200:
            print("SGLang server is not running or not healthy!")
            print("Please start the server with: python -m sglang.launch_server --model-path <your-model>")
            return
    except Exception as e:
        print(f"Cannot connect to SGLang server at {BASE_URL}: {e}")
        print("Please start the server with: python -m sglang.launch_server --model-path <your-model>")
        return
    
    tests = [
        test_anthropic_simple_message,
        test_anthropic_system_message,
        test_anthropic_complex_content,
        test_anthropic_streaming,
        test_anthropic_error_handling,
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
        
        print("-" * 30)
        time.sleep(1)  # Brief pause between tests
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Anthropic API integration is working correctly.")
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()