import requests
import json

# Test session params propagation with auto-create session
url = "http://localhost:8080/v1/completions"

headers = {
    "Content-Type": "application/json"
}

# Test 1: Auto-create session - should return session_id in metadata
print("=" * 60)
print("Test 1: Auto-create session (no id provided)")
print("=" * 60)
payload_auto_create = {
    "model": "default",
    "prompt": "Hello, world!",
    "max_tokens": 10,
    "semantic_event": "start"
}

response = requests.post(url, json=payload_auto_create, headers=headers)
print("Response:")
response_json = response.json()
print(json.dumps(response_json, indent=2))
print(f"Status code: {response.status_code}")

# Extract session_id from metadata
session_id = None
if "metadata" in response_json and "session_id" in response_json["metadata"]:
    session_id = response_json["metadata"]["session_id"]
    print(f"\n✓ Auto-created session_id: {session_id}")
else:
    print("\n✗ No session_id in response metadata")

# Test 2: Use the auto-created session ID for subsequent request
if session_id:
    print("\n" + "=" * 60)
    print("Test 2: Use auto-created session ID")
    print("=" * 60)
    payload_use_session = {
        "model": "default",
        "prompt": "How are you?",
        "max_tokens": 10,
        "session_params": {
            "id": session_id
        }
    }

    response = requests.post(url, json=payload_use_session, headers=headers)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status code: {response.status_code}")

# Test 3: Send semantic_event with existing session
if session_id:
    print("\n" + "=" * 60)
    print("Test 3: Send semantic_event with existing session")
    print("=" * 60)
    payload_semantic = {
        "model": "default",
        "prompt": "Summarize the conversation",
        "max_tokens": 10,
        "session_params": {
            "id": session_id
        },
        "semantic_event": "summary"
    }

    response = requests.post(url, json=payload_semantic, headers=headers)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status code: {response.status_code}")

# Test 4: Chat completions with auto-create session
print("\n" + "=" * 60)
print("Test 4: Chat completions with auto-create session")
print("=" * 60)
url_chat = "http://localhost:8080/v1/chat/completions"

payload_chat = {
    "model": "default",
    "messages": [
        {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 10,
    "semantic_event": "start"
}

response = requests.post(url_chat, json=payload_chat, headers=headers)
print("Chat Response:")
response_json = response.json()
print(json.dumps(response_json, indent=2))
print(f"Chat status code: {response.status_code}")

# Extract session_id from chat response
if "metadata" in response_json and "session_id" in response_json["metadata"]:
    chat_session_id = response_json["metadata"]["session_id"]
    print(f"\n✓ Auto-created chat session_id: {chat_session_id}")

    # Test 5: Send semantic_event in chat completions
    print("\n" + "=" * 60)
    print("Test 5: Send semantic_event in chat completions")
    print("=" * 60)
    payload_chat_summary = {
        "model": "default",
        "messages": [
            {"role": "user", "content": "Summarize our conversation"}
        ],
        "max_tokens": 10,
        "session_params": {
            "id": chat_session_id
        },
        "semantic_event": "summary"
    }

    response = requests.post(url_chat, json=payload_chat_summary, headers=headers)
    print("Chat Summary Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Chat status code: {response.status_code}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
