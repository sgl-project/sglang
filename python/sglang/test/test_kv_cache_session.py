"""
Test script to verify semantic-aware KV cache pruning during multi-round conversation.

This script:
1. Starts a session with system prompt + Q1
2. Builds up conversation history: Q1→A1→Q2→A2→Q3→A3
3. Checks KV cache state between each round
4. Sends summary request with semantic_event
5. Verifies cache pruning after summary
"""

import requests
import json
import time

BASE_URL = "http://localhost:8080"


def get_kv_cache_memory():
    """Get current KV cache memory state."""
    url = f"{BASE_URL}/v1/kv_cache"
    params = {"include": "memory,tree"}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "kv_cache_state" in data and len(data["kv_cache_state"]) > 0:
                state = data["kv_cache_state"][0]
                memory = state.get("memory", {})
                tree = state.get("tree", {})
                return {
                    "available_slots": memory.get("available_slots", 0),
                    "used_slots": memory.get("used_slots", 0),
                    "total_slots": memory.get("total_slots", 0),
                    "tree_total_size": tree.get("total_size", 0),
                    "tree_evictable": tree.get("evictable_size", 0),
                    "tree_protected": tree.get("protected_size", 0),
                }
        return None
    except Exception as e:
        print(f"Error getting KV cache: {e}")
        return None


def print_cache_state(label, cache_state):
    """Print cache state in compact format."""
    if cache_state:
        used = cache_state["used_slots"]
        total = cache_state["total_slots"]
        avail = cache_state["available_slots"]
        tree_size = cache_state["tree_total_size"]
        util = (used / total * 100) if total > 0 else 0
        print(f"  [{label}] Used: {used}/{total} ({util:.1f}%) | "
              f"Avail: {avail} | Tree: {tree_size}")
    else:
        print(f"  [{label}] Failed to get cache state")


def send_chat_completion(messages, session_id=None, semantic_event=None, max_tokens=15):
    """Send a chat completion request."""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if session_id:
        payload["session_params"] = {"id": session_id}
    if semantic_event:
        payload["semantic_event"] = semantic_event

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            returned_session_id = None
            if "metadata" in data and "session_id" in data["metadata"]:
                returned_session_id = data["metadata"]["session_id"]

            # Extract assistant's response
            content = ""
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                    if not content:
                        content = choice["message"].get("reasoning_content", "")

            return {
                "success": True,
                "session_id": returned_session_id,
                "content": content,
                "data": data
            }
        else:
            print(f"Error: {response.status_code} - {response.text[:100]}")
            return {"success": False, "error": response.text}
    except Exception as e:
        print(f"Exception: {e}")
        return {"success": False, "error": str(e)}


def main():
    print("="*70)
    print("Multi-Round Conversation with KV Cache Monitoring")
    print("="*70)

    session_id = None
    conversation_history = []

    # Initial cache state
    print("\n[Initial] Checking KV cache state...")
    initial_cache = get_kv_cache_memory()
    print_cache_state("Initial", initial_cache)
    baseline_used = initial_cache["used_slots"] if initial_cache else 0

    # Round 1: System prompt + Q1 (auto-create session)
    print("\n[Round 1] System + Q1 (auto-create session)")
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    result = send_chat_completion(
        conversation_history,
        semantic_event="start",
        max_tokens=10
    )

    if not result["success"]:
        print("Failed!")
        return

    session_id = result["session_id"]
    
    # Validate session_id format
    if isinstance(session_id, list):
        print(f"  ✗ ERROR: Session ID is a list: {session_id}")
        print(f"    Expected: string, Got: list")
        # Extract the string from the list for backward compatibility
        session_id = session_id[0] if session_id else None
    elif isinstance(session_id, str):
        print(f"  ✓ Session ID format correct (string)")
    else:
        print(f"  ✗ ERROR: Session ID has unexpected type: {type(session_id)}")
        
    print(f"  Session ID: {session_id}")
    print(f"  A1: {result['content'][:50]}...")

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": result["content"]})

    time.sleep(0.3)
    cache_r1 = get_kv_cache_memory()
    print_cache_state("After R1", cache_r1)

    # Round 2: Q2
    print("\n[Round 2] Q2 (building history)")
    conversation_history.append({"role": "user", "content": "What about 3+3?"})

    result = send_chat_completion(
        conversation_history,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A2: {result['content'][:50]}...")
        conversation_history.append({"role": "assistant", "content": result["content"]})
    else:
        print("  Failed!")

    time.sleep(0.3)
    cache_r2 = get_kv_cache_memory()
    print_cache_state("After R2", cache_r2)

    # Round 3: Q3
    print("\n[Round 3] Q3 (building history)")
    conversation_history.append({"role": "user", "content": "And 4+4?"})

    result = send_chat_completion(
        conversation_history,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A3: {result['content'][:50]}...")
        conversation_history.append({"role": "assistant", "content": result["content"]})
    else:
        print("  Failed!")

    time.sleep(0.3)
    cache_r3 = get_kv_cache_memory()
    print_cache_state("After R3", cache_r3)

    # Round 4: Q4
    print("\n[Round 4] Q4 (building history)")
    conversation_history.append({"role": "user", "content": "What is 5+5?"})

    result = send_chat_completion(
        conversation_history,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A4: {result['content'][:50]}...")
        conversation_history.append({"role": "assistant", "content": result["content"]})
    else:
        print("  Failed!")

    time.sleep(0.3)
    cache_r4 = get_kv_cache_memory()
    print_cache_state("After R4", cache_r4)

    # Check cache before summary
    print("\n" + "-"*50)
    print("Cache before summary:")
    before_summary_cache = get_kv_cache_memory()
    print_cache_state("Before Summary", before_summary_cache)
    before_used = before_summary_cache["used_slots"] if before_summary_cache else 0
    before_tree_size = before_summary_cache["tree_total_size"] if before_summary_cache else 0
    print(f"  Total cache growth: +{before_used - baseline_used} tokens")
    print(f"  Tree size: {before_tree_size} nodes")
    print("-"*50)

    # Round 5: Summary request with semantic_event
    print("\n[Round 5] Summary request (semantic_event='summary')")
    summary_messages = conversation_history + [
        {"role": "user", "content": "Summarize our math discussion."}
    ]

    result = send_chat_completion(
        summary_messages,
        session_id=session_id,
        semantic_event="summary",
        max_tokens=15
    )

    if result["success"]:
        print(f"  Summary: {result['content'][:60]}...")
    else:
        print("  Failed!")

    time.sleep(0.5)  # Allow pruning to complete
    cache_summary = get_kv_cache_memory()
    print_cache_state("After Summary", cache_summary)

    # Check cache after summary
    print("\n" + "-"*50)
    print("Cache after summary:")
    after_summary_cache = get_kv_cache_memory()
    print_cache_state("After Summary", after_summary_cache)
    after_used = after_summary_cache["used_slots"] if after_summary_cache else 0
    after_tree_size = after_summary_cache["tree_total_size"] if after_summary_cache else 0
    reduction = before_used - after_used
    tree_reduction = before_tree_size - after_tree_size if 'before_tree_size' in locals() else 0
    print(f"  Cache reduction: -{reduction} tokens")
    print(f"  Tree size reduction: -{tree_reduction} nodes")
    print("-"*50)
    
    # Validate pruning occurred
    if after_tree_size >= before_tree_size:
        print("  ✗ ERROR: Tree size did not decrease after summary!")
        print(f"    Before: {before_tree_size} nodes, After: {after_tree_size} nodes")
        print("    Possible issues:")
        print("      - Pruning logic not triggered")
        print("      - semantic_event not properly detected")
        print("      - last_node not available for pruning")
    else:
        print(f"  ✓ Tree size decreased from {before_tree_size} to {after_tree_size} nodes")

    # Round 6: New question after summary
    print("\n[Round 6] New question after summary")
    new_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Now what is 6+6?"}
    ]

    result = send_chat_completion(
        new_messages,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A6: {result['content'][:50]}...")
    else:
        print("  Failed!")

    time.sleep(0.3)
    cache_final = get_kv_cache_memory()
    print_cache_state("Final", cache_final)

    # Summary report
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Baseline tokens:     {baseline_used}")
    print(f"Before summary:      {before_used} (+{before_used - baseline_used})")
    print(f"After summary:       {after_used} (-{reduction})")
    print(f"Final:               {cache_final['used_slots'] if cache_final else 'N/A'}")
    print()

    if reduction > 0 and tree_reduction > 0:
        print("✓ SUCCESS: Cache was pruned after summary event")
        print(f"  Pruned {reduction} tokens from KV cache")
        print(f"  Pruned {tree_reduction} nodes from radix tree")
    elif reduction == 0 and tree_reduction == 0:
        print("⚠ WARNING: No cache reduction detected")
        print("  Possible reasons:")
        print("    - Pruning not triggered (check semantic_event handling)")
        print("    - Nodes had lock_ref > 0 (other sessions using same prefix)")
        print("    - Summary request not matched to existing cache")
        print("    - Server not running latest code with pruning logic")
    elif reduction > 0 and tree_reduction == 0:
        print("ℹ INFO: Token usage decreased but tree size unchanged")
        print("  This may indicate normal cache eviction, not pruning")
    else:
        print("ℹ INFO: Cache increased (new tokens added)")

    print("="*70)


if __name__ == "__main__":
    main()
