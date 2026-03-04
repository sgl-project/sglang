"""
Test script to verify semantic-aware KV cache pruning using AsyncOpenAI client.

This script mirrors test_kv_cache_session.py but uses the official AsyncOpenAI
client with extra_body parameter to pass custom parameters (semantic_event, session_params).

This validates that the semantic event handling works through the standard OpenAI client interface.
"""

import asyncio
import json
import time
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8080/v1"  # Include /v1 in base URL, no trailing slash
API_KEY = "dummy"  # SGLang doesn't require a real API key


class KVCacheMonitor:
    """Monitor KV cache state using HTTP endpoint."""

    def __init__(self, base_url):
        # Remove /v1 suffix if present, since kv_cache endpoint is at root
        self.base_url = base_url.rstrip('/v1').rstrip('/')

    async def get_state(self):
        """Get current KV cache memory state."""
        import aiohttp
        url = f"{self.base_url}/v1/kv_cache"
        params = {"include": "memory,tree"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
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


async def send_chat_completion(
    client: AsyncOpenAI,
    messages,
    session_id=None,
    semantic_event=None,
    max_tokens=15
):
    """Send a chat completion request using AsyncOpenAI client."""

    # Build extra_body with custom parameters
    extra_body = {}
    if session_id:
        extra_body["session_params"] = {"id": session_id}
    if semantic_event:
        extra_body["semantic_event"] = semantic_event

    try:
        print(f"  [Debug] Sending request with extra_body: {extra_body}")
        response = await client.chat.completions.create(
            model="default",
            messages=messages,
            max_completion_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )

        # Extract session ID from response metadata if available
        returned_session_id = None
        if hasattr(response, 'metadata') and response.metadata:
            returned_session_id = response.metadata.get('session_id')

        # Extract assistant's response
        content = ""
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and choice.message:
                content = choice.message.content or ""

        return {
            "success": True,
            "session_id": returned_session_id,
            "content": content,
            "response": response
        }
    except Exception as e:
        print(f"Error in chat completion: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def main():
    print("="*70)
    print("Multi-Round Conversation with KV Cache Monitoring (AsyncOpenAI)")
    print("="*70)

    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    print(f"  [Debug] Client base_url: {client.base_url}")
    print(f"  [Debug] Expected full URL: {client.base_url}/chat/completions")

    # Initialize cache monitor
    cache_monitor = KVCacheMonitor(BASE_URL)

    session_id = None
    conversation_history = []

    # Initial cache state
    print("\n[Initial] Checking KV cache state...")
    initial_cache = await cache_monitor.get_state()
    print_cache_state("Initial", initial_cache)
    # Use available_slots to track cache usage (decreasing = more used)
    baseline_avail = initial_cache["available_slots"] if initial_cache else 0

    # Round 1: System prompt + Q1 (auto-create session)
    print("\n[Round 1] System + Q1 (auto-create session)")
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    result = await send_chat_completion(
        client,
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

    await asyncio.sleep(0.3)
    cache_r1 = await cache_monitor.get_state()
    print_cache_state("After R1", cache_r1)

    # Round 2: Q2
    print("\n[Round 2] Q2 (building history)")
    conversation_history.append({"role": "user", "content": "What about 3+3?"})

    result = await send_chat_completion(
        client,
        conversation_history,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A2: {result['content'][:50]}...")
        conversation_history.append({"role": "assistant", "content": result["content"]})
    else:
        print("  Failed!")

    await asyncio.sleep(0.3)
    cache_r2 = await cache_monitor.get_state()
    print_cache_state("After R2", cache_r2)

    # Round 3: Q3
    print("\n[Round 3] Q3 (building history)")
    conversation_history.append({"role": "user", "content": "And 4+4?"})

    result = await send_chat_completion(
        client,
        conversation_history,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A3: {result['content'][:50]}...")
        conversation_history.append({"role": "assistant", "content": result["content"]})
    else:
        print("  Failed!")

    await asyncio.sleep(0.3)
    cache_r3 = await cache_monitor.get_state()
    print_cache_state("After R3", cache_r3)

    # Round 4: Q4
    print("\n[Round 4] Q4 (building history)")
    conversation_history.append({"role": "user", "content": "What is 5+5?"})

    result = await send_chat_completion(
        client,
        conversation_history,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A4: {result['content'][:50]}...")
        conversation_history.append({"role": "assistant", "content": result["content"]})
    else:
        print("  Failed!")

    await asyncio.sleep(0.3)
    cache_r4 = await cache_monitor.get_state()
    print_cache_state("After R4", cache_r4)

    # Check cache before summary
    print("\n" + "-"*50)
    print("Cache before summary:")
    before_summary_cache = await cache_monitor.get_state()
    print_cache_state("Before Summary", before_summary_cache)
    before_avail = before_summary_cache["available_slots"] if before_summary_cache else 0
    before_tree_size = before_summary_cache["tree_total_size"] if before_summary_cache else 0
    # Calculate cache usage: baseline - current_available (lower available = more used)
    cache_usage = baseline_avail - before_avail
    print(f"  Total cache usage: {cache_usage} slots (available decreased from {baseline_avail} to {before_avail})")
    print(f"  Tree size: {before_tree_size} nodes")
    print("-"*50)

    # Round 5: Summary request with semantic_event
    print("\n[Round 5] Summary request (semantic_event='summary')")
    summary_messages = conversation_history + [
        {"role": "user", "content": "Summarize our math discussion."}
    ]

    result = await send_chat_completion(
        client,
        summary_messages,
        session_id=session_id,
        semantic_event="summary",
        max_tokens=15
    )

    if result["success"]:
        print(f"  Summary: {result['content'][:60]}...")
    else:
        print("  Failed!")

    await asyncio.sleep(0.5)  # Allow pruning to complete
    cache_summary = await cache_monitor.get_state()
    print_cache_state("After Summary", cache_summary)

    # Check cache after summary
    print("\n" + "-"*50)
    print("Cache after summary:")
    after_summary_cache = await cache_monitor.get_state()
    print_cache_state("After Summary", after_summary_cache)
    after_avail = after_summary_cache["available_slots"] if after_summary_cache else 0
    after_tree_size = after_summary_cache["tree_total_size"] if after_summary_cache else 0
    # Calculate reduction: after_available - before_available (higher after = more freed)
    avail_recovery = after_avail - before_avail
    tree_reduction = before_tree_size - after_tree_size
    print(f"  Available slots recovered: +{avail_recovery} (from {before_avail} to {after_avail})")
    print(f"  Tree size reduction: -{tree_reduction} nodes (from {before_tree_size} to {after_tree_size})")
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

    result = await send_chat_completion(
        client,
        new_messages,
        session_id=session_id,
        max_tokens=10
    )

    if result["success"]:
        print(f"  A6: {result['content'][:50]}...")
    else:
        print("  Failed!")

    await asyncio.sleep(0.3)
    cache_final = await cache_monitor.get_state()
    print_cache_state("Final", cache_final)

    # Summary report
    print("\n" + "="*70)
    print("TEST SUMMARY (AsyncOpenAI Client)")
    print("="*70)
    final_avail = cache_final['available_slots'] if cache_final else 0
    total_avail_recovered = final_avail - baseline_avail if cache_final else 0
    print(f"Baseline available:  {baseline_avail}")
    print(f"Before summary:      {before_avail} (usage: {cache_usage} slots)")
    print(f"After summary:       {after_avail} (recovered: {avail_recovery} slots)")
    print(f"Final:               {final_avail} (new usage after Round 6: {baseline_avail - final_avail} slots)")
    print(f"Tree size change:    {before_tree_size} → {after_tree_size} nodes ({tree_reduction} pruned)")
    print()

    if tree_reduction > 0:
        print("✓ SUCCESS: Cache was pruned after summary event")
        print(f"  Recovered {avail_recovery} available slots")
        print(f"  Pruned {tree_reduction} nodes from radix tree")
    elif tree_reduction == 0 and avail_recovery == 0:
        print("⚠ WARNING: No cache reduction detected")
        print("  Possible reasons:")
        print("    - Pruning not triggered (check semantic_event handling)")
        print("    - Nodes had lock_ref > 0 (other sessions using same prefix)")
        print("    - Summary request not matched to existing cache")
        print("    - Server not running latest code with pruning logic")
    elif tree_reduction == 0 and avail_recovery > 0:
        print("ℹ INFO: Available slots increased but tree size unchanged")
        print("  This may indicate normal cache eviction, not pruning")
    else:
        print("ℹ INFO: Cache increased (new tokens added)")

    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
