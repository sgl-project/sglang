# Agentic Cache Control: Router-to-HiCache Command Plane

**Linear Issue**: [DYN-1986: Agentic Events from Router to KVBM](https://linear.app/nvidia/issue/DYN-1986/agentic-events-from-router-to-kvbm)

**Date**: 2025-02-04

**Authors**: Ishan Dhanani

---

## Overview

This document describes the design for implementing a **reverse control plane** that enables the Dynamo Router to send "intent signals" to SGLang's HiRadixCache. This complements the existing upward event flow (HiCache → Router) with downward commands (Router → HiCache) to optimize block lifecycles based on application-level semantics.

### Current State

KV events currently flow **upward** only:

```
SGLang HiRadixCache → ZMQ → Dynamo Event Plane → Router KvIndexer
```

The Router uses these events to build a prefix tree for KV-aware routing decisions.

### Goal

Enable the Router to send **downward** commands to HiCache for proactive cache management:

```
Router → NATS → Dynamo SGLang Wrapper → HiRadixCache
```

---

## Motivation

Advanced agentic workflows require cache lifecycle optimization that cannot be achieved through passive LRU eviction alone:

| Use Case | Problem | Solution |
|----------|---------|----------|
| **Context Rewrite** | Agent rewrites context, old blocks waste space | PRUNE: Explicitly free blocks after position X |
| **Session Idle** | User goes AFK, GPU memory wasted on idle session | PAUSE: Migrate to cold storage with TTL lease |
| **System Prompts** | High-value prefixes evicted under pressure | CACHE: Pin blocks with sticky bit |
| **Reasoning Tokens** | `<think>` blocks don't need long-term storage | THINK: Mark transient, purge on completion |
| **Cold Node Warming** | New worker has empty cache, slow first request | WARM: Proactively fetch from Mooncake |

---

## Architecture

### Design Principles

1. **No KVBM dependency**: SGLang manages its own cache via HiRadixCache. Dynamo's KVBM is for vLLM, not SGLang.
2. **Leverage existing infrastructure**: Use NATS event plane for command transport.
3. **Minimal SGLang changes**: Add methods to HiRadixCache, no architectural changes.
4. **Worker-targeted commands**: Router can send commands to specific workers (e.g., warm a cold node).

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      DYNAMO ROUTER (Rust)                       │
│                                                                 │
│  ┌─────────────┐     ┌──────────────────────────────────────┐  │
│  │  KvRouter   │────►│  AgenticCommandPublisher (NEW)       │  │
│  │             │     │                                      │  │
│  │  - Detects  │     │  Commands:                           │  │
│  │    cold     │     │  - WARM(worker, blocks, tier)        │  │
│  │    nodes    │     │  - PRUNE(worker, after_hash)         │  │
│  │  - Tracks   │     │  - CACHE(worker, blocks, pin)        │  │
│  │    sessions │     │  - THINK(worker, blocks, transient)  │  │
│  │  - Detects  │     │  - PAUSE(worker, blocks, ttl)        │  │
│  │    idle     │     │                                      │  │
│  └─────────────┘     └──────────────┬───────────────────────┘  │
│                                     │                          │
└─────────────────────────────────────┼──────────────────────────┘
                                      │
                    NATS Subject: kv-control-{worker_id}
                                      │
┌─────────────────────────────────────┼──────────────────────────┐
│                      WORKER         │                          │
│                                     ▼                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DynamoSglangControlSubscriber (NEW)                     │  │
│  │                                                          │  │
│  │  - Subscribes to kv-control-{worker_id}                  │  │
│  │  - Deserializes AgenticCommand                           │  │
│  │  - Dispatches to HiRadixCache methods                    │  │
│  └──────────────────────────────┬───────────────────────────┘  │
│                                 │                              │
│                                 ▼                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SGLang Engine                                           │  │
│  │    └─► HiRadixCache (extended)                          │  │
│  │          │                                               │  │
│  │          ├─ block_hash_index: Dict[int, TreeNode]       │  │
│  │          │                                               │  │
│  │          ├─ warm_from_storage(keys, tier)               │  │
│  │          ├─ prune_after_block(hash)                     │  │
│  │          ├─ pin_blocks(hashes) / unpin_blocks(hashes)   │  │
│  │          ├─ mark_transient(hashes)                      │  │
│  │          ├─ purge_transient(hashes)                     │  │
│  │          └─ pause_session(hashes, ttl, lease_id)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                 │                              │
│                                 ▼                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Mooncake / Storage Backend (L3)                         │  │
│  │                                                          │  │
│  │  - batch_get(): Fetch blocks for WARM                   │  │
│  │  - batch_set(): Store blocks for PAUSE                  │  │
│  │  - Shared across all workers in cluster                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. Command Protocol

#### AgenticCommand Enum (Rust)

```rust
// lib/llm/src/kv_router/agentic_commands.rs

use serde::{Deserialize, Serialize};

/// Commands from Router to Workers for cache lifecycle management.
/// Serialized as JSON and published to NATS subject kv-control-{worker_id}.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgenticCommand {
    /// Proactively fetch blocks from storage into local cache.
    /// Used for warming cold nodes before routing requests to them.
    Warm {
        /// SHA256 hex keys for storage lookup
        block_keys: Vec<String>,
        /// Target tier: "GPU" or "CPU_TIER1"
        target_tier: String,
    },

    /// Evict all blocks that are descendants of the given block.
    /// Used when agent performs context rewrite, preserving prefix up to position X.
    Prune {
        /// Block hash (int64) after which all descendants should be evicted
        after_block_hash: i64,
    },

    /// Set or clear sticky bit on blocks to resist/allow eviction.
    /// Used for pinning high-value prefixes (system prompts, tool definitions).
    Cache {
        /// Block hashes to pin/unpin
        block_hashes: Vec<i64>,
        /// true = pin (resist eviction), false = unpin (allow eviction)
        pin: bool,
    },

    /// Mark blocks as transient (skip backup) or purge them immediately.
    /// Used for reasoning tokens (<think>...</think>) that don't need persistence.
    Think {
        /// Block hashes to mark/purge
        block_hashes: Vec<i64>,
        /// true = mark as transient, false = purge immediately
        transient: bool,
    },

    /// Migrate blocks to cold storage with TTL lease.
    /// Used when session becomes idle (user AFK).
    Pause {
        /// Block hashes to migrate
        block_hashes: Vec<i64>,
        /// Time-to-live in seconds (None = indefinite)
        ttl_seconds: Option<u64>,
        /// Unique lease identifier for renew/revoke operations
        lease_id: String,
    },

    /// Renew an existing pause lease.
    RenewLease {
        lease_id: String,
        new_ttl_seconds: u64,
    },

    /// Revoke a pause lease and purge the session from storage.
    RevokeLease {
        lease_id: String,
    },
}
```

#### JSON Wire Format

```json
{"type": "Warm", "block_keys": ["abc123...", "def456..."], "target_tier": "CPU_TIER1"}
{"type": "Prune", "after_block_hash": 1234567890}
{"type": "Cache", "block_hashes": [111, 222, 333], "pin": true}
{"type": "Think", "block_hashes": [444, 555], "transient": true}
{"type": "Pause", "block_hashes": [666, 777], "ttl_seconds": 3600, "lease_id": "session-abc"}
```

---

### 2. Router: AgenticCommandPublisher

#### Location

`lib/llm/src/kv_router/agentic_commands.rs`

#### Interface

```rust
/// Publishes agentic commands to specific workers via NATS.
pub struct AgenticCommandPublisher {
    nats_client: async_nats::Client,
}

impl AgenticCommandPublisher {
    /// Create a new publisher connected to NATS.
    pub async fn new(nats_url: &str) -> Result<Self>;

    /// Send a command to a specific worker.
    pub async fn send(&self, worker_id: WorkerId, cmd: AgenticCommand) -> Result<()> {
        let subject = format!("kv-control-{}", worker_id);
        let payload = serde_json::to_vec(&cmd)?;
        self.nats_client.publish(subject, payload.into()).await?;
        Ok(())
    }

    /// Broadcast a command to all workers (e.g., cache invalidation).
    pub async fn broadcast(&self, cmd: AgenticCommand) -> Result<()> {
        let subject = "kv-control-broadcast";
        let payload = serde_json::to_vec(&cmd)?;
        self.nats_client.publish(subject, payload.into()).await?;
        Ok(())
    }
}
```

#### Integration with KvRouter

```rust
// In KvRouter or KvPushRouter

impl KvRouter {
    /// Warm a cold worker by pushing blocks from storage.
    pub async fn warm_worker(&self, worker_id: WorkerId, block_keys: Vec<String>) -> Result<()> {
        self.command_publisher.send(worker_id, AgenticCommand::Warm {
            block_keys,
            target_tier: "CPU_TIER1".to_string(),
        }).await
    }

    /// Called when router detects session idle (no requests for N seconds).
    pub async fn pause_idle_session(&self, worker_id: WorkerId, session_blocks: Vec<i64>) -> Result<()> {
        let lease_id = uuid::Uuid::new_v4().to_string();
        self.command_publisher.send(worker_id, AgenticCommand::Pause {
            block_hashes: session_blocks,
            ttl_seconds: Some(3600), // 1 hour default
            lease_id,
        }).await
    }
}
```

---

### 3. Worker: DynamoSglangControlSubscriber

#### Location

`components/src/dynamo/sglang/control_subscriber.py`

#### Implementation

```python
import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nats
    import sglang as sgl

logger = logging.getLogger(__name__)


class DynamoSglangControlSubscriber:
    """
    Subscribes to agentic commands from Dynamo Router and dispatches
    them to SGLang's HiRadixCache.
    """

    def __init__(
        self,
        engine: "sgl.Engine",
        worker_id: str,
        nats_client: "nats.NATS",
    ):
        self.engine = engine
        self.worker_id = worker_id
        self.nats = nats_client
        self.subject = f"kv-control-{worker_id}"
        self.broadcast_subject = "kv-control-broadcast"
        self._running = False

    async def start(self):
        """Start subscribing to control commands."""
        self._running = True

        # Subscribe to worker-specific commands
        self._sub = await self.nats.subscribe(self.subject)
        # Subscribe to broadcast commands
        self._broadcast_sub = await self.nats.subscribe(self.broadcast_subject)

        logger.info(f"Control subscriber started for {self.subject}")

        # Run both subscription handlers
        await asyncio.gather(
            self._handle_subscription(self._sub),
            self._handle_subscription(self._broadcast_sub),
        )

    async def _handle_subscription(self, sub):
        """Process messages from a subscription."""
        async for msg in sub.messages:
            if not self._running:
                break
            try:
                cmd = json.loads(msg.data.decode())
                await self._dispatch(cmd)
            except Exception as e:
                logger.exception(f"Failed to handle control command: {e}")

    async def _dispatch(self, cmd: dict):
        """Dispatch a command to the appropriate HiRadixCache method."""
        cache = self._get_cache()
        if cache is None:
            logger.warning("HiRadixCache not available, ignoring command")
            return

        cmd_type = cmd.get("type")
        logger.info(f"Received agentic command: {cmd_type}")

        if cmd_type == "Warm":
            await self._handle_warm(cache, cmd)
        elif cmd_type == "Prune":
            await self._handle_prune(cache, cmd)
        elif cmd_type == "Cache":
            await self._handle_cache(cache, cmd)
        elif cmd_type == "Think":
            await self._handle_think(cache, cmd)
        elif cmd_type == "Pause":
            await self._handle_pause(cache, cmd)
        elif cmd_type == "RenewLease":
            await self._handle_renew_lease(cache, cmd)
        elif cmd_type == "RevokeLease":
            await self._handle_revoke_lease(cache, cmd)
        else:
            logger.warning(f"Unknown command type: {cmd_type}")

    def _get_cache(self):
        """Get the HiRadixCache from the engine."""
        try:
            # Access path may vary based on SGLang version
            return self.engine.scheduler.tree_cache
        except AttributeError:
            return None

    async def _handle_warm(self, cache, cmd: dict):
        """Handle WARM command: prefetch blocks from storage."""
        block_keys = cmd.get("block_keys", [])
        target_tier = cmd.get("target_tier", "CPU_TIER1")
        count = cache.warm_from_storage(block_keys, target_tier)
        logger.info(f"Warmed {count} blocks to {target_tier}")

    async def _handle_prune(self, cache, cmd: dict):
        """Handle PRUNE command: evict descendants of a block."""
        after_hash = cmd.get("after_block_hash")
        count = cache.prune_after_block(after_hash)
        logger.info(f"Pruned {count} blocks after hash {after_hash}")

    async def _handle_cache(self, cache, cmd: dict):
        """Handle CACHE command: pin/unpin blocks."""
        block_hashes = cmd.get("block_hashes", [])
        pin = cmd.get("pin", True)
        if pin:
            cache.pin_blocks(block_hashes)
            logger.info(f"Pinned {len(block_hashes)} blocks")
        else:
            cache.unpin_blocks(block_hashes)
            logger.info(f"Unpinned {len(block_hashes)} blocks")

    async def _handle_think(self, cache, cmd: dict):
        """Handle THINK command: mark/purge transient blocks."""
        block_hashes = cmd.get("block_hashes", [])
        transient = cmd.get("transient", True)
        if transient:
            cache.mark_transient(block_hashes)
            logger.info(f"Marked {len(block_hashes)} blocks as transient")
        else:
            count = cache.purge_transient(block_hashes)
            logger.info(f"Purged {count} transient blocks")

    async def _handle_pause(self, cache, cmd: dict):
        """Handle PAUSE command: migrate to storage with lease."""
        block_hashes = cmd.get("block_hashes", [])
        ttl_seconds = cmd.get("ttl_seconds")
        lease_id = cmd.get("lease_id")
        cache.pause_session(block_hashes, ttl_seconds, lease_id)
        logger.info(f"Paused session {lease_id} with {len(block_hashes)} blocks")

    async def _handle_renew_lease(self, cache, cmd: dict):
        """Handle RENEW_LEASE command."""
        lease_id = cmd.get("lease_id")
        new_ttl = cmd.get("new_ttl_seconds")
        cache.renew_lease(lease_id, new_ttl)
        logger.info(f"Renewed lease {lease_id} for {new_ttl}s")

    async def _handle_revoke_lease(self, cache, cmd: dict):
        """Handle REVOKE_LEASE command."""
        lease_id = cmd.get("lease_id")
        cache.revoke_lease(lease_id)
        logger.info(f"Revoked lease {lease_id}")

    async def stop(self):
        """Stop the subscriber."""
        self._running = False
        if hasattr(self, "_sub"):
            await self._sub.unsubscribe()
        if hasattr(self, "_broadcast_sub"):
            await self._broadcast_sub.unsubscribe()
        logger.info("Control subscriber stopped")
```

---

### 4. SGLang HiRadixCache Extensions

#### TreeNode Attribute Extensions

```python
# python/sglang/srt/mem_cache/radix_cache.py

class TreeNode:
    def __init__(self, id: Optional[int] = None, priority: int = 0):
        # ... existing attributes ...

        # NEW: Agentic control attributes
        self.sticky: bool = False      # Resist eviction when True
        self.transient: bool = False   # Skip backup to host/storage when True
        self.lease_id: Optional[str] = None  # Associated pause lease
```

#### HiRadixCache Method Extensions

```python
# python/sglang/srt/mem_cache/hiradix_cache.py

class HiRadixCache(RadixCache):
    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        # ... existing init ...

        # NEW: Reverse lookup from block hash to TreeNode
        self.block_hash_index: Dict[int, TreeNode] = {}

        # NEW: Lease management for PAUSE operations
        self.active_leases: Dict[str, LeaseInfo] = {}

    # === BLOCK HASH INDEX MAINTENANCE ===

    def _index_node_hashes(self, node: TreeNode):
        """Add node's block hashes to the reverse index."""
        if node.hash_value is None:
            return
        for hash_str in node.hash_value:
            block_hash = hash_str_to_int64(hash_str)
            self.block_hash_index[block_hash] = node

    def _unindex_node_hashes(self, node: TreeNode):
        """Remove node's block hashes from the reverse index."""
        if node.hash_value is None:
            return
        for hash_str in node.hash_value:
            block_hash = hash_str_to_int64(hash_str)
            self.block_hash_index.pop(block_hash, None)

    # === WARM OPERATION ===

    def warm_from_storage(
        self,
        block_keys: List[str],
        target_tier: str = "CPU_TIER1",
    ) -> int:
        """
        Proactively fetch blocks from L3 storage into local cache.

        Args:
            block_keys: SHA256 hex keys for storage lookup
            target_tier: "GPU" or "CPU_TIER1"

        Returns:
            Number of blocks successfully warmed
        """
        if not self.enable_storage:
            logger.warning("Storage not enabled, cannot warm from storage")
            return 0

        # Allocate host memory for incoming blocks
        num_pages = len(block_keys)
        host_indices = self.token_to_kv_pool_host.alloc(num_pages * self.page_size)
        if host_indices is None:
            # Try to free space
            self.evict_host(num_pages * self.page_size)
            host_indices = self.token_to_kv_pool_host.alloc(num_pages * self.page_size)
            if host_indices is None:
                logger.warning("Cannot allocate host memory for warm operation")
                return 0

        # Fetch from storage backend
        target_locations = [host_indices[i * self.page_size].item() for i in range(num_pages)]
        target_sizes = [self.page_size] * num_pages

        success_count = self.cache_controller.storage_backend.batch_get(
            block_keys, target_locations, target_sizes
        )

        # Emit BlockStored events for successfully fetched blocks
        for i in range(success_count):
            self._record_prefetch_store_events(
                token_ids=[],  # Unknown without full context
                hash_values=[block_keys[i]],
                parent_hash=None,
                medium=MEDIUM_CPU_TIER1,
            )

        # Optionally promote to GPU if target_tier == "GPU"
        if target_tier == "GPU" and success_count > 0:
            # Load from host to GPU
            device_indices = self.cache_controller.load(
                host_indices=host_indices[:success_count * self.page_size],
                node_id=-1,  # No associated node
            )
            if device_indices is not None:
                for i in range(success_count):
                    self._record_prefetch_store_events(
                        token_ids=[],
                        hash_values=[block_keys[i]],
                        parent_hash=None,
                        medium=MEDIUM_GPU,
                    )

        logger.info(f"Warmed {success_count}/{num_pages} blocks to {target_tier}")
        return success_count

    # === PRUNE OPERATION ===

    def prune_after_block(self, after_block_hash: int) -> int:
        """
        Evict all blocks that are descendants of the given block.

        Used when agent performs context rewrite, preserving prefix up to position X.

        Args:
            after_block_hash: Block hash after which all descendants should be evicted

        Returns:
            Number of blocks evicted
        """
        node = self.block_hash_index.get(after_block_hash)
        if node is None:
            logger.warning(f"Block hash {after_block_hash} not found in index")
            return 0

        # Collect all descendant nodes
        descendants = []
        self._collect_descendants(node, descendants)

        # Evict each descendant
        evicted_count = 0
        for desc in descendants:
            if desc == node:
                continue  # Don't evict the anchor node itself
            if not desc.evicted:
                if desc.backuped:
                    self._evict_backuped(desc)
                else:
                    self._evict_regular(desc)
                evicted_count += 1

        logger.info(f"Pruned {evicted_count} descendant blocks")
        return evicted_count

    def _collect_descendants(self, node: TreeNode, result: List[TreeNode]):
        """Recursively collect all descendants of a node."""
        result.append(node)
        for child in node.children.values():
            self._collect_descendants(child, result)

    # === CACHE (PIN) OPERATION ===

    def pin_blocks(self, block_hashes: List[int]):
        """
        Mark blocks as sticky (resist eviction under LRU pressure).

        Used for pinning high-value prefixes like system prompts.

        Args:
            block_hashes: Block hashes to pin
        """
        pinned = 0
        for h in block_hashes:
            node = self.block_hash_index.get(h)
            if node is not None:
                node.sticky = True
                pinned += 1
        logger.info(f"Pinned {pinned}/{len(block_hashes)} blocks")

    def unpin_blocks(self, block_hashes: List[int]):
        """
        Remove sticky flag from blocks (allow normal eviction).

        Args:
            block_hashes: Block hashes to unpin
        """
        unpinned = 0
        for h in block_hashes:
            node = self.block_hash_index.get(h)
            if node is not None:
                node.sticky = False
                unpinned += 1
        logger.info(f"Unpinned {unpinned}/{len(block_hashes)} blocks")

    # === THINK (TRANSIENT) OPERATION ===

    def mark_transient(self, block_hashes: List[int]):
        """
        Mark blocks as transient (don't backup to host/storage).

        Used for reasoning tokens that don't need long-term persistence.

        Args:
            block_hashes: Block hashes to mark as transient
        """
        marked = 0
        for h in block_hashes:
            node = self.block_hash_index.get(h)
            if node is not None:
                node.transient = True
                marked += 1
        logger.info(f"Marked {marked}/{len(block_hashes)} blocks as transient")

    def purge_transient(self, block_hashes: List[int]) -> int:
        """
        Immediately evict transient blocks from GPU without backup.

        Called when reasoning phase completes (<think>...</think> ends).

        Args:
            block_hashes: Block hashes to purge

        Returns:
            Number of blocks purged
        """
        purged = 0
        for h in block_hashes:
            node = self.block_hash_index.get(h)
            if node is not None and node.transient and not node.evicted:
                # Evict without backup (skip write_backup)
                self._record_tier_remove_event(node, MEDIUM_GPU)
                self.cache_controller.mem_pool_device_allocator.free(node.value)
                self._unindex_node_hashes(node)
                self._delete_leaf(node)
                purged += 1
        logger.info(f"Purged {purged} transient blocks")
        return purged

    # === PAUSE (SESSION SUSPENSION) OPERATION ===

    def pause_session(
        self,
        block_hashes: List[int],
        ttl_seconds: Optional[int],
        lease_id: str,
    ):
        """
        Migrate blocks to cold storage with TTL lease.

        Used when session becomes idle (user AFK).

        Args:
            block_hashes: Block hashes belonging to the session
            ttl_seconds: Time-to-live in seconds (None = indefinite)
            lease_id: Unique lease identifier
        """
        # Record the lease
        self.active_leases[lease_id] = LeaseInfo(
            block_hashes=block_hashes,
            ttl_seconds=ttl_seconds,
            created_at=time.time(),
        )

        # Mark nodes with lease and trigger migration
        for h in block_hashes:
            node = self.block_hash_index.get(h)
            if node is not None:
                node.lease_id = lease_id

                # Proactively migrate: GPU -> Host -> Storage
                if not node.evicted and not node.backuped:
                    self.write_backup(node)

                if node.backuped and self.enable_storage:
                    self.write_backup_storage(node)

        logger.info(f"Paused session {lease_id} with {len(block_hashes)} blocks, TTL={ttl_seconds}s")

    def renew_lease(self, lease_id: str, new_ttl_seconds: int):
        """Renew a pause lease with new TTL."""
        if lease_id in self.active_leases:
            self.active_leases[lease_id].ttl_seconds = new_ttl_seconds
            self.active_leases[lease_id].created_at = time.time()
            logger.info(f"Renewed lease {lease_id} for {new_ttl_seconds}s")

    def revoke_lease(self, lease_id: str):
        """Revoke a pause lease and purge the session from all tiers."""
        lease = self.active_leases.pop(lease_id, None)
        if lease is None:
            logger.warning(f"Lease {lease_id} not found")
            return

        for h in lease.block_hashes:
            node = self.block_hash_index.get(h)
            if node is not None:
                node.lease_id = None
                # Evict from all tiers
                if not node.evicted:
                    if node.backuped:
                        self._evict_backuped(node)
                    else:
                        self._evict_regular(node)
                if node.backuped:
                    self.evict_host(len(node.host_value))

        logger.info(f"Revoked lease {lease_id}, purged {len(lease.block_hashes)} blocks")

    # === EVICTION POLICY MODIFICATIONS ===

    def evict(self, params: EvictParams) -> EvictResult:
        """
        Evict tokens from GPU cache.

        MODIFIED: Skip sticky nodes unless force_evict is True.
        """
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            # NEW: Skip sticky nodes (they resist eviction)
            if x.sticky:
                continue

            if not x.backuped:
                # NEW: Skip backup for transient nodes
                if x.transient:
                    num_evicted += self._evict_regular(x)
                elif self.cache_controller.write_policy == "write_back":
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            # ... rest of existing eviction logic ...

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def write_backup(self, node: TreeNode, write_back=False):
        """
        Write node to host memory.

        MODIFIED: Skip transient nodes.
        """
        # NEW: Don't backup transient nodes
        if node.transient:
            logger.debug(f"Skipping backup for transient node {node.id}")
            return 0

        # ... existing write_backup logic ...


@dataclass
class LeaseInfo:
    """Information about a pause lease."""
    block_hashes: List[int]
    ttl_seconds: Optional[int]
    created_at: float
```

---

## Implementation Plan

### Phase 1: Foundation (SGLang)

| Ticket | Description | Complexity |
|--------|-------------|------------|
| **SGLang-1** | Add `block_hash_index` to HiRadixCache, maintain on insert/evict | Medium |
| **SGLang-2** | Add `sticky`, `transient`, `lease_id` attributes to TreeNode | Low |
| **SGLang-3** | Implement `pin_blocks()` / `unpin_blocks()` | Low |
| **SGLang-4** | Modify `evict()` to skip sticky nodes | Low |

### Phase 2: Core Operations (SGLang)

| Ticket | Description | Complexity |
|--------|-------------|------------|
| **SGLang-5** | Implement `mark_transient()` / `purge_transient()` | Medium |
| **SGLang-6** | Modify `write_backup()` to skip transient nodes | Low |
| **SGLang-7** | Implement `prune_after_block()` | Medium |
| **SGLang-8** | Implement `warm_from_storage()` | Medium |

### Phase 3: Session Management (SGLang)

| Ticket | Description | Complexity |
|--------|-------------|------------|
| **SGLang-9** | Implement `pause_session()` with lease tracking | High |
| **SGLang-10** | Implement `renew_lease()` / `revoke_lease()` | Medium |
| **SGLang-11** | Add background TTL expiration checker | Medium |

### Phase 4: Dynamo Integration

| Ticket | Description | Complexity |
|--------|-------------|------------|
| **DYN-1** | Define `AgenticCommand` enum and wire format | Low |
| **DYN-2** | Implement `AgenticCommandPublisher` in Router | Medium |
| **DYN-3** | Implement `DynamoSglangControlSubscriber` | Medium |
| **DYN-4** | Integrate subscriber into `dynamo.sglang` startup | Low |

### Phase 5: Router Intelligence

| Ticket | Description | Complexity |
|--------|-------------|------------|
| **DYN-5** | Add cold node detection and WARM triggering | Medium |
| **DYN-6** | Add idle session detection and PAUSE triggering | Medium |
| **DYN-7** | Add API hints integration (cache_control) for CACHE | Medium |
| **DYN-8** | Add reasoning boundary detection for THINK | High |

---

## Testing Strategy

### Unit Tests

1. **HiRadixCache methods**: Test each new method in isolation
2. **Block hash index**: Test index consistency through insert/evict cycles
3. **Sticky eviction**: Test that sticky nodes resist eviction
4. **Transient skip backup**: Test that transient nodes aren't backed up

### Integration Tests

1. **Warm flow**: Router sends WARM → Worker prefetches from Mooncake
2. **Prune flow**: Router sends PRUNE → Descendant blocks evicted
3. **Pin flow**: Router sends CACHE(pin=true) → Blocks resist eviction
4. **Pause flow**: Router sends PAUSE → Blocks migrated to storage

### E2E Tests

1. **Cold node warming**: Start new worker, verify it gets warmed before routing
2. **Session suspension**: Simulate idle session, verify migration to storage
3. **Context rewrite**: Simulate agent rewrite, verify pruning works

---

## Open Questions

1. **Block key format**: Events emit int64 hashes, Mooncake uses SHA256 strings. Should we store both in events?

2. **Lease persistence**: Should leases survive worker restart? (Requires external storage)

3. **THINK detection**: How does Router know which blocks are reasoning tokens? Token masks? Special tokens?

4. **Rate limiting**: Should commands be rate-limited to prevent thundering herd?

---

## References

- [DYN-1986: Agentic Events from Router to KVBM](https://linear.app/nvidia/issue/DYN-1986/agentic-events-from-router-to-kvbm)
- [HiCache KV Events Worklog](../WORKLOG_HICACHE_KV_EVENTS.md)
- [Mooncake HiCache Integration](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1.html)
- [Dynamo Router Architecture](~/dynamo/lib/llm/src/kv_router/)
