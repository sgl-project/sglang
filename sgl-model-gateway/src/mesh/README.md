# Mesh Module

The Mesh module provides a distributed, eventually consistent state synchronization system for high-availability (HA) clusters. It enables multiple router nodes to share state information, coordinate rate limiting, and synchronize cache-aware routing policies across the cluster.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [HTTP API Reference](#http-api-reference)
- [Rate Limiting](#rate-limiting)
- [Cache-Aware Routing](#cache-aware-routing)
- [Service Discovery Integration](#service-discovery-integration)
- [Usage Examples](#usage-examples)

## Overview

The Mesh module implements a gossip-based protocol for state synchronization across cluster nodes. It uses Conflict-free Replicated Data Types (CRDTs) to ensure eventual consistency without requiring strong coordination or consensus protocols.

### Key Features

- **Eventual Consistency**: State converges across all nodes using CRDTs
- **Gossip Protocol**: Efficient peer-to-peer state propagation
- **Rate Limiting**: Distributed rate limiting with consistent hashing
- **Cache-Aware Routing**: Synchronized cache state for optimal routing
- **Service Discovery**: Integration with service discovery for dynamic membership
- **Topology Management**: Supports both full mesh and sparse mesh topologies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Mesh Module                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Gossip     │  │   CRDT       │  │   Stores    │     │
│  │   Protocol   │  │   Layer      │  │   Layer     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                           │                                 │
│  ┌──────────────────────────────────────────────┐         │
│  │         MeshSyncManager                       │         │
│  │  - Worker State Sync                          │         │
│  │  - Policy State Sync                           │         │
│  │  - Rate Limit Coordination                    │         │
│  │  - Tree Operation Sync                        │         │
│  └──────────────────────────────────────────────┘         │
│                           │                                 │
│  ┌──────────────────────────────────────────────┐         │
│  │         TopologyManager                       │         │
│  │  - Full Mesh (≤ threshold)                    │         │
│  │  - Sparse Mesh (> threshold, by region/AZ)    │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## HTTP API Reference

All mesh endpoints are prefixed with `/mesh` and require authentication.

### Cluster Status

#### GET `/mesh/status`

Get the current cluster status including node information and store counts.

**Response:**
```json
{
  "node_name": "node1",
  "node_count": 3,
  "nodes": [
    {
      "name": "node1",
      "address": "127.0.0.1:8000",
      "status": "Alive",
      "version": 1
    },
    {
      "name": "node2",
      "address": "127.0.0.1:8001",
      "status": "Alive",
      "version": 1
    }
  ],
  "stores": {
    "membership_count": 3,
    "worker_count": 0,
    "policy_count": 0,
    "app_count": 0
  }
}
```

**Status Codes:**
- `200 OK`: Success
- `503 Service Unavailable`: Mesh not enabled

### Health Check

#### GET `/mesh/health`

Get the health status of the mesh cluster.

**Response:**
```json
{
  "status": "healthy",
  "node_name": "node1",
  "cluster_size": 3,
  "stores_healthy": true
}
```

**Status Codes:**
- `200 OK`: Success
- `503 Service Unavailable`: Mesh not enabled

### Worker States

#### GET `/mesh/workers`

Get all worker states from the mesh store.

**Response:**
```json
[
  {
    "worker_id": "worker-1",
    "model_id": "model-1",
    "url": "http://worker1:8000",
    "health": true,
    "load": 0.75,
    "version": 1
  },
  {
    "worker_id": "worker-2",
    "model_id": "model-1",
    "url": "http://worker2:8000",
    "health": true,
    "load": 0.50,
    "version": 1
  }
]
```

**Status Codes:**
- `200 OK`: Success
- `503 Service Unavailable`: Mesh sync manager not available

#### GET `/mesh/workers/{worker_id}`

Get a specific worker state by worker ID.

**Path Parameters:**
- `worker_id` (string): The worker identifier

**Response:**
```json
{
  "worker_id": "worker-1",
  "model_id": "model-1",
  "url": "http://worker1:8000",
  "health": true,
  "load": 0.75,
  "version": 1
}
```

**Status Codes:**
- `200 OK`: Success
- `404 Not Found`: Worker not found
- `503 Service Unavailable`: Mesh sync manager not available

### Policy States

#### GET `/mesh/policies`

Get all policy states from the mesh store.

**Response:**
```json
[
  {
    "model_id": "model-1",
    "policy_type": "cache_aware",
    "config": "...",
    "version": 1
  }
]
```

**Status Codes:**
- `200 OK`: Success
- `503 Service Unavailable`: Mesh sync manager not available

#### GET `/mesh/policies/{model_id}`

Get a specific policy state by model ID.

**Path Parameters:**
- `model_id` (string): The model identifier

**Response:**
```json
{
  "model_id": "model-1",
  "policy_type": "cache_aware",
  "config": "...",
  "version": 1
}
```

**Status Codes:**
- `200 OK`: Success
- `404 Not Found`: Policy not found
- `503 Service Unavailable`: Mesh sync manager not available

### Application Configuration

#### GET `/mesh/config/{key}`

Get application configuration by key.

**Path Parameters:**
- `key` (string): The configuration key

**Response:**
```json
{
  "key": "config_key",
  "value": "68656c6c6f",  // Hex-encoded value
  "format": "hex"
}
```

**Status Codes:**
- `200 OK`: Success
- `404 Not Found`: Config not found
- `503 Service Unavailable`: Mesh not enabled

#### POST `/mesh/config`

Update application configuration.

**Request Body:**
```json
{
  "key": "config_key",
  "value": "68656c6c6f"  // Hex-encoded string (even length)
}
```

**Response:**
```json
{
  "status": "updated",
  "key": "config_key"
}
```

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid hex encoding or odd-length string
- `503 Service Unavailable`: Mesh not enabled

### Rate Limiting

#### POST `/mesh/rate-limit`

Set the global rate limit configuration.

**Request Body:**
```json
{
  "limit_per_second": 1000
}
```

**Response:**
```json
{
  "status": "updated",
  "limit_per_second": 1000
}
```

**Status Codes:**
- `200 OK`: Success
- `500 Internal Server Error`: Failed to serialize config
- `503 Service Unavailable`: Mesh not enabled

**Note:** Setting `limit_per_second` to `0` disables rate limiting.

#### GET `/mesh/rate-limit`

Get the global rate limit configuration.

**Response:**
```json
{
  "limit_per_second": 1000
}
```

**Status Codes:**
- `200 OK`: Success
- `404 Not Found`: Global rate limit not configured
- `500 Internal Server Error`: Failed to deserialize config
- `503 Service Unavailable`: Mesh not enabled

#### GET `/mesh/rate-limit/stats`

Get global rate limit statistics including current count and remaining capacity.

**Response:**
```json
{
  "limit_per_second": 1000,
  "current_count": 750,
  "remaining": 250
}
```

**Response Fields:**
- `limit_per_second`: The configured rate limit (0 means unlimited)
- `current_count`: Current aggregated count across all nodes
- `remaining`: Remaining capacity (`-1` if unlimited)

**Status Codes:**
- `200 OK`: Success
- `503 Service Unavailable`: Mesh sync manager not available

**How It Works:**
- Rate limit counters are distributed across cluster nodes using consistent hashing
- Each key is assigned to specific owner nodes
- Only owner nodes can increment counters
- Counter values are aggregated using CRDT (PNCounter) across all owners
- Counters are automatically reset periodically (default: every 1 second)

### Graceful Shutdown

#### POST `/mesh/shutdown`

Trigger a graceful shutdown of the mesh node.

**Response:**
```json
{
  "status": "shutdown initiated"
}
```

**Status Codes:**
- `202 Accepted`: Shutdown initiated
- `503 Service Unavailable`: Mesh not enabled

**Note:** This endpoint initiates the shutdown process asynchronously. The node will gracefully leave the cluster.

## Rate Limiting

The Mesh module provides distributed rate limiting using consistent hashing and CRDT counters.

### How It Works

1. **Consistent Hashing**: Each rate limit key is assigned to specific nodes (owners) based on hash
2. **Owner-Only Updates**: Only owner nodes can increment counters for their assigned keys
3. **CRDT Aggregation**: Counter values are merged across all owners using PNCounter (Positive-Negative Counter)
4. **Time Window Reset**: Counters are periodically reset (default: every 1 second)

### Usage Examples

#### Setting Global Rate Limit

Set a global rate limit of 1000 requests per second:

```bash
curl -X POST http://localhost:8000/mesh/rate-limit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"limit_per_second": 1000}'
```

**Response:**
```json
{
  "status": "updated",
  "limit_per_second": 1000
}
```

#### Checking Rate Limit Configuration

Get the current rate limit configuration:

```bash
curl http://localhost:8000/mesh/rate-limit \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "limit_per_second": 1000
}
```

#### Monitoring Rate Limit Statistics

Check current usage and remaining capacity:

```bash
curl http://localhost:8000/mesh/rate-limit/stats \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "limit_per_second": 1000,
  "current_count": 750,
  "remaining": 250
}
```

#### Disabling Rate Limiting

To disable rate limiting, set `limit_per_second` to `0`:

```bash
curl -X POST http://localhost:8000/mesh/rate-limit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"limit_per_second": 0}'
```

**Response:**
```json
{
  "limit_per_second": 0,
  "current_count": 0,
  "remaining": -1
}
```

### Complete Example: Rate Limiting Workflow

1. **Initialize Rate Limit:**
   ```bash
   # Set rate limit to 500 requests per second
   curl -X POST http://localhost:8000/mesh/rate-limit \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <token>" \
     -d '{"limit_per_second": 500}'
   ```

2. **Monitor Usage:**
   ```bash
   # Check current statistics
   curl http://localhost:8000/mesh/rate-limit/stats \
     -H "Authorization: Bearer <token>"
   ```

3. **Adjust Rate Limit:**
   ```bash
   # Increase to 2000 requests per second
   curl -X POST http://localhost:8000/mesh/rate-limit \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <token>" \
     -d '{"limit_per_second": 2000}'
   ```

### Key Concepts

- **Distributed Counters**: Counters are sharded across nodes, not replicated
- **Eventual Consistency**: Counter values converge across all nodes
- **Automatic Reset**: Counters reset periodically to implement time-window rate limiting
- **Membership Updates**: When nodes join/leave, ownership is automatically recalculated

## Cache-Aware Routing

The Mesh module synchronizes cache-aware routing tree operations across cluster nodes. This enables cache-aware routing policies to share cache state information using a **global radix tree**.

### Global Radix Tree: How Global State is Achieved

The cache-aware routing policy uses a **radix tree** (prefix tree) to track which workers have cached which request prefixes. The key innovation is making this tree **global** across all mesh nodes through state synchronization.

#### What is a Radix Tree?

A radix tree is a data structure that stores strings as a tree of character-based nodes. Each node represents a prefix segment, and the tree efficiently tracks which workers (tenants) have processed which request prefixes.

**Simple Example:**
```
Tree stores: "Hello" → worker1, "Help" → worker2

Structure:
Root
└── "H" → "ello" [worker1]
    └── "p" [worker2]
```

When routing a new request "Hello world", the tree finds the longest matching prefix ("Hello") and returns worker1, indicating worker1 likely has this cached.

#### How Global State is Achieved

The radix tree becomes **global** through mesh synchronization. Each node maintains a local copy of the tree, and all tree operations are synchronized across the cluster.

**Architecture:**

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Node 1    │         │   Node 2    │         │   Node 3    │
│             │         │             │         │             │
│ Local Tree  │◄───────►│ Local Tree  │◄───────►│ Local Tree  │
│             │  Mesh   │             │  Mesh   │             │
│             │  Sync   │             │  Sync   │             │
└─────────────┘         └─────────────┘         └─────────────┘
      │                       │                       │
      └───────────────────────┴───────────────────────┘
                    Global Tree State
              (Eventual Consistency via CRDT)
```

#### Synchronization Mechanism

**1. Operation-Based Synchronization:**

Instead of synchronizing the entire tree structure, only **tree operations** are synchronized:

- **Insert Operation**: `TreeOperation::Insert { text: "Hello", tenant: "worker1" }`
- **Remove Operation**: `TreeOperation::Remove { tenant: "worker1" }`

**2. Synchronization Flow:**

```
Node 1: Request arrives → Route to worker1
        ↓
        Insert into local tree: "Hello" → worker1
        ↓
        Generate operation: Insert("Hello", "worker1")
        ↓
        Sync to mesh via MeshSyncManager
        ↓
        Operation stored in PolicyStore with key: "tree:model-1"
        ↓
        Gossip protocol propagates to other nodes
        ↓
Node 2: Receives operation via gossip
        ↓
        Applies operation to local tree
        ↓
        Local tree now matches Node 1's tree
```

**3. State Storage:**

Tree state is stored in the `PolicyStore` (a CRDT map) with keys in the format `tree:{model_id}`:

```rust
// Tree state structure
TreeState {
    model_id: "llama-3",
    operations: [
        Insert("Hello", "worker1"),
        Insert("Help", "worker2"),
        Insert("World", "worker1"),
    ],
    version: 5
}
```

**4. Incremental Updates:**

The mesh uses incremental update collection to only send new operations:

- Each operation has a version number
- Only operations with versions > last_sent_version are transmitted
- Reduces network overhead and ensures efficient synchronization

**5. Eventual Consistency:**

- All nodes apply the same sequence of operations
- Operations are idempotent (can be applied multiple times safely)
- CRDT properties ensure convergence even with network partitions
- All nodes eventually have the same tree state

#### Complete Synchronization Example

**Initial State (All Nodes):**
```
All nodes have empty trees
```

**Step 1: Node 1 processes request**

```
Node 1:
1. Request: "Hello world" → Route to worker1
2. Insert locally: tree.insert("Hello world", "worker1")
3. Generate operation: Insert("Hello world", "worker1")
4. Sync to mesh: mesh_sync.sync_tree_operation("model-1", operation)
5. Operation stored in PolicyStore with version 1
```

**Step 2: Gossip propagation**

```
Gossip Protocol:
1. Node 1 sends state sync message to Node 2
2. Node 2 receives TreeState with operations: [Insert("Hello world", "worker1")]
3. Node 2 applies operation to local tree
4. Node 2's tree now matches Node 1's tree
5. Node 2 forwards to Node 3 (gossip continues)
```

**Step 3: Node 3 processes similar request**

```
Node 3:
1. Request: "Hello" arrives
2. Prefix match finds: "Hello" (partial match of "Hello world")
3. Match rate: 5/5 = 1.0 > cache_threshold
4. Route to worker1 (knows worker1 has "Hello" cached)
5. No new operation needed (already in tree)
```

**Step 4: Worker failure**

```
All Nodes:
1. Worker1 fails
2. Node 1 detects failure
3. Remove locally: tree.remove_tenant("worker1")
4. Generate operation: Remove("worker1")
5. Sync to mesh: mesh_sync.sync_tree_operation("model-1", operation)
6. All nodes receive and apply removal
7. All trees updated consistently
```

#### State Restoration on Startup

When a node restarts or joins the cluster:

```
1. Node starts with empty tree
2. CacheAwarePolicy.restore_tree_state_from_mesh() called
3. Retrieves TreeState from PolicyStore via mesh
4. Applies all operations sequentially:
   for operation in tree_state.operations {
       match operation {
           Insert(text, tenant) => tree.insert(text, tenant),
           Remove(tenant) => tree.remove_tenant(tenant),
       }
   }
5. Tree rebuilt to match cluster state
6. Node ready to route with full cache knowledge
```

#### Benefits of Global Synchronization

1. **Shared Cache Knowledge**: All nodes know which workers have cached which prefixes
2. **Optimal Routing**: Any node can route to the best worker based on cache state
3. **Fault Tolerance**: Tree state persists across node failures via mesh storage
4. **Automatic Recovery**: New nodes automatically get full cache state
5. **Eventual Consistency**: All nodes converge to the same view without coordination

### How It Works

1. **Tree Operations**: When cache-aware routing makes routing decisions, tree operations (insert/remove) are generated
2. **Mesh Synchronization**: Tree operations are automatically synchronized to the mesh via the sync manager
3. **State Restoration**: On startup, cache-aware policies restore tree state from the mesh
4. **Eventual Consistency**: Tree states converge across all nodes

### Integration

Cache-aware routing is automatically integrated with the mesh when:
- Mesh is enabled in the router configuration
- Cache-aware policy is configured for a model
- Tree operations are performed during routing

### Usage Examples

#### Checking Policy State

View the cache-aware policy state for a specific model:

```bash
curl http://localhost:8000/mesh/policies/model-1 \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "model_id": "model-1",
  "policy_type": "cache_aware",
  "config": "...",
  "version": 5
}
```

#### Viewing All Policy States

List all policy states across the cluster:

```bash
curl http://localhost:8000/mesh/policies \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
[
  {
    "model_id": "model-1",
    "policy_type": "cache_aware",
    "config": "...",
    "version": 5
  },
  {
    "model_id": "model-2",
    "policy_type": "cache_aware",
    "config": "...",
    "version": 3
  }
]
```

### Complete Example: Global Tree Synchronization Workflow

This example demonstrates how the radix tree becomes global through mesh synchronization.

**Initial Setup:**
- 3-node cluster: node1, node2, node3
- Model: llama-3
- All nodes start with empty trees

**Step 1: Node 1 processes first request**

```
Node 1:
Request: "The quick brown fox"
Model: llama-3

1. Local tree is empty → no prefix match
2. Route to worker1 (smallest tree)
3. Insert locally: tree.insert("The quick brown fox", "worker1")
4. Generate operation: Insert("The quick brown fox", "worker1")
5. Sync to mesh:
   mesh_sync.sync_tree_operation("llama-3", operation)
   ↓
   Operation stored in PolicyStore: tree:llama-3
   Version: 1
```

**Step 2: Mesh propagates to other nodes**

```
Gossip Protocol:
Node 1 → Node 2: Sends TreeState { operations: [Insert(...)], version: 1 }
Node 2 → Node 3: Forwards TreeState

Node 2:
1. Receives TreeState via gossip
2. Applies operation: tree.insert("The quick brown fox", "worker1")
3. Local tree now matches Node 1

Node 3:
1. Receives TreeState via gossip
2. Applies operation: tree.insert("The quick brown fox", "worker1")
3. Local tree now matches Node 1 and Node 2
```

**Step 3: Node 2 processes similar request**

```
Node 2:
Request: "The quick brown"
Model: llama-3

1. Prefix match finds: "The quick brown" (partial match)
2. Match rate: 17/19 = 0.89 > cache_threshold (0.7)
3. Route to worker1 (knows worker1 has this cached)
4. No new operation (tree already has this prefix)
5. All nodes already have this in their trees
```

**Step 4: Node 3 processes different request**

```
Node 3:
Request: "Hello world"
Model: llama-3

1. Prefix match: "" (no match)
2. Route to worker2 (smallest tree)
3. Insert locally: tree.insert("Hello world", "worker2")
4. Generate operation: Insert("Hello world", "worker2")
5. Sync to mesh:
   mesh_sync.sync_tree_operation("llama-3", operation)
   ↓
   Operation stored in PolicyStore: tree:llama-3
   Version: 2
6. Gossip propagates to Node 1 and Node 2
7. All nodes now have both prefixes in their trees
```

**Step 5: Node 1 restarts**

```
Node 1 (after restart):
1. CacheAwarePolicy initializes
2. Calls restore_tree_state_from_mesh()
3. Retrieves TreeState from PolicyStore:
   {
     model_id: "llama-3",
     operations: [
       Insert("The quick brown fox", "worker1"),
       Insert("Hello world", "worker2")
     ],
     version: 2
   }
4. Applies all operations sequentially:
   tree.insert("The quick brown fox", "worker1")
   tree.insert("Hello world", "worker2")
5. Tree rebuilt to match cluster state
6. Node 1 has full cache knowledge again
```

**Result:**
- All nodes have identical tree state
- Any node can route optimally based on cache
- State persists across restarts
- New nodes automatically get full state

### Tree State Storage

Tree states are stored in the PolicyStore with keys in the format: `tree:{model_id}`

The tree state contains:
- `model_id`: The model identifier
- `operations`: Sequence of tree operations (insert/remove)
- `version`: Version number for conflict resolution

### Benefits

- **Shared Cache Knowledge**: All nodes know which workers have cached which request prefixes
- **Optimal Routing**: Routes requests to workers with relevant cache data
- **Automatic Synchronization**: No manual intervention required
- **Fault Tolerance**: Tree state is preserved across node failures

### Example: Global Synchronization in Action

This example shows how tree operations are synchronized across nodes to create a global view.

**Scenario:** 3-node cluster, model "llama-3"

**Timeline:**

```
T0: All nodes have empty trees
    Node1: []
    Node2: []
    Node3: []

T1: Node1 processes "Hello world" → worker1
    Node1: [Insert("Hello world", "worker1")] → syncs to mesh
    Node2: [] (not yet received)
    Node3: [] (not yet received)

T2: Gossip propagates (Node1 → Node2 → Node3)
    Node1: [Insert("Hello world", "worker1")]
    Node2: [Insert("Hello world", "worker1")] ← applied from mesh
    Node3: [Insert("Hello world", "worker1")] ← applied from mesh

T3: Node2 processes "Hello" → worker1 (cache hit, no sync needed)
    Node3 processes "Help" → worker2
    Node3: [Insert("Hello world", "worker1"), Insert("Help", "worker2")] → syncs to mesh

T4: Gossip propagates
    Node1: [Insert("Hello world", "worker1"), Insert("Help", "worker2")] ← applied
    Node2: [Insert("Hello world", "worker1"), Insert("Help", "worker2")] ← applied
    Node3: [Insert("Hello world", "worker1"), Insert("Help", "worker2")]

T5: All nodes have identical tree state
    All nodes can route "Hello" → worker1 (cache hit)
    All nodes can route "Help" → worker2 (cache hit)
```

**Key Points:**

1. **Operations are the source of truth**: Tree structure is derived from operations
2. **Gossip ensures propagation**: Operations spread to all nodes automatically
3. **Eventual consistency**: All nodes converge to the same state
4. **No coordination needed**: Each node applies operations independently
5. **State persists**: Operations stored in PolicyStore survive node restarts

## Service Discovery Integration

The Mesh module integrates with service discovery systems to maintain cluster membership dynamically.

### Membership Updates

When service discovery detects membership changes (nodes joining or leaving), the mesh:

1. **Updates Hash Rings**: Rate limit hash rings are recalculated
2. **Updates Membership Store**: Node information is synchronized
3. **Ownership Transfer**: Rate limit ownership is transferred for failed nodes
4. **Topology Recalculation**: Peer connections are recalculated

### Topology Management

The mesh supports two topology modes:

**Full Mesh** (≤ threshold nodes, default: 10):
- All nodes connect to all other nodes
- Best for small clusters
- Maximum redundancy

**Sparse Mesh** (> threshold nodes):
- Nodes connect based on region/availability zone
- Reduces connection overhead
- Suitable for large clusters

### Node States

Nodes can be in different states:

- **Alive**: Node is healthy and reachable
- **Suspected**: Node may be unreachable (gossip-based detection)
- **Down**: Node is confirmed unreachable
- **Leaving**: Node is gracefully shutting down

### Service Discovery Flow

```
Service Discovery → Membership Update → Hash Ring Update → Ownership Transfer
                                      ↓
                              Topology Recalculation
                                      ↓
                              Peer Connection Update
```

### Configuration

Topology configuration can be set via:
- Region identifier (for sparse mesh)
- Availability zone identifier (for sparse mesh)
- Full mesh threshold (default: 10 nodes)

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful operation
- `202 Accepted`: Operation accepted (async)
- `400 Bad Request`: Invalid request format
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Mesh not enabled or service unavailable

Error responses follow this format:
```json
{
  "error": "Error message description"
}
```

## Usage Examples

### Complete Workflow: Setting Up Mesh with Rate Limiting and Cache-Aware Routing

This example demonstrates how to set up and use mesh features in a production environment.

#### 1. Enable Mesh in Configuration

Configure mesh in your router configuration file:

```yaml
mesh:
  enabled: true
  self_name: "router-node-1"
  self_addr: "0.0.0.0:8000"
  init_peer: "router-node-2:8000"  # Optional: initial peer for bootstrap
```

#### 2. Set Up Global Rate Limiting

```bash
# Set initial rate limit
curl -X POST http://localhost:8000/mesh/rate-limit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"limit_per_second": 1000}'

# Verify configuration
curl http://localhost:8000/mesh/rate-limit \
  -H "Authorization: Bearer <token>"
```

#### 3. Configure Cache-Aware Policy

When configuring a model with cache-aware routing, the mesh automatically handles synchronization:

```yaml
models:
  - model_id: "llama-3"
    policy:
      type: "cache_aware"
      config:
        cache_threshold: 0.7
        balance_abs_threshold: 10
        balance_rel_threshold: 1.5
        eviction_interval_secs: 300
        max_tree_size: 10000
```

#### 4. Monitor Cluster Status

```bash
# Check cluster health
curl http://localhost:8000/mesh/health \
  -H "Authorization: Bearer <token>"

# View cluster status
curl http://localhost:8000/mesh/status \
  -H "Authorization: Bearer <token>"
```

#### 5. Monitor Rate Limiting

```bash
# Check current rate limit statistics
curl http://localhost:8000/mesh/rate-limit/stats \
  -H "Authorization: Bearer <token>"

# Adjust rate limit based on load
curl -X POST http://localhost:8000/mesh/rate-limit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"limit_per_second": 2000}'
```

#### 6. Monitor Cache-Aware Policy State

```bash
# View policy state for a model
curl http://localhost:8000/mesh/policies/llama-3 \
  -H "Authorization: Bearer <token>"

# View all policy states
curl http://localhost:8000/mesh/policies \
  -H "Authorization: Bearer <token>"
```

#### 7. View Worker States

```bash
# List all worker states
curl http://localhost:8000/mesh/workers \
  -H "Authorization: Bearer <token>"

# Get specific worker state
curl http://localhost:8000/mesh/workers/worker-1 \
  -H "Authorization: Bearer <token>"
```

### Example: Multi-Node Setup

In a 3-node cluster setup:

**Node 1 (router-node-1):**
```bash
# Set rate limit
curl -X POST http://router-node-1:8000/mesh/rate-limit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"limit_per_second": 1000}'
```

**Node 2 and Node 3:**
- Automatically receive rate limit configuration via mesh synchronization
- Share cache-aware tree state across all nodes
- Maintain consistent state without manual configuration

**Verify Consistency:**
```bash
# Check rate limit on all nodes (should be consistent)
curl http://router-node-1:8000/mesh/rate-limit/stats
curl http://router-node-2:8000/mesh/rate-limit/stats
curl http://router-node-3:8000/mesh/rate-limit/stats
```

### Example: Dynamic Configuration Updates

Update application configuration dynamically:

```bash
# Store custom configuration
curl -X POST http://localhost:8000/mesh/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "key": "custom_feature_flag",
    "value": "74727565"  # hex for "true"
  }'

# Retrieve configuration
curl http://localhost:8000/mesh/config/custom_feature_flag \
  -H "Authorization: Bearer <token>"
```

## Authentication

All mesh endpoints require authentication. Configure authentication via the router's authentication middleware.

## See Also

- [CRDT Documentation](https://github.com/rust-crdt/rust-crdt)
- [Gossip Protocol](https://en.wikipedia.org/wiki/Gossip_protocol)
- [Consistent Hashing](https://en.wikipedia.org/wiki/Consistent_hashing)
