# Attention Explorer - Getting Started

This guide walks you through setting up and using the Attention Explorer to visualize and analyze LLM attention patterns.

## Overview

The Attention Explorer captures attention weights during LLM inference and:
- Computes **fingerprints** (20-dimensional vectors summarizing attention patterns)
- Classifies tokens into **manifold zones** (syntax_floor, semantic_bridge, structure_ripple, etc.)
- Creates **2D embeddings** for visualization using UMAP
- Provides **real-time classification** via a sidecar service

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  SGLang     │────▶│   Sidecar    │────▶│  Discovery   │
│  Server     │     │   Service    │     │  Job         │
└─────────────┘     └──────────────┘     └──────────────┘
       │                   │                    │
       │ attention         │ fingerprints       │ clusters
       │ tokens            │ + zones            │ + embeddings
       ▼                   ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  React UI   │◀────│  SQLite DB   │◀────│  Parquet     │
│             │     │              │     │  Files       │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Prerequisites

```bash
pip install numpy pandas pyarrow hdbscan umap-learn scikit-learn joblib
```

## Quick Start

### 1. Initialize Database

```bash
cd examples/attention_explorer
sqlite3 fingerprints.db < discovery/schema.sql
```

### 2. Start SGLang Server with Attention Capture

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --attention-backend triton \
    --return-attention-tokens \
    --attention-tokens-top-k 32 \
    --port 30000
```

### 3. Start the Sidecar Service

```bash
python rapids_sidecar.py \
    --port 9009 \
    --db fingerprints.db \
    --discovery-dir ./discovery_outputs \
    --online
```

### 4. Make Requests with Attention Capture

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"return_attention_tokens": True},
    stream=True,
)

for chunk in response:
    if hasattr(chunk.choices[0].delta, "attention_tokens"):
        print("Attention data:", chunk.choices[0].delta.attention_tokens)
```

### 5. Run Discovery Job

After collecting fingerprints, run discovery to create clusters:

```bash
python discovery/discovery_job.py \
    --db fingerprints.db \
    --output ./discovery_outputs \
    --hours 24 \
    --min-cluster-size 20
```

### 6. Start the UI

```bash
cd ui
npm install
npm run dev
```

Open http://localhost:5173 to view the Attention Explorer.

## Configuration Options

### SGLang Server

| Flag | Description |
|------|-------------|
| `--return-attention-tokens` | Enable attention token capture |
| `--attention-tokens-top-k N` | Number of top attention positions to return (default: 32) |
| `--attention-backend triton` | Use Triton backend (required for attention capture) |

### Sidecar Service

| Flag | Description |
|------|-------------|
| `--port N` | HTTP port (default: 9000) |
| `--db PATH` | SQLite database for fingerprint storage |
| `--discovery-dir PATH` | Directory containing discovery artifacts |
| `--online` | Enable online clustering mode |
| `--zmq-bind ADDR` | ZMQ bind address for streaming (e.g., `tcp://*:9001`) |

### Discovery Job

| Flag | Description |
|------|-------------|
| `--db PATH` | SQLite database with fingerprints |
| `--output PATH` | Output directory for artifacts |
| `--hours N` | Time window in hours (default: 24) |
| `--min-cluster-size N` | Minimum cluster size (default: 20) |
| `--long-running` | Enable coordinator mode with checkpointing |
| `--websocket-port N` | WebSocket port for live updates |

## API Endpoints

### Sidecar Service (default port 9009)

#### Health Check
```
GET /health
```

#### Add Fingerprint
```
POST /fingerprint
{
    "request_id": "uuid",
    "vector": [0.1, 0.2, ...],  // 20-dim fingerprint
    "step": 1
}
```

#### Classify Fingerprint
```
POST /classify
{
    "vector": [0.1, 0.2, ...]  // 20-dim fingerprint
}
```

Response:
```json
{
    "manifold": {
        "zone": "semantic_bridge",
        "confidence": 0.85,
        "cluster_id": 3,
        "cluster_label": "Cluster 3"
    },
    "schema_version": 1,
    "run_id": "2026-01-10T04-02-11"
}
```

#### Get Discovery Status
```
GET /discovery/status
```

#### Get Statistics
```
GET /stats
```

## Manifold Zones

The attention patterns are classified into zones based on their characteristics:

| Zone | Characteristics | Examples |
|------|----------------|----------|
| **syntax_floor** | High local mass, low entropy | JSON repair, bracket matching, formatting |
| **semantic_bridge** | Balanced mid-range attention | Coreference resolution, context retrieval |
| **structure_ripple** | Periodic patterns | Counting, tables, indentation tracking |
| **long_range** | High long-range attention | Document context, multi-turn dialogue |
| **diffuse** | Scattered attention | Uncertain predictions, exploration |

## Fingerprint Layout

The 20-dimensional fingerprint vector contains:

| Index | Name | Description |
|-------|------|-------------|
| 0 | local_mass | Attention mass on nearby tokens (positions 0-10) |
| 1 | mid_mass | Attention mass on mid-range tokens (positions 10-100) |
| 2 | long_mass | Attention mass on distant tokens (positions 100+) |
| 3 | entropy | Shannon entropy of attention distribution |
| 4-11 | histogram | 8-bin histogram of attention positions |
| 12-19 | layer_entropy | Per-layer entropy (8 layers) |

## Long-Running Discovery

For large datasets (100K+ fingerprints), use the coordinator mode:

```bash
python discovery/discovery_job.py \
    --db fingerprints.db \
    --output ./discovery_outputs \
    --long-running \
    --websocket-port 9010 \
    --max-memory-gb 16
```

Features:
- **Checkpointing**: Resume from interruptions
- **Memory-bounded UMAP**: Process large datasets in chunks
- **WebSocket updates**: Live progress in UI
- **Graceful shutdown**: Ctrl+C saves checkpoint

### Resuming from Checkpoint

```bash
python discovery/discovery_job.py \
    --db fingerprints.db \
    --output ./discovery_outputs \
    --resume RUN_ID
```

## Troubleshooting

### No attention tokens in response

1. Ensure `--return-attention-tokens` is set on the server
2. Use `extra_body={"return_attention_tokens": True}` in the client
3. Verify `--attention-backend triton` is used

### Discovery job fails with OOM

1. Reduce `--umap-sample-size` (default: 50000)
2. Use `--long-running` mode with `--max-memory-gb`
3. Increase chunk size with `--batch-size`

### Sidecar shows "discovery not available"

1. Run the discovery job first
2. Ensure `--discovery-dir` points to correct directory
3. Check that `latest` symlink exists in discovery directory

### UI shows "No clusters detected"

1. Run discovery job to create clusters
2. Reload sidecar or restart it
3. Check sidecar logs for errors

## Example Script

See `examples/manifold_quickstart.py` for a complete example:

```bash
python examples/manifold_quickstart.py
```
