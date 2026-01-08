# Attention Fingerprint Discovery Pipeline

Batch processing and real-time classification of attention fingerprints for the SGLang Attention Explorer.

## Overview

This module provides a three-stage pipeline:

1. **Storage**: SQLite database for streaming fingerprints from sidecar
2. **Discovery Job**: Batch pipeline that computes embeddings and clusters
3. **Online Classifier**: Real-time classification using discovery artifacts

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SGLang Server  │────▶│    Sidecar      │────▶│    SQLite DB    │
│  (fingerprints) │     │  (streaming)    │     │ (fingerprints)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                               ┌─────────────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Discovery Job  │  (hourly/daily)
                        │  - PCA + UMAP   │
                        │  - HDBSCAN      │
                        │  - Zone assign  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Parquet + Model │
                        │   Artifacts     │
                        └────────┬────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │   Sidecar   │       │     UI      │       │   Metrics   │
   │ (real-time  │       │ (manifold   │       │ (cluster    │
   │  classify)  │       │  view)      │       │  trends)    │
   └─────────────┘       └─────────────┘       └─────────────┘
```

## Installation

```bash
cd examples/attention_explorer/discovery
pip install -r requirements.txt
```

## Usage

### Quick Start: Prompt Harness for Manifold Discovery

The prompt harness drives structured probes to map different attention behaviors:

```bash
# Quick smoke test (5 minutes)
python prompt_harness.py --duration 5 --server http://localhost:8000

# Full 8-hour discovery run
python prompt_harness.py --duration 480 --server http://localhost:8000

# Generate analysis report
python prompt_harness.py --report --db fingerprints.db
```

The harness includes 7 probe packs designed to excite specific attention programs:

| Pack | Expected Zone | Purpose |
|------|---------------|---------|
| `json_repair` | syntax_floor | JSON/bracket repair, local structure |
| `coreference` | semantic_bridge | Pronoun resolution, entity tracking |
| `counting_tables` | structure_ripple | Counting, tables, periodic patterns |
| `code_editing` | mixed | Refactoring, type hints, error handling |
| `reasoning` | semantic_bridge | Multi-step logic puzzles |
| `adversarial` | diffuse | Edge cases, noise, stress tests |
| `natural` | semantic_bridge | Natural conversation baseline |

The report includes:
- Zone distribution and confusion matrix
- Cluster purity by probe pack
- Expected vs actual zone agreement

### 1. Initialize Database

```bash
sqlite3 fingerprints.db < schema.sql
```

### 2. Populate Database (via Sidecar)

The sidecar service writes fingerprints to the database:

```python
import sqlite3
import struct

def store_fingerprint(db_path, request_id, step, fingerprint, ...):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        INSERT INTO fingerprints (request_id, step, fingerprint, ...)
        VALUES (?, ?, ?, ...)
    ''', (request_id, step, struct.pack('<20f', *fingerprint), ...))
    conn.commit()
```

### 3. Run Discovery Job

```bash
# Basic usage
python discovery_job.py --db fingerprints.db --output ./discovery_outputs

# With custom parameters
python discovery_job.py \
    --db fingerprints.db \
    --output ./discovery_outputs \
    --hours 24 \                    # Time window
    --min-cluster-size 50 \         # HDBSCAN min_cluster_size
    --umap-neighbors 15 \           # UMAP n_neighbors
    --prototypes 5                  # Samples per cluster
```

### 4. Use Online Classifier

```python
from discovery import SidecarClassifier

# Initialize (loads latest discovery artifacts)
classifier = SidecarClassifier('./discovery_outputs')

# Classify fingerprint
result = classifier.classify(fingerprint_vector)
# Returns:
# {
#     'manifold': {
#         'zone': 'semantic_bridge',
#         'confidence': 0.85,
#         'cluster_id': 3,
#         'cluster_label': 'Code Explanation',
#     },
#     'schema_version': 1,
# }
```

## Output Artifacts

The discovery job produces these files in `discovery_outputs/{run_id}/`:

| File | Description |
|------|-------------|
| `manifest.json` | Job metadata and parameters |
| `embeddings.parquet` | 2D coordinates for each fingerprint |
| `clusters.parquet` | Cluster definitions and centroids |
| `prototypes.parquet` | Representative samples per cluster |
| `request_embeddings.parquet` | Request-level aggregated embeddings |
| `clusterer.joblib` | HDBSCAN model for approximate_predict |
| `embedding_models.joblib` | Scaler + PCA + UMAP models |

A `latest` symlink always points to the most recent run.

## Zone Labels

Fingerprints are assigned to one of three "attention programs":

| Zone | Description | Characteristics |
|------|-------------|-----------------|
| `syntax_floor` | Local attention patterns | High local mass, low entropy |
| `semantic_bridge` | Mid-range retrieval | Balanced distribution |
| `structure_ripple` | Long-range periodic | High long mass, periodic patterns |

## Schema Version

The fingerprint schema (v1) consists of 20 float32 values:

```
[0]  local_mass     - Attention mass in local window (0-32 tokens)
[1]  mid_mass       - Attention mass in mid range (32-256 tokens)
[2]  long_mass      - Attention mass in long range (256+ tokens)
[3]  entropy        - Attention entropy
[4-11] histogram    - 8-bin distance histogram
[12-19] layer_stats - Per-layer entropy (up to 8 layers)
```

## Scheduling

For production use, schedule the discovery job with cron or systemd:

```bash
# crontab -e
# Run every hour
0 * * * * cd /path/to/discovery && python discovery_job.py --db /path/to/fingerprints.db --output /path/to/outputs --hours 24

# Or as a systemd timer
# /etc/systemd/system/attention-discovery.timer
```

## Metrics

The classifier provides basic metrics:

```python
stats = classifier.stats
# {
#     'classification_count': 12345,
#     'run_id': '2026-01-07T00-00-00',
#     'cluster_count': 12,
# }
```

For detailed metrics, query the SQLite views:

```sql
-- Zone distribution over time
SELECT * FROM zone_distribution_hourly;

-- Cluster growth
SELECT * FROM cluster_growth;
```
