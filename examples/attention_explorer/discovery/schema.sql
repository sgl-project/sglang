-- =============================================================================
-- Attention Fingerprint Discovery Schema
-- =============================================================================
-- Storage for streaming attention fingerprints and discovery artifacts.
-- Designed for append-heavy writes (fingerprints) and read-heavy queries (UI).
--
-- Usage:
--   sqlite3 fingerprints.db < schema.sql
--
-- Schema Version: 1
-- =============================================================================

PRAGMA journal_mode = WAL;          -- Better concurrent read/write
PRAGMA synchronous = NORMAL;        -- Balance durability vs speed
PRAGMA cache_size = -64000;         -- 64MB cache
PRAGMA temp_store = MEMORY;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Fingerprints: One row per decoded token with attention fingerprint
-- This is the primary append-only table, write-heavy from sidecar
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Request identification
    request_id TEXT NOT NULL,              -- UUID grouping tokens in same request
    session_id TEXT,                       -- Optional: groups requests in conversation

    -- Token position
    step INTEGER NOT NULL,                 -- Decode step (1-indexed, matches server)
    token_id INTEGER,                      -- Vocabulary token ID
    token_text TEXT,                       -- Decoded text (may be partial UTF-8)

    -- Phase segmentation (for reasoning models like Qwen3-Next)
    think_phase TEXT CHECK(think_phase IN ('think', 'output', 'unknown')),
    segment_idx INTEGER DEFAULT 0,         -- Which think/output segment (0-indexed)

    -- The fingerprint vector (schema v1: 20 dimensions)
    -- Layout: [local_mass, mid_mass, long_mass, entropy,
    --          hist_0..hist_7, layer_entropy_0..layer_entropy_7]
    fingerprint BLOB NOT NULL,             -- Packed float32[20], little-endian

    -- Online classification (from sidecar's approximate_predict)
    manifold_zone TEXT CHECK(manifold_zone IN (
        'syntax_floor',      -- High local mass, low entropy
        'semantic_bridge',   -- Mid-range retrieval, balanced
        'structure_ripple',  -- Periodic patterns, counting/tables
        'unknown'
    )),
    manifold_confidence REAL,              -- 0.0 to 1.0
    cluster_id INTEGER DEFAULT -1,         -- From HDBSCAN (-1 = noise/unassigned)
    cluster_probability REAL,              -- HDBSCAN soft assignment

    -- MoE routing (for mixture-of-experts models)
    top_expert_ids BLOB,                   -- Packed int32[], top-k expert IDs
    router_entropy REAL,                   -- Entropy of router distribution
    expert_load_balance REAL,              -- How balanced the expert selection is

    -- Raw attention summary (optional, for detailed analysis)
    top_k_positions BLOB,                  -- Packed int32[k], attended positions
    top_k_scores BLOB,                     -- Packed float32[k], attention scores
    sink_token_mass REAL,                  -- Attention mass on position 0

    -- Capture metadata
    capture_layer_ids BLOB,                -- Packed int32[], which layers were captured
    schema_version INTEGER DEFAULT 1,
    model_id TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure uniqueness per request+step
    UNIQUE(request_id, step)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_fingerprints_request ON fingerprints(request_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_session ON fingerprints(session_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_cluster ON fingerprints(cluster_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_zone ON fingerprints(manifold_zone);
CREATE INDEX IF NOT EXISTS idx_fingerprints_created ON fingerprints(created_at);
CREATE INDEX IF NOT EXISTS idx_fingerprints_phase ON fingerprints(think_phase);


-- =============================================================================
-- REQUEST SUMMARY TABLE
-- =============================================================================

-- Aggregated stats per request (materialized by sidecar or trigger)
CREATE TABLE IF NOT EXISTS request_summary (
    request_id TEXT PRIMARY KEY,
    session_id TEXT,

    -- Token counts
    total_steps INTEGER NOT NULL,
    think_steps INTEGER DEFAULT 0,
    output_steps INTEGER DEFAULT 0,

    -- Zone distribution (percentages, sum to 1.0)
    syntax_floor_pct REAL DEFAULT 0,
    semantic_bridge_pct REAL DEFAULT 0,
    structure_ripple_pct REAL DEFAULT 0,
    unknown_pct REAL DEFAULT 0,

    -- Dominant characteristics
    dominant_zone TEXT,
    mean_entropy REAL,
    mean_local_mass REAL,
    mean_long_mass REAL,

    -- MoE summary
    expert_entropy_mean REAL,
    expert_switch_count INTEGER DEFAULT 0,  -- Times top expert changed
    unique_experts_used INTEGER DEFAULT 0,

    -- Cluster summary
    dominant_cluster_id INTEGER DEFAULT -1,
    cluster_transitions INTEGER DEFAULT 0,   -- Times cluster changed

    -- Content preview (for UI display)
    prompt_preview TEXT,                     -- First 500 chars of user message
    response_preview TEXT,                   -- First 500 chars of response

    -- Model info
    model_id TEXT,

    -- Timestamps
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_request_summary_session ON request_summary(session_id);
CREATE INDEX IF NOT EXISTS idx_request_summary_zone ON request_summary(dominant_zone);
CREATE INDEX IF NOT EXISTS idx_request_summary_cluster ON request_summary(dominant_cluster_id);
CREATE INDEX IF NOT EXISTS idx_request_summary_created ON request_summary(created_at);


-- =============================================================================
-- SESSION TABLE
-- =============================================================================

-- Optional: Track multi-turn conversations
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,

    -- Session metadata
    name TEXT,                              -- User-provided name
    model_id TEXT,

    -- Aggregates
    request_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =============================================================================
-- DISCOVERY RUN METADATA
-- =============================================================================

-- Track discovery job runs (for UI to know which artifacts to load)
CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id TEXT PRIMARY KEY,               -- ISO timestamp or UUID

    -- Job parameters
    time_window_start TIMESTAMP,
    time_window_end TIMESTAMP,
    fingerprint_count INTEGER,
    request_count INTEGER,

    -- Clustering results
    cluster_count INTEGER,
    noise_count INTEGER,                   -- Points not assigned to cluster

    -- Algorithm parameters (for reproducibility)
    embedding_method TEXT,                 -- e.g., 'pca_50_umap_2'
    clustering_method TEXT,                -- e.g., 'hdbscan'
    min_cluster_size INTEGER,

    -- Output paths
    output_dir TEXT,

    -- Status
    status TEXT CHECK(status IN ('running', 'completed', 'failed')),
    error_message TEXT,

    -- Timestamps
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_discovery_runs_status ON discovery_runs(status);
CREATE INDEX IF NOT EXISTS idx_discovery_runs_completed ON discovery_runs(completed_at);


-- =============================================================================
-- CLUSTER DEFINITIONS (cached from latest discovery run)
-- =============================================================================

-- Cluster metadata from most recent discovery job
-- This allows fast lookups without loading Parquet files
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,                  -- Which discovery run defined this

    -- Cluster identity
    label TEXT,                            -- Human/LLM-assigned label
    description TEXT,                      -- Longer description

    -- Zone assignment
    dominant_zone TEXT,
    zone_confidence REAL,

    -- Statistics
    size INTEGER,                          -- Number of points
    persistence REAL,                      -- HDBSCAN stability metric

    -- Centroid (for online assignment)
    centroid_x REAL,                       -- UMAP x coordinate
    centroid_y REAL,                       -- UMAP y coordinate
    centroid_fingerprint BLOB,             -- 20-dim fingerprint centroid

    -- Representative info
    medoid_request_id TEXT,                -- Most central request
    medoid_step INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES discovery_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_clusters_zone ON clusters(dominant_zone);


-- =============================================================================
-- PROTOTYPES (representative samples per cluster)
-- =============================================================================

CREATE TABLE IF NOT EXISTS prototypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,

    -- Ranking within cluster
    rank INTEGER NOT NULL,                 -- 0 = medoid, 1+ = nearest neighbors

    -- Reference to original fingerprint
    fingerprint_id INTEGER,
    request_id TEXT,
    step INTEGER,

    -- Cached display info
    token_text TEXT,
    context_before TEXT,                   -- ~50 chars before
    context_after TEXT,                    -- ~50 chars after
    prompt_preview TEXT,

    -- Coordinates
    x REAL,
    y REAL,

    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id),
    FOREIGN KEY (run_id) REFERENCES discovery_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_prototypes_cluster ON prototypes(cluster_id, rank);


-- =============================================================================
-- CONTROL SIGNALS (sidecar → inference server)
-- =============================================================================

-- Log of control signals sent back to the inference server
CREATE TABLE IF NOT EXISTS control_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    step INTEGER NOT NULL,

    -- Signal content
    signal_type TEXT CHECK(signal_type IN (
        'layer_adjustment',    -- Change capture layers
        'stride_adjustment',   -- Change capture stride
        'bias_injection',      -- Inject attention bias
        'early_stop'           -- Signal to stop generation
    )),

    -- Signal payload (JSON)
    payload TEXT,

    -- What triggered this signal
    trigger_reason TEXT,                   -- e.g., 'zone_transition', 'entropy_spike'

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_control_signals_request ON control_signals(request_id);


-- =============================================================================
-- VIEWS
-- =============================================================================

-- Recent fingerprints with zone info (for UI streaming)
CREATE VIEW IF NOT EXISTS recent_fingerprints AS
SELECT
    f.id,
    f.request_id,
    f.step,
    f.token_text,
    f.think_phase,
    f.manifold_zone,
    f.manifold_confidence,
    f.cluster_id,
    c.label as cluster_label,
    f.router_entropy,
    f.sink_token_mass,
    f.created_at
FROM fingerprints f
LEFT JOIN clusters c ON f.cluster_id = c.cluster_id
ORDER BY f.created_at DESC
LIMIT 10000;


-- Zone distribution over time (for trend visualization)
CREATE VIEW IF NOT EXISTS zone_distribution_hourly AS
SELECT
    strftime('%Y-%m-%d %H:00:00', created_at) as hour,
    manifold_zone,
    COUNT(*) as count,
    AVG(manifold_confidence) as avg_confidence
FROM fingerprints
WHERE manifold_zone IS NOT NULL
GROUP BY hour, manifold_zone
ORDER BY hour DESC;


-- Cluster growth over time
CREATE VIEW IF NOT EXISTS cluster_growth AS
SELECT
    strftime('%Y-%m-%d', created_at) as day,
    cluster_id,
    COUNT(*) as count
FROM fingerprints
WHERE cluster_id >= 0
GROUP BY day, cluster_id
ORDER BY day DESC, count DESC;


-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update request_summary when fingerprints are inserted
-- (Note: For high-throughput, consider doing this in application code instead)
CREATE TRIGGER IF NOT EXISTS update_request_summary_on_insert
AFTER INSERT ON fingerprints
BEGIN
    INSERT INTO request_summary (request_id, total_steps, model_id, started_at)
    VALUES (NEW.request_id, 1, NEW.model_id, NEW.created_at)
    ON CONFLICT(request_id) DO UPDATE SET
        total_steps = total_steps + 1,
        think_steps = think_steps + CASE WHEN NEW.think_phase = 'think' THEN 1 ELSE 0 END,
        output_steps = output_steps + CASE WHEN NEW.think_phase = 'output' THEN 1 ELSE 0 END,
        finished_at = NEW.created_at;
END;


-- Update session timestamp when request is added
CREATE TRIGGER IF NOT EXISTS update_session_on_request
AFTER INSERT ON request_summary
WHEN NEW.session_id IS NOT NULL
BEGIN
    UPDATE sessions SET
        request_count = request_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE session_id = NEW.session_id;
END;


-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert placeholder for unassigned cluster
INSERT OR IGNORE INTO clusters (cluster_id, run_id, label, dominant_zone, size)
VALUES (-1, 'initial', 'Unassigned', 'unknown', 0);


-- =============================================================================
-- SCHEMA VERSION TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema with fingerprints, requests, sessions, discovery runs, clusters');
