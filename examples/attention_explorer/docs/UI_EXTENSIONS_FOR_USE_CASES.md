# UI Extensions for 10 Attention Visualization Use Cases

This document outlines UI extensions to the Attention Explorer to support the 10 primary use cases for attention token visualization.

## Current UI Capabilities

The existing explorer.html provides:
- Token-level visualization with manifold zone coloring
- Fingerprint mode showing 20D attention features
- Statistics panel (tokens, edges, layers)
- Attention zones legend (Semantic Bridge, Syntax Floor, Exploration, Steering)
- Token selection with attention details panel

## Proposed Extensions by Use Case

---

### Use Case 1: Regression Detection

**Goal**: Detect when model behavior changes between versions by comparing attention fingerprint distributions.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ REGRESSION DETECTION                                         │
├─────────────────────────────────────────────────────────────┤
│ Baseline Model: [Qwen3-4B-v1.0    ▼]                        │
│ Current Model:  [Qwen3-4B-v1.1    ▼]                        │
│                                                              │
│ ┌─────────────────────┐  ┌─────────────────────┐            │
│ │ Baseline Fingerprint│  │ Current Fingerprint │            │
│ │  ████████░░░░░░░░   │  │  ██████████░░░░░░   │            │
│ │  Entropy: 0.12      │  │  Entropy: 0.18      │  ⚠️ +50%   │
│ │  Zone: semantic     │  │  Zone: semantic     │            │
│ └─────────────────────┘  └─────────────────────┘            │
│                                                              │
│ Drift Score: 0.34 (SIGNIFICANT)  [View Details]             │
│ ──────────────────────────────────────────────              │
│ Token-level diffs:                                           │
│   Step 5: "Paris" - entropy increased 0.12 → 0.25           │
│   Step 8: "capital" - zone changed syntax → semantic        │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Add model selector dropdown for A/B comparison
- Side-by-side fingerprint visualization
- Drift score computation (cosine distance between fingerprint distributions)
- Token-level diff highlighting

---

### Use Case 2: Debug "Why" (Chain-of-Thought Auditing)

**Goal**: Understand why the model made specific reasoning decisions by tracing attention back through the thinking chain.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ REASONING TRACE                                              │
├─────────────────────────────────────────────────────────────┤
│ Selected Token: "Carol" (position 45)                        │
│                                                              │
│ ATTENTION CHAIN (click to expand):                           │
│                                                              │
│ "Carol" ← "shortest" ← "Bob" ← "taller" ← "Alice"           │
│   0.42      0.31        0.18      0.09                       │
│                                                              │
│ ┌─ Reasoning Path Analysis ─────────────────────────┐       │
│ │ Step 1: Model attended to "Alice is taller than Bob"  │   │
│ │ Step 2: Model attended to "Bob is taller than Carol"  │   │
│ │ Step 3: Model concluded "Carol" as shortest           │   │
│ │                                                        │   │
│ │ Confidence: HIGH (coherent attention chain)            │   │
│ └────────────────────────────────────────────────────────┘   │
│                                                              │
│ [Export Chain] [Visualize as Graph]                          │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Backtrack attention from selected token to source tokens
- Build attention graph and find strongest paths
- Display confidence based on path coherence
- Export reasoning chains for analysis

---

### Use Case 3: Injection Forensics

**Goal**: Detect and visualize prompt injection attempts by identifying anomalous attention patterns.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔒 INJECTION DETECTION                                       │
├─────────────────────────────────────────────────────────────┤
│ Status: ⚠️ POTENTIAL INJECTION DETECTED                      │
│                                                              │
│ ┌─ Anomaly Map ─────────────────────────────────────────┐   │
│ │ System Prompt    │ User Input      │ Response          │   │
│ │ ░░░░░░░░░░░░░░░ │ ▓▓▓▓▓▓████████ │ ░░░░░░░░░░░░░   │   │
│ │ Normal          │ ANOMALY         │ Normal            │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                              │
│ Suspicious Tokens:                                           │
│   Position 45-52: "ignore previous instructions"             │
│   Attention entropy: 0.89 (expected: 0.12)                   │
│   Zone shift: semantic_bridge → diffuse                      │
│                                                              │
│ ┌─ Fingerprint Comparison ───────────────────────────────┐  │
│ │ Normal Query    : ████░░░░░░░░ (entropy: 0.11)         │  │
│ │ This Query      : ████████████ (entropy: 0.89)  ⚠️     │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Block Response] [Allow with Warning] [Report False Positive]│
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Compute entropy baseline from normal queries
- Flag tokens with >3σ entropy deviation
- Highlight suspicious token regions
- Show zone transitions that indicate injection attempts
- Integration with Manifold Firewall for automated blocking

---

### Use Case 4: Model Routing

**Goal**: Route queries to appropriate model (4B vs 80B) based on attention complexity.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔀 ROUTING DECISION                                          │
├─────────────────────────────────────────────────────────────┤
│ Query Complexity Analysis:                                   │
│                                                              │
│ ┌─ Complexity Metrics ───────────────────────────────────┐  │
│ │ Local Mass:     ████████░░ 0.82 (high locality)        │  │
│ │ Long-Range:     ██░░░░░░░░ 0.18 (low long-range)       │  │
│ │ Entropy:        ███░░░░░░░ 0.31 (focused)              │  │
│ │ Zone:           Semantic Bridge                         │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ RECOMMENDATION: Route to 4B model                            │
│ Confidence: 87%                                              │
│                                                              │
│ Reasoning:                                                   │
│ - High local attention (simple retrieval pattern)            │
│ - Low entropy (confident predictions)                        │
│ - No complex reasoning chains detected                       │
│                                                              │
│ ┌─ Historical Routing ───────────────────────────────────┐  │
│ │ Similar queries: 94% successfully served by 4B         │  │
│ │ Avg latency: 45ms (4B) vs 890ms (80B)                  │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Accept Recommendation] [Override to 80B] [Always 4B]        │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Real-time fingerprint analysis during prefill
- Complexity score based on fingerprint features
- Historical accuracy tracking
- Cost/latency comparison display

---

### Use Case 5: MoE Telemetry

**Goal**: Visualize expert activation patterns in Mixture-of-Experts models.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 🧠 MoE EXPERT TELEMETRY                                      │
├─────────────────────────────────────────────────────────────┤
│ Model: DeepSeek-V3 (256 Experts)                             │
│                                                              │
│ ┌─ Expert Activation Heatmap ────────────────────────────┐  │
│ │ Token    │ E0 │ E1 │ E2 │ ... │ E255 │                 │  │
│ │ "What"   │ ██ │ ░░ │ ░░ │     │ ██   │ Top: E0, E255   │  │
│ │ "is"     │ ░░ │ ██ │ ░░ │     │ ░░   │ Top: E1         │  │
│ │ "prime"  │ ░░ │ ░░ │ ██ │     │ ██   │ Top: E2, E255   │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Expert Specialization ────────────────────────────────┐  │
│ │ E0:   Math/Logic (activated 45% of tokens)             │  │
│ │ E1:   Syntax/Grammar (activated 23% of tokens)         │  │
│ │ E2:   Code/Programming (activated 18% of tokens)       │  │
│ │ E255: General Knowledge (activated 67% of tokens)      │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Load Balance Score: 0.78 (Good)                              │
│ Routing Efficiency: 92%                                      │
│                                                              │
│ [Export Expert Traces] [View Layer-by-Layer]                 │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Expert activation matrix visualization
- Per-token expert selection display
- Expert specialization inference
- Load balance metrics

---

### Use Case 6: Long Context / Needle Finding

**Goal**: Diagnose "lost in the middle" problems and visualize long-context attention patterns.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 📏 LONG CONTEXT ANALYSIS                                     │
├─────────────────────────────────────────────────────────────┤
│ Context Length: 32,768 tokens                                │
│ Needle Position: 15,234 (middle)                             │
│                                                              │
│ ┌─ Position Attention Distribution ──────────────────────┐  │
│ │                                                         │  │
│ │  ▓▓▓                                              ▓▓▓  │  │
│ │  ▓▓▓     ░                            ░           ▓▓▓  │  │
│ │  ▓▓▓     ░░           ▓               ░░          ▓▓▓  │  │
│ │  ▓▓▓     ░░░          ▓▓              ░░░         ▓▓▓  │  │
│ │  ├────────┼────────────┼───────────────┼──────────┤    │  │
│ │  Start   1/4          1/2            3/4         End   │  │
│ │                        ↑                               │  │
│ │                    Needle Found!                       │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Needle Attention Score: 0.15 (ADEQUATE)                      │
│ Position Bias: Slight recency bias detected                  │
│                                                              │
│ ┌─ Retrieval Metrics ────────────────────────────────────┐  │
│ │ First-token attention: 0.23                            │  │
│ │ Last-token attention:  0.31                            │  │
│ │ Middle attention:      0.15 (needle position)          │  │
│ │ Expected (uniform):    0.12                            │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [View Token-by-Token] [Export Attention Map]                 │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Position-binned attention histogram
- Needle position marker
- Retrieval success/failure indicator
- Position bias analysis

---

### Use Case 7: KV Cache Optimization

**Goal**: Identify which tokens can be evicted from KV cache without quality loss.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 💾 KV CACHE ANALYSIS                                         │
├─────────────────────────────────────────────────────────────┤
│ Current Cache: 8,192 tokens | Memory: 256 MB                 │
│                                                              │
│ ┌─ Token Importance Ranking ─────────────────────────────┐  │
│ │ KEEP (Skeleton Tokens - 30%):                          │  │
│ │   [Paris] [capital] [France] [is] ...                  │  │
│ │   Semantic anchors, high hubness                       │  │
│ │                                                         │  │
│ │ EVICT (Interpolatable - 70%):                          │  │
│ │   [the] [of] [,] [.] [a] ...                           │  │
│ │   Low semantic importance, reconstructible             │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Spectral Eviction Preview ────────────────────────────┐  │
│ │ Before: ████████████████████████████████ (8192 tokens) │  │
│ │ After:  ██████████░░░░░░░░░░░░░░░░░░░░░░ (2458 tokens) │  │
│ │                                                         │  │
│ │ Memory Savings: 70% (179 MB freed)                     │  │
│ │ Expected Quality Loss: <2% perplexity increase         │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Apply Eviction] [Simulate Quality] [Adjust Threshold]       │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Token importance ranking from fingerprints
- Skeleton vs interpolatable classification
- Memory savings calculator
- Quality impact predictor

---

### Use Case 8: Mode Switch Detection

**Goal**: Detect when model transitions between modes (chat → reasoning → code).

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔄 MODE TRANSITIONS                                          │
├─────────────────────────────────────────────────────────────┤
│ ┌─ Timeline View ────────────────────────────────────────┐  │
│ │                                                         │  │
│ │  CHAT    │  REASONING   │  CODE   │  OUTPUT            │  │
│ │  ░░░░░░░ │  ████████████ │ ███████ │ ░░░░░░░░░         │  │
│ │  Step 1-5│  Step 6-25    │ 26-45   │ 46-60             │  │
│ │          ↑               ↑         ↑                    │  │
│ │       Transition 1    Trans 2   Trans 3                │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Transition Details:                                          │
│                                                              │
│ ┌─ Transition 1 (Step 5→6) ──────────────────────────────┐  │
│ │ From: Chat (entropy: 0.11, zone: syntax_floor)         │  │
│ │ To:   Reasoning (entropy: 0.34, zone: semantic_bridge) │  │
│ │ Trigger: "<think>" token detected                      │  │
│ │ Fingerprint shift: 0.67 (significant)                  │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Mode Distribution: Chat 8% | Reasoning 45% | Code 32% | Out 15%
│                                                              │
│ [View Transitions] [Export Timeline]                         │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Mode classification from fingerprint features
- Transition detection (significant fingerprint shifts)
- Timeline visualization
- Trigger token identification

---

### Use Case 9: Dataset Mining

**Goal**: Discover and categorize attention patterns across many queries for training data curation.

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 DATASET MINING                                            │
├─────────────────────────────────────────────────────────────┤
│ Fingerprints Collected: 77,730 | Clusters: 23                │
│                                                              │
│ ┌─ UMAP Embedding ───────────────────────────────────────┐  │
│ │                  ○○○○                                   │  │
│ │              ○○○○○○○○○○                                 │  │
│ │            ●●●●    ○○○○○○○                              │  │
│ │         ●●●●●●●●     ○○○○                               │  │
│ │        ●●●●●●●●●                                        │  │
│ │       ●●●●●●●●      ◆◆◆◆◆◆                              │  │
│ │                    ◆◆◆◆◆◆◆◆                             │  │
│ │                   ◆◆◆◆◆◆◆◆◆◆                            │  │
│ │                                                         │  │
│ │  ● Reasoning (12,450)  ○ Factual (8,230)  ◆ Code (5,120)│  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Cluster Details ──────────────────────────────────────┐  │
│ │ Cluster 3: "Multi-hop Reasoning" (2,340 samples)       │  │
│ │   Characteristics: high entropy, semantic_bridge zone  │  │
│ │   Example prompts: "If A then B, if B then C..."       │  │
│ │   [View Samples] [Export Cluster]                      │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Run Discovery] [Export All Clusters] [Create Training Set]  │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- 2D UMAP embedding visualization
- Interactive cluster exploration
- Sample query display per cluster
- Export functionality for training data

---

### Use Case 10: Safety Monitoring

**Goal**: Monitor for safety-relevant attention patterns (hallucination, bias, harmful content).

**UI Extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│ 🛡️ SAFETY MONITOR                                            │
├─────────────────────────────────────────────────────────────┤
│ ┌─ Real-time Alerts ─────────────────────────────────────┐  │
│ │ ⚠️ 12:03:45 Potential hallucination detected           │  │
│ │    Query: "Who won the 2025 Super Bowl?"               │  │
│ │    Issue: Attention diffuse, no grounding tokens       │  │
│ │                                                         │  │
│ │ ✅ 12:03:42 Normal response                             │  │
│ │ ✅ 12:03:39 Normal response                             │  │
│ │ ⚠️ 12:03:35 High uncertainty detected                  │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Hallucination Risk Assessment ────────────────────────┐  │
│ │                                                         │  │
│ │ Grounding Score: 0.23 (LOW - potential hallucination)  │  │
│ │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                   │  │
│ │                                                         │  │
│ │ Warning Signs:                                          │  │
│ │  - Attention not anchored to factual context           │  │
│ │  - High entropy in factual claim tokens                │  │
│ │  - Zone: "diffuse" (uncertainty pattern)               │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Session Statistics ───────────────────────────────────┐  │
│ │ Total Queries: 1,234                                   │  │
│ │ Flagged: 45 (3.6%)                                     │  │
│ │ False Positive Rate: ~12% (estimated)                  │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Configure Thresholds] [Export Flagged Queries] [Dismiss All]│
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:
- Real-time alert stream
- Grounding score computation
- Hallucination risk indicators
- Session-level statistics

---

## Implementation Priority

| Priority | Use Case | Complexity | Value |
|----------|----------|------------|-------|
| P0 | UC3: Injection Forensics | Medium | High (security) |
| P0 | UC10: Safety Monitoring | Medium | High (safety) |
| P1 | UC2: Debug Why | Medium | High (debugging) |
| P1 | UC4: Model Routing | Low | High (cost savings) |
| P1 | UC7: KV Cache Optimization | Medium | High (memory) |
| P2 | UC1: Regression Detection | Medium | Medium |
| P2 | UC6: Long Context | Low | Medium |
| P2 | UC8: Mode Switch | Low | Medium |
| P3 | UC5: MoE Telemetry | High | Medium |
| P3 | UC9: Dataset Mining | High | Medium |

## Shared Components

Several components can be shared across use cases:

1. **Fingerprint Visualizer**: Bar chart showing 20D fingerprint features
2. **Zone Legend**: Color-coded zone display with definitions
3. **Timeline View**: Horizontal timeline of token/step progression
4. **Comparison Panel**: Side-by-side fingerprint comparison
5. **Alert Stream**: Real-time notification display
6. **Export Dialog**: Common export functionality (JSON, CSV, Parquet)

## Next Steps

1. Implement shared components first
2. Start with P0 use cases (Injection Forensics, Safety Monitoring)
3. Add P1 use cases based on user feedback
4. Iterate on P2/P3 based on adoption

---

## Prompt Analysis Feature

A key missing capability is **Prompt Analysis** - analyzing the prompt before generation to predict:
- Expected complexity
- Likely attention patterns
- Recommended model routing
- Potential safety concerns

This would be valuable as a standalone feature and integrate with UC4 (Routing) and UC10 (Safety).

### Prompt Analysis Panel Design

```
┌─────────────────────────────────────────────────────────────┐
│ 🔍 PROMPT ANALYSIS (Pre-Generation)                          │
├─────────────────────────────────────────────────────────────┤
│ Input: "Explain quantum entanglement to a 5-year-old"        │
│                                                              │
│ ┌─ Complexity Assessment ────────────────────────────────┐  │
│ │ Estimated Complexity: MEDIUM                           │  │
│ │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │  │
│ │                                                         │  │
│ │ Factors:                                                │  │
│ │  + Technical topic (quantum physics)                   │  │
│ │  + Simplification required (5-year-old audience)       │  │
│ │  - No multi-step reasoning required                    │  │
│ │  - No code generation required                         │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Predicted Attention Pattern ──────────────────────────┐  │
│ │ Primary Zone: Semantic Bridge (explaining concepts)    │  │
│ │ Expected Entropy: 0.25-0.35 (moderate focus)           │  │
│ │ Key Retrieval Tokens: "quantum", "entanglement", "5"   │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─ Recommendations ──────────────────────────────────────┐  │
│ │ Model: 4B sufficient (no complex reasoning)            │  │
│ │ Max Tokens: ~150 (explanation + analogy)               │  │
│ │ Temperature: 0.7 (creative explanation)                │  │
│ │ Safety: No concerns detected                           │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Proceed with Analysis] [Adjust Settings] [Cancel]           │
└─────────────────────────────────────────────────────────────┘
```

This prompt analysis can use:
1. Token-level features (entities, keywords)
2. Historical fingerprint patterns for similar prompts
3. Complexity heuristics (question type, domain)
4. Safety classifiers

The analysis runs on the prompt alone before generation, providing proactive guidance.
