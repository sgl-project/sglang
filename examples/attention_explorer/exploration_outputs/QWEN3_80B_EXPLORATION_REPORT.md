# Qwen3-80B Attention Fingerprint Exploration Report

**Date**: 2026-01-10
**Duration**: 4.06 hours
**Model**: Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
**Session ID**: 20260110_083804

## Executive Summary

This report documents a 4-hour exploration of attention patterns in the Qwen3-80B-FP8 model, a Mixture-of-Experts (MoE) architecture with 80B total parameters and 3B active parameters. The exploration collected 129,254 attention fingerprints across 96 diverse prompts, with 8 discovery runs analyzing the manifold structure.

### Key Findings

1. **Response Length is the Primary Routing Signal**: Zone distribution strongly correlates with response length rather than task type
   - Short responses (<500 tokens): 60-80% semantic_bridge
   - Long responses (>1000 tokens): 65-82% structure_ripple

2. **Minimal Syntax Floor Usage**: The MoE architecture shows only 0.8% syntax_floor attention across all categories, indicating reliance on mid/long-range attention patterns

3. **Stable Zone Distribution**: Zone proportions stabilized at ~65% structure_ripple, ~34% semantic_bridge, ~1% syntax_floor

4. **Manifold Fragmentation at Scale**: HDBSCAN clustering shows increasing fragmentation with data volume (6 → 1455 clusters)

## Methodology

### Data Collection
- **SGLang Server**: Triton attention backend with top-k=32 attention capture
- **Sidecar Service**: Real-time fingerprint storage and classification
- **Discovery Interval**: 30 minutes for cluster analysis

### Prompt Categories
| Category | Count | Description |
|----------|-------|-------------|
| instruction_following | 13 | Precise formatting tasks |
| factual | 13 | Knowledge retrieval |
| analysis | 13 | Deep analytical tasks |
| math | 12 | Calculations and proofs |
| creative | 11 | Poetry, stories, dialogue |
| reasoning | 11 | Logic puzzles, CRT problems |
| coding | 11 | Algorithm implementation |
| edge_cases | 5 | Paradoxes, philosophical |
| multi_turn_context | 3 | Follow-up questions |
| long_context | 2 | Document analysis |
| roleplay | 2 | Character simulation |

## Zone Distribution Analysis

### Final Distribution (129,254 tokens)
| Zone | Tokens | Percentage |
|------|--------|------------|
| structure_ripple | 84,674 | 65.5% |
| semantic_bridge | 43,565 | 33.7% |
| syntax_floor | 1,015 | 0.8% |

### Zone Characteristics

**structure_ripple (65.5%)**
- Dominant for responses >1000 tokens
- Associated with: analysis, creative writing, code implementation
- Indicates periodic/structural attention patterns

**semantic_bridge (33.7%)**
- Dominant for responses <500 tokens
- Associated with: factual queries, short reasoning, algorithm logic
- Indicates mid-range contextual attention

**syntax_floor (0.8%)**
- Minimal usage across all categories
- No strong correlation with any task type
- Suggests MoE architecture relies on semantic/structural patterns

## Category-Specific Patterns

### By Response Length
| Length Range | Dominant Zone | Typical Categories |
|--------------|---------------|-------------------|
| <300 tokens | 75-80% semantic_bridge | Simple factual, short instructions |
| 300-800 tokens | 55-60% semantic_bridge | Math calculations, algorithm logic |
| 800-1500 tokens | 55-65% structure_ripple | Reasoning puzzles, medium creative |
| >1500 tokens | 75-82% structure_ripple | Analysis, complex creative, code |

### Notable Findings by Category

**Coding**
- Binary search (simple algorithm): 67% semantic_bridge
- LRU cache (complex implementation): 58% structure_ripple
- Trie (data structure): 55-57% semantic_bridge
- Thread-safe singleton: 61% structure_ripple

**Reasoning**
- CRT bat/ball problem: 59% semantic_bridge (trick question stays in reasoning mode)
- Two guards riddle: 71% structure_ripple (complex step-by-step)
- Sheep riddle: 65% semantic_bridge (simple trick)

**Creative**
- Haiku (short): 75% semantic_bridge
- Sonnet (long): 53% structure_ripple
- Dialogue (medium): 72% structure_ripple

**Edge Cases**
- Paradox questions: 53% structure_ripple
- Philosophical (Big Bang): 74% structure_ripple
- Describing blue to blind: 81% structure_ripple

## Manifold Evolution

### Cluster Analysis Across Discovery Runs
| Run | Fingerprints | Clusters | Noise % | Notes |
|-----|--------------|----------|---------|-------|
| 1st | 17,907 | 6 | 0.04% | Initial consolidation |
| 2nd | 36,261 | 526 | 43% | Fragmentation begins |
| 3rd | 52,032 | 635 | 41% | Peak fragmentation |
| 4th | 70,790 | 53 | 0.3% | Reconsolidation |
| 5th | 85,866 | 60 | 0.2% | Stable manifold |
| 6th | 105,010 | 1,354 | 48% | Scale-induced fragmentation |
| 7th | 124,289 | 1,455 | 45% | Continued fragmentation |
| 8th | 129,254 | 1,527 | 44% | Final state |

### Key Observations
1. **Initial Consolidation**: Small data creates coherent clusters
2. **Mid-Session Stability**: 50K-90K fingerprints show stable ~60 clusters
3. **Large-Scale Fragmentation**: >100K fingerprints exceed HDBSCAN's ability to find coherent structure

## Throughput Analysis

### Token Generation Speed
| Time Window | Throughput (tok/s) | Notes |
|-------------|-------------------|-------|
| 0-30 min | 10.0-10.8 | Peak performance |
| 30-60 min | 7.1-7.6 | Thermal throttling |
| 60-120 min | 9.0-10.5 | Recovery |
| 120-240 min | 10.0-10.5 | Stable |

### Factors Affecting Throughput
- Thermal cycling on 80B model
- KV cache pressure at scale
- Recovery after thermal management

## Stochastic Variation

Repeated identical prompts showed significant variation:

**"List the days of the week in reverse order"**
| Run | Tokens | semantic_bridge | structure_ripple |
|-----|--------|-----------------|------------------|
| #42 | 1,843 | 20% | 80% |
| #43 | 944 | 29% | 70% |
| #71 | 237 | 99% | 0% |
| #89 | 534 | 51% | 49% |
| #90 | 620 | 43% | 56% |

This demonstrates that zone classification depends primarily on response length, which varies stochastically.

## Implications for Routing

### Quality Optimization
- Long-form tasks naturally engage structural attention patterns
- Short responses maintain semantic coherence
- Complex implementations benefit from structure_ripple mode

### Speed Optimization
- Force shorter responses to stay in faster semantic_bridge mode
- Truncation may affect quality of structural tasks

### Quantization Considerations
- structure_ripple zones may be more sensitive to quantization
- syntax_floor is minimal, suggesting less impact from local precision loss

## Recommendations

1. **Response Length Control**: Use max_tokens to influence zone engagement
2. **Task-Appropriate Settings**: Allow longer responses for analysis/creative
3. **Quantization Testing**: Focus testing on structure_ripple-heavy workloads
4. **Cluster Size Tuning**: For large-scale discovery, increase min_cluster_size

## Technical Details

### Discovery Parameters
- `min_cluster_size`: 10
- `min_samples`: 10
- `umap_neighbors`: 15
- `umap_min_dist`: 0.1
- `pca_components`: 50

### Fingerprint Dimensions
- [0-2] local_mass, mid_mass, long_mass
- [3] entropy
- [4-11] 8-bin histogram
- [12-19] 8 layer entropies

### Files Generated
- Database: `exploration_fingerprints.db` (129K fingerprints)
- Discovery runs: 8 complete runs
- Session summary: `session_20260110_083804_summary.json`
- Findings log: `findings_20260110_083804.jsonl`

## Conclusion

The Qwen3-80B-FP8 model shows a clear attention pattern hierarchy:
- **structure_ripple** for extended generation requiring document-level coherence
- **semantic_bridge** for focused reasoning and retrieval
- **syntax_floor** minimally used, suggesting MoE relies on higher-level patterns

Response length is the strongest predictor of zone engagement, more so than task category. This insight can guide routing decisions, quantization strategies, and inference optimization.

---
*Report generated: 2026-01-10T12:43:15*
