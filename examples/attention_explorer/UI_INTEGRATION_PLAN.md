# Attention Explorer UI Integration Plan

## Executive Summary

The Attention Explorer has **15+ sophisticated backend systems** but the UI currently exposes only basic token-level attention visualization. This plan outlines a progressive integration strategy that surfaces powerful capabilities without overwhelming users.

**Core Principle**: Progressive disclosure - show the right information at the right time.

---

## Current State Analysis

### What Works Well
1. **Token Grid** - Clear visualization of prompt/response tokens
2. **Zone Coloring** - Immediate visual feedback on attention patterns
3. **Attention Panel** - "Attends To" / "Attended By" shows clear relationships
4. **Streaming** - Real-time token arrival with edge capture
5. **Configuration** - Sensible defaults with expert controls

### What's Missing
| Backend System | Current UI Status | User Impact |
|----------------|-------------------|-------------|
| Manifold 2D Projection | Not exposed | Can't visualize attention space |
| Compass Router | Not exposed | No routing recommendations |
| Manifold Firewall | Not exposed | No hallucination warnings |
| Rotational Variance | Zone color only | No RV scores or timeline |
| Spectral Coherence | Not exposed | No complexity estimates |
| RoPE De-rotation | Not exposed | Can't see semantic attention |
| Layer Analysis | Button exists, empty | Can't analyze layer behavior |
| Cluster Metadata | Not exposed | Can't see cluster meaning |
| Threshold Tuner | Not exposed | Can't calibrate zones |

---

## Design Philosophy

### 1. Contextual Intelligence
Show insights when relevant, hide when not. Don't show hallucination warnings for simple math.

### 2. Visual Hierarchy
- **Primary**: Token grid + attention edges (current)
- **Secondary**: Zone info + stats (left panel)
- **Tertiary**: Deep analysis (expandable panels)

### 3. Entry Points
Different users need different depths:
- **Casual**: See tokens + colors (current default)
- **Explorer**: Click for details + patterns
- **Researcher**: Access full manifold analysis

### 4. Non-Intrusive Alerts
Warnings appear as subtle badges, not modal dialogs. Users can explore more if curious.

---

## Integration Roadmap

### Phase 1: Enhanced Token Insights (Low Risk)

**Goal**: Surface more backend data without changing layout.

#### 1.1 Token Tooltip Enhancement
Current: Shows token index on hover
Enhanced:
```
Token: "quantum"
Position: 47
Zone: semantic_bridge
Entropy: 0.73 (focused)
RV: 0.42 (mid-range)
Top Attention: [0] (94.2%)
```

**Implementation**: Extend `createTokenElement()` in both UIs.

#### 1.2 Attention Panel Enhancement
Current: Shows position + percentage
Enhanced: Add small sparkline showing attention distribution

```
"the" (pos 12) ████████░░ 82%  [syntax_floor]
"model" (pos 3) ███░░░░░░░ 31%  [semantic_bridge]
```

**Implementation**: Add zone badge to attention list items.

#### 1.3 Stats Panel Enhancement
Current: Tokens | Edges | Layers
Enhanced:
```
Tokens: 156  |  Edges: 1,240  |  Layers: 28
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zone Distribution:
[██████████░░░░░░░░░░] 48% Semantic Bridge
[████░░░░░░░░░░░░░░░░] 22% Syntax Floor
[███░░░░░░░░░░░░░░░░░] 18% Structure Ripple
[██░░░░░░░░░░░░░░░░░░] 12% Exploration
```

**Implementation**: Add zone histogram below stats.

---

### Phase 2: Manifold Visualization Panel (Medium Risk)

**Goal**: Add 2D manifold view as collapsible panel.

#### 2.1 Collapsible Manifold Panel
Position: Below token grid (full width, collapsed by default)
Expand: Click "Show Manifold" button or keyboard shortcut (M)

**Layout when expanded**:
```
┌──────────────────────────────────────────────────────────────┐
│  MANIFOLD VIEW                                    [Collapse] │
├─────────────────────────┬────────────────────────────────────┤
│                         │  Selected Cluster: #7              │
│     2D Scatter          │  Label: "Factual Retrieval"        │
│     (SVG/Canvas)        │  Size: 234 tokens (18%)            │
│                         │  Centroid: [0.42, -0.18]           │
│  [Current token ●]      │  Prototype: "The capital of..."    │
│  [Zone regions shaded]  │                                    │
│                         │  Zone: semantic_bridge (92%)       │
│                         │  Neighbors: #3, #12                │
├─────────────────────────┴────────────────────────────────────┤
│  Timeline: ●──●──●──●──●──●──●──●──●  (token trajectory)    │
└──────────────────────────────────────────────────────────────┘
```

**Interactions**:
- Hover token in grid → Highlight in manifold
- Click cluster → Show cluster info
- Drag to explore manifold space
- Timeline scrubber to see token sequence through manifold

**Data Source**: `discovery/bounded_umap.py` provides embeddings

---

### Phase 3: Intelligent Alerts System (Medium Risk)

**Goal**: Surface Compass Router and Manifold Firewall insights as non-intrusive alerts.

#### 3.1 Alert Badge System
Position: Top-right of token grid, next to "Connected" status

**Alert Types**:
```
┌─────────────────────────────────────────────────┐
│  COMPLEXITY: Moderate    ┊  CONFIDENCE: 0.87   │
│  ⚠ Zone drift at token 34                      │
└─────────────────────────────────────────────────┘
```

**Severity Levels** (from Manifold Firewall):
- 🟢 SAFE - Normal pattern (no badge)
- 🟡 WATCH - Minor drift (subtle yellow dot)
- 🟠 WARNING - Suspicious (orange badge, clickable)
- 🔴 ALERT - Likely hallucination (red badge, prominent)
- ⛔ CRITICAL - Strong signal (red pulse, tooltip shows details)

#### 3.2 Compass Router Insights Panel
Position: Collapsible section in left panel, below Configuration

```
┌─────────────────────────────────────────────────┐
│  ROUTING INSIGHTS            [?]                │
├─────────────────────────────────────────────────┤
│  Heading:  NORTHEAST (retrieval-focused)        │
│  Variance: ██████░░░░ 0.62 (moderate)          │
│                                                 │
│  Recommendation:                                │
│  ✓ Medium model suitable                        │
│  ✓ Chain-of-thought may help                   │
│  → Estimated complexity: MODERATE               │
└─────────────────────────────────────────────────┘
```

**Compass Rose Visualization** (optional enhancement):
```
        N (sink)
         │
    NW   │   NE
      \  │  /
   W ────┼──── E
      /  │  \
    SW   │   SE
         │
        S
     [●] Current
```

---

### Phase 4: Layer Analysis Modal (Low-Medium Risk)

**Goal**: Complete the existing "Layer Heatmap" button functionality.

#### 4.1 Layer Heatmap Modal
Trigger: Click "Layer Heatmap" button (already exists)

**Modal Content**:
```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER-BY-LAYER ATTENTION                              [✕ Close] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 0  ░░░░░░░░░░░░░░░░░░░░  (embedding, no attention)       │
│  Layer 1  ██████░░░░░░░░░░░░░░  entropy: 1.2 (focused)          │
│  Layer 2  ████████░░░░░░░░░░░░  entropy: 1.8                    │
│  ...                                                             │
│  Layer 27 ████████████████████  entropy: 3.1 (diffuse)          │
│                                                                  │
│  Selected: Layer 27                                              │
│  ├─ Zone: semantic_bridge (this layer)                          │
│  ├─ Top attended: [0] system, [12] "model", [45] "attention"    │
│  └─ Attention mass: 32% local, 48% mid, 20% long                │
│                                                                  │
│  [Apply Layer Filter]  [Show All Layers]  [Export Data]         │
└──────────────────────────────────────────────────────────────────┘
```

**Implementation**: Data exists in fingerprint `layers` field.

---

### Phase 5: RoPE De-rotation View (Advanced)

**Goal**: Show semantic vs positional attention breakdown.

#### 5.1 Semantic Attention Toggle
Position: New toggle in Configuration section

```
[ ] Word Mode
[✓] De-rotate RoPE  ← NEW
```

When enabled:
- Attention scores show semantic component only
- Edges colored by semantic strength
- Tooltip shows: "Raw: 82%, Semantic: 47%, Positional: 35%"

#### 5.2 Semantic Cluster Highlighting
When de-rotation active, tokens forming semantic clusters get special styling:
- Dashed border around cluster groups
- Cluster label on hover: "Semantic cluster: SUBJECT_VERB"

---

### Phase 6: Timeline & History View (Advanced)

**Goal**: Show attention evolution over generation.

#### 6.1 RV Timeline Plot
Position: Collapsible panel below manifold view

```
RV │    ╭──╮                  ╭─────╮
   │   ╱    ╲    ╭─╮         ╱       ╲
   │──╱      ╲──╱   ╲───────╱         ╲──
   └─┴────────┴──────┴───────┴──────────┴─▶ Tokens
     [syntax_floor]  [bridge]  [ripple]
```

Zones shown as colored background bands.

#### 6.2 Attention Flow Animation
Button: "Play Attention Flow"
- Animates edges appearing token-by-token
- Shows how attention patterns evolved during generation
- Speed control slider

---

### Phase 7: Threshold Calibration UI (Expert)

**Goal**: Allow researchers to tune zone classification.

#### 7.1 Calibration Panel
Access: Settings menu → "Advanced" → "Calibrate Zones"

```
┌─────────────────────────────────────────────────────────────┐
│  ZONE THRESHOLD CALIBRATION                                 │
├─────────────────────────────────────────────────────────────┤
│  Current Accuracy: 87.3%                                    │
│                                                             │
│  Thresholds:                                                │
│  ├─ Syntax Floor                                            │
│  │   local_mass: [0.5 ─────●───── 0.8] = 0.58              │
│  │   entropy:    [1.5 ───●─────── 3.0] = 2.1               │
│  │   rv_max:     [0.1 ─────●───── 0.4] = 0.25              │
│  │                                                          │
│  ├─ Semantic Bridge                                         │
│  │   [similar sliders]                                      │
│  │                                                          │
│  └─ Structure Ripple                                        │
│      [similar sliders]                                      │
│                                                             │
│  [Test on Current Data]  [Reset Defaults]  [Save Profile]   │
│                                                             │
│  Confusion Matrix:                                          │
│            Predicted                                        │
│          SYN  SEM  RIP                                      │
│  Actual SYN [94] [ 4] [ 2]                                  │
│         SEM [ 3] [89] [ 8]                                  │
│         RIP [ 1] [ 6] [93]                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Unified UI Layout Proposal

### Desktop Layout (1920x1080+)
```
┌────────────────────────────────────────────────────────────────────────────┐
│  SGLang Attention Explorer                    [Connected] [Alerts: 0] [?]  │
├──────────────────┬─────────────────────────────────────┬───────────────────┤
│                  │                                     │                   │
│   INPUT          │        TOKEN GRID                   │   TOKEN DETAILS   │
│   [textarea]     │   ┌─────────────────────────┐      │                   │
│                  │   │ PROMPT                   │      │   "quantum"       │
│   CONFIGURATION  │   │ [tok] [tok] [tok]       │      │   pos:47 | bridge │
│   ├─ Max Tokens  │   ├─────────────────────────┤      │                   │
│   ├─ Top-K       │   │ RESPONSE (streaming)    │      │   ATTENDS TO      │
│   ├─ Temperature │   │ [tok] [tok] [tok] ▌     │      │   ├─ "the" 82%    │
│   └─ [Advanced]  │   └─────────────────────────┘      │   ├─ "model" 31%  │
│                  │                                     │   └─ [more...]    │
│   STATS          │   ┌─────────────────────────────┐  │                   │
│   156 tokens     │   │ MANIFOLD VIEW [collapsed]   │  │   ATTENDED BY     │
│   1240 edges     │   │ [Click to expand]           │  │   ├─ "is" 44%     │
│   28 layers      │   └─────────────────────────────┘  │   └─ [more...]    │
│                  │                                     │                   │
│   ZONES          │   ┌─────────────────────────────┐  │   FINGERPRINT     │
│   [histogram]    │   │ TIMELINE [collapsed]        │  │   [mini radar]    │
│                  │   └─────────────────────────────┘  │                   │
│   ROUTING        │                                     │                   │
│   [compass mini] │                                     │                   │
│                  │                                     │                   │
└──────────────────┴─────────────────────────────────────┴───────────────────┘
```

### Mobile/Tablet Layout
- Single column, panels become accordions
- Token grid scrolls horizontally
- Details panel becomes slide-up drawer

---

## Implementation Priority Matrix

| Feature | Impact | Effort | Risk | Priority |
|---------|--------|--------|------|----------|
| Token tooltip enhancement | High | Low | Low | **P0** |
| Zone histogram in stats | High | Low | Low | **P0** |
| Attention list zone badges | Medium | Low | Low | **P0** |
| Alert badge system | High | Medium | Low | **P1** |
| Compass router mini-panel | High | Medium | Medium | **P1** |
| Layer heatmap modal | Medium | Medium | Low | **P1** |
| Manifold 2D panel | High | High | Medium | **P2** |
| RV timeline plot | Medium | Medium | Low | **P2** |
| De-rotation toggle | Medium | High | Medium | **P3** |
| Threshold calibration | Low | High | Low | **P3** |
| Attention flow animation | Low | High | Medium | **P4** |

---

## User Flow Examples

### Flow 1: Casual User
1. Enter prompt → Click Analyze
2. See tokens colored by zone
3. Click a token → See what it attends to
4. Done (never expands advanced panels)

### Flow 2: Explorer
1. Enter prompt → Click Analyze
2. Notice orange warning badge → Click to expand
3. See "Zone drift at token 34" → Click token 34
4. Expand manifold view → See token's position in cluster
5. Understand the pattern shift

### Flow 3: Researcher
1. Enter prompt with specific test case
2. Enable layer filter → Analyze layer 27 specifically
3. Open threshold calibration → Adjust for this model
4. Export data for further analysis
5. Train spectral router on collected samples

---

## Technical Considerations

### Performance
- Manifold view: Use WebGL/Canvas for 1000+ points
- Lazy-load advanced panels (don't compute until expanded)
- Cache fingerprints client-side for comparison

### Data Flow
```
SGLang API
    │
    ▼
attention_ws_server.py (adds streaming)
    │
    ├─► Token messages (current)
    │
    └─► Enriched messages (new)
        ├─ compass_routing: {heading, variance, recommendation}
        ├─ firewall_status: {severity, drift_events}
        ├─ spectral_coherence: float
        └─ derotated_attention: {...}
```

### Backwards Compatibility
- All new features are additive
- Old API responses still work (missing fields = hide panel)
- Feature detection: `if (data.compass_routing) showCompassPanel()`

---

## Success Metrics

### Quantitative
- Average session duration increases (engagement)
- "Expand advanced panel" rate (curiosity)
- Error badge click-through rate (utility)

### Qualitative
- Users can explain why a token was flagged
- Researchers export data for papers
- New users understand zones within 5 minutes

---

## Appendix: Component Specifications

### Alert Badge Component
```javascript
class AlertBadge {
  constructor(container) {
    this.severity = 'SAFE';  // SAFE|WATCH|WARNING|ALERT|CRITICAL
    this.events = [];
    this.expanded = false;
  }

  update(firewallStatus) {
    this.severity = firewallStatus.severity;
    this.events = firewallStatus.drift_events;
    this.render();
  }

  render() {
    // Badge: dot with color + optional count
    // Expanded: list of events with timestamps
  }
}
```

### Zone Histogram Component
```javascript
class ZoneHistogram {
  constructor(container) {
    this.counts = { syntax_floor: 0, semantic_bridge: 0, ... };
  }

  update(tokens) {
    // Count zones from token list
    // Render horizontal bar chart
  }
}
```

### Manifold Panel Component
```javascript
class ManifoldPanel {
  constructor(container) {
    this.embeddings = null;  // [[x,y], ...]
    this.clusters = null;
    this.collapsed = true;
  }

  async load() {
    // Fetch embeddings from backend
    // Initialize WebGL renderer
  }

  highlightToken(index) {
    // Flash the point, draw trajectory
  }
}
```

---

## Next Steps

1. **Immediate** (This session):
   - Implement P0 features (tooltip, histogram, zone badges)
   - Test with streaming UI

2. **Short-term** (Next sessions):
   - Implement P1 features (alerts, compass, layer modal)
   - User testing with sample prompts

3. **Medium-term**:
   - Implement P2 features (manifold, timeline)
   - Integration with discovery pipeline

4. **Long-term**:
   - P3/P4 features
   - Mobile optimization
   - Documentation and tutorials

---

*Document created: 2025-01-15*
*Author: Claude Code Assistant*
*Status: Draft - Ready for Review*
