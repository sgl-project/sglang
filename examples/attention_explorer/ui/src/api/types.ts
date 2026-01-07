// Data contracts for Latent Chat Explorer
// Matches SGLang attention capture output (schema_version: 1)

// ============================================================================
// CORE ENUMS & CONSTANTS
// ============================================================================

export type CaptureMode = 'raw' | 'sketch' | 'fingerprint';
export type ThinkPhase = 'think' | 'output';
export type ManifoldZone =
  | 'syntax_floor'
  | 'semantic_bridge'
  | 'long_range'
  | 'structure_ripple'
  | 'diffuse'
  | 'unknown';

export type Program = 'prod' | 'debug' | 'discovery';
export type View = 'chat' | 'inspect' | 'manifold' | 'router' | 'compare' | 'lens';

export const SCHEMA_VERSION = 1;

// ============================================================================
// ATTENTION DATA - RAW MODE
// ============================================================================

export interface AttentionLayerRaw {
  token_positions: number[];
  attention_scores: number[];
  topk_logits?: number[];
  logsumexp_candidates?: number;
  topk_mass?: number;
}

export interface AttentionEntryRaw {
  schema_version: 1;
  mode: 'raw';
  token_positions: number[];
  attention_scores: number[];
  layer_id: number;
  topk_logits?: number[];
  logsumexp_candidates?: number;
  topk_mass?: number;
  layers?: Record<number, AttentionLayerRaw>;
  decode_step: number;
  think_phase: ThinkPhase;
}

// ============================================================================
// ATTENTION DATA - SKETCH MODE
// ============================================================================

export interface AttentionSketch {
  top_hubs: number[];
  hub_scores: number[];
  dist_hist: number[];
  entropy: number;
  local_mass: number;
  mid_mass: number;
  long_mass: number;
  consensus_positions?: number[];
}

export interface AttentionEntrySketch {
  schema_version: 1;
  mode: 'sketch';
  sketch?: AttentionSketch;
  layer_id?: number;
  layer_sketches?: Record<number, AttentionSketch>;
  decode_step: number;
  think_phase: ThinkPhase;
}

// ============================================================================
// ATTENTION DATA - FINGERPRINT MODE
// ============================================================================

export interface Fingerprint {
  histogram: number[];
  local_mass: number;
  mid_mass: number;
  long_mass: number;
  entropy: number;
  fft_low?: number;
  consensus?: number;
  hubness?: number;
}

export interface AttentionEntryFingerprint {
  schema_version: 1;
  mode: 'fingerprint';
  fingerprint: number[];
  manifold: ManifoldZone;
  moe?: MoETelemetry;
  step: number;
  think_phase: ThinkPhase;
}

// ============================================================================
// MOE ROUTING DATA
// ============================================================================

export interface ExpertSelection {
  id: number;
  weight: number;
}

export interface MoELayerRouting {
  expert_ids: number[];
  expert_weights: number[];
  entropy: number;
}

export interface MoETelemetry {
  top_experts: Record<number, number[]>;
  expert_weights?: Record<number, number[]>;
  entropy: Record<number, number>;
  hubness: Record<number, number>;
  top_hubs: Record<number, number[]>;
  expert_churn?: number;
}

export interface MoERoutingEntry {
  decode_step: number;
  layers: Record<number, MoELayerRouting>;
  entropy_mean?: number;
  hubness_mean?: number;
  expert_churn?: number;
}

// ============================================================================
// LOGIT LENS DATA (Experimental)
// ============================================================================

export interface LogitLensLayerResult {
  layer_id: number;
  top_token_ids: number[];
  top_tokens: string[];
  top_probs: number[];
  entropy: number;
  kl_from_final?: number;
}

export interface LogitLensEntry {
  probed_layers: number[];
  total_layers: number;
  layers: Record<string, LogitLensLayerResult>;
  final?: {
    top_token_ids: number[];
    top_tokens: string[];
    top_probs: number[];
  };
  decode_step: number;
  token_text?: string;
}

export interface LogitLensSession {
  entries: LogitLensEntry[];
  model: string;
  total_layers: number;
}

// ============================================================================
// UNIFIED ATTENTION ENTRY
// ============================================================================

export type AttentionEntry =
  | AttentionEntryRaw
  | AttentionEntrySketch
  | AttentionEntryFingerprint;

export function isRawMode(entry: AttentionEntry): entry is AttentionEntryRaw {
  return entry.mode === 'raw';
}

export function isSketchMode(entry: AttentionEntry): entry is AttentionEntrySketch {
  return entry.mode === 'sketch';
}

export function isFingerprintMode(entry: AttentionEntry): entry is AttentionEntryFingerprint {
  return entry.mode === 'fingerprint';
}

// ============================================================================
// MANIFOLD & CLUSTERING
// ============================================================================

export interface ManifoldPoint {
  session_id: string;
  coords: [number, number];
  cluster_id: number;
  cluster_name?: string;
  manifold_zone: ManifoldZone;
  consensus: number;
  hubness: number;
  entropy: number;
  timestamp: string;
  model_id?: string;
}

export interface ClusterDefinition {
  cluster_id: number;
  name: string;
  centroid: [number, number];
  radius: number;
  avg_consensus: number;
  avg_hubness: number;
  avg_entropy: number;
  dominant_zone: ManifoldZone;
  top_experts?: number[];
  point_count: number;
  stability: number;
}

export interface ManifoldArtifacts {
  points: ManifoldPoint[];
  clusters: ClusterDefinition[];
  projection_method: 'parametric_umap' | 'centroid_knn' | 'none';
  created_at: string;
  model_id: string;
  point_count: number;
}

// ============================================================================
// API REQUEST/RESPONSE
// ============================================================================

export interface AttentionCaptureParams {
  return_attention_tokens?: boolean;
  top_k_attention?: number;
  attention_capture_layer_ids?: number[];
  attention_sketch_mode?: boolean;
  return_moe_routing?: boolean;
  moe_routing_top_k?: number;
}

export interface ChatCompletionWithAttention {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: 'assistant'; content: string };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  attention_tokens?: AttentionEntry[];
  moe_routing?: MoERoutingEntry[];
}

export interface ChatCompletionChunkWithAttention {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: { role?: 'assistant'; content?: string };
    finish_reason: string | null;
  }>;
  attention_token?: AttentionEntry;
  moe_routing_step?: MoERoutingEntry;
}

// ============================================================================
// UI STATE TYPES
// ============================================================================

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  tokens?: string[];
  attention?: AttentionEntry[];
  moe?: MoERoutingEntry[];
  fingerprint?: Fingerprint;
  manifold_zone?: ManifoldZone;
  timestamp: number;
  model?: string;
}

export interface ResolvedToken {
  text: string;
  index: number;
  type: 'input' | 'output';
  attention?: AttentionEntry;
  moe?: MoERoutingEntry;
  isSelected: boolean;
  isHovered: boolean;
  isAttended: boolean;
  isAnchor: boolean;
  attendedScore?: number;
  isLowMass: boolean;
}

export interface LayerInfo {
  id: number;
  name: string;
  hasData: boolean;
  isFullAttention: boolean;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export function extractFingerprint(entry: AttentionEntry): Fingerprint | null {
  if (isFingerprintMode(entry)) {
    const fp = entry.fingerprint;
    return {
      histogram: fp.slice(0, 16),
      local_mass: fp[16] ?? 0,
      mid_mass: fp[17] ?? 0,
      long_mass: fp[18] ?? 0,
      entropy: fp[19] ?? 0,
    };
  }

  if (isSketchMode(entry)) {
    const sketch = entry.sketch || Object.values(entry.layer_sketches || {})[0];
    if (!sketch) return null;
    return {
      histogram: sketch.dist_hist,
      local_mass: sketch.local_mass,
      mid_mass: sketch.mid_mass,
      long_mass: sketch.long_mass,
      entropy: sketch.entropy,
    };
  }

  // Compute fingerprint from raw attention data (client-side)
  if (isRawMode(entry)) {
    return computeFingerprintFromRaw(entry);
  }

  return null;
}

/**
 * Compute fingerprint metrics from raw attention positions and scores.
 * This enables Manifold/Router views without server fingerprint mode.
 *
 * Distance bands (relative to current position):
 * - Local: 0-32 tokens (syntax/immediate context)
 * - Mid: 33-256 tokens (semantic/paragraph level)
 * - Long: 257+ tokens (document-level/cross-context)
 */
function computeFingerprintFromRaw(entry: AttentionEntryRaw): Fingerprint {
  const positions = entry.token_positions || [];
  const scores = entry.attention_scores || [];

  // Also check multi-layer data
  const allPositions: number[] = [];
  const allScores: number[] = [];

  if (entry.layers) {
    for (const layer of Object.values(entry.layers)) {
      allPositions.push(...(layer.token_positions || []));
      allScores.push(...(layer.attention_scores || []));
    }
  } else {
    allPositions.push(...positions);
    allScores.push(...scores);
  }

  if (allPositions.length === 0) {
    return {
      histogram: new Array(16).fill(0),
      local_mass: 0,
      mid_mass: 0,
      long_mass: 0,
      entropy: 0,
    };
  }

  // Compute current position (assume last attended position + 1, or max position)
  const maxPos = Math.max(...allPositions);
  const currentPos = maxPos + 1;

  // Compute distance histogram (16 bins, log-scale)
  const histogram = new Array(16).fill(0);
  let localMass = 0;
  let midMass = 0;
  let longMass = 0;
  let totalMass = 0;

  for (let i = 0; i < allPositions.length; i++) {
    const pos = allPositions[i];
    const score = allScores[i] || 0;
    const distance = currentPos - pos;

    // Distance bands
    if (distance <= 32) {
      localMass += score;
    } else if (distance <= 256) {
      midMass += score;
    } else {
      longMass += score;
    }
    totalMass += score;

    // Log-scale histogram bin
    const bin = Math.min(15, Math.floor(Math.log2(Math.max(1, distance))));
    histogram[bin] += score;
  }

  // Normalize
  if (totalMass > 0) {
    localMass /= totalMass;
    midMass /= totalMass;
    longMass /= totalMass;
    for (let i = 0; i < histogram.length; i++) {
      histogram[i] /= totalMass;
    }
  }

  // Compute entropy from score distribution
  let entropy = 0;
  for (const score of allScores) {
    if (score > 0) {
      const p = score / totalMass;
      entropy -= p * Math.log2(p);
    }
  }

  // Compute consensus (how much top-k agrees across layers)
  let consensus = 0;
  if (entry.layers && Object.keys(entry.layers).length > 1) {
    const layerTopK = Object.values(entry.layers).map(l =>
      new Set((l.token_positions || []).slice(0, 5))
    );
    if (layerTopK.length >= 2) {
      let overlaps = 0;
      let comparisons = 0;
      for (let i = 0; i < layerTopK.length; i++) {
        for (let j = i + 1; j < layerTopK.length; j++) {
          const intersection = [...layerTopK[i]].filter(x => layerTopK[j].has(x));
          overlaps += intersection.length / 5;
          comparisons++;
        }
      }
      consensus = comparisons > 0 ? overlaps / comparisons : 0;
    }
  }

  // Compute hubness (concentration of attention on few positions)
  const sortedScores = [...allScores].sort((a, b) => b - a);
  const top3Mass = sortedScores.slice(0, 3).reduce((a, b) => a + b, 0);
  const hubness = totalMass > 0 ? top3Mass / totalMass : 0;

  return {
    histogram,
    local_mass: localMass,
    mid_mass: midMass,
    long_mass: longMass,
    entropy,
    consensus,
    hubness,
  };
}

export function classifyManifold(fp: Fingerprint): ManifoldZone {
  if (fp.local_mass > 0.5) return 'syntax_floor';
  if (fp.mid_mass > 0.5) return 'semantic_bridge';
  if (fp.long_mass > 0.5) return 'long_range';
  if (fp.fft_low && fp.fft_low > 0.6) return 'structure_ripple';
  return 'diffuse';
}

export function getTopKForLayer(
  entry: AttentionEntry,
  layerId: number,
  k: number = 5
): Array<{ position: number; score: number }> {
  if (isRawMode(entry)) {
    const layer = entry.layers?.[layerId] || (entry.layer_id === layerId ? entry : null);
    if (!layer) return [];

    return layer.token_positions
      .map((pos, i) => ({ position: pos, score: layer.attention_scores[i] }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  if (isSketchMode(entry)) {
    const sketch = entry.layer_sketches?.[layerId] || entry.sketch;
    if (!sketch) return [];

    return sketch.top_hubs
      .map((pos, i) => ({ position: pos, score: sketch.hub_scores[i] }))
      .slice(0, k);
  }

  return [];
}

// ============================================================================
// TRACE SESSION - CANONICAL DATA MODEL
// ============================================================================

/**
 * Segment types for trace analysis
 * - system: System prompt tokens
 * - user: User message tokens
 * - assistant_think: Assistant's reasoning tokens (inside <think> tags)
 * - assistant_final: Assistant's final output tokens
 */
export type SegmentType = 'system' | 'user' | 'assistant_think' | 'assistant_final';

/**
 * A contiguous segment of tokens with the same role/phase
 */
export interface Segment {
  id: string;
  type: SegmentType;
  startTokenIndex: number;
  endTokenIndex: number;  // Exclusive
  messageId: string;
}

/**
 * A single token in the trace with full context
 */
export interface TokenEntry {
  index: number;          // Global index in the trace
  tokenId?: number;       // Model's token ID (if available)
  text: string;           // Decoded text
  segmentId: string;      // Which segment this belongs to
  role: 'prompt' | 'generated';
}

/**
 * A decode step captures attention/routing for one generated token
 */
export interface DecodeStep {
  tokenIndex: number;     // Which token this step generated
  attention?: AttentionEntry;
  moe?: MoERoutingEntry;
  fingerprint?: Fingerprint;
  manifoldZone?: ManifoldZone;
}

/**
 * Aggregated session metrics
 */
export interface SessionMetrics {
  avgEntropy: number;
  avgLocalMass: number;
  avgMidMass: number;
  avgLongMass: number;
  avgHubness?: number;
  avgConsensus?: number;
  dominantZone: ManifoldZone;
}

/**
 * TraceSession is the canonical in-memory representation of a chat trace
 * All UI views should read from this structure
 */
export interface TraceSession {
  id: string;
  model: string;
  createdAt: number;
  updatedAt: number;

  // Raw messages (for display)
  messages: Message[];

  // Tokenized view
  tokens: TokenEntry[];
  segments: Segment[];

  // Per-token decode data
  steps: DecodeStep[];

  // Session-level metrics
  metrics?: SessionMetrics;

  // Streaming state
  isStreaming: boolean;
  streamingTokenIndex: number;
}

/**
 * Create an empty trace session
 */
export function createTraceSession(model: string): TraceSession {
  return {
    id: `trace-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    model,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    tokens: [],
    segments: [],
    steps: [],
    isStreaming: false,
    streamingTokenIndex: -1,
  };
}

/**
 * Detect segment boundaries from content
 * Parses <think> tags to separate reasoning from final output
 */
export function detectSegments(
  content: string,
  messageId: string,
  startIndex: number,
  tokens: string[]
): Segment[] {
  const segments: Segment[] = [];

  // Simple <think> tag detection
  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);

  if (thinkMatch) {
    const thinkStart = content.indexOf('<think>');
    const thinkEnd = content.indexOf('</think>') + '</think>'.length;

    // Count tokens before think section
    let charCount = 0;
    const preThinkTokens: number[] = [];
    const thinkTokens: number[] = [];
    const postThinkTokens: number[] = [];

    for (let i = 0; i < tokens.length; i++) {
      const tokenLen = tokens[i].length;
      const tokenStart = charCount;
      const tokenEnd = charCount + tokenLen;

      if (tokenEnd <= thinkStart) {
        preThinkTokens.push(i);
      } else if (tokenStart >= thinkEnd) {
        postThinkTokens.push(i);
      } else {
        thinkTokens.push(i);
      }

      charCount += tokenLen;
    }

    // Create segments
    if (preThinkTokens.length > 0) {
      segments.push({
        id: `${messageId}-pre`,
        type: 'assistant_final',
        startTokenIndex: startIndex + preThinkTokens[0],
        endTokenIndex: startIndex + preThinkTokens[preThinkTokens.length - 1] + 1,
        messageId,
      });
    }

    if (thinkTokens.length > 0) {
      segments.push({
        id: `${messageId}-think`,
        type: 'assistant_think',
        startTokenIndex: startIndex + thinkTokens[0],
        endTokenIndex: startIndex + thinkTokens[thinkTokens.length - 1] + 1,
        messageId,
      });
    }

    if (postThinkTokens.length > 0) {
      segments.push({
        id: `${messageId}-final`,
        type: 'assistant_final',
        startTokenIndex: startIndex + postThinkTokens[0],
        endTokenIndex: startIndex + postThinkTokens[postThinkTokens.length - 1] + 1,
        messageId,
      });
    }
  } else {
    // No think tags - single segment
    segments.push({
      id: `${messageId}-main`,
      type: 'assistant_final',
      startTokenIndex: startIndex,
      endTokenIndex: startIndex + tokens.length,
      messageId,
    });
  }

  return segments;
}

/**
 * Compute session metrics from decode steps
 */
export function computeSessionMetrics(steps: DecodeStep[]): SessionMetrics | undefined {
  if (steps.length === 0) return undefined;

  let totalEntropy = 0;
  let totalLocal = 0;
  let totalMid = 0;
  let totalLong = 0;
  let count = 0;

  for (const step of steps) {
    if (step.fingerprint) {
      totalEntropy += step.fingerprint.entropy;
      totalLocal += step.fingerprint.local_mass;
      totalMid += step.fingerprint.mid_mass;
      totalLong += step.fingerprint.long_mass;
      count++;
    }
  }

  if (count === 0) return undefined;

  const avgLocal = totalLocal / count;
  const avgMid = totalMid / count;
  const avgLong = totalLong / count;

  // Determine dominant zone
  let dominantZone: ManifoldZone = 'diffuse';
  if (avgLocal > 0.5) dominantZone = 'syntax_floor';
  else if (avgMid > 0.5) dominantZone = 'semantic_bridge';
  else if (avgLong > 0.3) dominantZone = 'long_range';

  return {
    avgEntropy: totalEntropy / count,
    avgLocalMass: avgLocal,
    avgMidMass: avgMid,
    avgLongMass: avgLong,
    dominantZone,
  };
}

// ============================================================================
// FINGERPRINT PCA LOADINGS & INTERPRETATION
// ============================================================================

/**
 * Feature schema describing each dimension of the 20D fingerprint vector.
 * Order: [local_mass, mid_mass, long_mass, entropy, histogram[0:16]]
 */
export interface FingerprintFeature {
  name: string;
  description: string;
  range: [number, number];  // Expected value range
  interpretation: {
    low: string;    // What low values mean
    high: string;   // What high values mean
  };
}

export const FINGERPRINT_FEATURE_SCHEMA: FingerprintFeature[] = [
  {
    name: 'local_mass',
    description: 'Attention weight on tokens within 8 positions',
    range: [0, 1],
    interpretation: {
      low: 'Minimal local context reliance',
      high: 'Strong syntax/grammar processing (immediate neighbors)',
    },
  },
  {
    name: 'mid_mass',
    description: 'Attention weight on tokens 8-255 positions back',
    range: [0, 1],
    interpretation: {
      low: 'Minimal paragraph-level context',
      high: 'Strong semantic retrieval (sentence/paragraph context)',
    },
  },
  {
    name: 'long_mass',
    description: 'Attention weight on tokens 256+ positions back',
    range: [0, 1],
    interpretation: {
      low: 'Limited long-range dependencies',
      high: 'Active document-level reasoning',
    },
  },
  {
    name: 'entropy',
    description: 'Normalized entropy of attention distribution',
    range: [0, 1],
    interpretation: {
      low: 'Focused attention on few positions (high confidence)',
      high: 'Diffuse attention (exploratory/uncertain)',
    },
  },
  // Histogram bins (log2-spaced distances)
  ...Array.from({ length: 16 }, (_, i) => ({
    name: `hist_bin_${i}`,
    description: `Attention in distance range [${Math.pow(2, i)}, ${Math.pow(2, i + 1) - 1}]`,
    range: [0, 1] as [number, number],
    interpretation: {
      low: `Minimal attention at ${Math.pow(2, i)}-${Math.pow(2, i + 1) - 1} token distance`,
      high: `Strong attention at ${Math.pow(2, i)}-${Math.pow(2, i + 1) - 1} token distance`,
    },
  })),
];

/**
 * A principal component loading for fingerprint interpretation.
 * Each PC captures a common attention pattern.
 */
export interface PCALoading {
  name: string;
  description: string;
  variance_explained: number;  // 0-1, how much variance this PC explains
  loadings: number[];          // 20 weights, one per feature
  interpretation: {
    negative: string;  // What negative scores mean
    positive: string;  // What positive scores mean
  };
}

/**
 * Pre-computed PCA loadings for fingerprint interpretation.
 * These are derived from analyzing common attention patterns in transformer models.
 *
 * The loadings help answer: "What kind of attention pattern is this?"
 */
export const FINGERPRINT_PCA_LOADINGS: PCALoading[] = [
  {
    name: 'PC1: Local vs Long-Range',
    description: 'Contrasts immediate neighbor attention with distant retrieval',
    variance_explained: 0.35,
    loadings: [
      0.6,   // local_mass (positive = more local)
      0.1,   // mid_mass
      -0.6,  // long_mass (negative = less long-range)
      -0.2,  // entropy
      // Histogram: high loadings on nearby bins, negative on distant
      0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.1, -0.2,
      -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3,
    ],
    interpretation: {
      negative: 'Long-range information retrieval (document memory, prior context)',
      positive: 'Local syntax processing (grammar, immediate continuity)',
    },
  },
  {
    name: 'PC2: Focused vs Diffuse',
    description: 'Measures attention concentration vs exploration',
    variance_explained: 0.25,
    loadings: [
      0.1,   // local_mass
      0.1,   // mid_mass
      0.1,   // long_mass
      -0.8,  // entropy (negative = low entropy = focused)
      // Histogram: concentrated bins positive, spread negative
      0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    interpretation: {
      negative: 'Diffuse attention (uncertainty, exploration, multiple options)',
      positive: 'Focused attention (high confidence, clear next token)',
    },
  },
  {
    name: 'PC3: Semantic Bridge',
    description: 'Captures mid-range semantic retrieval patterns',
    variance_explained: 0.15,
    loadings: [
      -0.3,  // local_mass
      0.7,   // mid_mass (positive = semantic retrieval)
      -0.2,  // long_mass
      0.1,   // entropy
      // Histogram: peak in mid-range bins
      -0.1, -0.1, 0.0, 0.3, 0.4, 0.4, 0.3, 0.2,
      0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
    ],
    interpretation: {
      negative: 'Either very local or very long-range attention',
      positive: 'Semantic context retrieval (paragraph-level reasoning)',
    },
  },
  {
    name: 'PC4: Structure Ripple',
    description: 'Periodic attention patterns (code, lists, structured text)',
    variance_explained: 0.10,
    loadings: [
      0.0,   // local_mass
      0.0,   // mid_mass
      0.0,   // long_mass
      -0.2,  // entropy
      // Alternating pattern in histogram (periodic structure)
      0.3, -0.2, 0.3, -0.2, 0.3, -0.2, 0.2, -0.1,
      0.2, -0.1, 0.1, -0.1, 0.1, 0.0, 0.0, 0.0,
    ],
    interpretation: {
      negative: 'Smooth, continuous attention flow',
      positive: 'Periodic structure (code blocks, numbered lists, repetition)',
    },
  },
];

/**
 * Project a fingerprint onto PCA space.
 * Returns scores for each principal component.
 */
export function projectFingerprintToPCA(fp: Fingerprint): number[] {
  // Reconstruct 20D vector from Fingerprint
  const vector = [
    fp.local_mass,
    fp.mid_mass,
    fp.long_mass,
    fp.entropy,
    ...fp.histogram,
  ];

  // Project onto each PC
  return FINGERPRINT_PCA_LOADINGS.map((pc) => {
    let score = 0;
    for (let i = 0; i < Math.min(vector.length, pc.loadings.length); i++) {
      score += vector[i] * pc.loadings[i];
    }
    return score;
  });
}

/**
 * Explain a fingerprint in natural language.
 * Returns an array of interpretations based on PCA projections.
 */
export interface FingerprintExplanation {
  pcName: string;
  score: number;
  intensity: 'weak' | 'moderate' | 'strong';
  interpretation: string;
}

export function explainFingerprint(fp: Fingerprint): FingerprintExplanation[] {
  const scores = projectFingerprintToPCA(fp);
  const explanations: FingerprintExplanation[] = [];

  scores.forEach((score, i) => {
    const pc = FINGERPRINT_PCA_LOADINGS[i];
    const absScore = Math.abs(score);

    // Determine intensity
    let intensity: 'weak' | 'moderate' | 'strong';
    if (absScore < 0.2) intensity = 'weak';
    else if (absScore < 0.5) intensity = 'moderate';
    else intensity = 'strong';

    // Only include moderate or strong signals
    if (intensity !== 'weak') {
      explanations.push({
        pcName: pc.name,
        score,
        intensity,
        interpretation: score > 0 ? pc.interpretation.positive : pc.interpretation.negative,
      });
    }
  });

  // Sort by absolute score (most significant first)
  return explanations.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
}

/**
 * Get a one-line summary of the fingerprint's dominant pattern.
 */
export function summarizeFingerprint(fp: Fingerprint): string {
  const explanations = explainFingerprint(fp);

  if (explanations.length === 0) {
    return 'Balanced attention pattern (no dominant characteristic)';
  }

  const top = explanations[0];
  return `${top.intensity.charAt(0).toUpperCase() + top.intensity.slice(1)} ${top.interpretation.toLowerCase()}`;
}

// ============================================================================
// CROSS-RUN COMPARISON TYPES
// ============================================================================

/**
 * Metrics difference between two trace sessions.
 * Positive values mean right session has higher values.
 */
export interface MetricsDiff {
  entropyDiff: number;      // right - left
  localMassDiff: number;
  midMassDiff: number;
  longMassDiff: number;
  hubnessDiff?: number;
  consensusDiff?: number;
}

/**
 * Fingerprint comparison with per-component differences.
 */
export interface FingerprintDiff {
  histogramDiff: number[];  // Per-bin difference (right - left)
  pcaDiff: number[];        // PC score differences
  cosineSimilarity: number; // 0-1, how similar the fingerprints are
  euclideanDistance: number; // L2 distance between fingerprints
}

/**
 * Zone comparison between two sessions.
 */
export interface ZoneComparison {
  leftZone: ManifoldZone;
  rightZone: ManifoldZone;
  zoneChanged: boolean;
  leftConfidence: number;   // How clearly the left session falls in its zone
  rightConfidence: number;  // How clearly the right session falls in its zone
}

/**
 * Complete comparison result between two trace sessions.
 */
export interface ComparisonResult {
  // Session identifiers
  leftTraceId: string;
  rightTraceId: string;

  // Session-level diffs
  metricsDiff: MetricsDiff;

  // Fingerprint comparison
  fingerprintDiff: FingerprintDiff;

  // Zone transition analysis
  zoneComparison: ZoneComparison;

  // Summary
  overallSimilarity: number;  // 0-1, aggregate similarity score
  keyDifferences: string[];   // Human-readable difference descriptions

  // Timestamps
  computedAt: number;
}

/**
 * Summary info for session selection dropdown.
 */
export interface SessionSummary {
  id: string;
  model: string;
  createdAt: number;
  messageCount: number;
  tokenCount: number;
  dominantZone: ManifoldZone;
  avgEntropy: number;
  label?: string;  // Optional user-friendly label
}
