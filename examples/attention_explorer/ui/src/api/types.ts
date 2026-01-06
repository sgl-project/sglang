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
export type View = 'chat' | 'inspect' | 'manifold' | 'router';

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

  return null;
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
