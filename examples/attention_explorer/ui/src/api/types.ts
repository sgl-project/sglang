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
