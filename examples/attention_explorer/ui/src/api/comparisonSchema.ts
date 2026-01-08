/**
 * Quantization Comparison Schema v1 - TypeScript Types
 *
 * Canonical types for comparing quantized model behavior against baseline.
 * Matches the JSON schema in schemas/quantization_comparison_v1.json
 */

// =============================================================================
// Enums
// =============================================================================

export type QuantizationMethod =
  | 'none'
  | 'sinq'
  | 'asinq'
  | 'awq'
  | 'gptq'
  | 'squeezellm'
  | 'fp8'
  | 'marlin';

export type TilingMode = '1D' | '2D';

export type ManifoldZone =
  | 'syntax_floor'
  | 'semantic_bridge'
  | 'structure_ripple'
  | 'long_range'
  | 'diffuse'
  | 'unknown';

export type QualityTier =
  | 'excellent'   // Jaccard >= 0.8
  | 'good'        // Jaccard >= 0.6
  | 'acceptable'  // Jaccard >= 0.4
  | 'degraded'    // Jaccard >= 0.2
  | 'failed';     // Jaccard < 0.2

export type PromptPack =
  | 'json_repair'
  | 'coreference'
  | 'counting_tables'
  | 'code_editing'
  | 'reasoning'
  | 'adversarial'
  | 'natural'
  | 'custom';

export type AttentionAggregation = 'mean' | 'max' | 'last_layer' | 'weighted';
export type PromptSource = 'inline' | 'file' | 'harness';
export type FingerprintLocation = 'server' | 'client';

// =============================================================================
// Configuration Types
// =============================================================================

export interface CalibrationConfig {
  dataset: string;
  num_samples: number;
  seq_length?: number;
}

export interface QuantizationConfig {
  method: QuantizationMethod;
  nbits?: number;
  group_size?: number;
  tiling_mode?: TilingMode;
  symmetric?: boolean;
  calibration?: CalibrationConfig;
  kernel_path?: string;
}

export interface ModelConfig {
  model_id: string;
  dtype: string;
  revision?: string;
  quantization?: QuantizationConfig;
  memory_mb?: number;
}

export interface DecodingConfig {
  max_tokens: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
  repetition_penalty?: number;
}

export interface SinkFilterConfig {
  enabled: boolean;
  indices: number[];
}

export interface AttentionCaptureConfig {
  top_k: number;
  layers?: number[];
  heads?: number[];
  aggregation?: AttentionAggregation;
  sink_filter?: SinkFilterConfig;
}

export interface FingerprintConfig {
  enabled: boolean;
  compute_location: FingerprintLocation;
  include_manifold_zone?: boolean;
}

export interface PrivacyMaskingConfig {
  enabled: boolean;
  token_threshold?: number;
}

export interface HarnessConfig {
  packs: PromptPack[];
  duration_minutes?: number;
}

export interface PromptsConfig {
  source: PromptSource;
  path?: string;
  harness_config?: HarnessConfig;
  count: number;
}

export interface EvaluationConfig {
  prompts: PromptsConfig;
  decoding: DecodingConfig;
  attention_capture: AttentionCaptureConfig;
  fingerprint?: FingerprintConfig;
  privacy_masking?: PrivacyMaskingConfig;
}

export interface HardwareConfig {
  gpu_type?: string;
  gpu_count?: number;
  gpu_memory_gb?: number;
  cuda_version?: string;
  driver_version?: string;
  attention_backend?: string;
  tensor_parallel?: number;
  pipeline_parallel?: number;
}

// =============================================================================
// Metrics Types
// =============================================================================

export interface JaccardMetrics {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  p5: number;
  p95: number;
}

export interface WeightedJaccardMetrics {
  mean: number;
  std: number;
}

export interface RankCorrelationMetrics {
  spearman_mean?: number;
  kendall_mean?: number;
}

export interface MassRetainedMetrics {
  mean: number;
  std: number;
}

export interface KLDivergenceMetrics {
  mean: number;
  std: number;
}

export interface OutputAgreementMetrics {
  exact_match_rate: number;
  token_accuracy?: number;
  edit_distance_mean?: number;
}

export interface SummaryMetrics {
  jaccard: JaccardMetrics;
  weighted_jaccard?: WeightedJaccardMetrics;
  rank_correlation?: RankCorrelationMetrics;
  mass_retained?: MassRetainedMetrics;
  kl_divergence?: KLDivergenceMetrics;
  output_agreement?: OutputAgreementMetrics;
  compression_ratio?: number;
  quality_tier: QualityTier;
}

export interface PromptRankCorrelation {
  spearman?: number;
  kendall?: number;
}

export interface PromptResult {
  prompt_id: string;
  prompt_text?: string;
  prompt_pack?: PromptPack;
  jaccard: number;
  weighted_jaccard?: number;
  rank_correlation?: PromptRankCorrelation;
  mass_retained?: number;
  kl_divergence?: number;
  output_match?: boolean;
  edit_distance?: number;
  baseline_zone?: ManifoldZone;
  candidate_zone?: ManifoldZone;
  zone_drift?: boolean;
  per_step_jaccard?: number[];
  per_layer_jaccard?: number[];
}

export interface ZoneTransitionMatrix {
  [baselineZone: string]: {
    [candidateZone: string]: number;
  };
}

export interface ClusterStability {
  baseline_cluster_count: number;
  candidate_cluster_count: number;
  cluster_purity: number;
  adjusted_rand_index?: number;
}

export interface FingerprintDistance {
  cosine_mean: number;
  euclidean_mean?: number;
}

export interface ManifoldAnalysis {
  zone_drift_rate: number;
  zone_transition_matrix: ZoneTransitionMatrix;
  drift_by_pack?: { [pack: string]: number };
  cluster_stability?: ClusterStability;
  fingerprint_distance?: FingerprintDistance;
}

export interface PerformanceMetrics {
  baseline_throughput_tps?: number;
  candidate_throughput_tps?: number;
  throughput_ratio?: number;
  baseline_latency_p50_ms?: number;
  candidate_latency_p50_ms?: number;
  baseline_latency_p99_ms?: number;
  candidate_latency_p99_ms?: number;
}

export interface ComparisonResults {
  summary: SummaryMetrics;
  per_prompt: PromptResult[];
  manifold_analysis?: ManifoldAnalysis;
  performance?: PerformanceMetrics;
}

export interface ComparisonMetadata {
  run_by?: string;
  purpose?: string;
  notes?: string;
  tags?: string[];
}

// =============================================================================
// Main Schema Type
// =============================================================================

export interface QuantizationComparisonV1 {
  schema_version: '1.0.0';
  comparison_id: string;
  timestamp: string;
  baseline: ModelConfig;
  candidate: ModelConfig;
  evaluation: EvaluationConfig;
  hardware?: HardwareConfig;
  results: ComparisonResults;
  metadata?: ComparisonMetadata;
}

// =============================================================================
// Helper Functions
// =============================================================================

export const SCHEMA_VERSION = '1.0.0';

export function classifyQualityTier(meanJaccard: number): QualityTier {
  if (meanJaccard >= 0.8) return 'excellent';
  if (meanJaccard >= 0.6) return 'good';
  if (meanJaccard >= 0.4) return 'acceptable';
  if (meanJaccard >= 0.2) return 'degraded';
  return 'failed';
}

export function getQualityTierColor(tier: QualityTier): string {
  switch (tier) {
    case 'excellent': return '#22c55e'; // green-500
    case 'good': return '#84cc16';      // lime-500
    case 'acceptable': return '#eab308'; // yellow-500
    case 'degraded': return '#f97316';   // orange-500
    case 'failed': return '#ef4444';     // red-500
  }
}

export function getQualityTierLabel(tier: QualityTier): string {
  switch (tier) {
    case 'excellent': return 'Excellent (≥80%)';
    case 'good': return 'Good (≥60%)';
    case 'acceptable': return 'Acceptable (≥40%)';
    case 'degraded': return 'Degraded (≥20%)';
    case 'failed': return 'Failed (<20%)';
  }
}

export function formatQuantConfig(config: QuantizationConfig): string {
  if (config.method === 'none') return 'None';

  const parts = [config.method.toUpperCase()];
  if (config.nbits) parts.push(`${config.nbits}b`);
  if (config.group_size) parts.push(`g${config.group_size}`);
  if (config.tiling_mode) parts.push(config.tiling_mode);

  return parts.join(' ');
}

export function formatModelConfig(config: ModelConfig): string {
  const modelName = config.model_id.split('/').pop() || config.model_id;
  const dtype = config.dtype.toUpperCase();

  if (config.quantization && config.quantization.method !== 'none') {
    return `${modelName} (${formatQuantConfig(config.quantization)})`;
  }

  return `${modelName} (${dtype})`;
}

/**
 * Validate a comparison report against the schema
 */
export function validateComparisonReport(
  report: unknown
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!report || typeof report !== 'object') {
    return { valid: false, errors: ['Report must be an object'] };
  }

  const r = report as Record<string, unknown>;

  // Required fields
  const required = [
    'schema_version',
    'comparison_id',
    'timestamp',
    'baseline',
    'candidate',
    'evaluation',
    'results',
  ];

  for (const field of required) {
    if (!(field in r)) {
      errors.push(`Missing required field: ${field}`);
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors };
  }

  // Schema version
  if (r.schema_version !== SCHEMA_VERSION) {
    errors.push(`Unsupported schema version: ${r.schema_version}`);
  }

  // Model configs
  for (const configName of ['baseline', 'candidate'] as const) {
    const config = r[configName] as Record<string, unknown> | undefined;
    if (!config?.model_id) {
      errors.push(`${configName} missing model_id`);
    }
    if (!config?.dtype) {
      errors.push(`${configName} missing dtype`);
    }
  }

  // Evaluation
  const evaluation = r.evaluation as Record<string, unknown> | undefined;
  if (!evaluation?.decoding) {
    errors.push('evaluation missing decoding config');
  }
  if (!evaluation?.attention_capture) {
    errors.push('evaluation missing attention_capture config');
  }

  // Results
  const results = r.results as Record<string, unknown> | undefined;
  if (!results?.summary) {
    errors.push('results missing summary');
  }
  if (!results?.per_prompt || !Array.isArray(results.per_prompt)) {
    errors.push('results missing per_prompt array');
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Compute summary statistics from per-prompt results
 */
export function computeSummaryFromPrompts(
  prompts: PromptResult[]
): Partial<SummaryMetrics> {
  if (prompts.length === 0) {
    return {};
  }

  const jaccards = prompts.map((p) => p.jaccard);
  jaccards.sort((a, b) => a - b);

  const mean = jaccards.reduce((a, b) => a + b, 0) / jaccards.length;
  const variance =
    jaccards.reduce((sum, j) => sum + (j - mean) ** 2, 0) / jaccards.length;
  const std = Math.sqrt(variance);

  const percentile = (arr: number[], p: number) => {
    const idx = Math.floor((arr.length - 1) * p);
    return arr[idx];
  };

  return {
    jaccard: {
      mean,
      std,
      min: jaccards[0],
      max: jaccards[jaccards.length - 1],
      median: percentile(jaccards, 0.5),
      p5: percentile(jaccards, 0.05),
      p95: percentile(jaccards, 0.95),
    },
    quality_tier: classifyQualityTier(mean),
  };
}

/**
 * Create a Sankey-ready data structure from zone transition matrix
 */
export function zoneTransitionToSankey(
  matrix: ZoneTransitionMatrix
): { source: string; target: string; value: number }[] {
  const links: { source: string; target: string; value: number }[] = [];

  for (const [source, targets] of Object.entries(matrix)) {
    for (const [target, value] of Object.entries(targets)) {
      if (value > 0) {
        links.push({
          source: `baseline_${source}`,
          target: `candidate_${target}`,
          value,
        });
      }
    }
  }

  return links;
}

// =============================================================================
// Blessed Config Registry Types
// =============================================================================

export interface BlessedQuantConfig {
  config_id: string;
  model_id: string;
  quantization: QuantizationConfig;
  quality_tier: QualityTier;
  mean_jaccard: number;
  zone_drift_rate: number;
  compression_ratio: number;
  approved_for_packs: PromptPack[];
  excluded_packs: PromptPack[];
  comparison_id: string;
  approved_timestamp: string;
  notes?: string;
}

export interface BlessedConfigRegistry {
  configs: BlessedQuantConfig[];
}

/**
 * Find the best quantization config for a model and prompt pack
 */
export function findBestConfigForPack(
  registry: BlessedConfigRegistry,
  modelId: string,
  pack: PromptPack,
  minQuality: QualityTier = 'acceptable'
): BlessedQuantConfig | null {
  const qualityOrder: QualityTier[] = [
    'failed',
    'degraded',
    'acceptable',
    'good',
    'excellent',
  ];
  const minQualityIdx = qualityOrder.indexOf(minQuality);

  const candidates = registry.configs.filter(
    (c) =>
      c.model_id === modelId &&
      c.approved_for_packs.includes(pack) &&
      !c.excluded_packs.includes(pack) &&
      qualityOrder.indexOf(c.quality_tier) >= minQualityIdx
  );

  if (candidates.length === 0) {
    return null;
  }

  // Sort by compression ratio (desc), then quality (desc)
  candidates.sort((a, b) => {
    if (b.compression_ratio !== a.compression_ratio) {
      return b.compression_ratio - a.compression_ratio;
    }
    return b.mean_jaccard - a.mean_jaccard;
  });

  return candidates[0];
}
