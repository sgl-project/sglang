import { useMemo, useState, useCallback } from 'react';
import { useSessionStore } from '../../stores/useSessionStore';
import { useUIStore } from '../../stores/useUIStore';
import { Fingerprint, ManifoldZone, CaptureMode } from '../../api/types';

// ============================================================================
// TYPES
// ============================================================================

interface DetectedMode {
  zone: ManifoldZone;
  label: string;
  description: string;
  confidence: number;
  evidence: Evidence[];
}

interface Evidence {
  metric: string;
  value: number;
  threshold: number;
  passes: boolean;
  weight: number;
}

interface RecommendedControl {
  name: string;
  value: string | number;
  hint: string;
  impact: 'cost' | 'quality' | 'bandwidth' | 'throughput' | 'latency';
  direction: 'up' | 'down' | 'neutral';
}

interface SamplingConfig {
  temperature: number;
  top_p: number;
  top_k: number;
  capture_mode: CaptureMode;
  capture_layer_ids: number[];
  attention_stride: number;
}

// ============================================================================
// DETECTION LOGIC
// ============================================================================

const ZONE_CONFIGS: Record<ManifoldZone, {
  label: string;
  description: string;
  thresholds: { metric: keyof Fingerprint; min?: number; max?: number; weight: number }[];
}> = {
  syntax_floor: {
    label: 'Structured / JSON',
    description: 'High local attention → formatting, JSON repair, code syntax patterns.',
    thresholds: [
      { metric: 'local_mass', min: 0.45, weight: 0.4 },
      { metric: 'entropy', max: 2.5, weight: 0.3 },
      { metric: 'consensus', min: 0.6, weight: 0.3 },
    ],
  },
  semantic_bridge: {
    label: 'Reasoning / Retrieval',
    description: 'High mid-range attention → retrieval augmentation, semantic connections.',
    thresholds: [
      { metric: 'mid_mass', min: 0.4, weight: 0.4 },
      { metric: 'hubness', min: 0.3, weight: 0.3 },
      { metric: 'entropy', min: 2.0, max: 4.0, weight: 0.3 },
    ],
  },
  long_range: {
    label: 'Planning / Cross-Document',
    description: 'High long-range attention → document-level planning, cross-context reasoning.',
    thresholds: [
      { metric: 'long_mass', min: 0.35, weight: 0.5 },
      { metric: 'entropy', min: 3.0, weight: 0.3 },
      { metric: 'local_mass', max: 0.3, weight: 0.2 },
    ],
  },
  structure_ripple: {
    label: 'Periodic / Rhythmic',
    description: 'Strong FFT signal → periodic patterns, structured repetition.',
    thresholds: [
      { metric: 'fft_low', min: 0.5, weight: 0.5 },
      { metric: 'mid_mass', min: 0.25, weight: 0.25 },
      { metric: 'consensus', min: 0.4, weight: 0.25 },
    ],
  },
  diffuse: {
    label: 'Creative / Exploratory',
    description: 'Distributed attention → creative generation, open-ended exploration.',
    thresholds: [
      { metric: 'entropy', min: 3.5, weight: 0.4 },
      { metric: 'local_mass', max: 0.35, weight: 0.3 },
      { metric: 'consensus', max: 0.5, weight: 0.3 },
    ],
  },
  unknown: {
    label: 'Unknown',
    description: 'Insufficient data for classification.',
    thresholds: [],
  },
};

function detectMode(fingerprint: Fingerprint | null): DetectedMode {
  if (!fingerprint) {
    return {
      zone: 'unknown',
      label: 'Unknown',
      description: 'No fingerprint data available. Send a message to analyze.',
      confidence: 0,
      evidence: [],
    };
  }

  let bestZone: ManifoldZone = 'diffuse';
  let bestScore = 0;
  let bestEvidence: Evidence[] = [];

  for (const [zone, config] of Object.entries(ZONE_CONFIGS) as [ManifoldZone, typeof ZONE_CONFIGS[ManifoldZone]][]) {
    if (zone === 'unknown') continue;

    const evidence: Evidence[] = [];
    let totalWeight = 0;
    let weightedScore = 0;

    for (const threshold of config.thresholds) {
      const rawValue = fingerprint[threshold.metric];
      // Skip array values (histogram) - only compare numeric metrics
      if (Array.isArray(rawValue)) continue;
      const value = (rawValue as number) ?? 0;
      let passes = true;

      if (threshold.min !== undefined && value < threshold.min) passes = false;
      if (threshold.max !== undefined && value > threshold.max) passes = false;

      evidence.push({
        metric: threshold.metric,
        value,
        threshold: threshold.min ?? threshold.max ?? 0,
        passes,
        weight: threshold.weight,
      });

      totalWeight += threshold.weight;
      if (passes) weightedScore += threshold.weight;
    }

    const score = totalWeight > 0 ? weightedScore / totalWeight : 0;

    if (score > bestScore) {
      bestScore = score;
      bestZone = zone;
      bestEvidence = evidence;
    }
  }

  const config = ZONE_CONFIGS[bestZone];

  return {
    zone: bestZone,
    label: config.label,
    description: config.description,
    confidence: bestScore,
    evidence: bestEvidence,
  };
}

// ============================================================================
// RECOMMENDATIONS
// ============================================================================

function getRecommendations(mode: DetectedMode): {
  sampling: SamplingConfig;
  controls: RecommendedControl[];
} {
  // Base config
  const sampling: SamplingConfig = {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
    capture_mode: 'sketch',
    capture_layer_ids: [7, 15, 23, 31],
    attention_stride: 4,
  };

  const controls: RecommendedControl[] = [];

  switch (mode.zone) {
    case 'syntax_floor':
      sampling.temperature = 0.1;
      sampling.top_p = 0.95;
      sampling.top_k = 10;
      sampling.capture_mode = 'fingerprint';
      sampling.capture_layer_ids = [31, 39, 47]; // Later layers for syntax
      sampling.attention_stride = 1;

      controls.push(
        { name: 'temperature', value: 0.1, hint: 'Low temp for deterministic structured output', impact: 'quality', direction: 'up' },
        { name: 'top_k', value: 10, hint: 'Narrow sampling for consistent formatting', impact: 'quality', direction: 'up' },
        { name: 'capture_layer_ids', value: '[31, 39, 47]', hint: 'Focus on syntax-heavy later layers', impact: 'bandwidth', direction: 'down' },
        { name: 'attention_stride', value: 1, hint: 'Dense capture for format validation', impact: 'cost', direction: 'up' },
      );
      break;

    case 'semantic_bridge':
      sampling.temperature = 0.5;
      sampling.top_p = 0.9;
      sampling.top_k = 40;
      sampling.capture_mode = 'sketch';
      sampling.capture_layer_ids = [7, 15, 23, 31];
      sampling.attention_stride = 4;

      controls.push(
        { name: 'temperature', value: 0.5, hint: 'Moderate temp for semantic coherence', impact: 'quality', direction: 'neutral' },
        { name: 'capture_mode', value: 'sketch', hint: 'Histogram + anchors for retrieval analysis', impact: 'throughput', direction: 'up' },
        { name: 'capture_layer_ids', value: '[7, 15, 23, 31]', hint: 'Mid-layers for semantic connections', impact: 'bandwidth', direction: 'neutral' },
        { name: 'attention_stride', value: 4, hint: 'Sample every 4 tokens after warmup', impact: 'cost', direction: 'down' },
      );
      break;

    case 'long_range':
      sampling.temperature = 0.6;
      sampling.top_p = 0.85;
      sampling.top_k = 30;
      sampling.capture_mode = 'sketch';
      sampling.capture_layer_ids = [3, 7, 15, 23];
      sampling.attention_stride = 8;

      controls.push(
        { name: 'temperature', value: 0.6, hint: 'Slightly elevated for creative planning', impact: 'quality', direction: 'neutral' },
        { name: 'capture_layer_ids', value: '[3, 7, 15, 23]', hint: 'Earlier layers capture long-range patterns', impact: 'bandwidth', direction: 'neutral' },
        { name: 'attention_stride', value: 8, hint: 'Sparse capture for long documents', impact: 'cost', direction: 'down' },
        { name: 'top_p', value: 0.85, hint: 'Tighter nucleus for coherent planning', impact: 'quality', direction: 'up' },
      );
      break;

    case 'structure_ripple':
      sampling.temperature = 0.3;
      sampling.top_p = 0.9;
      sampling.top_k = 20;
      sampling.capture_mode = 'fingerprint';
      sampling.capture_layer_ids = [15, 23, 31];
      sampling.attention_stride = 2;

      controls.push(
        { name: 'temperature', value: 0.3, hint: 'Low temp preserves periodic structure', impact: 'quality', direction: 'up' },
        { name: 'capture_mode', value: 'fingerprint', hint: 'FFT analysis for rhythm detection', impact: 'throughput', direction: 'up' },
        { name: 'attention_stride', value: 2, hint: 'Dense capture to track periodicity', impact: 'cost', direction: 'up' },
      );
      break;

    case 'diffuse':
    default:
      sampling.temperature = 0.9;
      sampling.top_p = 0.95;
      sampling.top_k = 100;
      sampling.capture_mode = 'sketch';
      sampling.capture_layer_ids = [7, 23, 39];
      sampling.attention_stride = 8;

      controls.push(
        { name: 'temperature', value: 0.9, hint: 'High temp for creative exploration', impact: 'quality', direction: 'neutral' },
        { name: 'top_k', value: 100, hint: 'Wide sampling for diverse outputs', impact: 'quality', direction: 'neutral' },
        { name: 'capture_mode', value: 'sketch', hint: 'Lightweight capture for exploration', impact: 'throughput', direction: 'up' },
        { name: 'attention_stride', value: 8, hint: 'Sparse sampling reduces overhead', impact: 'cost', direction: 'down' },
      );
      break;
  }

  return { sampling, controls };
}

// ============================================================================
// COMPONENTS
// ============================================================================

interface EvidenceMeterProps {
  evidence: Evidence;
}

function EvidenceMeter({ evidence }: EvidenceMeterProps) {
  const percentage = Math.min(100, Math.max(0, (evidence.value / (evidence.threshold || 1)) * 100));
  const displayPercentage = Math.min(percentage, 150); // Cap visual at 150%

  return (
    <div className={`evidence-meter ${evidence.passes ? 'passes' : 'fails'}`}>
      <div className="evidence-header">
        <span className="evidence-metric">{evidence.metric.replace('_', ' ')}</span>
        <span className="evidence-value">{evidence.value.toFixed(2)}</span>
      </div>
      <div className="evidence-bar">
        <div
          className="evidence-fill"
          style={{ width: `${Math.min(displayPercentage, 100)}%` }}
        />
        <div
          className="evidence-threshold"
          style={{ left: `${Math.min(100, (evidence.threshold / Math.max(evidence.value, evidence.threshold)) * 100)}%` }}
        />
      </div>
      <div className="evidence-footer">
        <span className="evidence-threshold-label">
          threshold: {evidence.threshold.toFixed(2)}
        </span>
        <span className={`evidence-status ${evidence.passes ? 'pass' : 'fail'}`}>
          {evidence.passes ? 'PASS' : 'FAIL'}
        </span>
      </div>
    </div>
  );
}

interface ControlRowProps {
  control: RecommendedControl;
  onApply?: () => void;
}

function ControlRow({ control }: ControlRowProps) {
  const impactIcon = {
    cost: '$',
    quality: 'Q',
    bandwidth: 'BW',
    throughput: 'TP',
    latency: 'L',
  }[control.impact];

  const directionSymbol = control.direction === 'up' ? '↑' : control.direction === 'down' ? '↓' : '~';

  return (
    <div className="control-row">
      <div className="control-left">
        <div className="control-name">
          <strong>{control.name}</strong>: {control.value}
        </div>
        <div className="control-hint">{control.hint}</div>
      </div>
      <div className={`control-right ${control.direction}`}>
        {impactIcon}: {directionSymbol}
      </div>
    </div>
  );
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export function RouterView() {
  const fingerprint = useSessionStore((state) => state.fingerprint);
  const program = useUIStore((state) => state.program);
  const setProgram = useUIStore((state) => state.setProgram);

  const [applied, setApplied] = useState(false);
  const [appliedConfig, setAppliedConfig] = useState<SamplingConfig | null>(null);

  // Detect mode from fingerprint
  const detectedMode = useMemo(() => detectMode(fingerprint), [fingerprint]);

  // Get recommendations based on detected mode
  const { sampling, controls } = useMemo(
    () => getRecommendations(detectedMode),
    [detectedMode]
  );

  // Handle apply
  const handleApply = useCallback(() => {
    setAppliedConfig(sampling);
    setApplied(true);

    // Store config for next request (could be expanded to actually send to server)
    localStorage.setItem('router-sampling-config', JSON.stringify(sampling));

    // Switch to appropriate program mode based on detection
    if (detectedMode.zone === 'syntax_floor') {
      setProgram('prod'); // Structured output benefits from prod mode
    } else if (detectedMode.zone === 'diffuse') {
      setProgram('discovery'); // Creative benefits from discovery
    }

    // Reset after 3 seconds
    setTimeout(() => setApplied(false), 3000);
  }, [sampling, detectedMode.zone, setProgram]);

  // Confidence display
  const confidencePercent = Math.round(detectedMode.confidence * 100);
  const confidenceClass =
    confidencePercent >= 70 ? 'high' : confidencePercent >= 40 ? 'medium' : 'low';

  return (
    <div className="card router-view">
      <div className="card-header">
        <div className="card-title">
          <span>Discovery Router</span>
          <span className="subtitle">
            Dynamic zone detection and adaptive sampling recommendations.
          </span>
        </div>
        <div className="badges">
          <span className="badge">sidecar feedback</span>
          <span className="badge">adaptive capture</span>
          <span className={`badge strong program-${program}`}>{program}</span>
        </div>
      </div>

      <div className="card-content router-layout">
        {/* Left Column: Detection */}
        <div className="router-detection">
          {/* Detected Mode */}
          <div className="section detected-mode-section">
            <div className="section-header">
              <span>Detected Mode</span>
              <span className={`badge strong zone-badge zone-${detectedMode.zone}`}>
                {detectedMode.label}
              </span>
            </div>

            <div className="mode-description">{detectedMode.description}</div>

            {/* Confidence Meter */}
            <div className="confidence-section">
              <div className="confidence-header">
                <span>Confidence</span>
                <span className={`confidence-value ${confidenceClass}`}>
                  {confidencePercent}%
                </span>
              </div>
              <div className="confidence-bar">
                <div
                  className={`confidence-fill ${confidenceClass}`}
                  style={{ width: `${confidencePercent}%` }}
                />
              </div>
            </div>
          </div>

          {/* Evidence */}
          {detectedMode.evidence.length > 0 && (
            <div className="section evidence-section">
              <div className="section-header">
                <span>Evidence</span>
                <span className="badge">
                  {detectedMode.evidence.filter((e) => e.passes).length}/
                  {detectedMode.evidence.length} passing
                </span>
              </div>
              <div className="evidence-grid">
                {detectedMode.evidence.map((ev, idx) => (
                  <EvidenceMeter key={idx} evidence={ev} />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Recommendations */}
        <div className="router-recommendations">
          {/* Recommended Controls */}
          <div className="section">
            <div className="section-header">
              <span>Recommended Controls</span>
              <span className="badge">zone-optimized</span>
            </div>
            <div className="control-list">
              {controls.map((ctrl, idx) => (
                <ControlRow key={idx} control={ctrl} />
              ))}
            </div>
          </div>

          {/* Apply Button */}
          <div className="section apply-section">
            <button
              className={`apply-btn ${applied ? 'applied' : ''}`}
              onClick={handleApply}
              disabled={detectedMode.zone === 'unknown'}
            >
              {applied ? 'Applied!' : 'Apply to Next Run'}
            </button>

            {appliedConfig && (
              <div className="applied-summary">
                <div className="applied-label">Applied Config:</div>
                <div className="applied-values">
                  <span>temp={appliedConfig.temperature}</span>
                  <span>top_p={appliedConfig.top_p}</span>
                  <span>stride={appliedConfig.attention_stride}</span>
                </div>
              </div>
            )}

            <div className="hint-text">
              Applies recommended sampling parameters to the next chat request.
              Config is stored locally and used by the attention stream hook.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
