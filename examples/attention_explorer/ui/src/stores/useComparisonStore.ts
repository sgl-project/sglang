import { create } from 'zustand';
import {
  TraceSession,
  ComparisonResult,
  MetricsDiff,
  FingerprintDiff,
  ZoneComparison,
  SessionSummary,
  Fingerprint,
  ManifoldZone,
  projectFingerprintToPCA,
} from '../api/types';
import { useTraceStore } from './useTraceStore';

interface ComparisonState {
  // Selected session IDs for comparison
  leftTraceId: string | null;
  rightTraceId: string | null;

  // Computed comparison result
  comparison: ComparisonResult | null;

  // Loading/error state
  isComputing: boolean;
  error: string | null;

  // Actions
  selectLeft: (traceId: string | null) => void;
  selectRight: (traceId: string | null) => void;
  swapSessions: () => void;
  clearComparison: () => void;
  computeComparison: () => void;

  // Helpers
  getAvailableSessions: () => SessionSummary[];
  getLeftSession: () => TraceSession | null;
  getRightSession: () => TraceSession | null;
}

/**
 * Compute aggregate fingerprint from decode steps.
 */
function computeAggregateFingerprint(trace: TraceSession): Fingerprint | null {
  const fingerprints = trace.steps
    .map((s) => s.fingerprint)
    .filter((fp): fp is Fingerprint => fp != null);

  if (fingerprints.length === 0) return null;

  // Average all fingerprint components
  const histogram = new Array(16).fill(0);
  let localMass = 0;
  let midMass = 0;
  let longMass = 0;
  let entropy = 0;

  for (const fp of fingerprints) {
    localMass += fp.local_mass;
    midMass += fp.mid_mass;
    longMass += fp.long_mass;
    entropy += fp.entropy;
    for (let i = 0; i < 16 && i < fp.histogram.length; i++) {
      histogram[i] += fp.histogram[i];
    }
  }

  const n = fingerprints.length;
  return {
    histogram: histogram.map((h) => h / n),
    local_mass: localMass / n,
    mid_mass: midMass / n,
    long_mass: longMass / n,
    entropy: entropy / n,
  };
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Compute Euclidean distance between two vectors.
 */
function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Convert fingerprint to 20D vector for similarity computation.
 */
function fingerprintToVector(fp: Fingerprint): number[] {
  return [
    fp.local_mass,
    fp.mid_mass,
    fp.long_mass,
    fp.entropy,
    ...fp.histogram.slice(0, 16),
  ];
}

/**
 * Compute zone confidence based on how clearly it falls into that zone.
 */
function computeZoneConfidence(fp: Fingerprint, zone: ManifoldZone): number {
  switch (zone) {
    case 'syntax_floor':
      return fp.local_mass;
    case 'semantic_bridge':
      return fp.mid_mass;
    case 'long_range':
      return fp.long_mass;
    case 'diffuse':
      // High entropy, no dominant mass region
      const maxMass = Math.max(fp.local_mass, fp.mid_mass, fp.long_mass);
      return (1 - maxMass) * (fp.entropy / 5); // Normalize entropy
    case 'structure_ripple':
      return fp.fft_low ?? 0;
    default:
      return 0;
  }
}

/**
 * Generate human-readable difference descriptions.
 */
function generateKeyDifferences(
  metricsDiff: MetricsDiff,
  zoneComparison: ZoneComparison,
  fingerprintDiff: FingerprintDiff
): string[] {
  const differences: string[] = [];

  // Zone change
  if (zoneComparison.zoneChanged) {
    differences.push(
      `Zone shift: ${zoneComparison.leftZone} → ${zoneComparison.rightZone}`
    );
  }

  // Entropy change
  if (Math.abs(metricsDiff.entropyDiff) > 0.1) {
    const pct = Math.round(
      (metricsDiff.entropyDiff / Math.max(0.01, 1 - Math.abs(metricsDiff.entropyDiff))) * 100
    );
    const direction = metricsDiff.entropyDiff > 0 ? 'higher' : 'lower';
    differences.push(`Session B has ${Math.abs(pct)}% ${direction} entropy`);
  }

  // Local mass change
  if (Math.abs(metricsDiff.localMassDiff) > 0.1) {
    if (metricsDiff.localMassDiff > 0) {
      differences.push('Session B focused more on local context');
    } else {
      differences.push('Session A focused more on local context');
    }
  }

  // Long-range change
  if (Math.abs(metricsDiff.longMassDiff) > 0.1) {
    if (metricsDiff.longMassDiff > 0) {
      differences.push('Session B has stronger long-range dependencies');
    } else {
      differences.push('Session A has stronger long-range dependencies');
    }
  }

  // Mid-range (semantic) change
  if (Math.abs(metricsDiff.midMassDiff) > 0.15) {
    if (metricsDiff.midMassDiff > 0) {
      differences.push('Session B has more semantic retrieval activity');
    } else {
      differences.push('Session A has more semantic retrieval activity');
    }
  }

  // Overall similarity note
  if (fingerprintDiff.cosineSimilarity > 0.9) {
    differences.push('Attention patterns are highly similar overall');
  } else if (fingerprintDiff.cosineSimilarity < 0.5) {
    differences.push('Attention patterns are significantly different');
  }

  return differences;
}

/**
 * Perform full comparison between two trace sessions.
 */
function compareTraces(
  left: TraceSession,
  right: TraceSession
): ComparisonResult {
  // Get metrics
  const leftMetrics = left.metrics;
  const rightMetrics = right.metrics;

  // Compute metrics diff
  const metricsDiff: MetricsDiff = {
    entropyDiff: (rightMetrics?.avgEntropy ?? 0) - (leftMetrics?.avgEntropy ?? 0),
    localMassDiff: (rightMetrics?.avgLocalMass ?? 0) - (leftMetrics?.avgLocalMass ?? 0),
    midMassDiff: (rightMetrics?.avgMidMass ?? 0) - (leftMetrics?.avgMidMass ?? 0),
    longMassDiff: (rightMetrics?.avgLongMass ?? 0) - (leftMetrics?.avgLongMass ?? 0),
    hubnessDiff:
      leftMetrics?.avgHubness != null && rightMetrics?.avgHubness != null
        ? rightMetrics.avgHubness - leftMetrics.avgHubness
        : undefined,
    consensusDiff:
      leftMetrics?.avgConsensus != null && rightMetrics?.avgConsensus != null
        ? rightMetrics.avgConsensus - leftMetrics.avgConsensus
        : undefined,
  };

  // Compute aggregate fingerprints
  const leftFp = computeAggregateFingerprint(left);
  const rightFp = computeAggregateFingerprint(right);

  let fingerprintDiff: FingerprintDiff;

  if (leftFp && rightFp) {
    const leftVec = fingerprintToVector(leftFp);
    const rightVec = fingerprintToVector(rightFp);

    const leftPCA = projectFingerprintToPCA(leftFp);
    const rightPCA = projectFingerprintToPCA(rightFp);

    fingerprintDiff = {
      histogramDiff: leftFp.histogram.map(
        (h, i) => (rightFp.histogram[i] ?? 0) - h
      ),
      pcaDiff: leftPCA.map((p, i) => (rightPCA[i] ?? 0) - p),
      cosineSimilarity: cosineSimilarity(leftVec, rightVec),
      euclideanDistance: euclideanDistance(leftVec, rightVec),
    };
  } else {
    fingerprintDiff = {
      histogramDiff: new Array(16).fill(0),
      pcaDiff: new Array(4).fill(0),
      cosineSimilarity: 0,
      euclideanDistance: 0,
    };
  }

  // Zone comparison
  const leftZone = leftMetrics?.dominantZone ?? 'unknown';
  const rightZone = rightMetrics?.dominantZone ?? 'unknown';

  const zoneComparison: ZoneComparison = {
    leftZone,
    rightZone,
    zoneChanged: leftZone !== rightZone,
    leftConfidence: leftFp ? computeZoneConfidence(leftFp, leftZone) : 0,
    rightConfidence: rightFp ? computeZoneConfidence(rightFp, rightZone) : 0,
  };

  // Generate key differences
  const keyDifferences = generateKeyDifferences(
    metricsDiff,
    zoneComparison,
    fingerprintDiff
  );

  // Overall similarity (weighted average of factors)
  const overallSimilarity =
    fingerprintDiff.cosineSimilarity * 0.5 +
    (zoneComparison.zoneChanged ? 0 : 0.3) +
    (1 - Math.min(1, Math.abs(metricsDiff.entropyDiff) * 2)) * 0.2;

  return {
    leftTraceId: left.id,
    rightTraceId: right.id,
    metricsDiff,
    fingerprintDiff,
    zoneComparison,
    overallSimilarity: Math.max(0, Math.min(1, overallSimilarity)),
    keyDifferences,
    computedAt: Date.now(),
  };
}

export const useComparisonStore = create<ComparisonState>((set, get) => {
  // Expose store for E2E testing/debugging
  if (typeof window !== 'undefined') {
    (window as any).__COMPARISON_STORE__ = { get };
  }

  return {
    leftTraceId: null,
    rightTraceId: null,
    comparison: null,
    isComputing: false,
    error: null,

    selectLeft: (traceId) => {
      set({ leftTraceId: traceId, comparison: null, error: null });
      // Auto-compute if both sessions are selected
      if (traceId && get().rightTraceId) {
        get().computeComparison();
      }
    },

    selectRight: (traceId) => {
      set({ rightTraceId: traceId, comparison: null, error: null });
      // Auto-compute if both sessions are selected
      if (traceId && get().leftTraceId) {
        get().computeComparison();
      }
    },

    swapSessions: () => {
      const { leftTraceId, rightTraceId } = get();
      set({
        leftTraceId: rightTraceId,
        rightTraceId: leftTraceId,
        comparison: null,
      });
      // Recompute after swap
      if (leftTraceId && rightTraceId) {
        get().computeComparison();
      }
    },

    clearComparison: () => {
      set({
        leftTraceId: null,
        rightTraceId: null,
        comparison: null,
        error: null,
      });
    },

    computeComparison: () => {
      const { leftTraceId, rightTraceId } = get();

      if (!leftTraceId || !rightTraceId) {
        set({ error: 'Both sessions must be selected', comparison: null });
        return;
      }

      if (leftTraceId === rightTraceId) {
        set({ error: 'Cannot compare a session with itself', comparison: null });
        return;
      }

      set({ isComputing: true, error: null });

      try {
        const left = get().getLeftSession();
        const right = get().getRightSession();

        if (!left) {
          set({
            error: `Left session not found: ${leftTraceId}`,
            isComputing: false,
          });
          return;
        }

        if (!right) {
          set({
            error: `Right session not found: ${rightTraceId}`,
            isComputing: false,
          });
          return;
        }

        const comparison = compareTraces(left, right);
        set({ comparison, isComputing: false });
      } catch (err) {
        set({
          error: err instanceof Error ? err.message : 'Comparison failed',
          isComputing: false,
        });
      }
    },

    getAvailableSessions: () => {
      const traceStore = useTraceStore.getState();
      const sessions: SessionSummary[] = [];

      // Add current trace if it exists and has data
      if (
        traceStore.currentTrace &&
        traceStore.currentTrace.steps.length > 0
      ) {
        const trace = traceStore.currentTrace;
        sessions.push({
          id: trace.id,
          model: trace.model,
          createdAt: trace.createdAt,
          messageCount: trace.messages.length,
          tokenCount: trace.tokens.length,
          dominantZone: trace.metrics?.dominantZone ?? 'unknown',
          avgEntropy: trace.metrics?.avgEntropy ?? 0,
          label: 'Current Session',
        });
      }

      // Add saved traces
      for (const trace of traceStore.savedTraces) {
        sessions.push({
          id: trace.id,
          model: trace.model,
          createdAt: trace.createdAt,
          messageCount: trace.messages.length,
          tokenCount: trace.tokens.length,
          dominantZone: trace.metrics?.dominantZone ?? 'unknown',
          avgEntropy: trace.metrics?.avgEntropy ?? 0,
        });
      }

      // Sort by creation date (newest first)
      return sessions.sort((a, b) => b.createdAt - a.createdAt);
    },

    getLeftSession: () => {
      const { leftTraceId } = get();
      if (!leftTraceId) return null;

      const traceStore = useTraceStore.getState();

      // Check current trace
      if (traceStore.currentTrace?.id === leftTraceId) {
        return traceStore.currentTrace;
      }

      // Check saved traces
      return traceStore.savedTraces.find((t) => t.id === leftTraceId) ?? null;
    },

    getRightSession: () => {
      const { rightTraceId } = get();
      if (!rightTraceId) return null;

      const traceStore = useTraceStore.getState();

      // Check current trace
      if (traceStore.currentTrace?.id === rightTraceId) {
        return traceStore.currentTrace;
      }

      // Check saved traces
      return traceStore.savedTraces.find((t) => t.id === rightTraceId) ?? null;
    },
  };
});
