import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  ManifoldPoint,
  ClusterDefinition,
  ManifoldArtifacts,
  ManifoldZone,
  Fingerprint,
  classifyManifold,
} from '../api/types';
import { useTraceStore } from './useTraceStore';

// Scope options for filtering which sessions to show
export type ManifoldScope = 'current' | 'recent' | 'saved' | 'all';

// How many recent sessions to show
const RECENT_SESSION_COUNT = 50;

// Default cluster colors by zone
const ZONE_COLORS: Record<ManifoldZone, string> = {
  syntax_floor: 'rgba(85, 214, 166, 0.8)',    // Green
  semantic_bridge: 'rgba(122, 162, 255, 0.8)', // Blue
  long_range: 'rgba(255, 204, 102, 0.8)',      // Yellow
  structure_ripple: 'rgba(255, 107, 107, 0.8)', // Red
  diffuse: 'rgba(156, 163, 175, 0.6)',         // Gray
  unknown: 'rgba(156, 163, 175, 0.4)',         // Light gray
};

interface ManifoldState {
  // Loaded artifacts (from file or generated)
  artifacts: ManifoldArtifacts | null;

  // Historical session points (persisted)
  sessionPoints: ManifoldPoint[];

  // Scope selection
  scope: ManifoldScope;

  // Selected point for details
  selectedPointId: string | null;

  // Hover point
  hoveredPointId: string | null;

  // Loading state
  isLoading: boolean;
  error: string | null;

  // Actions
  setScope: (scope: ManifoldScope) => void;
  selectPoint: (pointId: string | null) => void;
  hoverPoint: (pointId: string | null) => void;

  // Data management
  loadArtifacts: (artifacts: ManifoldArtifacts) => void;
  addSessionPoint: (point: ManifoldPoint) => void;
  clearSessionPoints: () => void;

  // Computed
  getFilteredPoints: () => ManifoldPoint[];
  getClusters: () => ClusterDefinition[];
  getPointColor: (point: ManifoldPoint) => string;
  getCurrentSessionPoint: () => ManifoldPoint | null;
}

// Project fingerprint to approximate UMAP coordinates
// This is a simple linear projection; real implementation would use a trained model
function projectFingerprint(fp: Fingerprint): [number, number] {
  // Use mass distribution to position in 2D space
  // x-axis: local vs long (local = left, long = right)
  // y-axis: entropy (low = bottom, high = top)
  const x = 0.5 + (fp.long_mass - fp.local_mass) * 0.3 + (fp.mid_mass - 0.33) * 0.2;
  const y = 0.5 - (fp.entropy - 2.0) * 0.15 + (fp.hubness ?? 0) * 0.1;

  // Clamp to [0.1, 0.9] to keep points away from edges
  return [
    Math.max(0.1, Math.min(0.9, x)),
    Math.max(0.1, Math.min(0.9, y)),
  ];
}

// Generate cluster definitions from points
function computeClusters(points: ManifoldPoint[]): ClusterDefinition[] {
  if (points.length === 0) return [];

  // Group by cluster_id
  const grouped = new Map<number, ManifoldPoint[]>();
  for (const point of points) {
    const existing = grouped.get(point.cluster_id) || [];
    existing.push(point);
    grouped.set(point.cluster_id, existing);
  }

  // Create cluster definitions
  const clusters: ClusterDefinition[] = [];
  for (const [clusterId, clusterPoints] of grouped) {
    if (clusterPoints.length === 0) continue;

    // Compute centroid
    const sumX = clusterPoints.reduce((s, p) => s + p.coords[0], 0);
    const sumY = clusterPoints.reduce((s, p) => s + p.coords[1], 0);
    const centroid: [number, number] = [sumX / clusterPoints.length, sumY / clusterPoints.length];

    // Compute radius (max distance from centroid)
    let maxDist = 0;
    for (const p of clusterPoints) {
      const dist = Math.sqrt(
        Math.pow(p.coords[0] - centroid[0], 2) + Math.pow(p.coords[1] - centroid[1], 2)
      );
      maxDist = Math.max(maxDist, dist);
    }

    // Compute averages
    const avgConsensus = clusterPoints.reduce((s, p) => s + p.consensus, 0) / clusterPoints.length;
    const avgHubness = clusterPoints.reduce((s, p) => s + p.hubness, 0) / clusterPoints.length;
    const avgEntropy = clusterPoints.reduce((s, p) => s + p.entropy, 0) / clusterPoints.length;

    // Determine dominant zone
    const zoneCounts = new Map<ManifoldZone, number>();
    for (const p of clusterPoints) {
      zoneCounts.set(p.manifold_zone, (zoneCounts.get(p.manifold_zone) || 0) + 1);
    }
    let dominantZone: ManifoldZone = 'unknown';
    let maxCount = 0;
    for (const [zone, count] of zoneCounts) {
      if (count > maxCount) {
        maxCount = count;
        dominantZone = zone;
      }
    }

    // Get name from first point or generate from zone
    const name = clusterPoints[0].cluster_name || formatZoneName(dominantZone);

    clusters.push({
      cluster_id: clusterId,
      name,
      centroid,
      radius: maxDist || 0.05, // Minimum radius
      avg_consensus: avgConsensus,
      avg_hubness: avgHubness,
      avg_entropy: avgEntropy,
      dominant_zone: dominantZone,
      point_count: clusterPoints.length,
      stability: 1.0, // Would be computed from temporal analysis
    });
  }

  return clusters.sort((a, b) => b.point_count - a.point_count);
}

function formatZoneName(zone: ManifoldZone): string {
  return zone
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export const useManifoldStore = create<ManifoldState>()(
  persist(
    (set, get) => ({
      artifacts: null,
      sessionPoints: [],
      scope: 'all',
      selectedPointId: null,
      hoveredPointId: null,
      isLoading: false,
      error: null,

      setScope: (scope) => set({ scope }),

      selectPoint: (pointId) => set({ selectedPointId: pointId }),

      hoverPoint: (pointId) => set({ hoveredPointId: pointId }),

      loadArtifacts: (artifacts) => {
        set({
          artifacts,
          sessionPoints: [...get().sessionPoints, ...artifacts.points],
          isLoading: false,
          error: null,
        });
      },

      addSessionPoint: (point) => {
        set((state) => ({
          sessionPoints: [...state.sessionPoints, point],
        }));
      },

      clearSessionPoints: () => {
        set({ sessionPoints: [], artifacts: null });
      },

      getFilteredPoints: () => {
        const { scope, sessionPoints } = get();
        const traceStore = useTraceStore.getState();

        switch (scope) {
          case 'current': {
            // Only show current session point
            const currentPoint = get().getCurrentSessionPoint();
            return currentPoint ? [currentPoint] : [];
          }
          case 'recent': {
            // Show last N sessions
            return sessionPoints.slice(-RECENT_SESSION_COUNT);
          }
          case 'saved': {
            // Show points from saved traces
            const savedIds = new Set(traceStore.savedTraces.map((t) => t.id));
            return sessionPoints.filter((p) => savedIds.has(p.session_id));
          }
          case 'all':
          default:
            return sessionPoints;
        }
      },

      getClusters: () => {
        const filteredPoints = get().getFilteredPoints();
        return computeClusters(filteredPoints);
      },

      getPointColor: (point) => {
        return ZONE_COLORS[point.manifold_zone] || ZONE_COLORS.unknown;
      },

      getCurrentSessionPoint: () => {
        const traceStore = useTraceStore.getState();
        const trace = traceStore.currentTrace;
        if (!trace || !trace.metrics) return null;

        const fingerprint: Fingerprint = {
          histogram: [],
          local_mass: trace.metrics.avgLocalMass,
          mid_mass: trace.metrics.avgMidMass,
          long_mass: trace.metrics.avgLongMass,
          entropy: trace.metrics.avgEntropy,
          consensus: trace.metrics.avgConsensus,
          hubness: trace.metrics.avgHubness,
        };

        const coords = projectFingerprint(fingerprint);

        return {
          session_id: trace.id,
          coords,
          cluster_id: -1, // Current session not assigned to cluster
          manifold_zone: classifyManifold(fingerprint),
          consensus: trace.metrics.avgConsensus ?? 0,
          hubness: trace.metrics.avgHubness ?? 0,
          entropy: trace.metrics.avgEntropy,
          timestamp: new Date(trace.createdAt).toISOString(),
          model_id: trace.model,
        };
      },
    }),
    {
      name: 'manifold-storage',
      partialize: (state) => ({
        sessionPoints: state.sessionPoints.slice(-200), // Keep last 200 points
        scope: state.scope,
      }),
    }
  )
);

// Helper to add current trace as a point when session completes
export function recordCurrentSession() {
  const manifoldStore = useManifoldStore.getState();

  const currentPoint = manifoldStore.getCurrentSessionPoint();
  if (currentPoint && currentPoint.session_id !== '') {
    // Check if already recorded
    const exists = manifoldStore.sessionPoints.some(
      (p) => p.session_id === currentPoint.session_id
    );
    if (!exists) {
      manifoldStore.addSessionPoint(currentPoint);
    }
  }
}
