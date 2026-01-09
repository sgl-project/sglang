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

// Discovery progress state
export interface DiscoveryProgress {
  run_id: string | null;
  stage: number;
  stage_name: string;
  percent_complete: number;
  items_processed: number;
  total_items: number;
  eta_seconds: number | null;
  is_running: boolean;
}

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

  // Live discovery state
  discoveryProgress: DiscoveryProgress;
  sseConnected: boolean;
  sidecarUrl: string;

  // Actions
  setScope: (scope: ManifoldScope) => void;
  selectPoint: (pointId: string | null) => void;
  hoverPoint: (pointId: string | null) => void;

  // Data management
  loadArtifacts: (artifacts: ManifoldArtifacts) => void;
  addSessionPoint: (point: ManifoldPoint) => void;
  addSessionPoints: (points: ManifoldPoint[]) => void;
  clearSessionPoints: () => void;

  // Live discovery SSE
  setSidecarUrl: (url: string) => void;
  connectToDiscovery: () => void;
  disconnectFromDiscovery: () => void;
  handleDiscoveryEvent: (eventType: string, data: any) => void;

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

// Default discovery progress state
const DEFAULT_DISCOVERY_PROGRESS: DiscoveryProgress = {
  run_id: null,
  stage: 0,
  stage_name: '',
  percent_complete: 0,
  items_processed: 0,
  total_items: 0,
  eta_seconds: null,
  is_running: false,
};

// Global SSE connection (outside React lifecycle)
let sseConnection: EventSource | null = null;

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
      discoveryProgress: DEFAULT_DISCOVERY_PROGRESS,
      sseConnected: false,
      sidecarUrl: 'http://localhost:9000',

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

      addSessionPoints: (points) => {
        set((state) => ({
          sessionPoints: [...state.sessionPoints, ...points],
        }));
      },

      clearSessionPoints: () => {
        set({ sessionPoints: [], artifacts: null });
      },

      setSidecarUrl: (url) => {
        set({ sidecarUrl: url });
        // Reconnect if already connected
        if (get().sseConnected) {
          get().disconnectFromDiscovery();
          get().connectToDiscovery();
        }
      },

      connectToDiscovery: () => {
        const { sidecarUrl, handleDiscoveryEvent } = get();

        // Close existing connection
        if (sseConnection) {
          sseConnection.close();
          sseConnection = null;
        }

        try {
          const sseUrl = `${sidecarUrl}/discovery/live`;
          console.log('[Manifold] Connecting to SSE:', sseUrl);

          sseConnection = new EventSource(sseUrl);

          sseConnection.onopen = () => {
            console.log('[Manifold] SSE connected');
            set({ sseConnected: true, error: null });
          };

          sseConnection.onerror = (event) => {
            console.error('[Manifold] SSE error:', event);
            set({ sseConnected: false, error: 'SSE connection error' });
          };

          // Handle different event types
          const eventTypes = [
            'status', 'progress', 'stage_start', 'stage_complete',
            'batch_complete', 'cluster_update', 'zone_stats',
            'run_start', 'run_complete', 'run_error', 'heartbeat'
          ];

          for (const eventType of eventTypes) {
            sseConnection.addEventListener(eventType, (event: MessageEvent) => {
              try {
                const data = JSON.parse(event.data);
                handleDiscoveryEvent(eventType, data);
              } catch (e) {
                console.warn('[Manifold] Failed to parse SSE event:', e);
              }
            });
          }

          // Generic message handler as fallback
          sseConnection.onmessage = (event: MessageEvent) => {
            try {
              const data = JSON.parse(event.data);
              handleDiscoveryEvent('message', data);
            } catch (e) {
              console.warn('[Manifold] Failed to parse SSE message:', e);
            }
          };

        } catch (e) {
          console.error('[Manifold] Failed to connect to SSE:', e);
          set({ sseConnected: false, error: `Failed to connect: ${e}` });
        }
      },

      disconnectFromDiscovery: () => {
        if (sseConnection) {
          sseConnection.close();
          sseConnection = null;
        }
        set({
          sseConnected: false,
          discoveryProgress: DEFAULT_DISCOVERY_PROGRESS,
        });
        console.log('[Manifold] SSE disconnected');
      },

      handleDiscoveryEvent: (eventType, data) => {
        console.log('[Manifold] SSE event:', eventType, data);

        switch (eventType) {
          case 'status':
          case 'progress':
            set((state) => ({
              discoveryProgress: {
                ...state.discoveryProgress,
                run_id: data.run_id || state.discoveryProgress.run_id,
                stage: data.stage ?? state.discoveryProgress.stage,
                stage_name: data.stage_name || state.discoveryProgress.stage_name,
                percent_complete: data.percent_complete ?? state.discoveryProgress.percent_complete,
                items_processed: data.items_processed ?? state.discoveryProgress.items_processed,
                total_items: data.total_items ?? state.discoveryProgress.total_items,
                eta_seconds: data.eta_seconds ?? state.discoveryProgress.eta_seconds,
                is_running: data.is_running ?? (data.run_id != null),
              },
            }));
            break;

          case 'stage_start':
            set((state) => ({
              discoveryProgress: {
                ...state.discoveryProgress,
                stage: data.stage,
                stage_name: data.stage_name,
                percent_complete: 0,
                is_running: true,
              },
            }));
            break;

          case 'stage_complete':
            set((state) => ({
              discoveryProgress: {
                ...state.discoveryProgress,
                stage: data.stage,
                stage_name: data.stage_name,
                percent_complete: 100,
              },
            }));
            break;

          case 'batch_complete':
            // Add new points from batch
            if (data.new_points && Array.isArray(data.new_points)) {
              const points: ManifoldPoint[] = data.new_points.map((p: any) => ({
                session_id: `discovery-${p.fingerprint_id || Math.random()}`,
                coords: [p.x || 0, p.y || 0] as [number, number],
                cluster_id: p.cluster_id ?? -1,
                manifold_zone: (p.zone || 'unknown') as ManifoldZone,
                consensus: 0,
                hubness: 0,
                entropy: 0,
                timestamp: new Date().toISOString(),
                model_id: 'discovery',
              }));
              get().addSessionPoints(points);
            }
            break;

          case 'run_start':
            set((state) => ({
              discoveryProgress: {
                ...state.discoveryProgress,
                run_id: data.run_id,
                stage: 0,
                stage_name: 'starting',
                percent_complete: 0,
                is_running: true,
              },
            }));
            break;

          case 'run_complete':
            set((state) => ({
              discoveryProgress: {
                ...state.discoveryProgress,
                is_running: false,
                percent_complete: 100,
              },
            }));
            break;

          case 'run_error':
            set((state) => ({
              discoveryProgress: {
                ...state.discoveryProgress,
                is_running: false,
              },
              error: data.message || 'Discovery run failed',
            }));
            break;

          case 'heartbeat':
            // Just update connection state
            break;

          default:
            // Unknown event type
            break;
        }
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
        sidecarUrl: state.sidecarUrl,
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
