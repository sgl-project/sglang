/**
 * Pareto Store - State management for quantization config selection
 *
 * Manages:
 * - Loading comparison results from Schema v1 files
 * - Computing Pareto frontier
 * - Filtering and selection state
 * - Blessed config registry
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  QuantizationComparisonV1,
  QualityTier,
  QuantizationMethod,
  TilingMode,
  classifyQualityTier,
} from '../api/comparisonSchema';
import {
  fetchBlessedConfigs,
  saveBlessedConfig,
  deleteBlessedConfig,
  checkSidecarAvailable,
} from '../api/blessedConfigsApi';

// ============================================================================
// TYPES
// ============================================================================

export interface ParetoPoint {
  id: string;
  configName: string;
  method: QuantizationMethod;
  nbits: number;
  groupSize: number;
  tilingMode: TilingMode;
  meanJaccard: number;
  weightedJaccard: number;
  spearman: number;
  massRetained: number;
  klDivergence: number;
  outputMatchRate: number;
  compressionRatio: number;
  memoryMb: number;
  qualityTier: QualityTier;
  isOnFrontier: boolean;
  comparisonId: string;
  timestamp: string;
}

export interface ParetoFilters {
  methods: QuantizationMethod[];
  minNbits: number;
  maxNbits: number;
  minQuality: QualityTier;
  minCompression: number;
  tilingModes: TilingMode[];
}

export interface BlessedConfig {
  id: string;
  configName: string;
  blessedAt: number;
  reason: string;
}

interface ParetoState {
  // Data
  points: ParetoPoint[];
  comparisons: QuantizationComparisonV1[];
  isLoading: boolean;
  error: string | null;

  // Filters
  filters: ParetoFilters;

  // Selection
  selectedPointId: string | null;
  hoveredPointId: string | null;

  // Blessed configs
  blessedConfigs: BlessedConfig[];
  backendAvailable: boolean;
  isSyncing: boolean;

  // Chart settings
  xAxis: 'compressionRatio' | 'memoryMb' | 'nbits';
  yAxis: 'meanJaccard' | 'weightedJaccard' | 'spearman' | 'massRetained';
  colorBy: 'method' | 'qualityTier' | 'tilingMode';

  // Actions
  loadFromJSON: (json: string) => void;
  loadFromComparison: (comparison: QuantizationComparisonV1) => void;
  loadDemoData: () => void;
  clearData: () => void;

  setFilters: (filters: Partial<ParetoFilters>) => void;
  resetFilters: () => void;

  selectPoint: (id: string | null) => void;
  hoverPoint: (id: string | null) => void;

  blessConfig: (pointId: string, reason: string) => void;
  unblessConfig: (id: string) => void;
  syncBlessedConfigs: () => Promise<void>;

  setXAxis: (axis: ParetoState['xAxis']) => void;
  setYAxis: (axis: ParetoState['yAxis']) => void;
  setColorBy: (colorBy: ParetoState['colorBy']) => void;
}

// ============================================================================
// DEFAULTS
// ============================================================================

const DEFAULT_FILTERS: ParetoFilters = {
  methods: ['sinq', 'asinq', 'awq', 'gptq', 'fp8'],
  minNbits: 2,
  maxNbits: 8,
  minQuality: 'degraded',
  minCompression: 1.0,
  tilingModes: ['1D', '2D'],
};

// ============================================================================
// PARETO COMPUTATION
// ============================================================================

function computeParetoFrontier(points: ParetoPoint[]): Set<string> {
  const frontierIds = new Set<string>();

  for (const point of points) {
    let dominated = false;

    for (const other of points) {
      if (point.id === other.id) continue;

      // Other dominates point if:
      // - Other has higher quality AND higher compression
      // - OR same quality and strictly higher compression
      // - OR strictly higher quality and same compression
      const otherBetterQuality = other.meanJaccard > point.meanJaccard;
      const otherBetterCompression = other.compressionRatio > point.compressionRatio;
      const sameQuality = Math.abs(other.meanJaccard - point.meanJaccard) < 0.001;
      const sameCompression = Math.abs(other.compressionRatio - point.compressionRatio) < 0.001;

      if (
        (otherBetterQuality && otherBetterCompression) ||
        (sameQuality && otherBetterCompression) ||
        (otherBetterQuality && sameCompression)
      ) {
        dominated = true;
        break;
      }
    }

    if (!dominated) {
      frontierIds.add(point.id);
    }
  }

  return frontierIds;
}

function comparisonToPoints(comparison: QuantizationComparisonV1): ParetoPoint[] {
  const points: ParetoPoint[] = [];

  const quant = comparison.candidate.quantization;
  if (!quant) return points;

  const summary = comparison.results.summary;
  const configName = `${quant.method}_${quant.nbits}b_g${quant.group_size}_${quant.tiling_mode}`;

  points.push({
    id: comparison.comparison_id,
    configName,
    method: quant.method as QuantizationMethod,
    nbits: quant.nbits ?? 4,
    groupSize: quant.group_size ?? 128,
    tilingMode: (quant.tiling_mode ?? '1D') as TilingMode,
    meanJaccard: summary.jaccard.mean,
    weightedJaccard: summary.weighted_jaccard?.mean ?? summary.jaccard.mean,
    spearman: summary.rank_correlation?.spearman_mean ?? 0,
    massRetained: summary.mass_retained?.mean ?? 0,
    klDivergence: summary.kl_divergence?.mean ?? 0,
    outputMatchRate: summary.output_agreement?.exact_match_rate ?? 0,
    compressionRatio: summary.compression_ratio ?? 1,
    memoryMb: comparison.candidate.memory_mb ?? 0,
    qualityTier: summary.quality_tier,
    isOnFrontier: false,
    comparisonId: comparison.comparison_id,
    timestamp: comparison.timestamp,
  });

  return points;
}

// ============================================================================
// DEMO DATA
// ============================================================================

function generateDemoData(): ParetoPoint[] {
  const methods: QuantizationMethod[] = ['sinq', 'asinq'];
  const nbitsOptions = [2, 3, 4, 5, 6, 8];
  const groupSizes = [64, 128];
  const tilingModes: TilingMode[] = ['1D', '2D'];

  const points: ParetoPoint[] = [];

  for (const method of methods) {
    for (const nbits of nbitsOptions) {
      for (const groupSize of groupSizes) {
        for (const tilingMode of tilingModes) {
          // Simulate realistic metrics
          const baseQuality = 0.3 + (nbits / 8) * 0.6; // Higher bits = better quality
          const methodBonus = method === 'asinq' ? 0.05 : 0;
          const tilingBonus = tilingMode === '2D' ? 0.02 : 0;
          const groupBonus = groupSize === 64 ? 0.03 : 0;

          const meanJaccard = Math.min(1, baseQuality + methodBonus + tilingBonus + groupBonus + (Math.random() * 0.1 - 0.05));
          const compressionRatio = 16 / nbits * (groupSize === 128 ? 1.05 : 1);

          const configName = `${method}_${nbits}b_g${groupSize}_${tilingMode}`;
          const id = `demo_${configName}`;

          points.push({
            id,
            configName,
            method,
            nbits,
            groupSize,
            tilingMode,
            meanJaccard,
            weightedJaccard: meanJaccard * 0.95,
            spearman: meanJaccard * 0.9,
            massRetained: meanJaccard * 0.85,
            klDivergence: (1 - meanJaccard) * 0.5,
            outputMatchRate: meanJaccard > 0.7 ? 0.8 : 0.4,
            compressionRatio,
            memoryMb: 4000 / compressionRatio,
            qualityTier: classifyQualityTier(meanJaccard),
            isOnFrontier: false,
            comparisonId: id,
            timestamp: new Date().toISOString(),
          });
        }
      }
    }
  }

  return points;
}

// ============================================================================
// STORE
// ============================================================================

export const useParetoStore = create<ParetoState>()(
  persist(
    (set, get) => ({
      // Initial state
      points: [],
      comparisons: [],
      isLoading: false,
      error: null,

      filters: DEFAULT_FILTERS,

      selectedPointId: null,
      hoveredPointId: null,

      blessedConfigs: [],
      backendAvailable: false,
      isSyncing: false,

      xAxis: 'compressionRatio',
      yAxis: 'meanJaccard',
      colorBy: 'method',

      // Actions
      loadFromJSON: (json: string) => {
        try {
          set({ isLoading: true, error: null });

          const data = JSON.parse(json);

          // Handle array of comparisons or single comparison
          const comparisons: QuantizationComparisonV1[] = Array.isArray(data)
            ? data
            : data.results
            ? [data]
            : data;

          let allPoints: ParetoPoint[] = [];

          for (const comparison of comparisons) {
            if (comparison.schema_version === '1.0.0') {
              allPoints = [...allPoints, ...comparisonToPoints(comparison)];
            }
          }

          // Compute Pareto frontier
          const frontierIds = computeParetoFrontier(allPoints);
          allPoints = allPoints.map((p) => ({
            ...p,
            isOnFrontier: frontierIds.has(p.id),
          }));

          set({
            comparisons,
            points: allPoints,
            isLoading: false,
          });
        } catch (e) {
          set({
            error: e instanceof Error ? e.message : 'Failed to parse JSON',
            isLoading: false,
          });
        }
      },

      loadFromComparison: (comparison: QuantizationComparisonV1) => {
        const { comparisons, points } = get();

        // Check if already loaded
        if (comparisons.some((c) => c.comparison_id === comparison.comparison_id)) {
          return;
        }

        const newPoints = comparisonToPoints(comparison);
        const allPoints = [...points, ...newPoints];

        // Recompute Pareto frontier
        const frontierIds = computeParetoFrontier(allPoints);
        const updatedPoints = allPoints.map((p) => ({
          ...p,
          isOnFrontier: frontierIds.has(p.id),
        }));

        set({
          comparisons: [...comparisons, comparison],
          points: updatedPoints,
        });
      },

      loadDemoData: () => {
        const demoPoints = generateDemoData();
        const frontierIds = computeParetoFrontier(demoPoints);
        const points = demoPoints.map((p) => ({
          ...p,
          isOnFrontier: frontierIds.has(p.id),
        }));

        set({ points, comparisons: [], error: null });
      },

      clearData: () => {
        set({
          points: [],
          comparisons: [],
          selectedPointId: null,
          hoveredPointId: null,
          error: null,
        });
      },

      setFilters: (newFilters: Partial<ParetoFilters>) => {
        set((state) => ({
          filters: { ...state.filters, ...newFilters },
        }));
      },

      resetFilters: () => {
        set({ filters: DEFAULT_FILTERS });
      },

      selectPoint: (id: string | null) => {
        set({ selectedPointId: id });
      },

      hoverPoint: (id: string | null) => {
        set({ hoveredPointId: id });
      },

      blessConfig: (pointId: string, reason: string) => {
        const { points, blessedConfigs, backendAvailable } = get();
        const point = points.find((p) => p.id === pointId);
        if (!point) return;

        // Remove if already blessed
        const filtered = blessedConfigs.filter((c) => c.id !== pointId);

        const newConfig: BlessedConfig = {
          id: pointId,
          configName: point.configName,
          blessedAt: Date.now(),
          reason,
        };

        set({
          blessedConfigs: [...filtered, newConfig],
        });

        // Save to backend if available
        if (backendAvailable) {
          saveBlessedConfig(newConfig).catch((err) => {
            console.warn('Failed to save blessed config to backend:', err);
          });
        }
      },

      unblessConfig: (id: string) => {
        const { backendAvailable } = get();

        set((state) => ({
          blessedConfigs: state.blessedConfigs.filter((c) => c.id !== id),
        }));

        // Delete from backend if available
        if (backendAvailable) {
          deleteBlessedConfig(id).catch((err) => {
            console.warn('Failed to delete blessed config from backend:', err);
          });
        }
      },

      syncBlessedConfigs: async () => {
        set({ isSyncing: true });

        try {
          // Check if backend is available
          const available = await checkSidecarAvailable();
          set({ backendAvailable: available });

          if (available) {
            // Fetch configs from backend
            const remoteConfigs = await fetchBlessedConfigs();
            const { blessedConfigs: localConfigs } = get();

            // Merge: keep local configs that aren't in remote, add all remote
            const remoteIds = new Set(remoteConfigs.map((c) => c.id));
            const localOnly = localConfigs.filter((c) => !remoteIds.has(c.id));

            // Upload local-only configs to backend
            for (const config of localOnly) {
              await saveBlessedConfig(config);
            }

            // Update store with merged configs
            set({
              blessedConfigs: [...remoteConfigs, ...localOnly],
              isSyncing: false,
            });
          } else {
            set({ isSyncing: false });
          }
        } catch (err) {
          console.warn('Failed to sync blessed configs:', err);
          set({ isSyncing: false, backendAvailable: false });
        }
      },

      setXAxis: (axis) => set({ xAxis: axis }),
      setYAxis: (axis) => set({ yAxis: axis }),
      setColorBy: (colorBy) => set({ colorBy }),
    }),
    {
      name: 'pareto-store',
      partialize: (state) => ({
        blessedConfigs: state.blessedConfigs,
        filters: state.filters,
        xAxis: state.xAxis,
        yAxis: state.yAxis,
        colorBy: state.colorBy,
      }),
      onRehydrateStorage: () => (state) => {
        // Sync blessed configs with backend after rehydration
        if (state) {
          state.syncBlessedConfigs();
        }
      },
    }
  )
);

// ============================================================================
// SELECTORS
// ============================================================================

export function useFilteredPoints(): ParetoPoint[] {
  const points = useParetoStore((state) => state.points);
  const filters = useParetoStore((state) => state.filters);

  const qualityOrder: QualityTier[] = ['failed', 'degraded', 'acceptable', 'good', 'excellent'];
  const minQualityIdx = qualityOrder.indexOf(filters.minQuality);

  return points.filter((p) => {
    if (!filters.methods.includes(p.method)) return false;
    if (p.nbits < filters.minNbits || p.nbits > filters.maxNbits) return false;
    if (qualityOrder.indexOf(p.qualityTier) < minQualityIdx) return false;
    if (p.compressionRatio < filters.minCompression) return false;
    if (!filters.tilingModes.includes(p.tilingMode)) return false;
    return true;
  });
}

export function useParetoFrontierPoints(): ParetoPoint[] {
  const filteredPoints = useFilteredPoints();
  return filteredPoints.filter((p) => p.isOnFrontier);
}
