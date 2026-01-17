/**
 * ParetoView - Main view for quantization config selection
 *
 * Features:
 * - Interactive Pareto frontier chart
 * - Filter controls for method, bits, quality
 * - Config detail card on selection
 * - Blessed config registry
 * - Data import from Schema v1 files
 */

import { useState, useCallback, useRef } from 'react';
import { useParetoStore, useFilteredPoints } from '../../stores/useParetoStore';
import { ParetoChart } from './ParetoChart';
import { ConfigCard, ConfigCardPlaceholder, BlessedConfigList } from './ConfigCard';
import { QualityTier, QuantizationMethod, TilingMode } from '../../api/comparisonSchema';

// ============================================================================
// FILTER PANEL
// ============================================================================

function FilterPanel() {
  const filters = useParetoStore((state) => state.filters);
  const setFilters = useParetoStore((state) => state.setFilters);
  const resetFilters = useParetoStore((state) => state.resetFilters);

  const methods: QuantizationMethod[] = ['sinq', 'asinq', 'awq', 'gptq', 'fp8'];
  const qualityTiers: QualityTier[] = ['excellent', 'good', 'acceptable', 'degraded', 'failed'];
  const tilingModes: TilingMode[] = ['1D', '2D'];

  const toggleMethod = (method: QuantizationMethod) => {
    const newMethods = filters.methods.includes(method)
      ? filters.methods.filter((m) => m !== method)
      : [...filters.methods, method];
    setFilters({ methods: newMethods });
  };

  const toggleTiling = (mode: TilingMode) => {
    const newModes = filters.tilingModes.includes(mode)
      ? filters.tilingModes.filter((m) => m !== mode)
      : [...filters.tilingModes, mode];
    setFilters({ tilingModes: newModes });
  };

  return (
    <div className="filter-panel">
      <div className="filter-header">
        <span>Filters</span>
        <button className="reset-btn" onClick={resetFilters}>Reset</button>
      </div>

      {/* Methods */}
      <div className="filter-section">
        <div className="filter-label">Methods</div>
        <div className="filter-chips">
          {methods.map((method) => (
            <button
              key={method}
              className={`filter-chip ${filters.methods.includes(method) ? 'active' : ''}`}
              onClick={() => toggleMethod(method)}
            >
              {method.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Bits Range */}
      <div className="filter-section">
        <div className="filter-label">Bits: {filters.minNbits} - {filters.maxNbits}</div>
        <div className="range-inputs">
          <input
            type="range"
            min={2}
            max={8}
            value={filters.minNbits}
            onChange={(e) => setFilters({ minNbits: parseInt(e.target.value) })}
          />
          <input
            type="range"
            min={2}
            max={8}
            value={filters.maxNbits}
            onChange={(e) => setFilters({ maxNbits: parseInt(e.target.value) })}
          />
        </div>
      </div>

      {/* Min Quality */}
      <div className="filter-section">
        <div className="filter-label">Min Quality</div>
        <select
          value={filters.minQuality}
          onChange={(e) => setFilters({ minQuality: e.target.value as QualityTier })}
          className="filter-select"
        >
          {qualityTiers.map((tier) => (
            <option key={tier} value={tier}>{tier}</option>
          ))}
        </select>
      </div>

      {/* Min Compression */}
      <div className="filter-section">
        <div className="filter-label">Min Compression: {filters.minCompression.toFixed(1)}x</div>
        <input
          type="range"
          min={1}
          max={8}
          step={0.5}
          value={filters.minCompression}
          onChange={(e) => setFilters({ minCompression: parseFloat(e.target.value) })}
        />
      </div>

      {/* Tiling Mode */}
      <div className="filter-section">
        <div className="filter-label">Tiling Mode</div>
        <div className="filter-chips">
          {tilingModes.map((mode) => (
            <button
              key={mode}
              className={`filter-chip ${filters.tilingModes.includes(mode) ? 'active' : ''}`}
              onClick={() => toggleTiling(mode)}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// AXIS CONTROLS
// ============================================================================

function AxisControls() {
  const xAxis = useParetoStore((state) => state.xAxis);
  const yAxis = useParetoStore((state) => state.yAxis);
  const colorBy = useParetoStore((state) => state.colorBy);
  const setXAxis = useParetoStore((state) => state.setXAxis);
  const setYAxis = useParetoStore((state) => state.setYAxis);
  const setColorBy = useParetoStore((state) => state.setColorBy);

  return (
    <div className="axis-controls">
      <div className="axis-group">
        <label>X Axis</label>
        <select value={xAxis} onChange={(e) => setXAxis(e.target.value as typeof xAxis)}>
          <option value="compressionRatio">Compression Ratio</option>
          <option value="memoryMb">Memory (MB)</option>
          <option value="nbits">Bits</option>
        </select>
      </div>
      <div className="axis-group">
        <label>Y Axis</label>
        <select value={yAxis} onChange={(e) => setYAxis(e.target.value as typeof yAxis)}>
          <option value="meanJaccard">Jaccard Similarity</option>
          <option value="weightedJaccard">Weighted Jaccard</option>
          <option value="spearman">Spearman Correlation</option>
          <option value="massRetained">Mass Retained</option>
        </select>
      </div>
      <div className="axis-group">
        <label>Color By</label>
        <select value={colorBy} onChange={(e) => setColorBy(e.target.value as typeof colorBy)}>
          <option value="method">Method</option>
          <option value="qualityTier">Quality Tier</option>
          <option value="tilingMode">Tiling Mode</option>
        </select>
      </div>
    </div>
  );
}

// ============================================================================
// DATA CONTROLS
// ============================================================================

function DataControls() {
  const loadFromJSON = useParetoStore((state) => state.loadFromJSON);
  const loadDemoData = useParetoStore((state) => state.loadDemoData);
  const clearData = useParetoStore((state) => state.clearData);
  const points = useParetoStore((state) => state.points);
  const error = useParetoStore((state) => state.error);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      loadFromJSON(text);
    } catch (err) {
      console.error('Failed to load file:', err);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [loadFromJSON]);

  return (
    <div className="data-controls">
      <div className="data-status">
        <span className="point-count">{points.length} configs loaded</span>
        {error && <span className="error-text">{error}</span>}
      </div>
      <div className="data-buttons">
        <button onClick={() => fileInputRef.current?.click()}>
          Import JSON
        </button>
        <button onClick={loadDemoData}>
          Load Demo
        </button>
        <button onClick={clearData} disabled={points.length === 0}>
          Clear
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// SUMMARY STATS
// ============================================================================

function SummaryStats() {
  const points = useFilteredPoints();

  if (points.length === 0) {
    return null;
  }

  const frontierCount = points.filter((p) => p.isOnFrontier).length;
  const avgJaccard = points.reduce((sum, p) => sum + p.meanJaccard, 0) / points.length;
  const avgCompression = points.reduce((sum, p) => sum + p.compressionRatio, 0) / points.length;

  // Best trade-off: highest Jaccard with compression > 2x
  const goodCompression = points.filter((p) => p.compressionRatio >= 2);
  const bestTradeoff = goodCompression.length > 0
    ? goodCompression.reduce((best, p) => p.meanJaccard > best.meanJaccard ? p : best)
    : null;

  return (
    <div className="summary-stats">
      <div className="stat">
        <div className="stat-value">{points.length}</div>
        <div className="stat-label">Configs</div>
      </div>
      <div className="stat">
        <div className="stat-value">{frontierCount}</div>
        <div className="stat-label">On Frontier</div>
      </div>
      <div className="stat">
        <div className="stat-value">{(avgJaccard * 100).toFixed(1)}%</div>
        <div className="stat-label">Avg Quality</div>
      </div>
      <div className="stat">
        <div className="stat-value">{avgCompression.toFixed(1)}x</div>
        <div className="stat-label">Avg Compression</div>
      </div>
      {bestTradeoff && (
        <div className="stat highlight">
          <div className="stat-value">{bestTradeoff.configName}</div>
          <div className="stat-label">Recommended</div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// MAIN VIEW
// ============================================================================

export function ParetoView() {
  const points = useParetoStore((state) => state.points);
  const selectedPointId = useParetoStore((state) => state.selectedPointId);
  const blessedConfigs = useParetoStore((state) => state.blessedConfigs);
  const blessConfig = useParetoStore((state) => state.blessConfig);

  const selectedPoint = points.find((p) => p.id === selectedPointId) || null;
  const isSelectedBlessed = selectedPointId
    ? blessedConfigs.some((c) => c.id === selectedPointId)
    : false;

  const [blessReason, setBlessReason] = useState('');
  const [showBlessModal, setShowBlessModal] = useState(false);

  const handleBless = useCallback(() => {
    if (selectedPointId && blessReason) {
      blessConfig(selectedPointId, blessReason);
      setBlessReason('');
      setShowBlessModal(false);
    }
  }, [selectedPointId, blessReason, blessConfig]);

  return (
    <div className="pareto-view card">
      <div className="card-header">
        <div className="card-title">
          <span>Pareto Frontier</span>
          <span className="subtitle">
            Select optimal quantization configs by quality vs compression trade-off
          </span>
        </div>
        <div className="badges">
          <span className="badge">SINQ evaluation</span>
          <span className="badge">schema v1</span>
        </div>
      </div>

      <div className="pareto-layout">
        {/* Left sidebar - Filters */}
        <div className="pareto-sidebar left">
          <FilterPanel />
          <BlessedConfigList />
        </div>

        {/* Main chart area */}
        <div className="pareto-main">
          <DataControls />
          <AxisControls />
          <SummaryStats />

          {points.length > 0 ? (
            <ParetoChart width={700} height={450} />
          ) : (
            <div className="empty-chart">
              <div className="empty-icon">📊</div>
              <div className="empty-title">No Data Loaded</div>
              <div className="empty-text">
                Import Schema v1 JSON files from SINQ evaluation or load demo data
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar - Selected config */}
        <div className="pareto-sidebar right">
          {selectedPoint ? (
            <ConfigCard
              point={selectedPoint}
              isBlessed={isSelectedBlessed}
              onBless={() => {
                if (isSelectedBlessed) {
                  // Already blessed, do nothing or unbless
                } else {
                  setShowBlessModal(true);
                }
              }}
            />
          ) : (
            <ConfigCardPlaceholder />
          )}
        </div>
      </div>

      {/* Bless modal */}
      {showBlessModal && (
        <div className="modal-overlay" onClick={() => setShowBlessModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">Bless Configuration</div>
            <div className="modal-body">
              <p>Mark this config as approved for production use.</p>
              <input
                type="text"
                placeholder="Reason for blessing..."
                value={blessReason}
                onChange={(e) => setBlessReason(e.target.value)}
                className="bless-input"
              />
            </div>
            <div className="modal-footer">
              <button onClick={() => setShowBlessModal(false)}>Cancel</button>
              <button
                className="primary"
                onClick={handleBless}
                disabled={!blessReason}
              >
                Bless
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
