/**
 * ConfigCard - Displays detailed metrics for a selected quantization config
 */

import { useParetoStore, ParetoPoint } from '../../stores/useParetoStore';
import { getQualityTierColor } from '../../api/comparisonSchema';

interface ConfigCardProps {
  point: ParetoPoint;
  onBless?: () => void;
  isBlessed?: boolean;
}

export function ConfigCard({ point, onBless, isBlessed }: ConfigCardProps) {
  const tierColor = getQualityTierColor(point.qualityTier);

  return (
    <div className={`config-card ${point.isOnFrontier ? 'on-frontier' : ''}`}>
      <div className="config-header">
        <div className="config-name">{point.configName}</div>
        <div
          className="quality-badge"
          style={{ backgroundColor: tierColor }}
        >
          {point.qualityTier}
        </div>
      </div>

      <div className="config-grid">
        {/* Primary metrics */}
        <div className="metric-section">
          <div className="section-title">Quality Metrics</div>
          <div className="metric-row">
            <span className="metric-label">Jaccard</span>
            <span className="metric-value">{(point.meanJaccard * 100).toFixed(1)}%</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{
                  width: `${point.meanJaccard * 100}%`,
                  backgroundColor: tierColor,
                }}
              />
            </div>
          </div>
          <div className="metric-row">
            <span className="metric-label">Weighted Jaccard</span>
            <span className="metric-value">{(point.weightedJaccard * 100).toFixed(1)}%</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: `${point.weightedJaccard * 100}%` }}
              />
            </div>
          </div>
          <div className="metric-row">
            <span className="metric-label">Spearman rho</span>
            <span className="metric-value">{point.spearman.toFixed(3)}</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: `${Math.max(0, point.spearman) * 100}%` }}
              />
            </div>
          </div>
          <div className="metric-row">
            <span className="metric-label">Mass Retained</span>
            <span className="metric-value">{(point.massRetained * 100).toFixed(1)}%</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: `${point.massRetained * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Efficiency metrics */}
        <div className="metric-section">
          <div className="section-title">Efficiency</div>
          <div className="metric-row">
            <span className="metric-label">Compression</span>
            <span className="metric-value highlight">{point.compressionRatio.toFixed(2)}x</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Memory</span>
            <span className="metric-value">{point.memoryMb.toFixed(0)} MB</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Output Match</span>
            <span className="metric-value">{(point.outputMatchRate * 100).toFixed(0)}%</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">KL Divergence</span>
            <span className="metric-value">{point.klDivergence.toFixed(4)}</span>
          </div>
        </div>

        {/* Config details */}
        <div className="metric-section">
          <div className="section-title">Configuration</div>
          <div className="config-detail">
            <span>Method:</span>
            <span className="method-badge">{point.method.toUpperCase()}</span>
          </div>
          <div className="config-detail">
            <span>Bits:</span>
            <span>{point.nbits}</span>
          </div>
          <div className="config-detail">
            <span>Group Size:</span>
            <span>{point.groupSize}</span>
          </div>
          <div className="config-detail">
            <span>Tiling:</span>
            <span>{point.tilingMode}</span>
          </div>
        </div>
      </div>

      {/* Frontier indicator */}
      {point.isOnFrontier && (
        <div className="frontier-indicator">
          <span className="frontier-icon">*</span>
          On Pareto Frontier - No config dominates this one
        </div>
      )}

      {/* Actions */}
      <div className="config-actions">
        <button
          className={`bless-btn ${isBlessed ? 'blessed' : ''}`}
          onClick={onBless}
        >
          {isBlessed ? 'Blessed' : 'Bless Config'}
        </button>
        <button className="copy-btn" onClick={() => {
          const configJson = JSON.stringify({
            method: point.method,
            nbits: point.nbits,
            group_size: point.groupSize,
            tiling_mode: point.tilingMode,
          }, null, 2);
          navigator.clipboard.writeText(configJson);
        }}>
          Copy JSON
        </button>
      </div>
    </div>
  );
}

/**
 * ConfigCardPlaceholder - Shown when no config is selected
 */
export function ConfigCardPlaceholder() {
  return (
    <div className="config-card placeholder">
      <div className="placeholder-content">
        <div className="placeholder-icon">?</div>
        <div className="placeholder-text">
          Click a point on the chart to see detailed metrics
        </div>
        <div className="placeholder-hint">
          Points on the Pareto frontier represent optimal trade-offs
        </div>
      </div>
    </div>
  );
}

/**
 * BlessedConfigList - Shows all blessed configurations
 */
export function BlessedConfigList() {
  const blessedConfigs = useParetoStore((state) => state.blessedConfigs);
  const points = useParetoStore((state) => state.points);
  const unblessConfig = useParetoStore((state) => state.unblessConfig);
  const selectPoint = useParetoStore((state) => state.selectPoint);

  if (blessedConfigs.length === 0) {
    return (
      <div className="blessed-list empty">
        <div className="empty-text">No blessed configs yet</div>
        <div className="empty-hint">
          Bless configs to mark them as approved for production use
        </div>
      </div>
    );
  }

  return (
    <div className="blessed-list">
      <div className="blessed-header">
        <span>Blessed Configs</span>
        <span className="count">{blessedConfigs.length}</span>
      </div>
      {blessedConfigs.map((config) => {
        const point = points.find((p) => p.id === config.id);

        return (
          <div
            key={config.id}
            className="blessed-item"
            onClick={() => selectPoint(config.id)}
          >
            <div className="blessed-name">{config.configName}</div>
            <div className="blessed-meta">
              <span className="blessed-reason">{config.reason}</span>
              <span className="blessed-date">
                {new Date(config.blessedAt).toLocaleDateString()}
              </span>
            </div>
            {point && (
              <div className="blessed-metrics">
                <span>J: {(point.meanJaccard * 100).toFixed(0)}%</span>
                <span>C: {point.compressionRatio.toFixed(1)}x</span>
              </div>
            )}
            <button
              className="unbless-btn"
              onClick={(e) => {
                e.stopPropagation();
                unblessConfig(config.id);
              }}
            >
              x
            </button>
          </div>
        );
      })}
    </div>
  );
}
