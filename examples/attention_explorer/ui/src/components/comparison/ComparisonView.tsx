import { useComparisonStore } from '../../stores/useComparisonStore';
import { SessionSelector } from './SessionSelector';
import { MetricsDiffCard } from './MetricsDiffCard';
import { FingerprintDiffChart } from './FingerprintDiffChart';
import { ZoneBadgeComparison } from './ZoneBadgeComparison';

export function ComparisonView() {
  const leftTraceId = useComparisonStore((state) => state.leftTraceId);
  const rightTraceId = useComparisonStore((state) => state.rightTraceId);
  const comparison = useComparisonStore((state) => state.comparison);
  const error = useComparisonStore((state) => state.error);
  const isComputing = useComparisonStore((state) => state.isComputing);
  const swapSessions = useComparisonStore((state) => state.swapSessions);
  const clearComparison = useComparisonStore((state) => state.clearComparison);

  const hasSelection = leftTraceId && rightTraceId;

  return (
    <div className="card comparison-view">
      <div className="card-header">
        <div className="card-title">
          <span>Compare Sessions</span>
          <span className="subtitle">Side-by-side attention pattern analysis</span>
        </div>
        <div className="badges">
          {comparison && (
            <span className="badge strong">
              {Math.round(comparison.overallSimilarity * 100)}% Similar
            </span>
          )}
        </div>
      </div>

      <div className="card-content comparison-layout">
        {/* Session Selection Row */}
        <div className="comparison-selectors">
          <div className="selector-group">
            <label className="selector-label">Session A</label>
            <SessionSelector side="left" />
          </div>

          <div className="swap-button-container">
            <button
              className="swap-btn"
              onClick={swapSessions}
              disabled={!hasSelection}
              title="Swap sessions"
            >
              &#x21C4;
            </button>
          </div>

          <div className="selector-group">
            <label className="selector-label">Session B</label>
            <SessionSelector side="right" />
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="comparison-error">
            <span className="error-icon">!</span>
            {error}
          </div>
        )}

        {/* Loading state */}
        {isComputing && (
          <div className="comparison-loading">
            <span className="spinner" />
            Computing comparison...
          </div>
        )}

        {/* Empty state */}
        {!comparison && !error && !isComputing && (
          <div className="comparison-empty">
            <div className="empty-icon">&#x2696;</div>
            <h3>Select Two Sessions to Compare</h3>
            <p>Choose sessions from the dropdowns above to see their attention pattern differences.</p>
            <p className="empty-hint">
              Save traces from the Chat view to add them to the comparison list.
            </p>
          </div>
        )}

        {/* Comparison Results */}
        {comparison && !isComputing && (
          <div className="comparison-results">
            {/* Summary Section */}
            <div className="comparison-summary">
              <h3>Summary</h3>
              <div className="summary-score">
                <span className="score-value">
                  {Math.round(comparison.overallSimilarity * 100)}%
                </span>
                <span className="score-label">Similar</span>
              </div>
              <div className="summary-differences">
                <h4>Key Differences:</h4>
                <ul>
                  {comparison.keyDifferences.map((diff, i) => (
                    <li key={i}>{diff}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Zone Comparison */}
            <div className="comparison-section">
              <h3>Manifold Zone</h3>
              <ZoneBadgeComparison
                leftZone={comparison.zoneComparison.leftZone}
                rightZone={comparison.zoneComparison.rightZone}
                leftConfidence={comparison.zoneComparison.leftConfidence}
                rightConfidence={comparison.zoneComparison.rightConfidence}
                zoneChanged={comparison.zoneComparison.zoneChanged}
              />
            </div>

            {/* Metrics Comparison */}
            <div className="comparison-section">
              <h3>Metrics Comparison</h3>
              <div className="metrics-grid">
                <MetricsDiffCard
                  label="Entropy"
                  leftValue={
                    (comparison.fingerprintDiff.pcaDiff[1] ?? 0) * -1 + 0.5
                  }
                  rightValue={0.5}
                  diff={comparison.metricsDiff.entropyDiff}
                  description="Attention distribution spread"
                />
                <MetricsDiffCard
                  label="Local Mass"
                  leftValue={0.5 - comparison.metricsDiff.localMassDiff / 2}
                  rightValue={0.5 + comparison.metricsDiff.localMassDiff / 2}
                  diff={comparison.metricsDiff.localMassDiff}
                  description="Attention on nearby tokens"
                />
                <MetricsDiffCard
                  label="Mid Mass"
                  leftValue={0.5 - comparison.metricsDiff.midMassDiff / 2}
                  rightValue={0.5 + comparison.metricsDiff.midMassDiff / 2}
                  diff={comparison.metricsDiff.midMassDiff}
                  description="Paragraph-level attention"
                />
                <MetricsDiffCard
                  label="Long Mass"
                  leftValue={0.5 - comparison.metricsDiff.longMassDiff / 2}
                  rightValue={0.5 + comparison.metricsDiff.longMassDiff / 2}
                  diff={comparison.metricsDiff.longMassDiff}
                  description="Document-level attention"
                />
              </div>
            </div>

            {/* Histogram Comparison */}
            <div className="comparison-section">
              <h3>Distance Distribution</h3>
              <FingerprintDiffChart
                histogramDiff={comparison.fingerprintDiff.histogramDiff}
                cosineSimilarity={comparison.fingerprintDiff.cosineSimilarity}
              />
            </div>

            {/* PCA Projection */}
            <div className="comparison-section">
              <h3>PCA Interpretation</h3>
              <div className="pca-comparison">
                <div className="pca-row">
                  <span className="pca-label">PC1 (Local vs Long)</span>
                  <span
                    className={`pca-diff ${
                      comparison.fingerprintDiff.pcaDiff[0] > 0
                        ? 'positive'
                        : 'negative'
                    }`}
                  >
                    {comparison.fingerprintDiff.pcaDiff[0] > 0 ? '+' : ''}
                    {comparison.fingerprintDiff.pcaDiff[0]?.toFixed(2) ?? '0.00'}
                  </span>
                  <span className="pca-desc">
                    {comparison.fingerprintDiff.pcaDiff[0] > 0.1
                      ? 'B is more local-focused'
                      : comparison.fingerprintDiff.pcaDiff[0] < -0.1
                      ? 'B is more long-range'
                      : 'Similar local/long balance'}
                  </span>
                </div>
                <div className="pca-row">
                  <span className="pca-label">PC2 (Focused vs Diffuse)</span>
                  <span
                    className={`pca-diff ${
                      comparison.fingerprintDiff.pcaDiff[1] > 0
                        ? 'positive'
                        : 'negative'
                    }`}
                  >
                    {comparison.fingerprintDiff.pcaDiff[1] > 0 ? '+' : ''}
                    {comparison.fingerprintDiff.pcaDiff[1]?.toFixed(2) ?? '0.00'}
                  </span>
                  <span className="pca-desc">
                    {comparison.fingerprintDiff.pcaDiff[1] > 0.1
                      ? 'B is more focused'
                      : comparison.fingerprintDiff.pcaDiff[1] < -0.1
                      ? 'B is more diffuse'
                      : 'Similar focus level'}
                  </span>
                </div>
              </div>
            </div>

            {/* Clear button */}
            <button className="clear-comparison-btn" onClick={clearComparison}>
              Clear Comparison
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
