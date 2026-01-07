interface FingerprintDiffChartProps {
  histogramDiff: number[];
  cosineSimilarity: number;
}

// Distance labels for histogram bins (log2 scale)
const BIN_LABELS = [
  '1', '2', '4', '8', '16', '32', '64', '128',
  '256', '512', '1K', '2K', '4K', '8K', '16K', '32K+'
];

export function FingerprintDiffChart({
  histogramDiff,
  cosineSimilarity,
}: FingerprintDiffChartProps) {
  // Find max absolute diff for scaling
  const maxAbsDiff = Math.max(0.1, ...histogramDiff.map(Math.abs));

  // Scale factor for bar height (max bar = 100%)
  const scale = 100 / maxAbsDiff;

  return (
    <div className="fingerprint-diff-chart">
      <div className="similarity-badge">
        <span className="similarity-label">Cosine Similarity</span>
        <span
          className="similarity-value"
          style={{
            color:
              cosineSimilarity > 0.8
                ? '#55d6a6'
                : cosineSimilarity > 0.5
                ? '#ffcc66'
                : '#ff7b72',
          }}
        >
          {(cosineSimilarity * 100).toFixed(0)}%
        </span>
      </div>

      <div className="histogram-diff-container">
        {/* Y-axis label */}
        <div className="y-axis">
          <span className="y-label positive">B higher</span>
          <span className="y-label neutral">equal</span>
          <span className="y-label negative">A higher</span>
        </div>

        {/* Bars */}
        <div className="histogram-bars">
          {histogramDiff.map((diff, i) => {
            const height = Math.abs(diff) * scale;
            const isPositive = diff >= 0;

            return (
              <div key={i} className="histogram-bar-wrapper">
                <div className="bar-column">
                  {/* Positive region (B higher) */}
                  <div className="bar-region positive">
                    {isPositive && (
                      <div
                        className="diff-bar positive"
                        style={{ height: `${height}%` }}
                        title={`Bin ${i}: B is ${(diff * 100).toFixed(1)}% higher`}
                      />
                    )}
                  </div>

                  {/* Zero line */}
                  <div className="zero-line" />

                  {/* Negative region (A higher) */}
                  <div className="bar-region negative">
                    {!isPositive && (
                      <div
                        className="diff-bar negative"
                        style={{ height: `${height}%` }}
                        title={`Bin ${i}: A is ${(-diff * 100).toFixed(1)}% higher`}
                      />
                    )}
                  </div>
                </div>

                {/* X-axis label */}
                <span className="bin-label">{BIN_LABELS[i]}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-color positive" /> B has more attention
        </span>
        <span className="legend-item">
          <span className="legend-color negative" /> A has more attention
        </span>
      </div>

      <div className="chart-hint">
        Distance histogram shows attention mass by token distance (log scale).
        Bars show the difference between sessions at each distance bin.
      </div>
    </div>
  );
}
