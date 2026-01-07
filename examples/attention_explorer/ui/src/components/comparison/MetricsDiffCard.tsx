interface MetricsDiffCardProps {
  label: string;
  leftValue: number;
  rightValue: number;
  diff: number;
  description: string;
}

export function MetricsDiffCard({
  label,
  leftValue,
  rightValue,
  diff,
  description,
}: MetricsDiffCardProps) {
  const absDiff = Math.abs(diff);
  const percentChange = Math.round(diff * 100);
  const isSignificant = absDiff > 0.05;

  // Determine color based on direction and significance
  const getDiffColor = () => {
    if (!isSignificant) return '#888';
    return diff > 0 ? '#55d6a6' : '#ff7b72';
  };

  const getArrowSymbol = () => {
    if (!isSignificant) return '';
    return diff > 0 ? ' \u2191' : ' \u2193';
  };

  // Normalize values for bar display (0-100%)
  const leftBar = Math.max(0, Math.min(100, leftValue * 100));
  const rightBar = Math.max(0, Math.min(100, rightValue * 100));

  return (
    <div className="metrics-diff-card">
      <div className="diff-card-header">
        <span className="diff-card-label">{label}</span>
        <span
          className="diff-card-delta"
          style={{ color: getDiffColor() }}
        >
          {isSignificant ? (
            <>
              {percentChange > 0 ? '+' : ''}
              {percentChange}%{getArrowSymbol()}
            </>
          ) : (
            <span style={{ opacity: 0.6 }}>~same</span>
          )}
        </span>
      </div>

      <div className="diff-card-bars">
        <div className="bar-row">
          <span className="bar-label">A</span>
          <div className="bar-track">
            <div
              className="bar-fill bar-a"
              style={{ width: `${leftBar}%` }}
            />
          </div>
          <span className="bar-value">{leftValue.toFixed(2)}</span>
        </div>
        <div className="bar-row">
          <span className="bar-label">B</span>
          <div className="bar-track">
            <div
              className="bar-fill bar-b"
              style={{ width: `${rightBar}%` }}
            />
          </div>
          <span className="bar-value">{rightValue.toFixed(2)}</span>
        </div>
      </div>

      <div className="diff-card-description">{description}</div>
    </div>
  );
}
