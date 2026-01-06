import { useMemo } from 'react';

interface DistanceHistogramProps {
  histogram: number[];
  onBinClick?: (binIndex: number) => void;
  selectedBin?: number | null;
  compact?: boolean;
}

// Log-binned distance ranges (typical for attention patterns)
const BIN_LABELS = [
  '0', '1', '2', '3-4', '5-7', '8-15', '16-31', '32-63',
  '64-127', '128-255', '256-511', '512-1K', '1K-2K', '2K-4K', '4K-8K', '8K+'
];

const BIN_ZONES: Array<'local' | 'mid' | 'long'> = [
  'local', 'local', 'local',  // 0-2
  'mid', 'mid', 'mid',        // 3-15
  'long', 'long', 'long', 'long', 'long', 'long', 'long', 'long', 'long', 'long'  // 16+
];

function getZoneColor(zone: 'local' | 'mid' | 'long'): string {
  switch (zone) {
    case 'local': return 'var(--accent)';
    case 'mid': return 'var(--secondary)';
    case 'long': return 'var(--warn)';
  }
}

export function DistanceHistogram({
  histogram,
  onBinClick,
  selectedBin,
  compact = false,
}: DistanceHistogramProps) {
  // Normalize histogram values
  const normalizedHist = useMemo(() => {
    const max = Math.max(...histogram, 0.01);
    return histogram.map(v => v / max);
  }, [histogram]);

  // Find dominant zone
  const dominantZone = useMemo(() => {
    const zoneMass = { local: 0, mid: 0, long: 0 };
    histogram.forEach((v, i) => {
      const zone = BIN_ZONES[i] || 'long';
      zoneMass[zone] += v;
    });
    const total = zoneMass.local + zoneMass.mid + zoneMass.long;
    if (total === 0) return null;

    if (zoneMass.local / total > 0.5) return 'local';
    if (zoneMass.mid / total > 0.5) return 'mid';
    if (zoneMass.long / total > 0.3) return 'long';
    return 'diffuse';
  }, [histogram]);

  if (histogram.length === 0) {
    return (
      <div className="distance-histogram empty">
        <span className="hint-text">No histogram data</span>
      </div>
    );
  }

  return (
    <div className={`distance-histogram ${compact ? 'compact' : ''}`}>
      <div className="histogram-bars">
        {normalizedHist.slice(0, 16).map((value, i) => {
          const zone = BIN_ZONES[i] || 'long';
          const isSelected = selectedBin === i;
          const actualValue = histogram[i];

          return (
            <div
              key={i}
              className={`histogram-bar-container ${isSelected ? 'selected' : ''}`}
              onClick={() => onBinClick?.(i)}
              title={`${BIN_LABELS[i]}: ${(actualValue * 100).toFixed(1)}%`}
            >
              <div
                className="histogram-bar"
                style={{
                  height: `${Math.max(value * 100, 2)}%`,
                  backgroundColor: getZoneColor(zone),
                  opacity: isSelected ? 1 : 0.7,
                }}
              />
            </div>
          );
        })}
      </div>

      {!compact && (
        <div className="histogram-axis">
          <span>0</span>
          <span className="zone-marker local">Local</span>
          <span className="zone-marker mid">Mid</span>
          <span className="zone-marker long">Long</span>
        </div>
      )}

      {dominantZone && !compact && (
        <div className="histogram-summary">
          <span className={`zone-badge ${dominantZone}`}>
            {dominantZone === 'diffuse' ? 'Diffuse' : `${dominantZone.charAt(0).toUpperCase() + dominantZone.slice(1)} Dominant`}
          </span>
        </div>
      )}
    </div>
  );
}
