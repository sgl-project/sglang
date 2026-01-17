import { useMemo } from 'react';

interface FFTSpectrumProps {
  histogram: number[];
  compact?: boolean;
}

// Simple DFT for small arrays (histogram is typically 16 bins)
function computeDFT(signal: number[]): number[] {
  const N = signal.length;
  const magnitudes: number[] = [];

  // Only compute first half (Nyquist)
  for (let k = 0; k < Math.floor(N / 2); k++) {
    let real = 0;
    let imag = 0;

    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      real += signal[n] * Math.cos(angle);
      imag -= signal[n] * Math.sin(angle);
    }

    magnitudes.push(Math.sqrt(real * real + imag * imag) / N);
  }

  return magnitudes;
}

// Zone labels based on dominant frequency
function getZoneLabel(peakIndex: number, totalBins: number): string {
  const freq = peakIndex / totalBins;
  if (freq < 0.15) return 'Syntax Floor';
  if (freq < 0.35) return 'Semantic Bridge';
  if (freq < 0.6) return 'Structure Ripple';
  return 'High Frequency';
}

function getZoneDescription(peakIndex: number): string {
  if (peakIndex <= 1) return 'Smooth, local attention pattern';
  if (peakIndex <= 2) return 'Periodic semantic structure';
  if (peakIndex <= 4) return 'Recurring structural patterns';
  return 'High-frequency oscillations';
}

export function FFTSpectrum({ histogram, compact = false }: FFTSpectrumProps) {
  const { spectrum, peakIndex, lowEnergy, highEnergy } = useMemo(() => {
    if (histogram.length < 4) {
      return { spectrum: [], peakIndex: 0, lowEnergy: 0, highEnergy: 0 };
    }

    const fft = computeDFT(histogram);

    // Skip DC component (index 0)
    const acSpectrum = fft.slice(1);

    // Normalize
    const max = Math.max(...acSpectrum, 0.01);
    const normalized = acSpectrum.map(v => v / max);

    // Find peak (excluding DC)
    let peakIdx = 0;
    let peakVal = 0;
    for (let i = 0; i < normalized.length; i++) {
      if (normalized[i] > peakVal) {
        peakVal = normalized[i];
        peakIdx = i;
      }
    }

    // Compute low vs high energy ratio
    const midpoint = Math.floor(normalized.length / 2);
    const low = normalized.slice(0, midpoint).reduce((a, b) => a + b, 0);
    const high = normalized.slice(midpoint).reduce((a, b) => a + b, 0);
    const total = low + high || 1;

    return {
      spectrum: normalized,
      peakIndex: peakIdx + 1, // +1 because we skipped DC
      lowEnergy: low / total,
      highEnergy: high / total,
    };
  }, [histogram]);

  if (spectrum.length === 0) {
    return (
      <div className="fft-spectrum empty">
        <span className="hint-text">No frequency data</span>
      </div>
    );
  }

  const zoneLabel = getZoneLabel(peakIndex, spectrum.length);
  const zoneDesc = getZoneDescription(peakIndex);

  return (
    <div className={`fft-spectrum ${compact ? 'compact' : ''}`}>
      <div className="spectrum-bars">
        {spectrum.map((value, i) => {
          const isPeak = i === peakIndex - 1;

          return (
            <div
              key={i}
              className={`spectrum-bar-container ${isPeak ? 'peak' : ''}`}
              title={`Freq ${i + 1}: ${(value * 100).toFixed(0)}%`}
            >
              <div
                className="spectrum-bar"
                style={{
                  height: `${Math.max(value * 100, 2)}%`,
                  opacity: isPeak ? 1 : 0.6,
                }}
              />
              {isPeak && <div className="peak-marker">▼</div>}
            </div>
          );
        })}
      </div>

      {!compact && (
        <>
          <div className="spectrum-axis">
            <span>DC</span>
            <span>Low freq</span>
            <span>High freq</span>
          </div>

          <div className="spectrum-summary">
            <div className="zone-label">{zoneLabel}</div>
            <div className="zone-desc hint-text">{zoneDesc}</div>
          </div>

          <div className="energy-split">
            <div
              className="energy-bar low"
              style={{ flex: lowEnergy }}
              title={`Low energy: ${(lowEnergy * 100).toFixed(0)}%`}
            />
            <div
              className="energy-bar high"
              style={{ flex: highEnergy }}
              title={`High energy: ${(highEnergy * 100).toFixed(0)}%`}
            />
          </div>
          <div className="energy-labels">
            <span>Low {(lowEnergy * 100).toFixed(0)}%</span>
            <span>High {(highEnergy * 100).toFixed(0)}%</span>
          </div>
        </>
      )}
    </div>
  );
}
