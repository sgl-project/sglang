import { useSessionStore } from '../../stores/useSessionStore';
import { useUIStore } from '../../stores/useUIStore';
import { KPICard } from './KPICard';
import { TopKList } from './TopKList';
import { DistanceHistogram } from './DistanceHistogram';
import { FFTSpectrum } from './FFTSpectrum';
import { getTopKForLayer, isRawMode, isFingerprintMode, isSketchMode, AttentionEntry, extractFingerprint } from '../../api/types';

// Extract fingerprint metrics from any attention entry
function getMetricsFromAttention(entry: AttentionEntry | null): {
  localMass: number;
  midMass: number;
  longMass: number;
  entropy: number;
  mode: string;
} | null {
  if (!entry) return null;

  if (isFingerprintMode(entry)) {
    const fp = entry.fingerprint;
    return {
      localMass: fp[16] ?? fp[0] ?? 0,
      midMass: fp[17] ?? fp[1] ?? 0,
      longMass: fp[18] ?? fp[2] ?? 0,
      entropy: fp[19] ?? fp[3] ?? 0,
      mode: 'fingerprint',
    };
  }

  if (isSketchMode(entry)) {
    const sketch = entry.sketch || Object.values(entry.layer_sketches || {})[0];
    if (sketch) {
      return {
        localMass: sketch.local_mass,
        midMass: sketch.mid_mass,
        longMass: sketch.long_mass,
        entropy: sketch.entropy,
        mode: 'sketch',
      };
    }
  }

  if (isRawMode(entry)) {
    return {
      localMass: 0,
      midMass: 0,
      longMass: 0,
      entropy: 0,
      mode: 'raw',
    };
  }

  return null;
}

export function InsightPanel() {
  const fingerprint = useSessionStore((state) => state.fingerprint);
  const messages = useSessionStore((state) => state.messages);
  const selectedTokenIndex = useUIStore((state) => state.selectedTokenIndex);
  const selectedLayerId = useUIStore((state) => state.selectedLayerId);

  // Get attention data for selected token from stored messages
  const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
  const selectedAttention =
    selectedTokenIndex !== null ? (lastAssistant?.attention?.[selectedTokenIndex] ?? null) : null;

  // Extract metrics from the stored attention data
  const tokenMetrics = getMetricsFromAttention(selectedAttention);

  // Get top-k from attention entry (only available in raw mode)
  // If layerId is -1 (auto), try common last-layer IDs or use entry's layer_id
  const getEffectiveLayerId = (): number => {
    if (selectedLayerId !== -1) return selectedLayerId;
    // Try to get layer_id from the attention entry itself
    if (selectedAttention && 'layer_id' in selectedAttention) {
      return (selectedAttention as any).layer_id;
    }
    // Fallback: try common last-layer positions (31 for 32-layer, 27 for 28-layer models)
    return 31;
  };
  const layerId = getEffectiveLayerId();
  const topK = selectedAttention ? getTopKForLayer(selectedAttention, layerId, 5) : [];

  // Use token-level metrics if available, fallback to session fingerprint
  const displayMetrics = tokenMetrics || (fingerprint ? {
    localMass: fingerprint.local_mass,
    midMass: fingerprint.mid_mass,
    longMass: fingerprint.long_mass,
    entropy: fingerprint.entropy,
    mode: 'session',
  } : null);

  const manifoldZone = displayMetrics
    ? displayMetrics.localMass > 0.5
      ? 'syntax_floor'
      : displayMetrics.midMass > 0.5
      ? 'semantic_bridge'
      : 'diffuse'
    : 'unknown';

  return (
    <div className="card insight-panel">
      <div className="card-header">
        <div className="card-title">
          <span>Insight Lens</span>
          <span className="subtitle">what the model "looked at"</span>
        </div>
        <span className="badge">
          {selectedTokenIndex !== null ? `Token #${selectedTokenIndex}` : 'No token selected'}
        </span>
      </div>
      <div className="card-content">
        {/* KPI Row */}
        <div className="kpi-row">
          <KPICard
            label="Manifold Zone"
            value={manifoldZone.replace('_', ' ')}
            hint="From fingerprint (distance histogram + hubness)."
          />
          <KPICard
            label="Consensus"
            value={fingerprint ? `${((fingerprint.consensus ?? 0) * 100).toFixed(0)}%` : '-'}
            hint="How often different layers agree on the same anchors."
            progress={fingerprint?.consensus}
          />
        </div>

        {/* Token Lens */}
        <div className="section">
          <div className="section-header">
            <span>Token Lens</span>
            <span className="badge">
              {selectedTokenIndex !== null ? 'active' : 'idle'}
            </span>
          </div>

          {selectedTokenIndex !== null && topK.length > 0 ? (
            <>
              <div className="metric-row" style={{ marginBottom: '8px' }}>
                <span className="badge" style={{ background: 'rgba(85, 214, 166, 0.15)' }}>
                  {tokenMetrics?.mode || 'raw'}
                </span>
                <span className="badge">{topK.length} anchors</span>
              </div>
              <div className="hint-text" style={{ marginBottom: '8px' }}>
                <strong>Top attended tokens</strong>:
              </div>
              <TopKList items={topK} />
            </>
          ) : selectedTokenIndex !== null && tokenMetrics ? (
            <div className="hint-text" style={{ padding: '12px', textAlign: 'center' }}>
              <div style={{ fontSize: '20px', marginBottom: '8px', opacity: 0.7 }}>📊</div>
              <div>Server is in <strong>fingerprint mode</strong>.</div>
              <div style={{ fontSize: '11px', marginTop: '8px', opacity: 0.7 }}>
                Attention metrics are shown below. For detailed token-level anchors,
                restart the server without --attention-fingerprint-mode.
              </div>
            </div>
          ) : selectedTokenIndex !== null ? (
            <div className="hint-text" style={{ padding: '12px', textAlign: 'center' }}>
              <div style={{ fontSize: '20px', marginBottom: '8px', opacity: 0.5 }}>🔍</div>
              <div>No attention data for this token.</div>
              <div style={{ fontSize: '11px', marginTop: '4px', opacity: 0.7 }}>
                Make sure attention capture is enabled on the server.
              </div>
            </div>
          ) : (
            <div className="hint-text" style={{ padding: '12px', textAlign: 'center' }}>
              <div style={{ fontSize: '20px', marginBottom: '8px', opacity: 0.5 }}>👆</div>
              <div>Click any <strong>assistant token</strong> to see its attention.</div>
            </div>
          )}
        </div>

        {/* Token-level Fingerprint Metrics */}
        {displayMetrics && (
          <div className="section">
            <div className="section-header">
              <span>Attention Profile</span>
              <span className="badge">{displayMetrics.mode}</span>
            </div>
            <div className="metric-chips">
              <div className="chip">
                <strong>Entropy</strong> {displayMetrics.entropy.toFixed(2)}
              </div>
              <div className="chip">
                <strong>Local</strong> {(displayMetrics.localMass * 100).toFixed(0)}%
              </div>
              <div className="chip">
                <strong>Mid</strong> {(displayMetrics.midMass * 100).toFixed(0)}%
              </div>
              <div className="chip">
                <strong>Long</strong> {(displayMetrics.longMass * 100).toFixed(0)}%
              </div>
            </div>
            {/* Visual bar for attention distribution */}
            <div style={{ display: 'flex', gap: '2px', marginTop: '8px', height: '24px' }}>
              <div style={{
                flex: displayMetrics.localMass,
                background: 'rgba(122, 162, 255, 0.6)',
                borderRadius: '4px 0 0 4px',
                minWidth: displayMetrics.localMass > 0.01 ? '4px' : '0'
              }} title={`Local: ${(displayMetrics.localMass * 100).toFixed(0)}%`} />
              <div style={{
                flex: displayMetrics.midMass,
                background: 'rgba(85, 214, 166, 0.6)',
                minWidth: displayMetrics.midMass > 0.01 ? '4px' : '0'
              }} title={`Mid: ${(displayMetrics.midMass * 100).toFixed(0)}%`} />
              <div style={{
                flex: displayMetrics.longMass || 0.01,
                background: 'rgba(255, 204, 102, 0.6)',
                borderRadius: '0 4px 4px 0',
                minWidth: displayMetrics.longMass > 0.01 ? '4px' : '0'
              }} title={`Long: ${(displayMetrics.longMass * 100).toFixed(0)}%`} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--muted)', marginTop: '4px' }}>
              <span>Local (0-2)</span>
              <span>Mid (3-7)</span>
              <span>Long (8+)</span>
            </div>
          </div>
        )}

        {/* Distance & Rhythm Section */}
        {selectedAttention && (() => {
          const fp = extractFingerprint(selectedAttention);
          if (!fp?.histogram || fp.histogram.length === 0) return null;
          return (
            <div className="section">
              <div className="section-header">
                <span>Distance & Rhythm</span>
                <span className="badge">spectral</span>
              </div>
              <div className="distance-rhythm-container">
                <div className="distance-section">
                  <div className="subsection-label">Distance Distribution</div>
                  <DistanceHistogram histogram={fp.histogram} compact />
                </div>
                <div className="rhythm-section">
                  <div className="subsection-label">Frequency Spectrum</div>
                  <FFTSpectrum histogram={fp.histogram} compact />
                </div>
              </div>
            </div>
          );
        })()}

        {/* Session-level Fingerprint (from sidecar) */}
        {fingerprint && displayMetrics?.mode !== 'session' && (
          <div className="section">
            <div className="section-header">
              <span>Session Fingerprint</span>
              <span className="badge">sidecar</span>
            </div>
            <div className="metric-chips">
              <div className="chip">
                <strong>Hubness</strong> {(fingerprint.hubness ?? 0).toFixed(2)}
              </div>
              <div className="chip">
                <strong>Consensus</strong> {(fingerprint.consensus ?? 0).toFixed(2)}
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="card-footer hint-text">
        This is a live UI. Data comes from SGLang attention capture.
      </div>
    </div>
  );
}
