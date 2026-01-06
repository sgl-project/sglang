import { useSessionStore } from '../../stores/useSessionStore';
import { useUIStore } from '../../stores/useUIStore';
import { KPICard } from './KPICard';
import { TopKList } from './TopKList';
import { getTopKForLayer, isRawMode, isFingerprintMode, isSketchMode, extractFingerprint, AttentionEntry } from '../../api/types';

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

  // Get attention data for selected token
  const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
  const selectedAttention =
    selectedTokenIndex !== null ? lastAssistant?.attention?.[selectedTokenIndex] : null;

  // Extract metrics from selected attention
  const tokenMetrics = getMetricsFromAttention(selectedAttention);

  // Get top-k for selected layer (only works for raw/sketch modes)
  const layerId = selectedLayerId === -1 ? 31 : selectedLayerId;
  const topK = selectedAttention ? getTopKForLayer(selectedAttention, layerId, 5) : [];

  const topkMass =
    selectedAttention && isRawMode(selectedAttention) ? selectedAttention.topk_mass ?? 0 : 0;

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
            <span className="badge">{selectedTokenIndex !== null ? 'active' : 'idle'}</span>
          </div>
          <div className="hint-text">
            When you select an assistant token, we highlight its <strong>top-k attended context tokens</strong>.
            Low <strong>topk_mass</strong> indicates diffuse attention (top-k may miss signal).
          </div>

          {selectedTokenIndex !== null && topK.length > 0 ? (
            <>
              <div className="metric-row">
                <span className="badge">layer {layerId}</span>
                <span className="badge">topk_mass {topkMass.toFixed(2)}</span>
                <span className="badge">{topkMass < 0.55 ? 'diffuse' : 'captured'}</span>
              </div>
              <TopKList items={topK} />
            </>
          ) : (
            <div className="hint-text">Select any assistant token to populate the lens.</div>
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

        {/* Session-level Fingerprint (from sidecar) */}
        {fingerprint && fingerprint !== displayMetrics && (
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
