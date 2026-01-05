import { useSessionStore } from '../../stores/useSessionStore';
import { useUIStore } from '../../stores/useUIStore';
import { KPICard } from './KPICard';
import { TopKList } from './TopKList';
import { getTopKForLayer, isRawMode } from '../../api/types';

export function InsightPanel() {
  const fingerprint = useSessionStore((state) => state.fingerprint);
  const messages = useSessionStore((state) => state.messages);
  const selectedTokenIndex = useUIStore((state) => state.selectedTokenIndex);
  const selectedLayerId = useUIStore((state) => state.selectedLayerId);

  // Get attention data for selected token
  const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
  const selectedAttention =
    selectedTokenIndex !== null ? lastAssistant?.attention?.[selectedTokenIndex] : null;

  // Get top-k for selected layer
  const layerId = selectedLayerId === -1 ? 31 : selectedLayerId; // Default to last layer
  const topK = selectedAttention ? getTopKForLayer(selectedAttention, layerId, 5) : [];

  const topkMass =
    selectedAttention && isRawMode(selectedAttention) ? selectedAttention.topk_mass ?? 0 : 0;

  const manifoldZone = fingerprint
    ? fingerprint.local_mass > 0.5
      ? 'syntax_floor'
      : fingerprint.mid_mass > 0.5
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

        {/* Fingerprint Metrics */}
        {fingerprint && (
          <div className="section">
            <div className="section-header">
              <span>Fingerprint</span>
              <span className="badge">20D → UMAP</span>
            </div>
            <div className="metric-chips">
              <div className="chip">
                <strong>Hubness</strong> {(fingerprint.hubness ?? 0).toFixed(2)}
              </div>
              <div className="chip">
                <strong>Entropy</strong> {fingerprint.entropy.toFixed(2)}
              </div>
              <div className="chip">
                <strong>Local</strong> {(fingerprint.local_mass * 100).toFixed(0)}%
              </div>
              <div className="chip">
                <strong>Mid</strong> {(fingerprint.mid_mass * 100).toFixed(0)}%
              </div>
              <div className="chip">
                <strong>Long</strong> {(fingerprint.long_mass * 100).toFixed(0)}%
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
