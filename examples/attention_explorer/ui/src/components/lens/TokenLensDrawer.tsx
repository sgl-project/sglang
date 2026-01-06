import { useEffect, useCallback } from 'react';
import { useUIStore, DrawerTab } from '../../stores/useUIStore';
import { useSessionStore } from '../../stores/useSessionStore';
import {
  AttentionEntry,
  isRawMode,
  isFingerprintMode,
  getTopKForLayer,
  extractFingerprint,
  MoERoutingEntry,
} from '../../api/types';
import { DistanceHistogram } from './DistanceHistogram';
import { FFTSpectrum } from './FFTSpectrum';

interface TopKItem {
  position: number;
  score: number;
  normalizedScore: number; // Score normalized excluding sink token
  distance: number;
  text?: string;
  segment?: 'prompt' | 'think' | 'output';
  isSinkToken: boolean;
}

function getDistanceLabel(distance: number): string {
  if (distance <= 2) return 'local';
  if (distance <= 7) return 'mid';
  return 'long';
}

function getDistanceColor(distance: number): string {
  if (distance <= 2) return 'var(--accent)';
  if (distance <= 7) return 'var(--secondary)';
  return 'var(--warn)';
}

// Extract top-k with distance info and sink token handling
// The "sink token" (position 0, usually BOS/system prompt) often absorbs
// 50-90% of attention mass, making other connections invisible.
// We normalize scores excluding the sink to make semantic connections visible.
function getTopKWithDistance(
  entry: AttentionEntry | null,
  tokenIndex: number,
  layerId: number,
  tokens: string[],
  k: number = 8
): TopKItem[] {
  if (!entry) return [];

  const topK = getTopKForLayer(entry, layerId, k);

  // Identify sink token (position 0) and compute normalization
  // Sink token typically absorbs 50-90% of attention, making other connections invisible
  const nonSinkTotal = topK
    .filter((item) => item.position !== 0)
    .reduce((sum, item) => sum + item.score, 0);

  return topK.map((item) => {
    const isSinkToken = item.position === 0;
    // Normalize non-sink tokens relative to each other
    // This makes semantic connections visible even when sink absorbs most attention
    const normalizedScore = isSinkToken
      ? item.score
      : nonSinkTotal > 0.01
      ? item.score / nonSinkTotal
      : item.score;

    return {
      position: item.position,
      score: item.score,
      normalizedScore,
      distance: Math.abs(tokenIndex - item.position),
      text: tokens[item.position] || `[${item.position}]`,
      segment: item.position < tokens.length ? 'output' : 'prompt',
      isSinkToken,
    };
  });
}

// Links Tab - shows top-k attended tokens
function LinksTab({
  attention,
  tokenIndex,
  layerId,
  tokens,
}: {
  attention: AttentionEntry | null;
  tokenIndex: number;
  layerId: number;
  tokens: string[];
}) {
  const topK = getTopKWithDistance(attention, tokenIndex, layerId, tokens);

  if (topK.length === 0) {
    const isFingerprint = attention ? isFingerprintMode(attention) : false;
    return (
      <div className="drawer-empty">
        <div className="drawer-empty-icon">🔗</div>
        <div>No attention links available</div>
        <div className="hint-text">
          {isFingerprint
            ? 'Server is in fingerprint mode - no token-level links'
            : 'Enable attention capture on the server'}
        </div>
      </div>
    );
  }

  // Separate sink token, local hits, and remote hits
  const sinkToken = topK.find((t) => t.isSinkToken);
  const localHits = topK.filter((t) => !t.isSinkToken && t.distance <= 7);
  const remoteHits = topK.filter((t) => !t.isSinkToken && t.distance > 7);

  return (
    <div className="drawer-links">
      {/* Sink token warning - shown if sink absorbs significant attention */}
      {sinkToken && sinkToken.score > 0.3 && (
        <div className="sink-warning">
          <span className="sink-icon">🕳️</span>
          <span>
            Sink token (pos 0) absorbs {(sinkToken.score * 100).toFixed(0)}% of
            attention
          </span>
        </div>
      )}

      {localHits.length > 0 && (
        <div className="links-section">
          <div className="links-section-header">
            <span>Nearby tokens</span>
            <span className="badge">{localHits.length}</span>
          </div>
          <div className="topk-list">
            {localHits.map((item, i) => (
              <div key={i} className="topk-row">
                <div className="topk-left">
                  <span className="topk-token">{item.text}</span>
                  <span className="topk-hint">
                    pos {item.position} · {getDistanceLabel(item.distance)} (
                    {item.distance})
                  </span>
                </div>
                <div
                  className="topk-right"
                  style={{ color: getDistanceColor(item.distance) }}
                >
                  <span title={`Raw: ${(item.score * 100).toFixed(1)}%`}>
                    {(item.normalizedScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {remoteHits.length > 0 && (
        <div className="links-section">
          <div className="links-section-header">
            <span>Remote hits</span>
            <span className="badge">{remoteHits.length}</span>
          </div>
          <div className="topk-list remote-hits">
            {remoteHits.map((item, i) => (
              <div key={i} className="topk-row remote">
                <div className="topk-left">
                  <span className="topk-token">{item.text}</span>
                  <span className="topk-hint">
                    pos {item.position} · long ({item.distance} away)
                  </span>
                </div>
                <div className="topk-right" style={{ color: 'var(--warn)' }}>
                  <span title={`Raw: ${(item.score * 100).toFixed(1)}%`}>
                    {(item.normalizedScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Signal Tab - shows hubness/entropy/consensus metrics
function SignalTab({ attention }: { attention: AttentionEntry | null }) {
  const fingerprint = attention ? extractFingerprint(attention) : null;

  if (!fingerprint && !attention) {
    return (
      <div className="drawer-empty">
        <div className="drawer-empty-icon">📊</div>
        <div>No signal data available</div>
      </div>
    );
  }

  // Get mode-specific data
  let topkMass = 0;
  let entropy = fingerprint?.entropy ?? 0;
  let localMass = fingerprint?.local_mass ?? 0;
  let midMass = fingerprint?.mid_mass ?? 0;
  let longMass = fingerprint?.long_mass ?? 0;

  if (attention && isRawMode(attention)) {
    topkMass = attention.topk_mass ?? 0;
  }

  // Compute zone
  const zone =
    localMass > 0.5
      ? 'Syntax Floor'
      : midMass > 0.5
      ? 'Semantic Bridge'
      : longMass > 0.3
      ? 'Long Range'
      : 'Diffuse';

  return (
    <div className="drawer-signal">
      <div className="signal-zone">
        <div className="signal-zone-label">Manifold Zone</div>
        <div className="signal-zone-value">{zone}</div>
      </div>

      <div className="signal-metrics">
        <div className="signal-metric">
          <div className="signal-metric-label">Entropy</div>
          <div className="signal-metric-value">{entropy.toFixed(2)}</div>
        </div>
        {topkMass > 0 && (
          <div className="signal-metric">
            <div className="signal-metric-label">Top-K Mass</div>
            <div className="signal-metric-value">
              {(topkMass * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>

      <div className="signal-distribution">
        <div className="signal-dist-header">Attention Distribution</div>
        <div className="signal-dist-bars">
          <div
            className="signal-bar local"
            style={{ flex: localMass }}
            title={`Local: ${(localMass * 100).toFixed(0)}%`}
          />
          <div
            className="signal-bar mid"
            style={{ flex: midMass }}
            title={`Mid: ${(midMass * 100).toFixed(0)}%`}
          />
          <div
            className="signal-bar long"
            style={{ flex: longMass || 0.01 }}
            title={`Long: ${(longMass * 100).toFixed(0)}%`}
          />
        </div>
        <div className="signal-dist-labels">
          <span>Local {(localMass * 100).toFixed(0)}%</span>
          <span>Mid {(midMass * 100).toFixed(0)}%</span>
          <span>Long {(longMass * 100).toFixed(0)}%</span>
        </div>
      </div>

      {fingerprint?.histogram && fingerprint.histogram.length > 0 && (
        <div className="signal-spectral">
          <div className="signal-spectral-section">
            <div className="signal-spectral-header">Distance Histogram</div>
            <DistanceHistogram histogram={fingerprint.histogram} />
          </div>
          <div className="signal-spectral-section">
            <div className="signal-spectral-header">Frequency Spectrum</div>
            <FFTSpectrum histogram={fingerprint.histogram} />
          </div>
        </div>
      )}
    </div>
  );
}

// MoE Tab - shows expert routing info
function MoETab({ moe }: { moe: MoERoutingEntry | null }) {
  if (!moe) {
    return (
      <div className="drawer-empty">
        <div className="drawer-empty-icon">🔀</div>
        <div>No MoE routing data</div>
        <div className="hint-text">
          Enable return_moe_routing in capture params
        </div>
      </div>
    );
  }

  const layers = Object.entries(moe.layers).slice(0, 4); // Show first 4 layers

  return (
    <div className="drawer-moe">
      <div className="moe-summary">
        <div className="moe-metric">
          <span className="moe-metric-label">Avg Entropy</span>
          <span className="moe-metric-value">
            {(moe.entropy_mean ?? 0).toFixed(2)}
          </span>
        </div>
        {moe.expert_churn !== undefined && (
          <div className="moe-metric">
            <span className="moe-metric-label">Expert Churn</span>
            <span className="moe-metric-value">
              {(moe.expert_churn * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      <div className="moe-layers">
        {layers.map(([layerId, routing]) => (
          <div key={layerId} className="moe-layer">
            <div className="moe-layer-header">
              <span>Layer {layerId}</span>
              <span className="badge">H={routing.entropy.toFixed(2)}</span>
            </div>
            <div className="moe-experts">
              {routing.expert_ids.slice(0, 4).map((expertId, i) => (
                <div key={i} className="moe-expert">
                  <span className="moe-expert-id">E{expertId}</span>
                  <span className="moe-expert-weight">
                    {(routing.expert_weights[i] * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TokenLensDrawer() {
  const drawerState = useUIStore((s) => s.drawerState);
  const drawerTokenIndex = useUIStore((s) => s.drawerTokenIndex);
  const drawerTab = useUIStore((s) => s.drawerTab);
  const selectedLayerId = useUIStore((s) => s.selectedLayerId);
  const setDrawerTab = useUIStore((s) => s.setDrawerTab);
  const unpinDrawer = useUIStore((s) => s.unpinDrawer);
  const clearHoverTimeout = useUIStore((s) => s.clearHoverTimeout);

  const messages = useSessionStore((s) => s.messages);

  // Get attention data for the drawer token
  const lastAssistant = messages.filter((m) => m.role === 'assistant').pop();
  const attention =
    drawerTokenIndex !== null
      ? lastAssistant?.attention?.[drawerTokenIndex] ?? null
      : null;
  const moe =
    drawerTokenIndex !== null
      ? lastAssistant?.moe?.[drawerTokenIndex] ?? null
      : null;
  const tokens = lastAssistant?.tokens ?? [];

  // Effective layer ID
  const layerId =
    selectedLayerId !== -1
      ? selectedLayerId
      : attention && 'layer_id' in attention
      ? (attention as any).layer_id
      : 31;

  // Handle escape key to close drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && drawerState === 'pinned') {
        unpinDrawer();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [drawerState, unpinDrawer]);

  // Keep drawer open when hovering over it
  const handleMouseEnter = useCallback(() => {
    clearHoverTimeout();
  }, [clearHoverTimeout]);

  const handleMouseLeave = useCallback(() => {
    // Let the token's mouse leave handler deal with closing
  }, []);

  if (drawerState === 'closed' || drawerTokenIndex === null) {
    return null;
  }

  const tabs: { id: DrawerTab; label: string; icon: string }[] = [
    { id: 'links', label: 'Links', icon: '🔗' },
    { id: 'signal', label: 'Signal', icon: '📊' },
    { id: 'moe', label: 'MoE', icon: '🔀' },
  ];

  return (
    <div
      className={`token-lens-drawer ${drawerState}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="drawer-header">
        <div className="drawer-title">
          <span className="drawer-icon">🔍</span>
          <span>Token #{drawerTokenIndex}</span>
          {drawerState === 'pinned' && (
            <span className="drawer-pin-badge">pinned</span>
          )}
        </div>
        {drawerState === 'pinned' && (
          <button className="drawer-close" onClick={unpinDrawer}>
            ×
          </button>
        )}
      </div>

      <div className="drawer-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`drawer-tab ${drawerTab === tab.id ? 'active' : ''}`}
            onClick={() => setDrawerTab(tab.id)}
          >
            <span className="drawer-tab-icon">{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="drawer-content">
        {drawerTab === 'links' && (
          <LinksTab
            attention={attention}
            tokenIndex={drawerTokenIndex}
            layerId={layerId}
            tokens={tokens}
          />
        )}
        {drawerTab === 'signal' && <SignalTab attention={attention} />}
        {drawerTab === 'moe' && <MoETab moe={moe} />}
      </div>

      <div className="drawer-footer">
        <span className="hint-text">
          {drawerState === 'hovering'
            ? 'Click token to pin'
            : 'Press Esc to close'}
        </span>
      </div>
    </div>
  );
}
