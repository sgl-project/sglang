import { useCallback } from 'react';
import { ManifoldMap } from '../manifold/ManifoldMap';
import { ClusterList } from '../manifold/ClusterList';
import { useSessionStore } from '../../stores/useSessionStore';
import { useManifoldStore } from '../../stores/useManifoldStore';
import { useTraceStore } from '../../stores/useTraceStore';

// Panel showing selected session details
function SessionDetailsPanel() {
  const selectedPointId = useManifoldStore((state) => state.selectedPointId);
  const sessionPoints = useManifoldStore((state) => state.sessionPoints);
  const savedTraces = useTraceStore((state) => state.savedTraces);
  const loadTrace = useTraceStore((state) => state.loadTrace);

  if (!selectedPointId) {
    return (
      <div className="session-details-panel empty">
        <span className="empty-text">Click a point to see session details</span>
      </div>
    );
  }

  const point = sessionPoints.find((p) => p.session_id === selectedPointId);
  const savedTrace = savedTraces.find((t) => t.id === selectedPointId);

  if (!point) {
    return (
      <div className="session-details-panel empty">
        <span className="empty-text">Session not found</span>
      </div>
    );
  }

  const handleLoadSession = () => {
    if (savedTrace) {
      loadTrace(selectedPointId);
    }
  };

  return (
    <div className="session-details-panel">
      <div className="details-header">
        <span className="details-title">Session Details</span>
        <span className="details-id">{point.session_id.slice(0, 16)}...</span>
      </div>

      <div className="details-metrics">
        <div className="metric-row">
          <span className="metric-label">Zone</span>
          <span className={`metric-value zone-${point.manifold_zone}`}>
            {point.manifold_zone.replace('_', ' ')}
          </span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Consensus</span>
          <span className="metric-value">{point.consensus.toFixed(3)}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Hubness</span>
          <span className="metric-value">{point.hubness.toFixed(3)}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Entropy</span>
          <span className="metric-value">{point.entropy.toFixed(3)}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Timestamp</span>
          <span className="metric-value">
            {new Date(point.timestamp).toLocaleString()}
          </span>
        </div>
        {point.model_id && (
          <div className="metric-row">
            <span className="metric-label">Model</span>
            <span className="metric-value">{point.model_id}</span>
          </div>
        )}
      </div>

      {savedTrace ? (
        <button className="load-session-btn" onClick={handleLoadSession}>
          Load Session Transcript
        </button>
      ) : (
        <div className="no-transcript-hint">
          Transcript not saved. Save sessions to view later.
        </div>
      )}
    </div>
  );
}

export function ManifoldView() {
  const fingerprint = useSessionStore((state) => state.fingerprint);
  const selectPoint = useManifoldStore((state) => state.selectPoint);

  const handlePointClick = useCallback(
    (sessionId: string) => {
      selectPoint(sessionId);
    },
    [selectPoint]
  );

  return (
    <div className="card manifold-view">
      <div className="card-header">
        <div className="card-title">
          <span>Manifold Map</span>
          <span className="subtitle">Where this request lives in latent behavior space.</span>
        </div>
        <div className="badges">
          <span className="badge">UMAP</span>
          <span className="badge">HDBSCAN</span>
          {fingerprint && (
            <span className="badge strong">
              {fingerprint.local_mass > 0.5
                ? 'syntax_floor'
                : fingerprint.mid_mass > 0.5
                ? 'semantic_bridge'
                : 'diffuse'}
            </span>
          )}
        </div>
      </div>

      <div className="card-content manifold-layout">
        {/* Left: Map and current metrics */}
        <div className="manifold-main">
          {fingerprint && (
            <div className="metric-chips">
              <div className="chip">
                <strong>Consensus</strong> {(fingerprint.consensus ?? 0).toFixed(2)}
              </div>
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
          )}

          <ManifoldMap onPointClick={handlePointClick} height={280} />

          <div className="hint-text">
            Click a point to see details. Points are colored by manifold zone.
            The "you" marker shows your current session's position.
          </div>
        </div>

        {/* Right: Cluster list and session details */}
        <div className="manifold-sidebar">
          <ClusterList />
          <SessionDetailsPanel />
        </div>
      </div>
    </div>
  );
}
