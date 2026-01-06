import { ManifoldMap } from '../manifold/ManifoldMap';
import { useSessionStore } from '../../stores/useSessionStore';

export function ManifoldView() {
  const fingerprint = useSessionStore((state) => state.fingerprint);

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
      <div className="card-content">
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
        <ManifoldMap />
        <div className="hint-text">
          Click a point to load its session. In a real build, this plot is served from a nightly job
          (UMAP/HDBSCAN artifacts), and new points are projected online.
        </div>
      </div>
    </div>
  );
}
