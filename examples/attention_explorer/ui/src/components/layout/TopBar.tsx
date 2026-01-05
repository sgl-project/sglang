import { useUIStore } from '../../stores/useUIStore';
import { View, Program } from '../../api/types';

const VIEWS: { id: View; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'inspect', label: 'Inspect' },
  { id: 'manifold', label: 'Manifold' },
  { id: 'router', label: 'Router' },
];

const PROGRAMS: { id: Program; label: string }[] = [
  { id: 'prod', label: 'Production Chat' },
  { id: 'debug', label: 'Debug (Raw Attention)' },
  { id: 'discovery', label: 'Discovery (Fingerprint + Clusters)' },
];

export function TopBar() {
  const view = useUIStore((state) => state.view);
  const program = useUIStore((state) => state.program);
  const isConnected = useUIStore((state) => state.isConnected);
  const modelName = useUIStore((state) => state.modelName);
  const setView = useUIStore((state) => state.setView);
  const setProgram = useUIStore((state) => state.setProgram);

  return (
    <div className="topbar">
      <div className="brand">
        <div className="logo">∿</div>
        <div>
          <span className="brand-title">Latent Chat Explorer</span>
          <div className="brand-subtitle">Chat-first. Insights on hover/tap.</div>
        </div>
      </div>

      <div className="tabs">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={`tab ${view === v.id ? 'active' : ''}`}
            onClick={() => setView(v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div className="right">
        <div className="pill">
          <span className={`dot ${isConnected ? 'connected' : ''}`} />
          <span>{modelName}</span>
        </div>
        <div className="pill">
          <span>Program:</span>
          <select
            value={program}
            onChange={(e) => setProgram(e.target.value as Program)}
            className="program-select"
          >
            {PROGRAMS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
