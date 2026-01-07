import { useState, useRef } from 'react';
import { useUIStore } from '../../stores/useUIStore';
import { useTraceStore } from '../../stores/useTraceStore';
import { View, Program } from '../../api/types';
import { OverheadHUD } from './OverheadHUD';

const VIEWS: { id: View; label: string; icon: string }[] = [
  { id: 'chat', label: 'Chat', icon: '💬' },
  { id: 'inspect', label: 'Inspect', icon: '🔍' },
  { id: 'manifold', label: 'Manifold', icon: '🌐' },
  { id: 'router', label: 'Router', icon: '⚡' },
];

const PROGRAMS: { id: Program; label: string; description: string }[] = [
  { id: 'prod', label: 'Production', description: 'Fast chat with basic attention' },
  { id: 'debug', label: 'Debug', description: 'Raw attention + layer data' },
  { id: 'discovery', label: 'Discovery', description: 'Fingerprints + clustering' },
];

export function TopBar() {
  const view = useUIStore((state) => state.view);
  const program = useUIStore((state) => state.program);
  const isConnected = useUIStore((state) => state.isConnected);
  const modelName = useUIStore((state) => state.modelName);
  const setView = useUIStore((state) => state.setView);
  const setProgram = useUIStore((state) => state.setProgram);

  const currentTrace = useTraceStore((state) => state.currentTrace);
  const exportTraceAsJSONL = useTraceStore((state) => state.exportTraceAsJSONL);
  const importTraceFromJSONL = useTraceStore((state) => state.importTraceFromJSONL);

  const [programChanging, setProgramChanging] = useState(false);
  const [viewChanging, setViewChanging] = useState<View | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const content = await file.text();
      importTraceFromJSONL(content);
    } catch (error) {
      console.error('Failed to import trace:', error);
      alert('Failed to import trace. Please check the file format.');
    }

    // Reset input so same file can be imported again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Show feedback when program changes
  const handleProgramChange = (newProgram: Program) => {
    setProgramChanging(true);
    setProgram(newProgram);
    setTimeout(() => setProgramChanging(false), 800);
  };

  // Show feedback when view changes
  const handleViewChange = (newView: View) => {
    setViewChanging(newView);
    setView(newView);
    setTimeout(() => setViewChanging(null), 300);
  };

  const currentProgram = PROGRAMS.find(p => p.id === program);

  return (
    <div className="topbar">
      <div className="brand">
        <div className="logo" style={{
          animation: programChanging ? 'pulse 0.5s ease-in-out' : undefined
        }}>∿</div>
        <div>
          <span className="brand-title">Latent Chat Explorer</span>
          <div className="brand-subtitle">{currentProgram?.description || 'Chat-first insights'}</div>
        </div>
      </div>

      <div className="tabs">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={`tab ${view === v.id ? 'active' : ''} ${viewChanging === v.id ? 'changing' : ''}`}
            onClick={() => handleViewChange(v.id)}
            style={{
              transform: viewChanging === v.id ? 'scale(0.95)' : undefined,
              transition: 'all 0.15s ease'
            }}
          >
            <span style={{ marginRight: '4px' }}>{v.icon}</span>
            {v.label}
          </button>
        ))}
      </div>

      <div className="right">
        {/* Export/Import buttons */}
        <div style={{ display: 'flex', gap: '4px', marginRight: '8px' }}>
          <button
            onClick={exportTraceAsJSONL}
            disabled={!currentTrace}
            title="Export trace as JSONL"
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              background: currentTrace ? 'rgba(122, 162, 255, 0.2)' : 'rgba(100, 100, 100, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '4px',
              color: currentTrace ? '#7aa2ff' : '#666',
              cursor: currentTrace ? 'pointer' : 'not-allowed',
            }}
          >
            Export
          </button>
          <button
            onClick={handleImportClick}
            title="Import trace from JSONL"
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              background: 'rgba(85, 214, 166, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '4px',
              color: '#55d6a6',
              cursor: 'pointer',
            }}
          >
            Import
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".jsonl,.ndjson"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
        <OverheadHUD />
        <div className={`pill ${isConnected ? 'connected-pill' : 'disconnected-pill'}`}>
          <span className={`dot ${isConnected ? 'connected' : ''}`} style={{
            animation: isConnected ? 'pulse 2s infinite' : undefined
          }} />
          <span style={{
            maxWidth: '200px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}>
            {modelName}
          </span>
        </div>
        <div className={`pill program-pill ${programChanging ? 'changing' : ''}`} style={{
          border: programChanging ? '1px solid rgba(122, 162, 255, 0.5)' : undefined,
          background: programChanging ? 'rgba(122, 162, 255, 0.1)' : undefined,
          transition: 'all 0.3s ease'
        }}>
          <span className={`program-indicator ${program}`} style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: program === 'prod' ? '#55d6a6' : program === 'debug' ? '#ffcc66' : '#7aa2ff',
            marginRight: '6px'
          }} />
          <select
            value={program}
            onChange={(e) => handleProgramChange(e.target.value as Program)}
            className="program-select"
            style={{ cursor: 'pointer' }}
          >
            {PROGRAMS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          {programChanging && (
            <span className="loading-spinner" style={{
              marginLeft: '6px',
              width: '12px',
              height: '12px',
              border: '2px solid rgba(122, 162, 255, 0.3)',
              borderTopColor: 'rgba(122, 162, 255, 0.8)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite'
            }} />
          )}
        </div>
      </div>
    </div>
  );
}
