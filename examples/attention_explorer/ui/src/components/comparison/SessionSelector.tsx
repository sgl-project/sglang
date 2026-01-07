import { useComparisonStore } from '../../stores/useComparisonStore';
import { ManifoldZone } from '../../api/types';

interface SessionSelectorProps {
  side: 'left' | 'right';
}

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();

  if (isToday) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function getZoneColor(zone: ManifoldZone): string {
  switch (zone) {
    case 'syntax_floor':
      return '#55d6a6';
    case 'semantic_bridge':
      return '#7aa2ff';
    case 'long_range':
      return '#c678dd';
    case 'structure_ripple':
      return '#ffcc66';
    case 'diffuse':
      return '#ff7b72';
    default:
      return '#888';
  }
}

export function SessionSelector({ side }: SessionSelectorProps) {
  const leftTraceId = useComparisonStore((state) => state.leftTraceId);
  const rightTraceId = useComparisonStore((state) => state.rightTraceId);
  const selectLeft = useComparisonStore((state) => state.selectLeft);
  const selectRight = useComparisonStore((state) => state.selectRight);
  const getAvailableSessions = useComparisonStore(
    (state) => state.getAvailableSessions
  );

  const sessions = getAvailableSessions();
  const selectedId = side === 'left' ? leftTraceId : rightTraceId;
  const otherSelectedId = side === 'left' ? rightTraceId : leftTraceId;
  const selectFn = side === 'left' ? selectLeft : selectRight;

  const selectedSession = sessions.find((s) => s.id === selectedId);

  return (
    <div className="session-selector">
      <select
        value={selectedId || ''}
        onChange={(e) => selectFn(e.target.value || null)}
        className="session-dropdown"
      >
        <option value="">-- Select a session --</option>
        {sessions.map((session) => (
          <option
            key={session.id}
            value={session.id}
            disabled={session.id === otherSelectedId}
          >
            {session.label || `${session.model} - ${formatDate(session.createdAt)}`}
            {session.id === otherSelectedId && ' (selected)'}
          </option>
        ))}
      </select>

      {/* Preview of selected session */}
      {selectedSession && (
        <div className="session-preview">
          <div className="preview-row">
            <span className="preview-label">Model</span>
            <span className="preview-value">{selectedSession.model}</span>
          </div>
          <div className="preview-row">
            <span className="preview-label">Messages</span>
            <span className="preview-value">{selectedSession.messageCount}</span>
          </div>
          <div className="preview-row">
            <span className="preview-label">Tokens</span>
            <span className="preview-value">{selectedSession.tokenCount}</span>
          </div>
          <div className="preview-row">
            <span className="preview-label">Zone</span>
            <span
              className="preview-value zone-badge"
              style={{ color: getZoneColor(selectedSession.dominantZone) }}
            >
              {selectedSession.dominantZone.replace('_', ' ')}
            </span>
          </div>
          <div className="preview-row">
            <span className="preview-label">Entropy</span>
            <span className="preview-value">
              {selectedSession.avgEntropy.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {!selectedSession && (
        <div className="session-preview empty">
          <span className="empty-text">No session selected</span>
        </div>
      )}
    </div>
  );
}
