import { useUIStore, OverheadStats } from '../../stores/useUIStore';

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatRate(stats: OverheadStats): string {
  if (!stats.streamStartTime || stats.tokenCount === 0) return '-';
  const elapsed = (stats.lastTokenTime || Date.now()) - stats.streamStartTime;
  if (elapsed <= 0) return '-';
  const tokensPerSec = (stats.tokenCount / elapsed) * 1000;
  return `${tokensPerSec.toFixed(1)} tok/s`;
}

function getModeColor(mode: OverheadStats['attentionMode']): string {
  switch (mode) {
    case 'raw': return '#ffcc66';       // Yellow - highest overhead
    case 'sketch': return '#7aa2ff';    // Blue - medium
    case 'fingerprint': return '#55d6a6'; // Green - lowest
    default: return '#666';
  }
}

export function OverheadHUD() {
  const overheadStats = useUIStore((state) => state.overheadStats);
  const isConnected = useUIStore((state) => state.isConnected);

  // Don't show if not connected or no data
  if (!isConnected || overheadStats.tokenCount === 0) {
    return null;
  }

  const rate = formatRate(overheadStats);
  const bytes = formatBytes(overheadStats.attentionBytes);
  const mode = overheadStats.attentionMode;
  const modeColor = getModeColor(mode);

  return (
    <div className="overhead-hud" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      padding: '4px 10px',
      background: 'rgba(30, 35, 50, 0.8)',
      borderRadius: '6px',
      fontSize: '11px',
      fontFamily: 'monospace',
      color: '#b0b8c8',
      border: '1px solid rgba(255, 255, 255, 0.08)',
    }}>
      {/* Token count */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <span style={{ color: '#666' }}>tok:</span>
        <span style={{ color: '#ddd' }}>{overheadStats.tokenCount}</span>
      </div>

      {/* Generation rate */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <span style={{ color: '#666' }}>rate:</span>
        <span style={{ color: '#55d6a6' }}>{rate}</span>
      </div>

      {/* Attention overhead */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <span style={{ color: '#666' }}>attn:</span>
        <span style={{ color: '#7aa2ff' }}>{bytes}</span>
      </div>

      {/* Mode indicator */}
      {mode && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: modeColor,
          }} />
          <span style={{ color: modeColor }}>{mode}</span>
        </div>
      )}
    </div>
  );
}
