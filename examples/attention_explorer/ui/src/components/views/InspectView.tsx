import { MessageList } from '../chat/MessageList';
import { LayerSelector } from '../lens/LayerSelector';
import { SegmentToggle } from '../lens/SegmentToggle';
import { useUIStore } from '../../stores/useUIStore';

export function InspectView() {
  const segment = useUIStore((state) => state.segment);
  const setSegment = useUIStore((state) => state.setSegment);

  return (
    <div className="card inspect-view">
      <div className="card-header">
        <div className="card-title">
          <span>Inspector</span>
          <span className="subtitle">Multi-layer token lens: attention anchors, hubs, and distance structure.</span>
        </div>
        <div className="badges">
          <span className="badge">multi-layer</span>
          <span className="badge">hub tokens</span>
        </div>
      </div>
      <div className="card-content">
        <div className="inspect-controls">
          <SegmentToggle value={segment} onChange={setSegment} />
          <LayerSelector />
        </div>
        <div className="inspect-hint">
          Select an assistant token. Then switch layers below. You'll see: <strong>anchors</strong> (high consensus),
          <strong>syntax floor</strong> (local), and <strong>semantic bridge</strong> (mid/long offsets).
        </div>
        <MessageList />
      </div>
    </div>
  );
}
