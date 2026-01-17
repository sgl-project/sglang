interface SegmentToggleProps {
  value: 'think' | 'output';
  onChange: (value: 'think' | 'output') => void;
}

export function SegmentToggle({ value, onChange }: SegmentToggleProps) {
  return (
    <div className="segment-toggle">
      <button
        className={`seg-tab ${value === 'think' ? 'active' : ''}`}
        onClick={() => onChange('think')}
      >
        Think
      </button>
      <button
        className={`seg-tab ${value === 'output' ? 'active' : ''}`}
        onClick={() => onChange('output')}
      >
        Output
      </button>
    </div>
  );
}
