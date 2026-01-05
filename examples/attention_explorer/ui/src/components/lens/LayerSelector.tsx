import { useUIStore } from '../../stores/useUIStore';

const LAYERS = [
  { id: 7, label: 'Layer 7 (early)' },
  { id: 15, label: 'Layer 15' },
  { id: 23, label: 'Layer 23 (mid)' },
  { id: 31, label: 'Layer 31 (last)' },
];

export function LayerSelector() {
  const selectedLayerId = useUIStore((state) => state.selectedLayerId);
  const setLayer = useUIStore((state) => state.setLayer);

  return (
    <div className="layer-selector">
      {LAYERS.map((layer) => (
        <button
          key={layer.id}
          className={`seg-tab ${selectedLayerId === layer.id ? 'active' : ''}`}
          onClick={() => setLayer(layer.id)}
        >
          {layer.label}
        </button>
      ))}
    </div>
  );
}
