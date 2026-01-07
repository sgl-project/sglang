import { useState, useMemo } from 'react';
import { LogitLensEntry, LogitLensLayerResult } from '../../api/types';

interface LogitLensPanelProps {
  entries: LogitLensEntry[];
  totalLayers: number;
  model?: string;
}

function LayerPrediction({
  layer,
  isFirst,
  isLast,
}: {
  layer: LogitLensLayerResult;
  isFirst: boolean;
  isLast: boolean;
}) {
  const topToken = layer.top_tokens[0] || `<${layer.top_token_ids[0]}>`;
  const topProb = layer.top_probs[0];
  const entropy = layer.entropy;
  const klDiv = layer.kl_from_final;

  // Color based on probability
  const probColor =
    topProb > 0.8 ? '#55d6a6' : topProb > 0.5 ? '#7aa2ff' : topProb > 0.2 ? '#ffcc66' : '#ff7b72';

  return (
    <div className={`layer-prediction ${isLast ? 'final' : ''}`}>
      <div className="layer-header">
        <span className="layer-id">L{layer.layer_id}</span>
        {isFirst && <span className="layer-label">early</span>}
        {isLast && <span className="layer-label final">final</span>}
      </div>
      <div className="prediction-content">
        <span className="predicted-token" style={{ color: probColor }}>
          {topToken}
        </span>
        <div className="prediction-stats">
          <span className="prob-value">{(topProb * 100).toFixed(1)}%</span>
          <span className="entropy-value" title="Entropy">
            H={entropy.toFixed(2)}
          </span>
          {klDiv !== undefined && (
            <span className="kl-value" title="KL divergence from final">
              KL={klDiv.toFixed(2)}
            </span>
          )}
        </div>
      </div>
      {/* Top-k alternatives */}
      <div className="alternatives">
        {layer.top_tokens.slice(1, 4).map((token, idx) => (
          <span key={idx} className="alt-token">
            {token} <span className="alt-prob">{(layer.top_probs[idx + 1] * 100).toFixed(0)}%</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function TokenEvolution({
  entry,
  expanded,
  onToggle,
}: {
  entry: LogitLensEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  const layers = Object.values(entry.layers).sort((a, b) => a.layer_id - b.layer_id);
  const finalPrediction = entry.final?.top_tokens?.[0] || '?';
  const tokenText = entry.token_text || `step ${entry.decode_step}`;

  // Check if prediction evolved (early != final)
  const earlyPrediction = layers[0]?.top_tokens?.[0];
  const evolved = earlyPrediction && earlyPrediction !== finalPrediction;

  return (
    <div className={`token-evolution ${expanded ? 'expanded' : ''} ${evolved ? 'evolved' : ''}`}>
      <div className="evolution-header" onClick={onToggle}>
        <span className="token-text">{tokenText}</span>
        <span className="step-number">Step {entry.decode_step}</span>
        <span className="final-prediction">
          {finalPrediction}
        </span>
        {evolved && <span className="evolution-badge">evolved</span>}
        <span className="expand-icon">{expanded ? '−' : '+'}</span>
      </div>

      {expanded && (
        <div className="layers-timeline">
          {layers.map((layer, idx) => (
            <LayerPrediction
              key={layer.layer_id}
              layer={layer}
              isFirst={idx === 0}
              isLast={idx === layers.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function LogitLensPanel({ entries, totalLayers, model }: LogitLensPanelProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const [showOnlyEvolved, setShowOnlyEvolved] = useState(false);

  const toggleStep = (step: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(step)) {
        next.delete(step);
      } else {
        next.add(step);
      }
      return next;
    });
  };

  const filteredEntries = useMemo(() => {
    if (!showOnlyEvolved) return entries;

    return entries.filter((entry) => {
      const layers = Object.values(entry.layers).sort((a, b) => a.layer_id - b.layer_id);
      const earlyPrediction = layers[0]?.top_tokens?.[0];
      const finalPrediction = entry.final?.top_tokens?.[0] || layers[layers.length - 1]?.top_tokens?.[0];
      return earlyPrediction !== finalPrediction;
    });
  }, [entries, showOnlyEvolved]);

  // Summary statistics
  const stats = useMemo(() => {
    let evolvedCount = 0;
    let totalEntropy = 0;
    let totalKL = 0;
    let klCount = 0;

    entries.forEach((entry) => {
      const layers = Object.values(entry.layers).sort((a, b) => a.layer_id - b.layer_id);
      const earlyPrediction = layers[0]?.top_tokens?.[0];
      const finalPrediction = entry.final?.top_tokens?.[0] || layers[layers.length - 1]?.top_tokens?.[0];

      if (earlyPrediction !== finalPrediction) {
        evolvedCount++;
      }

      layers.forEach((layer) => {
        totalEntropy += layer.entropy;
        if (layer.kl_from_final !== undefined) {
          totalKL += layer.kl_from_final;
          klCount++;
        }
      });
    });

    return {
      total: entries.length,
      evolved: evolvedCount,
      avgEntropy: entries.length > 0 ? totalEntropy / (entries.length * 4) : 0, // ~4 layers typically
      avgKL: klCount > 0 ? totalKL / klCount : 0,
    };
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="logit-lens-panel empty">
        <div className="empty-message">
          <span className="empty-icon">🔬</span>
          <h3>No Logit Lens Data</h3>
          <p>
            Enable logit lens capture with <code>return_logit_lens=True</code> to visualize how token
            predictions evolve through model layers.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="logit-lens-panel">
      <div className="panel-header">
        <h3>Logit Lens</h3>
        {model && <span className="model-name">{model}</span>}
        <span className="layer-count">{totalLayers} layers</span>
      </div>

      <div className="lens-stats">
        <div className="stat">
          <span className="stat-label">Tokens</span>
          <span className="stat-value">{stats.total}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Evolved</span>
          <span className="stat-value evolved">{stats.evolved}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Avg Entropy</span>
          <span className="stat-value">{stats.avgEntropy.toFixed(2)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Avg KL</span>
          <span className="stat-value">{stats.avgKL.toFixed(2)}</span>
        </div>
      </div>

      <div className="lens-controls">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={showOnlyEvolved}
            onChange={(e) => setShowOnlyEvolved(e.target.checked)}
          />
          Show only evolved tokens
        </label>
      </div>

      <div className="tokens-list">
        {filteredEntries.map((entry) => (
          <TokenEvolution
            key={entry.decode_step}
            entry={entry}
            expanded={expandedSteps.has(entry.decode_step)}
            onToggle={() => toggleStep(entry.decode_step)}
          />
        ))}
      </div>
    </div>
  );
}
