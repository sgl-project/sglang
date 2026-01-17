import { LogitLensPanel } from '../lens/LogitLensPanel';
import { LogitLensEntry } from '../../api/types';

// Mock data for demonstration
const MOCK_LOGIT_LENS_ENTRIES: LogitLensEntry[] = [
  {
    decode_step: 1,
    probed_layers: [0, 10, 20, 31],
    total_layers: 32,
    layers: {
      '0': {
        layer_id: 0,
        top_token_ids: [279, 262, 257],
        top_tokens: ['the', 'a', 'an'],
        top_probs: [0.32, 0.28, 0.15],
        entropy: 2.8,
        kl_from_final: 1.2,
      },
      '10': {
        layer_id: 10,
        top_token_ids: [279, 2911, 262],
        top_tokens: ['the', 'function', 'a'],
        top_probs: [0.45, 0.25, 0.12],
        entropy: 2.1,
        kl_from_final: 0.6,
      },
      '20': {
        layer_id: 20,
        top_token_ids: [2911, 279, 1917],
        top_tokens: ['function', 'the', 'class'],
        top_probs: [0.58, 0.22, 0.08],
        entropy: 1.6,
        kl_from_final: 0.3,
      },
      '31': {
        layer_id: 31,
        top_token_ids: [2911, 279, 1917],
        top_tokens: ['function', 'the', 'class'],
        top_probs: [0.72, 0.15, 0.05],
        entropy: 1.1,
      },
    },
    final: {
      top_token_ids: [2911, 279, 1917],
      top_tokens: ['function', 'the', 'class'],
      top_probs: [0.72, 0.15, 0.05],
    },
    token_text: 'function',
  },
  {
    decode_step: 2,
    probed_layers: [0, 10, 20, 31],
    total_layers: 32,
    layers: {
      '0': {
        layer_id: 0,
        top_token_ids: [1917, 825, 2134],
        top_tokens: ['class', 'method', 'def'],
        top_probs: [0.25, 0.22, 0.18],
        entropy: 2.9,
        kl_from_final: 0.4,
      },
      '10': {
        layer_id: 10,
        top_token_ids: [2134, 1917, 825],
        top_tokens: ['def', 'class', 'method'],
        top_probs: [0.42, 0.28, 0.12],
        entropy: 2.0,
        kl_from_final: 0.2,
      },
      '20': {
        layer_id: 20,
        top_token_ids: [2134, 1917, 825],
        top_tokens: ['def', 'class', 'method'],
        top_probs: [0.65, 0.18, 0.08],
        entropy: 1.4,
        kl_from_final: 0.1,
      },
      '31': {
        layer_id: 31,
        top_token_ids: [2134, 1917, 825],
        top_tokens: ['def', 'class', 'method'],
        top_probs: [0.78, 0.12, 0.04],
        entropy: 0.9,
      },
    },
    final: {
      top_token_ids: [2134, 1917, 825],
      top_tokens: ['def', 'class', 'method'],
      top_probs: [0.78, 0.12, 0.04],
    },
    token_text: 'def',
  },
  {
    decode_step: 3,
    probed_layers: [0, 10, 20, 31],
    total_layers: 32,
    layers: {
      '0': {
        layer_id: 0,
        top_token_ids: [12768, 8251, 6426],
        top_tokens: ['calculate', 'compute', 'process'],
        top_probs: [0.35, 0.30, 0.15],
        entropy: 2.2,
        kl_from_final: 0.1,
      },
      '10': {
        layer_id: 10,
        top_token_ids: [12768, 8251, 6426],
        top_tokens: ['calculate', 'compute', 'process'],
        top_probs: [0.55, 0.25, 0.08],
        entropy: 1.6,
        kl_from_final: 0.05,
      },
      '20': {
        layer_id: 20,
        top_token_ids: [12768, 8251, 6426],
        top_tokens: ['calculate', 'compute', 'process'],
        top_probs: [0.70, 0.18, 0.05],
        entropy: 1.2,
        kl_from_final: 0.02,
      },
      '31': {
        layer_id: 31,
        top_token_ids: [12768, 8251, 6426],
        top_tokens: ['calculate', 'compute', 'process'],
        top_probs: [0.82, 0.12, 0.03],
        entropy: 0.8,
      },
    },
    final: {
      top_token_ids: [12768, 8251, 6426],
      top_tokens: ['calculate', 'compute', 'process'],
      top_probs: [0.82, 0.12, 0.03],
    },
    token_text: 'calculate',
  },
];

export function LogitLensView() {
  // In a real implementation, this would come from a store or API
  const entries = MOCK_LOGIT_LENS_ENTRIES;
  const totalLayers = 32;
  const model = 'Qwen3-80B';

  return (
    <div className="logit-lens-view">
      <div className="view-header">
        <h2>Logit Lens</h2>
        <p className="view-description">
          Visualize how token predictions evolve through model layers. Early layers often predict
          common tokens, while deeper layers refine predictions based on context.
        </p>
      </div>

      <div className="logit-lens-content">
        <LogitLensPanel entries={entries} totalLayers={totalLayers} model={model} />
      </div>

      <div className="logit-lens-info">
        <h4>About Logit Lens</h4>
        <p>
          The logit lens technique projects intermediate layer hidden states through the model's
          unembedding matrix to see what tokens the model would predict at each layer. This helps
          understand:
        </p>
        <ul>
          <li>
            <strong>When the model commits</strong> - At which layer does the final prediction
            become stable?
          </li>
          <li>
            <strong>Prediction evolution</strong> - How do candidate tokens change through layers?
          </li>
          <li>
            <strong>Entropy progression</strong> - Does uncertainty decrease monotonically?
          </li>
          <li>
            <strong>KL divergence</strong> - How different is each layer from the final
            distribution?
          </li>
        </ul>
      </div>
    </div>
  );
}
