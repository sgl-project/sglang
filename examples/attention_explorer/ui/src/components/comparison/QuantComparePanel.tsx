import { useState } from 'react';
import {
  QuantizationComparison,
  QuantComparePromptResult,
  QuantCompareStatus,
  getQuantQualityTier,
} from '../../api/types';

interface QuantComparePanelProps {
  comparison: QuantizationComparison | null;
  onLoadResults?: () => void;
}

function StatusBadge({ status }: { status: QuantCompareStatus }) {
  const colors = {
    PASS: { bg: 'rgba(85, 214, 166, 0.2)', text: '#55d6a6', border: 'rgba(85, 214, 166, 0.4)' },
    WARN: { bg: 'rgba(255, 204, 102, 0.2)', text: '#ffcc66', border: 'rgba(255, 204, 102, 0.4)' },
    FAIL: { bg: 'rgba(255, 102, 102, 0.2)', text: '#ff6666', border: 'rgba(255, 102, 102, 0.4)' },
  };
  const c = colors[status];

  return (
    <span
      style={{
        padding: '2px 8px',
        borderRadius: '4px',
        fontSize: '11px',
        fontWeight: 600,
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.border}`,
      }}
    >
      {status}
    </span>
  );
}

function JaccardBar({ value, showLabel = true }: { value: number; showLabel?: boolean }) {
  const percentage = Math.round(value * 100);
  const color =
    value >= 0.8 ? '#55d6a6' : value >= 0.5 ? '#ffcc66' : '#ff6666';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
      <div
        style={{
          flex: 1,
          height: '8px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '4px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${percentage}%`,
            height: '100%',
            background: color,
            borderRadius: '4px',
            transition: 'width 0.3s ease',
          }}
        />
      </div>
      {showLabel && (
        <span style={{ fontSize: '12px', color, fontWeight: 600, minWidth: '40px' }}>
          {percentage}%
        </span>
      )}
    </div>
  );
}

function PromptResultCard({ result }: { result: QuantComparePromptResult }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      style={{
        background: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '8px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          cursor: 'pointer',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: '13px',
              color: '#fff',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {result.prompt}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
          <JaccardBar value={result.mean_jaccard} />
          <StatusBadge status={result.status} />
          <span style={{ fontSize: '12px', color: '#888' }}>
            {expanded ? '▼' : '▶'}
          </span>
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            <div>
              <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px' }}>MIN</div>
              <JaccardBar value={result.min_jaccard} />
            </div>
            <div>
              <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px' }}>MEAN</div>
              <JaccardBar value={result.mean_jaccard} />
            </div>
            <div>
              <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px' }}>MAX</div>
              <JaccardBar value={result.max_jaccard} />
            </div>
          </div>

          {result.tokens_compared !== undefined && (
            <div style={{ fontSize: '11px', color: '#888', marginBottom: '8px' }}>
              Tokens compared: {result.tokens_compared}
              {result.divergent_count !== undefined && result.divergent_count > 0 && (
                <span style={{ color: '#ffcc66', marginLeft: '8px' }}>
                  ({result.divergent_count} divergent)
                </span>
              )}
            </div>
          )}

          {result.bf16Response && (
            <div style={{ marginBottom: '8px' }}>
              <div style={{ fontSize: '10px', color: '#7aa2ff', marginBottom: '4px' }}>BF16 Response</div>
              <div style={{ fontSize: '12px', color: '#ccc', background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '4px' }}>
                {result.bf16Response.slice(0, 200)}
                {result.bf16Response.length > 200 && '...'}
              </div>
            </div>
          )}

          {result.int4Response && (
            <div>
              <div style={{ fontSize: '10px', color: '#55d6a6', marginBottom: '4px' }}>INT4 Response</div>
              <div style={{ fontSize: '12px', color: '#ccc', background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '4px' }}>
                {result.int4Response.slice(0, 200)}
                {result.int4Response.length > 200 && '...'}
              </div>
            </div>
          )}

          {result.per_token_jaccard && result.per_token_jaccard.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ fontSize: '10px', color: '#888', marginBottom: '8px' }}>Per-Token Jaccard</div>
              <div style={{ display: 'flex', gap: '1px', height: '24px', background: 'rgba(0,0,0,0.2)', borderRadius: '4px', overflow: 'hidden' }}>
                {result.per_token_jaccard.slice(0, 50).map((j, i) => (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      background: j >= 0.8 ? '#55d6a6' : j >= 0.5 ? '#ffcc66' : '#ff6666',
                      opacity: 0.3 + j * 0.7,
                    }}
                    title={`Token ${i}: ${Math.round(j * 100)}%`}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function QuantComparePanel({ comparison, onLoadResults }: QuantComparePanelProps) {
  if (!comparison) {
    return (
      <div style={{ textAlign: 'center', padding: '40px 20px' }}>
        <div style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.5 }}>&#x2696;</div>
        <h3 style={{ margin: '0 0 8px', color: '#fff' }}>No Quantization Comparison Loaded</h3>
        <p style={{ color: '#888', fontSize: '13px', marginBottom: '16px' }}>
          Run a SINQ comparison script to generate results, then load them here.
        </p>
        {onLoadResults && (
          <button
            onClick={onLoadResults}
            style={{
              padding: '8px 16px',
              background: 'rgba(122, 162, 255, 0.2)',
              border: '1px solid rgba(122, 162, 255, 0.4)',
              borderRadius: '6px',
              color: '#7aa2ff',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            Load Results
          </button>
        )}
      </div>
    );
  }

  const qualityTier = getQuantQualityTier(comparison.overall_mean_jaccard);
  const tierColors = {
    EXCELLENT: { bg: 'rgba(85, 214, 166, 0.15)', text: '#55d6a6', border: 'rgba(85, 214, 166, 0.3)' },
    ACCEPTABLE: { bg: 'rgba(255, 204, 102, 0.15)', text: '#ffcc66', border: 'rgba(255, 204, 102, 0.3)' },
    POOR: { bg: 'rgba(255, 102, 102, 0.15)', text: '#ff6666', border: 'rgba(255, 102, 102, 0.3)' },
  };
  const tierColor = tierColors[qualityTier];

  const passCount = comparison.results.filter((r) => r.status === 'PASS').length;
  const warnCount = comparison.results.filter((r) => r.status === 'WARN').length;
  const failCount = comparison.results.filter((r) => r.status === 'FAIL').length;

  return (
    <div>
      {/* Header Summary */}
      <div
        style={{
          background: tierColor.bg,
          border: `1px solid ${tierColor.border}`,
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '20px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
          <div>
            <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>MODEL</div>
            <div style={{ fontSize: '16px', color: '#fff', fontWeight: 600 }}>{comparison.model}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>QUANTIZATION</div>
            <div style={{ fontSize: '16px', color: tierColor.text, fontWeight: 600 }}>{comparison.quantization}</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '11px', color: '#888', marginBottom: '8px' }}>OVERALL JACCARD SIMILARITY</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div
                style={{
                  flex: 1,
                  height: '12px',
                  background: 'rgba(0, 0, 0, 0.3)',
                  borderRadius: '6px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${Math.round(comparison.overall_mean_jaccard * 100)}%`,
                    height: '100%',
                    background: tierColor.text,
                    borderRadius: '6px',
                  }}
                />
              </div>
              <span style={{ fontSize: '24px', fontWeight: 700, color: tierColor.text }}>
                {Math.round(comparison.overall_mean_jaccard * 100)}%
              </span>
            </div>
          </div>
          <div
            style={{
              padding: '8px 16px',
              background: tierColor.bg,
              border: `2px solid ${tierColor.border}`,
              borderRadius: '8px',
            }}
          >
            <span style={{ fontSize: '14px', fontWeight: 700, color: tierColor.text }}>{qualityTier}</span>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', marginTop: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#55d6a6' }} />
            <span style={{ fontSize: '12px', color: '#888' }}>{passCount} Pass</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ffcc66' }} />
            <span style={{ fontSize: '12px', color: '#888' }}>{warnCount} Warn</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ff6666' }} />
            <span style={{ fontSize: '12px', color: '#888' }}>{failCount} Fail</span>
          </div>
        </div>
      </div>

      {/* Individual Results */}
      <div>
        <h3 style={{ fontSize: '14px', color: '#fff', marginBottom: '12px' }}>
          Prompt Results ({comparison.results.length})
        </h3>
        {comparison.results.map((result, i) => (
          <PromptResultCard key={i} result={result} />
        ))}
      </div>
    </div>
  );
}
