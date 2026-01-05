import { useSessionStore } from '../../stores/useSessionStore';

export function RouterView() {
  const fingerprint = useSessionStore((state) => state.fingerprint);

  const manifoldZone = fingerprint
    ? fingerprint.local_mass > 0.5
      ? 'syntax_floor'
      : fingerprint.mid_mass > 0.5
      ? 'semantic_bridge'
      : fingerprint.long_mass > 0.5
      ? 'long_range'
      : 'diffuse'
    : 'unknown';

  return (
    <div className="card router-view">
      <div className="card-header">
        <div className="card-title">
          <span>Discovery Router</span>
          <span className="subtitle">How clustering feeds back into adaptive capture + sampling.</span>
        </div>
        <div className="badges">
          <span className="badge">sidecar feedback</span>
          <span className="badge">adaptive capture</span>
        </div>
      </div>
      <div className="card-content">
        <div className="hint-text">
          This program shows "what the discovery loop would do" after observing the first few decode steps.
          It sends a control signal back to SGLang: which layers to probe next, stride, max steps, and
          whether to switch to sketch mode.
        </div>

        <div className="section">
          <div className="section-header">
            <span>Detected Mode</span>
            <span className="badge strong">{manifoldZone.replace('_', ' ')}</span>
          </div>
          <div className="hint-text">
            {manifoldZone === 'syntax_floor' && 'High local mass → formatting/JSON repair patterns.'}
            {manifoldZone === 'semantic_bridge' && 'High mid mass → retrieval/reasoning connections.'}
            {manifoldZone === 'long_range' && 'High long mass → cross-document planning.'}
            {manifoldZone === 'diffuse' && 'No dominant pattern detected.'}
            {manifoldZone === 'unknown' && 'No fingerprint data available yet.'}
          </div>
        </div>

        <div className="section">
          <div className="section-header">
            <span>Recommended Controls</span>
            <span className="badge">sidecar → scheduler</span>
          </div>
          <div className="control-list">
            <div className="control-row">
              <div className="control-left">
                <div>
                  <strong>next_capture_layer_ids</strong>: [7, 23, 39]
                </div>
                <div className="control-hint">Probe semantic layers; de-emphasize last-layer syntax.</div>
              </div>
              <div className="control-right">cost: +low</div>
            </div>
            <div className="control-row">
              <div className="control-left">
                <div>
                  <strong>attention_stride</strong>: 8
                </div>
                <div className="control-hint">After cluster stabilizes, sample every 8 tokens.</div>
              </div>
              <div className="control-right">bandwidth: ↓</div>
            </div>
            <div className="control-row">
              <div className="control-left">
                <div>
                  <strong>capture_mode</strong>: sketch
                </div>
                <div className="control-hint">Store histograms + anchors instead of per-token edges.</div>
              </div>
              <div className="control-right">throughput: ↑</div>
            </div>
            <div className="control-row">
              <div className="control-left">
                <div>
                  <strong>structured_output</strong>: temperature=0.1
                </div>
                <div className="control-hint">Structured formats benefit from low randomness.</div>
              </div>
              <div className="control-right">quality: ↑</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
