export const LTX25Deployment = () => {
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Deployment Target',
      items: [
        { id: 'h200', label: '1x H200', subtitle: 'no extra flags', default: true },
        { id: 'tight', label: '1 GPU, tight VRAM', subtitle: 'layerwise offload', default: false },
        { id: 'sp2', label: '2 GPUs', subtitle: 'sequence parallel', default: false },
        { id: 'tp2', label: '2 GPUs', subtitle: 'tensor parallel', default: false },
        { id: 'cfg2', label: '2 GPUs', subtitle: 'CFG parallel', default: false },
      ],
    },
    precision: {
      name: 'precision',
      title: 'Precision',
      items: [
        { id: 'bf16', label: 'bf16', subtitle: 'default', default: true },
        { id: 'fp8', label: 'fp8', subtitle: 'online, -18 GB', default: false },
      ],
    },
    weights: {
      name: 'weights',
      title: 'Weights',
      items: [
        { id: 'distilled', label: 'Distilled', subtitle: '8 steps, unguided', default: true },
        { id: 'dev', label: 'Dev / SFT', subtitle: 'steps + CFG', default: false },
      ],
    },
    pipeline: {
      name: 'pipeline',
      title: 'Pipeline',
      items: [
        { id: 'one-stage', label: 'One Stage', subtitle: '960x544', default: true },
        { id: 'two-stage', label: 'Two Stage', subtitle: '1920x1088', default: false },
      ],
    },
    decoder: {
      name: 'decoder',
      title: 'Decoder',
      items: [
        { id: 'vae', label: 'VAE', subtitle: 'default, fast', default: true },
        { id: 'diffusion', label: 'Diffusion', subtitle: 'slower, more detail', default: false },
      ],
    },
    duration: {
      name: 'duration',
      title: 'Clip Length',
      items: [
        { id: 'fixed', label: 'Fixed', subtitle: '--num-frames', default: true },
        { id: 'auto', label: 'Auto', subtitle: 'duration head', default: false },
      ],
    },
  };

  const REPO_ID = 'Lightricks/LTX-2.5-Diffusers';
  const PIPELINE_CLASSES = {
    'one-stage': 'LTX2Pipeline',
    'two-stage': 'LTX2TwoStagePipeline',
  };

  const [values, setValues] = useState({
    hardware: 'h200',
    precision: 'bf16',
    weights: 'distilled',
    pipeline: 'one-stage',
    decoder: 'vae',
    duration: 'fixed',
  });
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode = html.classList.contains('dark') ||
        html.getAttribute('data-theme') === 'dark' ||
        html.style.colorScheme === 'dark';
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
    return () => observer.disconnect();
  }, []);

  const handleRadioChange = (key, id) => {
    setValues((prev) => ({ ...prev, [key]: id }));
  };

  const getParallelFlags = () => {
    const map = {
      tight: ` \\\n  --dit-layerwise-offload`,
      sp2: ` \\\n  --num-gpus 2 \\\n  --ulysses-degree 2`,
      tp2: ` \\\n  --num-gpus 2 \\\n  --tp-size 2`,
      cfg2: ` \\\n  --num-gpus 2 \\\n  --enable-cfg-parallel`,
    };
    return map[values.hardware] || '';
  };

  const generateCommand = () => {
    let command = `sglang serve \\\n  --model-path ${REPO_ID}`;
    command += ` \\\n  --pipeline-class-name ${PIPELINE_CLASSES[values.pipeline]}`;
    if (values.weights === 'dev') {
      command += ` \\\n  --model-variant dev`;
    }
    if (values.precision === 'fp8') {
      command += ` \\\n  --quantization fp8`;
    }
    command += getParallelFlags();
    command += ` \\\n  --port 30000`;

    // The distilled DiT runs unguided, so there is no negative branch to split.
    if (values.hardware === 'cfg2' && values.weights !== 'dev') {
      command += `\n\n# Note: CFG parallel does nothing on the distilled weights (they run\n#   unguided). Pick "Dev / SFT" above, or use sequence/tensor parallel.`;
    }

    // Per-request flags belong on the generate call, not the server.
    const requestFlags = [];
    if (values.pipeline === 'two-stage') {
      requestFlags.push('--height 1088 --width 1920');
    }
    if (values.weights === 'dev') {
      requestFlags.push('--num-inference-steps 30 --guidance-scale 3.0');
    }
    if (values.duration === 'auto') {
      requestFlags.push('--auto-duration');
    }
    if (values.decoder === 'diffusion') {
      requestFlags.push('--use-diffusion-decoder');
    }
    if (requestFlags.length > 0) {
      command += `\n\n# Per-request flags (pass these to \`sglang generate\`, or as request fields):\n#   ${requestFlags.join(' ')}`;
    }
    return command;
  };

  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}</div>
          <div style={itemsStyle}>
            {option.items.map((item) => {
              const isChecked = values[option.name] === item.id;
              return (
                <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}>
                  <input
                    type="radio"
                    name={option.name}
                    checked={isChecked}
                    onChange={() => handleRadioChange(key, item.id)}
                    style={{ display: 'none' }}
                  />
                  {item.label}
                  {item.subtitle && (
                    <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>
                      {item.subtitle}
                    </small>
                  )}
                </label>
              );
            })}
          </div>
        </div>
      ))}

      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
