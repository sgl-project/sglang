export const GLM47Deployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'b200', label: 'B200', default: true },
        { id: 'gb200', label: 'GB200', default: false },
        { id: 'h200', label: 'H200', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'nvfp4', label: 'NVFP4', default: true },
        { id: 'fp8', label: 'FP8', default: false },
        { id: 'bf16', label: 'BF16', default: false }
      ]
    },
    gpus: {
      name: 'gpus',
      title: 'Number of GPUs',
      items: [
        { id: '2', label: '2', default: false },
        { id: '4', label: '4', default: true },
        { id: '8', label: '8', default: false }
      ]
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      type: 'checkbox',
      items: [
        { id: 'tp', label: 'TP', subtitle: 'Tensor Parallel', default: true, required: true },
        { id: 'dp', label: 'DP', subtitle: 'Data Parallel', default: false },
        { id: 'ep', label: 'EP', subtitle: 'Expert Parallel', default: false },
        { id: 'mtp', label: 'MTP', subtitle: 'Multi-token Prediction', default: false }
      ]
    },
    thinking: {
      name: 'thinking',
      title: 'Thinking Capabilities',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ]
    }
  };

  // §3.2 support matrix — single source of truth for the greyed-out controls and
  // generateCommand. hardware -> weight type -> allowed TP sizes (missing key = unsupported).
  const SUPPORT = {
    b200: { nvfp4: [2, 4, 8], fp8: [4, 8], bf16: [8] },
    gb200: { nvfp4: [2, 4], fp8: [4] },
    h200: { fp8: [8], bf16: [8] },
    mi300x: { fp8: [2, 4, 8], bf16: [4, 8] },
    mi325x: { fp8: [2, 4, 8], bf16: [4, 8] },
    mi355x: { fp8: [2, 4, 8], bf16: [4, 8] },
  };
  const quantSupported = (hw, q) => Boolean(SUPPORT[hw] && SUPPORT[hw][q]);
  const allowedTps = (hw, q) => (SUPPORT[hw] && SUPPORT[hw][q]) || [];
  const firstSupportedQuant = (hw) => Object.keys(SUPPORT[hw] || {})[0] || 'fp8';

  // Initialize state
  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      if (option.type === 'checkbox') {
        initialState[key] = option.items.filter(item => item.default).map(item => item.id);
      } else {
        const defaultItem = option.items.find(item => item.default);
        initialState[key] = defaultItem ? defaultItem.id : option.items[0].id;
      }
    });
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);

  // Detect dark mode
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

  const handleRadioChange = (optionName, value) => {
    setValues(prev => {
      const next = { ...prev, [optionName]: value };
      // Keep weight type + GPU count within the §3.2 matrix as hardware/quant change,
      // so the displayed command is always a supported configuration.
      if (optionName === 'hardware' || optionName === 'quantization') {
        if (!quantSupported(next.hardware, next.quantization)) {
          next.quantization = firstSupportedQuant(next.hardware);
        }
        const tps = allowedTps(next.hardware, next.quantization);
        if (tps.length && !tps.includes(parseInt(next.gpus, 10))) {
          next.gpus = String(tps.includes(4) ? 4 : tps[0]);
        }
      }
      return next;
    });
  };

  const handleCheckboxChange = (optionName, itemId, isChecked) => {
    setValues(prev => {
      const currentValues = prev[optionName] || [];
      if (isChecked) {
        return { ...prev, [optionName]: [...currentValues, itemId] };
      } else {
        return { ...prev, [optionName]: currentValues.filter(id => id !== itemId) };
      }
    });
  };

  // Generate command
  const generateCommand = () => {
    const { hardware, quantization, gpus, strategy, thinking, toolcall } = values;
    const strategyArray = Array.isArray(strategy) ? strategy : [];

    const isNvidiaBlackwell = hardware === 'b200' || hardware === 'gb200';
    const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x';

    // Only emit §3.2-supported commands; guards any stale (greyed-out) selection.
    if (!quantSupported(hardware, quantization)) {
      return (
        `# ${quantization.toUpperCase()} is not supported on ${hardware.toUpperCase()} per the §3.2 matrix.\n` +
        `# Pick a highlighted weight type above.`
      );
    }

    // Pick model checkpoint by weight type
    let modelName = 'zai-org/GLM-4.7';
    if (quantization === 'nvfp4') {
      modelName = 'nvidia/GLM-4.7-NVFP4';
    } else if (quantization === 'fp8') {
      modelName = 'zai-org/GLM-4.7-FP8';
    }

    let cmd = 'python -m sglang.launch_server \\\n';
    cmd += `  --model ${modelName}`;

    if (isAMD) {
      // AMD (MI300X / MI325X / MI355X): validated pre-Blackwell command shape.
      // TP is fixed per chip + weight type, so the GPU-count selector is unused here.
      let tpValue = 4; // MI300X / MI325X default
      if (hardware === 'mi355x') {
        tpValue = quantization === 'fp8' ? 2 : 4; // MI355X: TP=2 FP8, TP=4 BF16
      }
      cmd += ` \\\n  --tp ${tpValue}`;

      // MI300X/MI325X BF16 requires extra flags
      if ((hardware === 'mi300x' || hardware === 'mi325x') && quantization === 'bf16') {
        cmd += ` \\\n  --max-context-length 8192 \\\n  --mem-fraction-static 0.9`;
      }
      if (strategyArray.includes('dp')) {
        cmd += ` \\\n  --dp 8 \\\n  --enable-dp-attention`;
      }
      if (strategyArray.includes('ep')) {
        cmd += ` \\\n  --ep 8`;
      }
    } else {
      // NVIDIA (B200 / GB200 / H200): TP follows the "Number of GPUs" selector,
      // clamped to a §3.2-supported value for the chosen hardware + weight type.
      const tps = allowedTps(hardware, quantization);
      let tpValue = parseInt(gpus, 10) || tps[0];
      if (!tps.includes(tpValue)) {
        tpValue = tps.includes(4) ? 4 : tps[0];
      }
      cmd += ` \\\n  --tp-size ${tpValue}`;

      // Blackwell + NVFP4: enable EP when the user selected it
      if (isNvidiaBlackwell && quantization === 'nvfp4' && strategyArray.includes('ep')) {
        cmd += ` \\\n  --ep ${tpValue}`;
      }
      // Blackwell + NVFP4: leave headroom for cuda-graph capture
      if (isNvidiaBlackwell && quantization === 'nvfp4') {
        cmd += ` \\\n  --mem-fraction-static 0.85`;
      }
    }

    // MTP / EAGLE speculative decoding (all platforms)
    if (strategyArray.includes('mtp')) {
      cmd = 'SGLANG_ENABLE_SPEC_V2=1 ' + cmd;
      cmd += ` \\\n  --speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4`;
    }

    if (toolcall === 'enabled') {
      cmd += ` \\\n  --tool-call-parser glm47`;
    }

    // glm45 is the registered reasoning detector; glm47 is only valid for tool-call.
    if (thinking === 'enabled') {
      cmd += ` \\\n  --reasoning-parser glm45`;
    }

    return cmd;
  };

  // Styles - with dark mode support
  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const disabledStyle = { cursor: 'not-allowed', opacity: 0.5 };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  // Which Deployment Strategy toggles apply (mirrors generateCommand): DP only on
  // AMD; EP only on AMD or Blackwell + NVFP4 — greyed otherwise.
  const hwSel = values.hardware;
  const isAMDSel = hwSel === 'mi300x' || hwSel === 'mi325x' || hwSel === 'mi355x';
  const isBlackwellSel = hwSel === 'b200' || hwSel === 'gb200';
  const strategyApplies = (id) => {
    if (id === 'dp') return isAMDSel;
    if (id === 'ep') return isAMDSel || (isBlackwellSel && values.quantization === 'nvfp4');
    return true; // tp (required) and mtp (all platforms)
  };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        // GPU count is fixed (greyed) on AMD; on NVIDIA individual counts are greyed
        // per the §3.2 matrix. Weight types unsupported on the hardware are greyed too.
        const gpusGroupAMD = key === 'gpus' && isAMDSel;
        return (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}{gpusGroupAMD ? ' (N/A for AMD)' : ''}</div>
          <div style={itemsStyle}>
            {option.type === 'checkbox' ? (
              option.items.map(item => {
                const isChecked = (values[option.name] || []).includes(item.id);
                const isDisabled = item.required || (key === 'strategy' && !strategyApplies(item.id));
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}>
                    <input type="checkbox" checked={isChecked} disabled={isDisabled} onChange={(e) => handleCheckboxChange(option.name, item.id, e.target.checked)} style={{ display: 'none' }} />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              })
            ) : (
              option.items.map(item => {
                const isChecked = values[option.name] === item.id;
                const isDisabled =
                  (key === 'gpus' && (gpusGroupAMD || !allowedTps(values.hardware, values.quantization).includes(parseInt(item.id, 10)))) ||
                  (key === 'quantization' && !quantSupported(values.hardware, item.id));
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}>
                    <input type="radio" name={option.name} value={item.id} checked={isChecked} disabled={isDisabled} onChange={() => !isDisabled && handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              })
            )}
          </div>
        </div>
        );
      })}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
