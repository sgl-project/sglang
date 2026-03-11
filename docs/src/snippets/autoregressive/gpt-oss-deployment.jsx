export const GPTOSSDeployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'b200', label: 'B200', default: true },
        { id: 'h200', label: 'H200', default: false },
        { id: 'h100', label: 'H100', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    modelsize: {
      name: 'modelsize',
      title: 'Model Size',
      items: [
        { id: '120b', label: '120B', subtitle: 'MOE', default: true },
        { id: '20b', label: '20B', subtitle: 'MOE', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'mxfp4', label: 'MXFP4', default: true },
        { id: 'bf16', label: 'BF16', default: false }
      ]
    },
    reasoningParser: {
      name: 'reasoningParser',
      title: 'Reasoning Parser',
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
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false }
      ]
    }
  };

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
    setValues(prev => ({ ...prev, [optionName]: value }));
  };

  // Generate command
  const generateCommand = () => {
    const { hardware, modelsize, quantization, reasoningParser, toolcall, speculative } = values;

    // Model configurations
    const modelConfigs = {
      '120b': {
        baseName: '120b',
        h100: { tp: 8 },
        h200: { tp: 8 },
        b200: { tp: 8 },
        mi300x: { tp: 8 },
        mi325x: { tp: 8 },
        mi355x: { tp: 8 }
      },
      '20b': {
        baseName: '20b',
        h100: { tp: 1 },
        h200: { tp: 1 },
        b200: { tp: 1 },
        mi300x: { tp: 1 },
        mi325x: { tp: 1 },
        mi355x: { tp: 1 }
      }
    };

    const config = modelConfigs[modelsize];
    if (!config) {
      return `# Error: Unknown model size: ${modelsize}`;
    }

    const hwConfig = config[hardware];
    if (!hwConfig) {
      return `# Error: Unknown hardware platform: ${hardware}`;
    }

    const quantSuffix = quantization === 'bf16' ? '-bf16' : '';
    const orgPrefix = quantization === 'bf16' ? 'lmsys' : 'openai';
    const modelName = `${orgPrefix}/gpt-oss-${config.baseName}${quantSuffix}`;

    let cmd = '';

    // MI30x GPUs with speculative decoding: Work In Progress
    if ((hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x') && speculative === 'enabled') {
      return '# MI30x GPUs Speculative Decoding: Work In Progress';
    }

    // MI300X/MI325X MXFP4: Work In Progress (only MI355X with gfx950 supports MXFP4)
    if ((hardware === 'mi300x' || hardware === 'mi325x') && quantization === 'mxfp4') {
      return '# MI300X/MI325X GPUs with MXFP4 quantization: Work In Progress';
    }

    // AMD MI30x requires SGLANG_USE_AITER=0 due to YaRN RoPE precision issues
    if (hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x') {
      cmd += 'SGLANG_USE_AITER=0 ';
    }

    if (speculative === 'enabled') {
      cmd += 'SGLANG_ENABLE_SPEC_V2=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 ';
    }

    cmd += 'python -m sglang.launch_server \\\n';

    cmd += `  --model ${modelName}`;

    if (hwConfig.tp > 1) {
      cmd += ` \\\n  --tp ${hwConfig.tp}`;
    }

    // Add reasoning parser if enabled
    if (reasoningParser === 'enabled') {
      cmd += ` \\\n  --reasoning-parser gpt-oss`;
    }

    // Add tool call parser if enabled
    if (toolcall === 'enabled') {
      cmd += ` \\\n  --tool-call-parser gpt-oss`;
    }

    // Add speculative decoding if enabled (MI30x handled above)
    if (speculative === 'enabled') {
      cmd += ` \\\n  --speculative-algorithm EAGLE3 \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4`;

      if (modelsize === '120b') {
        cmd += ` \\\n  --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3`;
      } else if (modelsize === '20b') {
        cmd += ` \\\n  --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16`;
      }
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

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}</div>
          <div style={itemsStyle}>
            {option.type === 'checkbox' ? (
              option.items.map(item => {
                const isChecked = (values[option.name] || []).includes(item.id);
                const isDisabled = item.required;
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
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}>
                    <input type="radio" name={option.name} value={item.id} checked={isChecked} onChange={() => handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              })
            )}
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

