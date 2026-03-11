export const Llama31Deployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h100', label: 'H100', default: true },
        { id: 'h200', label: 'H200', default: false },
        { id: 'b200', label: 'B200', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    modelsize: {
      name: 'modelsize',
      title: 'Model Size',
      items: [
        { id: '8b', label: '8B', default: false },
        { id: '70b', label: '70B', default: true },
        { id: '405b', label: '405B', default: false }
      ]
    },
    category: {
      name: 'category',
      title: 'Category',
      items: [
        { id: 'base', label: 'Base', default: false },
        { id: 'instruct', label: 'Instruct', default: true }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'bf16', label: 'BF16', default: true },
        { id: 'fp8', label: 'FP8', default: false }
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
    optimization: {
      name: 'optimization',
      title: 'Optimization Mode',
      items: [
        { id: 'basic', label: 'Basic', default: true },
        { id: 'throughput', label: 'Throughput Optimized', default: false },
        { id: 'latency', label: 'Latency Optimized', default: false }
      ]
    }
  };

  // Initialize state
  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      const defaultItem = option.items.find(item => item.default);
      initialState[key] = defaultItem ? defaultItem.id : option.items[0].id;
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
    const { hardware, optimization, modelsize, category, toolcall, quantization } = values;

    const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x';

    // Model size mapping
    const sizeMap = {
      '8b': '8B',
      '70b': '70B',
      '405b': '405B'
    };
    const sizeToken = sizeMap[modelsize] || '70B';
    const categorySuffix = category === 'instruct' ? '-Instruct' : '';

    // Determine model path
    let modelPath;
    if (quantization === 'fp8' && category === 'instruct') {
      if (modelsize === '405b') {
        // Meta official FP8 for 405B
        modelPath = `meta-llama/Llama-3.1-${sizeToken}${categorySuffix}-FP8`;
      } else if (isAMD) {
        // AMD FP8-KV variants for 70B/8B on AMD GPUs
        modelPath = `amd/Llama-3.1-${sizeToken}${categorySuffix}-FP8-KV`;
      } else {
        modelPath = `meta-llama/Llama-3.1-${sizeToken}${categorySuffix}`;
      }
    } else {
      modelPath = `meta-llama/Llama-3.1-${sizeToken}${categorySuffix}`;
    }

    // Determine TP size
    let tpSize;
    if (isAMD) {
      // AMD GPU TP configuration
      const amdTpConfig = {
        'mi300x': {
          '405b': { bf16: 8, fp8: 4 },
          '70b': { bf16: 1, fp8: 1 },
          '8b': { bf16: 1, fp8: 1 }
        },
        'mi325x': {
          '405b': { bf16: 8, fp8: 4 },
          '70b': { bf16: 1, fp8: 1 },
          '8b': { bf16: 1, fp8: 1 }
        },
        'mi355x': {
          '405b': { bf16: 4, fp8: 2 },
          '70b': { bf16: 1, fp8: 1 },
          '8b': { bf16: 1, fp8: 1 }
        }
      };
      tpSize = quantization === 'fp8'
        ? amdTpConfig[hardware][modelsize].fp8
        : amdTpConfig[hardware][modelsize].bf16;
    } else {
      // NVIDIA GPU TP configuration
      if (modelsize === '405b') {
        tpSize = 8;
      } else if (modelsize === '70b' && (hardware === 'h100' || hardware === 'h200')) {
        tpSize = 2;
      }
    }

    // Build command args
    const args = [];
    args.push(`--model-path ${modelPath}`);

    if (tpSize) {
      args.push(`--tp ${tpSize}`);
    }

    // Add quantization flag only if not using FP8 variant model
    if (quantization === 'fp8' && category !== 'instruct') {
      args.push(`--quantization fp8`);
    }

    // NVIDIA-specific optimizations
    if (!isAMD) {
      if (optimization === 'throughput') {
        args.push(`--enable-dp-attention`);
        args.push(`--mem-fraction-static 0.85`);
      } else if (optimization === 'latency') {
        args.push(`--speculative-algorithm EAGLE3`);
        args.push(`--speculative-num-steps 3`);
        args.push(`--speculative-eagle-topk 1`);
        args.push(`--speculative-num-draft-tokens 4`);
        if (modelsize === '8b' && category === 'instruct') {
          args.push(`--speculative-draft-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`);
        } else {
          args.push(`--speculative-draft-model-path \${EAGLE3_MODEL_PATH}`);
        }
        args.push(`--disable-shared-experts-fusion`);
        args.push(`--max-running-requests 64`);
        args.push(`--mem-fraction-static 0.85`);
        args.push(`--kv-cache-dtype fp8_e4m3`);
        args.push(`--context-length 32768`);
      }
    }

    if (toolcall === 'enabled') {
      args.push(`--tool-call-parser llama3`);
    }

    let cmd = 'sglang serve \\\n';
    cmd += `  ${args.join(' \\\n  ')}`;

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

