export const DeepSeekV3Deployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h100', label: 'H100', default: false },
        { id: 'h200', label: 'H200', default: false },
        { id: 'b200', label: 'B200', default: true },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'fp8', label: 'FP8', default: true },
        { id: 'fp4', label: 'FP4', default: false }
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

  // Detect dark mode - prioritize page theme over system preference
  useEffect(() => {
    const checkDarkMode = () => {
      // Check Mintlify's theme class on html element
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
    const { hardware, quantization, strategy, thinking, toolcall } = values;
    const strategyArray = Array.isArray(strategy) ? strategy : [];

    // Validation - H100/H200/MI300X/MI325X only supports FP8
    if (['h100', 'h200', 'mi300x', 'mi325x'].includes(hardware) && quantization === 'fp4') {
      return '# Error: This hardware only supports FP8 quantization\n# Please select FP8 quantization or use B200/MI355X hardware';
    }

    const modelPath = quantization === 'fp4' ? 'nvidia/DeepSeek-V3-0324-NVFP4' : 'deepseek-ai/DeepSeek-V3';

    let cmd = 'python3 -m sglang.launch_server \\\n';
    cmd += `  --model-path ${modelPath}`;

    if (strategyArray.includes('tp')) cmd += ' \\\n  --tp 8';
    if (strategyArray.includes('dp')) cmd += ' \\\n  --dp 8 \\\n  --enable-dp-attention';
    if (strategyArray.includes('ep')) cmd += ' \\\n  --ep 8';
    if (strategyArray.includes('mtp')) {
      cmd = 'SGLANG_ENABLE_SPEC_V2=1 ' + cmd;
      cmd += ' \\\n  --speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4';
    }

    cmd += ' \\\n  --enable-symm-mem # Optional: improves performance, but may be unstable';

    if (hardware === 'b200') {
      cmd += ' \\\n  --kv-cache-dtype fp8_e4m3 # Optional: enables fp8 kv cache and fp8 attention kernels';
    }

    if (thinking === 'enabled') cmd += ' \\\n  --reasoning-parser deepseek-v3';
    if (toolcall === 'enabled') cmd += ' \\\n  --tool-call-parser deepseekv3 \\\n  --chat-template examples/chat_template/tool_chat_template_deepseekv3.jinja';

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

