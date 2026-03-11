export const MiMoV2FlashDeployment = () => {
  // Config options based on MiMoConfigGenerator
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
        { id: 'h100', label: 'H100', default: false }
      ]
    },
    modelname: {
      name: 'modelname',
      title: 'Model Name',
      items: [
        { id: 'mimo-v2-flash', label: 'MiMo-V2-Flash', default: true }
      ]
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      type: 'checkbox',
      items: [
        { id: 'tp', label: 'TP 8 (Required)', default: true, required: true },
        { id: 'dp', label: 'DP Attention (DP 2)', default: true },
        { id: 'mtp', label: 'Multi-token Prediction (MTP)', default: true },
        { id: 'optimization', label: 'Performance Optimizations', default: true }
      ]
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning & Tools',
      type: 'checkbox',
      items: [
        { id: 'reasoning', label: 'Reasoning Parser (Qwen3)', default: true },
        { id: 'toolcall', label: 'Tool Call Parser', default: true }
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
    const { hardware, modelname, strategy, reasoning } = values;

    const modelPath = 'XiaomiMiMo/MiMo-V2-Flash';
    const strategyArray = Array.isArray(strategy) ? strategy : [];
    const reasoningArray = Array.isArray(reasoning) ? reasoning : [];

    let cmd = 'SGLANG_ENABLE_SPEC_V2=1 python3 -m sglang.launch_server \\\n';
    cmd += `  --model-path ${modelPath} \\\n`;
    cmd += `  --trust-remote-code \\\n`;
    cmd += `  --tp-size 8`;

    // DP settings
    if (strategyArray.includes('dp')) {
      cmd += ` \\\n  --dp-size 2 \\\n  --enable-dp-attention`;
    }

    // Performance Optimizations
    if (strategyArray.includes('optimization')) {
      cmd += ` \\\n  --mem-fraction-static 0.75 \\\n  --max-running-requests 128 \\\n  --chunked-prefill-size 16384 \\\n  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \\\n  --attention-backend fa3`;
    }

    // MTP/Speculative settings
    if (strategyArray.includes('mtp')) {
      cmd += ` \\\n  --speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4 \\\n  --enable-multi-layer-eagle`;
    }

    // Reasoning Parser
    if (reasoningArray.includes('reasoning')) {
      cmd += ` \\\n  --reasoning-parser qwen3`;
    }

    // Tool Call Parser
    if (reasoningArray.includes('toolcall')) {
      cmd += ` \\\n  --tool-call-parser mimo`;
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

