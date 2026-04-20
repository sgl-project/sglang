export const DeepSeekV32Deployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/DeepSeekConfigGenerator/index.js.
  //
  // Model variants:
  //   DeepSeek-V3.2, V3.2-Exp, V3.2-Speciale    → deepseek-ai/ family, TP=8
  //   DeepSeek-V3.2-NVFP4                         → nvidia/ family, B200 only, TP=4
  //   DeepSeek-V3.2-MXFP4                         → amd/ family, MI300X/MI355X only, TP=8
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',   label: 'H200',          default: true  },
        { id: 'b200',   label: 'B200',          default: false },
        { id: 'mi300x', label: 'MI300X',        default: false },
        { id: 'mi355x', label: 'MI355X',        default: false }
      ]
    },
    modelname: {
      name: 'modelname',
      title: 'Model Name',
      getDynamicItems: (values) => {
        const hw = values.hardware;
        const isB200 = hw === 'b200';
        const isAMD = hw === 'mi300x' || hw === 'mi355x';
        return [
          { id: 'v32',         label: 'DeepSeek-V3.2',           default: !isB200 && !isAMD },
          { id: 'v32speciale', label: 'DeepSeek-V3.2-Speciale',  default: false },
          { id: 'v32exp',      label: 'DeepSeek-V3.2-Exp',       default: false },
          { id: 'v32nvfp4',    label: 'DeepSeek-V3.2-NVFP4',     default: isB200,  disabled: !isB200, disabledReason: 'NVFP4 requires B200 (Blackwell)' },
          { id: 'v32mxfp4',    label: 'DeepSeek-V3.2-MXFP4',     default: isAMD,   disabled: !isAMD,  disabledReason: 'MXFP4 requires AMD MI300X/MI355X' }
        ];
      }
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      type: 'checkbox',
      condition: (values) => values.modelname !== 'v32nvfp4' && values.modelname !== 'v32mxfp4',
      items: [
        { id: 'tp',  label: 'TP',                       default: true,  required: true },
        { id: 'dp',  label: 'DP attention',              default: false },
        { id: 'ep',  label: 'EP',                        default: false },
        { id: 'mtp', label: 'Multi-token Prediction',    default: false }
      ]
    },
    reasoningParser: {
      name: 'reasoningParser',
      title: 'Reasoning Parser',
      condition: (values) => values.modelname !== 'v32nvfp4' && values.modelname !== 'v32mxfp4',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      condition: (values) => values.modelname !== 'v32nvfp4' && values.modelname !== 'v32mxfp4' && values.modelname !== 'v32speciale',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    }
  };

  const resolveItems = (option, vals) => {
    if (typeof option.getDynamicItems === 'function') return option.getDynamicItems(vals);
    return option.items;
  };

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      if (option.type === 'checkbox') {
        const items = resolveItems(option, initialState);
        initialState[key] = items.filter(i => i.default).map(i => i.id);
      } else {
        const items = resolveItems(option, initialState);
        const def = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled) || items[0];
        initialState[key] = def.id;
      }
    }
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
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

  // When hardware changes, re-resolve model name defaults (NVFP4→B200, MXFP4→AMD).
  useEffect(() => {
    setValues(prev => {
      const next = { ...prev };
      for (const [key, option] of Object.entries(options)) {
        if (typeof option.getDynamicItems !== 'function') continue;
        const items = option.getDynamicItems(next);
        const current = items.find(i => i.id === next[key]);
        if (!current || current.disabled) {
          const fallback = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled);
          if (fallback) next[key] = fallback.id;
        }
      }
      return next;
    });
  }, [values.hardware]);

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

  const generateCommand = () => {
    const { hardware, modelname, strategy, reasoningParser, toolcall } = values;

    const isNvfp4 = modelname === 'v32nvfp4';
    const isMxfp4 = modelname === 'v32mxfp4';
    const isAMD = hardware === 'mi300x' || hardware === 'mi355x';

    // Validation: NVFP4 requires B200
    if (isNvfp4 && hardware !== 'b200') {
      return `# Error: DeepSeek-V3.2-NVFP4 requires NVIDIA B200 (Blackwell) hardware\n# Please select "B200" for Hardware Platform or choose a different model`;
    }

    // Validation: MXFP4 requires AMD MI300X/MI355X
    if (isMxfp4 && !isAMD) {
      return `# Error: DeepSeek-V3.2-MXFP4 requires AMD MI300X/MI355X hardware\n# Please select "MI300X" or "MI355X" for Hardware Platform or choose a different model`;
    }

    // Validation: Speciale doesn't support tool calling
    if (modelname === 'v32speciale' && toolcall === 'enabled') {
      return `# Error: DeepSeek-V3.2-Speciale doesn't support tool calling\n# Please select "Disabled" for Tool Call Parser or choose a different model`;
    }

    // Model name mapping
    const modelMap = {
      'v32':         'DeepSeek-V3.2',
      'v32exp':      'DeepSeek-V3.2-Exp',
      'v32speciale': 'DeepSeek-V3.2-Speciale',
      'v32nvfp4':    'DeepSeek-V3.2-NVFP4',
      'v32mxfp4':    'DeepSeek-V3.2-mxfp4'
    };

    let modelFamily;
    if (isNvfp4) modelFamily = 'nvidia';
    else if (isMxfp4) modelFamily = 'amd';
    else modelFamily = 'deepseek-ai';

    const modelName = `${modelFamily}/${modelMap[modelname]}`;

    // NVFP4: fixed config
    if (isNvfp4) {
      let cmd = 'sglang serve \\\n';
      cmd += `  --model ${modelName}`;
      cmd += ' \\\n  --tp 4';
      cmd += ' \\\n  --quantization modelopt_fp4';
      cmd += ' \\\n  --moe-runner-backend flashinfer_trtllm';
      return cmd;
    }

    // MXFP4: fixed config for AMD
    if (isMxfp4) {
      let cmd = 'sglang serve \\\n';
      cmd += `  --model ${modelName}`;
      cmd += ' \\\n  --tp 8';
      cmd += ' \\\n  --trust-remote-code';
      return cmd;
    }

    let cmd = 'sglang serve \\\n';
    cmd += `  --model ${modelName}`;

    // Hardware platform specific parameters
    if (isAMD) {
      cmd += ' \\\n  --trust-remote-code';
      cmd += ' \\\n  --nsa-prefill-backend tilelang';
      cmd += ' \\\n  --nsa-decode-backend tilelang';
      cmd += ' \\\n  --cuda-graph-max-bs 64';
    }

    // Strategy configurations
    const strategyArray = Array.isArray(strategy) ? strategy : [];
    const tpSize = 8;
    const dpSize = 8;
    const epSize = 8;
    cmd += ` \\\n  --tp ${tpSize}`;
    if (strategyArray.includes('dp')) {
      cmd += ` \\\n  --dp ${dpSize} \\\n  --enable-dp-attention`;
    }
    if (strategyArray.includes('ep')) {
      cmd += ` \\\n  --ep ${epSize}`;
    }

    // Multi-token prediction (MTP) configuration
    if (strategyArray.includes('mtp')) {
      cmd += ' \\\n  --speculative-algorithm EAGLE';
      cmd += ' \\\n  --speculative-num-steps 3';
      cmd += ' \\\n  --speculative-eagle-topk 1';
      cmd += ' \\\n  --speculative-num-draft-tokens 4';
    }

    // Add tool-call-parser if enabled (not supported for Speciale)
    if (toolcall === 'enabled' && modelname !== 'v32speciale') {
      if (modelname === 'v32exp') {
        cmd += ' \\\n  --tool-call-parser deepseekv31';
      } else if (modelname === 'v32') {
        cmd += ' \\\n  --tool-call-parser deepseekv32';
      }
    }

    // Add reasoning-parser when enabled
    if (reasoningParser === 'enabled') {
      cmd += ' \\\n  --reasoning-parser deepseek-v3';
    }

    // Add chat-template if tool calling is enabled (only for v32exp)
    if (toolcall === 'enabled' && modelname === 'v32exp') {
      cmd += ' \\\n  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja';
    }

    return cmd;
  };

  // Styles
  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const disabledStyle = { cursor: 'not-allowed', opacity: 0.4 };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        if (typeof option.condition === 'function' && !option.condition(values)) return null;
        const items = resolveItems(option, values);
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {option.type === 'checkbox' ? (
                items.map(item => {
                  const isChecked = (values[option.name] || []).includes(item.id);
                  const isDisabled = item.required || !!item.disabled;
                  return (
                    <label
                      key={item.id}
                      style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? { ...disabledStyle, ...(item.required ? {} : {}) } : {}) }}
                      title={item.disabledReason || ''}
                    >
                      <input type="checkbox" checked={isChecked} disabled={isDisabled} onChange={(e) => handleCheckboxChange(option.name, item.id, e.target.checked)} style={{ display: 'none' }} />
                      {item.label}
                      {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                    </label>
                  );
                })
              ) : (
                items.map(item => {
                  const isChecked = values[option.name] === item.id;
                  const isDisabled = !!item.disabled;
                  return (
                    <label
                      key={item.id}
                      style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}
                      title={item.disabledReason || ''}
                    >
                      <input
                        type="radio"
                        name={option.name}
                        value={item.id}
                        checked={isChecked}
                        disabled={isDisabled}
                        onChange={() => !isDisabled && handleRadioChange(option.name, item.id)}
                        style={{ display: 'none' }}
                      />
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
