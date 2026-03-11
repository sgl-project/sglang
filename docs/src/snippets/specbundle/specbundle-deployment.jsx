export const SpecBundleDeployment = () => {
  // Config options based on SpecBundleConfigGenerator - matching original structure exactly
  const baseConfig = {
    options: {
      mode: {
        name: 'mode',
        title: 'Launch Mode',
        renderType: 'radio',
        items: [
          { id: 'with-server', label: 'With Server', subtitle: 'Launch SGLang server & Benchmark concurrently', default: true },
          { id: 'without-server', label: 'Without Server', subtitle: 'Connect to an existing server (--skip-launch-server)', default: false }
        ]
      },
      common: {
        name: 'common',
        title: 'Common Configuration',
        renderType: 'inputs',
        items: [
          { id: 'modelPath', label: 'Model Path', type: 'text', placeholder: 'e.g., meta-llama/Llama-3.1-8B-Instruct', default: 'meta-llama/Llama-3.1-8B-Instruct', description: 'Path to the target model.' },
          { id: 'port', label: 'Port', type: 'number', default: 30000, description: 'Port to launch/connect the SGLang server.' },
          { id: 'configList', label: 'Config List', type: 'text', default: '1,3,1,4', description: 'Format: <batch-size>,<num-steps>,<topk>,<num-draft-tokens>' },
          { id: 'benchmarkList', label: 'Benchmark List', type: 'textarea', default: 'mtbench:5 ceval:5:accountant', description: 'Format: <benchmark-name>:<num-prompts>:<subset>. Supported: aime, ceval, financeqa, gpqa, gsm8k, humaneval, livecodebench, math500, mmlu, mmstar, mtbench, simpleqa' }
        ]
      },
      server: {
        name: 'server',
        title: 'Server Configuration',
        renderType: 'inputs',
        requiredMode: 'with-server',
        items: [
          { id: 'draftModelPath', label: 'Draft Model Path', type: 'text', placeholder: 'Path to draft model', default: '', description: 'Path to the speculative draft model.' },
          { id: 'tpSize', label: 'TP Size', type: 'number', default: 1, description: 'Number of GPUs for Tensor Parallelism.' },
          { id: 'memFraction', label: 'Memory Fraction Static', type: 'number', step: '0.1', default: 0.9, description: 'The memory fraction for the static memory.' },
          { id: 'attentionBackend', label: 'Attention Backend', type: 'text', default: '', description: 'The attention backend used in sglang' },
          { id: 'trustRemoteCode', label: 'Trust Remote Code', type: 'checkbox', default: true, description: 'Whether to trust remote code.' }
        ]
      }
    }
  };

  // Initialize state - matching original logic
  const getInitialState = () => {
    const initialState = {};
    Object.values(baseConfig.options).forEach(option => {
      if (option.renderType === 'radio') {
        const defaultItem = option.items.find(item => item.default);
        initialState[option.name] = defaultItem ? defaultItem.id : option.items[0].id;
      } else if (option.renderType === 'inputs') {
        option.items.forEach(item => {
          initialState[item.id] = item.default;
        });
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

  // Get display options based on current mode
  const getDisplayOptions = () => {
    const options = {};
    const currentMode = values.mode;
    Object.entries(baseConfig.options).forEach(([key, option]) => {
      if (option.requiredMode && option.requiredMode !== currentMode) {
        return;
      }
      options[key] = option;
    });
    return options;
  };

  const handleRadioChange = (optionName, itemId) => {
    setValues(prev => ({ ...prev, [optionName]: itemId }));
  };

  const handleInputChange = (itemId, value) => {
    setValues(prev => ({ ...prev, [itemId]: value }));
  };

  const handleCheckboxChange = (itemId, checked) => {
    setValues(prev => ({ ...prev, [itemId]: checked }));
  };

  // Generate command - matching original logic
  const generateCommand = () => {
    const { mode, modelPath, port, configList, benchmarkList, draftModelPath, tpSize, memFraction, attentionBackend, trustRemoteCode } = values;

    let cmd = 'python bench_eagle3.py';
    if (modelPath) cmd += ` \\\n  --model-path ${modelPath}`;
    if (port) cmd += ` \\\n  --port ${port}`;
    if (configList) cmd += ` \\\n  --config-list ${configList}`;
    if (benchmarkList) cmd += ` \\\n  --benchmark-list ${benchmarkList.replace(/\n/g, ' ')}`;

    if (mode === 'without-server') {
      cmd += ' \\\n  --skip-launch-server';
    } else {
      if (draftModelPath) cmd += ` \\\n  --speculative-draft-model-path ${draftModelPath}`;
      if (tpSize) cmd += ` \\\n  --tp-size ${tpSize}`;
      if (memFraction) cmd += ` \\\n  --mem-fraction-static ${memFraction}`;
      if (attentionBackend) cmd += ` \\\n  --attention-backend ${attentionBackend}`;
      if (trustRemoteCode) cmd += ` \\\n  --trust-remote-code`;
    }

    return cmd;
  };

  // Styles
  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'flex-start', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '180px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit', paddingTop: '4px' };
  const contentStyle = { flex: 1 };
  const itemsStyle = { display: 'flex', rowGap: '4px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center' };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const inputGroupStyle = { display: 'flex', flexDirection: 'column', gap: '8px' };
  const inputRowStyle = { display: 'flex', alignItems: 'flex-start', gap: '12px' };
  const inputLabelStyle = { fontSize: '13px', fontWeight: '500', minWidth: '180px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit', paddingTop: '8px' };
  const inputContentStyle = { flex: 1, display: 'flex', flexDirection: 'column' };
  const inputStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#4b5563' : '#d1d5db'}`, borderRadius: '4px', fontSize: '13px', background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit', width: '100%', boxSizing: 'border-box' };
  const textareaStyle = { ...inputStyle, minHeight: '60px', resize: 'vertical' };
  const descStyle = { color: isDark ? '#9ca3af' : '#666', marginTop: '4px', fontSize: '11px' };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  const displayOptions = getDisplayOptions();

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(displayOptions).map(([key, option]) => (
        <div key={key}>
          {/* Render Radio Group - title on left */}
          {option.renderType === 'radio' && (
            <div style={cardStyle}>
              <div style={titleStyle}>{option.title}</div>
              <div style={{ ...contentStyle, ...itemsStyle }}>
                {option.items.map(item => {
                  const isChecked = values[option.name] === item.id;
                  return (
                    <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}>
                      <input type="radio" name={option.name} value={item.id} checked={isChecked} onChange={() => handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                      {item.label}
                      {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                    </label>
                  );
                })}
              </div>
            </div>
          )}

          {/* Render Input Group - each input has label on left */}
          {option.renderType === 'inputs' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {option.items.map(item => (
                <div key={item.id} style={cardStyle}>
                  <div style={inputLabelStyle}>{item.label}</div>
                  <div style={inputContentStyle}>
                    {item.type === 'textarea' ? (
                      <textarea
                        value={values[item.id]}
                        onChange={(e) => handleInputChange(item.id, e.target.value)}
                        style={textareaStyle}
                      />
                    ) : item.type === 'checkbox' ? (
                      <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', gap: '8px', padding: '4px 0' }}>
                        <input
                          type="checkbox"
                          checked={values[item.id]}
                          onChange={(e) => handleCheckboxChange(item.id, e.target.checked)}
                          style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                        />
                        <span style={{ fontSize: '13px', color: isDark ? '#e5e7eb' : 'inherit' }}>Enabled</span>
                      </label>
                    ) : (
                      <input
                        type={item.type}
                        value={values[item.id]}
                        placeholder={item.placeholder}
                        step={item.step}
                        onChange={(e) => handleInputChange(item.id, e.target.value)}
                        style={inputStyle}
                      />
                    )}
                    {item.description && <small style={descStyle}>{item.description}</small>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      <div style={cardStyle}>
        <div style={titleStyle}>Generated Command</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
