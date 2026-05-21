export const Nemotron3NanoOmniDeployment = () => {
  const MODEL_PATHS = {
    reasoning: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning',
    bf16: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-BF16',
    fp8: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-FP8',
    nvfp4: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-NVFP4',
  };

  const options = {
    model: {
      name: 'model',
      title: 'Model',
      items: [
        { id: 'reasoning', label: 'Reasoning', default: true },
        { id: 'bf16', label: 'BF16', default: false },
        { id: 'fp8', label: 'FP8', default: false },
        { id: 'nvfp4', label: 'NVFP4', default: false },
      ],
    },
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h100', label: 'H100', default: true },
        { id: 'h200', label: 'H200', default: false },
        { id: 'b200', label: 'B200', default: false },
        { id: 'a100', label: 'A100', default: false },
        { id: 'l40s', label: 'L40S', default: false },
      ],
    },
    tp: {
      name: 'tp',
      title: 'Tensor Parallel (TP)',
      items: [
        { id: '1', label: 'TP=1', default: false },
        { id: '2', label: 'TP=2', default: false },
        { id: '4', label: 'TP=4', default: true },
        { id: '8', label: 'TP=8', default: false },
      ],
    },
    kvcache: {
      name: 'kvcache',
      title: 'KV Cache DType',
      items: [
        { id: 'none', label: 'None', default: true },
        { id: 'fp8_e4m3', label: 'fp8_e4m3', default: false },
      ],
    },
    thinking: {
      name: 'thinking',
      title: 'Reasoning Parser',
      items: [
        { id: 'thinking_on', label: 'Enabled', default: true },
        { id: 'thinking_off', label: 'Disabled', default: false },
      ],
      commandRule: (value) => value === 'thinking_on' ? '--reasoning-parser deepseek-r1' : null,
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'toolcall_on', label: 'Enabled', default: true },
        { id: 'toolcall_off', label: 'Disabled', default: false },
      ],
      commandRule: (value) => value === 'toolcall_on' ? '--tool-call-parser qwen3_coder' : null,
    },
  };

  const generateCommand = (values) => {
    const { tp, kvcache, model, hardware } = values;

    if (model === 'nvfp4' && hardware !== 'b200') {
      return '# NVFP4 requires Blackwell hardware. Please select B200.';
    }

    if (hardware === 'l40s' && tp === '1') {
      return '# TP=1 is not supported on L40S for this model. Please use TP=2 or higher.';
    }

    const modelPath = MODEL_PATHS[model] || MODEL_PATHS.reasoning;

    let cmd = 'sglang serve \\\n';
    cmd += `  --model-path ${modelPath} \\\n`;
    cmd += '  --host 0.0.0.0 \\\n';
    cmd += '  --port 30000 \\\n';
    cmd += '  --trust-remote-code \\\n';
    cmd += `  --tp ${tp} \\\n`;

    if (kvcache && kvcache !== 'none') {
      cmd += `  --kv-cache-dtype ${kvcache} \\\n`;
    }

    for (const [key, option] of Object.entries(options)) {
      if (option.commandRule) {
        const rule = option.commandRule(values[key]);
        if (rule) {
          cmd += `  ${rule} \\\n`;
        }
      }
    }

    cmd = cmd.trimEnd();
    if (cmd.endsWith('\\')) {
      cmd = cmd.slice(0, -1).trimEnd();
    }

    return cmd;
  };

  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      const items = option.items || [];
      const defaultItem = items.find((item) => item.default);
      initialState[key] = defaultItem ? defaultItem.id : items[0]?.id || '';
    });
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode =
        html.classList.contains('dark') ||
        html.getAttribute('data-theme') === 'dark' ||
        html.style.colorScheme === 'dark';
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme', 'style'],
    });
    return () => observer.disconnect();
  }, []);

  const handleRadioChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  const command = generateCommand(values);

  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const disabledStyle = { cursor: 'not-allowed', opacity: 0.5 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        const items = option.items || [];
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {items.map((item) => {
                const isChecked = values[option.name] === item.id;
                const isDisabled = Boolean(item.disabled);
                return (
                  <label
                    key={item.id}
                    title={item.disabledReason || ''}
                    style={{
                      ...labelBaseStyle,
                      ...(isChecked ? checkedStyle : {}),
                      ...(isDisabled ? disabledStyle : {}),
                    }}
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
                  </label>
                );
              })}
            </div>
          </div>
        );
      })}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>
  );
};
