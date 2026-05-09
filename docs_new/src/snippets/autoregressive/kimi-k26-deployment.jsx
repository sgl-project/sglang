export const KimiK26Deployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/KimiK26ConfigGenerator/index.js.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
        { id: 'b200', label: 'B200', default: false },
        { id: 'b300', label: 'B300', default: false },
        { id: 'gb200', label: 'GB200', default: false },
        { id: 'gb300', label: 'GB300', default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi350x', label: 'MI350X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false },
      ],
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true },
      ],
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true },
      ],
    },
    dpattention: {
      name: 'dpattention',
      title: 'DP Attention',
      items: [
        { id: 'disabled', label: 'Disabled', subtitle: 'Low Latency', default: true },
        { id: 'enabled', label: 'Enabled', subtitle: 'High Throughput', default: false },
      ],
    },
  };

  const modelConfigs = {
    h200: { tp: 8 },
    b200: { tp: 8 },
    b300: { tp: 8 },
    gb200: { tp: 4 },
    gb300: { tp: 4 },
    mi300x: { tp: 4 },
    mi325x: { tp: 4 },
    mi350x: { tp: 4 },
    mi355x: { tp: 4 },
  };

  const resolveItems = (option, values) =>
    typeof option.getDynamicItems === 'function' ? option.getDynamicItems(values) : option.items || [];

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, initialState);
      const def = items.find((item) => item.default && !item.disabled) || items.find((item) => !item.disabled) || items[0];
      initialState[key] = def.id;
    }
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

  const generateCommand = () => {
    const { hardware, reasoning, toolcall, dpattention } = values;
    const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi350x' || hardware === 'mi355x';
    const hwConfig = modelConfigs[hardware];
    const tpValue = hwConfig.tp;

    let cmd = '';

    if (isAMD) {
      cmd += 'SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \\\n';
    }

    cmd += 'sglang serve \\\n';
    cmd += '  --model-path moonshotai/Kimi-K2.6';
    cmd += ` \\\n  --tp ${tpValue}`;
    if (isAMD) {
      cmd += ' \\\n  --mem-fraction-static 0.8';
    }
    cmd += ' \\\n  --trust-remote-code';

    if (dpattention === 'enabled') {
      cmd += ` \\\n  --dp ${tpValue} \\\n  --enable-dp-attention`;
    }

    if (reasoning === 'enabled') {
      cmd += ' \\\n  --reasoning-parser kimi_k2';
    }

    if (toolcall === 'enabled') {
      cmd += ' \\\n  --tool-call-parser kimi_k2';
    }

    if (isAMD) {
      cmd += ' \\\n  --kv-cache-dtype fp8_e4m3';
    }

    cmd += ' \\\n  --host 0.0.0.0 \\\n  --port 30000';
    return cmd;
  };

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
        const items = resolveItems(option, values);
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {items.map((item) => {
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
              })}
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
