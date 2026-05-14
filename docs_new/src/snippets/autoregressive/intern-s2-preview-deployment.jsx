export const InternS2PreviewDeployment = () => {
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
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
    mtp: {
      name: 'mtp',
      title: 'Multi-Token Prediction',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled', label: 'Enabled', default: false },
      ],
    },
  };

  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      const defaultItem = option.items.find((item) => item.default);
      initialState[key] = defaultItem ? defaultItem.id : option.items[0].id;
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

  const generateCommand = () => {
    const { reasoning, toolcall, mtp } = values;
    const tpValue = 8;

    const flags = [];
    flags.push('  --model-path internLM/Intern-S2-Preview');
    flags.push(`  --tp ${tpValue}`);
    if (reasoning === 'enabled') flags.push('  --reasoning-parser qwen3');
    if (toolcall === 'enabled') flags.push('  --tool-call-parser qwen3_coder');
    if (mtp === 'enabled') {
      flags.push('  --mamba-scheduler-strategy extra_buffer');
      flags.push("  --speculative-algo 'NEXTN'");
      flags.push('  --speculative-eagle-topk 1');
      flags.push('  --speculative-num-steps 3');
      flags.push('  --speculative-num-draft-tokens 4');
    }
    flags.push('  --mem-fraction-static 0.8');
    flags.push('  --host 0.0.0.0');
    flags.push('  --port 30000');

    const envPrefix = mtp === 'enabled' ? 'SGLANG_ENABLE_SPEC_V2=1 \\\n' : '';
    return `${envPrefix}sglang serve \\\n${flags.join(' \\\n')}`;
  };

  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = {
    padding: '8px 12px',
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
    borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`,
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    background: isDark ? '#1f2937' : '#fff',
  };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = {
    padding: '4px 10px',
    border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`,
    borderRadius: '3px',
    cursor: 'pointer',
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '500',
    fontSize: '13px',
    transition: 'all 0.2s',
    userSelect: 'none',
    minWidth: '45px',
    textAlign: 'center',
    flex: 1,
    background: isDark ? '#374151' : '#fff',
    color: isDark ? '#e5e7eb' : 'inherit',
  };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = {
    flex: 1,
    padding: '12px 16px',
    background: isDark ? '#111827' : '#f5f5f5',
    borderRadius: '6px',
    fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    fontSize: '12px',
    lineHeight: '1.5',
    color: isDark ? '#e5e7eb' : '#374151',
    whiteSpace: 'pre-wrap',
    overflowX: 'auto',
    margin: 0,
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
  };

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
                    value={item.id}
                    checked={isChecked}
                    onChange={() => handleRadioChange(option.name, item.id)}
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
