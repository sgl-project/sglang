export const Qwen25VLDeployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'mi300x', label: 'MI300X', default: true },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    modelsize: {
      name: 'modelsize',
      title: 'Model Size',
      items: [
        { id: '72b', label: '72B', subtitle: 'Dense', default: true },
        { id: '32b', label: '32B', subtitle: 'Dense', default: false },
        { id: '7b', label: '7B', subtitle: 'Dense', default: false },
        { id: '3b', label: '3B', subtitle: 'Dense', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'bf16', label: 'BF16', default: true }
      ]
    }
  };

  // Model configurations
  const modelConfigs = {
    '72b': {
      baseName: '72B',
      mi300x: { tp: 8, ep: 0 },
      mi325x: { tp: 8, ep: 0 },
      mi355x: { tp: 8, ep: 0 }
    },
    '32b': {
      baseName: '32B',
      mi300x: { tp: 2, ep: 0 },
      mi325x: { tp: 2, ep: 0 },
      mi355x: { tp: 2, ep: 0 }
    },
    '7b': {
      baseName: '7B',
      mi300x: { tp: 1, ep: 0 },
      mi325x: { tp: 1, ep: 0 },
      mi355x: { tp: 1, ep: 0 }
    },
    '3b': {
      baseName: '3B',
      mi300x: { tp: 1, ep: 0 },
      mi325x: { tp: 1, ep: 0 },
      mi355x: { tp: 1, ep: 0 }
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
    const { hardware, modelsize: modelSize } = values;

    const config = modelConfigs[modelSize];
    if (!config) {
      return `# Error: Unknown model size: ${modelSize}`;
    }

    const hwConfig = config[hardware];
    if (!hwConfig) {
      return `# Error: Unknown hardware platform: ${hardware}`;
    }

    const modelName = `Qwen/Qwen2.5-VL-${config.baseName}-Instruct`;

    let cmd = 'python -m sglang.launch_server \\\n';
    cmd += `  --model ${modelName}`;

    if (hwConfig.tp > 1) {
      cmd += ` \\\n  --tp ${hwConfig.tp}`;
    }

    if ((hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x') && modelSize === '72b') {
      cmd += ` \\\n  --context-length 128000`;
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
            {option.items.map(item => {
              const isChecked = values[option.name] === item.id;
              const isDisabled = item.disabled;
              return (
                <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}>
                  <input type="radio" name={option.name} value={item.id} checked={isChecked} disabled={isDisabled} onChange={() => handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                  {item.label}
                  {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
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

