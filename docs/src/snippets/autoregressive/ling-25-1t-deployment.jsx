export const Ling251TDeployment = () => {
  // Config options
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
        { id: 'b200', label: 'B200', default: false },
        { id: 'gb200', label: 'GB200', default: false },
        { id: 'gb300', label: 'GB300', default: false }
      ]
    },
    parallelism: {
      name: 'parallelism',
      title: 'Parallelism Strategy',
      items: [
        { id: 'tp4pp2', label: 'TP4 + PP2', default: true },
        { id: 'tp8', label: 'TP8', default: false }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'enabled', label: 'Enabled', default: true },
        { id: 'disabled', label: 'Disabled', default: false }
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
    const { hardware, parallelism, toolcall } = values;

    const isGB = hardware === 'gb200' || hardware === 'gb300';
    const envPrefix = isGB ? 'NCCL_IB_DISABLE=1 ' : '';

    let tp, pp;
    if (isGB && parallelism === 'tp8') {
      tp = 8;
      pp = null;
    } else if (isGB) {
      tp = 4;
      pp = 2;
    } else {
      tp = 8;
      pp = 2;
    }

    const needMemFrac = hardware === 'h200' || (isGB && parallelism !== 'tp8');

    const generateNodeCmd = (rank) => {
      let cmd = `${envPrefix}python3 -m sglang.launch_server \\\n`;
      cmd += `  --model-path inclusionAI/Ling-2.5-1T \\\n`;
      cmd += `  --trust-remote-code \\\n`;
      cmd += `  --tp-size ${tp} \\\n`;
      if (pp) {
        cmd += `  --pp-size ${pp} \\\n`;
      }
      cmd += `  --nnodes 2 \\\n`;
      cmd += `  --node-rank ${rank} \\\n`;
      if (rank === 0) {
        cmd += `  --host 0.0.0.0 \\\n`;
        cmd += `  --port \${PORT} \\\n`;
      }
      cmd += `  --dist-init-addr \${MASTER_IP}:\${DIST_PORT}`;
      if (toolcall === 'enabled') {
        cmd += ` \\\n  --tool-call-parser qwen`;
      }
      if (needMemFrac) {
        cmd += ` \\\n  --mem-frac 0.95`;
      }
      return cmd;
    };

    let output = `# MASTER_IP is Node 0 IP. PORT and DIST_PORT can be assigned by yourself.\n\n`;
    output += `# Node 0:\n`;
    output += generateNodeCmd(0);
    output += `\n\n\n# Node 1:\n`;
    output += generateNodeCmd(1);

    return output;
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

  const isGB = values.hardware === 'gb200' || values.hardware === 'gb300';

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        // Only show parallelism for GB200/GB300
        if (key === 'parallelism' && !isGB) return null;
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {option.type === 'checkbox' ? (
                option.items.map(item => {
                  const isChecked = (values[option.name] || []).includes(item.id);
                  const isItemDisabled = item.required;
                  return (
                    <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isItemDisabled ? disabledStyle : {}) }}>
                      <input type="checkbox" checked={isChecked} disabled={isItemDisabled} onChange={(e) => handleCheckboxChange(option.name, item.id, e.target.checked)} style={{ display: 'none' }} />
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
        );
      })}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
