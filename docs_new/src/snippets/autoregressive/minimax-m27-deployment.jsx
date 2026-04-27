export const MiniMaxM27Deployment = () => {
  // Config options. `getDynamicItems(values)` is evaluated at render time so that
  // e.g. the 2-GPU option is only enabled on AMD or GB300 hardware.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',   label: 'H200',   default: true  },
        { id: 'b200',   label: 'B200',   default: false },
        { id: 'gb300',  label: 'GB300',  default: false },
        { id: 'a100',   label: 'A100',   default: false },
        { id: 'h100',   label: 'H100',   default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    gpuCount: {
      name: 'gpuCount',
      title: 'GPU Count',
      getDynamicItems: (values) => {
        const hw = values.hardware;
        const isAMD = hw === 'mi300x' || hw === 'mi325x' || hw === 'mi355x';
        const isGB300 = hw === 'gb300';
        const canUse2GPU = isAMD || isGB300;
        return [
          { id: '2gpu', label: '2', default: canUse2GPU,  disabled: !canUse2GPU },
          { id: '4gpu', label: '4', default: !canUse2GPU, disabled: false },
          { id: '8gpu', label: '8', default: false,       disabled: isGB300 }
        ];
      }
    },
    thinking: {
      name: 'thinking',
      title: 'Thinking Capabilities',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    }
  };

  // Helper: resolve an option's items (static or dynamic) given current values
  const resolveItems = (option, values) => {
    if (typeof option.getDynamicItems === 'function') {
      return option.getDynamicItems(values);
    }
    return option.items;
  };

  const getInitialState = () => {
    const initialState = {};
    // Resolve hardware first so gpuCount's dynamic items can see it
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, initialState);
      const defaultItem = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled) || items[0];
      initialState[key] = defaultItem.id;
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

  // When hardware changes, re-evaluate gpuCount so disabled/default shifts apply
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

  // Generate command mirrors sgl-cookbook src/components/autoregressive/MiniMaxM27ConfigGenerator/index.js
  const generateCommand = () => {
    const { hardware, gpuCount, thinking, toolcall } = values;

    const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x';
    const isGB300 = hardware === 'gb300';
    const canUse2GPU = isAMD || isGB300;

    if (gpuCount === '2gpu' && !canUse2GPU) {
      return '# Please select compatible hardware\n# 2-GPU requires AMD MI300X/MI325X/MI355X or GB300';
    }

    const modelName = 'MiniMaxAI/MiniMax-M2.7';

    let cmd = 'sglang serve \\\n';
    cmd += `  --model-path ${modelName}`;

    if (gpuCount === '8gpu') {
      cmd += ' \\\n  --tp 8';
      cmd += ' \\\n  --ep 8';
    } else if (gpuCount === '4gpu') {
      cmd += ' \\\n  --tp 4';
      if (isAMD) cmd += ' \\\n  --ep 4';
    } else if (gpuCount === '2gpu') {
      cmd += ' \\\n  --tp 2';
      if (isAMD) cmd += ' \\\n  --ep 2';
    }

    if (toolcall === 'enabled') cmd += ' \\\n  --tool-call-parser minimax-m2';
    if (thinking === 'enabled') cmd += ' \\\n  --reasoning-parser minimax-append-think';

    cmd += ' \\\n  --trust-remote-code';
    cmd += ' \\\n  --mem-fraction-static 0.85';

    if (isAMD) {
      cmd += ' \\\n  --kv-cache-dtype fp8_e4m3';
      cmd += ' \\\n  --attention-backend triton';
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
        const items = resolveItems(option, values);
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {items.map(item => {
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
