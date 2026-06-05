export const Hunyuan3PreviewDeployment = () => {
  // Hunyuan 3 Preview (~276B total / ~20B active MoE) — BF16 only.
  // ~552GB weights; 80GB-class GPUs (A100/H100) cannot fit single-node.
  //   H200 (141GB): tp=8
  //   B200 (180GB): tp=8
  //   B300 (275GB): tp=4
  //   GB300 (275GB, 4-GPU node): tp=4
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',  label: 'H200',  default: true  },
        { id: 'b200',  label: 'B200',  default: false },
        { id: 'b300',  label: 'B300',  default: false },
        { id: 'gb300', label: 'GB300', default: false }
      ]
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
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
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding (MTP)',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  subtitle: 'Low Latency', default: false }
      ]
    }
  };

  const modelConfigs = {
    h200:  { tp: 8, mem: 0.9 },
    b200:  { tp: 8, mem: 0.9 },
    b300:  { tp: 4, mem: 0.9 },
    gb300: { tp: 4, mem: 0.9 }
  };

  const resolveItems = (option, values) => {
    if (typeof option.getDynamicItems === 'function') return option.getDynamicItems(values);
    return option.items;
  };

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, initialState);
      const def = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled) || items[0];
      initialState[key] = def.id;
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

  const handleRadioChange = (optionName, value) => {
    setValues(prev => ({ ...prev, [optionName]: value }));
  };

  const generateCommand = () => {
    const { hardware } = values;
    const isBlackwell = hardware === 'b200' || hardware === 'b300' || hardware === 'gb300';
    const hwConfig = modelConfigs[hardware];
    if (!hwConfig) return '# Configuration not available for the selected hardware.';

    const modelName = 'tencent/Hy3-preview';
    const tpValue = hwConfig.tp;
    const memFraction = hwConfig.mem;
    const enableSpec = values.speculative === 'enabled';

    let cmd = '';
    if (enableSpec) cmd += 'SGLANG_ENABLE_SPEC_V2=1 ';
    cmd += 'sglang serve \\\n';
    cmd += `  --model-path ${modelName}`;
    cmd += ` \\\n  --tp ${tpValue}`;

    if (values.reasoning === 'enabled') cmd += ' \\\n  --reasoning-parser hunyuan';
    if (values.toolcall  === 'enabled') cmd += ' \\\n  --tool-call-parser hunyuan';
    if (enableSpec) {
      cmd += ' \\\n  --speculative-algorithm EAGLE';
      cmd += ' \\\n  --speculative-num-steps 3';
      cmd += ' \\\n  --speculative-eagle-topk 1';
      cmd += ' \\\n  --speculative-num-draft-tokens 4';
    }

    cmd += ' \\\n  --trust-remote-code';
    cmd += ` \\\n  --mem-fraction-static ${memFraction}`;

    if (isBlackwell) cmd += ' \\\n  --attention-backend trtllm_mha';

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
        if (typeof option.condition === 'function' && !option.condition(values)) return null;
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
