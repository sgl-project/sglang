export const Ring251TDeployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/Ring25ConfigGenerator/index.js.
  //
  // GPU requirements:
  //   H200 / B200 / GB200 / GB300 / MI355X: single-node (tp per platform)
  //   MI300X / MI325X: two nodes, tp-size 8, pp-size 2 (multi-node scripts)
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',   label: 'H200',   default: true  },
        { id: 'b200',   label: 'B200',   default: false },
        { id: 'gb200',  label: 'GB200',  default: false },
        { id: 'gb300',  label: 'GB300',  default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: true },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    }
  };

  const modelConfigs = {
    h200:   { fp8: { tp: 8 } },
    b200:   { fp8: { tp: 8 } },
    gb200:  { fp8: { tp: 4 } },
    gb300:  { fp8: { tp: 4 } },
    mi300x: { fp8: { tp: 8, pp: 2, nnodes: 2 } },
    mi325x: { fp8: { tp: 8, pp: 2, nnodes: 2 } },
    mi355x: { fp8: { tp: 8 } }
  };

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

  // Generate command — byte-identical to sgl-cookbook Ring25ConfigGenerator
  const generateCommand = () => {
    const { hardware, reasoning, toolcall } = values;
    const modelName = 'inclusionAI/Ring-2.5-1T';
    const amdMultiNode = hardware === 'mi300x' || hardware === 'mi325x';

    // Extra flags from reasoning / toolcall
    const extraFlags = [];
    if (reasoning === 'enabled') extraFlags.push('--reasoning-parser deepseek-r1');
    if (toolcall === 'enabled') extraFlags.push('--tool-call-parser qwen');

    if (amdMultiNode) {
      const hwConfig = modelConfigs[hardware].fp8;
      const tpSize = hwConfig.tp;
      const ppSize = hwConfig.pp;

      const buildAmdNodeCmd = (nodeRank) => {
        let cmd = 'sglang serve \\\n';
        cmd += `--model-path ${modelName} \\\n`;
        cmd += '--trust-remote-code \\\n';
        cmd += `--tp-size ${tpSize} \\\n`;
        cmd += `--pp-size ${ppSize} \\\n`;
        cmd += `--nnodes ${hwConfig.nnodes} \\\n`;
        cmd += `--node-rank ${nodeRank} \\\n`;
        if (nodeRank === 0) {
          cmd += '--host 0.0.0.0 \\\n';
          cmd += '--port 30000 \\\n';
        }
        cmd += '--dist-init-addr ${MASTER_IP}:${DIST_PORT} \\\n';
        cmd += '--attention-backend triton \\\n';
        cmd += '--model-loader-extra-config \'{"enable_multithread_load": "true","num_threads": 64}\' \\\n';
        cmd += '--mem-frac 0.95';
        extraFlags.forEach((flag) => {
          cmd += ` \\\n${flag}`;
        });
        return cmd;
      };

      const envBlock =
        'export MASTER_IP=<your-node0-ip> # Replace with the IP of Node 0\n' +
        'export PORT=30000\n' +
        'export DIST_PORT=20000\n' +
        '# Replace <nic-ifname> with your actual NIC interface name\n' +
        'export GLOO_SOCKET_IFNAME=<nic-ifname>\n' +
        'export TP_SOCKET_IFNAME=<nic-ifname>\n';

      let out = envBlock + '\n';

      out += '\n# Node 0:\n';
      out += buildAmdNodeCmd(0);

      out += '\n\n\n# Node 1:\n';
      out += buildAmdNodeCmd(1);

      return out;
    }

    // Single-node path (H200, B200, GB200, GB300, MI355X)
    const hwConfig = modelConfigs[hardware].fp8;
    const tpValue = hwConfig.tp;

    let cmd = 'sglang serve \\\n';
    cmd += `  --model-path ${modelName}`;
    cmd += ` \\\n  --tp ${tpValue}`;
    cmd += ' \\\n  --trust-remote-code';

    extraFlags.forEach((flag) => {
      cmd += ` \\\n  ${flag}`;
    });

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
      ))}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
