export const LagunaXS2Deployment = () => {
  // Config options for Laguna-XS.2 (poolside)
  //
  //   poolside/Laguna-XS.2        BF16  -- H200 and B200
  //   poolside/Laguna-XS.2-FP8    FP8   -- H200 and B200; first-launch DeepGEMM
  //                                        JIT pre-compile is multi-session and slow.
  //                                        Pre-warm with `python3 -m sglang.compile_deep_gemm`.
  //   poolside/Laguna-XS.2-NVFP4  NVFP4 -- Blackwell-only (B200); raises
  //                                        NotImplementedError on Hopper.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200', label: 'H200', default: true },
        { id: 'b200', label: 'B200/GB200', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'bf16',  label: 'BF16',  default: true },
        { id: 'fp8',   label: 'FP8',   default: false },
        { id: 'nvfp4', label: 'NVFP4', default: false }
      ]
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true }
      ]
    },
    dpAttention: {
      name: 'dpAttention',
      title: 'DP Attention',
      items: [
        { id: 'disabled', label: 'Disabled', subtitle: 'Low Latency',     default: true },
        { id: 'enabled',  label: 'Enabled',  subtitle: 'High Throughput', default: false }
      ]
    }
  };

  const modelByQuant = {
    bf16:  'poolside/Laguna-XS.2',
    fp8:   'poolside/Laguna-XS.2-FP8',
    nvfp4: 'poolside/Laguna-XS.2-NVFP4'
  };

  const resolveItems = (option, values) => {
    if (typeof option.getDynamicItems === 'function') {
      return option.getDynamicItems(values);
    }
    return option.items;
  };

  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      const items = resolveItems(option, {});
      const defaultItem = items.find(item => item.default && !item.disabled);
      initialState[key] = defaultItem ? defaultItem.id : items[0].id;
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
    setValues(prev => {
      const next = { ...prev, [optionName]: value };
      Object.entries(options).forEach(([key, option]) => {
        if (key === optionName) return;
        const items = resolveItems(option, next);
        const current = items.find(it => it.id === next[key]);
        if (!current || current.disabled) {
          const fallback = items.find(it => !it.disabled);
          if (fallback) next[key] = fallback.id;
        }
      });
      return next;
    });
  };

  const generateCommand = () => {
    const { hardware, quantization, reasoning, toolcall, dpAttention } = values;

    if (hardware === 'h200' && quantization === 'nvfp4') {
      return '# Error: NVFP4 is Blackwell-only. Select B200, or pick BF16/FP8 for H200.';
    }

    const modelId = modelByQuant[quantization];
    if (!modelId) return `# Error: Unknown quantization: ${quantization}`;

    const tp = 8;
    const lines = [
      'sglang serve \\',
      `  --model-path ${modelId} \\`,
      `  --tp ${tp}`
    ];

    if (dpAttention === 'enabled') {
      lines[lines.length - 1] += ' \\';
      lines.push(`  --dp ${tp} \\`);
      lines.push('  --enable-dp-attention');
    }

    if (reasoning === 'enabled') {
      lines[lines.length - 1] += ' \\';
      lines.push('  --reasoning-parser poolside_v1');
    }

    if (toolcall === 'enabled') {
      lines[lines.length - 1] += ' \\';
      lines.push('  --tool-call-parser poolside_v1');
    }

    lines[lines.length - 1] += ' \\';
    lines.push('  --host 0.0.0.0 \\');
    lines.push('  --port 30000');

    return lines.join('\n');
  };

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
      {Object.entries(options).map(([key, option]) => {
        const items = resolveItems(option, values);
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {items.map(item => {
                const isChecked = values[option.name] === item.id;
                return (
                  <label
                    key={item.id}
                    style={{
                      ...labelBaseStyle,
                      ...(isChecked ? checkedStyle : {}),
                      ...(item.disabled ? disabledStyle : {})
                    }}
                  >
                    <input
                      type="radio"
                      name={option.name}
                      value={item.id}
                      checked={isChecked}
                      disabled={item.disabled}
                      onChange={() => !item.disabled && handleRadioChange(option.name, item.id)}
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
