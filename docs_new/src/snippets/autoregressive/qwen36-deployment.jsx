export const Qwen36Deployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/Qwen36ConfigGenerator/index.js.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h100', label: 'H100', default: true },
        { id: 'h200', label: 'H200', default: false },
        { id: 'b200', label: 'B200', default: false },
      ],
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      items: [
        { id: 'fp8', label: 'FP8', default: true },
        { id: 'bf16', label: 'BF16', default: false },
      ],
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true },
      ],
      commandRule: (value) => value === 'enabled' ? '--reasoning-parser qwen3' : null,
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true },
      ],
      commandRule: (value) => value === 'enabled' ? '--tool-call-parser qwen3_coder' : null,
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding (MTP)',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true },
      ],
      commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4' : null,
    },
    mambaCache: {
      name: 'mambaCache',
      title: 'Mamba Radix Cache',
      getDynamicItems: (values) => {
        const mtpEnabled = values.speculative === 'enabled';
        if (mtpEnabled) {
          return [
            { id: 'v1', label: 'V1', default: false, disabled: true },
            { id: 'v2', label: 'V2', default: true },
          ];
        }
        return [
          { id: 'v1', label: 'V1', default: true },
          { id: 'v2', label: 'V2', default: false },
        ];
      },
      commandRule: (value) => value === 'v2' ? '--mamba-scheduler-strategy extra_buffer' : null,
    },
  };

  const modelConfigs = {
    h100: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
    h200: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
    b200: { bf16: { tp: 1, mem: 0.8 }, fp8: { tp: 1, mem: 0.8 } },
  };

  const resolveItems = (option, vals) =>
    typeof option.getDynamicItems === 'function' ? option.getDynamicItems(vals) : option.items;

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

  useEffect(() => {
    setValues((prev) => {
      const next = { ...prev };
      for (const [key, option] of Object.entries(options)) {
        if (typeof option.getDynamicItems !== 'function') continue;
        const items = option.getDynamicItems(next);
        const current = items.find((item) => item.id === next[key]);
        if (!current || current.disabled) {
          const fallback = items.find((item) => item.default && !item.disabled) || items.find((item) => !item.disabled);
          if (fallback) next[key] = fallback.id;
        }
      }
      return next;
    });
  }, [values.speculative]);

  const handleRadioChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  const generateCommand = () => {
    const { hardware, quantization, speculative } = values;
    const hwConfig = modelConfigs[hardware]?.[quantization];
    if (!hwConfig) {
      return '# Please select a valid hardware and quantization combination';
    }

    const quantSuffix = quantization === 'fp8' ? '-FP8' : '';
    const modelName = `Qwen/Qwen3.6-35B-A3B${quantSuffix}`;

    let cmd = '';
    if (speculative === 'enabled') {
      cmd += 'SGLANG_ENABLE_SPEC_V2=1 ';
    }

    cmd += `sglang serve --model-path ${modelName}`;
    if (hwConfig.tp > 1) {
      cmd += ` \\\n  --tp ${hwConfig.tp}`;
    }

    const adjustedValues = {
      ...values,
      mambaCache: speculative === 'enabled' ? 'v2' : values.mambaCache,
    };

    for (const [key, option] of Object.entries(options)) {
      if (key === 'quantization' || key === 'hardware') continue;
      if (!option.commandRule) continue;
      const rule = option.commandRule(adjustedValues[key]);
      if (rule) {
        cmd += ` \\\n  ${rule}`;
      }
    }

    if (hardware === 'b200') {
      cmd += ` \\\n  --attention-backend trtllm_mha`;
    }

    cmd += ` \\\n  --mem-fraction-static ${hwConfig.mem}`;
    return cmd;
  };

  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const disabledStyle = { cursor: 'not-allowed', opacity: 0.4 };
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
