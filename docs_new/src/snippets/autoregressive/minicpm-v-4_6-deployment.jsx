export const MiniCPMV46Deployment = () => {
  // NVIDIA platforms listed in chronological generation order:
  //   - A100 (Ampere, sm_80): FA3 falls back to flashinfer.
  //   - H100 / H200 (Hopper, sm_90a): same kernel family.
  //   - B200 (Blackwell, sm_100a): sglang auto-picks trtllm_mha; pinned
  //     explicitly here for safety.
  // B300 / GB300 (sm_103a) require the CUDA-13 image variant (`-cu130`)
  // and are not exposed in this generator.
  //
  // mem-fraction-static values are conservative defaults; re-tune for
  // your workload.
  //
  // Required flags (any hardware):
  //   --trust-remote-code   tokenizer / preprocessor loading
  //   --dtype bfloat16      released ckpt config.json has no torch_dtype;
  //                         without forcing bf16 the GDN causal_conv1d
  //                         triton kernel fails on bf16/fp16 branch merge.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'a100', label: 'A100', default: false },
        { id: 'h100', label: 'H100', default: false },
        { id: 'h200', label: 'H200', default: true  },
        { id: 'b200', label: 'B200', default: false },
      ],
    },
    variant: {
      name: 'variant',
      title: 'Variant',
      items: [
        { id: 'base',     label: 'Base',     subtitle: 'MiniCPM-V-4.6',          default: true  },
        { id: 'thinking', label: 'Thinking', subtitle: 'MiniCPM-V-4.6-Thinking', default: false },
      ],
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'enabled',  label: 'enabled',  default: false  },
        { id: 'disabled', label: 'disabled', default: true },
      ],
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'enabled',  label: 'enabled',  default: false },
        { id: 'disabled', label: 'disabled', default: true  },
      ],
    },
    mambaCache: {
      name: 'mambaCache',
      title: 'Mamba Radix Cache',
      items: [
        { id: 'v1', label: 'V1', default: false  },
        { id: 'v2', label: 'V2', default: true },
      ],
    },
  };

  // Per-hardware tp / mem-fraction-static recommendations (BF16 only).
  // Conservative defaults; re-tune once the released parameter count is known.
  const modelConfigs = {
    a100: { tp: 1, mem: 0.7 },   // 80GB,  Ampere
    h100: { tp: 1, mem: 0.7 },   // 80GB,  Hopper
    h200: { tp: 1, mem: 0.5 },   // 141GB, Hopper
    b200: { tp: 1, mem: 0.4 },   // 180GB, Blackwell
  };

  const generateCommand = (values) => {
    const { variant, hardware, reasoning, toolcall, mambaCache } = values;

    const hwConfig = modelConfigs[hardware];
    if (!hwConfig) return `# Error: Unknown hardware platform`;

    const { tp, mem } = hwConfig;
    const isBlackwell = hardware === 'b200';
    const modelPath = variant === 'thinking'
      ? 'openbmb/MiniCPM-V-4.6-Thinking'
      : 'openbmb/MiniCPM-V-4.6';

    let cmd = `sglang serve --model-path ${modelPath}`;
    if (tp > 1) {
      cmd += ` \\\n  --tp ${tp}`;
    }
    cmd += ` \\\n  --trust-remote-code`;
    cmd += ` \\\n  --dtype bfloat16`;
    if (isBlackwell) {
      cmd += ` \\\n  --attention-backend trtllm_mha`;
    }
    cmd += ` \\\n  --mem-fraction-static ${mem}`;
    if (reasoning === 'enabled') {
      cmd += ` \\\n  --reasoning-parser qwen3`;
    }
    if (toolcall === 'enabled') {
      cmd += ` \\\n  --tool-call-parser qwen3_coder`;
    }
    if (mambaCache === 'v2') {
      cmd += ` \\\n  --mamba-scheduler-strategy extra_buffer`;
    }
    cmd += ` \\\n  --host 0.0.0.0 --port 30000`;

    return cmd;
  };

  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      if (option.type === 'checkbox') {
        initialState[key] = (option.items || [])
          .filter((item) => item.default)
          .map((item) => item.id);
        return;
      }
      if (option.type === 'text') {
        initialState[key] = option.default || '';
        return;
      }
      let items = option.items || [];
      if (option.getDynamicItems) {
        const defaultValues = {};
        Object.entries(options).forEach(([innerKey, innerOption]) => {
          if (innerOption.type === 'checkbox') {
            defaultValues[innerKey] = (innerOption.items || [])
              .filter((item) => item.default)
              .map((item) => item.id);
          } else if (innerOption.type === 'text') {
            defaultValues[innerKey] = innerOption.default || '';
          } else if (innerOption.items && innerOption.items.length > 0) {
            const defaultItem = innerOption.items.find((item) => item.default);
            defaultValues[innerKey] = defaultItem ? defaultItem.id : innerOption.items[0].id;
          }
        });
        items = option.getDynamicItems(defaultValues);
      }
      const defaultItem = items && items.find((item) => item.default);
      initialState[key] = defaultItem ? defaultItem.id : items && items[0] ? items[0].id : '';
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

  const handleCheckboxChange = (optionName, itemId, isChecked) => {
    setValues((prev) => {
      const currentValues = prev[optionName] || [];
      if (isChecked) {
        return { ...prev, [optionName]: [...currentValues, itemId] };
      }
      return {
        ...prev,
        [optionName]: currentValues.filter((id) => id !== itemId),
      };
    });
  };

  const handleTextChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  const command = generateCommand(values);

  const containerStyle = {
    maxWidth: '900px',
    margin: '0 auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  };
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
  const titleStyle = {
    fontSize: '13px',
    fontWeight: '600',
    minWidth: '140px',
    flexShrink: 0,
    color: isDark ? '#e5e7eb' : 'inherit',
  };
  const itemsStyle = {
    display: 'flex',
    rowGap: '2px',
    columnGap: '6px',
    flexWrap: 'wrap',
    alignItems: 'center',
    flex: 1,
  };
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
  const checkedStyle = {
    background: '#D45D44',
    color: 'white',
    borderColor: '#D45D44',
  };
  const disabledStyle = {
    cursor: 'not-allowed',
    opacity: 0.5,
  };
  const subtitleStyle = {
    display: 'block',
    fontSize: '9px',
    marginTop: '1px',
    lineHeight: '1.1',
    opacity: 0.7,
  };
  const textInputStyle = {
    flex: 1,
    padding: '8px 10px',
    borderRadius: '4px',
    border: `1px solid ${isDark ? '#4b5563' : '#d1d5db'}`,
    background: isDark ? '#111827' : '#fff',
    color: isDark ? '#e5e7eb' : '#111827',
    fontSize: '13px',
  };
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
      {Object.entries(options).map(([key, option]) => {
        if (option.condition && !option.condition(values)) {
          return null;
        }
        const items = option.getDynamicItems ? option.getDynamicItems(values) : option.items || [];
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {option.type === 'text' ? (
                <input
                  type="text"
                  value={values[option.name] || ''}
                  placeholder={option.placeholder || ''}
                  onChange={(event) => handleTextChange(option.name, event.target.value)}
                  style={textInputStyle}
                />
              ) : option.type === 'checkbox' ? (
                (option.items || []).map((item) => {
                  const isChecked = (values[option.name] || []).includes(item.id);
                  const isDisabled =
                    item.required ||
                    (typeof item.disabledWhen === 'function' && item.disabledWhen(values));
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
                        type="checkbox"
                        checked={isChecked}
                        disabled={isDisabled}
                        onChange={(event) =>
                          handleCheckboxChange(option.name, item.id, event.target.checked)
                        }
                        style={{ display: 'none' }}
                      />
                      {item.label}
                      {item.subtitle && (
                        <small
                          style={{
                            ...subtitleStyle,
                            color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit',
                          }}
                        >
                          {item.subtitle}
                        </small>
                      )}
                    </label>
                  );
                })
              ) : (
                items.map((item) => {
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
                      {item.subtitle && (
                        <small
                          style={{
                            ...subtitleStyle,
                            color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit',
                          }}
                        >
                          {item.subtitle}
                        </small>
                      )}
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
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>
  );
};
