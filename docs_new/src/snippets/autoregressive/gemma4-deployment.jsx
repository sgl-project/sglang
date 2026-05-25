export const Gemma4Deployment = () => {
  const options = {
    modelSize: {
      name: 'modelSize',
      title: 'Model Variant',
      items: [
        { id: 'e2b', label: 'E2B (~2B)', default: false },
        { id: 'e4b', label: 'E4B (~4B)', default: true },
        { id: '31b', label: '31B (Dense)', default: false },
        { id: '26b-a4b', label: '26B-A4B (MoE)', default: false },
      ]
    },
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      getDynamicItems: (values) => {
        const size = values.modelSize;
        const showMI300X = size === '31b' || size === '26b-a4b';
        return [
          { id: 'h200', label: 'H200', default: true },
          { id: 'b200', label: 'B200', default: false },
          { id: 'mi300x', label: 'MI300X', default: false, disabled: !showMI300X },
        ];
      }
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true }
      ],
      commandRule: (value) => value === 'enabled' ? '--reasoning-parser gemma4' : null
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled', label: 'Enabled', default: true }
      ],
      commandRule: (value) => value === 'enabled' ? '--tool-call-parser gemma4' : null
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding (MTP)',
      condition: (values) => !['mi300x'].includes(values.hardware),
      items: [
        { id: 'disabled', label: 'Disabled', subtitle: 'Baseline', default: true },
        { id: 'enabled', label: 'Enabled', subtitle: 'Lower Latency', default: false }
      ]
    },
  };

  const modelConfigs = {
    h200: {
      e2b: { tp: 1, mem: 0.85 },
      e4b: { tp: 1, mem: 0.85 },
      '31b': { tp: 2, mem: 0.85 },
      '26b-a4b': { tp: 1, mem: 0.85 },
    },
    b200: {
      e2b: { tp: 1, mem: 0.9 },
      e4b: { tp: 1, mem: 0.9 },
      '31b': { tp: 1, mem: 0.9 },
      '26b-a4b': { tp: 1, mem: 0.9 },
    },
    mi300x: {
      '31b': { tp: 1, mem: 0.80 },
      '26b-a4b': { tp: 1, mem: 0.80 },
    },
  };

  const generateCommand = (values) => {
    const { hardware, modelSize } = values;

    const hwConfig = modelConfigs[hardware]?.[modelSize];
    if (!hwConfig) return `# Error: Unknown hardware/model combination`;

    let { tp, mem } = hwConfig;

    const modelNames = {
      'e2b': 'google/gemma-4-E2B-it',
      'e4b': 'google/gemma-4-E4B-it',
      '31b': 'google/gemma-4-31B-it',
      '26b-a4b': 'google/gemma-4-26B-A4B-it',
    };

    const mtpEnabled = values.speculative === 'enabled';
    if (mtpEnabled && modelSize === '26b-a4b' && hardware !== 'mi300x') {
      tp = 2;
    }

    let cmd = `sglang serve --model-path ${modelNames[modelSize]}`;
    if (tp > 1) {
      cmd += ` \\\n  --tp ${tp}`;
    }

    Object.entries(options).forEach(([key, option]) => {
      if (key === 'modelSize' || key === 'hardware') return;
      if (option.commandRule) {
        const rule = option.commandRule(values[key]);
        if (rule) cmd += ` \\\n  ${rule}`;
      }
    });

    if (mtpEnabled) {
      cmd += ` \\\n  --speculative-algorithm NEXTN`;
      cmd += ` \\\n  --speculative-draft-model-path ${modelNames[modelSize]}-assistant`;
      cmd += ` \\\n  --speculative-num-steps 5`;
      cmd += ` \\\n  --speculative-num-draft-tokens 6`;
      cmd += ` \\\n  --speculative-eagle-topk 1`;
    }

    cmd += ` \\\n  --mem-fraction-static ${mem}`;
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
