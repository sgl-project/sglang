export const Wan21Deployment = () => {
  const MODELSIZE_DEFS = [
    {
      id: '14b',
      label: '14B',
      subtitle: 'High-quality, 480P/720P',
      default: true,
      validTasks: ['t2v', 'i2v'],
    },
    {
      id: '1_3b',
      label: '1.3B',
      subtitle: 'Lightweight, 480P',
      default: false,
      validTasks: ['t2v'],
    },
  ];

  const modelConfigs = {
    't2v-14b': {
      repoId: 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
      supportedLoras: [
        { id: 'general', label: 'General Wan2.1 LoRA', path: 'NIVEDAN/wan2.1-lora' },
      ],
    },
    't2v-1_3b': {
      repoId: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
      supportedLoras: [],
    },
    'i2v-14b': {
      repoId: 'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers',
      supportedLoras: [
        { id: 'fight', label: 'Fight Style LoRA', path: 'valiantcat/Wan2.1-Fight-LoRA' },
      ],
    },
  };

  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [{ id: 'mi300x', label: 'MI300X/MI325X/MI355X', default: true }],
    },
    task: {
      name: 'task',
      title: 'Task Type',
      items: [
        { id: 't2v', label: 'Text-to-Video (T2V)', default: true },
        { id: 'i2v', label: 'Image-to-Video (I2V)', default: false },
      ],
    },
    modelsize: {
      name: 'modelsize',
      title: 'Model Variant',
      items: MODELSIZE_DEFS.map(({ validTasks, ...rest }) => rest),
    },
    bestPractice: {
      name: 'bestPractice',
      title: 'Sequence Parallelism',
      items: [
        { id: 'off', label: 'Standard', default: true },
        { id: 'on', label: 'Best Practice (4 GPUs)', default: false },
      ],
    },
  };

  function modelSizeItemsForTask(task) {
    return MODELSIZE_DEFS.filter((item) => item.validTasks.includes(task)).map(
      ({ validTasks, ...rest }) => rest
    );
  }

  const getInitialState = () => {
    const task = 't2v';
    const sizes = modelSizeItemsForTask(task);
    const modelsize = sizes.find((size) => size.default)?.id || sizes[0].id;
    const configKey = `${task}-${modelsize}`;
    const supported = modelConfigs[configKey]?.supportedLoras || [];
    return {
      hardware: 'mi300x',
      task,
      modelsize,
      bestPractice: 'off',
      selectedLoraPath: supported[0]?.path ?? '',
    };
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

  const handleRadioChange = (optionName, itemId) => {
    setValues((prev) => {
      let next = { ...prev, [optionName]: itemId };

      if (optionName === 'task') {
        const sizes = modelSizeItemsForTask(itemId);
        if (!sizes.some((size) => size.id === next.modelsize)) {
          next.modelsize = sizes.find((size) => size.default)?.id || sizes[0].id;
        }
      }

      if (optionName === 'task' || optionName === 'modelsize') {
        const configKey = `${next.task}-${next.modelsize}`;
        const supported = modelConfigs[configKey]?.supportedLoras || [];
        if (supported.length === 0) {
          next.selectedLoraPath = '';
        } else if (
          next.selectedLoraPath &&
          !supported.some((lora) => lora.path === next.selectedLoraPath)
        ) {
          next.selectedLoraPath = supported[0].path;
        }
      }

      return next;
    });
  };

  const handleLoraToggle = (path) => {
    setValues((prev) => ({
      ...prev,
      selectedLoraPath: prev.selectedLoraPath === path ? '' : path,
    }));
  };

  const handleTextChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  const generateCommand = () => {
    const { task, modelsize, selectedLoraPath, bestPractice } = values;
    const configKey = `${task}-${modelsize}`;
    const config = modelConfigs[configKey];

    if (!config) {
      return '# Error: Invalid configuration';
    }

    let command = `sglang serve \\\n  --model-path ${config.repoId} \\\n  --dit-layerwise-offload true`;

    if (bestPractice === 'on') {
      command += ` \\\n  --num-gpus 4 \\\n  --ulysses-degree 2 \\\n  --enable-cfg-parallel`;
    }

    if (selectedLoraPath) {
      command += ` \\\n  --lora-path ${selectedLoraPath}`;
    }

    return command;
  };

  const modelSizeItems = modelSizeItemsForTask(values.task);
  const loraConfigKey = `${values.task}-${values.modelsize}`;
  const availableLoras = modelConfigs[loraConfigKey]?.supportedLoras || [];
  const command = generateCommand();

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
      {Object.entries(options).map(([key, option]) => (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}</div>
          <div style={itemsStyle}>
            {(key === 'modelsize' ? modelSizeItems : option.items).map((item) => {
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

      {availableLoras.length > 0 && (
        <div style={cardStyle}>
          <div style={titleStyle}>Select LoRA Model (Only some of the supported LoRAs are listed here)</div>
          <div style={itemsStyle}>
            {availableLoras.map((lora) => {
              const isChecked = values.selectedLoraPath === lora.path;
              return (
                <label
                  key={lora.id}
                  style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}
                  onClick={(event) => {
                    event.preventDefault();
                    handleLoraToggle(lora.path);
                  }}
                >
                  <input
                    type="radio"
                    name="selectedLoraPath"
                    value={lora.path}
                    checked={isChecked}
                    readOnly
                    style={{ display: 'none' }}
                  />
                  {lora.label}
                  <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>
                    {lora.path}
                  </small>
                </label>
              );
            })}
          </div>
        </div>
      )}

      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>
  );
};
