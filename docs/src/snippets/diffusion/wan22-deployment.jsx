export const Wan22Deployment = () => {
  // Config options based on Wan2_2ConfigGenerator
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'b200', label: 'B200', default: true },
        { id: 'h200', label: 'H200', default: false }
      ]
    },
    task: {
      name: 'task',
      title: 'Task Type',
      items: [
        { id: 'i2v', label: 'Image-to-Video (I2V)', default: false },
        { id: 't2v', label: 'Text-to-Video (T2V)', default: true },
        { id: 'ti2v', label: 'Text/Image-to-Video (TI2V)', default: false }
      ]
    },
    modelsize: {
      name: 'modelsize',
      title: 'Model Size',
      items: [
        {
          id: '14b',
          label: 'A14B',
          subtitle: 'Diffusers (A14B)',
          default: true,
          validTasks: ['i2v', 't2v']
        },
        {
          id: '5b',
          label: '5B',
          subtitle: 'Diffusers',
          default: false,
          validTasks: ['ti2v']
        }
      ]
    },
    bestPractice: {
      name: 'bestPractice',
      title: 'Sequence Parallelism',
      items: [
        { id: 'off', label: 'Standard', default: true },
        { id: 'on', label: 'Best Practice (4 GPUs)', default: false }
      ]
    }
  };

  // Model configurations
  const modelConfigs = {
    'i2v-14b': {
      repoId: 'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
      supportedLoras: [
        { id: 'distill', label: 'Distill', path: 'lightx2v/Wan2.2-Distill-Loras' }
      ]
    },
    't2v-14b': {
      repoId: 'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
      supportedLoras: [
        { id: 'arcane', label: 'Arcane Jinx', path: 'Cseti/wan2.2-14B-Arcane_Jinx-lora-v1' }
      ]
    },
    'ti2v-5b': {
      repoId: 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
      supportedLoras: []
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
    initialState.selectedLoraPath = 'none';
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

  // Get available LoRAs based on current task and modelsize
  const availableLoras = () => {
    const configKey = `${values.task}-${values.modelsize}`;
    return modelConfigs[configKey]?.supportedLoras || [];
  };

  const handleRadioChange = (optionName, itemId) => {
    setValues(prev => {
      const newValues = { ...prev, [optionName]: itemId };
      
      // Auto-update modelsize based on task
      if (optionName === 'task') {
        newValues.modelsize = (itemId === 'ti2v') ? '5b' : '14b';
      }

      // Reset LoRA if not supported after change
      const configKey = `${newValues.task}-${newValues.modelsize}`;
      const nextSupported = modelConfigs[configKey]?.supportedLoras || [];
      const isValid = nextSupported.some(l => l.path === prev.selectedLoraPath);
      if (!isValid) {
        newValues.selectedLoraPath = 'none';
      }
      return newValues;
    });
  };

  const handleLoraToggle = (path) => {
    setValues(prev => ({
      ...prev,
      selectedLoraPath: prev.selectedLoraPath === path ? 'none' : path
    }));
  };

  // Generate command
  const generateCommand = () => {
    const { task, modelsize, selectedLoraPath, bestPractice } = values;
    const configKey = `${task}-${modelsize}`;
    const config = modelConfigs[configKey];

    if (!config) return `# Error: Invalid configuration`;

    let command = `sglang serve \\
  --model-path ${config.repoId} \\
  --dit-layerwise-offload true`;

    // Add Best Practice parameters if enabled
    if (bestPractice === 'on') {
      command += ` \\
  --num-gpus 4 \\
  --ulysses-degree 4`;
    }

    // Only add --lora-path if a LoRA is selected and it's not 'none'
    if (selectedLoraPath && selectedLoraPath !== 'none') {
      command += ` \\
  --lora-path ${selectedLoraPath}`;
    }

    return command;
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
            {(() => {
              // Filter items based on validTasks for modelsize option
              const itemsToDisplay = key === 'modelsize'
                ? option.items.filter(item => item.validTasks.includes(values.task))
                : option.items;
              
              return itemsToDisplay.map(item => {
                const isChecked = values[option.name] === item.id;
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}>
                    <input type="radio" name={option.name} value={item.id} checked={isChecked} onChange={() => handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                    {item.label}
                    {item.subtitle && <small style={{ ...subtitleStyle, color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{item.subtitle}</small>}
                  </label>
                );
              });
            })()}
          </div>
        </div>
      ))}
      
      {/* LoRA Selection Section */}
      <div style={cardStyle}>
        <div style={titleStyle}>Select LoRA Model</div>
        <div style={itemsStyle}>
          {availableLoras().length === 0 && (
            <div style={{ color: isDark ? '#999' : '#666', fontSize: '12px', padding: '8px' }}>
              No LoRA models available for this model.
            </div>
          )}
          {availableLoras().map(lora => {
            const isSelected = values.selectedLoraPath === lora.path;
            return (
              <label
                key={lora.id}
                style={{ ...labelBaseStyle, ...(isSelected ? checkedStyle : {}), cursor: 'pointer' }}
                onClick={(e) => {
                  e.preventDefault();
                  handleLoraToggle(lora.path);
                }}
              >
                <input
                  type="radio"
                  name="loraModelSelection"
                  checked={isSelected}
                  readOnly
                  style={{ display: 'none' }}
                />
                {lora.label}
                <small style={{ ...subtitleStyle, color: isSelected ? 'rgba(255,255,255,0.85)' : 'inherit' }}>{lora.path}</small>
              </label>
            );
          })}
        </div>
      </div>
      
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};

