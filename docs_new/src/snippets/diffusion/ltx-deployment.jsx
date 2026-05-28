export const LTXDeployment = () => {
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Deployment Target',
      items: [
        { id: 'h200', label: '1x H200', subtitle: 'resident', default: true },
        { id: 'h200-2gpu', label: '2 GPUs', subtitle: 'CFG parallel', default: false },
        { id: 'h200-4gpu', label: '4 GPUs', subtitle: 'TP2 + CFG', default: false },
        { id: 'standard', label: 'Standard CUDA', subtitle: 'Snapshot mode', default: false },
        { id: 'official', label: 'Official Match', subtitle: 'Original switching', default: false },
      ],
    },
    model: {
      name: 'model',
      title: 'Model',
      items: [
        { id: 'ltx23', label: 'LTX-2.3', default: true },
        { id: 'ltx2', label: 'LTX-2', default: false },
      ],
    },
    pipeline: {
      name: 'pipeline',
      title: 'Pipeline',
      items: [
        { id: 'two-stage', label: 'Two Stage', default: true, validModels: ['ltx2', 'ltx23'] },
        { id: 'two-stage-hq', label: 'Two Stage HQ', subtitle: 'High Quality', default: false, validModels: ['ltx23'] },
        { id: 'one-stage', label: 'One Stage', default: false, validModels: ['ltx2', 'ltx23'] },
      ],
    },
  };

  const modelConfigs = {
    ltx2: {
      repoId: 'Lightricks/LTX-2',
      pipelines: {
        'one-stage': 'LTX2Pipeline',
        'two-stage': 'LTX2TwoStagePipeline',
      },
      supportedLoras: [],
    },
    ltx23: {
      repoId: 'Lightricks/LTX-2.3',
      pipelines: {
        'one-stage': 'LTX2Pipeline',
        'two-stage': 'LTX2TwoStagePipeline',
        'two-stage-hq': 'LTX2TwoStageHQPipeline',
      },
      supportedLoras: [
        {
          id: 'transition',
          path: 'valiantcat/LTX-2.3-Transition-LORA',
          weightName: 'ltx2.3-transition.safetensors',
          validPipelines: ['two-stage', 'two-stage-hq'],
        },
      ],
    },
  };

  const getInitialState = () => ({
    hardware: 'h200',
    model: 'ltx23',
    pipeline: 'two-stage',
    selectedLoraPath: 'none',
  });

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

  const availableLoras = (() => {
    const config = modelConfigs[values.model];
    return (config?.supportedLoras || []).filter((lora) => lora.validPipelines.includes(values.pipeline));
  })();

  const handleRadioChange = (optionName, itemId) => {
    setValues((prev) => {
      const next = { ...prev, [optionName]: itemId };

      const validPipeline = options.pipeline.items.some((item) => (
        item.id === next.pipeline && item.validModels.includes(next.model)
      ));
      if (!validPipeline) {
        next.pipeline = 'two-stage';
      }

      const config = modelConfigs[next.model];
      const nextSupported = (config?.supportedLoras || []).filter((lora) => lora.validPipelines.includes(next.pipeline));
      const isValid = nextSupported.some((lora) => lora.path === prev.selectedLoraPath);
      if (!isValid) {
        next.selectedLoraPath = 'none';
      }
      return next;
    });
  };

  const handleLoraToggle = (path) => {
    setValues((prev) => ({
      ...prev,
      selectedLoraPath: prev.selectedLoraPath === path ? 'none' : path,
    }));
  };

  const getDeviceMode = () => {
    if (values.hardware.startsWith('h200')) {
      return 'resident';
    }
    if (values.hardware === 'official') {
      return 'original';
    }
    return 'snapshot';
  };

  const getParallelFlags = () => {
    const parallelFlagsMap = {
      'h200-2gpu': ` \\\n  --num-gpus 2 \\\n  --enable-cfg-parallel`,
      'h200-4gpu': ` \\\n  --num-gpus 4 \\\n  --tp-size 2 \\\n  --enable-cfg-parallel`,
    };
    return parallelFlagsMap[values.hardware] || '';
  };

  const generateCommand = () => {
    const config = modelConfigs[values.model];
    const pipelineClass = config.pipelines[values.pipeline];
    if (!pipelineClass) {
      return '# Error: Invalid configuration';
    }

    let command = `sglang serve \\\n  --model-path ${config.repoId} \\\n  --pipeline-class-name ${pipelineClass}`;
    command += getParallelFlags();
    if (values.model === 'ltx23' && values.pipeline !== 'one-stage') {
      command += ` \\\n  --ltx2-two-stage-device-mode ${getDeviceMode()}`;
    }

    const selectedLora = availableLoras.find((lora) => lora.path === values.selectedLoraPath);
    if (selectedLora) {
      command += ` \\\n  --lora-path ${selectedLora.path} \\\n  --lora-weight-name ${selectedLora.weightName}`;
    }

    command += ` \\\n  --port 30000`;
    return command;
  };

  const containerStyle = { maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px' };
  const cardStyle = { padding: '8px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`, borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`, borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '12px', background: isDark ? '#1f2937' : '#fff' };
  const titleStyle = { fontSize: '13px', fontWeight: '600', minWidth: '140px', flexShrink: 0, color: isDark ? '#e5e7eb' : 'inherit' };
  const itemsStyle = { display: 'flex', rowGap: '2px', columnGap: '6px', flexWrap: 'wrap', alignItems: 'center', flex: 1 };
  const labelBaseStyle = { padding: '4px 10px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`, borderRadius: '3px', cursor: 'pointer', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontWeight: '500', fontSize: '13px', transition: 'all 0.2s', userSelect: 'none', minWidth: '45px', textAlign: 'center', flex: 1, background: isDark ? '#374151' : '#fff', color: isDark ? '#e5e7eb' : 'inherit' };
  const checkedStyle = { background: '#D45D44', color: 'white', borderColor: '#D45D44' };
  const subtitleStyle = { display: 'block', fontSize: '9px', marginTop: '1px', lineHeight: '1.1', opacity: 0.7 };
  const commandDisplayStyle = { flex: 1, padding: '12px 16px', background: isDark ? '#111827' : '#f5f5f5', borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace", fontSize: '12px', lineHeight: '1.5', color: isDark ? '#e5e7eb' : '#374151', whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}` };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        const itemsToDisplay = key === 'pipeline'
          ? option.items.filter((item) => item.validModels.includes(values.model))
          : option.items;

        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {itemsToDisplay.map((item) => {
                const isChecked = values[option.name] === item.id;
                return (
                  <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}) }}>
                    <input
                      type="radio"
                      name={option.name}
                      checked={isChecked}
                      onChange={() => handleRadioChange(key, item.id)}
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
        );
      })}

      <div style={cardStyle}>
        <div style={titleStyle}>Select LoRA Model</div>
        <div style={itemsStyle}>
          {availableLoras.length === 0 && (
            <div style={{ color: isDark ? '#999' : '#666', fontSize: '12px', padding: '8px' }}>
              No LoRA models available for this configuration.
            </div>
          )}
          {availableLoras.map((lora) => {
            const isSelected = values.selectedLoraPath === lora.path;
            return (
              <label
                key={lora.id}
                style={{ ...labelBaseStyle, ...(isSelected ? checkedStyle : {}) }}
                onClick={(event) => {
                  event.preventDefault();
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
                {lora.id}
                <small style={{ ...subtitleStyle, color: isSelected ? 'rgba(255,255,255,0.85)' : 'inherit' }}>
                  {lora.path}
                </small>
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
