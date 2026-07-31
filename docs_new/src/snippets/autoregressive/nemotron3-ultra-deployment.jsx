export const Nemotron3UltraDeployment = () => {
  const MODEL_PATHS = {
    bf16: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
    nvfp4: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4',
  };

  // Verified {model, hardware, tp} combinations. Any tuple not in this list is
  // blocked by `generateCommand` so the UI cannot emit an unvalidated launch.
  // Keep in sync with the "Supported GPUs" section of Nemotron3-Ultra.mdx.
  const VERIFIED_CONFIGS = [
    { model: 'bf16',  hardware: 'h100',  tp: '16', multinode: true },
    { model: 'bf16',  hardware: 'h200',  tp: '16', multinode: true },
    { model: 'bf16',  hardware: 'b200',  tp: '8'  },
    { model: 'bf16',  hardware: 'b300',  tp: '8'  },
    { model: 'nvfp4', hardware: 'b200',  tp: '4'  },
    { model: 'nvfp4', hardware: 'b200',  tp: '8'  },
    { model: 'nvfp4', hardware: 'b300',  tp: '4'  },
    { model: 'nvfp4', hardware: 'b300',  tp: '8'  },
    { model: 'nvfp4', hardware: 'gb200', tp: '4'  },
    { model: 'nvfp4', hardware: 'gb300', tp: '4'  },
  ];

  const findVerified = (model, hardware, tp) =>
    VERIFIED_CONFIGS.find((c) => c.model === model && c.hardware === hardware && c.tp === tp);

  const verifiedHardwareForModel = (model) =>
    [...new Set(VERIFIED_CONFIGS.filter((c) => c.model === model).map((c) => c.hardware))];

  const verifiedTpForModelHardware = (model, hardware) =>
    [...new Set(VERIFIED_CONFIGS.filter((c) => c.model === model && c.hardware === hardware).map((c) => c.tp))];

  // DP attention is verified at dp=2 for BF16, and dp in {2,4,8} for NVFP4. SGLang
  // requires tp_size % dp_size == 0, so dp is capped at both the selected TP and the
  // max verified TP for this model+hardware (whichever is smaller).
  const dpCandidatesForModel = (model) => (model === 'bf16' ? ['2'] : ['2', '4', '8']);

  const maxVerifiedTpForModelHardware = (model, hardware) => {
    const tps = verifiedTpForModelHardware(model, hardware).map(Number);
    return tps.length ? Math.max(...tps) : 0;
  };

  const verifiedDpForModelHardwareTp = (model, hardware, tp) => {
    const cap = Math.min(Number(tp) || 0, maxVerifiedTpForModelHardware(model, hardware));
    return dpCandidatesForModel(model).filter((d) => Number(d) <= cap);
  };

  const options = {
    model: {
      name: 'model',
      title: 'Model',
      items: [
        { id: 'bf16',   label: 'BF16',   default: false },
        { id: 'nvfp4',  label: 'NVFP4',  default: true,  subtitle: 'Blackwell only' },
      ]
    },
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      getDynamicItems: (values) => {
        const supported = new Set(verifiedHardwareForModel(values.model));
        const base = [
          { id: 'h100',  label: 'H100',  default: false },
          { id: 'h200',  label: 'H200',  default: false },
          { id: 'b200',  label: 'B200',  default: true  },
          { id: 'gb200', label: 'GB200', default: false },
          { id: 'b300',  label: 'B300',  default: false },
          { id: 'gb300', label: 'GB300', default: false }
        ];
        return base.map((it) => {
          const ok = supported.has(it.id);
          return {
            ...it,
            disabled: !ok,
            disabledReason: ok ? '' : `${values.model.toUpperCase()} is not verified on ${it.label}`
          };
        });
      }
    },
    tp: {
      name: 'tp',
      title: 'Tensor Parallel (TP)',
      getDynamicItems: (values) => {
        const supported = new Set(verifiedTpForModelHardware(values.model, values.hardware));
        const base = [
          { id: '4',  label: 'TP=4'  },
          { id: '8',  label: 'TP=8'  },
          { id: '16', label: 'TP=16', subtitle: '2-node' }
        ];
        return base.map((it) => {
          const ok = supported.has(it.id);
          return {
            ...it,
            default: ok && supported.size === 1,
            disabled: !ok,
            disabledReason: ok ? '' : `TP=${it.id} is not verified for ${values.model.toUpperCase()} on ${values.hardware.toUpperCase()}`
          };
        });
      }
    },
    ep: {
      name: 'ep',
      title: 'Expert Parallel (EP)',
      items: [
        { id: 'enabled',  label: 'Enabled', subtitle: 'EP = TP' },
        { id: 'disabled', label: 'Disabled', default: true }
      ],
      // This MoE only supports ep_size == 1 or ep_size == tp_size; when on, EP equals TP.
      commandRule: (value, state) => value === 'enabled' ? `--ep ${state.tp}` : null
    },
    dpattention: {
      name: 'dpattention',
      title: 'DP Attention',
      getDynamicItems: (values) => {
        const allowed = new Set(verifiedDpForModelHardwareTp(values.model, values.hardware, values.tp));
        const base = [
          { id: 'disabled', label: 'Disabled', subtitle: 'Low latency',    default: true },
          { id: '2',        label: 'DP=2',      subtitle: 'High throughput' },
          { id: '4',        label: 'DP=4',      subtitle: 'High throughput' },
          { id: '8',        label: 'DP=8',      subtitle: 'High throughput' }
        ];
        return base.map((it) => {
          if (it.id === 'disabled') return it;
          const ok = allowed.has(it.id);
          return {
            ...it,
            disabled: !ok,
            disabledReason: ok ? '' : `DP=${it.id} is not verified for ${values.model.toUpperCase()} on ${values.hardware.toUpperCase()} at TP=${values.tp}`
          };
        });
      },
      // dp_size must divide tp_size; only emit when the selected DP is valid for the current TP.
      commandRule: (value, state) =>
        value && value !== 'disabled' &&
        dpCandidatesForModel(state.model).includes(value) &&
        Number(value) <= Number(state.tp)
          ? `--dp ${value} \\\n  --enable-dp-attention`
          : null
    },
    mtp: {
      name: 'mtp',
      title: 'Multi-token Prediction (MTP)',
      items: [
        { id: 'enabled',  label: 'Enabled',  default: true  },
        { id: 'disabled', label: 'Disabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--speculative-algorithm EAGLE \\\n  --speculative-num-steps 5 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 6' : null
    },
    kvcache: {
      name: 'kvcache',
      title: 'KV Cache DType',
      items: [
        { id: 'none',     label: 'None',     default: true  },
        { id: 'fp8_e4m3', label: 'fp8_e4m3', default: false },
        { id: 'bf16',     label: 'bf16',     default: false }
      ]
    },
    mambabackend: {
      name: 'mambabackend',
      title: 'Mamba Backend',
      items: [
        { id: 'triton',     label: 'Triton',     subtitle: 'Default', default: true  },
        { id: 'flashinfer', label: 'FlashInfer', subtitle: 'Faster',  default: false }
      ],
      commandRule: (value) => value === 'flashinfer' ? '--mamba-backend flashinfer' : null
    },
    mambassmdtype: {
      name: 'mambassmdtype',
      title: 'Mamba SSM DType',
      items: [
        { id: 'default', label: 'Default', subtitle: 'Model config', default: true  },
        { id: 'float16', label: 'float16', subtitle: 'Less memory',   default: false }
      ],
      commandRule: (value) => value === 'float16' ? '--mamba-ssm-dtype float16' : null
    },
    mambastochasticrounding: {
      name: 'mambastochasticrounding',
      title: 'Mamba Stochastic Rounding',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  subtitle: 'FP16 SSM' }
      ],
      commandRule: (value, state) =>
        value === 'enabled' && state.mambassmdtype === 'float16'
          ? '--enable-mamba-cache-stochastic-rounding'
          : null
    },
    thinking: {
      name: 'thinking',
      title: 'Reasoning Parser',
      items: [
        { id: 'enabled',  label: 'Enabled',  default: true  },
        { id: 'disabled', label: 'Disabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--reasoning-parser nemotron_3' : null
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [
        { id: 'enabled',  label: 'Enabled',  default: true  },
        { id: 'disabled', label: 'Disabled', default: false }
      ],
      commandRule: (value) => value === 'enabled' ? '--tool-call-parser qwen3_coder' : null
    }
  };

  const renderVerifiedMatrix = () => {
    const byModel = {};
    for (const c of VERIFIED_CONFIGS) {
      (byModel[c.model] ||= []).push(c);
    }
    return Object.entries(byModel)
      .map(([m, cs]) => {
        const lines = cs.map((c) => {
          const node = c.multinode ? ', 2-node' : '';
          return `#     - ${c.hardware.toUpperCase()} @ TP=${c.tp}${node}`;
        });
        return `#   ${m.toUpperCase()}:\n${lines.join('\n')}`;
      })
      .join('\n');
  };

  const generateCommand = (values) => {
    const { tp, kvcache, model, hardware } = values;
    const cfg = findVerified(model, hardware, tp);

    // Block any combination that is not in the verified support matrix.
    if (!cfg) {
      return [
        `# ERROR: ${model.toUpperCase()} on ${hardware.toUpperCase()} with TP=${tp} is not a verified configuration.`,
        `# The launch command has been suppressed to avoid running an unvalidated setup.`,
        `#`,
        `# Verified configurations:`,
        renderVerifiedMatrix(),
      ].join('\n');
    }

    const modelPath = MODEL_PATHS[model] || MODEL_PATHS['bf16'];

    let cmd = `python3 -m sglang.launch_server \\\n`;
    cmd += `  --model-path ${modelPath} \\\n`;
    cmd += `  --trust-remote-code \\\n`;
    cmd += `  --tp ${tp} \\\n`;

    for (const [key, option] of Object.entries(options)) {
      if (option.commandRule) {
        const rule = option.commandRule(values[key], values);
        if (rule) {
          cmd += `  ${rule} \\\n`;
        }
      }
    }

    cmd += `  --mamba-radix-cache-strategy extra_buffer \\\n`;
    if (['b200', 'gb200', 'b300', 'gb300'].includes(hardware)) {
      cmd += `  --attention-backend trtllm_mha \\\n`;
    }

    if (kvcache && kvcache !== 'none') {
      cmd += `  --kv-cache-dtype ${kvcache} \\\n`;
    }

    if (cfg.multinode) {
      cmd += `  --dist-init-addr <head-node-ip>:5000 \\\n`;
      cmd += `  --nnodes 2 \\\n`;
      cmd += `  --node-rank <0|1> \\\n`;
    }

    cmd = cmd.trimEnd();
    if (cmd.endsWith('\\')) {
      cmd = cmd.slice(0, -1).trimEnd();
    }

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
      const items = option.getDynamicItems
        ? option.getDynamicItems(initialState)
        : option.items || [];
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
