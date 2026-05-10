export const KimiK25Deployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/KimiK25ConfigGenerator/index.js.
  //
  // GPU requirements:
  //   H200: tp=8
  //   B300: tp=8
  //   MI300X: tp=4 (64 heads / 4 = 16 heads per GPU, AITER MLA requires heads_per_gpu % 16 == 0)
  //   MI325X: tp=4 (same constraint as MI300X)
  //   MI350X: tp=4 (same constraint as MI300X)
  //   MI355X: tp=4 (same constraint as MI300X)
  //
  // NVFP4 quantization is only supported on NVIDIA Blackwell (B300).
  // Speculative decoding is only supported on H200 and B300.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',   label: 'H200',   default: true  },
        { id: 'b300',   label: 'B300',   default: false },
        { id: 'mi300x', label: 'MI300X', default: false },
        { id: 'mi325x', label: 'MI325X', default: false },
        { id: 'mi350x', label: 'MI350X', default: false },
        { id: 'mi355x', label: 'MI355X', default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: (values) => {
        const hw = values.hardware;
        const isB300 = hw === 'b300';
        return [
          { id: 'int4',  label: 'INT4',  subtitle: 'initial model',   default: true },
          { id: 'nvfp4', label: 'NVFP4', subtitle: 'Blackwell only',  default: false, disabled: !isB300, disabledReason: 'NVFP4 only on B300' }
        ];
      }
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
    dpattention: {
      name: 'dpattention',
      title: 'DP Attention',
      items: [
        { id: 'disabled', label: 'Disabled', subtitle: 'Low Latency',     default: true  },
        { id: 'enabled',  label: 'Enabled',  subtitle: 'High Throughput', default: false }
      ]
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding',
      condition: (values) => values.hardware === 'h200' || values.hardware === 'b300',
      items: [
        { id: 'disabled', label: 'Disabled', default: true  },
        { id: 'enabled',  label: 'Enabled',  default: false }
      ]
    }
  };

  const modelConfigs = {
    h200:   { tp: 8 },
    b300:   { tp: 8 },
    mi300x: { tp: 4 },
    mi325x: { tp: 4 },
    mi350x: { tp: 4 },
    mi355x: { tp: 4 }
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

  // When hardware changes, re-resolve quantization defaults (NVFP4 only on B300).
  useEffect(() => {
    setValues(prev => {
      const next = { ...prev };
      for (const [key, option] of Object.entries(options)) {
        if (typeof option.getDynamicItems !== 'function') continue;
        const items = option.getDynamicItems(next);
        const current = items.find(i => i.id === next[key]);
        if (!current || current.disabled) {
          const fallback = items.find(i => i.default && !i.disabled) || items.find(i => !i.disabled);
          if (fallback) next[key] = fallback.id;
        }
      }
      return next;
    });
  }, [values.hardware]);

  const handleRadioChange = (optionName, value) => {
    setValues(prev => ({ ...prev, [optionName]: value }));
  };

  // Generate command - mirrors sgl-cookbook's config.generateCommand(values) exactly.
  const generateCommand = () => {
    const { hardware, quantization, speculative } = values;
    const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi350x' || hardware === 'mi355x';

    // NVFP4 is only supported on NVIDIA Blackwell (B300)
    if (quantization === 'nvfp4' && hardware !== 'b300') {
      return '# NVFP4 quantization is only supported on NVIDIA Blackwell GPUs (B300)';
    }

    // Speculative decoding only supported on H200 and B300
    if (speculative === 'enabled' && hardware !== 'h200' && hardware !== 'b300') {
      return '# Speculative Decoding for Kimi-K2.5 is only supported on H200 and B300';
    }

    // Model path depends on quantization
    const modelName = quantization === 'nvfp4'
      ? 'nvidia/Kimi-K2.5-NVFP4'
      : 'moonshotai/Kimi-K2.5';

    const hwConfig = modelConfigs[hardware];
    const tpValue = hwConfig.tp;

    let cmd = '';

    // AMD ROCm environment variables
    if (isAMD) {
      cmd += 'SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 ';
    }

    // Speculative decoding env var
    if (speculative === 'enabled') {
      cmd += 'SGLANG_ENABLE_SPEC_V2=1 ';
    }

    // If we added any env vars above, break to a new line for readability
    if (isAMD || speculative === 'enabled') {
      cmd += '\\\n';
    }

    cmd += 'sglang serve \\\n';
    cmd += `  --model-path ${modelName}`;
    cmd += ` \\\n  --tp ${tpValue}`;
    cmd += ' \\\n  --trust-remote-code';

    // DP Attention: --dp matches --tp
    if (values.dpattention === 'enabled') {
      cmd += ` \\\n  --dp ${tpValue} \\\n  --enable-dp-attention`;
    }

    // Reasoning parser
    if (values.reasoning === 'enabled') {
      cmd += ' \\\n  --reasoning-parser kimi_k2';
    }

    // Tool call parser
    if (values.toolcall === 'enabled') {
      cmd += ' \\\n  --tool-call-parser kimi_k2';
    }

    // Speculative decoding (EAGLE3)
    if (speculative === 'enabled') {
      cmd += ' \\\n  --speculative-algorithm EAGLE3 \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4 \\\n  --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3';
    }

    // AMD: FP8 KV cache for memory efficiency
    if (isAMD) {
      cmd += ' \\\n  --kv-cache-dtype fp8_e4m3';
    }

    cmd += ' \\\n  --host 0.0.0.0 \\\n  --port 30000';

    return cmd;
  };

  // Styles
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
