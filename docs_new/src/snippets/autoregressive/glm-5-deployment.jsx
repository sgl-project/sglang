export const GLM5Deployment = () => {
  // Config mirrors sgl-cookbook src/components/autoregressive/GLM5ConfigGenerator/index.js.
  //
  // Supported quantization per hardware:
  //   H100 / H200 / MI300X / MI325X / MI355X → BF16 (AMD only) + FP8 (NV only)
  //   B200 → NVFP4 (default), FP8, BF16
  //
  // BF16 always needs 2x GPUs compared to FP8. AMD only supports BF16.
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [
        { id: 'h200',   label: 'H200',            default: true  },
        { id: 'b200',   label: 'B200',            default: false },
        { id: 'h100',   label: 'H100',            default: false },
        { id: 'mi300x', label: 'MI300X/MI325X',   default: false },
        { id: 'mi355x', label: 'MI355X',          default: false }
      ]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: (values) => {
        const hw = values.hardware;
        const isAMD = hw === 'mi300x' || hw === 'mi355x';
        const isB200 = hw === 'b200';
        return [
          { id: 'bf16',  label: 'BF16',  subtitle: 'Full Weights',        default: isAMD },
          { id: 'fp8',   label: 'FP8',   subtitle: 'High Throughput',     default: !isAMD && !isB200, disabled: isAMD,  disabledReason: 'FP8 not verified on AMD' },
          { id: 'nvfp4', label: 'NVFP4', subtitle: 'Highest Throughput',  default: isB200,            disabled: !isB200, disabledReason: 'NVFP4 only on B200' }
        ];
      }
    },
    reasoning: {
      name: 'reasoning',
      title: 'Reasoning Parser',
      condition: (values) => values.quantization !== 'nvfp4',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      condition: (values) => values.quantization !== 'nvfp4',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    },
    dpattention: {
      name: 'dpattention',
      title: 'DP Attention',
      condition: (values) => values.quantization !== 'nvfp4',
      items: [
        { id: 'disabled', label: 'Disabled', subtitle: 'Low Latency',      default: true },
        { id: 'enabled',  label: 'Enabled',  subtitle: 'High Throughput',  default: false }
      ]
    },
    speculative: {
      name: 'speculative',
      title: 'Speculative Decoding',
      condition: (values) => values.hardware !== 'mi300x' && values.hardware !== 'mi355x' && values.quantization !== 'nvfp4',
      items: [
        { id: 'disabled', label: 'Disabled', default: false },
        { id: 'enabled',  label: 'Enabled',  default: true  }
      ]
    }
  };

  // BF16 always 2× the GPUs of FP8.
  const modelConfigs = {
    h100:   { fp8: { tp: 16, mem: 0.85 }, bf16: { tp: 32, mem: 0.85 } },
    h200:   { fp8: { tp: 8,  mem: 0.85 }, bf16: { tp: 16, mem: 0.85 } },
    b200:   { nvfp4: { tp: 4, mem: 0.9 }, fp8: { tp: 8, mem: 0.9 }, bf16: { tp: 16, mem: 0.9 } },
    mi300x: { bf16: { tp: 8, mem: 0.80 } },
    mi355x: { bf16: { tp: 8, mem: 0.80 } }
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

  // When hardware changes, re-resolve quantization (and downstream) defaults to
  // stay consistent (AMD→BF16, B200→NVFP4, etc.).
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

  const generateCommand = () => {
    const { hardware, quantization } = values;
    const isAMD = hardware === 'mi300x' || hardware === 'mi355x';
    const isNVFP4 = quantization === 'nvfp4';
    const effectiveQuant = isAMD ? 'bf16' : quantization;

    let modelName;
    if (isNVFP4) {
      modelName = 'nvidia/GLM-5-NVFP4';
    } else {
      const suffix = effectiveQuant === 'fp8' ? '-FP8' : '';
      modelName = `zai-org/GLM-5${suffix}`;
    }

    const hwConfig = modelConfigs[hardware][effectiveQuant];
    const tpValue = hwConfig.tp;
    const memFraction = hwConfig.mem;

    let cmd = 'sglang serve \\\n';
    cmd += `  --model-path ${modelName}`;
    cmd += ` \\\n  --tp ${tpValue}`;

    // NVFP4 B200: trtllm NSA backends, flashinfer fusion, FP8 KV cache.
    if (isNVFP4) {
      cmd += ' \\\n  --trust-remote-code';
      cmd += ' \\\n  --quantization modelopt_fp4';
      cmd += ' \\\n  --kv-cache-dtype fp8_e4m3';
      cmd += ' \\\n  --nsa-decode-backend trtllm';
      cmd += ' \\\n  --nsa-prefill-backend trtllm';
      cmd += ' \\\n  --moe-runner-backend flashinfer_trtllm';
      cmd += ' \\\n  --enable-flashinfer-allreduce-fusion';
      cmd += ' \\\n  --enable-dp-lm-head';
      cmd += ' \\\n  --disable-radix-cache';
      cmd += ' \\\n  --max-prefill-tokens 32768';
      cmd += ' \\\n  --chunked-prefill-size 32768';
      cmd += ` \\\n  --mem-fraction-static ${memFraction}`;
      cmd += ' \\\n  --scheduler-recv-interval 10';
      cmd += ' \\\n  --tokenizer-worker-num 6';
      return cmd;
    }

    // AMD-specific: NSA tilelang backend.
    if (isAMD) {
      cmd += ' \\\n  --trust-remote-code';
      cmd += ' \\\n  --nsa-prefill-backend tilelang';
      cmd += ' \\\n  --nsa-decode-backend tilelang';
      cmd += ' \\\n  --chunked-prefill-size 131072';
      cmd += ' \\\n  --watchdog-timeout 1200';
    }

    if (values.dpattention === 'enabled') {
      cmd += ` \\\n  --dp ${tpValue} \\\n  --enable-dp-attention`;
    }
    if (values.reasoning === 'enabled') cmd += ' \\\n  --reasoning-parser glm45';
    if (values.toolcall  === 'enabled') cmd += ' \\\n  --tool-call-parser glm47';
    if (values.speculative === 'enabled') {
      cmd += ' \\\n  --speculative-algorithm EAGLE';
      cmd += ' \\\n  --speculative-num-steps 3';
      cmd += ' \\\n  --speculative-eagle-topk 1';
      cmd += ' \\\n  --speculative-num-draft-tokens 4';
    }

    // B200 FP8: consolidated optimized flags.
    if (hardware === 'b200' && effectiveQuant === 'fp8') {
      cmd += ' \\\n  --ep 1';
      cmd += ' \\\n  --quantization fp8';
      cmd += ' \\\n  --attention-backend nsa';
      cmd += ' \\\n  --nsa-decode-backend trtllm';
      cmd += ' \\\n  --nsa-prefill-backend trtllm';
      cmd += ' \\\n  --moe-runner-backend flashinfer_trtllm';
      cmd += ' \\\n  --enable-flashinfer-allreduce-fusion';
    }

    cmd += ` \\\n  --mem-fraction-static ${memFraction}`;
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
