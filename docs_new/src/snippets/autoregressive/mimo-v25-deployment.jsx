export const MiMoV25Deployment = ({ defaultModel = 'standard' }) => {
  const options = {
    model: {
      name: 'model',
      title: 'Model Variant',
      items: [
        { id: 'standard', label: 'MiMo-V2.5', default: defaultModel === 'standard' },
        { id: 'pro', label: 'MiMo-V2.5-Pro', default: defaultModel === 'pro' }
      ]
    },
    platform: {
      name: 'platform',
      title: 'Platform',
      items: [
        { id: 'nvidia', label: 'NVIDIA CUDA', default: true }
      ]
    }
  };

  const getInitialState = () => {
    const initialState = {};
    Object.entries(options).forEach(([key, option]) => {
      const defaultItem = option.items.find(item => item.default && !item.disabled) || option.items.find(item => !item.disabled) || option.items[0];
      initialState[key] = defaultItem.id;
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
    setValues(prev => ({ ...prev, [optionName]: value }));
  };

  const generateStandardCommand = () => {
    let cmd = 'sglang serve \\\n';
    cmd += '  --model-path XiaomiMiMo/MiMo-V2.5 \\\n';
    cmd += '  --served-model-name mimo-v2.5 \\\n';
    cmd += '  --log-level-http warning \\\n';
    cmd += '  --enable-cache-report \\\n';
    cmd += '  --pp-size 1 \\\n';
    cmd += '  --dp-size 2 \\\n';
    cmd += '  --tp-size 8 \\\n';
    cmd += '  --enable-dp-attention \\\n';
    cmd += '  --moe-a2a-backend deepep \\\n';
    cmd += '  --deepep-mode auto \\\n';
    cmd += '  --decode-log-interval 1 \\\n';
    cmd += '  --page-size 1 \\\n';
    cmd += '  --host 0.0.0.0 \\\n';
    cmd += '  --port 9001 \\\n';
    cmd += '  --trust-remote-code \\\n';
    cmd += '  --watchdog-timeout 1000000 \\\n';
    cmd += '  --mem-fraction-static 0.65 \\\n';
    cmd += '  --chunked-prefill-size 16384 \\\n';
    cmd += '  --reasoning-parser qwen3 \\\n';
    cmd += '  --tool-call-parser mimo \\\n';
    cmd += '  --context-length 262144 \\\n';
    cmd += '  --collect-tokens-histogram \\\n';
    cmd += '  --enable-metrics \\\n';
    cmd += '  --load-balance-method round_robin \\\n';
    cmd += '  --allow-auto-truncate \\\n';
    cmd += '  --enable-metrics-for-all-schedulers \\\n';
    cmd += '  --quantization fp8 \\\n';
    cmd += '  --skip-server-warmup \\\n';
    cmd += '  --moe-dense-tp-size 1 \\\n';
    cmd += '  --enable-dp-lm-head \\\n';
    cmd += '  --disable-tokenizer-batch-decode \\\n';
    cmd += '  --mm-enable-dp-encoder \\\n';
    cmd += '  --attention-backend fa3 \\\n';
    cmd += '  --mm-attention-backend fa3';
    return cmd;
  };

  const generateProCommand = () => {
    let cmd = 'SGLANG_ENABLE_SPEC_V2=1 \\\n';
    cmd += 'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \\\n';
    cmd += 'sglang serve \\\n';
    cmd += '  --model-path XiaomiMiMo/MiMo-V2.5-Pro \\\n';
    cmd += '  --trust-remote-code \\\n';
    cmd += '  --pp-size 1 \\\n';
    cmd += '  --dp-size 2 \\\n';
    cmd += '  --ep-size 16 \\\n';
    cmd += '  --tp-size 16 \\\n';
    cmd += '  --moe-dense-tp-size 1 \\\n';
    cmd += '  --enable-dp-attention \\\n';
    cmd += '  --moe-a2a-backend deepep \\\n';
    cmd += '  --dist-init-addr ${LWS_LEADER_IP}:20000 \\\n';
    cmd += '  --node-rank ${LWS_WORKER_INDEX} \\\n';
    cmd += '  --nnodes ${LWS_GROUP_SIZE} \\\n';
    cmd += '  --page-size 64 \\\n';
    cmd += '  --attention-backend fa3 \\\n';
    cmd += '  --quantization fp8 \\\n';
    cmd += '  --mem-fraction-static 0.7 \\\n';
    cmd += '  --max-running-requests 128 \\\n';
    cmd += '  --cuda-graph-max-bs 64 \\\n';
    cmd += '  --chunked-prefill-size 32768 \\\n';
    cmd += '  --context-length 1048576 \\\n';
    cmd += '  --tokenizer-worker-num 64 \\\n';
    cmd += '  --speculative-algorithm EAGLE \\\n';
    cmd += '  --speculative-num-steps 3 \\\n';
    cmd += '  --speculative-eagle-topk 1 \\\n';
    cmd += '  --speculative-num-draft-tokens 4 \\\n';
    cmd += '  --enable-multi-layer-eagle \\\n';
    cmd += '  --host 0.0.0.0 \\\n';
    cmd += '  --port 9001 \\\n';
    cmd += '  --reasoning-parser mimo \\\n';
    cmd += '  --tool-call-parser mimo \\\n';
    cmd += '  --watchdog-timeout 3600 \\\n';
    cmd += '  --model-loader-extra-config \'{"enable_multithread_load": "true", "num_threads": 64}\' \\\n';
    cmd += '  --swa-full-tokens-ratio 0.3';
    return cmd;
  };

  const generateCommand = () => values.model === 'pro' ? generateProCommand() : generateStandardCommand();

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
      {Object.entries(options).map(([key, option]) => (
        <div key={key} style={cardStyle}>
          <div style={titleStyle}>{option.title}</div>
          <div style={itemsStyle}>
            {option.items.map(item => {
              const isChecked = values[option.name] === item.id;
              const isDisabled = !!item.disabled;
              return (
                <label key={item.id} style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}>
                  <input type="radio" name={option.name} value={item.id} checked={isChecked} disabled={isDisabled} onChange={() => !isDisabled && handleRadioChange(option.name, item.id)} style={{ display: 'none' }} />
                  {item.label}
                </label>
              );
            })}
          </div>
        </div>
      ))}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
