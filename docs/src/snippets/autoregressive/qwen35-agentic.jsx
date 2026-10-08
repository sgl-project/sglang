export const qwen35AgenticRecipes = {
  "model": "nvidia/Qwen3.5-397B-A17B-NVFP4",
  "image": "lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729",
  "commonFlags": [
    "--enable-cache-report",
    "--trust-remote-code",
    "--quantization modelopt_fp4",
    "--kv-cache-dtype fp8_e4m3",
    "--pipeline-parallel-size 1",
    "--data-parallel-size 1",
    "--mamba-radix-cache-strategy extra_buffer",
    "--mamba-ssm-dtype bfloat16",
    "--context-length 262144",
    "--page-size 64",
    "--attention-backend trtllm_mha",
    "--speculative-algorithm NEXTN",
    "--speculative-eagle-topk 1",
    "--mem-fraction-static 0.85",
    "--watchdog-timeout 1000000",
    "--enable-hierarchical-cache",
    "--hicache-write-policy write_back",
    "--hicache-io-backend kernel",
    "--hicache-mem-layout page_first_direct",
    "--enable-metrics",
    "--enable-linear-replayssm-spec"
  ],
  "commonEnv": [
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    "NCCL_CUMEM_ENABLE=1",
    "NCCL_NVLS_ENABLE=0",
    "SGLANG_MOE_NVFP4_DISPATCH=1",
    "SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1",
    "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1",
    "FLASHINFER_DISABLE_VERSION_CHECK=1",
    "SGLANG_DG_CACHE_DIR=/configs/deepgemm-cache",
    "FLASHINFER_WORKSPACE_BASE=/configs/flashinfer-cache",
    "SGLANG_CACHE_DIR=/configs/sglang-cache",
    "SGLANG_FLASHINFER_AUTOTUNE_CACHE=1",
    "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
    "SGLANG_HEALTH_CHECK_TIMEOUT=1800",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0",
    "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER=1"
  ],
  "profiles": [
    {
      "id": "tp4-mtp6-c4",
      "gpus": 4,
      "concurrency": 4,
      "flags": [
        "--tensor-parallel-size 4",
        "--expert-parallel-size 1",
        "--moe-dense-tp-size 4",
        "--moe-a2a-backend none",
        "--load-balance-method round_robin",
        "--mamba-track-interval 8192",
        "--mamba-max-states-per-path 3",
        "--max-mamba-cache-size 512",
        "--moe-runner-backend flashinfer_cutedsl",
        "--linear-attn-decode-backend triton",
        "--disable-shared-experts-fusion",
        "--speculative-num-steps 6",
        "--speculative-num-draft-tokens 7",
        "--speculative-moe-runner-backend flashinfer_cutedsl",
        "--speculative-moe-a2a-backend none",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--max-running-requests 1",
        "--pp-max-micro-batch-size 1",
        "--prefill-max-requests 1",
        "--cuda-graph-max-bs-decode 1",
        "--cuda-graph-bs-decode 1",
        "--disable-prefill-cuda-graph",
        "--stream-interval 20",
        "--decode-log-interval 10",
        "--weight-loader-prefetch-checkpoints",
        "--weight-loader-prefetch-num-threads 4",
        "--hicache-size 32",
        "--disable-attn-tp-gather"
      ],
      "env": [
        "SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS=1",
        "SGLANG_ENABLE_JIT_DEEPGEMM=true"
      ]
    },
    {
      "id": "tp2-mtp7-c36",
      "gpus": 2,
      "concurrency": 36,
      "flags": [
        "--tensor-parallel-size 2",
        "--expert-parallel-size 1",
        "--moe-dense-tp-size 2",
        "--moe-a2a-backend none",
        "--load-balance-method round_robin",
        "--mamba-track-interval 8192",
        "--mamba-max-states-per-path 3",
        "--max-mamba-cache-size 1536",
        "--moe-runner-backend flashinfer_trtllm",
        "--linear-attn-decode-backend triton",
        "--disable-shared-experts-fusion",
        "--speculative-num-steps 7",
        "--speculative-num-draft-tokens 8",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--speculative-moe-a2a-backend none",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--prefill-decode-interval 0",
        "--max-running-requests 4",
        "--pp-max-micro-batch-size 4",
        "--prefill-max-requests 4",
        "--cuda-graph-max-bs-decode 4",
        "--cuda-graph-bs-decode 1 2 3 4",
        "--stream-interval 20",
        "--decode-log-interval 10",
        "--weight-loader-prefetch-checkpoints",
        "--weight-loader-prefetch-num-threads 4",
        "--hicache-size 128",
        "--disable-attn-tp-gather"
      ],
      "env": [
        "SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS=1",
        "SGLANG_ENABLE_JIT_DEEPGEMM=true"
      ]
    },
    {
      "id": "tp2-mtp5-c40",
      "gpus": 2,
      "concurrency": 40,
      "flags": [
        "--tensor-parallel-size 2",
        "--expert-parallel-size 1",
        "--moe-dense-tp-size 2",
        "--moe-a2a-backend none",
        "--load-balance-method round_robin",
        "--mamba-track-interval 8192",
        "--mamba-max-states-per-path 3",
        "--max-mamba-cache-size 1536",
        "--moe-runner-backend flashinfer_trtllm",
        "--linear-attn-decode-backend triton",
        "--disable-shared-experts-fusion",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--speculative-moe-a2a-backend none",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--prefill-decode-interval 0",
        "--max-running-requests 2",
        "--pp-max-micro-batch-size 2",
        "--prefill-max-requests 2",
        "--cuda-graph-max-bs-decode 2",
        "--cuda-graph-bs-decode 1 2",
        "--stream-interval 20",
        "--decode-log-interval 10",
        "--weight-loader-prefetch-checkpoints",
        "--weight-loader-prefetch-num-threads 4",
        "--hicache-size 128",
        "--disable-attn-tp-gather"
      ],
      "env": [
        "SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS=1",
        "SGLANG_ENABLE_JIT_DEEPGEMM=true"
      ]
    },
    {
      "id": "tp2-mtp7-c44",
      "gpus": 2,
      "concurrency": 44,
      "flags": [
        "--tensor-parallel-size 2",
        "--expert-parallel-size 1",
        "--moe-dense-tp-size 2",
        "--moe-a2a-backend none",
        "--load-balance-method round_robin",
        "--mamba-track-interval 8192",
        "--mamba-max-states-per-path 3",
        "--max-mamba-cache-size 1536",
        "--moe-runner-backend flashinfer_trtllm",
        "--linear-attn-decode-backend triton",
        "--disable-shared-experts-fusion",
        "--speculative-num-steps 7",
        "--speculative-num-draft-tokens 8",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--speculative-moe-a2a-backend none",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--prefill-decode-interval 0",
        "--max-running-requests 8",
        "--pp-max-micro-batch-size 8",
        "--prefill-max-requests 8",
        "--cuda-graph-max-bs-decode 8",
        "--cuda-graph-bs-decode 1 2 3 4 5 6 7 8",
        "--stream-interval 20",
        "--decode-log-interval 10",
        "--weight-loader-prefetch-checkpoints",
        "--weight-loader-prefetch-num-threads 4",
        "--hicache-size 128",
        "--disable-attn-tp-gather"
      ],
      "env": [
        "SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS=1",
        "SGLANG_ENABLE_JIT_DEEPGEMM=true"
      ]
    },
    {
      "id": "tp2-ep2-mtp3-c64",
      "gpus": 2,
      "concurrency": 64,
      "flags": [
        "--tensor-parallel-size 2",
        "--expert-parallel-size 2",
        "--mamba-track-interval 1048576",
        "--mamba-max-states-per-path 1",
        "--max-mamba-cache-size 512",
        "--moe-runner-backend flashinfer_cutedsl",
        "--linear-attn-prefill-backend flashinfer",
        "--speculative-num-steps 3",
        "--speculative-num-draft-tokens 4",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--linear-replayssm-cache-len 8",
        "--max-prefill-tokens 16384",
        "--chunked-prefill-size 16384",
        "--max-running-requests 224",
        "--cuda-graph-max-bs-decode 224",
        "--tokenizer-worker-num 6",
        "--stream-interval 50",
        "--scheduler-recv-interval 10",
        "--allow-auto-truncate",
        "--hicache-ratio 1.8",
        "--weight-loader-drop-cache-after-load"
      ],
      "env": [
        "SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS=2",
        "SGLANG_ENABLE_JIT_DEEPGEMM=false"
      ]
    }
  ]
};

export const Qwen35AgenticCommands = ({ recipes, isDark }) => {
  const [profileId, setProfileId] = useState(recipes.profiles[0].id);
  const [role, setRole] = useState('worker');
  const [copyState, setCopyState] = useState('');
  const profile = recipes.profiles.find(item => item.id === profileId);
  const flagValue = name => {
    const flag = profile.flags.find(item => item.startsWith(`--${name} `));
    return flag ? flag.slice(name.length + 3) : null;
  };

  const generateCommand = () => {
    if (role === 'router') {
      return [
        '# Run on the host that serves client traffic. Replace 127.0.0.1 if the worker runs on another host.',
        'python3 -m sglang_router.launch_router \\',
        '  --worker-urls http://127.0.0.1:6100 \\',
        '  --host 0.0.0.0 \\',
        '  --port 8000',
      ].join('\n');
    }
    const flags = [
      `--model-path /model`,
      `--served-model-name ${recipes.model}`,
      ...recipes.commonFlags,
      ...profile.flags,
      '--host 0.0.0.0',
      '--port 6100',
    ];
    return [
      `# ${profile.id}; use image ${recipes.image}`,
      '# Run inside the staged container (see the setup above).',
      `export CUDA_VISIBLE_DEVICES=${profile.gpus === 4 ? '0,1,2,3' : '0,1'}`,
      ...recipes.commonEnv.map(value => `export ${value}`),
      ...profile.env.map(value => `export ${value}`),
      'sglang serve \\\n  ' + flags.join(' \\\n  '),
    ].join('\n');
  };

  const command = generateCommand();
  useEffect(() => { setCopyState(''); }, [command]);
  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopyState('Copied');
    } catch {
      setCopyState('Copy failed; select the command below to copy it.');
    }
  };
  const card = {
    padding: '10px 12px', border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
    borderLeft: '3px solid #D45D44', borderRadius: '4px', background: isDark ? '#1f2937' : '#fff',
    color: isDark ? '#e5e7eb' : 'inherit', fontSize: '13px',
  };
  const control = {
    padding: '6px 8px', border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`,
    borderRadius: '3px', background: isDark ? '#374151' : '#fff', color: 'inherit',
    fontSize: '13px', maxWidth: '100%',
  };
  const ep = flagValue('expert-parallel-size');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <p style={{ margin: '6px 0', fontSize: '13px' }}>
        Use the pinned serving container after completing the{' '}
        <a href="#3-4-b300-nvfp4-agentic-aggregated-serving-with-hicache-and-mtp" style={{ color: '#D45D44', textDecoration: 'underline' }}>checkpoint and container setup below</a>.
      </p>
      <div style={card}>
        <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '12px' }}>
          <strong style={{ minWidth: '140px' }}>Serving variant</strong>
          <select aria-label="Serving variant" value={profileId} onChange={event => setProfileId(event.target.value)} style={{ ...control, flex: 1 }}>
            {recipes.profiles.map(item => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
        </label>
        <p style={{ margin: '8px 0 0', lineHeight: '1.5' }}>
          One aggregated worker on {profile.gpus} GPUs of a single B300 node: TP{flagValue('tensor-parallel-size')}, EP{ep};
          {' '}MTP {flagValue('speculative-num-steps')} steps; running-request limit {flagValue('max-running-requests')};
          {' '}HiCache host pool {flagValue('hicache-size') ? `${flagValue('hicache-size')} GB` : `${flagValue('hicache-ratio')}x the device KV pool`}.
        </p>
        <p style={{ margin: '4px 0 0', opacity: 0.8 }}>
          The c suffix identifies the client concurrency the variant was measured at; the serving limits come from the selected recipe.
        </p>
      </div>
      <div style={{ ...card, display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
        <strong style={{ minWidth: '140px' }}>Command role</strong>
        {[
          ['worker', 'Worker'], ['router', 'Router'],
        ].map(([id, label]) => (
          <button key={id} type="button" aria-pressed={role === id} onClick={() => setRole(id)}
            style={{ ...control, cursor: 'pointer', ...(role === id ? { background: '#D45D44', color: 'white', borderColor: '#D45D44' } : {}) }}>
            {label}
          </button>
        ))}
      </div>
      <div style={card}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
          <strong>{role === 'router' ? 'Router' : 'Worker'} command</strong>
          <button type="button" onClick={copyCommand} style={{ ...control, marginLeft: 'auto', cursor: 'pointer' }}>Copy command</button>
          <span role="status" aria-live="polite">{copyState}</span>
        </div>
        <pre aria-label="Generated agentic command" style={{
          padding: '12px 16px', margin: 0, background: isDark ? '#111827' : '#f5f5f5',
          borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
          fontSize: '12px', lineHeight: '1.5', whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: '520px',
        }}>{command}</pre>
      </div>
    </div>
  );
};
