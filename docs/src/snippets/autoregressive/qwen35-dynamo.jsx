export const qwen35DynamoRecipes = {
  "model": "nvidia/Qwen3.5-397B-A17B-NVFP4-V2",
  "image": "lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928",
  "commonFlags": [
    "--model-path /model",
    "--served-model-name nvidia/Qwen3.5-397B-A17B-NVFP4-V2",
    "--trust-remote-code",
    "--reasoning-parser qwen3",
    "--tool-call-parser qwen3_coder",
    "--quantization modelopt_mixed",
    "--fp4-gemm-backend flashinfer_cutlass",
    "--kv-cache-dtype fp8_e4m3",
    "--mamba-scheduler-strategy no_buffer",
    "--mamba-ssm-dtype bfloat16",
    "--attention-backend trtllm_mha",
    "--mm-attention-backend triton_attn",
    "--linear-attn-decode-backend flashinfer",
    "--disaggregation-transfer-backend mooncake",
    "--disable-radix-cache",
    "--context-length 9236",
    "--page-size 64"
  ],
  "commonEnv": [
    "NO_COLOR=1",
    "PYTHONUNBUFFERED=1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600",
    "TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=3600",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600",
    "NCCL_MNNVL_ENABLE=1",
    "NCCL_NVLS_ENABLE=0",
    "NCCL_CUMEM_ENABLE=1",
    "MC_FORCE_MNNVL=1",
    "SGLANG_ENABLE_SPEC_V2=1",
    "SGLANG_DG_CACHE_DIR=/configs/deepgemm-cache",
    "FLASHINFER_WORKSPACE_BASE=/configs/flashinfer-cache",
    "SGLANG_ENABLE_JIT_DEEPGEMM=true",
    "SGLANG_ENABLE_FLASHINFER_GEMM=true",
    "FLASHINFER_DISABLE_VERSION_CHECK=1",
    "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000",
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
    "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0",
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1",
    "SGLANG_HEALTH_CHECK_TIMEOUT=3600",
    "SGLANG_HEALTH_STARTING_OK=1",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0"
  ],
  "profiles": [
    {
      "id": "1p1d-tp4-tp4-c1",
      "prefillWorkers": 1,
      "decodeWorkers": 1,
      "hosts": 2,
      "gpusPerHost": 4,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 128000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 128",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 1500000",
        "--max-mamba-cache-size 256",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 256",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 1",
        "--disaggregation-bootstrap-port 31000"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "1p1d-tp4-tp4-c4",
      "prefillWorkers": 1,
      "decodeWorkers": 1,
      "hosts": 2,
      "gpusPerHost": 4,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 128000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 1",
        "--stream-interval 30",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 128",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 1500000",
        "--max-mamba-cache-size 256",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 256",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 1",
        "--disaggregation-bootstrap-port 31000"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "1p1d-tp4-tp4-c8",
      "prefillWorkers": 1,
      "decodeWorkers": 1,
      "hosts": 2,
      "gpusPerHost": 4,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 128000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 128",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 1500000",
        "--max-mamba-cache-size 256",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 256",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 50",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 1",
        "--disaggregation-bootstrap-port 31000"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "3p5d-tp4-tp4-c128",
      "prefillWorkers": 3,
      "decodeWorkers": 5,
      "hosts": 8,
      "gpusPerHost": 4,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 128000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 128",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 1500000",
        "--max-mamba-cache-size 256",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 256",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 1",
        "--disaggregation-bootstrap-port 31000"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "2p2d-tp2-tp2-c512",
      "prefillWorkers": 2,
      "decodeWorkers": 2,
      "hosts": 4,
      "gpusPerHost": 2,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 2",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 64000",
        "--max-running-requests 64",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 2",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 128",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 750000",
        "--max-mamba-cache-size 128",
        "--max-running-requests 84",
        "--cuda-graph-max-bs 88",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 1"
      ],
      "prefillEnv": [
        "ETCD_LEASE_TTL=120"
      ],
      "decodeEnv": [
        "ETCD_LEASE_TTL=120",
        "MC_TE_METRIC=true",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "2p3d-tp2-tp2-c768",
      "prefillWorkers": 2,
      "decodeWorkers": 3,
      "hosts": 5,
      "gpusPerHost": 2,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 2",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 64000",
        "--max-running-requests 64",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 2",
        "--data-parallel-size 1",
        "--expert-parallel-size 1",
        "--mamba-track-interval 128",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 750000",
        "--max-mamba-cache-size 128",
        "--max-running-requests 86",
        "--cuda-graph-max-bs 88",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 10",
        "--stream-interval 30",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 1"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "7p1d-dep4-dep16-c5120",
      "prefillWorkers": 7,
      "decodeWorkers": 1,
      "hosts": 11,
      "gpusPerHost": 4,
      "decodeNodes": 4,
      "frontends": 3,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 4",
        "--expert-parallel-size 4",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 128000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 65536",
        "--max-prefill-tokens 65536",
        "--scheduler-recv-interval 1",
        "--stream-interval 50",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 16",
        "--data-parallel-size 16",
        "--expert-parallel-size 16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mamba-track-interval 128",
        "--moe-a2a-backend flashinfer",
        "--moe-runner-backend flashinfer_cutedsl",
        "--speculative-moe-a2a-backend none",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--disable-shared-experts-fusion",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--cuda-graph-max-bs 256",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 2200000",
        "--max-mamba-cache-size 2048",
        "--max-running-requests 2048",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 1",
        "--stream-interval 50",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 50"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_MOE_NVFP4_DISPATCH=1",
        "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "7p1d-dep4-dep16-c8192",
      "prefillWorkers": 7,
      "decodeWorkers": 1,
      "hosts": 11,
      "gpusPerHost": 4,
      "decodeNodes": 4,
      "frontends": 3,
      "requestPlane": "tcp",
      "dynamoVersion": "1.3.0.dev20260708",
      "installDynamo": false,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 4",
        "--expert-parallel-size 4",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 128000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 65536",
        "--max-prefill-tokens 65536",
        "--scheduler-recv-interval 1",
        "--stream-interval 50",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 16",
        "--data-parallel-size 16",
        "--expert-parallel-size 16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mamba-track-interval 128",
        "--moe-a2a-backend flashinfer",
        "--moe-runner-backend flashinfer_cutedsl",
        "--speculative-moe-a2a-backend none",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--disable-shared-experts-fusion",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--cuda-graph-max-bs 256",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 2200000",
        "--max-mamba-cache-size 3413",
        "--max-running-requests 3413",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 1",
        "--stream-interval 50",
        "--watchdog-timeout 1000000",
        "--decode-log-interval 50"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_MOE_NVFP4_DISPATCH=1",
        "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    },
    {
      "id": "7p4d-dep4-dep4-c12288",
      "prefillWorkers": 7,
      "decodeWorkers": 4,
      "hosts": 11,
      "gpusPerHost": 4,
      "decodeNodes": 1,
      "frontends": 1,
      "requestPlane": "nats",
      "dynamoVersion": "1.2.1",
      "installDynamo": true,
      "prefillFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 4",
        "--expert-parallel-size 4",
        "--enable-dp-attention",
        "--enable-dp-attention-local-control-broadcast",
        "--enable-dp-lm-head",
        "--mamba-track-interval 2048",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--max-total-tokens 256000",
        "--max-running-requests 128",
        "--cuda-graph-max-bs 4",
        "--chunked-prefill-size 262144",
        "--max-prefill-tokens 131072",
        "--scheduler-recv-interval 1",
        "--stream-interval 50",
        "--load-balance-method round_robin",
        "--watchdog-timeout 1000000",
        "--log-level info"
      ],
      "decodeFlags": [
        "--tensor-parallel-size 4",
        "--data-parallel-size 4",
        "--expert-parallel-size 4",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mamba-track-interval 128",
        "--moe-a2a-backend none",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-moe-a2a-backend none",
        "--speculative-moe-runner-backend flashinfer_trtllm",
        "--disable-shared-experts-fusion",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--disable-flashinfer-autotune",
        "--cuda-graph-max-bs 512",
        "--mem-fraction-static 0.94",
        "--max-total-tokens 2052736",
        "--max-mamba-cache-size 1056",
        "--max-running-requests 1056",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 32768",
        "--scheduler-recv-interval 1",
        "--disaggregation-decode-polling-interval 8",
        "--stream-interval 50",
        "--soft-watchdog-timeout 240",
        "--watchdog-timeout 300",
        "--decode-log-interval 50"
      ],
      "prefillEnv": [],
      "decodeEnv": [
        "MC_TE_METRIC=true",
        "SGLANG_MOE_NVFP4_DISPATCH=1",
        "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024",
        "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1",
        "SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000"
      ]
    }
  ]
};

export const Qwen35DynamoCommands = ({ recipes, isDark }) => {
  const [profileId, setProfileId] = useState(recipes.profiles[0].id);
  const [role, setRole] = useState('prefill');
  const [rank, setRank] = useState('0');
  const [discoveryHost, setDiscoveryHost] = useState('192.0.2.10');
  const [decodeLeader, setDecodeLeader] = useState('192.0.2.20');
  const [copyState, setCopyState] = useState('');
  const profile = recipes.profiles.find(item => item.id === profileId);
  const isMultiNodeDecode = role === 'decode' && profile.decodeNodes > 1;
  const addressesReady = discoveryHost.trim() && (!isMultiNodeDecode || decodeLeader.trim());
  const shellQuote = value => "'" + value.replace(/'/g, "'\\''") + "'";

  const generateCommand = () => {
    if (!addressesReady) return '# Enter the discovery host and, for TP16 decode, the decode leader.';
    const lines = [
      `# ${profile.id} — ${role}; use image ${recipes.image}`,
      `# Dynamo package pair: ${profile.dynamoVersion}`,
    ];
    if (profile.installDynamo) {
      lines.push(
        'python3 -m pip install --break-system-packages --no-deps \\',
        '  --extra-index-url https://pypi.nvidia.com \\',
        '  ai-dynamo==1.2.1 ai-dynamo-runtime==1.2.1',
      );
    }
    lines.push(
      `export ETCD_ENDPOINTS=${shellQuote(`http://${discoveryHost.trim()}:2379`)}`,
      `export NATS_SERVER=${shellQuote(`nats://${discoveryHost.trim()}:4222`)}`,
      `export DYN_REQUEST_PLANE=${profile.requestPlane}`,
      'export DYN_SKIP_SGLANG_LOG_FORMATTING=1',
      'export DYN_LOG=info,dynamo_runtime::pipeline::network::ingress::push_handler=warn',
    );
    if (role === 'frontend') {
      lines.push(
        `# Run on ${profile.frontends} distinct frontend host${profile.frontends > 1 ? 's; configure nginx as described below' : ''}.`,
        'python3 -m dynamo.frontend --http-port=30000 --enforce-disagg',
      );
      return lines.join('\n');
    }
    const isPrefill = role === 'prefill';
    const roleEnv = isPrefill ? profile.prefillEnv : profile.decodeEnv;
    lines.push(
      '# Run inside the staged worker container after configuring fabric interfaces below.',
      `export CUDA_VISIBLE_DEVICES=${profile.gpusPerHost === 2 ? '0,1' : '0,1,2,3'}`,
      ...recipes.commonEnv.map(value => `export ${value}`),
      ...roleEnv.map(value => `export ${value}`),
      `export DYN_SYSTEM_PORT=${isPrefill ? 7500 : 7501}`,
    );
    const flags = [
      ...recipes.commonFlags,
      ...(isPrefill ? profile.prefillFlags : profile.decodeFlags),
      '--host 0.0.0.0',
      '--port 6100',
      `--nccl-port ${isPrefill ? 17500 : 17501}`,
      `--request-plane ${profile.requestPlane}`,
      `--disaggregation-mode ${role}`,
    ];
    if (isPrefill) flags.push('--disaggregation-bootstrap-port 31000');
    if (isMultiNodeDecode) {
      flags.push(
        `--nnodes ${profile.decodeNodes}`,
        `--node-rank ${rank}`,
        `--dist-init-addr ${shellQuote(`${decodeLeader.trim()}:29500`)}`,
      );
    }
    lines.push('python3 -m dynamo.sglang \\\n  ' + flags.join(' \\\n  '));
    return lines.join('\n');
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
  const mtpSteps = profile.decodeFlags.find(flag => flag.startsWith('--speculative-num-steps ')).split(' ')[1];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <p style={{ margin: '6px 0', fontSize: '13px' }}>
        Use the pinned serving container after completing the{' '}
        <a href="#3-3-gb200-nvfp4-v2-with-dynamo-prefill-decode-disaggregation" style={{ color: '#D45D44', textDecoration: 'underline' }}>checkpoint, service, and fabric setup below</a>.
      </p>
      <div style={card}>
        <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '12px' }}>
          <strong style={{ minWidth: '140px' }}>Serving variant</strong>
          <select aria-label="Serving variant" value={profileId} onChange={event => setProfileId(event.target.value)} style={{ ...control, flex: 1 }}>
            {recipes.profiles.map(item => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
        </label>
        <p style={{ margin: '8px 0 0', lineHeight: '1.5' }}>
          {profile.prefillWorkers} prefill + {profile.decodeWorkers} decode workers across {profile.hosts} physical hosts;
          {' '}{profile.gpusPerHost} GPUs per host exposed to each worker.
          {' '}Decode MTP: {mtpSteps} steps. Dynamo {profile.dynamoVersion}; {profile.requestPlane.toUpperCase()} requests.
          {' '}{profile.frontends} frontend replica{profile.frontends > 1 ? 's behind nginx' : ''}.
        </p>
        <p style={{ margin: '4px 0 0', opacity: 0.8 }}>
          The c suffix identifies the source workload point; serving limits come from the selected recipe.
        </p>
      </div>
      <div style={{ ...card, display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
        <strong style={{ minWidth: '140px' }}>Command role</strong>
        {[
          ['prefill', 'Prefill worker'], ['decode', 'Decode worker'], ['frontend', 'Frontend'],
        ].map(([id, label]) => (
          <button key={id} type="button" aria-pressed={role === id} onClick={() => setRole(id)}
            style={{ ...control, cursor: 'pointer', ...(role === id ? { background: '#D45D44', color: 'white', borderColor: '#D45D44' } : {}) }}>
            {label}
          </button>
        ))}
      </div>
      <div style={{ ...card, display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
        <label>Discovery host{' '}
          <input aria-label="Discovery host" value={discoveryHost} onChange={event => setDiscoveryHost(event.target.value)} style={control} />
        </label>
        {isMultiNodeDecode && <>
          <label>Decode leader{' '}
            <input aria-label="Decode leader" value={decodeLeader} onChange={event => setDecodeLeader(event.target.value)} style={control} />
          </label>
          <label>Decode rank{' '}
            <select aria-label="Decode rank" value={rank} onChange={event => setRank(event.target.value)} style={control}>
              {['0', '1', '2', '3'].map(value => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
        </>}
        <span style={{ width: '100%', opacity: 0.8 }}>Replace the example IPs with addresses reachable from every serving container.</span>
      </div>
      <div style={card}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
          <strong>{role === 'frontend' ? 'Frontend' : role === 'prefill' ? 'Prefill worker' : 'Decode worker'} command</strong>
          <button type="button" onClick={copyCommand} disabled={!addressesReady} style={{ ...control, marginLeft: 'auto', cursor: 'pointer' }}>Copy command</button>
          <span role="status" aria-live="polite">{copyState}</span>
        </div>
        <pre aria-label="Generated Dynamo command" style={{
          padding: '12px 16px', margin: 0, background: isDark ? '#111827' : '#f5f5f5',
          borderRadius: '6px', fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
          fontSize: '12px', lineHeight: '1.5', whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: '520px',
        }}>{command}</pre>
      </div>
    </div>
  );
};
