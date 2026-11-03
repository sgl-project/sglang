####################################################################################################
# Intel XPU (Level Zero) dev: elastic hybrid-SWA pools on one card.
#
# Differences from dev.sh (CUDA):
#   - kvcached is built for XPU from a local checkout (KVCACHED_BACKEND=xpu).
#   - --page-size 1: elastic pools need it. intel_xpu serves page size 1 through the
#     gather fallback in sgl-kernel-xpu#454 (slower than its paged 64/128 kernel);
#     without that build, SGLang forces page size 128 and startup is refused.
#   - kvcached's own SGLang autopatch must stay off; it would replace the same pools.
#   - SGLANG_ELASTIC_MEM_CU_PAGE_SIZE is also kvcached's page size here, so it must
#     be a multiple of 2 MiB (the default is 2 MiB).

# kvcached with the XPU backend
KVCACHED_SRC=${KVCACHED_SRC:-$HOME/kvcached}
(cd "${KVCACHED_SRC}" && KVCACHED_BACKEND=xpu pip install -e . --no-build-isolation)
python3 -c "import kvcached.vmm_ops as v; assert v.backend_name() == 'XPU', v.backend_name()"
python3 -c "from sgl_kernel import flash_attn as f; assert '_flash_attn_with_kvcache_page_size_1' in vars(f), 'sgl-kernel-xpu lacks #454'"

##############################
# gpt-oss-20b (MXFP4 runs natively on XPU, ~13 GB)
##############################
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export SGLANG_ENABLE_ELASTIC_MEM_POOL=1
export ENABLE_KVCACHED=false
export KVCACHED_AUTOPATCH=false
export PORT=${PORT:-30000}

python3 -m sglang.launch_server \
  --log-level debug \
  --model openai/gpt-oss-20b \
  --device xpu \
  --attention-backend intel_xpu \
  --page-size 1 \
  --swa-full-tokens-ratio 1.0 \
  --context-length 65536 \
  --port ${PORT} \
  > nohup.emem.xpu.port.${PORT}.out 2>&1 &
# Comparison runs: --attention-backend triton --page-size 1 (elastic), and
# intel_xpu --page-size 128 with SGLANG_ENABLE_ELASTIC_MEM_POOL=0 (paged kernel).

python3 -m sglang.bench_serving \
  --backend sglang \
  --port ${PORT} \
  --dataset-name random \
  --num-prompts 256 \
  --random-input 4096 \
  --random-output 1024 \
  --random-range-ratio 0.5 \
  --max-concurrency 64
