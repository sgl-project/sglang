---
title: "Paged Experts"
metatags:
    description: "SGLang Paged Experts: serve an MoE model whose experts do not fit in GPU memory by keeping K of E experts per layer on the GPU and paging the rest from pinned host memory. Computation stays on the GPU."
---

Paged Experts serves a Mixture-of-Experts (MoE) model whose expert weights do not fit in GPU
memory. Each MoE layer keeps **K of its E routed experts in a K-slot table on the GPU** and holds
all E experts in pinned host memory; every step, the experts the step routes to that are not
resident are copied into slots before the expert GEMM runs.

## Why Paged Experts?

Expert weights are most of an MoE model's parameters, yet each token is routed to only `top_k`
of the `E` experts per layer, and consecutive tokens reuse many of the same experts. Paged
Experts keeps a working set of K experts per layer resident and pages the rest in from host
memory as routing requires them, so expert capacity is bounded by host RAM rather than VRAM.

**Computation stays on the GPU.** Only expert *storage* is host-backed: experts are copied to the
GPU and computed there by the model's own fused-MoE kernel. This differs from `--cpu-offload-gb`
and CPU expert offload, which move the computation to the CPU.

**Results are unchanged.** Routing is untouched (E-wide), and the paged layer produces the same
output as the unpaged one: bit-identical for unquantized, FP8 and INT8 experts (with
`--enable-deterministic-inference`), and within the Marlin kernel's own run-to-run noise for
the other quantized formats. See [Correctness](#correctness).

## Usage

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --enable-paged-experts
```

At startup the KV cache keeps room for one request of the model's context, and the rest of the
static memory holds as many experts per layer (K) as fit; the log names the K it chose and what
would change it. See [Choosing K](#choosing-k) to trade context or concurrency for resident
experts, or to set K yourself.

The default CUDA graphs stay on: decode graphs are captured up to `K // top_k` concurrent
requests, and breakable prefill graphs run the paged MoE layers as eager breaks.

## Related Parameters

- **`--enable-paged-experts`**: Enable Paged Experts. Off by default.
- **`--paged-experts-num-resident K|auto`**: Routed experts per MoE layer kept on the GPU, at least
  the router's `top_k`; larger values page less but leave less memory for the KV cache. `auto` (the
  default) sizes K after the model is loaded; see [Choosing K](#choosing-k).
- **`--paged-experts-enable-disk`**: Let Paged Experts write to disk: keep each MoE layer's
  experts in the [expert cache](#the-expert-cache), so that later starts read them instead of
  loading and repacking them, and serve the experts that do not fit host memory from there
  (see [Disk tier](#disk-tier)). Off by default; requires `--enable-paged-experts`.
- **`--paged-experts-host-memory-gb G`**: Pinned host memory, in GiB, for the experts of all MoE
  layers. When every expert fits, nothing changes; otherwise the [disk tier](#disk-tier) holds the
  rest, with `--paged-experts-enable-disk`, and without it the start fails. Unset, every expert is
  pinned as before; with `--paged-experts-enable-disk`, the experts that do not fit what is
  available at startup (within the container's memory limit) go to the disk tier, which then
  keeps 10 GiB of that free. Requires `--enable-paged-experts`.

## Supported Configurations and Limitations

- **Quantization:**

  | Experts | MoE runner | Notes |
  |---|---|---|
  | unquantized (bf16/fp16) | triton | |
  | block-quantized FP8 (e.g. `weight_block_size` `[128, 128]`) | triton | |
  | compressed-tensors FP8, per-channel or block, dynamic activations (e.g. `FP8-dynamic` checkpoints) | triton | per-tensor FP8 is not supported |
  | W8A8 FP8 (`--quantization w8a8_fp8`), W8A8 INT8 (`--quantization w8a8_int8`) | triton | |
  | block-quantized INT8 | triton | |
  | GPTQ int4/int8, AWQ int4 | Marlin | |
  | compressed-tensors W4A16/W8A16 | Marlin | |
  | compressed-tensors NVFP4 | Marlin | weight-only: nvfp4a16, or w4a4 before Blackwell |
  | MXFP4 (e.g. gpt-oss) | Marlin | pass `--moe-runner-backend marlin` |
  | MXFP4 routed experts of an FP8 checkpoint (e.g. DeepSeek-V4) | Marlin | pass `--moe-runner-backend marlin` |
  | ModelOpt NVFP4 | Marlin | pass `--moe-runner-backend marlin`; runs W4A16 |

  Other quantization methods and MoE runners are rejected at startup.
- **Single GPU:** tensor, expert, pipeline and data parallelism, EPLB, fused shared experts
  (use `--disable-shared-experts-fusion`) and `--enable-fused-moe-sum-all-reduce` are rejected
  at startup.
- **Speculative decoding** is rejected at startup.
- **Load formats:** `--load-format auto` (the default), `safetensors`, `pt`, `npcache`,
  `mistral`, `fastsafetensors`, `instanttensor` and `runai_streamer`. The others are rejected
  at startup: `dummy` has no real weights, `gguf` holds GGUF-quantized experts, and the rest
  load the weights without Paged Experts' loader and post-load step.
- **CUDA graphs:** decode graphs may use the `full` backend (the default) or be disabled;
  prefill graphs may use the `breakable` backend (the default on CUDA) or be disabled. Other
  combinations are rejected at startup.
- **Host memory:** the host store holds all E experts of every MoE layer and is page-locked, so
  the host must be able to pin that much memory, or the disk tier holds the rest. Under WSL2,
  pinned memory is capped at about half of system RAM: set `--paged-experts-host-memory-gb`
  below that.
- **Disk tier** ([below](#disk-tier); R is the experts per layer in host memory): x86-64 hosts
  only for now. The first start writes every layer's file once, about the size of the expert
  weights, and keeps 8 GiB of the disk free; the space is checked at startup. The checkpoint must
  load through the safetensors loaders with mmap (see Startup; not a checkpoint that stacks all
  experts in one tensor), or the start fails. With decode CUDA graphs on and R < E, a step waits
  at a layer for the experts it reads from disk, and the scheduler thread waits for each decode
  step (no overlap scheduling); a step that gives up waiting (after 2 s) is replayed. On hybrid
  (mamba / linear-attention) models, the step's recurrent state is copied before it runs and
  restored before a replay. Refused then, as they accumulate state a replay would count twice:
  `--enable-linear-replayssm`, `--enable-linear-replayssm-spec`, `--enable-unified-memory`,
  `--expert-distribution-recorder-mode` and `SGLANG_OPT_USE_ONLINE_COMPRESS`. With `auto`, K
  grows at most to R.
- **Startup:** with mapped safetensors (`--load-format auto`, `safetensors` or `mistral`, without
  `--weight-loader-disable-mmap`), loading makes no second copy of the experts: each expert is
  read from the checkpoint once, straight into the pinned host store. With other loaders, and
  for checkpoints that store every expert in one tensor (e.g. gpt-oss), the experts are first
  loaded into a pageable copy of every expert, as large as the host store. With
  `--paged-experts-enable-disk`, later starts read the experts from the
  [expert cache](#the-expert-cache) instead.

## How It Works

### The K-slot table and the host store

The paged method wraps the layer's own fused-MoE method. It creates the layer's expert tensors
with K rows instead of E, so the GPU holds only the K slots. Each tensor's weight loader is
redirected: a load of one expert's weights does not run yet, but waits as a view of the mapped
checkpoint file. After loading, the waiting loads run K experts at a time, each through
SGLang's native per-expert loader (sharding, gate/up placement, scale layout) into K rows laid
out as the GPU slots, and the wrapped method's post-load step runs on those rows as it would
without paging (on the Marlin runner, the repack into Marlin's layout). Each chunk's result is
copied into the host store, a page-locked table of all E experts of each per-expert tensor, and
the chunk's checkpoint pages are given back to the kernel, so the checkpoint never occupies
memory beside the host store. The slots then hold experts 0..K-1.

A repack must treat each expert on its own to run K at a time: the step runs once more on
experts 1..K in a chunk of their own and must give the same bytes. A step that does not repack
must leave the loaded weights as they are, since paging copies host rows verbatim. Either is
checked at startup. Loads that cannot wait (a checkpoint tensor holding every expert, or a
loader whose tensors are not views of mapped files) go to a table of every expert instead,
which the chunks are taken from.

### A step: waves, page-in, merge

Every forward step of every MoE layer:

1. The distinct experts the step routes to are split into **waves** of at most K, the experts
   already resident first. Decode at
   moderate batch sizes needs one wave; a prefill chunk usually needs several.
2. Each wave makes its experts resident, evicting the least recently used experts outside the
   wave, and copies the missing ones from the host store into their slots.
3. The routed expert ids are remapped to slots and the wrapped fused-MoE kernel runs over the
   wave's own entries **without its final top-k sum**, returning each routed (token, expert)
   output already multiplied by its router weight.
4. Each entry is written into one merged buffer by the one wave that holds its expert, and the
   merged entries go through the same top-k sum the kernel would have performed, so the result
   does not depend on how the experts were split into waves.

### Captured decode

With decode CUDA graphs on, a step whose routed entries fit the K slots (`tokens × top_k ≤ K`,
i.e. decode up to `K // top_k` requests) is served by a single wave decided and paged on the
GPU, with no host synchronization, so the decode graph captures it:

- a JIT kernel makes the LRU decision on the device and writes the page-in plan (experts, slots,
  count) to GPU buffers;
- a second kernel copies the planned experts of every paged tensor from the pinned host store
  into their slots, reading the host memory directly through its unified address and the count
  on the device, so each graph replay moves exactly the experts it decided on.

Decode capture batch sizes are capped at `K // top_k` (logged at startup); larger batches run
without a graph. Prefill steps plan their waves on the host: under a breakable prefill graph the
paged MoE layer is an eager break between graph segments, so the rest of the layer stack stays
captured. Both paths share one LRU state, so residency carries over between captured and
uncaptured steps.

### The expert cache

Loading the experts takes most of a start for large MoE models, and formats on the Marlin
runner also repack them. With `--paged-experts-enable-disk`, every MoE layer with more experts
than K (under `--paged-experts-num-resident auto`, more than the router's top-k: K is sized
after loading) keeps its experts on disk after loading, in the layout of the host store, one
file per layer under `$SGLANG_CACHE_DIR/paged_experts/<model>-<key>/` (by default
`~/.cache/sglang`; `<model>` is the Hugging Face id, or the folder name and a hash of its path).
The files take about as much disk as the experts: on an RTX 5070 Ti with 62 GB of RAM and a
PCIe 5.0 NVMe, Qwen3-30B-A3B-GPTQ-Int4 keeps 13.9 GiB (`load_model` 33 s, 22 s from the cache),
Qwen3-30B-A3B-FP8 27.0 GiB (51 s, 29 s) and Qwen3-Next-80B-A3B-Instruct-NVFP4 40.7 GiB. After
loading, one log line names the model's directory and what the cache did: layers read, rebuilt
or written. Without the flag, a start whose experts the cache could keep logs its size and
location once.

- **A later start reads the files instead of loading and repacking.** It reads a layer's file
  straight into the host store, then still loads a few groups of experts from the checkpoint
  with the current code, each as wide as the router's top-k whatever K: the first experts, the
  last ones, the first ones shifted by one, and a group sampled anew at every start, so that
  over restarts every expert is checked. On the Marlin runner it repacks them. It compares
  their bytes with what it read; a file that does not give the same bytes for all of them, does
  not fit its layer or cannot be read is rebuilt from every expert and replaced, and a warning
  names the first group that differed. With mapped safetensors (see Startup), the other
  experts are not loaded at all unless the file is rebuilt; with other loaders, and for
  checkpoints that store every expert in one tensor, every expert is still loaded and the
  cache saves the repack only.
- **The key** covers the checkpoint files the loader read (path, size and modification time of
  the weight files and of the folder's `*.json`) and the torch, triton and kernel package
  versions (the model's directory), and per layer its format and MoE runner, its tensors and
  the content of the source files that repack the experts (the layer's file). A change to how
  the experts are loaded reaches every expert, so the checked groups catch it without a new
  key. The cache needs the default loader (`--load-format auto`, `safetensors`, `pt`, `npcache`,
  `mistral`, `fastsafetensors`, `instanttensor`), which reports the files it read.
- **One set per model.** Writing a file under a new key first removes the model's directories
  under other keys and the layer's file under another layer key, also to make the space for
  it. So the cache does not grow over time: it holds one set of files per model it has
  served. Starts that alternate `--quantization` or `--moe-runner-backend` on the same
  checkpoint replace each other's layer files at every start. Check the space with
  `du -sh $SGLANG_CACHE_DIR/paged_experts`; to clear a model, delete its `<model>-*`
  directory (the next start with the flag rebuilds it). A model loaded from a local folder gets
  a directory named by a hash; the startup log names it. An SGLang, torch or kernel package
  upgrade changes the key: the next start rebuilds every file once.
- **Writes** run in the background with direct I/O and are published complete or not at all
  (a staging file, then a rename), so concurrent starts can share the directory; a start that
  is killed mid-write leaves a staging file that a later start removes. A write is skipped
  when it would leave less than 8 GiB free: on a nearly full disk, the layers written until
  then stay, and the warning says how much the next one needed. Reading or writing the cache
  never fails a start; it only costs the time it would have saved. On the
  [disk tier](#disk-tier) the files are required instead: a first build that cannot write them
  fails the start.
- **Who can write the directory controls the served weights**: a start serves the cached
  experts outside the checked groups as they are on disk. Keep `$SGLANG_CACHE_DIR` private to
  the users who run the server, as for the other caches there.
- **Not detected:** a change that alters only some experts (for example a branch that depends
  on the values of their scales) outside the files in the key, until a sampled group covers
  one of those experts; and a checkpoint file or cache file altered in place, outside the
  checked groups, with its size and modification time kept (no checksum, as SGLang's other
  caches). If in doubt, delete the model's directory.

In a container, mount `$SGLANG_CACHE_DIR` (or a directory it points to) to keep the files
across restarts; without a mount they are written into the container's own filesystem, which
takes the disk space and loses them with the container.

### Disk tier

With `--paged-experts-enable-disk`, when the host budget cannot hold every expert of every layer,
each layer keeps all E experts in its [expert cache](#the-expert-cache) file and R of them in
pinned host rows; the K GPU slots are filled from the rows. The tier runs on x86-64 hosts for
now: on others (e.g. Grace), such a model fails at startup. With `--paged-experts-host-memory-gb`
and without `--paged-experts-enable-disk`, such a model fails at startup with the sizes and the
flag. Layers routed by token id (DeepSeek-V4's hash layers) keep every expert in host memory when
they take at most half the budget; the other layers split the rest evenly (over all hidden
layers: a model with dense layers pins less than the budget).

For example, DeepSeek-V4-Flash (137 GiB of experts) on a 16 GB GPU (RTX 5070 Ti) with 62 GB of
RAM, as measured (about 14 tok/s decode; with the default `--mem-fraction-static` the weights
leave no room for the KV cache on this card):

```bash
SGLANG_OPT_FP8_WO_A_GEMM=1 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --enable-paged-experts --paged-experts-enable-disk --paged-experts-host-memory-gb 45 \
  --moe-runner-backend marlin --mem-fraction-static 0.92 --max-running-requests 1 \
  --context-length 4096 --json-model-override-args '{"max_position_embeddings":4096}' \
  --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.5 --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-prefill disabled --disable-radix-cache
```

`SGLANG_OPT_FP8_WO_A_GEMM=1` keeps DeepSeek-V4's `wo_a` in FP8 on SM120 (upstream option), the
4096-token context limits the RoPE table and the KV cache, and one prefill chunk per prompt reads
each expert from disk once.

Set `--paged-experts-host-memory-gb` on a machine where the rest of the server needs more than
10 GiB of host memory: pinned memory cannot be swapped out.

DeepSeek-V4 on a small GPU needs `--max-running-requests`: without it, SGLang reserves the
per-request state of DeepSeek-V4's compressed attention for its default of 256 requests (about
2.5 GB for DeepSeek-V4-Flash) before Paged Experts sizes anything, which can leave no room for
the sliding-window pool and fail the start. `--max-running-requests 1` reserves it for one.

- **The first start writes the files once**, so a model larger than RAM never has to fit in
  RAM: the experts' loads are replayed K at a time from the mapped checkpoint (see Startup)
  straight into the file being built (a write through a mapping of the file, not direct I/O),
  through the repack on the Marlin runner. The build keeps 8 GiB of the disk free: its space is
  checked at startup, and again for each file. A failed build leaves no file behind. Later starts check the same experts as for a model that fits and do not
  read the other experts from the checkpoint; nor do they read them from the file, except the
  ones the slots and host rows start with.
- **The host rows are inclusive:** moving an expert into a slot copies it from its host row,
  which keeps it; the expert the slot held is dropped (the disk has it), so nothing is written
  back. An expert only on disk is first read into the least recently used host row (whose
  expert is dropped), with one read per expert, bypassing the page cache (`O_DIRECT`), and
  several in flight.
- **Captured decode** decides on the GPU as before and copies the experts that are in host rows
  inside the graph. An expert only on disk is flagged: the layer's plan kernel writes the
  layer's state to pinned host memory before the layer copies its host-row hits, a host thread
  reads the flagged experts while those copies run, and the step waits at the layer until they
  are in their slots, then runs on with the experts it routed to. The host thread is the one
  that replays the graph, so it returns once the step ends: the overlap scheduler cannot prepare
  the next batch during a decode step. A step that gives up waiting (after 2 s) runs on, and the
  host copies the experts in after it and replays the step, with the recurrent state restored;
  the log warns about such steps. A decode step only writes its own KV entries and outputs, so
  replaying it gives the same result. ReplaySSM, `--enable-unified-memory`,
  `--expert-distribution-recorder-mode` and DeepSeek-V4's `SGLANG_OPT_USE_ONLINE_COMPRESS`
  accumulate state a replay would count twice: they are refused with decode CUDA graphs.
- **Prefill reads from disk:** a prompt of a few thousand tokens routes to nearly every expert of
  every layer, so its time to first token is that of reading most of the experts (DeepSeek-V4-Flash:
  about 8.5 s for a 1.5k-token prompt on a PCIe 5.0 NVMe).
- **Checkpoints:** the tier needs the experts' loads as views of mapped safetensors files, one
  tensor per expert (`--load-format auto`, `safetensors` or `mistral`): checkpoints that stack all
  experts in one tensor (gpt-oss) are refused.

Expert packs (`--load-format expert_pack`) are a different SSD tier: prebuilt packs for
specific models, read from SSD into a GPU cache. The disk tier needs no preparation; it serves
checkpoints that load as mapped safetensors with one tensor per expert.

## Choosing K

The K-slot table and the KV cache come out of the same static budget (`--mem-fraction-static`).
One resident expert per layer costs `per_expert_bytes × num_moe_layers` of GPU memory; the more
experts stay resident, the fewer are paged in per step and the faster decode is, up to a plateau.

### `auto` (the default)

With `--paged-experts-num-resident auto`, or the flag omitted, K is chosen once every weight is
loaded, in two steps:

1. **The KV cache comes first.** By default it keeps room for one request of the model's own
   context. `--context-length` sets that length, and `--max-running-requests`, if you set it, how
   many such requests it aims to hold at once: one full-length request is a promise, more of them a
   target that the experts never take memory from. `--max-total-tokens` sets the KV cache's size
   directly instead. As always in SGLang, requests beyond what the KV cache holds at a time queue.
2. **The experts take the rest.** What the static budget has left after the KV cache is allocated
   goes to the expert slots, one slot per layer at a time, up to all experts.

So under `auto`, `--context-length`, `--max-running-requests` and `--max-total-tokens` decide
the KV cache, and they are also the knobs that trade context and concurrency for resident experts.
At startup the server logs the chosen K, the KV cache it kept and why, and how many full-length
requests fit at once:

```
Paged experts: K=63 of 128 experts per MoE layer resident (6.85 GiB); KV cache 40960 tokens,
1 request(s) of 40960 at once (aim 40960: the model's context 40960 x 1 request(s)). To pin:
--paged-experts-num-resident 63
```

If a `--max-total-tokens` you set, or one request of a `--context-length` you set, cannot be met
even at K = `top_k`, the server stops at startup and names the flag. If fewer requests of that
length fit than `--max-running-requests` asks for, the log says how many do, and the rest queue.
If one request of the model's own context does not fit, K stays at `top_k`, the KV cache takes what
is left, and requests are capped at its size, as SGLang caps them whenever the KV cache is shorter
than the context; a warning says so.

On hybrid models with Mamba layers and no `--max-mamba-cache-size`, the Mamba state pool holds
the requests the KV cache aims for: the `--max-running-requests` you set, or one; never more
than SGLang's own share of the budget (`--mamba-full-memory-ratio`), and never so many that one
full-length request no longer fits in the KV cache next to it: the rest queue. Each running
request needs its own Mamba state however short it is, so these are the requests that run at a
time: without `--max-running-requests`, one. SGLang's startup warning that requests are capped
by the Mamba state refers to this; set `--max-running-requests` to run more.

Ways to give the experts more memory:

- a shorter `--context-length` or `--max-total-tokens`, if your requests are shorter than the
  model's context, or fewer `--max-running-requests`;
- `--kv-cache-dtype fp8`, which halves the KV cache's cost per token;
- a larger `--mem-fraction-static`, if the GPU has activation memory to spare.

`auto` cannot be combined with `--enable-memory-saver`, `--enable-lora`,
`--enable-nvidia-megakernel` or post-capture KV sizing (`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`); set
an explicit K with those.

### An explicit K

`--paged-experts-num-resident K` is used as given, and the KV cache takes what is left, as without
paging; if the server then runs out of memory for the KV cache at startup, lower K. The K that
`auto` logs can be pinned this way.

For example, Qwen3-30B-A3B-GPTQ-Int4 (2.43 MB per expert, 48 MoE layers, 117 MB per slot) on a
16 GB GPU with `--mem-fraction-static 0.88` (the log above): its own 40,960-token context leaves
K = 63 of 128 experts (208 tok/s), `--context-length 16384` K = 86 (231 tok/s) and
`--max-total-tokens 10240` K = 92 (233 tok/s). Decode speed on this model is flat from K ≈ 88 up
and falls below it.

## Correctness

Prompt and greedy decode logprobs are compared with the unpaged server under
`--enable-deterministic-inference --attention-backend triton`, eager and with decode captured:

- **triton runner (unquantized, FP8, INT8):** bit-identical;
- **Marlin runner (GPTQ, AWQ, compressed-tensors, MXFP4, NVFP4):** Marlin's reduction depends on
  the batch shape, so a paged step (which runs the kernel on a different number of experts)
  differs from the unpaged one within Marlin's own batch-to-batch noise. On SM90 and newer the
  Marlin MoE kernel also reduces with atomic adds and is not reproducible from run to run (two
  identical unpaged NVFP4 servers differ by up to about 1 nat). Accuracy is unchanged: on GSM8K
  (200 questions), gpt-oss-20b MXFP4 scores 0.96–0.97 paged (16 of 32 experts resident) and
  0.975 unpaged.
