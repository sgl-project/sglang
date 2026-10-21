# PPU Answer MVP

This directory holds the registered PPU Answer nightly tests together with the
public, versioned inputs they read. The evaluator lives in
`python/sglang/test/kits/answer_eval_kit.py` and the on-machine driver in
`python/sglang/test/kits/answer_suite_kit.py`.

```
answer_eval/
├── configs/<model family>/<model config>.json   reviewed execution contracts
├── dataset/                                     corpus, quality profile, schema
└── test_ppu_*.py                                registered entry points
```

The test files sit here rather than in `test/registered/ppu/` so that the suites,
the contracts they execute, and the corpus they are judged against are one
reviewable unit. Nothing about discovery changes: `run_suite.py` and the
`check-registered-tests` hook both walk `test/registered/**/*.py`, and a suite is
owned by the `register_ppu_ci` call inside the file, not by its directory.

Eight suites cover six models. A suite is a file, because `register_ppu_ci`
registers per file; a file holds only configs that agree on the checkpoint family
and on the node count, because every test in a suite runs in one process off one
`SGLANG_PPU_ANSWER_TEST_CONFIG` and a node count decides which workflow can run
it at all. Configs that agree on both do share a suite — Qwen3.5's four, GLM-5.2's
three and MiniMax-M2.7's three do — and the caller picks which one by naming its
config. Two models appear twice for the same reason: the 2.4T checkpoint's two
quantisations need 2 nodes and 4, and Kimi-K2.6's W8A8-INT8 needs 2 where its
other format needs 1.

| Suite | Test file | Model | Configs | Devices |
| --- | --- | --- | --- | --- |
| `nightly-answer-1-ppu` | `test_ppu_qwen38_answer.py` | Qwen3.8-27B, BF16 | 2 | 1 |
| `nightly-answer-8-ppu` | `test_ppu_qwen35_answer.py` | Qwen3.5-397B-A17B | 4 | 8 |
| `nightly-answer-8-glm52-ppu` | `test_ppu_glm52_answer.py` | GLM-5.2 | 3 | 8 |
| `nightly-answer-8-kimi26-ppu` | `test_ppu_kimi_k26_answer.py` | Kimi-K2.6 | 1 | 8 |
| `nightly-answer-8-minimax27-ppu` | `test_ppu_minimax_m27_answer.py` | MiniMax-M2.7 | 3 | 8 |
| `nightly-answer-16-ppu` | `test_ppu_qwen38_a95b_mxfp4_answer.py` | Qwen3.8-2.4T-A95B-MXFP4-FP8 | 1 | 16, over 2 nodes |
| `nightly-answer-16-kimi26-ppu` | `test_ppu_kimi_k26_w8a8_answer.py` | Kimi-K2.6-W8A8-INT8 | 1 | 16, over 2 nodes |
| `nightly-answer-32-ppu` | `test_ppu_qwen38_a95b_answer.py` | Qwen3.8-2.4T-A95B-FP8 | 1 | 32, over 4 nodes |

The device count stays in the suite name even where the name also carries a model
family: it is the scheduling fact a reader needs first, and the four 8-device
suites differ from each other only in which model they hold. Kimi-K2.6 is the one
model whose name appears on two different device counts, which is why its
two-node suite keeps the model in the name as well.

Every suite is now executed on the ZW-M890P (144GiB) K8s boards through the
`-144g` reviewed configs. The ZW810E (96GiB) line has been retired from active
execution: no workflow schedules or dispatches it any more, and its no-suffix
configs stay in the tree only as the measured baseline the 144GiB numbers are
read against — see [Measured baseline (ZW810E)](#measured-baseline-zw810e).

| Line | Suites | Devices | Workflow |
| --- | --- | --- | --- |
| single board | the five 8-device suites, `nightly-answer-1-ppu` included | 1–8 | `test-ppu-answer.yml` |
| two nodes | `nightly-answer-16-ppu`, `nightly-answer-16-kimi26-ppu` | 16 | `test-ppu-answer-16.yml` |
| four nodes | `nightly-answer-32-ppu` | 32 | `test-ppu-answer-32.yml` |

The 8-device suites, and the two two-node ones, have 144GiB configs only: their
checkpoints are 116.2 GiB to 1272.1 GiB, and a ZW810E comparison is not something
this cluster can schedule at those sizes. See
[The ported 144GiB entries](#the-ported-144gib-entries) for where each config
came from and what it costs.

`nightly-answer-32-ppu` has one config and four nodes. Its checkpoint is 2324.7
GiB over 213 shards, which no 96GiB node count this cluster can gang-schedule
would hold, so there is no ZW810E sibling to compare against; it has a workflow
of its own, `test-ppu-answer-32.yml` — see
[The four-node line](#the-four-node-line). `nightly-answer-16-ppu` is the same
checkpoint quantised to MXFP4-FP8, 1272.1 GiB, which two nodes hold, and
`nightly-answer-16-kimi26-ppu` is Kimi-K2.6-W8A8-INT8 at 968.3 GiB, which one
node does not. Both entries are in `test-ppu-answer-16.yml` and have measured
10/10 v0.5.18 runs.

Each of the three lines keeps its own file, concurrency group, artifacts, and
dispatch entry. `.github/workflows/nightly-test-ppu-answer.yml` is the
orchestrator over all three: it owns no runner and runs no test itself, but fans
out to the three reusable workflows through `workflow_call`, pinning each to one
commit through `ref` so all three judge the same revision and handing down the
PPU credentials with `secrets: inherit`. It gives the Answer line a single
scheduled and dispatchable entry point, kept separate from the general
`nightly-test-ppu.yml` so Answer failures, timeouts, scheduling, and artifacts
stay isolated; its cron is offset to 05:00 Beijing. The three callees carry no
`needs` between them — the boards they claim are disjoint (a single board, two
nodes, four nodes) so they run in parallel, and a failure in one does not
suppress another's evidence. Periodic execution requires this file (or a caller)
to sit on the repository's default branch, because GitHub evaluates `schedule`
only there; until then the line is exercised by dispatch, on the orchestrator or
on any single callee. See [The ZW-M890P line](#the-zw-m890p-line) for the
single-board file's internals.

The current phase enforces only request integrity and deterministic facts and
quality checks. LLM-as-Judge is intentionally deferred. Open-ended cases are
therefore marked `hard_constraints_only` in the result rather than being
presented as fully semantic evaluations.

## Runner configuration

`configs/<model family>/<model config>.json` are the reviewed sources of truth
for hardware selection, checkpoint identity and path, SGLang server parameters,
request generation parameters, timeouts, dataset, and quality profile. The file
name spells the identity its `test_id` declares, with the ZW810E baseline
carrying no board suffix: `configs/qwen3.8/27b-bf16.json` is
`qwen3.8-27b-bf16-answer-96g` and its ZW-M890P sibling
`configs/qwen3.8/27b-bf16-144g.json` is `qwen3.8-27b-bf16-answer-144g`. The
suffix is the per-device capacity rather than the board name, which is how the
internal plans separate these same two boards (`answer_96g`, `answer_144g`) and
what `answer_expected_hardware` renders into provenance (`zw810e-8x96g`,
`zw-m890p-8x144g`, and `zw-m890p-4nx8x144g` for the four-node entry). The
workflow reads the same file the test will read to select
the visible devices and to warm the checkpoint page cache, then exports its path
as `SGLANG_PPU_ANSWER_TEST_CONFIG`. This keeps the Python tests generic and makes
the exact execution contract visible in the workflow log.

Only the `hardware` section and the checkpoint path differ between a board pair.
Every server and generation parameter is held identical on purpose, so that a
difference between two verdicts is attributable to the board alone.

The two models share `dataset/answer_cases_zh_v1.json` and
`dataset/quality_profile.json`: the ten prompts are public general knowledge with
reviewed reference facts, so they identify a judgement standard rather than a
model, and a shared file keeps the two suites comparable by construction.

`evaluation.dataset` and `evaluation.quality_profile` are resolved against this
directory, which each test file declares as `data_root`, rather than against the
directory holding the config. Resolving from the config would put the shared
corpus above it, reachable only through `..`, which `validate_test_config`
rejects so that no config can name inputs outside the reviewed tree. `data_root`
is a class attribute with no environment override, unlike the config path: the
corpus a verdict was produced against stays pinned to the checkout.

CI runs each suite the same way every other registered test runs:

```bash
cd test
python3 run_suite.py --hw ppu --suite nightly-answer-8-ppu --nightly \
  --timeout-per-file 14400
```

`run_suite.py` invokes each file as `python3 <file> -f`, so the configuration
arrives through the environment variable. A local run can select the contract
the same way:

```bash
SGLANG_PPU_ANSWER_TEST_CONFIG=test/registered/ppu/answer_eval/configs/qwen3.5/397b-a17b-w8a8-int8.json \
  python3 test/registered/ppu/answer_eval/test_ppu_qwen35_answer.py
```

Each test file also names its own config as the default, so the variable is only
needed to depart from it. A config handed to the wrong file still fails loudly
rather than silently testing something else: validation ties the parallel
degrees to the declared devices and the preflight ties the checkpoint to the
devices actually visible.

For local pytest users, `test/registered/ppu/conftest.py` exposes the same
selection as `--answer-test-config <path>`; that option is not part of the CI
path.

The tests validate the configuration, checkpoint directory, and `config.json`
before starting SGLang. The test configuration digest and checkpoint
configuration digest are included in provenance; the first on-machine run must
establish the reviewed checkpoint digest baseline.

Candidate generation is deterministic for every model: temperature 0 and top-p 1
throughout, so the rule-based verdict is reproducible. Only the output budget
moves, and only where a checkpoint's own template forces it to: 2048 tokens for
the eleven entries that can be asked not to think; 8192 for the two 2.4T entries
and two-node Kimi entry; 16384 for MiniMax W8A8; and 32768 for MiniMax FP8
channelwise and MXFP4, whose template has no off switch at all.

Server parameters follow the internal case each config was ported from, so they
differ by model rather than by house rule. The 397B is TP=8, FA3 attention, static
memory fraction 0.8; the 27B is TP=1, FA3, 0.85, and `unquant`, which is how
`server_args` spells an explicit opt-out for a BF16 checkpoint; the 2.4T FP8 entry
is TP=8 × PP=4 and the MXFP4-FP8 one TP=8 × PP=2; GLM-5.2 serves through the
sparse `dsa` backend with `flashmla_sparse` prefill and `flashmla_kv` decode, and
Kimi-K2.6 names `fa3` prefill against `flashmla` decode with no unified backend at
all. Every config whose checkpoint the loader handles unaided leaves the
quantization flag off so the checkpoint's own declaration stands; the W8A8-INT8
ones name `w8a8_int8` and the BF16 ones say `unquant` — see
[Parity with the internal launcher](#parity-with-the-internal-launcher). Naming
`w8a8_int8` is required rather than cosmetic: read off the checkpoint instead, that
format resolves to the compressed-tensors path, whose W8A8-INT8 fused-MoE scheme
raises `NotImplementedError` outside NPU
(`sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`), which
is how the GLM-5.2 and MiniMax-M2.7 INT8 entries first failed (run 34085800820)
while the Qwen3.5 one, which already named the format, served the same suite.
Each config carries the `watchdog_timeout` its source case carries, 600 everywhere
except the GLM-5.2 channelwise entry's 24000.

`dist_timeout` is set from the btv1.5 `server_cmds` set, which passes it on every
single command it lists while the `answer_144g` plan these configs were ported
from passes it on none: 6000 for the Kimi-K2.6, MiniMax-M2.7, Qwen3.5 and two of
the GLM-5.2 entries, which is what that set's own SGLang eval cases use; 24000 for
the GLM-5.2 channelwise entry, matching the context-parallel case it was ported
from, where the value equals its watchdog; and 60000 for the two 2.4T entries, the
only value that model's file carries — it lists no SGLang eval case, so the figure
comes from its performance cases. The two 27B entries are left without it, because
at TP=1 there is no multi-rank group for it to bound. This is alignment with the
reference commands, **not** a fix for anything measured: the value reaches only the
device-side groups, since `parallel_state.py` builds every `gloo` companion group
with its own hard-coded 120-minute `gloo_timeout` instead, so it is unrelated to
the resolver failure recorded under
[The ZW-M890P line](#the-zw-m890p-line). Those cases keep `dist_timeout` and
`watchdog_timeout` equal; outside the channelwise entry these do not, because a
watchdog is what makes a wedged forward give the board back promptly and 600 has
been measured to be enough.

`SGLANG_WARMUP_TIMEOUT` is 3600 on every config, the one value the btv1.5
`server_cmds` set uses — all 121 commands across its thirteen model files, without
an exception to carry over. Unlike `dist_timeout` this one has measured headroom
behind it. It bounds the single warmup request `_wait_and_warmup` sends before the
server reports itself ready, and unset it is not unbounded: `http_server.py` falls
back to 600 seconds, and a warmup that overruns kills the process tree rather than
failing soft. That request is where DeepGEMM compiles the kernels the first real
forward needs, and on the 144GiB board it took 65 seconds for Kimi-K2.6 MXFP4 and
299 to 325 for the three Qwen3.5 entries — better than half the default already,
and not a warm-cache best case either, since those requests log
`Try DeepGEMM JIT Compiling` while they run (run 34085800820). The much larger
figures in those logs belong to CUDA-graph capture, 621 to 1009 seconds, which
finishes before the HTTP server listens and is bounded by
`startup_timeout_seconds` instead.

## The ZW-M890P line

`.github/workflows/test-ppu-answer.yml` runs the single-board suites against
the 144GiB board, which is reachable only through the K8s cluster. It is a sibling
workflow rather than a second matrix dimension of the bare-metal one because
everything around the test differs, while the test itself does not:

- **Selection.** The board is chosen by `node_selector: board-type=ZW-M890P`, and
  the work runs in a worker pod submitted by `ppu-distributed-action` from a
  CPU-only orchestration shell that holds no device.
- **Devices.** `nproc_per_node` is a resource request, not isolation. A pod that
  asks for one PPU still sees all eight `/dev/alixpu_ppu*` nodes, and
  `torch.cuda.device_count()` still returns 8, so the preflight — which requires
  the visible device count to equal the configured one — would fail the
  single-card entry outright. `CUDA_VISIBLE_DEVICES` is therefore exported inside
  the pod from the same config the test reads, and every entry requests all eight
  PPUs so that no second pod can land on the board and contend for a device with
  a server that has already claimed most of its memory. The internal btv1.5 plan
  schedules its answer cases as `1node8ppu` for the same reason.
- **Lanes, not a matrix.** Eleven entries, one job id each, arranged into eight
  lanes so at most eight boards are held at once: three lanes carry two entries,
  the second declaring `needs` on the first, and five carry one. A matrix would
  have been the obvious shape and was the first one tried, and it does not work
  above `max-parallel: 1`. `flytiger-eco/ppu-distributed-action` builds both the
  NAS directory it stages the source to and the name of the K8s job it submits
  out of `$GITHUB_JOB`, and a matrix leg is not a separate job id — so twelve
  legs staged into one directory, and the second leg to copy a git pack file,
  mode 444, died on `EPERM`; all twelve failed within two minutes of starting
  (run 34082162222). Had the copy succeeded they would then have submitted twelve
  pods under one K8s job name, each leg's cleanup deleting the others' pods. The
  value cannot be redirected from the caller either: `GITHUB_JOB` is set by the
  runner and a job-level `env:` entry of the same name does not override it
  (measured, run 34084233664), and the pod name comes from an expression the
  action fixes at step level. `needs` with `if: ${{ !cancelled() }}` is the
  `fail-fast: false` of this shape — a lane's second entry runs on its
  predecessor's verdict whether that verdict was green or red, while a cancelled
  run stops asking for boards. Each entry carries a slug distinct from its suite,
  because three entries share `nightly-answer-8-ppu` and three more share
  `nightly-answer-8-glm52-ppu`: the slug is what names the NAS results directory
  and the artifact, and `upload-artifact@v4` fails outright on a repeated name.
  `--suite` still receives the registered name.
- **One body, eleven jobs.** The two long shell bodies each entry runs — the
  board-side one that installs, warms and calls `run_suite.py`, and the
  orchestration-side one that carries the report off the NAS — are
  `scripts/ci/ppu/run_answer_suite_board.sh` and
  `scripts/ci/ppu/collect_answer_evidence.sh`, so eleven jobs do not carry eleven
  copies of them; what the workflow still states per entry is the K8s action's
  inputs. A local composite action would have been the tidier shape and does not
  work on this runner group: it drives the job through a container hook, so
  `actions/checkout` populates a workspace inside the container
  (`/__w/sglang-for-sail/sglang-for-sail`) while the runner resolves `uses: ./`
  against its own filesystem (`/home/runner/_work/...`), and all twelve jobs
  failed on a missing `action.yml` seconds after a checkout that reported success
  (run 34085102675). A `uses:` naming a published repository is unaffected,
  because the runner downloads that itself, which is why the K8s action works at
  all — and it is the same split filesystem that silently drops
  `GITHUB_STEP_SUMMARY` here. Everything the pod needs that does not vary between
  entries is exported by the board script rather than repeated in eleven
  `extra_env` strings; the secrets, the provenance of the run and the four values
  that do vary are passed.
- **Checkpoint path.** The 397B weights live under `T-HEAD/v3.5/` on this NAS
  rather than under `qwen/v3.5/` as on the ZW810E line; the 27B path is the same
  on both. The path is a per-config field, so this costs nothing beyond the two
  new configs — `checkpoint_name` is unchanged, because only the parent directory
  differs.
- **Evidence channel.** The action returns pod logs but not pod files, so the
  report travels over the NAS that both sides mount: the pod writes
  `SGLANG_PPU_ANSWER_RESULTS_DIR` under `/mnt/wl_nas/devops/<run>/<job>/`, and the
  orchestration shell reads the same bytes under `/wl_nas/...`, publishes the
  summary, and uploads the artifact. The entry slug is part of the path as well
  as the job id, so the directory reads without a mapping from one to the other.
  The NAS copy is read and left
  in place rather than moved: the pod writes as root and the orchestration shell
  is a different, non-root uid, so it can read those bytes but not unlink them.
  A step summary does not survive this runner at all, so the failing cases reach
  the run page as annotations instead; see
  [Results and annotations](#results-and-annotations).
- **Provenance.** `base_image_digest` is null on this path. The orchestration
  shell has no Docker daemon to inspect the image with and the pod cannot see its
  own digest, so only the image tag is recorded; on the bare-metal line the digest
  is resolved by `docker image inspect`. The daemon is not the only way to ask,
  though: the performance lines resolve the digest from the registry over HTTPS in
  `scripts/ci/ppu/resolve_image_digest.sh` and pin the pods to it, and this line
  could be wired the same way. It has not been, so the null here is a gap rather
  than a limit of the path.
- **Trigger.** No cron. GitHub honours `schedule` only on the default branch, so a
  cron on a version branch would never fire while claiming the workflow is
  scheduled. The line is dispatched by hand or through `workflow_call`.
- **Warm.** The page-cache warm runs inside the pod, immediately before
  `run_suite.py`. It has to: the cache it warms belongs to the node that will then
  load the weights, which no orchestration-shell step can reach.
- **Gloo interface.** The board script exports `GLOO_SOCKET_IFNAME=lo`. Gloo picks
  its address by resolving the pod's own hostname, and on some nodes of this
  cluster that hostname has no address: across the eight nodes one batch landed
  on, six ranks logged torch's own `Unable to resolve hostname to a (local)
  address ... Manually set the network interface to bind to with
  GLOO_SOCKET_IFNAME` and fell back to loopback, while two raised out of the
  fallback — `[enforce fail at .../gloo/transport/tcp/device.cc:99] rv == 0. -5 vs
  0`, `EAI_NODATA` — and killed three entries at `torch.distributed.new_group`
  before a weight was read (run 34085800820, nodes swu10/swu12/swu15). The failure
  did not track the attention backend or the checkpoint, and no two of the eight
  entries shared a node, so it is the node's resolver rather than contention.
  Naming the interface takes the resolver out of the path. `lo` is right for this
  script and only this script: its entries are single pods whose ranks are
  processes in one network namespace. The multi-node entries must not copy it;
  they derive the interface that faces their peers instead, through
  `scripts/ci/ppu/answer_gloo_iface.sh` — see *The group is told which interface
  gloo binds to* below.

The board's identity and capacity are measured rather than assumed. An inventory
probe submitted to this cluster on 2026-09-02 reported eight devices named
`ZW-M890P` with compute capability 8.9, 39 multiprocessors, 32MB of L2, and
`total_memory` 147456MB — 144.0GiB exactly — under torch 2.11.0, with both
checkpoints present (94 and 18 shards) and the JIT and Hugging Face caches in
place. `hardware.generation` is a free-form string that nothing validates against
a list, so a guessed value would be indistinguishable from a measured one in the
provenance record; `zw-m890p` and `memory_gib_per_device: 144` are what the
hardware reported. Board type alone would not have settled the capacity in any
case: the internal plans schedule this same board in both a 96GiB and a 144GiB
configuration.

The two entries that predate this batch have since been run on this board; see
[Measured baseline (ZW-M890P)](#measured-baseline-zw-m890p). The nine added here
have not been run yet.

## The four-node line

`nightly-answer-32-ppu` serves one checkpoint across four ZW-M890P nodes at
TP=32. Its internal plan,
`btv1.5_P1_sglang_1node_func_answer_4nodes.csv`, schedules it as `4node8ppu` on
`M890P`, which is where the topology comes from: the case JSON itself carries only
`tp: 32`. The suite body is the same ten prompts and the same deterministic
evaluator; what is new is that four processes have to become one server, and
that only one of them can grade the result.

**The config states the topology, the environment states the rendezvous.**
`hardware.nnodes` is the reviewed fact — it says the checkpoint is served across
four nodes — and `hardware.visible_devices` keeps meaning *per node*, so
`validate_test_config` requires `tp_size * pp_size == len(visible_devices) *
nnodes`, which is the same product SGLang's own `check_server_args` measures
against the node count. There
is deliberately no `devices_per_node` field: it would restate the length of a
list that is already there, and two spellings of one number drift. `nnodes`
defaults to 1, so every existing config and the string
`answer_expected_hardware` renders for it are untouched.

**The 32 devices are 8-way tensor parallel over 4 pipeline stages, not 32-way
tensor parallel.** This is the one place the entry departs from the number the
internal case states, and the reason is arithmetic in the checkpoint rather than
a preference. `config.json` declares `moe_intermediate_size: 2048` and
`quantization_config.weight_block_size: [128, 128]`, and the FP8 path shards the
expert weight column-wise before it lays out the scales, so
`create_weights` in `layers/quantization/fp8.py` requires
`2048 / tp_size` to be a multiple of 128 — satisfied up to `tp_size` 16, and
`tp_size` 32 gives 64. Run `33813065317` is that failure, reached only after the
cross-node group had formed and weight loading had begun:

```
ValueError: The output_size of gate's and up's weight = 64 is not divisible by
weight quantization block_n = 128.
```

A second constraint is independent of quantization: the hybrid layers declare
`linear_num_key_heads: 16`, which 32 does not divide, so the linear-attention
heads cannot be split 32 ways either. Both are satisfied by `tp_size` 8, and the
remaining factor of four goes to the 92 layers: `pp_size` 4. SGLang lays a rank
out as `pp_rank * tp_size + tp_rank`, so each node ends up holding exactly one
pipeline stage — every tensor-parallel collective stays inside a node, and the
only cross-node traffic is the pipeline's point-to-point hand-off. That is a
lighter demand on the fabric than `tp_size` 32 would have made, not a heavier
one.

This is also the shape the vendor documents for this checkpoint on four nodes:
`server_cmds/LLM_Serving/BTV1.5/Qwen3.8-2.4T-A95B.md` in `model-test-cases`
gives `--tp-size 8 --pp-size 4` for the FP8 weights, and its 32-way command is a
different configuration entirely — expert parallel over DeepEP with data-parallel
attention, which needs a transport PPU does not have here. The vendor's own
evaluation command for the sibling MXFP4 checkpoint reads `--nnodes 2 --tp-size
16`, where `2048 / 16` is exactly 128.

`pp_size` is optional in the schema and absent from every other config, so those
entries emit the command line they emitted before the field existed; a config
that names it emits `--pp-size` immediately after `--tp-size`.

Which node a given process is, and where the group meets, are not reviewable
facts: they are decided by whatever launched the four pods. `resolve_distributed_runtime`
reads them from the environment and returns `None` for a single-node config:

| Variable | Meaning |
| --- | --- |
| `NODE_RANK` | this pod's rank, injected per pod by `ppu-distributed-action` |
| `NNODES` | the group size the launcher started, checked against `hardware.nnodes` |
| `MASTER_ADDR`, `MASTER_PORT` | rank 0's address, composed into `--dist-init-addr` |
| `SGLANG_PPU_ANSWER_NODE_RANK` | overrides the rank |
| `SGLANG_PPU_ANSWER_DIST_INIT_ADDR` | overrides the address, as `host:port` |

`NNODES` is checked rather than used: the group size is stated twice, once in the
reviewed config and once in the workflow that asks the cluster for boards, and a
launcher that started too few pods leaves every rank it did start with a valid
one. The shortfall would then surface only as a rendezvous that never completes,
after the group had held the boards it did get for the whole 5400s startup
budget. A launcher that states nothing is left alone, which is the bare-metal
case, and a single-node config is never asked, so the `NNODES=1` the action
injects for every single-board entry cannot fail one of those.

The two overrides are not conveniences. `ppu-distributed-action` asks for host
networking while leaving `spec.dnsPolicy` at ClusterFirst, so the pods get the
host resolver and the cluster-internal name it puts in `MASTER_ADDR` does not
resolve — measured identically on all four ranks of run `33750074634`. A patch
is with the action's owners; until it lands, the caller discovers rank 0's
address another way and passes it through the override. Everything the launch
needs is otherwise derived: the resolved rendezvous becomes `--nnodes`,
`--node-rank`, and `--dist-init-addr`, appended to the argument list the
single-node path already produced, and `build_answer_server_args` refuses a
multi-node config launched without one — the alternative is four independent rank
0 processes each trying to fit 2324.7 GiB onto eight devices.

**Every node runs the same file.** `AnswerSuiteMixin` does not branch until after
the server is up, because it does not have to: for `node_rank >= 1`, SGLang's
`launch_server` brings up its schedulers and then serves a dummy health endpoint,
so `popen_launch_server` returns on all four nodes and one launch path covers the
group. The branch is in what happens next.

| | Rank 0 | Ranks 1–3 |
| --- | --- | --- |
| Holds | the tokenizer and the HTTP API | its eight devices in the group |
| Does | generates, grades, writes the report | waits |
| Passes when | the corpus passes | it held the group until rank 0 was done |
| Fails when | a case fails or the server dies | its own server dies first, or rank 0 never finishes |

A worker cannot return early: its schedulers own their slice of every request, so
leaving would tear the group down under rank 0. It also cannot wait forever, so
the hold is bounded by the reviewed request budget for the whole corpus plus
`WORKER_HOLD_MARGIN_SECONDS`.

**The nodes address each other through the results directory.** They have no
other shared surface — the action streams worker-0's log and returns no files —
so `SGLANG_PPU_ANSWER_RESULTS_DIR` must be an absolute path on the NAS every pod
mounts, and the multi-node path refuses a relative one rather than letting each
pod write to its own copy of the same name. Underneath it, `ranks/` carries:

- `rank-<n>-devices.json`, each node's own device inventory, written staged and
  renamed because a reader on NFS can otherwise observe a partial file. Rank 0
  reads all four into the report's `accelerator` record, so the provenance
  describes the thirty-two devices the verdict was produced on rather than the
  eight rank 0 could see. A node whose inventory never arrived is named in
  `node_ranks_without_inventory` rather than dropped.
- `rank0-complete`, written when rank 0 is done. Workers poll it with a
  `listdir` rather than an `exists`, since a readdir revalidates the negative
  lookup NFS would otherwise cache.
- `rendezvous`, the address rank 0 published for the group to meet at, written by
  `scripts/ci/ppu/answer_rendezvous.sh` before anything else and read by the
  other three. Also staged and renamed, for the same reason.
- `rank-<n>.log` and `rank-<n>.status`, each node's own output and its own exit
  code, written by `scripts/ci/ppu/run_answer_suite_node.sh`.

Rank 0 writes its report where the workflow collects it; a worker writes its own
under `ranks/rank-<n>/`, so a worker's view of a run it did not grade cannot
overwrite the verdict. The single-node report is byte-identical to what it was:
the multi-node keys are additive.

The sentinel is written *before* rank 0 kills its own server. In the other order,
killing rank 0 makes the workers' schedulers exit, each worker sees its own server
die, and a healthy run reports three failed pods.

The exchange is under test without a board.
`TestPPUAnswerMultiNodeExchange` in `test_ppu_answer_eval_unit.py` binds the mixin
to each of four ranks against a real temporary directory with a stubbed device,
and covers the report destinations, the inventory round trip including a node that
reported nothing, the release, the three worker outcomes, and the teardown order.
It skips itself where torch is absent, since the driver needs it and the evaluator
does not.

**Measured baseline, four nodes.**
[Run 33849322347](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/33849322347),
2026-09-04, four ZW-M890P boards (thirty-two devices), driver 1.6.1, SDK 2.1.1,
at `b7b48ee`. **Ten of ten cases passed**, verdict `passed`, no suspect case, and
all four ranks exited 0.

| Phase | Cost | Observation |
| --- | --- | --- |
| `torch.distributed` init | 29.5s | the group forms across all four boards |
| Weight load | 14m42s | 213 shards, `type=Qwen3_5MoeForCausalLM`, `quant=fp8`, 72.36 GiB per device |
| CUDA graph capture | 5m30s | against the 46.4 GiB left free per device |
| Launch to ready | 22m44s | `/generate` warm-up 200 OK, well inside the 5400s `startup_timeout_seconds` |
| Ten graded requests | 57s | every one 200 OK; 16:01:35 to 16:02:32 |
| Registered file, total | 1455s | against `est_time` 7200 |

`max_total_num_tokens` comes out at 3875020 with a 262144 context and
`max_running_requests` 96. The 503s on `/health_generate` before ready are the
driver polling a scheduler that has not finished warming, not a fault.

The reasoning split is visible in the report: at `reasoning_effort: low` the
longest case spends 389 of its 418 completion tokens on reasoning and none of it
reaches the graded text — `final_answer` for the letter-count case is `4个。` — and
`reasoning_sha256` records the reasoning separately for every case. The widest
case uses 418 of the 8192 token budget, so `max_tokens` is not the binding
constraint at this effort.

`startup_timeout_seconds` 5400 and `est_time` 7200 remain as they were: both hold
comfortably against these numbers, and leaving headroom for a cold page cache is
deliberate on a node whose memory is smaller than the checkpoint tree.
`WORKER_HOLD_MARGIN_SECONDS` 900 was never approached — the workers finished
within 27s of rank 0.

**The workflow is its own file, dispatchable and orchestrated.**
`.github/workflows/test-ppu-answer-32.yml` claims four whole boards, so it is
not one more lane in the btv1.5 workflow — adding it there would make every
routine btv1.5 dispatch ask the cluster for four more boards — but it is now
also reached by the `nightly-test-ppu-answer.yml` orchestrator through
`workflow_call`, even though it has not yet had a green measured run. `nnodes: 4` is what makes
the action gang-schedule: it creates a PodGroup with `minMember: 4`, so the group
either gets all four boards or waits, rather than half a group holding sixteen
devices while the rest never arrives. `nproc_per_node: 8` is the whole board on
each, which is both what TP=32 across four nodes needs and the only fence that
keeps a second pod off a board this run has claimed.

Five things differ from the single-board entries, each for a measured reason.

**Rank 0 publishes the rendezvous, first thing.**
`scripts/ci/ppu/answer_rendezvous.sh` runs before the dependency install: rank 0
detects its own address and writes it into `ranks/rendezvous`, and the other three
read it there and export it as `SGLANG_PPU_ANSWER_DIST_INIT_ADDR`. It has to be
done this way because the name the action injects as `MASTER_ADDR` does not
resolve in these pods, and it is done *early* so a worker waits only on its own
install rather than on rank 0's. The address is detected in the order SGLang's own
`get_local_ip_auto` uses — explicit host IP, then the source address the kernel
would pick for an outbound route, then the hostname — so the group meets at the
address the server would have chosen for itself. The detection is written against
the standard library rather than imported from `sglang`, because it runs before the
editable install and the rendezvous of a test should not need the tree under test
to be importable.

**The group is told which RoCE GID to use.** `scripts/ci/ppu/answer_gid_index.sh`
runs next and exports `NCCL_IB_GID_INDEX`. Left to itself pccl picks index 0 on
these hosts, which is the link-local `fe80::<EUI-64>` address formed from the
bond's MAC: it does not cross the L3 fabric between two nodes, so the group
builds its queue pairs over the out-of-band TCP channel, logs `Connected all
rings`, and then dies on its first payload with `IBV_WC_RETRY_EXC_ERR` — packets
leave and the peer never acknowledges them. What the bonds do carry is a global
ULA under `fd03::/8`, present as a consecutive GID pair, RoCE v1 then v2. A
two-node sweep of all six indices on these boards measured exactly one that
works:

| GID | address | result |
| --- | --- | --- |
| 0 | `fe80:…:<MAC>`, link-local | hang, then `SIGABRT` — **this is what pccl picks by itself** |
| 1 | `fe80:…:<MAC>`, link-local | `ibv_modify_qp` fails, connection timed out |
| 2 | `fd03:45c2:1:XXXX::1`, global | `IBV_WC_RETRY_EXC_ERR` |
| 3 | `fd03:45c2:1:XXXX::1`, global | `all_reduce` completes, checksum correct |
| 4, 5 | `fe80:45c2:…` | `IBV_WC_RETRY_EXC_ERR` |

The script derives the index — highest non-link-local GID, which is the v2 entry
of the routable pair — rather than writing `3` down, because the table is built
in whatever order the kernel added the addresses, and a node that ordered them
differently would otherwise fall silently back to a GID that drops every packet.
It exits non-zero if a device exposes no routable GID or if the four devices
disagree, both being conditions under which the group must not start: failing at
the entrance costs seconds, failing on the fabric costs the whole run.

Note the boundary: this compensates for a *host* condition, RoCE bonds with no
IPv4, which is what leads pccl's own preference — a v2 GID derived from the
device's IPv4 — to find no match and fall back. On a host whose bonds carry
IPv4, the automatic choice is already correct and this script would simply agree
with it.

**The group is told which interface gloo binds to.**
`scripts/ci/ppu/answer_gloo_iface.sh` runs next and exports
`GLOO_SOCKET_IFNAME`. Gloo carries the CPU side of every process group SGLang
creates, and it picks its address by resolving the pod's own hostname — the same
host condition that costs the single-board entries their ranks, with a second
failure mode when the group spans nodes. On the two-node entry of run
34109451085, whose rank 0 landed on `na131t-cloud-swu12`, three ranks logged
torch's `Unable to resolve hostname to a (local) address ... Manually set the
network interface to bind to with GLOO_SOCKET_IFNAME` and fell back to loopback,
advertising `127.0.0.1` to a peer on the other node that can only reach its own;
the fourth raised out of the resolver instead and killed the server at
`torch.distributed.new_group` before a weight was read — `[enforce fail at
.../gloo/transport/tcp/device.cc:99] rv == 0. -5 vs 0`, `EAI_NODATA`, the name
resolving to no address at all. All ten cases were recorded as
`server_start_failed`, six minutes in, and the four-node entry shares the cause:
it had only been landing on nodes whose hostname resolves.

The interface is derived rather than named, because these hosts number their
bonds differently and a wrong name would put the group back on loopback. The
derivation is the address the group already agreed to meet at: rank 0 publishes
it, so on rank 0 it is a local address and on every other node it is the address
whose route selects the interface facing rank 0 — either way the kernel's own
source-address choice for that destination names the interface gloo has to use.
Measured on a node of this fleet with eight interfaces: the rendezvous address
yields the interface that owns it, an off-fleet destination yields the interface
its route selects, and a destination that resolves to loopback is refused. That
refusal is deliberate — the script exits non-zero with its interface inventory on
stderr rather than letting gloo fall back silently, since a fallback costs the
whole multi-hour run and the refusal costs seconds. Written against the standard
library for the same reason as the rendezvous script: it runs before the editable
install and must not depend on the tree under test.

**No page-cache warm.** A warm reads the whole checkpoint tree, and at 2324.7 GiB
that is larger than one node's 2266 GiB of memory, so the beginning of it is
already evicted by the time the read ends; run on all four nodes it would also
read four times what the group needs, since each node loads roughly its own
quarter. It would cost hours of NAS traffic to leave the cache no warmer than it
started. The single-board entries keep their warm, where the checkpoint does fit.

**Each node keeps its own log and its own exit code.** The action streams
worker-0's log and no other pod's, and a worker is exactly the node that would
report a device shortfall or a lost server, so
`scripts/ci/ppu/run_answer_suite_node.sh` tees each node's output into `ranks/`
and records its status there. `tee` rather than a redirect, so rank 0's log still
arrives live over a run this long; `PIPESTATUS` rather than the pipeline's status,
because handing back `tee`'s exit code would turn every failed suite green. The
collect step turns those status files into annotations: rank 0's verdict is
written before the workers are released, so a worker that failed afterwards is
visible in nothing else. The suite itself still runs through `run_suite.py`, so
the executed set is what `register_ppu_ci` declares.

## The ported 144GiB entries

Twelve configs were added from the internal btv1.5 `answer_144g` plan, against
checkpoints already staged on this NAS; eleven remain, the `kimi2.6-w4a8-int8`
entry having been removed when its format proved unservable on this board line
(see the departures below). Each row names the internal
`llm_infer_sglang_evalscope` case its server parameters came from, and the
checkpoint as measured on `na131t-ppu810e-test001`, which mounts the same NAS the
cluster does. Every one of them holds exactly the number of shards its own
`model.safetensors.index.json` declares, so none is a partial copy — read from the
index, not from the `-of-NNNNN` suffix the filenames carry, which on
`MiniMax-M2.7-W8A8-INT8` is stale.

| Entry | Suite | Source case | Checkpoint | Size |
| --- | --- | --- | --- | --- |
| `glm5.2-w8a8-int8` | `-8-glm52-` | `glm-5.1-w8a8-int8_3001` | `v5.2/GLM-5.2-W8A8-INT8` | 704.4 GiB, 282 shards |
| `glm5.2-mxfp4-fp8` | `-8-glm52-` | `glm-5.2-fp8_3001` | `v5.2/GLM-5.2-MXFP4-FP8` | 383.0 GiB, 282 |
| `glm5.2-fp8-channelwise` | `-8-glm52-` | `glm-5.2-fp8_channel_cp_3001` | `v5.2/GLM-5.2-FP8-Channelwise` | 704.4 GiB, 282 |
| `kimi2.6-mxfp4-fp8` | `-8-kimi26-` | `kimi-k2.6-w8a8-int8_3001` | `k2.6/Kimi-K2.6-MXFP4-FP8` | 516.2 GiB, 64 |
| `kimi2.6-w8a8-int8` | `-16-kimi26-` | `kimi-k2.6-w8a8-int8_3001` | `k2.6/Kimi-K2.6-Quant-W-INT8-PerChannel-A-INT8-PerToken` | 968.3 GiB, 64 |
| `minimax2.7-fp8-channelwise` | `-8-minimax27-` | `minimax-m3-bf16_3001` | `M2.7/MiniMax-M2.7-FP8-Channelwise` | 214.7 GiB, 86 |
| `minimax2.7-mxfp4-fp8` | `-8-minimax27-` | `minimax-m3-bf16_3001` | `M2.7/MiniMax-M2.7-MXFP4-FP8` | 116.2 GiB, 86 |
| `minimax2.7-w8a8-int8` | `-8-minimax27-` | `minimax-m3-bf16_3001` | `m2.7/MiniMax-M2.7-W8A8-INT8` | 214.6 GiB, 125 |
| `qwen3.5-397b-fp8-channelwise` | `-8-` | this repository's own `-144g` entry | `v3.5/Qwen3.5-397B-A17B-FP8-Channelwise` | 379.0 GiB, 94 |
| `qwen3.5-397b-mxfp4-fp8` | `-8-` | this repository's own `-144g` entry | `v3.5/Qwen3.5-397B-A17B-MXFP4-FP8` | 215.6 GiB, 94 |
| `qwen3.8-2.4t-a95b-mxfp4-fp8` | `-16-` | `qwen3.7-mxfp4-fp8_3001` | `v3.8/Qwen3.8-2.4T-A95B-...-MXFP4-...` | 1272.1 GiB, 213 |

Six of the eleven entries name a source case that does not carry their
checkpoint, in four groups. That is the plan's own situation rather than a
substitution:

- **GLM-5.2 W8A8-INT8.** The plan has no GLM-5.2 W8A8 case, and neither does the
  btv1.5 `server_cmds` set, which lists only `fp8-channel` and `mxfp4-fp8` for
  GLM-5.1 and GLM-5.2 alike; this entry exists because the NAS holds the weights.
  The nearest case is `glm-5.1-w8a8-int8_3001`, whose parameters are identical to
  `glm-5-w8a8-int8_3001`, so the W8A8 serving line is stable across those
  revisions and is what this entry carries, with `quantization` named for the
  reason given under the departures below.
- **Kimi-K2.6, the MXFP4 format.** `kimi-k2.6-w8a8-int8_3001` is the only Kimi
  answer case, and its own format is the W8A8 entry below, so what carries over
  here is everything that is not format-specific — TP=8, `fa3` prefill against
  `flashmla` decode, memory fraction 0.8. The entry leaves the quantization flag
  off, which its checkpoint's own declaration covers; the W4A8 sibling that once
  shared this derivation is removed, the gap recorded under the departures
  below.
- **MiniMax-M2.7, all three formats.** `minimax-m3-bf16_3001` is the only MiniMax
  case on the sglang side; `minimax-m3-mxfp4-fp8_3001` exists on the vllm side
  only. Its parameters are TP=8, `fa3`, 0.8, and `watchdog_timeout` 600, none of
  which is specific to BF16 or to M3.
- **Qwen3.8-2.4T MXFP4-FP8.** Server parameters come from
  `qwen3.7-mxfp4-fp8_3001`; the topology and the template handling come from this
  repository's four-node FP8 entry, which is the same checkpoint at a different
  quantization and the only place either has been established.

`kimi2.6-w8a8-int8` is the one entry whose source case names its exact
checkpoint, which is why it is the only Kimi config that states a `quantization`
at all. It departs from that case in topology instead, for a reason of
arithmetic recorded under Capacity below.

**Where these ports depart from their sources.** Eight departures beyond the
deterministic generation line already described in
[Relation to the internal test cases](#relation-to-the-internal-test-cases):

- The sparse-attention parameters are spelled `dsa_*`, not the `nsa_*` the GLM
  cases use. This tree renamed the whole family and keeps the old spellings only
  as deprecated aliases, so an `nsa_*` config would work today, warn, and stop
  working without notice.
- `SGLANG_NSA_DUAL_STREAM=0`, which the GLM cases export, is refused by the
  environment whitelist. Nothing in this tree reads that name — the only
  `DUAL_STREAM` symbols are the module constants `DUAL_STREAM_TOKEN_THRESHOLD` in
  `dsa_indexer.py`, `qwen3_5.py`, and `qwen3_next.py` — so accepting it would let
  a config state a setting no run honours.
- `enable_metrics` and `tool_call_parser`, which the GLM channelwise case sets,
  are not carried: nothing collects the metrics endpoint on this path, and no
  request in this corpus asks for a tool call.
- Kimi and MiniMax get a `reasoning_parser` their source cases do not pass, for
  the reason already recorded for the Qwen entries: the requests ask for
  `separate_reasoning`, and the parser is what keeps a reasoning block out of the
  graded text.
- The GLM-5.2 and MiniMax-M2.7 W8A8-INT8 entries name `quantization` where their
  source cases leave it to the checkpoint. This is forced, not preferred: those
  checkpoints declare compressed-tensors, and that path's W8A8-INT8 fused-MoE
  scheme raises `NotImplementedError` on anything but NPU, which is how both
  entries failed before the flag was added (run 34085800820). Naming the format
  routes the MoE layers to the `w8a8_int8` implementation the Qwen3.5 entry has
  been serving this suite with.
- `kimi2.6-w4a8-int8` had no equivalent escape, and the gap is why the entry is
  gone. Its checkpoint was W-INT4-per-channel against A-INT8-per-token,
  `BASE_QUANTIZATION_METHODS` offers no `w4a8_int8`, and `w4afp8` is a different
  activation type, so the loader's own choice of `mixed_precision_w4` was the
  only route available. That route reached the board and failed inside the
  kernel — `[ACEXT][ERROR] CUDA
  runtime error: invalid argument` from `compute_occupancy.h:59`, under
  `acext.fusedmoe_wrapper` — which no server parameter this schema carries can
  redirect. The entry reddened on `server_start_failed` on every run and was
  dropped from the suite.
- `dist_timeout` is carried although the plan's cases pass none, taking the value
  the btv1.5 `server_cmds` set uses for each model; the reasoning, and the reason
  it is not paired with `watchdog_timeout` the way those commands pair it, is under
  [Runner configuration](#runner-configuration) above.
- `SGLANG_WARMUP_TIMEOUT` reaches every config, where only the GLM-5.2 channelwise
  port carried it from its own source case. The btv1.5 set puts 3600 on every
  command it lists, and the measurement under
  [Runner configuration](#runner-configuration) says why that is worth following
  rather than a formality: the fallback this replaces is 600 seconds, half of which
  the Qwen3.5 entries already spend.

**The MiniMax budgets are run-validated.** All three entries retain a 900s request
timeout. W8A8 remains at `max_tokens` 16384 after completing nightly run
35528870244. FP8-channelwise and MXFP4 use 32768: targeted run 35551775837
completed the previously critical FP8-channelwise case in 6728 output tokens,
and MXFP4 met its configured threshold with only its accepted known failure.

**Capacity.** These are headroom checks against the measured checkpoint sizes, not
predictions of what the server will actually reserve:

| Group | Static pool at the configured fraction | Largest checkpoint on it |
| --- | --- | --- |
| 8 × 144 GiB, fraction 0.8 | 921 GiB | 704.4 GiB (GLM-5.2 W8A8-INT8) |
| 8 × 144 GiB, fraction 0.9 | 1036 GiB | 704.4 GiB (GLM-5.2 channelwise) |
| 16 × 144 GiB, fraction 0.8 | 1843 GiB | 1272.1 GiB (2.4T MXFP4-FP8) |

That third row is why `kimi2.6-w8a8-int8` is a two-node entry while its source
case is a single-node one. Kimi-K2.6-W8A8-INT8 is 968.3 GiB, which is 84 per cent
of one node's 1152 GiB of device memory: at the source case's own fraction of 0.8
it exceeds the 921 GiB pool and cannot finish loading, and 0.9 would leave about 8
GiB per device for the KV cache with nothing measured to say that is workable.
Two nodes at TP=8 × PP=2 keep the fraction at 0.8 against a 1843 GiB pool. The
internal plan schedules its `answer_144g` cases as `1node8ppu` and does not list
this one, which is consistent with the arithmetic above.

**Time.** The measured page-cache warm rate is about 4.15 s/GiB at
`WARM_PARALLELISM` 8 on both boards — 26m17s for 379.0 GiB on ZW810E, 3m34s for
51.7 GiB on ZW-M890P — which puts the 704.4 GiB entries near 49 minutes of warm
before a weight load begins. The per-entry budgets follow from that: pod timeouts of
210 to 330 minutes and `timeout-per-file` of 10800 or 14400 seconds, with the
larger figures on the GLM and MiniMax entries. `est_time` is 7200 for GLM,
MiniMax, both 2.4T suites, and the two-node Kimi one, and 5400 for the
single-board Kimi suite; none of these is measured either, and the first
successful run of each is what should replace it.

**The two-node line.** `.github/workflows/test-ppu-answer-16.yml` runs
`nightly-answer-16-ppu` and `nightly-answer-16-kimi26-ppu` on two ZW-M890P nodes
each at TP=8 × PP=2, one entry at a time. It is the four-node workflow with
`nnodes: 2`, so it keeps every mechanism that line established —
the rendezvous file exchange, the RDMA GID index resolution, the gloo interface
derivation, `run_answer_suite_node.sh`
for the non-zero ranks, and the rank status collection — and differs only in the
group size and the configs it names. The two entries are there for different
reasons: 1272.1 GiB does not fit one node's static pool at any fraction, while
968.3 GiB fits neither 0.8 nor, with any credible KV cache left over, 0.9. Like
the four-node entry neither runs a page-cache warm, but for a different reason:
both checkpoints would fit in one node's 2266 GiB of host memory, yet under PP=2
each node loads roughly its own half, so warming the whole tree on both nodes
would read twice what the group needs and double the run's NAS traffic. Whether
the halves are clean enough for that to matter is not measured.

**What was not ported.** The NAS stages 25 checkpoints; 14 are covered, the eleven
ported here plus the three that already had entries. The remaining eleven fall into
two classes. Each architecture below was read from the checkpoint's own
`config.json` on `na131t-ppu810e-test001`, and each claim of a missing registration
is that the name appears nowhere under `python/sglang/`.

- **Four architectures this tree does not register**, over five checkpoints:
  `Glm5NextForConditionalGeneration` (GLM-5.3-Flash),
  `KimiK3ForConditionalGeneration` (Kimi-K3),
  `MiniMaxM3SparseForConditionalGeneration` (both MiniMax-M3 formats), and
  `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next). A port would fail at
  load rather than produce a verdict. Registering an architecture is its own change
  with its own evidence, not part of a config port — as the `Qwen3_5MoeForCausalLM`
  work in the four-node line shows.
- **DeepSeek-V4, five checkpoints.** All five are `DeepseekV4ForCausalLM`, which
  this tree does register, but none carries a chat template: no `chat_template*`
  file and no `chat_template` key in `tokenizer_config.json`. Every request in this
  corpus is a chat completion. A template is a decision about how the model is
  prompted, which belongs with whoever owns the checkpoint. Kimi-K3 is in the same
  position on top of its missing registration.

`minimax2.7-w8a8-int8`, in the table above, was at one point in a third class here,
and the reason was wrong. That checkpoint was first reported as an incomplete copy
missing shards 125 through 130. It is not: its shards are numbered `00000` to
`00124` against a filename suffix of `-of-00130`, and it is the suffix that is
stale. The index declares exactly the 125 files that are present, none absent and
none undeclared, and the tree holds 214.6 GiB of tensors — the same volume as the
complete FP8-Channelwise copy, which is the expected result for two 8-bit
quantisations of one model. With the stated reason for skipping it withdrawn, it
is an entry.

## Relation to the internal test cases

All three suites are ports of internal `llm_infer_sglang_evalscope` answer cases
(`qwen3.5-397b-a17b-w8a8-int8_3001`, `qwen3.8-27b-bf16_3001`, and the btv1.5
`answer_4nodes/qwen3.8-2.4t-a95b-fp8_3001`). The port keeps
the checkpoint, device count, attention backend, memory fraction, and
quantization intent, and deliberately departs in two places:

- generation parameters are pinned to the deterministic house line (temperature
  0, top-p 1) instead of the sampling values the internal cases carry, because a
  rule-based verdict must be reproducible;
- serving parameters that the reviewed schema does not model — `page_size`,
  `stream_interval`, and `cuda_graph_max_bs_decode` — are
  left at SGLang defaults. `validate_test_config` accepts exactly the reviewed
  parameter set, so adding one of them is a schema change with its own review
  rather than a silent config edit.

The btv1.5 plan for the 144GiB board carries `qwen3.8-27b-fp8_3001` where this
repository keeps BF16, and no BF16 entry of its own. The port stays on BF16
deliberately: the two `-144g` configs are the `-96g` ones with a different board,
and nothing else, so a difference in verdicts has one candidate explanation. An
FP8 entry is a separate case with its own baseline, not a substitution.

The four-node case departs in one more place, which is not a choice: its internal
plan reaches the group through the cluster's own launcher, and this repository
reaches it through `ppu-distributed-action`, so the rank and the rendezvous arrive
by different means. `served_model_name` is `Qwen3.8-2.4T-A95B` while
`checkpoint_name` is `Qwen3.8-2.4T-A95B-FP8`, which is the internal case's own
spelling: the served name is what a client asks for and the checkpoint name is
what was loaded, and the reviewed schema keeps them separate for exactly this.

## Parity with the internal launcher

The internal cases are executed by `model_mate`, whose `SGLangServerCmd`
(`utils/commands/ServerCmd.py`, branch `sglang/h20_golden`) composes the launch.
The command it builds and the one `popen_launch_server` builds were compared
flag by flag; what follows is what that established.

**Identical.** Both invoke `sglang serve` rather than
`python3 -m sglang.launch_server`. Both reach a group through
`--dist-init-addr host:port`, `--nnodes`, and `--node-rank`, and both emit those
three only when the node count exceeds one, so a single-node launch is given no
rendezvous on either side. Both derive `--tp-size` from the case's `tp`, with the
one deliberate exception recorded above: the 2.4T FP8 entry splits the group as
`--tp-size 8 --pp-size 4`, because 32-way tensor parallelism cannot shard that
checkpoint's FP8 blocks at all.

**Quantization, which was a defect here.** `model_mate` does not derive the flag
from the case's `data_type` — that field only names the log directory — so a case
states a format or does not. Across the 51 internal btv1.5 sglang evalscope and
answer cases the line is sharp: only `data_type: w8a8-int8` carries
`quantization: w8a8_int8`, while `fp8`, `fp8-channel`, `mxfp4-fp8`, `awq`,
`gptq-int4`, `gptq-int8`, and `bf16` all leave it unset. The reason is visible in
`server_args.py`, where `fp8` is annotated *MOE + linear online quantization*:
naming it asks for unquantised weights to be quantised at load time, which
overrides what an already-quantised checkpoint declares in its own
`config.json`. The 2.4T config therefore states `null`, and
`build_answer_server_args` omits a null rather than rendering it. `unquant` on
the two BF16 configs is not the same case: `server_args` maps it to `None` on
arrival, and the flag it sets alongside is read only under `is_sm100_supported()`,
so on PPU it is exactly the internal cases' silence, spelled explicitly.

**`watchdog_timeout` 600** is now part of the reviewed schema and matches the
internal answer cases, which set it almost uniformly.

**`reasoning_parser`, a deliberate departure.** The internal Qwen answer cases do
not pass it; these configs pass one on all five. The requests here ask for
`separate_reasoning`, and the parser is what keeps a reasoning block out of the
graded text; the two entries with measured baselines were produced with it. The
internal cases that do pass one are the models whose templates need a different
parser (`glm45`, `deepseek-v4`, `minimax-append-think`).

Four configs pass `qwen3` and ask their template for `enable_thinking: false`.
The 2.4T entry cannot: its checkpoint ships a template that answers
`raise_exception('Disabling thinking is not supported.')` to exactly that
argument, which is why every request in run 33841139864 came back `400 Bad
Request` while `/generate` was answering 200. That template grades the pass
instead of switching it off — `reasoning_effort`, one of `xhigh` (its default),
`medium`, or `low`, rejected by name otherwise — so the config states `low`, and
`max_tokens` rises to 8192 with a 600s request timeout because a reasoning pass
that cannot be disabled still has to fit. The parser becomes `qwen3-thinking`,
which is the same `Qwen3Detector` with `force_reasoning` set: the template emits
`<|im_start|>assistant\n<think>\n` itself, so `<think>` arrives in the prompt and
the completion opens mid-reasoning with only `</think>` to come. `qwen3` decides
it is looking at reasoning by finding `<think>` in the text, so on this
checkpoint it would hand the whole reasoning pass to the grader as content.
Both findings were reproduced against the checkpoint's own template under
jinja2 3.1.2, in the image the pods run.

**Bind address, unchanged.** `model_mate` binds `--host 0.0.0.0`; here the host
comes from `DEFAULT_URL_FOR_TEST`, which is the loopback. Nothing off-node
reaches the HTTP API: the only client runs in rank 0's own pod, and the workers
coordinate through the rendezvous and the NAS, never over HTTP.

**Group environment variables.** `SGLangServerCmd._config_env` exports
`MASTER_ADDR`, `NNODES`, and `RANK` around the launch, and `_server_environment`
now does the same for a multi-node launch — for parity, not for a consumer this
repository can point at. On the path these configs take SGLang reads the
rendezvous from `--dist-init-addr` alone: `MASTER_PORT` matters only behind an
`env://` init-method override, and `MASTER_ADDR` only behind a `nixl` a2a
backend. What the PPU runtime below reads is not visible here, so the group
states itself the way the framework whose runs are the baseline states it.
`MASTER_ADDR` is taken from the rendezvous actually in force rather than copied
from the injected variable of the same name, since an override exists precisely
when the injected one does not resolve.

One mechanism has no analogue and needs none: `model_mate` skips any parameter
whose value is falsy, which is why an internal case's `ep: 0` produces no flag.
The reviewed schema lists its parameters exhaustively instead, so there is no
value that silently disappears.

## Measured baseline (ZW810E)

The first on-machine run was executed on `ptg-ppu-02` (16 × PPU-ZW810E, 96GiB
per device, driver 1.6.1, SDK 2.1.1) on 2026-09-01 using eight devices, and it
establishes the reference cost of the 397B test:

| Phase | Cost | Observation |
| --- | --- | --- |
| Checkpoint page-cache warm | 26m17s | ~250MB/s across eight parallel readers |
| Weight load | 66s | 47.35 GB per device, 47.64 GB free afterwards |
| CUDA graph capture | ~3m | 18.66 GB free afterwards |
| Ten candidate generations | ~4m | 98.6 decode tokens/s |
| Test body total | 7m31s | well inside the 280-minute step budget |

Two properties of this host are load-bearing and are the reason the warm step
carries the comment it does: the checkpoint is only reachable over NFS at about
57MB/s per stream, and the page cache is reclaimed back to its baseline within
30 minutes regardless of free memory. Neither is a property of the test, and a
run that skips or defers the warm will exhaust `startup_timeout_seconds`
instead of producing a verdict.

During capture each of the eight TP ranks emits one
`Scheduler watchdog timeout (soft=True)` record with a py-spy dump. The
watchdog is soft, no process is killed, and the run proceeds normally; the only
effect is several hundred extra log lines.

The run reported nine of ten cases passing. `deepseek-letter-count` answered
`3` where the reviewed fact is `4`, with `finish_reason=stop` and complete
usage accounting, so the failure is a model-capability result rather than a
serving or evaluation defect. `fact_rule_failed` is classified `hard_fail`
by the kit, so the case's per-case verdict is `failed`. Whether that reddens
the *run* is now a separate decision: declared in this model's
`evaluation.baseline.known_failures` as a known model-capability miss, the
case is tolerated and the run stays green until a genuinely new failure
appears. See "Regression gate and per-model baseline" below.

The 27B entry was measured on the same host on 2026-09-02 using one device:

| Phase | Cost | Observation |
| --- | --- | --- |
| Checkpoint page-cache warm | 3m34s | 51.7GiB in 18 shards, ~247MB/s in parallel |
| Weight load | 11.4s | 51.05 GB used, 44.93 GB free afterwards |
| CUDA graph capture | 87.5s | batch sizes up to 33, 13.96 GB free afterwards |
| Server startup, launch to ready | 2m43s | against the 1800s `startup_timeout_seconds` |
| Ten candidate generations | ~27s | ~37 decode tokens/s |
| Test body total | 3m10s | reported elapsed 199s against `est_time` 1200 |
| Whole job | 8m41s | against the 60-minute job budget |

The generous budgets are kept deliberately: they cover a cold checkpoint. At the
measured single-stream 57MB/s, 51.7GiB needs about 15m30s, which together with
capture still fits inside `startup_timeout_seconds`, so this entry produces a
verdict even if the warm step is skipped — unlike the 397B entry, for which the
warm is a hard dependency. `max_total_num_tokens` came out at 264724 with
`context_len` 262144, so `mem_fraction_static` 0.85 leaves the KV pool ample
room on a 96GiB device.

That run reported seven of ten cases passing, all three failures being
`fact_rule_failed` on objective cases:

| Case | Answer | Reviewed fact |
| --- | --- | --- |
| `deepseek-letter-count` | `2` | 4 |
| `henan-bordering-provinces` | includes 湖南, omits 河北 | the six bordering provinces |
| `red-ball-probability` | `12.5` | 1/7, about 14.29% |

The first two are model-capability results: the request path is clean and the
rules are correct — the second even reproduces the error that the reviewed rule
was written to correct in the internal golden. The third is a wrong answer as
well, but it also exposed a gap in the `probability` rule, which accepted `a/b`,
`X%`, or a bare number already in `[0, 1]`. The prompt asks for a percentage, so
a bare `14.29` was discarded exactly as `12.5` was, and no correctly formed
bare-percentage answer could pass at all. A case now declares how a bare number
may be read, and this one admits either reading:

```json
{"type": "probability", "target": 0.142857142857, "tolerance": 0.0006,
 "bare_number_units": ["percent", "probability"], "description": "..."}
```

The declaration is the only way in: the default stays the probability reading
alone and an unknown unit fails the run, because inferring `percent` from a value
above 1 would admit a rounded `14` against a target of 1/7 under a loose enough
tolerance. The two readings of one bare number are alternatives for a single
claim, so the matching one is kept — otherwise `答案是0.1429` would contradict
itself, its percentage reading being both asserted and wrong. Neither the
tolerance nor the severity classification is changed, so `12.5` and a rounded
`14` still fail and the verdict for the run above stands. Changing what a rule
accepts changes the judging standard, so `dataset/answer_cases_zh_v1.json` is now
`revision` 2: an annotation keyed on revision 1 must not be read as though it had
been produced under the current standard.

## Regression gate and per-model baseline

The per-case verdict and the run's red/green are two separate judgements. A
case is still `failed` the moment any `hard_fail` finding lands — the rules
are unchanged and the golden facts stay strict. What is separate is what turns
a *run* red: a run regresses only when a **new** problem appears, not because
the checkpoint keeps missing a fact it has always missed. Two mechanisms carry
this.

**Finding severity.** Every finding carries a `severity`. Engine-internal
checks — a bad `finish_reason`, garbled or repeated output, cross-case
duplication, request errors — are `critical`: they mean the serving stack or
the transcript is broken, and they always red the run. A fact-rule finding is
`model` by default (the checkpoint simply got a fact wrong); a dataset rule may
set `"severity": "critical"` when its miss would instead signal a broken
contract that must always gate. The per-case verdict ignores severity; only the
run-level gate reads it.

**Per-model baseline.** A model config may declare `evaluation.baseline`:

```json
"baseline": {
  "min_score": 7,
  "known_failures": ["deepseek-letter-count", "henan-bordering-provinces", "red-ball-probability"],
  "note": "..."
}
```

`known_failures` are the case ids this checkpoint is known to miss for
model-capability reasons; `min_score` is the floor of passing cases. The run
regresses (reddens) when any of the following holds: a failure carries a
`critical` finding; a `model` failure lands on a case **not** in
`known_failures` (a new miss); or the passing-case count drops below
`min_score`. A case in `known_failures` that now passes is reported as an
`unexpected_pass` — green, but a signal to tighten the baseline. The gate lands
in `summary.gate` of the result JSON; the suite assertion reads
`summary.gate.regressed`, and its failure message names exactly the cases that
regressed so a red run points straight at the new problem.

With no baseline the allowlist is empty and the floor is unset, so any miss
reddens — identical to the earlier strict behaviour, which is why an unseeded
config is unaffected.

### Seeding a baseline

A baseline is seeded only from a run verified correct. To seed a model: run its
suite once, confirm each miss is a genuine model-capability result (clean
request path, `finish_reason=stop`, rule correct) rather than an infrastructure
failure, then record those case ids in `known_failures` and set `min_score` to
the passing count. Never add a case to `known_failures` to silence a `critical`
finding — severity keeps that impossible by construction, and doing so would
hide a real regression.

Fifteen of the sixteen configs carry a baseline. Fourteen are seeded from the
temperature-0 verdicts of the v0.5.13 Answer PR-stack. The twelve M890P entries
take their `known_failures` and `min_score` from the reference grading — run
`34100574370` for the ten single-board `-144g` entries, run `34944181107` for
the two-node `qwen3.8/2.4t-a95b-mxfp4-fp8-144g`, and run `33849322347` for the
four-node `qwen3.8/2.4t-a95b-fp8-144g` — where every miss is a `fact_rule_failed`
on a shared-corpus golden case (`deepseek-letter-count` in eight entries,
`red-ball-probability` and `henan-bordering-provinces` on the weaker
checkpoints) with `finish_reason=stop` and a clean request path. The two retired
ZW810E (`-96g`) configs take their baselines from the on-machine ptg-ppu-02
measured baseline above; they are not validated by the M890P probe.

The fifteenth baseline, `minimax2.7/mxfp4-fp8-144g.json`, is seeded not from
v0.5.13 but from the v0.5.18 M890P probe run `35315230192` (9/10 at
temperature 0), because v0.5.13 never produced a valid verdict for it — the
report builder crashed on the empty-answer invariant before grading. Its one
miss, `henan-bordering-provinces`, is a model-severity `fact_rule_failed` with
`finish_reason=stop` and a clean request path, the same case already accepted as
a known model-capability miss on `qwen3.8/27b-bf16-144g`. See the cross-version
comparison below.

One baseline has moved since that seeding. `glm5.2/mxfp4-fp8-144g.json` drops
from a strict 10 to `min_score` 9 with `red-ball-probability` allowlisted: on
the current stack the checkpoint misses that probability probe in the
non-thinking direct-answer mode, a model-severity `fact_rule_failed` rather
than a serving fault.

The remaining config stays unseeded and therefore strict, because no run has
yet produced a clean, stable model verdict for it.
`kimi2.6/w8a8-int8-144g-2n.json` is not stable across v0.5.18 runs: it graded
10/10 on run `34961981320` but 6/10 on the probe run `35315230192`, where
`spring-dawn-poem` exhausted its whole 8192-token budget in reasoning and
returned an empty answer at `finish_reason=length` — a `critical` finding, never
eligible for `known_failures`. It earns its first baseline only from a clean,
repeatable M890P probe, not by silencing a critical. The other formerly unseeded
config, `kimi2.6/w4a8-int8-144g.json`, reddened on `server_start_failed` through
the open ACEXT W4A8 kernel gap — infrastructure that was never eligible for
`known_failures` — and is removed with the entry itself.

## v0.5.18 cross-version regression comparison

The v0.5.18 branch was probed in full on 2026-09-18 as run
[35315230192](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/35315230192),
source revision `83dbd4295b341b6892e7fc8db652503db814f06f`, fifteen M890P
entries on the `sglang0.5.13-py312` base image, one shared dataset digest
(`891275f6`). Each entry was compared case-for-case against its v0.5.13 Answer
PR-stack history (single-board run `34100574370`, two-node `34132812028`,
four-node `33849322347`). The comparison judges on verdict, reason_code, and the
failing-case set rather than byte-identical answers: greedy decoding at
temperature 0 is not byte-reproducible across kernels or SDK versions, so an
objective answer whose text differs but still grades correct is a benign format
difference, not a regression.

Eleven entries had comparable v0.5.13 history:

| Entry | v0.5.13 | v0.5.18 | Classification |
| --- | --- | --- | --- |
| `qwen3.8-27b-bf16` | 7/10 | 7/10 | consistent |
| `qwen3.5-397b-fp8-channelwise` | 9/10 | 9/10 | consistent |
| `qwen3.5-397b-w8a8-int8` | 9/10 | 9/10 | benign byte diff |
| `qwen3.5-397b-mxfp4-fp8` | 9/10 | 9/10 | benign byte diff |
| `glm5.2-fp8-channelwise` | 8/10 | 8/10 | benign byte diff |
| `glm5.2-mxfp4-fp8` | 10/10 | 10/10 | benign byte diff |
| `minimax2.7-w8a8-int8` | 10/10 | 10/10 | benign byte diff |
| `kimi2.6-mxfp4-fp8` | 9/10 | 9/10 | benign byte diff |
| `32` (four-node) | 10/10 | 10/10 | benign byte diff |
| `glm5.2-w8a8-int8` | 8/10 | **0/10** | **real regression** |
| `minimax2.7-fp8-channelwise` | 10/10 | **9/10** | **real regression** |

Nine of the eleven are regression-consistent: the same cases pass, the same
cases miss, and every miss keeps its v0.5.13 reason_code and severity. The seven
marked "benign byte diff" differ only in the byte form of one or more objective
answers that still grade correct — for example `glm5.2-mxfp4-fp8`, where six
objective answers changed text but all ten still passed. Their baselines are
unchanged.

Two entries are real regressions and stay red; neither baseline is loosened to
hide them:

- `glm5.2-w8a8-int8` fell from 8/10 to **0/10**. All ten cases now degenerate:
  `finish_reason=length` with `periodic_fragment_repeat` and
  `repeated_4gram_coverage` (`critical`) on every case. The gate reads
  `regressed=true`, `score_below_baseline=true` against its `min_score=8` — the
  baseline catches the regression exactly as intended.
- `minimax2.7-fp8-channelwise` fell from 10/10 to **9/10**. The open-ended
  `xian-three-day-trip` newly fails with a `fact_rule_failed`
  (`finish_reason=stop`, model severity). The gate reads `regressed=true`,
  `new_regressions=[xian-three-day-trip]` against its `min_score=10`.

The remaining four probe entries had no comparable v0.5.13 verdict in this set:

- `16-qwen3.8-2.4t-a95b-mxfp4-fp8` graded 10/10, matching its existing
  `min_score=10` baseline. That baseline was seeded from the two-node run
  `34944181107`, which is not among the three history runs downloaded for this
  comparison, so the 10/10 here simply satisfies the gate rather than being
  re-diffed against an artifact.
- `minimax2.7-mxfp4-fp8` graded 9/10 with one model-severity miss and no
  critical, and is seeded from this run (above).
- `kimi2.6-w4a8-int8` reddened on `server_start_failed` (ACEXT W4A8 kernel gap)
  and `kimi2.6-w8a8-int8-2n` carried a `critical` token-exhaustion finding; both
  were unseeded at the time, and the w4a8 entry has since been removed.

The two retired ZW810E (`-96g`) configs are not part of this fifteen-entry M890P
probe; they keep their own measured baselines and are excluded from the probe's
coverage claim.

## Measured baseline (ZW-M890P)

Both entries were run on the btv1.5 cluster on 2026-09-02 (run `33645062958`,
eight ZW-M890P devices of 144GiB, torch 2.11.0). The verdicts are the ZW810E ones,
case for case:

| Suite | Passed | Failing cases | Job | Test body |
| --- | --- | --- | --- | --- |
| `nightly-answer-1-ppu` (27B, 1 device) | 7/10 | `deepseek-letter-count`, `henan-bordering-provinces`, `red-ball-probability` | 19m06s | 399.8s |
| `nightly-answer-8-ppu` (397B, 8 devices) | 9/10 | `deepseek-letter-count` | 26m39s | 770.6s |

The failing answers are the ZW810E ones character for character: `2`, the same
Henan sentence that includes 湖南 and omits 河北, and `12.5` from the 27B; `3` from
the 397B. At temperature 0 that is the expected result, and it is what makes the
board a controlled variable — the `-144g` configs differ from the `-96g` ones only
in `hardware` and, for the 397B, in the checkpoint's parent directory, and no
verdict moves.

Where the board does show is in capacity and in the cost of using it:

| Phase | 27B, 1 device | 397B, 8 devices | ZW810E counterpart |
| --- | --- | --- | --- |
| Free device memory before load | 143.98 GB | 143.07 GB per rank | 96GiB board |
| Weight load | 3.8s, 51.05 GB | 21.9–24.5s, 47.35 GB per rank | 11.4s / 66s |
| CUDA graph capture | 260.8s, batch sizes to 78 | 621.7s | 87.5s (to 33) / ~3m |
| Launch to ready | 6m08s | 12m23s | 2m43s / — |
| `max_total_num_tokens` | 616103 | 2467089 | 264724 / — |
| Free memory after capture | 20.88 GB | 26.33 GB | 13.96 GB / 18.66 GB |

Capture is what grew. `mem_fraction_static` is a fraction, so the same 0.85 on a
144GiB device yields a KV pool 2.3× the 96GiB one; the captured batch-size list is
capped by `max_running_requests`, which the pool raises from 33 to 78 for the 27B,
and every additional batch size is another graph to record. Both entries still
reach ready far inside the 1800s `startup_timeout_seconds` and finish well inside
their step budgets. The 397B also reproduces the soft `Scheduler watchdog timeout`
record on each of the eight ranks during capture that the ZW810E line reports, with
the same absence of consequence.

Dependency install and the page-cache warm cannot be separated in these logs — the
pod's stdout reaches the runner in batches, so both land on a single flush
timestamp — but together they account for roughly 12 minutes for the 27B and 13.5
for the 397B. The install dominates: a worker pod starts from the base image and
re-resolves the wheels on every run, where the bare-metal host reuses what is
already installed. That is a property of the K8s path rather than of the test, and
it is why this line's pod budgets are 120 and 330 minutes.

One defect surfaced in that run and is fixed in the workflow as it now stands: the
evidence step also tried to delete the NAS copy after taking it, which fails with
`EPERM` on every file and turned the step red on both entries even though the copy
and the upload had succeeded. The step now only reads.

The fix was re-run on the same cluster (run `33690732486`, both entries against
`2f48d78`): `Collect Answer evidence` and `Upload Answer evidence` are green on
both, each artifact carries all five files, and the verdicts and failing cases are
unchanged — 7/10 and 9/10, same case ids. The remaining red step is the suite's own
assertion, which is the intended signal. That run also shows what the warm buys:
with the checkpoints already in the node's page cache from the run above, the jobs
took 10m56s and 18m15s instead of 19m06s and 26m39s. The table above keeps the cold
numbers, since a nightly on an otherwise idle board is the cold case.

## The twelve-entry eight-wide line

The twelve ported entries run as one eight-wide dispatch. The reference grading is
[run 34100574370](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/34100574370),
2026-09-07, at `265d498`, which is the first run to carry all three of the
robustness fixes below. Eleven of the twelve reached a verdict; the results are:

| Entry | Passed | Failing / error | Note |
| --- | --- | --- | --- |
| `glm5.2-mxfp4-fp8` | 10/10 | — | |
| `minimax2.7-w8a8-int8` | 10/10 | — | |
| `minimax2.7-fp8-channelwise` | 10/10 | — | checkout rescued, then clean |
| `qwen3.5-397b-w8a8-int8` | 9/10 | `deepseek-letter-count` | |
| `qwen3.5-397b-mxfp4-fp8` | 9/10 | `deepseek-letter-count` | rescued after 3× GnuTLS |
| `qwen3.5-397b-fp8-channelwise` | 9/10 | `deepseek-letter-count` | rescued after 4× GnuTLS |
| `kimi2.6-mxfp4-fp8` | 9/10 | `deepseek-letter-count` | |
| `glm5.2-fp8-channelwise` | 8/10 | `deepseek-letter-count`, `red-ball-probability` | |
| `glm5.2-w8a8-int8` | 8/10 | `deepseek-letter-count`, `red-ball-probability` | |
| `qwen3.8-27b-bf16` | 7/10 | `deepseek-letter-count`, `henan-bordering-provinces`, `red-ball-probability` | |
| `kimi2.6-w4a8-int8` | error | — | ACEXT W4A8 kernel gap, below |
| `minimax2.7-mxfp4-fp8` | error | — | empty-answer report crash, below |

The ten graded entries are stable against the eight-wide run before this one,
[run 34091089985](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/34091089985)
at `17b5aeb`: every case id that failed there fails here and no other, and the
27B entry is verdict-for-verdict identical across its third, fourth, and fifth
graded runs. `deepseek-letter-count` is the failing case in eight of the eleven
reports — it asks for a letter count the models get wrong at temperature 0 — and
this concentration is a property of the question, not of any one checkpoint.

Three fixes land here and each is confirmed on this run:

- **`dist_timeout` (`63370a8983`, fifteen configs).** Every entry that carries it
  reaches ready and grades, none regressed; the value is the btv1.5 `server_cmds`
  one, passed on every launch line there.
- **`SGLANG_WARMUP_TIMEOUT=3600` (`5adf87868b`, seventeen configs).** Warmup
  passes on every entry; no entry stalled at the warmup request.
- **Second checkout attempt (`265d498b75`, twelve jobs, four workflows).** Four
  entries hit the github.com egress fault this run — `qwen3.5-397b-fp8-channelwise`
  four times, `qwen3.5-397b-mxfp4-fp8` three, `glm5.2-fp8-channelwise` and
  `kimi2.6-w4a8-int8` once each — and none died at checkout, where three of the
  previous run's entries did. The two heaviest cases exhausted the three attempts
  `actions/checkout` makes on its own and were carried by the second checkout;
  both then graded 9/10. The lighter two were absorbed by the built-in retry. The
  mechanism is now confirmed on real traffic, not only by construction.

Two entries error rather than grade, and both are known:

- `kimi2.6-w4a8-int8` is the open W4A8 kernel gap: it clears checkout and weight
  load and then dies in `Capture cuda graph failed` with an ACEXT `invalid
  argument` on `compute_occupancy.h`. This is a platform kernel gap, tracked in
  the gap section above, not a defect in the port.
- `minimax2.7-mxfp4-fp8` errors on a report-builder invariant, not on serving.
  All ten cases returned `200 OK`; with `separate_reasoning` on and the `minimax`
  reasoning parser, one case's whole output was classed as reasoning and its
  answer `content` came back empty. An empty final answer is already a hard-fail
  finding (`empty_final_answer`), so the run summary would be 9/10 — but
  `build_label_candidates` then builds a label record whose `candidate_answer` is
  that empty string, which violates the annotation invariant that exactly one
  candidate-answer field be non-empty, and the whole report build raises instead.
  `3c46b11d9c` skips an empty final answer when selecting label candidates, the
  same way infrastructure failures are already skipped there, so this entry reports
  9/10 rather than raising. It was a kit robustness gap unrelated to the three fixes
  above, none of which touch response parsing or candidate selection, and it is not
  yet confirmed on a run of this entry.

## The two-node line — measured

The first two-node run to reach a verdict is
[run 34132812028](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/34132812028),
2026-09-07, with the pods at `61dd1838`. One of the two entries graded:

| Entry | Passed | Failing / error | Note |
| --- | --- | --- | --- |
| `qwen3.8-2.4t-a95b-mxfp4-fp8` | 10/10 | — | one suspect: `henan-bordering-provinces`, 4-gram coverage 0.55 |
| `kimi2.6-w8a8-int8` | error | — | all ten `server_start_failed`; PP defect below |

That 10/10 is what the gloo derivation was for: on the run before it every case of
this entry was `server_start_failed` six minutes in, and here both nodes reached
ready and graded. The suite itself took 1903.5 s and the job 47m53s, against an
`est_time` of 7200 for this suite — the first measured figure that could replace it.

`kimi2.6-w8a8-int8` failed before serving, and its cause is not in the traceback it
printed. All sixteen schedulers — `PP0 TP0` through `PP0 TP7` on rank 0 and
`PP1 TP0` through `PP1 TP7` on rank 1 — raised `RecursionError: maximum recursion
depth exceeded` out of torch's `named_parameters`, both node status files recorded
255, and all ten cases came back `server_start_failed`. What actually happened is a
PP defect hidden by a `LazyValue` one, fixed together in `7559dea240`:

- `DeepseekV2ForCausalLM`, which is this checkpoint's language model, built its
  `routed_experts_weights_of_layer` by walking `enumerate(self.model.layers)` and
  reading `layer.mlp`. Under `pp_size > 1` `make_layers` pads that list with
  `PPMissingLayer` for every layer another stage holds, and the placeholder has no
  `mlp`, so the walk raises `AttributeError`. Every single-board entry runs at
  `pp_size` 1 and never sees a placeholder, which is why this had not shown before
  a two-node entry graded. It now walks
  `range(self.model.start_layer, self.model.end_layer)`, the bounds `qwen3_5` and
  `gpt_oss` already use. `GlmMoeDsaForCausalLM` subclasses this class and is
  carried by the same fix.
- `LazyValue` is what turned that `AttributeError` into the `RecursionError`.
  `__getattr__` forwarded to `self.value`, and Python falls back to `__getattr__`
  not only for a name an object lacks but also whenever a property getter raises
  `AttributeError` — so the creator's error arrived back at `__getattr__` under the
  name `value`, which read `self.value` again, which ran the creator again, since
  it is only retired on success. 981 rounds of that exhausted the stack, and what
  got reported was wherever the last round happened to stand. Names the class owns
  now stop at `__getattr__`, and a creator's `AttributeError` is re-raised as
  `RuntimeError` with the original attached: a bare `AttributeError` leaving the
  getter is indistinguishable from `value` being absent, and the caller reads the
  property as `getattr(model, name, None)`, where it would be swallowed into a
  silent `None` — a model that failed to build its expert list would present as a
  model with no experts. `TestLazyValue` in
  `test/registered/unit/utils/test_common.py` locks both halves.

Three other creators in this tree have the same `enumerate` shape and would fail
the same way under `pp_size > 1`: `mimo_v2.py`, `glm4_moe_lite.py`, and
`qwen3_next.py`. None of the three is an entry of any suite here, so none can be
verified on this fleet, and they are recorded rather than changed.

### v0.5.18 verification

Both two-node entries reached a clean verdict on 2026-09-15 after the v0.5.18
port fixes were applied:

| Entry | Run | Source revision | Cases | Suite / job duration |
| --- | --- | --- | --- | --- |
| `qwen3.8-2.4t-a95b-mxfp4-fp8` | [34944181107](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/34944181107) | `7b0821c36c37b4ad30a1e8f1df4126ed2405d0ae` | 10/10 | 1801 / 2172 s |
| `kimi2.6-w8a8-int8` | [34961981320](https://github.com/flytiger-eco/sglang-for-sail/actions/runs/34961981320) | `5094d105da74282bf76d921fd1d6ad5771f1fafd` | 10/10 | 1527 / 1690 s |

The A95B run verifies the `LazyValue` hardening and the missing
`PPMissingLayer` guard in `qwen3_5_text.py`; the workflow body also carries the
PCCL MNNVL opt-out required for two-node initialization on this cluster. The
Kimi run verifies that `thinking: true` produces balanced reasoning blocks and
that an 8192-token completion budget was sufficient on that run. With 2048
tokens, eight of ten cases ended at `finish_reason=length` before a final
answer; with thinking disabled, the parser received an unmatched closing tag and
reasoning leaked into graded text. The budget is not sufficient on every run,
however: on the later probe run `35315230192` this entry dropped to 6/10 and
`spring-dawn-poem` consumed the full 8192 tokens in reasoning, ending at
`finish_reason=length` with an empty answer. Reasoning-token consumption on this
checkpoint varies run to run — `red-ball-probability` alone took 4789 reasoning
tokens on the probe — so the config stays unseeded pending a stable budget.

These runs satisfy the workflow's original requirement for measured green
results. `test-ppu-answer-16.yml` remains dispatch-only until a
repository-default-branch caller is added; placing a `schedule` trigger only on
the v0.5.18 branch would not run because GitHub evaluates schedules exclusively
from the default branch. The current timeout budgets remain conservative until
repeated cold-cache runs establish a safe lower bound.

**A lost evidence upload no longer reports a passing suite as failed.** Both jobs
of that run ended `failure`, and for the 10/10 entry the only non-green step was
the artifact upload: `CreateArtifact` was reset (`ECONNRESET`) four seconds after
the step announced thirteen files, with no retry of its own. The verdict was never
at risk — it leaves the collect step over annotations, and rank 0's log carries the
per-case table — but the run page said failed. `7559dea240` marks the upload
`continue-on-error`, waits 30 seconds, and tries once more with `overwrite: true`,
which v4 needs because the first attempt can be reset after it has already created
the artifact. Fourteen jobs across the three Answer workflows carry it. What a lost
upload still costs is the machine-readable copy — `result.json`, `junit.xml`, and
the logs of every rank but rank 0 — which exists nowhere else.

## Results and annotations

The workflow uploads `result.json` (rule findings and provenance), `summary.md`,
`junit.xml`, and — because `SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS` is set to
`1` — `result.raw.json` and `label_candidates.jsonl`, which carry the candidate
answers verbatim. Publishing the candidates is deliberate: the prompts are
public general knowledge that already lives in this directory, the answers are
this project's own model output, and a red nightly is otherwise not diagnosable
without occupying eight devices for a second run. `result.json` stays redacted
so the schema-stable report keeps one shape whether or not raw collection is
enabled.

The same switch also prints the candidates into the job log, so a failure can be
read without downloading anything: after the summary table the test emits one
block per case with the prompt, the answer, and the observed value that tripped
each rule. `ci_utils.run_unittest_files` runs the test file as a plain
subprocess with an inherited stdout, so the block reaches the log whether the
run ends green or red. Because both the artifact and the log carry candidate
text, set the switch back to `0` for any dataset whose prompts or answers cannot
be published — one switch covers both surfaces.

One step further out, the run page itself names the failures. `summary.md` opens
with a `### Failing cases` list — one line per failing case carrying the rule
sentence the dataset declares, for example `deepseek-letter-count` against
`答案包含数字 4` — and both workflows turn each of those lines into a GitHub
annotation, plus one notice with the pass count. Annotations were chosen over the
step summary as the primary channel because **a step summary is not always
collected**: the K8s runner drives its job through a container hook that does not
share the job container's filesystem with the runner process, so bytes written to
`GITHUB_STEP_SUMMARY` there are dropped without an error — the check runs of the
first two runs on that board reported a summary of length zero while their
annotations arrived intact. Annotations travel over the step's stdout, which
reaches the runner on both paths. The step summary is still written, since it is
the richer surface where it works.

The two workflows read that list back with a `grep` for the bullet prefix, because
the orchestration container is not guaranteed a JSON parser — no `python3`, no
`jq`. That prefix appears on no other line of the document, and
`test_summary_names_the_failing_cases_for_annotations` locks the shape on the
evaluator's side, which is where a bash snippet cannot.

Both raw files are written with escaped non-ASCII (`\uXXXX`) so that an unpaired
surrogate in a candidate answer cannot fail the write. Read them with a JSON
tool, which decodes the escapes: `jq '.cases[] | select(.case_id=="...")'`, or
`jq -r '.cases[] | "[\(.verdict)] \(.case_id)\n  \(.final_answer)\n"'` for every
answer at once. The log block, in contrast, is already plain text: only
characters that UTF-8 cannot encode stay escaped there.

In provenance, `source_revision` is the authoritative identifier of what ran: it
is the full commit SHA of the checkout, injected by the workflow.
`package_versions.sglang` reads `0.0.0`, which is a property of the CI install
path rather than a collection defect. `ppu_install_dependency.sh` swaps in
`pyproject_other.toml`, whose version is dynamic, and installs with
`--no-build-isolation`, so `setuptools-scm` is never present to supply one; had
it run and found no tag, the configured `fallback_version` would have produced
`0.0.0.dev0` instead. The checkout is `--no-tags --depth=2` in any case, so no
tag is reachable on the machine to describe against. Recover the version number
from the SHA in a full clone:

```bash
git checkout <source_revision>
python3 python/tools/get_version_tag.py   # e.g. 0.5.13+v0.1.0-121-g5c9c6bce54
```

Public data belongs here:

- prompts, reviewed reference facts, and quality profiles;
- synthetic mutations that are safe to disclose;
- explicitly released calibration snapshots.

Pending blind reviews, adjudication records, and unrevealed holdout samples
belong in a private annotation repository. Records should conform to
`dataset/annotation_record.schema.json`. GitHub artifacts are evidence for a run,
not the long-term annotation system of record, so promote a candidate into that
repository rather than relying on the 30-day artifact retention.
