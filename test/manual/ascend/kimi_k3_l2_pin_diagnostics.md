# Kimi K3 Nightly NPU pinned-memory diagnostics

This reference change collects the missing runtime evidence for
`RuntimeError: host buffer must use pinned memory` during Mamba L2 D2H backup.
The cause remains unconfirmed. The original exception is re-raised after
collection; the buffer, allocator, transfer mode, and test parameters are retained.

## Run the existing Nightly automation

Use the original four-node case:

```text
test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py
```

Its execution path is:

```text
nightly-test-npu.yml
  -> nightly-test-npu-e2e-multi-node.yml
  -> run_npu_e2e_test.py
  -> k8s_multi_pd_mix.yaml.jinja2
  -> run_npu_testcase.sh
  -> test_npu_kimi_k3_w4a8_32p_gpqa.py
```

From an account with upstream Actions permissions, run the existing workflow
against this PR's source. Replace `PR_NUMBER` with the reference PR number:

```bash
gh workflow run nightly-test-npu.yml --repo sgl-project/sglang --ref main \
  -f ref=refs/pull/PR_NUMBER/head \
  -f job_filter=nightly-poc-multi-node-mix-tests \
  -f accuracy_dataset=gsm8k \
  -f image_a3=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-a3
```

`job_filter` selects the existing mix group, including its other Kimi cases.
The workflow matrix is unchanged. For an isolated run of just the failing case,
use the same driver command in the Nightly runner environment, with the checked
out PR already copied to its run-scoped shared directory:

```bash
cd python/sglang/test/ascend/e2e
python3 -u run_npu_e2e_test.py \
  --env ci \
  --image swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-a3 \
  --sglang-source-relative-path tests/sglang-PIN_DIAG_RUN_ID \
  --metrics-data-file /root/.cache/tests/output/PIN_DIAG_RUN_ID/accuracy/test_npu_kimi_k3_w4a8_32p_gpqa \
  --test-case test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py \
  --kube-name-space sgl-project \
  --kube-job-type multi-pd-mix \
  --kube-job-name-prefix ascend-PIN_DIAG_RUN_ID \
  --run-id PIN_DIAG_RUN_ID \
  --accuracy-dataset gsm8k \
  --node-size 4 \
  --sglang-is-in-ci
```

Replace `PIN_DIAG_RUN_ID` consistently with the run's unique numeric ID. Keep the normal
Nightly checkout mode (`use_image_sglang` empty/false), so the server imports the
PR's mounted source. The image continues to provide torch, torch_npu, CANN, and
sgl-kernel-npu. Record the actual image identity rather than inferring it from
the mutable image tag.

The four-node case is outside `/rerun-test`'s supported NPU suites; run it through
the multi-node automation above.

## Evidence collected automatically

Before starting K3, `run_npu_testcase.sh` runs the metadata collector using its
already prepared interpreter and SDK environment. This subprocess imports only
the standard library; it does not initialize an NPU or prime torch_npu's caches.
It records package versions, build-tagged kernel wheel filename, version-file
Git IDs, CANN/driver version files, and library hashes even if startup later fails.

The Kubernetes driver also logs each running pod's `image_id`.

On the original pinned exception, the server records:

- Original host buffer data/storage pointers, storage offset, shape, strides,
  dtype, and byte count; source device tensor metadata.
- Matching entries in the existing ACL/MemFabric registration table.
- ACL pointer attributes for both host pointers and current device/context.
- Python and ATen `is_pinned`, dispatcher registrations, and thread-local keys.
- DRIVER/RUNTIME versions returned by torch_npu, loaded library paths/hashes,
  and the loaded Mamba source hash.
- The cached pointer-attribute compatibility decision for the verified
  torch_npu ELF. Reading this cache requires an exact SHA256 and instruction
  match. Other builds skip the binary-layout-specific read and retain the
  remaining evidence. No C++ functions are invoked through ctypes.

Search the job log for `NPU_PIN_DIAGNOSTIC`. JSON files are saved under:

```text
${METRICS_DATA_FILE}/npu_pin_diagnostics/
```

The existing Nightly artifact upload collects that directory. Filenames include
the pod hostname, process ID, and timestamp to distinguish workers.

## Interpret the failure record

Check the cached decision before interpreting version strings. A false decision
can route external ACL registrations through torch_npu's allocator-membership
fallback. If the decision is true, examine the storage pointer's ACL location
type and the registration's coverage. A difference between the storage and data
pointers can matter because the native pinned check inspects storage.

Compare Python/ATen results with ACL attributes and dispatcher/TLS state. A valid
ACL mapping does not by itself establish which check rejected the tensor.
Collection runs after the native exception returns to the Python boundary;
restored or inconsistent state may require follow-up instrumentation at the C++
check itself.

## Baseline investigation limits

Single-node A3 checks passed for an ACL-registered 3,337,224,192-byte host tensor
with shape `[123, 69, 1, 6, 128, 128]` (FP32), using an older native library and
a two-slot device pool. A separate 16-process test held 230,691,897,344 bytes of
registered host memory concurrently; all 80 buffers remained HOST/pinned and all
16 native D2H copies passed. These are negative reproduction results, rather
than evidence that the original Nightly issue is fixed. The test team still
needs to run the original multi-node case with these diagnostics.
