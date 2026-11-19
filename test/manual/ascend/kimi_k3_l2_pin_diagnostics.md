# Kimi K3 L2 pinned-memory container patch

This package collects evidence for the Mamba D2H "host buffer must use pinned
memory" error. Its root cause remains unconfirmed. The original exception,
allocator, transfer mode, and four-node test parameters are retained.

## Apply patch to container

Pass this zip URL to the test runner's **Apply patch to container** /
**container_patches** entry:

~~~text
https://raw.githubusercontent.com/hanwlax/sglang/codex/k3-hicache-pin-diagnostics/test/manual/ascend/kimi_k3_l2_pin_patch/k3-pin-diagnostics.zip
~~~

The archive has install.sh at its root. The existing multi-node template
downloads/extracts it and each Pod runs the installer before the original test.
This PR ships only the package and guide; runtime/launcher files are unchanged.

[Reviewable package sources](kimi_k3_l2_pin_patch/) include a reproducible builder:

~~~bash
python3 test/manual/ascend/kimi_k3_l2_pin_patch/build_zip.py /tmp/k3-pin-diagnostics.zip
cmp /tmp/k3-pin-diagnostics.zip test/manual/ascend/kimi_k3_l2_pin_patch/k3-pin-diagnostics.zip
sha256sum /tmp/k3-pin-diagnostics.zip
~~~

Replace the URL's branch name with the published commit SHA for immutable bytes.

## Existing Nightly automation

The reusable multi-node template accepts container_patches. A caller in the
upstream CI environment can use this job with its normal trigger/permissions:

~~~yaml
jobs:
  k3-pin-diagnostics:
    uses: sgl-project/sglang/.github/workflows/nightly-test-npu-e2e-multi-node.yml@main
    with:
      ref: main
      runner: linux-amd64-cpu-4
      test_type: accuracy
      test_config_name: kimi_k3_w4a8_32p_gpqa
      test_case: test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py
      node_size: 4
      prefill_decode_deployment: mix
      accuracy_dataset: gsm8k
      image: swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-a3
      container_patches: https://raw.githubusercontent.com/hanwlax/sglang/codex/k3-hicache-pin-diagnostics/test/manual/ascend/kimi_k3_l2_pin_patch/k3-pin-diagnostics.zip
    secrets: inherit
~~~

This runs main plus the prints installed inside each Pod. Keep source, image,
and dataset aligned with the failing run. Cluster credentials/shared PVC are
required. **The current top-level nightly-test-npu.yml does not expose/forward
container_patches**; do not add that input to its dispatch command. Use the
test-team entry that forwards it, the reusable-template caller above, or the
driver below. This PR does not change workflow inputs.

For the existing driver's single-case route, run in the CI runner after its
normal source-copy preparation:

~~~bash
run_id=123456789  # Replace with the run's unique numeric ID.
checkout="/root/.cache/tests/sglang-${run_id}"
patch_url=https://raw.githubusercontent.com/hanwlax/sglang/codex/k3-hicache-pin-diagnostics/test/manual/ascend/kimi_k3_l2_pin_patch/k3-pin-diagnostics.zip
curl -fsSL "${patch_url}" -o /tmp/k3-pin-diagnostics.zip
mkdir -p "${checkout}/container-patches/k3-pin"
python3 -m zipfile -e /tmp/k3-pin-diagnostics.zip "${checkout}/container-patches/k3-pin"

cd python/sglang/test/ascend/e2e
python3 -u run_npu_e2e_test.py \
  --env ci \
  --image swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-a3 \
  --sglang-source-relative-path "tests/sglang-${run_id}" \
  --metrics-data-file "/root/.cache/tests/output/${run_id}/accuracy/test_npu_kimi_k3_w4a8_32p_gpqa" \
  --test-case test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py \
  --kube-name-space sgl-project \
  --kube-job-type multi-pd-mix \
  --kube-job-name-prefix "ascend-${run_id}" \
  --run-id "${run_id}" \
  --container-patches container-patches/k3-pin \
  --accuracy-dataset gsm8k \
  --node-size 4 \
  --sglang-is-in-ci
~~~

The workflow input takes an HTTPS zip URL. The driver's --container-patches
takes an already extracted directory relative to the mounted checkout.

## Installer and manual application

The installer matches the launcher's package selection: mounted checkout by
default, installed image package for SGLANG_USE_IMAGE=true. It only installs
the Mamba exception hook and its helper, without editing the executing launcher.
A file lock and atomic writes handle four Pods sharing a checkout. Repeated
installs verify the same hook/helper; unexpected source context or a different
existing helper fails installation.

To apply manually, extract the archive and run inside the container:

~~~bash
cd /tmp/k3-pin-diagnostics
bash install.sh /sgl-workspace/sglang/python/sglang
~~~

The optional argument is the exact package directory imported by the server.
Verify "K3_PIN_PATCH applied" or "K3_PIN_PATCH already applied", the target path,
and source hash in every Pod's log. The original source is saved beside it as
mamba.py.k3-pin-original. Fresh Pods/run-scoped checkouts are the normal reset.

## Evidence and limits

The installer runs a standard-library-only metadata subprocess without NPU
initialization. In Nightly it runs before SDK setup, so its paths are pre-setup
observations. On failure the collector records actual loaded libraries and
torch_npu-reported versions again.

Search Pod logs for NPU_PIN_DIAGNOSTIC. JSON records are written under:

~~~text
${METRICS_DATA_FILE}/npu_pin_diagnostics/
~~~

The existing artifact upload collects that directory. Records include
hostname/PID/time, package/build versions, wheel basename, Git IDs, and hashes.
Failure records add original data/storage pointers, offsets/shapes/strides,
ACL registration coverage and pointer attributes, Python/ATen pinned results,
dispatcher/TLS state, and loaded source/library hashes.

For one verified torch_npu ELF, an exact SHA256 and instruction check guard
reading the cached compatibility decision. Other builds skip that read.
No C++ functions are called through ctypes. Exception unwinding can restore
state; inconsistent observations may require native-check instrumentation.

The patch cannot modify the Kubernetes driver outside its Pods. Retain image
IDs separately while Pods exist, for example:

~~~bash
kubectl -n sgl-project get pods -l "run-id=${run_id}" \
  -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .status.containerStatuses[*]}{.imageID}{" "}{end}{"\n"}{end}'
~~~

Earlier A3 probes passed with a 3,337,224,192-byte registered host tensor and
16 processes holding 230,691,897,344 registered host bytes. These are negative
reproduction results, not evidence of a fix. The collector ran successfully on
A3, with its failure entry manually invoked after a successful native copy.
The original four-node failure still needs test-team capture.
