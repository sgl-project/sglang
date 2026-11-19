# Kimi K3 L2 pinned-memory diagnostics

The diagnostic files travel with the PR checkout. The original K3 accuracy case
automatically runs their local install.sh through the existing container patch
step. No diagnostic archive, URL input, or runtime download is needed.

The error's root cause remains unconfirmed. The hook preserves the original
exception, allocator, transfer mode, and four-node test parameters.

## Run through Nightly

From an account with upstream Actions permissions:

~~~bash
gh workflow run nightly-test-npu.yml --repo sgl-project/sglang --ref main \
  -f ref=refs/pull/51513/head \
  -f job_filter=nightly-poc-multi-node-mix-tests \
  -f accuracy_dataset=gsm8k \
  -f image_a3=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-a3
~~~

The workflow checks out the PR, copies its files to the run-scoped shared PVC,
and mounts them in each Pod at /root/sglang. The launcher automatically includes
test/manual/ascend/kimi_k3_l2_pin_patch in its local container patch directories
only for test_npu_kimi_k3_w4a8_32p_gpqa.py. No container_patches URL input is needed.
The existing mix group also runs its other Kimi cases; they do not select these
diagnostics. The workflow matrix and original case configuration are unchanged.

Keep the default checkout mode (use_image_sglang empty/false) for the PR source.
If the existing image-code mode is explicitly selected, the installer instead
patches the image package that the launcher will import.

## Files applied inside the container

[The local directory](kimi_k3_l2_pin_patch/) contains install.sh, install.py,
mamba-pin-diagnostics.patch, and npu_pin_diagnostics.py. The installer checks the
patch in a temporary tree and applies only the Mamba exception hook plus its
helper. A lock and atomic writes handle four Pods sharing a checkout. Repeated
installs verify the same hook/helper; unexpected source context or a different
existing helper fails installation and stops that Pod's launcher.

Verify "K3_PIN_PATCH applied" or "K3_PIN_PATCH already applied", the target path,
and source hash in each Pod's log. The original source is saved beside it as
mamba.py.k3-pin-original. Fresh Pods/run-scoped checkouts are the normal reset.

For manual application after the files have been copied into a container:

~~~bash
bash /root/sglang/test/manual/ascend/kimi_k3_l2_pin_patch/install.sh \
  /sgl-workspace/sglang/python/sglang
~~~

The optional argument is the exact sglang package directory imported by the
server. Omit it in the normal Nightly launcher to follow its package selection.

For direct use of run_npu_e2e_test.py, use the PR checkout already copied to the
shared directory and keep the original K3 test-case argument. The launcher adds
the local diagnostic directory automatically. Its existing --container-patches
option can still specify other local patch directories when needed.

## Evidence and limits

The installer runs a standard-library-only metadata subprocess without NPU
initialization. In Nightly this precedes SDK setup, so those paths are pre-setup
observations. On the original pinned exception, the collector records active
loaded libraries and torch_npu-reported versions again, and re-raises the error.

Search Pod logs for NPU_PIN_DIAGNOSTIC. JSON records go to:

~~~text
${METRICS_DATA_FILE}/npu_pin_diagnostics/
~~~

The existing artifact upload collects that directory. Records include package/
build versions, wheel basename, Git IDs, hashes, original data/storage pointers,
buffer shapes/strides/offsets, ACL registration coverage and pointer attributes,
Python/ATen pinned results, and dispatcher/TLS state. One verified torch_npu ELF
supports a compatibility-cache read guarded by SHA256 and instruction checks;
other builds skip it. No C++ functions are invoked through ctypes.
Post-exception state can require follow-up native-check instrumentation.

Earlier A3 probes and the collector passed, but did not reproduce the original
failure. The installer passed first/repeat installation in a temporary copy of
the actual A3 container's default code; its default source was unchanged.
The original four-node failure still needs test-team capture.
