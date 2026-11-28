# Kimi K3: diagnose the image code through Nightly YAML

This reference change runs the original four-node K3 case using the published
image's SGLang code. Diagnostic files are embedded in the workflow YAML.
It requires neither a PR Python checkout nor diagnostic file/archive downloads.
The root cause remains unconfirmed; allocator, transfer mode, test parameters,
and the original exception are retained.

## Use the modified workflow definitions

The test team must load both changed YAML files:

- .github/workflows/nightly-test-npu.yml sets use_image_sglang=true for the
  original K3 accuracy case.
- .github/workflows/nightly-test-npu-e2e-multi-node.yml generates the diagnostic
  files from literal inline content and supplies their directory to the existing
  per-Pod container patch hook.

Setting only the test-code ref to PR #51513 does not load the YAML changes.
Put these two workflow changes on the CI configuration branch used by the
test team. For example, once that branch exists in the CI repository:

~~~bash
ci_yaml_branch=YOUR_CI_CONFIGURATION_BRANCH
gh workflow run nightly-test-npu.yml --repo sgl-project/sglang \
  --ref "${ci_yaml_branch}" \
  -f ref=main \
  -f job_filter=nightly-poc-multi-node-mix-tests \
  -f accuracy_dataset=gsm8k \
  -f image_a3=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-a3
~~~

Here --ref selects the modified workflow definitions; ref=main keeps the normal
test/harness checkout on main. The server uses the image package because the
existing driver receives use_image_sglang=true. No PR Python code is used.
The mix filter includes the other existing Kimi cases; their runtime selection
and patch behavior are unchanged.

## Files applied to the image

1. The reusable YAML creates install.sh, install.py,
   mamba-pin-diagnostics.patch, and npu_pin_diagnostics.py through literal
   here-documents. The payload belongs to the workflow definition.
2. The normal source-copy step carries the generated container-patches directory
   onto the shared PVC mounted into every Pod.
3. The existing --container-patches option passes that directory to
   SGLANG_NPU_CONTAINER_PATCHES. The Pod launcher's existing hook runs install.sh.
4. With SGLANG_USE_IMAGE=true, the installer locates and patches the installed
   sglang package, so the server imports the instrumented image code.

The inline step runs only for the original accuracy case in image mode.
No checkout changes to run_npu_testcase.sh or production Python modules are
needed. [The separate plain files](kimi_k3_l2_pin_patch/) are review/manual copies
of the inline payload; automation does not read them from a PR checkout.

The installer uses a lock and atomic writes and supports repeated installation.
Unexpected source context or a different helper produces an installation error.
Verify "K3_PIN_PATCH applied" or "K3_PIN_PATCH already applied", the actual image
package path, and source hash in every Pod. Without that marker, installation
has not been confirmed; retain the installer error. An original-source backup
is kept beside mamba.py. Fresh Pods reset these image-package edits.

## Evidence and limits

The installer runs standard-library metadata collection before NPU
initialization. This precedes SDK setup in the existing launcher; failure
records capture actual loaded libraries and torch_npu runtime versions again.

Search Pod logs for NPU_PIN_DIAGNOSTIC. The existing artifact upload collects:

~~~text
${METRICS_DATA_FILE}/npu_pin_diagnostics/
~~~

Records include versions/builds/hashes, original buffer data/storage pointers,
shape/strides/offsets, ACL registration coverage and attributes, Python/ATen
pinned results, and dispatcher/TLS state. Reading a compatibility cache in one
verified torch_npu ELF requires an exact SHA256 and instruction match; other
builds skip it. No C++ functions are invoked through ctypes. Post-exception
state may require follow-up instrumentation at the native check.

Earlier A3 probes and collector validation passed without reproducing the
original failure. The installer passed on a temporary copy of the actual
container source. The original four-node capture still requires the test team.
