---
title: Cambricon MLUs
description: "Install SGLang on Cambricon MLUs with matching vendor runtime packages."
---

This guide describes how to prepare the Cambricon software environment and install SGLang for MLU CI environment validation. MLU inference support is provided separately.

## Prerequisites

Install the MLU driver and firmware on the host before proceeding. Select packages matching the component version configuration, container OS, CPU architecture, and Python version.

## MLU software components

Component versions are defined in [`scripts/ci/mlu/component_versions.properties`](https://github.com/sgl-project/sglang/blob/main/scripts/ci/mlu/component_versions.properties). Before installing SGLang, install the components at the versions specified in this file for the SGLang commit you intend to run.

The table below lists the MLU software components and their roles.

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Component</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowSpan="5">PyTorch stack</td>
      <td><code>pytorch</code></td>
      <td>CPU build of the PyTorch deep learning framework</td>
    </tr>
    <tr>
      <td><code>torchvision</code></td>
      <td>CPU build of the PyTorch vision package</td>
    </tr>
    <tr>
      <td><code>torchaudio</code></td>
      <td>CPU build of the PyTorch audio package</td>
    </tr>
    <tr>
      <td><code>torch_mlu</code></td>
      <td>PyTorch MLU device adapter</td>
    </tr>
    <tr>
      <td><code>torch_mlu_ops</code></td>
      <td>PyTorch third-party operator library on MLU</td>
    </tr>
    <tr>
      <td>Toolkit</td>
      <td><code>cntoolkit</code></td>
      <td>Cambricon Neuware Toolkit, distributed as a bundle of development and runtime packages</td>
    </tr>
    <tr>
      <td rowSpan="3">Operator libraries</td>
      <td><code>cnnl</code></td>
      <td>Cambricon Network Library for neural network operators</td>
    </tr>
    <tr>
      <td><code>cnnlextra</code></td>
      <td>CNNL Extra Library, including transformer, quantization, and detection operators</td>
    </tr>
    <tr>
      <td><code>mlu-ops</code></td>
      <td>Cambricon open-source operator library for neural networks</td>
    </tr>
    <tr>
      <td rowSpan="2">Communication libraries</td>
      <td><code>cncl</code></td>
      <td>Cambricon Communication Library for communicator management and collective communication</td>
    </tr>
    <tr>
      <td><code>cnclep</code></td>
      <td>Cambricon Neuware Communication Library for Expert Parallelism, providing high-throughput, low-latency MoE communication</td>
    </tr>
    <tr>
      <td>Kernel programming</td>
      <td><code>triton</code></td>
      <td>A language and compiler for custom deep learning operations on the MLU backend</td>
    </tr>
    <tr>
      <td>Disaggregated architecture</td>
      <td><code>mooncake</code></td>
      <td>MLU-adapted Mooncake with a KV cache-centric architecture for disaggregated LLM inference and training</td>
    </tr>
    <tr>
      <td>Distributed execution</td>
      <td><code>ray</code></td>
      <td>MLU-adapted Ray for distributed computing and scaling AI workloads</td>
    </tr>
  </tbody>
</table>

## Installation

### Create a Docker container

Contact Cambricon official support at [ecosystem@cambricon.com](mailto:ecosystem@cambricon.com) to obtain the container image.

Set `MLU_IMAGE` to the image address supplied by Cambricon, using the tag specified by `MLU_CI_IMAGE_TAG` in the version configuration. The example below uses two MLU devices on an x86_64 host and a container with Ubuntu and Python matching that image tag.

Run on the host:

```bash
docker run -it --user root --network=host --shm-size=16g \
  --device=/dev/cambricon_ctl \
  --device=/dev/cambricon_dev0 --device=/dev/cambricon_dev1 \
  --workdir /workspace \
  "$MLU_IMAGE" bash
```

The device mappings above use the nodes exposed by the reference host driver. Adjust them to match your host's device configuration. Run the remaining installation commands inside the container.

### Install the MLU components

Contact Cambricon official support at [ecosystem@cambricon.com](mailto:ecosystem@cambricon.com) to obtain the required software packages and installation guidance.

### Install SGLang

Use the Python environment where you installed the MLU components:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_mlu]"
```

MLU installation uses the shared `pyproject_other.toml` configuration. The `all_mlu`
extra includes the runtime dependencies and tracing support. The native libraries and framework packages must already be installed.

## Verify the CI environment

Install the test dependencies, then run the single-device RMSNorm and two-device SGLang communicator suites:

```bash
pip install -e "python[dev_mlu]"
python test/run_suite.py --hw mlu --suite pr-test-1-mlu
python test/run_suite.py --hw mlu --suite pr-test-2-mlu
```

These tests validate the MLU RMSNorm implementation and SGLang communicator collectives without loading a model or starting an inference server.
