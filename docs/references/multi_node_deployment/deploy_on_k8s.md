# Deploy On Kubernetes

This document is for deploying a RoCE network-based SGLang two-node inference service on a Kubernetes (K8S) cluster.

[LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws) is a Kubernetes API that aims to address common deployment patterns of AI/ML inference workloads. A major use case is for multi-host/multi-node distributed inference.

SGLang can also be deployed with LWS on Kubernetes for distributed model serving.

Please see this guide for more details on deploying SGLang on Kubernetes using LWS.

Here we take the deployment of DeepSeek-R1 as an example.

## Prerequisites

1. At least two Kubernetes nodes, each with two H20 systems and eight GPUs, are required.

2. Make sure your K8S cluster has LWS correctly installed. If it hasn't been set up yet, please follow the [installation instructions](https://github.com/kubernetes-sigs/lws/blob/main/site/content/en/docs/installation/_index.md). **Note:** For LWS versions ≤0.5.x, you must use the Downward API to obtain `LWS_WORKER_INDEX`, as native support for this feature was introduced in v0.6.0.

## Basic example

For the basic example documentation, refer to [Deploy Distributed Inference Service with SGLang and LWS on GPUs](https://github.com/kubernetes-sigs/lws/tree/main/docs/examples/sglang).

However, that document only covers the basic NCCL socket mode.

In this section, we’ll make some simple modifications to adapt the setup to the RDMA scenario.

## RDMA RoCE case

* Check your env:

```bash
[root@node1 ~]# ibstatus
Infiniband device 'mlx5_bond_0' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe64:c79a
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_bond_1' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe6e:c3ec
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_bond_2' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe73:0dd7
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_bond_3' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe36:f7ff
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet
```

* Prepare the `lws.yaml` file for deploying on k8s.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: sglang
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-leader
            image: sglang:latest
            securityContext:
              privileged: true
            env:
              - name: NCCL_IB_GID_INDEX
                value: "3"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --mem-fraction-static
              -  "0.93"
              - --torch-compile-max-bs
              - "8"
              - --max-running-requests
              - "20"
              - --tp
              - "16" # Size of Tensor Parallelism
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --host
              - "0.0.0.0"
              - --port
              - "40000"
            resources:
              limits:
                nvidia.com/gpu: "8"
            ports:
              - containerPort: 40000
            readinessProbe:
              tcpSocket:
                port: 40000
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model
                mountPath: /work/models
              - name: ib
                mountPath: /dev/infiniband
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model
            hostPath:
              path: '< your models dir >' # modify it according your models dir
          - name: ib
            hostPath:
              path: /dev/infiniband
    workerTemplate:
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-worker
            image: sglang:latest
            securityContext:
              privileged: true
            env:
            - name: NCCL_IB_GID_INDEX
              value: "3"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --mem-fraction-static
              - "0.93"
              - --torch-compile-max-bs
              - "8"
              - --max-running-requests
              - "20"
              - --tp
              - "16" # Size of Tensor Parallelism
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
            resources:
              limits:
                nvidia.com/gpu: "8"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model
                mountPath: /work/models
              - name: ib
                mountPath: /dev/infiniband
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: ib
            hostPath:
              path: /dev/infiniband
          - name: model
            hostPath:
              path: /data1/models/deepseek_v3_moe
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-leader
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: sglang
    role: leader
  ports:
    - protocol: TCP
      port: 40000
      targetPort: 40000

```

* Then use  `kubectl apply -f lws.yaml` you will get this output.

```text
NAME           READY   STATUS    RESTARTS       AGE
sglang-0       0/1     Running   0              9s
sglang-0-1     1/1     Running   0              9s
```

Wait for the sglang leader (`sglang-0`) status to change to 1/1, which indicates it is `Ready`.

You can use the command `kubectl logs -f sglang-0` to view the logs of the leader node.

Once successful, you should see output like this:

```text
[2025-02-17 05:27:24 TP1] Capture cuda graph end. Time elapsed: 84.89 s
[2025-02-17 05:27:24 TP6] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP0] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP7] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP3] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP2] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP4] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP1] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP5] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24] INFO:     Started server process [1]
[2025-02-17 05:27:24] INFO:     Waiting for application startup.
[2025-02-17 05:27:24] INFO:     Application startup complete.
[2025-02-17 05:27:24] INFO:     Uvicorn running on http://0.0.0.0:40000 (Press CTRL+C to quit)
[2025-02-17 05:27:25] INFO:     127.0.0.1:48908 - "GET /get_model_info HTTP/1.1" 200 OK
[2025-02-17 05:27:25 TP0] Prefill batch. #new-seq: 1, #new-token: 7, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue-req: 0
[2025-02-17 05:27:32] INFO:     127.0.0.1:48924 - "POST /generate HTTP/1.1" 200 OK
[2025-02-17 05:27:32] The server is fired up and ready to roll!
```

If it doesn’t start up successfully, please follow these steps to check for any remaining issues. Thanks!

### Debug

* Set `NCCL_DEBUG=TRACE` to check if it is a NCCL communication problem.

This should resolve most NCCL-related issues.

***Notice: If you find that NCCL_DEBUG=TRACE is not effective in the container environment, but the process is stuck or you encounter hard-to-diagnose issues, try switching to a different container image. Some images may not handle standard error output properly.***

#### RoCE scenario

* Please make sure that RDMA devices are available in the cluster environment.
* Please make sure that the nodes in the cluster have Mellanox NICs with RoCE. In this example, we use Mellanox ConnectX 5 model NICs, and the proper OFED driver has been installed. If not, please refer to the document [Install OFED Driver](https://docs.nvidia.com/networking/display/mlnxofedv461000/installing+mellanox+ofed) to install the driver.
* Check your env:

  ```shell
  $ lspci -nn | grep Eth | grep Mellanox
  0000:7f:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0000:7f:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0000:c7:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0000:c7:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:08:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:08:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:a2:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:a2:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  ```

* Check the OFED driver:

  ```shell
  ofed_info -s
  OFED-internal-23.07-0.5.0:
  ```

* Show RDMA link status and check IB devices:

  ```shell
  $ rdma link show
  8/1: mlx5_bond_0/1: state ACTIVE physical_state LINK_UP netdev reth0
  9/1: mlx5_bond_1/1: state ACTIVE physical_state LINK_UP netdev reth2
  10/1: mlx5_bond_2/1: state ACTIVE physical_state LINK_UP netdev reth4
  11/1: mlx5_bond_3/1: state ACTIVE physical_state LINK_UP netdev reth6

  $ ibdev2netdev
  8/1: mlx5_bond_0/1: state ACTIVE physical_state LINK_UP netdev reth0
  9/1: mlx5_bond_1/1: state ACTIVE physical_state LINK_UP netdev reth2
  10/1: mlx5_bond_2/1: state ACTIVE physical_state LINK_UP netdev reth4
  11/1: mlx5_bond_3/1: state ACTIVE physical_state LINK_UP netdev reth6
  ```

* Test RoCE network speed on the host:

  ```shell
  yum install qperf
  # for server：
  execute qperf
  # for client
  qperf -t 60 -cm1 <server_ip>   rc_rdma_write_bw
  ```

* Check RDMA accessible in your container:

  ```shell
  # ibv_devices
  # ibv_devinfo
  ```

## Keys to success

* In the YAML configuration above, pay attention to the NCCL environment variable. For older versions of NCCL, you should check the NCCL_IB_GID_INDEX environment setting.
* NCCL_SOCKET_IFNAME is also crucial, but in a containerized environment, this typically isn’t an issue.
* In some cases, it’s necessary to configure GLOO_SOCKET_IFNAME correctly.
* NCCL_DEBUG is essential for troubleshooting, but I've found that sometimes it doesn't show error logs within containers. This could be related to the Docker image you're using. You may want to try switching images if needed.
* Avoid using Docker images based on Ubuntu 18.04, as they tend to have compatibility issues.

## Remaining issues

* In Kubernetes, Docker, or Containerd environments, we use hostNetwork to prevent performance degradation.
* We utilize privileged mode, which  isn’t secure. Additionally, in containerized environments, full GPU isolation cannot be achieved.

## TODO

* Integrated with [k8s-rdma-shared-dev-plugin](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin).
