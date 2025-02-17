# Deploying a RoCE Network-Based SGLANG Two-Node Inference Service on a Kubernetes (K8S) Cluster

LeaderWorkerSet (LWS) is a Kubernetes API that aims to address common deployment patterns of AI/ML inference workloads. A major use case is for multi-host/multi-node distributed inference.

Sglang can also be deployed with LWS on Kubernetes for distributed model serving.

Please see this guide for more details on deploying SGLang on Kubernetes using LWS.

Here we take the deployment of deepseekR1 as an example.

## Prerequisites

1. At least 2* H20 *8 GPU k8s node
2. K8S cluster with LWS correctly installed, if not , please follow  [this doc](https://github.com/kubernetes-sigs/lws/blob/main/docs/setup/install.md)


## Basic Example

Basic Example Doc has been introduced  here. please [visit this guide](https://github.com/kubernetes-sigs/lws/tree/main/docs/examples/sglang)

However,that document only describes the basic NCCL socket mode.

Here , we make some simple modifications to adapt to the `RDMA` scenario

## RDMA ROCE case

hardware:

* 4* 200Gb cx7 X 2 nodes
* 8*H20 per node


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

lws.yaml is

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
              capabilities:
                add: [ "IPC_LOCK" ]
            env:
             # - name: NCCL_DEBUG
             #   value: TRACE
             # - name: NCCL_SOCKET_IFNAME
             #   value: eth0,bond
             # - name: NCCL_IB_HCA
             #   value: mlx5
             # - name: NCCL_SOCKET_NTHREADS
             #   value: "16"
             # - name: NCCL_NSOCKS_PERTHREAD
             #   value: "4"
             # - name: NCCL_IB_DISABLE
             #   value: "0"
              - name: NCCL_IB_GID_INDEX
                value: "3"
              - name: LWS_WORKER_INDEX
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
             # - --enable-flashinfer-mla
             # - --disable-radix-cache
              - --mem-fraction-static
              -  "0.93"
              - --torch-compile-max-bs
              - "8"
              - --max-running-requests
              - "50"
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
#                rdma/hca: "8"
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
              path: /data1/maas_hosted_models/models/deepseek_v3_moe
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
              capabilities:
                add: [ "IPC_LOCK" ]
            env:
            #- name: NCCL_SOCKET_NTHREADS
            #  value: "16"
            #- name: NCCL_NSOCKS_PERTHREAD
            #  value: "4"
            #- name: NCCL_IB_DISABLE
            #  value: "0"
            - name: NCCL_IB_GID_INDEX
              value: "3"
            # - name: NCCL_DEBUG
            #  value: TRACE
            #- name: NCCL_SOCKET_IFNAME
            #  value: eth0,bond
            #- name: NCCL_IB_HCA
            #  value: mlx5
            - name: LWS_WORKER_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
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
              - "50"
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
#                rdma/hca: "8"
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

Then use  `kubectl apply -f lws.yaml` you will get this output.

```text
NAME           READY   STATUS    RESTARTS       AGE
sglang-0       0/1     Running   0              9s
sglang-0-1     1/1     Running   0              9s
```

Waiting  for sglang leader `sglang-0` 's status becoming `1/1` which means Ready...

Now if successfully , you will get output  like this.

using command `kubectl logs sglang-0` to view the leader node's log.

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

if not successfully startup, please follow this steps to check or see the remaining issues... thanks.

### Debug

* Set `NCCL_DEBUG=TRACE` to check if it is a nccl communication problem


This will solve most NCCL problems.

#### ROCE env

* Please make sure that RDMA devices are available in the cluster environment.
* Please make sure that the nodes in the cluster have mellanox NICs with RoCE. In this example, we use mellanox ConnectX 5 model NICs, and the proper OFED driver has been installed, if not, please refer to the document Install OFED Driver to install the driver.
* Env Check:
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
* ofed driver
  ```shell
   ofed_info -s
  OFED-internal-23.07-0.5.0:
  ```
* rdma link show and check ib dev
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
* test roce network speed in th host
 ```shell
  yum install qperf
  # for server：
  excute qperf
  # for client
  qperf -t 60 -cm1 <server_ip>   rc_rdma_write_bw
```

* check rdma accessible in  your container...
 ```shell
   # ibv_devices
   # ibv_devinfo
  ```

## Keys to Success

*  NCCL's env in the above yaml. for some low version nccl. you should check env `NCCL_IB_GID_INDEX`
* `NCCL_SOCKET_IFNAME` is also important but in container env, it's not a problem
* `NCCL_DEBUG` is the most important..... but i find some times it will not show the error log in container.. maybe  problems of the image. you can switch your docker images
*  Don't use ubuntu 18.04'based docker images...

## Remaining issues

* In K8s/Docker/Containerd Case, we just use `hostnetwork`  to avoid performance loss
* We use the `privileged` mode... it's not safe and in container, we can't isolate the gpu

## Todo

* integrated with [k8s rdma share plugin](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin).
