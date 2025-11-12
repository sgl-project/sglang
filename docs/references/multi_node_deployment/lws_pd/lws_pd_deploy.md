# LWS Based PD Deploy

## 0. Prerequisites

1. k8s >=1.26
2. lws installed on k8s.

## 1. Image Preparation

`lmsysorg/sglang:deepep`

## 2. Deployment Manifest Files

***Notice: We will package all deployment files into Helm Chart format in the near future. Interested community members can contact us to contribute***

### Prefill

Prefill manifest file [prefill.yaml](lws-examples/p.yaml)

*Note: The NodeSelector section, model location section, and taint toleration section can be adjusted according to your actual deployment environment*

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseekr10528-prefill-main
spec:
  leaderWorkerTemplate:
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        containers:
        - command:
          - python3
          - -m
          - sglang.launch_server
          - --port
          - "30000"
          - --host
          - "0.0.0.0"
          - --model-path
          - /work/models
          - --disaggregation-ib-device
          # should modify according your rdma env
          - mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
          - --chunked-prefill-size
          - "524288"
          - --max-prefill-tokens
          - "32768"
          - --page-size
          - "64"
          #          - --init-expert-location
          #          - /home/aiges/tuned/attachment_ep_statistics/prefill_in1024.json
          - --ep-dispatch-algorithm
          - dynamic
          - --eplb-algorithm
          - deepseek
          #          - --deepep-config
          #          -  /home/aiges/tuned/tuned_8sms.json
          - --enable-dp-lm-head
          - --enable-dp-attention
          - --dp-size
          - "16"
          - --disable-radix-cache
          - --moe-a2a-backend
          - deepep
          - --disaggregation-mode
          - prefill
          - --mem-fraction-static
          - "0.7"
          - --context-length
          - "32768"
          - --tp
          - "16"
          - --dist-init-addr
          - $(LWS_LEADER_ADDRESS):20102
          - --nnodes
          - $(LWS_GROUP_SIZE)
          - --node-rank
          - $(LWS_WORKER_INDEX)
          - --trust-remote-code
          - --ep-num-redundant-experts
          - "32"
          - --moe-dense-tp-size
          - "1"
          - --max-running-requests
          - "1024"
          env:
#          - name: NVSHMEM_HCA_PE_MAPPING
#            value: "mlx5_bond_0:1:2,mlx5_bond_1:1:2,mlx5_bond_2:1:2,mlx5_bond_3:1:2"
#          - name: NVSHMEM_HCA_LIST
#            value: "mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"
          - name: NVSHMEM_IB_GID_INDEX
            value: "3"
          - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
            value: "1"
          - name: SGLANG_SET_CPU_AFFINITY
            value: "true"
          - name: SGLANG_ENABLE_JIT_DEEPGEMM
            value: "1"
          - name: NCCL_IB_QPS_PER_CONNECTION
            value: "8"
          - name: NCCL_IB_SPLIT_DATA_ON_QPS
            value: "1"
          - name: NCCL_NET_PLUGIN
            value: none
          - name: NCCL_IB_TC
            value: "136"
          - name: NCCL_MIN_NCHANNELS
            value: "4"
          - name: MC_TE_METRIC
            value: "false"
          - name: NCCL_IB_SL
            value: "5"
          - name: NCCL_IB_HCA
            value: ^=mlx5_0,mlx5_5,mlx5_6
          - name: LWS_WORKER_INDEX
            valueFrom:
              fieldRef:
                fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
          image: lmsysorg/sglang:deepep
          name: sglang-leader
          ports:
          - containerPort: 30000
            protocol: TCP
          readinessProbe:
            periodSeconds: 30
            tcpSocket:
              port: 30000
          resources:
            limits:
              nvidia.com/gpu: "8"
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
            privileged: true
          volumeMounts:
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /work/models
            name: model
          - mountPath: /dev/infiniband
            name: ib
          - mountPath: /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs
            name: cf
          - mountPath: /root/.cache
            name: sgl-cache
        dnsPolicy: ClusterFirstWithHostNet
        hostIPC: true
        hostNetwork: true
        nodeSelector:
          pd: "yes"
        tolerations:
        - key: pd
          operator: Exists
        - key: node-role
          operator: Exists
        volumes:
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            # modify according to you deployment env
            path: /data1/maas_hosted_models/models/DeepSeek-R1-0528/deepseek_r1_0528
          name: model
        - hostPath:
            path: /dev/infiniband
          name: ib
        - hostPath:
            # modify according to you deployment env
            path: /data1/maas_hosted_models/models/fused_moe_triton/configs
          name: cf
        - hostPath:
            # modify according to you deployment env
            path: /data1/sgl_cache
            type: DirectoryOrCreate
          name: sgl-cache
    restartPolicy: RecreateGroupOnPodRestart
    size: 2
    workerTemplate:
      metadata: {}
      spec:
        containers:
        - command:
          - python3
          - -m
          - sglang.launch_server
          - --model-path
          - /work/models
          - --disaggregation-ib-device
          - mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
          - --chunked-prefill-size
          - "524288"
          - --max-prefill-tokens
          - "32768"
          - --page-size
          - "64"
          #- --init-expert-location
          #- /home/aiges/tuned/attachment_ep_statistics/prefill_in1024.json
          - --ep-dispatch-algorithm
          - dynamic
          - --eplb-algorithm
          - deepseek
#          - --deepep-config
#          -  /home/aiges/tuned/tuned_8sms.json
          - --enable-dp-lm-head
          - --enable-dp-attention
          - --dp-size
          - "16"
          - --disable-radix-cache
          - --moe-a2a-backend
          - deepep
          - --disaggregation-mode
          - prefill
          - --mem-fraction-static
          - "0.7"
          - --context-length
          - "32768"
          - --tp
          - "16"
          - --dist-init-addr
          - $(LWS_LEADER_ADDRESS):20102
          - --nnodes
          - $(LWS_GROUP_SIZE)
          - --node-rank
          - $(LWS_WORKER_INDEX)
          - --trust-remote-code
          - --ep-num-redundant-experts
          - "32"
          - --moe-dense-tp-size
          - "1"
          - --max-running-requests
          - "1024"
          env:
          - name: SGLANG_SET_CPU_AFFINITY
            value: "true"
          - name: SGLANG_HACK_DEEPEP_NUM_SMS
            value: "8"
          - name: SGLANG_HACK_DEEPEP_NEW_MODE
            value: "0"
#          - name: NVSHMEM_HCA_PE_MAPPING
#            value: "mlx5_bond_0:1:2,mlx5_bond_1:1:2,mlx5_bond_2:1:2,mlx5_bond_3:1:2"
#          - name: NVSHMEM_HCA_LIST
#            value: "mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"
          - name: NCCL_IB_HCA
            value: ^=mlx5_0,mlx5_5,mlx5_6
          - name: NVSHMEM_IB_TRAFFIC_CLASS
            value: "16"
          - name: NVSHMEM_IB_GID_INDEX
            value: "3"
          - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
            value: "1"
          - name: CUDA_LAUNCH_BLOCKING
            value: "0"
          - name: SGLANG_MOONCAKE_TRANS_THREAD
            value: "8"
          - name: SGLANG_ENABLE_JIT_DEEPGEMM
            value: "1"
          - name: SGL_CHUNKED_PREFIX_CACHE_THRESHOLD
            value: "0"
          - name: NCCL_IB_QPS_PER_CONNECTION
            value: "8"
          - name: NCCL_IB_SPLIT_DATA_ON_QPS
            value: "1"
          - name: NCCL_NET_PLUGIN
            value: none
          - name: NCCL_IB_TC
            value: "136"
          - name: NCCL_MIN_NCHANNELS
            value: "4"
          - name: MC_TE_METRIC
            value: "true"
          - name: NCCL_IB_SL
            value: "5"
          - name: LWS_WORKER_INDEX
            valueFrom:
              fieldRef:
                fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
          image: lmsysorg/sglang:deepep
          name: sglang-worker
          ports:
          - containerPort: 30001
            protocol: TCP
          resources:
            limits:
              nvidia.com/gpu: "8"
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
            privileged: true
          volumeMounts:

          - mountPath: /root/.cache
            name: sgl-cache
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /work/models
            name: model
          - mountPath: /dev/infiniband
            name: ib
          - mountPath: /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs
            name: cf
        dnsPolicy: ClusterFirstWithHostNet
        hostIPC: true
        hostNetwork: true
        nodeSelector:
          pd: "yes"
        tolerations:
        - key: pd
          operator: Exists
        - key: node-role
          operator: Exists
        volumes:
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            path: /dev/infiniband
          name: ib
        - hostPath:
            path: /data1/maas_hosted_models/models/DeepSeek-R1-0528/deepseek_r1_0528
          name: model
        - hostPath:
            path: /data1/maas_hosted_models/models/fused_moe_triton/configs
          name: cf
        - hostPath:
            path: /data1/sgl_cache
            type: DirectoryOrCreate
          name: sgl-cache

```

### Decode

Decode node deployment manifest file [decode.yaml](lws-examples/d.yaml)

*Note: The NodeSelector section, model location section, and taint toleration section can be adjusted according to your actual deployment environment*

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseekr10528-decode-main
spec:
  leaderWorkerTemplate:
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        containers:
        - command:
          - python3
          - -m
          - sglang.launch_server
          - --port
          - "30000"
          - --host
          - "0.0.0.0"
          - --model-path
          - /work/models
          - --chunked-prefill-size
          - "262144"
          - --page-size
          - "64"
          - --enable-dp-attention
          - --enable-dp-lm-head
          - --dp-size
          - "16"
          - --moe-a2a-backend
          - deepep
          - --disaggregation-mode
          - decode
          - --mem-fraction-static
          -  "0.849"
          - --context-length
          - "32768"
          - --disaggregation-ib-device
          - "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
          - --cuda-graph-max-bs
          - "64"
          - --max-running-requests
          - "2048"
          - --tp-size
          - "16" # Size of Tensor Parallelism
          - --dist-init-addr
          - $(LWS_LEADER_ADDRESS):20102
          - --nnodes
          - $(LWS_GROUP_SIZE)
          - --node-rank
          - $(LWS_WORKER_INDEX)
          - --trust-remote-code
          - --ep-num-redundant-experts
          - "32"
          - --moe-dense-tp-size
          - "1"
          env:
          - name: CUDA_LAUNCH_BLOCKING
            value: "0"
          - name: NVSHMEM_IB_GID_INDEX
            value: "3"
          - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
            value: "1"
          - name:  NCCL_IB_QPS_PER_CONNECTION
            value: "8"
          - name: NCCL_IB_SPLIT_DATA_ON_QPS
            value: "1"
          - name: NCCL_NET_PLUGIN
            value: "none"
          - name: NCCL_IB_TC
            value: "136"
          - name: NCCL_MIN_NCHANNELS
            value: "4"
          - name: NCCL_IB_SL
            value: "5"
          - name: MC_TE_METRIC
            value: "true"
          - name: SGLANG_MOONCAKE_TRANS_THREAD
            value: "16"
          - name: SGLANG_ENABLE_JIT_DEEPGEMM
            value: "1"
          - name: NCCL_IB_HCA
            value: ^=mlx5_0,mlx5_5,mlx5_6
          - name: LWS_WORKER_INDEX
            valueFrom:
              fieldRef:
                fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
          image: lmsysorg/sglang:deepep
          name: sglang-leader
          ports:
          - containerPort: 30000
            protocol: TCP
          readinessProbe:
            periodSeconds: 30
            tcpSocket:
              port: 30000
          resources:
            limits:
              nvidia.com/gpu: "8"
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
            privileged: true
          volumeMounts:
          - mountPath: /root/.cache
            name: sgl-cache
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /work/models
            name: model
          - mountPath: /dev/infiniband
            name: ib
          - mountPath: /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs
            name: cf
        dnsPolicy: ClusterFirstWithHostNet
        hostIPC: true
        hostNetwork: true
        nodeSelector:
          pd: "yes"
        tolerations:
        - key: pd
          operator: Exists
        - key: node-role
          operator: Exists
        volumes:
        - hostPath:
            path: /data1/sgl_cache1
            type: DirectoryOrCreate
          name: sgl-cache
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            path: /data1/maas_hosted_models/models/DeepSeek-R1-0528/deepseek_r1_0528
          name: model
        - hostPath:
            path: /dev/infiniband
          name: ib
        - hostPath:
            path: /data1/maas_hosted_models/models/fused_moe_triton/configs
          name: cf
    restartPolicy: RecreateGroupOnPodRestart
    size:  2
    workerTemplate:
      metadata: {}
      spec:
        containers:
        - command:
          - python3
          - -m
          - sglang.launch_server
          - --model-path
          - /work/models
          - --chunked-prefill-size
          - "262144"
          - --page-size
          - "64"
          - --enable-dp-attention
          - --enable-dp-lm-head
            #- --enable-two-batch-overlap
          - --dp-size
          - "16"
          - --moe-a2a-backend
          - deepep
          - --disaggregation-mode
          - decode
          - --mem-fraction-static
          -  "0.849"
          - --context-length
          - "32768"
          - --disaggregation-ib-device
          # should modify according your rdma env
          - "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
          - --cuda-graph-max-bs
          - "64"
          - --max-running-requests
          - "2048"
          - --tp-size
          - "16" # Size of Tensor Parallelism
          - --dist-init-addr
          - $(LWS_LEADER_ADDRESS):20102
          - --nnodes
          - $(LWS_GROUP_SIZE)
          - --node-rank
          - $(LWS_WORKER_INDEX)
          - --trust-remote-code
          - --ep-num-redundant-experts
          - "32"
          - --moe-dense-tp-size
          - "1"
          env:
          - name: SGLANG_HACK_DEEPEP_NUM_SMS
            value: "24"
          - name: SGLANG_HACK_DEEPEP_NEW_MODE
            value: "0"
          - name: NVSHMEM_IB_TRAFFIC_CLASS
            value: "16"
          - name: NVSHMEM_IB_GID_INDEX
            value: "3"
          - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
            value: "1"
          - name:  NCCL_IB_QPS_PER_CONNECTION
            value: "8"
          - name: NCCL_IB_SPLIT_DATA_ON_QPS
            value: "1"
          - name: NCCL_NET_PLUGIN
            value: "none"
          - name: NCCL_IB_TC
            value: "136"
          - name: NCCL_MIN_NCHANNELS
            value: "4"
          - name: MC_TE_METRIC
            value: "true"
          - name: NCCL_IB_SL
            value: "5"
          - name: SGLANG_MOONCAKE_TRANS_THREAD
            value: "16"
          - name: SGLANG_ENABLE_JIT_DEEPGEMM
            value: "1"
          - name: NCCL_IB_HCA
            value: ^=mlx5_0,mlx5_5,mlx5_6
          - name: LWS_WORKER_INDEX
            valueFrom:
              fieldRef:
                fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
          image: lmsysorg/sglang:deepep
          name: sglang-worker
          ports:
          - containerPort: 30001
          resources:
            limits:
              nvidia.com/gpu: "8"
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
            privileged: true
          volumeMounts:
          - mountPath: /root/.cache
            name: sgl-cache
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /work/models
            name: model
          - mountPath: /dev/infiniband
            name: ib
          - mountPath: /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs
            name: cf
        dnsPolicy: ClusterFirstWithHostNet
        hostIPC: true
        hostNetwork: true
        nodeSelector:
          pd: "yes"
        tolerations:
        - key: pd
          operator: Exists
        - key: node-role
          operator: Exists
        volumes:
        - hostPath:
            path: /data1/sgl_cache1
            type: DirectoryOrCreate
          name: sgl-cache
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            path: /dev/infiniband
          name: ib
        - hostPath:
            # modify according to you deployment env
            path: /data1/maas_hosted_models/models/DeepSeek-R1-0528/deepseek_r1_0528
          name: model
        - hostPath:
            # modify according to you deployment env
            path: /data1/maas_hosted_models/models/fused_moe_triton/configs
          name: cf
  networkConfig:
    subdomainPolicy: Shared
  replicas: 1
  rolloutStrategy:
    rollingUpdateConfiguration:
      maxSurge: 0
      maxUnavailable: 1
    type: RollingUpdate
  startupPolicy: LeaderCreated
```

Execute separately:

```bash
kubectl apply -f p.yaml
kubectl apply -f d.yaml
```

At this point, we have completed the deployment of the 1P1D SGlang engine part.

To allow our users to directly experience the model API, we still need a load balancer to handle sequential calls between prefill and decode. Different companies implement LBs differently, and the community will also officially release a new LB component written in Rust in the near future.

Currently, we use a static K8S service + minilb approach to implement model API calls.

### Creating Service for Prefill and Decode

#### Create prefill k8s service
[p-svc.yaml](lws-examples/p-svc.yaml)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: deepseekr10528-prefill-main
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: deepseekr10528-prefill-main
    role: leader
  ports:
    - protocol: TCP
      port: 30000
      targetPort: 30000
```
Execute `kubectl apply -f p-svc.yaml`

#### Create decode k8s service
[d-svc.yaml](lws-examples/d-svc.yaml)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: deepseekr10528-decode-main
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: deepseekr10528-decode-main
    role: leader
  ports:
    - protocol: TCP
      port: 30000
      targetPort: 30000
```
Execute `kubectl apply -f d-svc.yaml`

#### Deploy minilb and lb service
[lb.yaml](lws-examples/lb.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseekr10528-lb-main
  labels:
    app: deepseekr10528-lb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseekr10528-lb
  template:
    metadata:
      labels:
        app: deepseekr10528-lb
    spec:
      nodeSelector:
          pd: "yes"
      tolerations:
        - key: pd
          operator: Exists
        - key: node-role
          operator: Exists
      containers:
        - name: sgl-minilb
          image: lmsysorg/sglang:deepep
          command:
          - python
          - -m
          - sglang_router.launch_router
          - --pd-disaggregation
          - --prefill
          - http://deepseekr10528-prefill-main:30000
          - --decode
          - http://deepseekr10528-decode-main:30000
          - --host
          - 0.0.0.0
          - --port
          -  "8000"
          ports:
            - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: deepseekr10528-lb-service
spec:
  type: NodePort
  selector:
    app: deepseekr10528-lb
  ports:
    - protocol: TCP
      port: 8000         # Service Port（In-Cluster）
      targetPort: 8000   # Exposed Container
      nodePort: 30800
```
Execute `kubectl apply -f lb.yaml`

After waiting for all model deployments to succeed, you will get the following output:

```bash
[root@ecs-001]# kubectl get po
deepseekr10528-decode-main-0             1/1     Running   0          74m
deepseekr10528-decode-main-0-1           1/1     Running   0          74m
deepseekr10528-lb-main-9c5dbfc57-6lcbd   1/1     Running   0          22m
deepseekr10528-prefill-main-0            1/1     Running   0          74m
deepseekr10528-prefill-main-0-1          1/1     Running   0          74m
[root@ecs-cbm-x1-pd-cpu-001 main_doc]# kubectl  get svc |grep dee
deepseekr10528-decode-main    ClusterIP   None             <none>        <none>           97m
deepseekr10528-lb-service     NodePort    172.16.242.169   <none>        8000:30800/TCP   22m
deepseekr10528-prefill-main   ClusterIP   None             <none>        <none>           97m
```

At this point, select a nodePort:30800 to access:

```bash
[root@ecs-001]# curl -X POST "http://{nodePort}:30800/v1/chat/completions" \
>     -H "Content-Type: application/json" \
>     -H "Authorization: Bearer None" \
>     -d '{
>        "rid":"ccccdd",
>         "model": "r1",
>         "messages": [
>             {"role": "system", "content": "0: You are a helpful AI assistant"},
>             {"role": "user", "content": "你是谁？."}
>         ],
>         "max_tokens":221
>     }'
{"id":"ccccdd","object":"chat.completion","created":1750252498,"model":"qwen2","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\n嗯，用户问了一个很基础的自我介绍问题"你是谁？"。这可能是第一次互动时的常规开场白，也可能是想确认我的身份和功能范围。\n\n用户没有提供任何背景信息，语气简洁中性。这种场景下新用户的可能性较高，需要给出清晰友好的自我介绍，同时突出实用价值来降低陌生感。\n\n考虑到中文用户，应该用简体中文回复。重点要说明三点：身份归属（深度求索）、功能定位（AI助手）、服务范围（学习/工作/生活）。结尾用开放性问题引导对话很关键——既能了解需求，又能避免让用户面对空白输入框时不知所措。\n\n用波浪线结尾可以软化语气，那个笑脸表情😊刚好能中和AI的机械感。不过要控制表情符号数量，避免显得轻浮。\n</think>\n你好呀！我是你的AI助手，由深度求索公司（DeepSeek）开发的语言模型，名字叫 **DeepSeek-R1**。你可以把我当成一个知识丰富、随叫随到的小帮手～😊\n\n我的任务就是陪你聊天、解答问题、","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":14,"total_tokens":235,"completion_tokens":221,"prompt_tokens_details":null}}

```
## FAQ

1. The current deployment startup parameters may not be fully compatible with all RDMA scenarios. Different RDMA NCCL-related environment configurations may be needed in different network environments.

2. Some preset, optimized configurations for EPLB are not used here. You can adjust them according to [6017](https://github.com/sgl-project/sglang/issues/6017) as needed.
