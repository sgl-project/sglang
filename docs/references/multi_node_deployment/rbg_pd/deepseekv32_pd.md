# DeepSeekV32-Exp RBG Based PD Deploy

## 0. Prerequisites

1. k8s >=1.26
2. lws installed on k8s.
3. rbg installed on k8s.

For RBG installation, please refer to: https://github.com/sgl-project/rbg

## 1. Image Preparation

`lmsysorg/sglang:latest`


### 2. All In One manifest file

*Note: The NodeSelector section, model location section, and taint toleration section can be adjusted according to your actual deployment environment*

rbg-dsv32.yml

```yaml
apiVersion: workloads.x-k8s.io/v1alpha1
kind: RoleBasedGroup
metadata:
  name: deepseek-rbg-32exp
  namespace: default
spec:
  roles:
    - name: prefill
      replicas: 1
      workload:
        apiVersion: leaderworkerset.x-k8s.io/v1
        kind: LeaderWorkerSet
      restartPolicy: None
      leaderWorkerSet:
        size: 1
        patchLeaderTemplate:
          metadata:
            labels:
              role: leader
              pd_role: prefill
          spec:
            containers:
            - command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --port
              - "30000"
              - --trust-remote
              - --host
              -  0.0.0.0
              - --disaggregation-ib-device
              -  mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
              - --disable-radix-cache
              - --chunked-prefill-size
              - "131072"
              - --page-size
              - "64"
    #          - --enable-eplb
              - --ep-dispatch-algorithm
              - dynamic
              - --eplb-algorithm
              - deepseek
              - --enable-dp-lm-head
              - --enable-dp-attention
              - --dp-size
              - "8"
              - --moe-a2a-backend
              - deepep
              - --deepep-mode
              - normal
              - --disaggregation-mode
              - prefill
              - --mem-fraction-static
              - "0.8"
              - --max-prefill-tokens
              - "32768"
              - --context-length
              - "32768"
              - --tp
              - "8"
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
              - name: LWS_WORKER_INDEX
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
              livenessProbe:
                failureThreshold: 3000
                httpGet:
                  path: /health
                  port: 30000
                initialDelaySeconds: 300
                periodSeconds: 60
                successThreshold: 1
                timeoutSeconds: 10
              readinessProbe:
                failureThreshold: 20
                httpGet:
                  path: /health
                  port: 30000
                periodSeconds: 30
                successThreshold: 1
                timeoutSeconds: 10
              name: sglang
              ports:
              - containerPort: 30000
                name: sglang-http
                protocol: TCP

        patchWorkerTemplate: {}
      template:
        metadata:
          labels:
            inference-framework: sglang
            inference-stack.io/monitoring: "enabled"
        spec:
            containers:
            - name: sglang
              image: lmsysorg/sglang:latest
              env:
                - name: SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
                  value: "1"
                - name: CUDA_LAUNCH_BLOCKING
                  value: "0"
                - name:  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT
                  value: "1000000000"
                - name: NVSHMEM_IB_TRAFFIC_CLASS
                  value: "16"
                - name: NVSHMEM_DISABLE_P2P
                  value: "0"
                - name: ENABLE_METRICS
                  value: "true"
                - name: NVSHMEM_IB_GID_INDEX
                  value: "3"
                - name: NVSHMEM_IB_SL
                  value: "5"
                - name: SGLANG_SET_CPU_AFFINITY
                  value: "true"
                - name: SGL_ENABLE_JIT_DEEPGEMM
                  value: "1"
                - name:  NCCL_IB_QPS_PER_CONNECTION
                  value: "8"
                - name: NCCL_IB_SPLIT_DATA_ON_QPS
                  value: "1"
                - name: NCCL_NET_PLUGIN
                  value: "none"
                - name: NCCL_IB_TC
                  value: "136"
                - name: NCCL_IB_SL
                  value: "5"
                - name: NCCL_IB_TIMEOUT
                  value: "22"
                - name: NCCL_IB_GID_INDEX
                  value: "3"
                - name: NCCL_MIN_NCHANNELS
                  value: "4"
                - name: NCCL_SOCKET_IFNAME
                  value: bond0
                - name: GLOO_SOCKET_IFNAME
                  value: bond0
                - name: NCCL_IB_HCA
                  value: ^=mlx5_0,mlx5_5,mlx5_6
                - name: NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME
                  value: "bond0"
                - name: MC_TE_METRIC
                  value: "false"
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
                - mountPath: /sgl-workspace/sglang
                  name: src

            dnsPolicy: ClusterFirstWithHostNet
            hostIPC: true
            hostNetwork: true
            nodeSelector:
              pd: "yes"
            tolerations:
              - key: pd
                operator: Exists
            volumes:
            - hostPath:
                path: /var/run/sys-topology
              name: topo
            - hostPath:
                path: /data1/sgl_cache4
                type: DirectoryOrCreate
              name: sgl-cache
            - emptyDir:
                medium: Memory
              name: dshm
            - hostPath:
                path: /data/DeepSeek-V3.2-Exp
              name: model
            - hostPath:
                path: /dev/infiniband
              name: ib
            - hostPath:
                path: /data/src/sglang
                type: DirectoryOrCreate
              name: src

    - name: decode
      replicas: 1
      workload:
        apiVersion: leaderworkerset.x-k8s.io/v1
        kind: LeaderWorkerSet
      leaderWorkerSet:
        size: 1
        patchLeaderTemplate:
          metadata:
            labels:
              role: leader
              pd_role: decode
          spec:
            containers:
            - command:
                  - python3
                  - -m
                  - sglang.launch_server
                  - --model-path
                  - /work/models
                  - --port
                  - "30000"
                  - --trust-remote
                  - --host
                  -  0.0.0.0
                  - --disaggregation-ib-device
                  -  mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
                  - --chunked-prefill-size
                  - "131072"
                  - --eplb-rebalance-layers-per-chunk
                  - "29"
                  - --page-size
                  - "64"
                  - --enable-dp-attention
                  - --enable-dp-lm-head
                  - --dp-size
                  - "8"
                  - --moe-a2a-backend
                  - deepep
                  - --deepep-mode
                  - low_latency
                  - --disaggregation-mode
                  - decode
                  - --mem-fraction-static
                  -  "0.8"
                  - --context-length
                  - "32768"
                  - --max-running-requests
                  - "2048"
                  - --tp-size
                  - "8" # Size of Tensor Parallelism
                  - --cuda-graph-max-bs
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
              env:
              - name: LWS_WORKER_INDEX
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
              livenessProbe:
                failureThreshold: 30000
                httpGet:
                  path: /health
                  port: 30000
                initialDelaySeconds: 300
                periodSeconds: 60
                successThreshold: 1
                timeoutSeconds: 10
              name: sglang
              readinessProbe:
                failureThreshold: 20
                httpGet:
                  path: /health
                  port: 30000
                periodSeconds: 30
                successThreshold: 1
                timeoutSeconds: 10
        patchWorkerTemplate:
          spec:
            containers:
            - command:
                - python3
                - -m
                - sglang.launch_server
                - --model-path
                - /work/models
                - --crash-dump-folder
                -  /log
                - --chunked-prefill-size
                - "262144"
                - --eplb-rebalance-layers-per-chunk
                - "29"
                - --page-size
                - "64"
                - --enable-dp-attention
                - --enable-dp-lm-head
                - --dp-size
                - "32"
                - --moe-a2a-backend
                - "deepep"
                - --deepep-mode
                - low_latency
                - --disaggregation-mode
                - decode
                - --mem-fraction-static
                -  "0.849"
                - --context-length
                - "32768"
                - --disaggregation-ib-device
                -  mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
                - --max-running-requests
                - "4096"
                - --cuda-graph-max-bs
                - "16"
                - --tp-size
                - "8" # Size of Tensor Parallelism
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
              - name: LWS_WORKER_INDEX
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']
              name: sglang
      template:
        metadata:
          labels:
            inference-framework: sglang-unuse
            inference-stack.io/monitoring: "enabled"
        spec:
            containers:
            - image: lmsysorg/sglang:latest
              name: sglang
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
                - mountPath: /sgl-workspace/sglang
                  name: src
              env:
                - name: SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
                  value: "1"
                - name: SGLANG_DISAGGREGATION_WAITING_TIMEOUT
                  value: "100000000"
                - name: NVSHMEM_DISABLE_P2P
                  value: "0"
                - name: NVSHMEM_IB_TRAFFIC_CLASS
                  value: "16"
                - name: NVSHMEM_IB_SL
                  value: "5"
                - name: ENABLE_METRICS
                  value: "true"
                - name: CUDA_LAUNCH_BLOCKING
                  value: "0"
                - name: NVSHMEM_IB_GID_INDEX
                  value: "3"
                - name:  NCCL_IB_QPS_PER_CONNECTION
                  value: "8"
                - name: NCCL_IB_SPLIT_DATA_ON_QPS
                  value: "1"
                - name: NCCL_NET_PLUGIN
                  value: "none"
                - name: NCCL_IB_TC
                  value: "136"
                - name: NCCL_IB_SL
                  value: "5"
                - name: NCCL_IB_TIMEOUT
                  value: "22"
                - name: NCCL_IB_GID_INDEX
                  value: "3"
                - name: NCCL_MIN_NCHANNELS
                  value: "4"
                - name: NCCL_SOCKET_IFNAME
                  value: bond0
                - name: GLOO_SOCKET_IFNAME
                  value: bond0
                - name: NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME
                  value: "bond0"
                - name: NCCL_IB_HCA
                  value: ^=mlx5_0,mlx5_5,mlx5_6
                - name: MC_TE_METRIC
                  value: "false"
                - name: SGL_ENABLE_JIT_DEEPGEMM
                  value: "1"
            dnsPolicy: ClusterFirstWithHostNet
            hostIPC: true
            hostNetwork: true
            nodeSelector:
              pd: "yes"
            tolerations:
            - key: pd
              operator: Exists
            volumes:
            - hostPath:
                path: /var/run/sys-topology
              name: topo
            - hostPath:
                path: /data1/sgl_cache4
                type: DirectoryOrCreate
              name: sgl-cache
            - hostPath:
                path: /data/src/sglang
                type: DirectoryOrCreate
              name: src
            - emptyDir:
                medium: Memory
              name: dshm
            - hostPath:
                path: /data/DeepSeek-V3.2-Exp
              name: model
            - hostPath:
                path: /dev/infiniband
              name: ib
    - name: router
      replicas: 1
      dependencies: [ "decode", "prefill" ]
      template:
        spec:
          containers:
            - name: scheduler
              image: lmsysorg/sglang:latest
              command:
              - sh
              - -c
              - >
                python3 -m sglang_router.launch_router
                --host 0.0.0.0
                --port 8080
                --pd-disaggregation
                --policy random
                --service-discovery
                --service-discovery-namespace ${NAMESPACE}
                --service-discovery-port 30000
                --prefill-selector pd_role=prefill
                --decode-selector pd_role=decode
                --max-payload-size 2147483648
                --worker-startup-timeout-secs 1200
              env:
              - name: NAMESPACE
                valueFrom:
                  fieldRef:
                    apiVersion: v1
                    fieldPath: metadata.namespace
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: deepseek-rbg-32exp
  name: deepseek-rbg-32exp
  namespace: default
spec:
  ports:
    - name: http
      port: 8080
      protocol: TCP
      targetPort: 8080
      nodePort: 30080

  selector:
    rolebasedgroup.workloads.x-k8s.io/name: deepseek-rbg-32exp
    rolebasedgroup.workloads.x-k8s.io/role: router
  type: NodePort

```

```bash
[root@ecs-001]# kubectl get po -n default
deepseek-rbg-32exp-decode-main-0             1/1     Running   0          74m
deepseek-rbg-32exp-decode-0-1                1/1     Running   0          74m
deepseek-rbg-32exp-router-9c5dbfc57          1/1     Running   0          22m
deepseek-rbg-32exp-prefill-0                 1/1     Running   0          74m

[root@ecs-cbm-x1-pd-cpu-001 main_doc]# kubectl  get svc |grep dee
deepseek-rbg-32exp-decode             ClusterIP   None             <none>        <none>           97m
deepseek-rbg-32exp-router-service     NodePort    172.16.242.169   <none>        8000:30800/TCP   22m
deepseek-rbg-32exp-prefill            ClusterIP   None             <none>        <none>           97m
```

At this point, select a nodePort:30800 to access:

```bash
[root@ecs-001]# curl -X POST "http://{nodePort}:30800/v1/chat/completions" \
>     -H "Content-Type: application/json" \
>     -H "Authorization: Bearer None" \
>     -d '{
>        "rid":"ccccdd",
>         "model": "dsv32",
>         "messages": [
>             {"role": "system", "content": "0: You are a helpful AI assistant"},
>             {"role": "user", "content": "你是谁？."}
>         ],
>         "max_tokens":221
>     }'
{"id":"ccccdd","object":"chat.completion","created":1750252498,"model":"qwen2","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\n嗯，用户问了一个很基础的自我介绍问题"你是谁？"。这可能是第一次互动时的常规开场白，也可能是想确认我的身份和功能范围。\n\n用户没有提供任何背景信息，语气简洁中性。这种场景下新用户的可能性较高，需要给出清晰友好的自我介绍，同时突出实用价值来降低陌生感。\n\n考虑到中文用户，应该用简体中文回复。重点要说明三点：身份归属（深度求索）、功能定位（AI助手）、服务范围（学习/工作/生活）。结尾用开放性问题引导对话很关键——既能了解需求，又能避免让用户面对空白输入框时不知所措。\n\n用波浪线结尾可以软化语气，那个笑脸表情😊刚好能中和AI的机械感。不过要控制表情符号数量，避免显得轻浮。\n</think>\n你好呀！我是你的AI助手，由深度求索公司（DeepSeek）开发的语言模型，名字叫 **DeepSeek-V32**。你可以把我当成一个知识丰富、随叫随到的小帮手～😊\n\n我的任务就是陪你聊天、解答问题、","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":14,"total_tokens":235,"completion_tokens":221,"prompt_tokens_details":null}}

```
## FAQ

1. The current deployment startup parameters may not be fully compatible with all RDMA scenarios. Different RDMA NCCL-related environment configurations may be needed in different network environments.

2. Please ensure that the sglang code in the image has incorporated the changes from [PR #10912](https://github.com/sgl-project/sglang/pull/10912).
