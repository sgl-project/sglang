{{/*
Expand the name of the chart.
*/}}
{{- define "sglang-leaderworkerset.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "sglang-leaderworkerset.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sglang-leaderworkerset.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sglang-leaderworkerset.labels" -}}
helm.sh/chart: {{ include "sglang-leaderworkerset.chart" . }}
{{ include "sglang-leaderworkerset.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sglang-leaderworkerset.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sglang-leaderworkerset.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common SGLang command arguments for prefill leader
*/}}
{{- define "sglang-leaderworkerset.prefill.leader.command" -}}
- python3
- -m
- sglang.launch_server
- --port
- {{ .Values.prefill.config.port | quote }}
- --host
- {{ .Values.prefill.config.host | quote }}
- --model-path
- {{ .Values.global.model.path }}
- --disaggregation-ib-device
- {{ .Values.global.rdma.ibDevice }}
- --chunked-prefill-size
- {{ .Values.prefill.config.chunkedPrefillSize | quote }}
- --max-prefill-tokens
- {{ .Values.prefill.config.maxPrefillTokens | quote }}
- --page-size
- {{ .Values.prefill.config.pageSize | quote }}
- --ep-dispatch-algorithm
- {{ .Values.prefill.config.epDispatchAlgorithm }}
- --eplb-algorithm
- {{ .Values.prefill.config.eplbAlgorithm }}
{{- if .Values.prefill.config.enableDpLmHead }}
- --enable-dp-lm-head
{{- end }}
{{- if .Values.prefill.config.enableDpAttention }}
- --enable-dp-attention
{{- end }}
- --dp-size
- {{ .Values.prefill.config.dpSize | quote }}
{{- if .Values.prefill.config.disableRadixCache }}
- --disable-radix-cache
{{- end }}
- --moe-a2a-backend
- {{ .Values.prefill.config.moeA2aBackend }}
- --disaggregation-mode
- {{ .Values.prefill.config.disaggregationMode }}
- --mem-fraction-static
- {{ .Values.prefill.config.memFractionStatic | quote }}
- --context-length
- {{ .Values.global.model.contextLength | quote }}
- --tp
- {{ .Values.prefill.config.tp | quote }}
- --dist-init-addr
- $(LWS_LEADER_ADDRESS):{{ .Values.global.distInit.port }}
- --nnodes
- $(LWS_GROUP_SIZE)
- --node-rank
- $(LWS_WORKER_INDEX)
{{- if .Values.global.model.trustRemoteCode }}
- --trust-remote-code
{{- end }}
- --ep-num-redundant-experts
- {{ .Values.prefill.config.epNumRedundantExperts | quote }}
- --moe-dense-tp-size
- {{ .Values.prefill.config.moeDenseTpSize | quote }}
- --max-running-requests
- {{ .Values.prefill.config.maxRunningRequests | quote }}
{{- end }}

{{/*
Common SGLang command arguments for decode leader
*/}}
{{- define "sglang-leaderworkerset.decode.leader.command" -}}
- python3
- -m
- sglang.launch_server
- --port
- {{ .Values.decode.config.port | quote }}
- --host
- {{ .Values.decode.config.host | quote }}
- --model-path
- {{ .Values.global.model.path }}
- --chunked-prefill-size
- {{ .Values.decode.config.chunkedPrefillSize | quote }}
- --page-size
- {{ .Values.decode.config.pageSize | quote }}
{{- if .Values.decode.config.enableDpAttention }}
- --enable-dp-attention
{{- end }}
{{- if .Values.decode.config.enableDpLmHead }}
- --enable-dp-lm-head
{{- end }}
- --dp-size
- {{ .Values.decode.config.dpSize | quote }}
- --moe-a2a-backend
- {{ .Values.decode.config.moeA2aBackend }}
- --disaggregation-mode
- {{ .Values.decode.config.disaggregationMode }}
- --mem-fraction-static
- {{ .Values.decode.config.memFractionStatic | quote }}
- --context-length
- {{ .Values.global.model.contextLength | quote }}
- --disaggregation-ib-device
- {{ .Values.global.rdma.ibDevice | quote }}
- --cuda-graph-max-bs
- {{ .Values.decode.config.cudaGraphMaxBs | quote }}
- --max-running-requests
- {{ .Values.decode.config.maxRunningRequests | quote }}
- --tp-size
- {{ .Values.decode.config.tpSize | quote }}
- --dist-init-addr
- $(LWS_LEADER_ADDRESS):{{ .Values.global.distInit.port }}
- --nnodes
- $(LWS_GROUP_SIZE)
- --node-rank
- $(LWS_WORKER_INDEX)
{{- if .Values.global.model.trustRemoteCode }}
- --trust-remote-code
{{- end }}
- --ep-num-redundant-experts
- {{ .Values.decode.config.epNumRedundantExperts | quote }}
- --moe-dense-tp-size
- {{ .Values.decode.config.moeDenseTpSize | quote }}
{{- end }}

{{/*
Common SGLang command arguments for decode worker
*/}}
{{- define "sglang-leaderworkerset.decode.worker.command" -}}
- python3
- -m
- sglang.launch_server
- --model-path
- {{ .Values.global.model.path }}
- --chunked-prefill-size
- {{ .Values.decode.config.chunkedPrefillSize | quote }}
- --page-size
- {{ .Values.decode.config.pageSize | quote }}
{{- if .Values.decode.config.enableDpAttention }}
- --enable-dp-attention
{{- end }}
{{- if .Values.decode.config.enableDpLmHead }}
- --enable-dp-lm-head
{{- end }}
- --dp-size
- {{ .Values.decode.config.dpSize | quote }}
- --moe-a2a-backend
- {{ .Values.decode.config.moeA2aBackend }}
- --disaggregation-mode
- {{ .Values.decode.config.disaggregationMode }}
- --mem-fraction-static
- {{ .Values.decode.config.memFractionStatic | quote }}
- --context-length
- {{ .Values.global.model.contextLength | quote }}
- --disaggregation-ib-device
- {{ .Values.global.rdma.ibDevice | quote }}
- --cuda-graph-max-bs
- {{ .Values.decode.config.cudaGraphMaxBs | quote }}
- --max-running-requests
- {{ .Values.decode.config.maxRunningRequests | quote }}
- --tp-size
- {{ .Values.decode.config.tpSize | quote }}
- --dist-init-addr
- $(LWS_LEADER_ADDRESS):{{ .Values.global.distInit.port }}
- --nnodes
- $(LWS_GROUP_SIZE)
- --node-rank
- $(LWS_WORKER_INDEX)
{{- if .Values.global.model.trustRemoteCode }}
- --trust-remote-code
{{- end }}
- --ep-num-redundant-experts
- {{ .Values.decode.config.epNumRedundantExperts | quote }}
- --moe-dense-tp-size
- {{ .Values.decode.config.moeDenseTpSize | quote }}
{{- end }}

{{/*
Common SGLang command arguments for prefill worker
*/}}
{{- define "sglang-leaderworkerset.prefill.worker.command" -}}
- python3
- -m
- sglang.launch_server
- --model-path
- {{ .Values.global.model.path }}
- --disaggregation-ib-device
- {{ .Values.global.rdma.ibDevice }}
- --chunked-prefill-size
- {{ .Values.prefill.config.chunkedPrefillSize | quote }}
- --max-prefill-tokens
- {{ .Values.prefill.config.maxPrefillTokens | quote }}
- --page-size
- {{ .Values.prefill.config.pageSize | quote }}
- --ep-dispatch-algorithm
- {{ .Values.prefill.config.epDispatchAlgorithm }}
- --eplb-algorithm
- {{ .Values.prefill.config.eplbAlgorithm }}
{{- if .Values.prefill.config.enableDpLmHead }}
- --enable-dp-lm-head
{{- end }}
{{- if .Values.prefill.config.enableDpAttention }}
- --enable-dp-attention
{{- end }}
- --dp-size
- {{ .Values.prefill.config.dpSize | quote }}
{{- if .Values.prefill.config.disableRadixCache }}
- --disable-radix-cache
{{- end }}
- --moe-a2a-backend
- {{ .Values.prefill.config.moeA2aBackend }}
- --disaggregation-mode
- {{ .Values.prefill.config.disaggregationMode }}
- --mem-fraction-static
- {{ .Values.prefill.config.memFractionStatic | quote }}
- --context-length
- {{ .Values.global.model.contextLength | quote }}
- --tp
- {{ .Values.prefill.config.tp | quote }}
- --dist-init-addr
- $(LWS_LEADER_ADDRESS):{{ .Values.global.distInit.port }}
- --nnodes
- $(LWS_GROUP_SIZE)
- --node-rank
- $(LWS_WORKER_INDEX)
{{- if .Values.global.model.trustRemoteCode }}
- --trust-remote-code
{{- end }}
- --ep-num-redundant-experts
- {{ .Values.prefill.config.epNumRedundantExperts | quote }}
- --moe-dense-tp-size
- {{ .Values.prefill.config.moeDenseTpSize | quote }}
- --max-running-requests
- {{ .Values.prefill.config.maxRunningRequests | quote }}
{{- end }}
