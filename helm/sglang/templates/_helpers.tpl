{{/*
Expand the name of the chart.
*/}}
{{- define "sglang.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sglang.fullname" -}}
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
{{- define "sglang.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sglang.labels" -}}
helm.sh/chart: {{ include "sglang.chart" . }}
{{ include "sglang.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sglang.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sglang.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sglang.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sglang.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "sglang.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}

{{/*
Return image pull secrets
*/}}
{{- define "sglang.imagePullSecrets" -}}
{{- if .Values.image.pullSecrets }}
imagePullSecrets:
{{- range .Values.image.pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the HuggingFace secret name
*/}}
{{- define "sglang.hfSecretName" -}}
{{- if .Values.hfSecret.existingSecret }}
{{- .Values.hfSecret.existingSecret }}
{{- else if .Values.hfSecret.externalSecret.secretStoreRef.name }}
{{- printf "%s-hf-token" (include "sglang.fullname" .) }}
{{- else }}
{{- printf "%s-hf-token" (include "sglang.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Determine if we should create a secret for HF token
*/}}
{{- define "sglang.createHfSecret" -}}
{{- if and (not .Values.hfSecret.existingSecret) (not .Values.hfSecret.externalSecret.secretStoreRef.name) .Values.hfSecret.token }}
{{- true }}
{{- end }}
{{- end }}

{{/*
Determine if we should create an external secret for HF token
*/}}
{{- define "sglang.createExternalSecret" -}}
{{- if and (not .Values.hfSecret.existingSecret) .Values.hfSecret.externalSecret.secretStoreRef.name }}
{{- true }}
{{- end }}
{{- end }}
