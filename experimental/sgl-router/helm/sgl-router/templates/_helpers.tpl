{{/*
Expand the name of the chart.
*/}}
{{- define "sgl-router.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name. Used for Deployment + Service names.
*/}}
{{- define "sgl-router.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels — recommended Kubernetes labels per
https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/
*/}}
{{- define "sgl-router.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "sgl-router.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels — narrower subset used in Deployment.spec.selector +
Service.spec.selector. Must NOT include any label that mutates between
releases (helm.sh/chart, app.kubernetes.io/version).
*/}}
{{- define "sgl-router.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sgl-router.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Resolve the pod's service account name.
*/}}
{{- define "sgl-router.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "sgl-router.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Resolve the Service's `spec.ports[0].port`. Defaults to
`config.server.port` so a `service.port: ""` (the chart default) keeps
the LB-facing port in sync with the container port. An operator that
needs a different externally-visible port (e.g. an L4 LB that rewrites
80 → 8090) sets `service.port: 80` explicitly.
*/}}
{{- define "sgl-router.servicePort" -}}
{{- $svcPort := .Values.service.port -}}
{{- if or (eq (printf "%v" $svcPort) "") (kindIs "invalid" $svcPort) -}}
{{- .Values.config.server.port -}}
{{- else -}}
{{- $svcPort -}}
{{- end -}}
{{- end -}}
