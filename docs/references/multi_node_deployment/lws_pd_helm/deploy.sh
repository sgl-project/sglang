#!/bin/bash

# SGLang PD Helm Deployment Script
# Usage: ./deploy.sh [install|upgrade|uninstall] [release-name] [namespace]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CHART_DIR="$SCRIPT_DIR/sglang-leaderworkerset"

# Default parameters
ACTION="${1:-install}"
RELEASE_NAME="${2:-sglang-leaderworkerset}"
NAMESPACE="${3:-default}"
VALUES_FILE="${4:-}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check k8s connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Unable to connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check LWS CRD
    if ! kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io &> /dev/null; then
        log_error "LeaderWorkerSet CRD not installed, please install LWS Operator first"
        log_info "Install command: kubectl apply -f https://github.com/kubernetes-sigs/lws/releases/download/v0.3.0/manifests.yaml"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

validate_values() {
    log_info "Validating configuration file..."
    
    if [[ -n "$VALUES_FILE" && ! -f "$VALUES_FILE" ]]; then
        log_error "Configuration file $VALUES_FILE does not exist"
        exit 1
    fi
    
    # Use helm template validation
    local template_cmd_arr=(helm template "$RELEASE_NAME" "$CHART_DIR")
    if [[ -n "$VALUES_FILE" ]]; then
        template_cmd_arr+=("-f" "$VALUES_FILE")
    fi
    
    if ! "${template_cmd_arr[@]}" --dry-run &> /dev/null; then
        log_error "Helm template validation failed"
        exit 1
    fi
    
    log_info "Configuration file validation passed"
}

install_chart() {
    log_info "Installing SGLang PD Chart..."
    
    local install_cmd_arr=(helm install "$RELEASE_NAME" "$CHART_DIR")
    
    if [[ "$NAMESPACE" != "default" ]]; then
        install_cmd_arr+=("--namespace" "$NAMESPACE" "--create-namespace")
    fi
    
    if [[ -n "$VALUES_FILE" ]]; then
        install_cmd_arr+=("-f" "$VALUES_FILE")
    fi
    
    log_info "Executing command: ${install_cmd_arr[*]}"
    
    if "${install_cmd_arr[@]}"; then
        log_info "Installation successful!"
        show_post_install_info
    else
        log_error "Installation failed"
        exit 1
    fi
}

upgrade_chart() {
    log_info "Upgrading SGLang PD Chart..."
    
    local upgrade_cmd_arr=(helm upgrade "$RELEASE_NAME" "$CHART_DIR")
    
    if [[ "$NAMESPACE" != "default" ]]; then
        upgrade_cmd_arr+=("--namespace" "$NAMESPACE")
    fi
    
    if [[ -n "$VALUES_FILE" ]]; then
        upgrade_cmd_arr+=("-f" "$VALUES_FILE")
    fi
    
    log_info "Executing command: ${upgrade_cmd_arr[*]}"
    
    if "${upgrade_cmd_arr[@]}"; then
        log_info "Upgrade successful!"
        show_post_install_info
    else
        log_error "Upgrade failed"
        exit 1
    fi
}

uninstall_chart() {
    log_info "Uninstalling SGLang PD Chart..."
    
    local uninstall_cmd_arr=(helm uninstall "$RELEASE_NAME")
    
    if [[ "$NAMESPACE" != "default" ]]; then
        uninstall_cmd_arr+=("--namespace" "$NAMESPACE")
    fi
    
    log_info "Executing command: ${uninstall_cmd_arr[*]}"
    
    if "${uninstall_cmd_arr[@]}"; then
        log_info "Uninstallation successful!"
    else
        log_error "Uninstallation failed"
        exit 1
    fi
}

show_post_install_info() {
    log_info "Post-deployment information:"
    echo
    echo "1. Check deployment status:"
    echo "   kubectl get leaderworkerset -n $NAMESPACE"
    echo "   kubectl get pods -n $NAMESPACE"
    echo "   kubectl get svc -n $NAMESPACE"
    echo
    echo "2. View detailed information:"
    echo "   helm status $RELEASE_NAME -n $NAMESPACE"
    echo
    echo "3. View logs:"
    echo "   kubectl logs -l app.kubernetes.io/instance=$RELEASE_NAME -n $NAMESPACE"
    echo
}

show_usage() {
    echo "Usage: $0 [ACTION] [RELEASE_NAME] [NAMESPACE] [VALUES_FILE]"
    echo
    echo "Parameters:"
    echo "  ACTION       - Operation type: install, upgrade, uninstall (default: install)"
    echo "  RELEASE_NAME - Helm release name (default: sglang-leaderworkerset)" 
    echo "  NAMESPACE    - Kubernetes namespace (default: default)"
    echo "  VALUES_FILE  - Custom values file path (optional)"
    echo
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 install my-sglang sglang-system"
    echo "  $0 install my-sglang sglang-system ./values-prod.yaml"
    echo "  $0 upgrade my-sglang sglang-system ./values-prod.yaml"
    echo "  $0 uninstall my-sglang sglang-system"
}

main() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    check_prerequisites
    
    case "$ACTION" in
        install)
            validate_values
            install_chart
            ;;
        upgrade)
            validate_values
            upgrade_chart
            ;;
        uninstall)
            uninstall_chart
            ;;
        *)
            log_error "Unknown operation: $ACTION"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
