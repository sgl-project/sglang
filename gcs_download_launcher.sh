#!/bin/bash
#
# This launcher downloads model files from GCS to local model directory before
# launching the actual command.
#
# If GCS URI is passed as an environment variable, set GCS_URI_ENV_KEY to the
# environment variable name.
# If GCS URI is passed as an argument, set GCS_URI_ARG_KEY to the argument name.
# The argument must be in the format of '--$GCS_URI_ARG_KEY=gs://*'. Do not
# separate argument name and value with spaces.
# This script will also try reading from AIP_STORAGE_URI or AIP_STORAGE_DIR.
# Note that AIP_STORAGE_DIR is expected to be a local path, so it bypasses the
# download process.
#
# Input priority: AIP_STORAGE_DIR > AIP_STORAGE_URI > GCS_URI_ENV_KEY > GCS_URI_ARG_KEY.
# Will output the local model directory to GCS_URI_ENV_KEY and GCS_URI_ARG_KEY
# if they are set. Both will be updated if both set.
#
# Requires google-cloud-sdk as a dependency (for gcloud storage CLI).

set -e

readonly LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-"/tmp/model_dir"}

update_model_id() {
  if [[ ! -z "$GCS_URI_ENV_KEY" ]]; then
    echo "Updating env var $GCS_URI_ENV_KEY to $AIP_STORAGE_DIR."
    export "$GCS_URI_ENV_KEY"="$AIP_STORAGE_DIR"
  fi

  if [[ ! -z "$GCS_URI_ARG_KEY" ]]; then
    echo "Updating args $GCS_URI_ARG_KEY to $AIP_STORAGE_DIR."
    updated=0
    for (( i=1; i <= $#; i++)); do
      arg="${!i}"
      if [[ "$arg" == "--$GCS_URI_ARG_KEY="* ]]; then
        echo "Found $arg, updating to $AIP_STORAGE_DIR."
        set -- "${@:1:(($i-1))}" "--$GCS_URI_ARG_KEY=$AIP_STORAGE_DIR" "${@:$(($i+1))}";
        updated=1
        break
      fi
    done
    if [[ $updated -eq 0 ]]; then
      echo "Appending args $GCS_URI_ARG_KEY to $AIP_STORAGE_DIR."
      set -- "$@" "--$GCS_URI_ARG_KEY=$AIP_STORAGE_DIR";
    fi
  fi
  LAUNCH_CMD=("$@")
}

maybe_download_model() {
  if [[ -z "$GCS_URI_ENV_KEY" ]] && [[ -z "$GCS_URI_ARG_KEY" ]]; then
    echo "Internal error: Required GCS_URI_ENV_KEY or GCS_URI_ARG_KEY."
    exit 1
  fi

  LAUNCH_CMD=("$@")
  gcs_uri=""
  if [[ ! -z "$AIP_STORAGE_DIR" ]]; then
    # AIP_STORAGE_DIR is expected to be a local path.
    echo "AIP_STORAGE_DIR set, proceeding to run the launcher."
    update_model_id "$@"
    return
  elif [[ $AIP_STORAGE_URI == gs://* ]]; then
    # Check AIP_STORAGE_URI environment variable.
    echo "AIP_STORAGE_URI set and starts with 'gs://', proceeding to download from GCS."
    gcs_uri="$AIP_STORAGE_URI"
  elif [[ ! -z "$GCS_URI_ENV_KEY" ]] && [[ ${!GCS_URI_ENV_KEY} == gs://* ]]; then
    # Check custom environment variable.
    echo "Custom environment variable ${GCS_URI_ENV_KEY} set and starts with 'gs://', proceeding to download from GCS."
    gcs_uri="${!GCS_URI_ENV_KEY}"
  elif [[ ! -z "$GCS_URI_ARG_KEY" ]]; then
    # Check custom args.
    for arg in "$@"; do
      if [[ "$arg" == "--$GCS_URI_ARG_KEY=gs://"* ]]; then
        gcs_uri="${arg#*=}"
        echo "Custom args ${GCS_URI_ARG_KEY} set and starts with 'gs://', proceeding to download from GCS."
        break
      elif [[ "$arg" == "--$GCS_URI_ARG_KEY" ]]; then
        echo "Found $GCS_URI_ARG_KEY, but it's not in the format of '--$GCS_URI_ARG_KEY=gs://*'."
        echo "Ensure the value of $GCS_URI_ARG_KEY is within the same arg, separated by '='."
        exit 1
      fi
    done
  fi

  if [[ -z "$gcs_uri" ]]; then
    echo "No GCS URI found, proceeding to run the launcher."
    return
  fi

  # Remove trailing '/' if any.
  gcs_uri="${gcs_uri%%/}"
  export AIP_STORAGE_DIR="$LOCAL_MODEL_DIR/${gcs_uri##gs://}"

  # Create the target directory.
  mkdir -p "$AIP_STORAGE_DIR"
  echo "Downloading model from ${gcs_uri} to ${AIP_STORAGE_DIR}."

  # Use gcloud storage CLI to copy the content from GCS to the target directory.
  if gcloud storage cp -r "$gcs_uri/*" "$AIP_STORAGE_DIR"; then
    echo "Model downloaded successfully to ${AIP_STORAGE_DIR}."
    update_model_id "$@"
  else
    echo "Failed to download model from GCS."
    exit 1
  fi
}

prefecth_model() {
  local model_val="$1"
  find "$model_val" -name '*safetensors' -type f -print0 | \
  xargs -0 -r -I {} -P 0 sh -c 'echo "$(date) - Prefetching weight: {}"; dd if="{}" of=/dev/null' &
}

maybe_prefetch_model() {
  if [[ ! -z "$VERTEX_MODEL_LOADER" ]]; then
    for (( i=1; i <= $#; i++)); do
      arg="${!i}"
      if [[ "$arg" == "--$GCS_URI_ARG_KEY="* ]]; then
        model_val=${arg#"--$GCS_URI_ARG_KEY="}
        model_val=${model_val%/}
        if [[ ! -d "$model_val" ]]; then
          break
        fi
        echo "Found $arg, updating prefetch command."
        prefecth_model "$model_val"
        break
      fi
    done
  fi
}

run_local_command() {
  echo "Launch command: " "${LAUNCH_CMD[@]}"
  "${LAUNCH_CMD[@]}"
}

maybe_download_model "$@"
maybe_prefetch_model "${LAUNCH_CMD[@]}"
run_local_command
