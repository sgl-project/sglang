#!/bin/bash
# Assemble the streaming SANA-WM model dir for the sglang diffusers runtime.
#
# test-anas-smoke is NOT a turnkey diffusers pipeline — it ships only raw artifacts
# (a DMD .pt DiT, a causal VAE, refiner_diffusers, gemma3_12b, a yaml). This script
# builds a runnable pipeline dir, sourcing each component from its AUTHORITATIVE place
# (verified against minimal-sanawm infer.py / pipeline.py):
#
#   DiT weights         -> test-anas-smoke sana_dit/model.pt  (converted to safetensors;
#                          see convert_streaming_dit.py — bit-identical to the reference)
#   VAE                 -> test-anas-smoke ltx2_causal_vae
#   stage-1 text encoder-> Efficient-Large-Model/gemma-2-2b-it  (bit-identical to the
#                          bidirectional model's text_encoder, so we reuse that)
#   model_index/config/scheduler -> bidirectional materialized dir (generic / same arch)
#   refiner transformer + connectors -> test-anas-smoke refiner_diffusers   <-- the fix
#   refiner text encoder + tokenizer -> test-anas-smoke gemma3_12b          <-- the fix
#
# The refiner had been wrongly pointed at the BIDIRECTIONAL refiner (3503/3510 tensors
# differ); the streaming refiner is trained to clean up the streaming DMD stage-1.
set -euo pipefail
MODEL=${1:-/data/yihao/sana-wm-streaming-model}
SS=/root/.cache/huggingface/hub/models--Hao-Zhe--test-anas-smoke/snapshots/21d9055d58eaaf0dc89de848d72cfc88037d08cd

echo "Re-pointing $MODEL/refiner to the test-anas-smoke streaming refiner..."
rm -rf "$MODEL/refiner"
mkdir -p "$MODEL/refiner"
ln -s "$SS/refiner_diffusers/transformer" "$MODEL/refiner/transformer"
ln -s "$SS/refiner_diffusers/connectors"  "$MODEL/refiner/connectors"
ln -s "$SS/gemma3_12b"                     "$MODEL/refiner/text_encoder"
echo "Done. refiner now:"
ls -l "$MODEL/refiner"
