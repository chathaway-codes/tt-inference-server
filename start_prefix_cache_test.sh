#!/bin/bash
set -euo pipefail

CONTAINER_NAME="qwen-dflash-test"

# Remove existing test container if present
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting container $CONTAINER_NAME with GDN Prefix Caching, Sticky Sessions, and RAGGED=1 (Restart: unless-stopped)..."

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --user 1000:1000 \
  --workdir /home/tt/work \
  --ipc host \
  --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  --device /dev/tenstorrent:/dev/tenstorrent \
  -p 127.0.0.1:8000:8000 \
  -v /mnt/md0/tt-model-cache/qwen3.8-27b-p150x2:/cache \
  -v /mnt/md0/huggingface:/hf:ro \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /home/charles/p3/tt-inference-server/lever_n_overlay/qwen36:/opt/tt-metal/models/demos/blackhole/qwen36 \
  -v /home/charles/p3/tt-inference-server/lever_n_overlay/vllm_tt_plugin:/opt/tt-venv/lib/python3.12/site-packages/vllm_tt_plugin \
  -e HF_MODEL="Qwen/Qwen3.8-27B" \
  -e MESH_DEVICE="P150x2" \
  -e TT_MESH_GRAPH_DESC_PATH="/opt/tt-metal/models/demos/blackhole/qwen36/mesh/p150_x2_2link_mesh_graph_descriptor.textproto" \
  -e QWEN36_DFLASH_TP="1" \
  -e VLLM_CONFIGURE_LOGGING="1" \
  -e HF_HUB_OFFLINE="1" \
  -e DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4" \
  -e QWEN36_DRAFTER="dflash2" \
  -e QWEN36_DFLASH_SERVE_BLOCK="32" \
  -e QWEN36_DFLASH_RAGGED="1" \
  -e QWEN36_DFLASH_FOLD_SEED="1" \
  -e QWEN36_SKIP_VISION="1" \
  -e VLLM_RPC_TIMEOUT="900000" \
  -e TT_CACHE_PATH="/cache/tensors-tp2" \
  -e TT_QWEN35_TEXT_VER="qwen36_blackhole" \
  -e QWEN36_MAX_TOKENS_ALL_USERS="524288" \
  -e TORCHDYNAMO_DISABLE="1" \
  -e ARCH_NAME="blackhole" \
  -e QWEN36_DFLASH_BLOCK="8" \
  -e QWEN36_GDN_SPEC_FUSED="1" \
  -e QWEN36_PREFILL_BUCKET_TRACE="0" \
  -e QWEN_SDPA_BF8="1" \
  -e EXTRA_MODELS_DIR="/opt/tt-metal/models/demos/blackhole/qwen36/vllm_bundle" \
  -e TT_VLLM_BUILTIN_MODELS="0" \
  -e TT_MODEL_KIND="vllm-plugin" \
  -e HF_HOME="/hf" \
  -e TT_METAL_CACHE="/cache" \
  -e QWEN_PREFIX_REUSE="1" \
  -e QWEN_FAST_STICKY_SESSIONS="1" \
  -e QWEN_PREFIX_STORE_GIB="8" \
  tt-model/qwen3.8-27b-p150x2:arm64 \
  vllm serve Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 262144 \
  --enable-prefix-caching \
  --block-size 64 \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --port 8000 \
  --additional-config '{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824, "sample_on_device_mode": "decode_only"}}'

echo "Container launched with Prefix Caching and RAGGED=1."
