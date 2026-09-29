#!/bin/bash
# Profile tp2-dflash2 of this image: TP=2 + lossless DFlash2 speculative decoding, 1-4 greedy users,
# 64K context. Rendered by hand from reference-x86/tt-model.yaml (serve.env merged with the profile's
# env; args in the order tt-model composes them, see serve-tp2.sh) and checked against
# reference-x86/tt_kernel_manifest.json serve_profiles[2].
#
# Needs the drafter in the mounted HF cache first (the cache is mounted read-only):
#   hf download incoai/Qwen3.8-27B-DFlash2 --revision dedf8df68adfb1afeaf7b7480c0a0243108177b4
#
# Deliberately NOT set here (unlike serve-tp2.sh):
#   TT_MODEL_CLASS_OVERRIDES  the bundle's main_class (Qwen36DFlashForCausalLM) must win
#   QWEN36_MTP                tp2 sets it to 0; this profile doesn't
set -euo pipefail

if [ ! -e /dev/tenstorrent ]; then
  cat >&2 <<'USAGE'
This image needs Tenstorrent devices and a specific host setup, and was started without
them. It is normally launched with:

    tt-model serve <org/name>

To run it directly, all of these are required:

    docker run --rm \
      --device /dev/tenstorrent \
      --ipc host \
      --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
      --volume "$HOME/.cache/huggingface:/hf" \
      --volume "<a writable dir>:/cache" \
      --user "$(id -u):$(id -g)" \
      --publish 8000:8000 \
      <image>

The hugepages mount must be exactly /dev/hugepages-1G: umd matches that path in
/proc/mounts. Weights are NOT in this image; they are read from the mounted HF cache.
Under ROOTLESS docker use --user 0:0 instead: that daemon maps your uid to container
root, so "$(id -u)" there names an identity that owns none of the mounts.
USAGE
  exit 1
fi

export ARCH_NAME=blackhole
export DFLASH_WEIGHTS=incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4
export HF_HUB_OFFLINE=1
export HF_MODEL=Qwen/Qwen3.8-27B
export MESH_DEVICE=P150x2
export QWEN36_DFLASH_BLOCK=8
export QWEN36_DFLASH_FOLD_SEED=1
export QWEN36_DFLASH_SERVE_BLOCK=32
export QWEN36_DFLASH_TP=1
export QWEN36_DRAFTER=dflash2
export QWEN36_GDN_SPEC_FUSED=1
export QWEN36_MAX_TOKENS_ALL_USERS=262144
export QWEN36_PREFILL_BUCKET_TRACE=0
export QWEN36_SKIP_VISION=1
export QWEN_SDPA_BF8=1
export TORCHDYNAMO_DISABLE=1
export TT_CACHE_PATH=/cache/tensors-tp2
export TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/models/demos/blackhole/qwen36/mesh/p150_x2_2link_mesh_graph_descriptor.textproto
export TT_QWEN35_TEXT_VER=qwen36_blackhole
export VLLM_CONFIGURE_LOGGING=1
export VLLM_RPC_TIMEOUT=900000

exec vllm serve Qwen/Qwen3.8-27B --max-model-len 65536 --max-num-seqs 4 --block-size 64 --additional-config '{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824, "sample_on_device_mode": "decode_only"}}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning_parser qwen3 --max-num-batched-tokens 65536 --no-async-scheduling --port 8000
