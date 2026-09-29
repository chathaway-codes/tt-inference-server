#!/bin/bash
# Profile p1d1 of this image: prefill on chip 0, decode on chip 1 (TP=1 each), KV handed over the
# ethernet fabric; up to 8 concurrent decodes, 64K context. Rendered by hand from
# reference-x86/tt-model.yaml (serve.env merged with the profile's env; args in the order tt-model
# composes them, see serve-tp2.sh) and checked against reference-x86/tt_kernel_manifest.json
# serve_profiles[1].
#
# TT_SERVE_LAUNCHER=pd_container makes the plugin's general-plugin hook exec the PD supervisor (mpirun
# over /opt/openmpi-v5.0.7-ulfm) instead of a plain `vllm serve`. TT_PD_CONNS=2 = trained eth links
# between the two cards; a one-cable pair may need 1 (check the rank logs).
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
export HF_HUB_OFFLINE=1
export HF_MODEL=Qwen/Qwen3.8-27B
export MESH_DEVICE=P150
export QWEN36_FORCE_TP_PATH=1
export QWEN36_MAX_TOKENS_ALL_USERS=163840
export QWEN36_MTP=0
export QWEN36_PREFILL_BUCKET_TRACE=1
export QWEN36_SKIP_VISION=1
export QWEN_SDPA_BF8=1
export TORCHDYNAMO_DISABLE=1
export TT_CACHE_PATH=/cache/tensors-tp1
export TT_METAL_OPERATION_TIMEOUT_SECONDS=600
export TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36ForCausalLM,Qwen3_5ForConditionalGeneration=models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36ForCausalLM
export TT_PD_ALLOW_FUSED_CONV=1
export TT_PD_CHIPS=0,1
export TT_PD_CONNS=2
export TT_PD_D_POOL=163840
export TT_PD_EXPORT_SLOTS=1
export TT_PD_FABRIC_CLAIM_LEASE_S=0
export TT_PD_GDN_FUSED=2
export TT_PD_LEASE=30
export TT_PD_P_POOL=65536
export TT_PD_SSE_KEEPALIVE=0
export TT_PD_TAG=ttm
export TT_QWEN35_TEXT_VER=qwen36_blackhole
export TT_SERVE_LAUNCHER=pd_container
export VLLM_CONFIGURE_LOGGING=1
export VLLM_RPC_TIMEOUT=900000

exec vllm serve Qwen/Qwen3.8-27B --max-model-len 65536 --max-num-seqs 8 --block-size 64 --additional-config '{"tt": {"l1_small_size": 24576, "trace_region_size": 536870912}}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning_parser qwen3 --max-num-batched-tokens 65536 --no-enable-prefix-caching --no-async-scheduling --port 8000
