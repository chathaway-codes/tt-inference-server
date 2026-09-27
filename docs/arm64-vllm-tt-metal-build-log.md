# ARM64 vLLM tt-metal build log (reproducible commands)

This is the exact sequence of commands used to build and check the aarch64 vLLM tt-metal image
that tt-cli launches for LLMs, with results and timings. The plan and background are in
[`arm64-vllm-tt-metal-image-plan.md`](arm64-vllm-tt-metal-image-plan.md).

All commands run from the **repo root** (`tt-inference-server/`) unless noted otherwise.
Logs went to `/tmp/*.log`; they're referenced below but not checked in.

## Quick reproduce (everything, in order)

```sh
# 1. v0.18.0 build context + arm64 Dockerfile + tt-cli's model_spec.json
git worktree add /tmp/ttis-v0.18.0 v0.18.0
cp vllm-tt-metal/vllm.tt-metal.src.dev.arm64.Dockerfile /tmp/ttis-v0.18.0/vllm-tt-metal/
cp ~/.local/share/tenstorrent/tools/tt-inference-server-v0.18.0/src/model_spec.json /tmp/ttis-v0.18.0/

# 2. Free the name tt-cli uses (it was a second tag on the base image; the base keeps its own tag)
docker rmi ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7

# 3. Build (~15 min)
docker build -f /tmp/ttis-v0.18.0/vllm-tt-metal/vllm.tt-metal.src.dev.arm64.Dockerfile \
  --build-arg TT_METAL_DOCKERFILE_URL=tt-metalium-release:local \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=c49bb7625e \
  --build-arg TT_VLLM_COMMIT_SHA_OR_TAG=6b4a3a7b4e04c3b7f4f60dea97a7e156c3956d46 \
  --build-arg CONTAINER_APP_UID=1000 --progress=plain \
  -t vllm-tt-metal-src-dev-ubuntu-22.04-arm64:0.18.0-c49bb76-6b4a3a7 /tmp/ttis-v0.18.0 \
  > /tmp/vllm-arm64-build-1.log 2>&1

# 4. The name tt-cli looks for (release = re-tagged dev, as in build_docker_images.py)
docker tag vllm-tt-metal-src-dev-ubuntu-22.04-arm64:0.18.0-c49bb76-6b4a3a7 \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7

git worktree remove /tmp/ttis-v0.18.0   # optional cleanup
```

## Environment

| | |
|---|---|
| Host | Ubuntu 24.04.4 **aarch64**, 80 cores, 125 GB RAM, motherboard `MP32-AR0-JG` |
| Devices | 2x Tenstorrent Blackhole P150 (`/dev/tenstorrent/{0,1}`), FW bundle 19.13.2, KMD 2.10.0, 1 GB hugepages |
| Docker | 29.7.2, buildx 0.36.1 |
| Base image | `tt-metalium-release:local` (`bd99e0346dfe`, arm64, Ubuntu 22.04, clang-20, uv 0.9.26 in `/usr/local/bin`; built per `chathaway-tt-metal/dockerfile/ARM64_RELEASE_BUILD_PLAN.md`) |

## Pinned versions

| Component | Pin | Where it comes from |
|---|---|---|
| Dockerfile / app sources | tt-inference-server `v0.18.0` (`d5913e81`) | tt-cli v0.18.0 |
| tt-metal | `c49bb7625e` | `release_model_spec.json` (Llama-3.1-8B-Instruct) |
| vLLM | `tenstorrent/vllm` `6b4a3a7b4e04c3b7f4f60dea97a7e156c3956d46` (reports `0.1.dev14175+g6b4a3a7b4`) | same |
| tt-smi | `v3.1.1` | Dockerfile default |
| `model_spec.json` | tt-cli's copy (`release_version` 0.18.0) | not in the v0.18.0 tag; generated at build time upstream |

## Step 1: Dockerfile

`vllm-tt-metal/vllm.tt-metal.src.dev.arm64.Dockerfile` is the v0.18.0 Dockerfile (byte-identical
to tt-cli's copy) with these changes, each marked `arm64:` in the file:

1. `ARG TT_METAL_DOCKERFILE_URL=tt-metalium-release:local` default.
2. `build_metal.sh --toolchain-path cmake/aarch64-linux-clang-20-libstdcpp-toolchain.cmake` when
   `uname -m` is `aarch64` (the script's default is the x86_64 toolchain, which adds
   `-march=x86-64-v3`). `CMAKE_BUILD_PARALLEL_LEVEL` defaults to 48 to keep unity builds within RAM.
3. `create_venv.sh` with `CXX=clang++-20 CC=clang-20` (the base has no clang-17).
4. In both stages, `PATH` without `/opt/venv/bin` and `VIRTUAL_ENV=` (the base activates its own
   ttnn venv).

`CONTAINER_APP_UID=1000`: the `volume_id_tt_transformers-Llama-3.1-8B-Instruct` volume tt-cli
created is owned by 1000.

## Step 2: Build

Passed on the **first attempt**, 01:12 to 01:27 UTC (about 15 minutes, cold except for the base):

| Stage | Time |
|---|---|
| tt-metal clone + clang-20 build + `create_venv.sh` | 603 s |
| vLLM install (`VLLM_TARGET_DEVICE=empty`) | 35 s |
| tt-smi (Rust/pyluwen) | 40 s |
| Image export | 54 s |

Result: `cb3dc0ac124e`, arm64, 16.8 GB.

## Step 3: Checks without a device

```sh
IMG=vllm-tt-metal-src-dev-ubuntu-22.04-arm64:0.18.0-c49bb76-6b4a3a7
docker image inspect $IMG --format '{{.Architecture}} {{json .Config.Entrypoint}}'
docker run --rm $IMG --help | head -30
docker run --rm --entrypoint bash $IMG -c 'source $PYTHON_ENV_DIR/bin/activate && python --version &&
  python -c "import ttnn, vllm; print(vllm.__version__)" && ls $TT_SMI_DIR/.venv/bin/tt-smi && id'
```

- `arm64`, entrypoint `run_vllm_api_server.py "$@"`.
- `--help` loads the `tt` platform plugin (`vllm_tt_plugin`) and prints the entrypoint's options.
  It uses `parse_known_args`, so `--enable-auto-tool-choice --tool-call-parser llama3_json` are
  passed on to vLLM.
- Python 3.10.12, `import ttnn, vllm` OK, tt-smi present, runs as `uid=1000(container_app_user)`.

## Step 4: On device

**Card state.** The first on-device attempts ran right after another vLLM server (`ttserve`) had
been stopped. Device 1 failed firmware init (`TT_THROW: Device 1: Timeout (10000 ms) waiting for
physical cores to finish` / `Device 1 init: failed to initialize FW! Try resetting the board.`).
A hard reboot of the host fixed it. That's a card-state problem, not an image problem.

### 4a. ttnn smoke test

```sh
docker run --rm --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --entrypoint bash vllm-tt-metal-src-dev-ubuntu-22.04-arm64:0.18.0-c49bb76-6b4a3a7 -c 'source $PYTHON_ENV_DIR/bin/activate && python -c "
import ttnn, torch
d = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=d)
print(\"SMOKE_OK\", ttnn.to_torch(ttnn.add(a, a))[0, 0].item())
ttnn.close_device(d)"'
```

`SMOKE_OK 2.0` in 13.7 s. The only warnings were the known `Unknown motherboard 'MP32-AR0-JG'`
lines (it falls back to bus IDs; harmless here).

### 4b. tt-cli command

The **unchanged** tt-cli command reached weight loading and then failed:

```
FileNotFoundError: No such file or directory: /home/container_app_user/cache_root/model_file_symlinks_map/Llama-3.1-8B-Instruct/model-00001-of-00004.safetensors
```

That's caused by this host's HF cache layout, not by the image. The weights are behind two
levels of symlinks, and the second level points outside the model directory that tt-cli mounts:

```
models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e.../model-00001-of-00004.safetensors -> ../../blobs/2b1879...
models--meta-llama--Llama-3.1-8B-Instruct/blobs/2b1879...  -> ../../blobs/37/37e46c...   (= ~/.cache/huggingface/hub/blobs/37/...)
```

With one extra read-only mount (`hub/blobs` at `readonly_weights_mount/blobs`) so the
second-level links resolve, it works:

```sh
docker run -d --name tt-inference-server-d517878a \
  --env-file ~/.local/share/tenstorrent/tools/tt-inference-server-v0.18.0/src/.env \
  --ipc host --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --publish 0.0.0.0:8000:8000 \
  --volume volume_id_tt_transformers-Llama-3.1-8B-Instruct:/home/container_app_user/cache_root \
  --mount type=bind,src=$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,dst=/home/container_app_user/readonly_weights_mount/Llama-3.1-8B-Instruct,readonly \
  --mount type=bind,src=$HOME/.cache/huggingface/hub/blobs,dst=/home/container_app_user/readonly_weights_mount/blobs,readonly \
  -e MODEL_WEIGHTS_DIR=/home/container_app_user/readonly_weights_mount/Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7 \
  --model meta-llama/Llama-3.1-8B-Instruct --tt-device p150 --no-auth \
  --enable-auto-tool-choice --tool-call-parser llama3_json
```

On a host with a standard HF cache (real files in `models--.../blobs/`), the extra mount isn't needed.

Cold start (empty tensor cache), ready after **about 8 minutes** (`startup_time: 490.5 seconds`):

| Phase | Duration |
|---|---|
| Entrypoint, arg parsing, model registry | ~37 s |
| Device open, HF weights loaded | ~12 s |
| Weight conversion/caching, 32 layers (~7 GB into `cache_root`) | 3 min 03 s |
| KV cache allocation, 32 layers | 2 min 21 s |
| Prefill/decode warmup incl. kernel compile | ~83 s |
| Trace capture | ~5 s |
| Background trace capture up to 64K context (after the API is up) | ~102 s |

The vLLM command the entrypoint logged:

```
vllm serve --enable-auto-tool-choice --tool-call-parser llama3_json --model meta-llama/Llama-3.1-8B-Instruct \
  --block_size 64 --max_model_len 65536 --max_num_seqs 32 --max_num_batched_tokens 65536 \
  --max-log-len 32 --seed 9472 --additional_config '{"tt": {"trace_region_size": 900000000}}'
```

Requests:

| Request | Result |
|---|---|
| `GET /v1/models` | `meta-llama/Llama-3.1-8B-Instruct`, `max_model_len` 65536 |
| Chat "What is the capital of France?" | `The capital of France is Paris.` |
| Tool call (`get_weather`, `tool_choice: auto`) | `tool_calls: get_weather {"city": "Paris"}`, `finish_reason: tool_calls` |
| 128 forced tokens (`ignore_eos`), single user | 6.94 s end to end, about 18.4 tok/s including prefill |

No `ERROR` or `Traceback` lines. Warnings worth knowing about (none fatal):

- `Unknown motherboard 'MP32-AR0-JG'`: falls back to bus IDs (as in the forge work).
- `env var ARCH_NAME is already set to wormhole_b0, overriding with blackhole`: the Dockerfile
  bakes in `ARCH_NAME=wormhole_b0` (same as upstream); the entrypoint corrects it.
- `Unknown model Llama-3.1-8B-Instruct on device P150, setting MAX_PREFILL_CHUNK_SIZE to 4`:
  the runtime logs still show `max_prefill_chunk_size: 4096`.
- Kernel compile `-Wunused-but-set-variable` warnings.

## Not done yet

- Warm-start time (the tensor cache is now populated in the volume).
- tt-inference-server's benchmark/eval workflow, to compare throughput and accuracy with x86.
- Multi-card runs.
- tt-cli: the unchanged command fails on hosts whose HF cache uses a shared second-level blob
  store (as here); tt-cli would have to mount the resolved blob paths.
