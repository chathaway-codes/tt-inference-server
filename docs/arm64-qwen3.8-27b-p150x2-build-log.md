# ARM64 Qwen3.8-27B P150x2 build log (reproducible commands)

The exact commands used to build and test the native aarch64 image `tt-model/qwen3.8-27b-p150x2:arm64`
(Qwen/Qwen3.8-27B on 2x Blackhole P150; profiles `tp2`, `p1d1`, `tp2-dflash2`), with results and
timings. Plan and background: [`arm64-qwen3.8-27b-p150x2-plan.md`](arm64-qwen3.8-27b-p150x2-plan.md).

All commands run from the **repo root** (`tt-inference-server/`) unless noted otherwise.
Logs went to `/tmp/*.log`; they're referenced below but not checked in.

## Quick reproduce

```sh
# 1. Host prep (once): drafter into the shared HF cache, UID-1000 cache dir. /mnt/md0 is root-owned
#    and sudo needs a password, so write through root containers.
docker run --rm -v /mnt/md0/huggingface:/hf -e HF_HOME=/hf python:3.12-slim sh -c \
  'pip install -q huggingface_hub && hf download incoai/Qwen3.8-27B-DFlash2 --revision dedf8df68adfb1afeaf7b7480c0a0243108177b4'
docker run --rm -v /mnt/md0:/m alpine sh -c \
  'mkdir -p /m/tt-model-cache/qwen3.8-27b-p150x2 && chown -R 1000:1000 /m/tt-model-cache'

# 2. Build
docker build --progress=plain -f vllm-tt-metal/qwen3.8-27b-p150x2.arm64/Dockerfile \
  -t tt-model/qwen3.8-27b-p150x2:arm64 vllm-tt-metal/qwen3.8-27b-p150x2.arm64
docker tag tt-model/qwen3.8-27b-p150x2:arm64 tt-model/qwen3.8-27b-p150x2:arm64-aa20afaf-ab5f7f3

# 3. Serve (profile = tp2 | tp2-dflash2 | p1d1)
docker run -d --name qwen-arm64-<profile> --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume /mnt/md0/huggingface:/hf:ro \
  --volume /mnt/md0/tt-model-cache/qwen3.8-27b-p150x2:/cache \
  --user 1000:1000 --publish 127.0.0.1:8000:8000 \
  tt-model/qwen3.8-27b-p150x2:arm64 /usr/local/bin/serve-<profile>.sh
# stop with `docker stop` (SIGTERM closes the mesh), never docker kill / rm -f
```

## Environment

| | |
|---|---|
| Host | Ubuntu 24.04 **aarch64**, 80 cores, 125 GB RAM, motherboard `MP32-AR0-JG` |
| Devices | 2x Blackhole P150 (`/dev/tenstorrent/{0,1}`), ethernet-linked, FW 19.13.2, KMD 2.10.0, 1 GB hugepages |
| Docker | 29.7.2, buildx 0.36.1 |
| Base image | `tt-metalium-release:local` (`bd99e0346dfe`, arm64, Ubuntu 22.04, clang-20, uv 0.9.26) |
| Disk at start | `/` 66 GB free (the plan assumed ~180 GB), `/mnt/md0` 553 GB free |

## Pinned versions

| Component | Pin |
|---|---|
| tt-metal | `tenstorrent/tt-metal@aa20afaf172ac35370b8d112edc0f771ee6d2571` (ttnn dist version pinned to `0.65.2.dev9815`) |
| TT vLLM plugin | `changh95/vllm-tt-plugin@ab5f7f3d795ee519b5d96f41f31cc1e0f635d6a7` (non-editable) |
| vLLM | `0.26.0` sdist, `VLLM_TARGET_DEVICE=empty` |
| Python packages | `vllm-tt-metal/qwen3.8-27b-p150x2.arm64/requirements-x86-lock.txt` = the x86 venv's 216 dists minus 5 (below), installed `--no-deps` |
| Qwen weights | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (already on disk) |
| DFlash2 drafter | `incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4` |

## Step 0: Host prep (2026-09-29)

- Cache dir `/mnt/md0/tt-model-cache/qwen3.8-27b-p150x2` created, owned 1000:1000.
- Drafter download: 04:20:27 to 04:21:32 UTC (about 1 min incl. pulling `python:3.12-slim`).
  `model.safetensors` = 3,848,817,896 bytes, sha256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`
  (matches the `67fc76d6...` in `tt-model.yaml`). The hub CLI is `hf` (huggingface_hub 2.0.0 has no `[cli]` extra).
- Drafter resolution checked in the source: `tt/dflash2.py` calls
  `snapshot_download(repo, revision=rev, local_files_only=HF_HUB_OFFLINE == "1")`, so the read-only
  `/hf` mount works offline.

## Step 1: Dockerfile and scripts

Files in `vllm-tt-metal/qwen3.8-27b-p150x2.arm64/`:

| File | What |
|---|---|
| `Dockerfile` | Two stages on `tt-metalium-release:local`; deviations from x86 marked `arm64:` |
| `requirements-x86-lock.txt` | Exact pins generated from `reference-x86/x86-venv-packages.txt` |
| `entrypoint.sh` | Identical to `reference-x86/entrypoint.sh` |
| `serve-tp2.sh` | `reference-x86/serve-default.sh` + one comment line (default CMD; `serve-default.sh` is a symlink to it) |
| `serve-tp2-dflash2.sh`, `serve-p1d1.sh` | Rendered from `tt-model.yaml`, checked against `tt_kernel_manifest.json` `serve_profiles` |
| `verify.sh` | x86 `verify.sh` verbatim + an arm64 block (arch, Python 3.12, torch/torchvision/transformers/numpy/vllm versions, ttnn from `/opt/tt-metal`) |

Design decisions (found while writing it; differ from the plan's sketch):

1. **No `create_venv.sh`.** The x86 venv isn't a `requirements-dev.txt` venv (216 dists; loguru 0.7.3,
   transformers 5.17.0, pandas 3.0.6; no pytest/black/diffusers). The venv is created the same way
   `create_venv.sh` does it (`uv python install 3.12` + `uv venv --relocatable --managed-python`), then
   the x86 lock is installed with `--no-deps`. That also sidesteps `vllm-overrides.txt`'s
   `transformers<5.15.0` cap: the validated image had 5.17.0.
2. **Venv before the C++ build.** `build_metal.sh` passes `Python3_EXECUTABLE=$(command -v python3)` to
   CMake, so the 3.12 venv must be first on `PATH` during the build (the base's `/opt/venv` is 3.10).
3. **Layer order**: apt, venv, clone, build (drops objects/CPM cache in the same layer), then the pip
   steps. Python-side fixes reuse the cached compile.
4. **Dropped from the lock** (besides `ttnn`, `vllm`, `vllm_tt_plugin`, which are built from the pins):
   `pyluwen` (sdist-only on aarch64; needs Rust) and `tt_smi` (its only dependent; a CLI the serving
   path doesn't import). Every other pinned dist has an aarch64 (glibc <= 2.35) or pure-Python wheel;
   checked against PyPI and the PyTorch CPU index before the first build.
5. **vLLM Rust extensions** are optional in 0.26.0 and skipped (no cargo). The x86 `vllm` package has
   no `_rust_*.so` / `vllm-rs` either.
6. **Trim** mirrors the x86 image: `build_Release/lib` (+ `build` symlink), `runtime/`, `tt_metal/`,
   `ttnn/`, `tools/`, `setup.py`, `pyproject.toml`, and the `source.code` allowlist of `models/`.

## Step 2: Build attempts

| # | Start (UTC) | Duration | Result | Root cause / fix |
|---|---|---|---|---|
| 1 | 04:30:05 | 21 s | FAIL, runtime stage user setup (`Dockerfile:152`) | The base's `SHELL` is `bash -euo pipefail`; there is no UID 1000 in the base, so `getent passwd 1000` exits 2 and `existing="$(getent ... \| cut ...)"` fails. The x86 image's line assumed plain `sh`. BuildKit cancelled the tt-metal clone. Cached: builder apt, venv (Python 3.12.12, uv 0.9.26), runtime apt |
| 2 | 04:31:46 | 10.7 min | FAIL, builder plugin step | Fix for 1: `getent ... \| cut ... \|\| true` (marked `arm64:`); worked. tt-metal clone 56 s, **compile + in-layer trim 484 s** (`/opt/tt-metal` 1.7 GB after), lock install 27.5 s, ttnn editable 30.9 s, vLLM sdist 28.8 s. Then the plugin RUN `rm -rf`'d its own cwd and `uv pip freeze` failed with "Current directory does not exist" (plugin itself had installed). Disk: 65 GB free before, 121 GB after (freed by something else on the host) |
| 3 | 04:43:40 | 65 s | FAIL, runtime `verify.sh` (arm64 block) | Fix for 2: `cd /` before `rm -rf`; worked. Trim `du`: build_Release 122M, runtime 422M, tt_metal 203M, ttnn 80M, models 15M, tools 4.3M; venv 4.3G; uv Pythons 180M. `uv pip check`: only `vllm requires opencv-python-headless>=4.13.0, 4.11.0.86 installed` (same pin as x86, by design of `vllm-overrides.txt`). verify.sh: arch + Python 3.12 OK; torch 2.11.0+cpu, torchvision 0.26.0+cpu, transformers 5.17.0, numpy 1.26.4 all match, but my check expected `vllm.__version__ == "0.26.0+empty"`; the module reports `0.26.0` (the `+empty` is only in the dist metadata, which is what the x86 list records) |
| 4 | 04:46:19 | 1 min 55 s | **PASS** | Fix for 3: expect `vllm.__version__ == "0.26.0"` and add `importlib.metadata.version("vllm") == "0.26.0+empty"`. verify.sh 81.5 s, all checks pass. Image `sha256:b818e580f532...`, arm64, 8.78 GB (x86: 6.44 GB). Tagged `:arm64` and `:arm64-aa20afaf-ab5f7f3` |

A cold build (no cache) is therefore about 12 min on this host: clone 1 min, tt-metal compile 8 min,
Python steps about 2 min, runtime stage and verify.sh about 2 min.

## Step 3: Checks without a device

All passed (attempt 4 build agent):

- `docker image inspect`: `arm64`, 8,782,588,717 bytes, entrypoint `/usr/local/bin/entrypoint.sh`,
  cmd `/usr/local/bin/serve-tp2.sh`, labels as the x86 image + `org.opencontainers.image.architecture=arm64`.
- Python 3.12.12; `torch 2.11.0+cpu`, `torchvision 0.26.0+cpu`, `transformers 5.17.0`, `numpy 1.26.4`,
  `vllm.__version__ 0.26.0` (dist `0.26.0+empty`); TT platform plugin activates on import.
- `Qwen36DFlashForCausalLM` (with `QWEN36_DRAFTER=dflash2`) and `Qwen36ForCausalLM` import.
- `serve-tp2.sh` and `serve-tp2-dflash2.sh` print the usage message and exit 1 without devices.
- User `uid=1000(tt)`; env matches the x86 image (`PYTHON_ENV_DIR` present but empty); `/opt/tt-metal`
  846 MB (x86: about 1.1 GB incl. the duplicated `build/`), `/opt/tt-venv` 4.3 GB; `pyvenv.cfg` home
  `/usr/local/share/uv/cpython-3.12.12-linux-aarch64-gnu/bin`, relocatable; SFPI in `runtime/sfpi`.
- verify.sh (x86 checks verbatim): `EXTRA_MODELS_DIR` resolves
  `Qwen3_5ForConditionalGeneration -> ...qwen36_vllm_dflash:Qwen36DFlashForCausalLM`; plugin kv_transfer
  modules, `tt_serve_launcher` entry point, PD mesh descriptor resource, `ttnn.distributed.ttrun`,
  `mpirun`, 2-link descriptor (`count: 2`) all OK; `pd_container --dry-run` exits 0 with the expected plan
  (chips 0/1, board p150x2, FABRIC_2D, pools 65536/163840). Only notice: "Triton is installed but 0 active
  driver(s) found ... Disabling Triton" (expected, no GPU).
- Freeze vs the x86 list: missing only `pyluwen`, `tt-smi` (intended); `ttnn` is `-e file:///opt/tt-metal`;
  the plugin shows as a direct file URL (pinned git checkout); every other version identical.
  `uv pip check` reports one incompatibility, `vllm` wants `opencv-python-headless>=4.13.0` but 4.11.0.86
  is installed; the x86 venv has the same pin (`vllm-overrides.txt`: numpy<2 wins).
- No `tt-smi` CLI in the image (dropped with pyluwen).

## Step 4: On device

### 4a. tp2

### 4b. tp2-dflash2

### 4c. Comparison

## Not done / open

- **Device tests blocked (2026-09-29 ~04:50 UTC).** The tt-studio stack came up at about 03:57 UTC, after
  the plan was written: `tt_studio_backend_api_prod` has `/dev/tenstorrent` mapped into it and publishes
  host port 8000. Per the plan's rule 3, waiting for the user before any device test.
