# ARM64 forge build log (reproducible commands)

This is the exact sequence of commands used to build and check the aarch64 forge pieces, with
results and timings. The plan and background are in
[`arm64-forge-image-plan.md`](arm64-forge-image-plan.md).

All commands run from the **repo root** (`tt-inference-server/`) unless noted otherwise.
Build logs went to `/tmp/*.log`; they're referenced below but not checked in.

## Quick reproduce (everything, in order)

```sh
# 1. All aarch64 wheels (pjrt-plugin-tt, vllm_tt, torch-xla) -> tt-media-server/wheels/
#    Build the two heavy stages one after the other: with --target wheels alone, BuildKit would
#    run them concurrently (64 ninja + 48 Bazel jobs), which may not fit in 125 GB of RAM.
#    That concurrent run was never tried.
F=tt-media-server/Dockerfile.forge-builder.arm64; CTX=tt-media-server/scripts/forge_arm64
docker build -f $F --target pjrt-wheel -t tt-forge-builder:pjrt-wheel-arm64 $CTX
docker build -f $F --target torch-xla-wheel -t tt-forge-builder:torch-xla-wheel-arm64 $CTX
docker build -f $F --target wheels --output type=local,dest=tt-media-server/wheels $CTX

# 2. Runtime image with the TT stack
docker build --platform linux/arm64 -f tt-media-server/Dockerfile.forge.arm64 \
  --build-arg FORGE_WHEELS=wheels -t tt-media-server-forge:arm64-tt .

# 3. Serve ResNet on device 0 and query it (Step 5)
docker run -d --name forge-arm64-resnet --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G -p 127.0.0.1:8123:8000 \
  -e MODEL_RUNNER=tt-xla-resnet -e DEVICE_IDS="(0)" tt-media-server-forge:arm64-tt
```

From a cold cache on this host, step 1 should take about 1 hour: about 20 minutes for the
toolchain, 20 to 25 for pjrt, and 15 for torch-xla. That's estimated from the separate runs
below; the exact sequence above hasn't been run cold end to end. Step 2 takes about 8 minutes.

## Environment

| | |
|---|---|
| Host | Ubuntu 24.04.4, Linux 6.8.0-139-generic **aarch64**, 80 cores, 125 GB RAM, motherboard `MP32-AR0-JG` |
| Devices | 2x Tenstorrent Blackhole (`/dev/tenstorrent/{0,1}`), 1 GB hugepages at `/dev/hugepages-1G` |
| Docker | 29.7.2, buildx v0.36.1 (BuildKit is needed for `RUN --mount=type=cache`) |
| Base image | `tt-metalium-release:local` (built locally; Ubuntu 22.04, arm64, clang-20, ttnn `0.75.0rc10+gd04395ed862`) |

## Pinned versions

| Component | Pin | Where it comes from |
|---|---|---|
| tt-forge (x86 reference) | `1.5.0.dev20260812001218` | `tt_model_runners/forge_runners/requirements.txt` |
| tt-xla | `3cfcfcfdefac5e310f9864c82548021636e18771` | `pjrt-plugin-tt` wheel `Summary` |
| tt-mlir | `da6f1ce081af95630f5d679392e50a4bafa7c6ea` | tt-xla `third_party/CMakeLists.txt` |
| tt-metal | `5beed318d0f0d1c6212e605947fb0be80c9e0a1d` | tt-mlir `third_party/CMakeLists.txt` |
| LLVM | `4efe170d858eb54432f520abb4e7f0086236748b` | tt-mlir `env/CMakeLists.txt` |
| SFPI | `7.67.0` | tt-metal `tt_metal/sfpi-version` |
| {fmt} | `11.1.4` | tt-metal CPM pin (seen in the build log) |
| torch-xla | `2.9.0+gited8a445` | `pjrt-plugin-tt` `Requires-Dist` |
| tt_lang | `1.1.5.dev20260704+light` | `pjrt-plugin-tt` `Requires-Dist` |
| torch / jax | `2.11.0+cpu` / `0.7.1` | `pjrt-plugin-tt` `Requires-Dist` |

## Step 0: Read the x86 package metadata (Phase 0)

```sh
mkdir -p /tmp/forge-meta && cd /tmp/forge-meta
python3 -m pip download --no-deps --only-binary=:all: --python-version 3.12 \
  --platform manylinux_2_34_x86_64 --platform linux_x86_64 \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com \
  tt-forge==1.5.0.dev20260812001218 pjrt-plugin-tt==1.5.0.dev20260812001218

# Dependencies and commit SHAs
unzip -p tt_forge-*.whl '*.dist-info/METADATA'
unzip -p pjrt_plugin_tt-*.whl '*.dist-info/METADATA' | grep -E '^(Summary|Requires-Dist)'
unzip -p pjrt_plugin_tt-*.whl '*.dist-info/entry_points.txt'
# What `tt-forge-install` does (SFPI installer, hardcodes x86_64)
unzip -p pjrt_plugin_tt-*.whl ttxla_tools/install_sfpi.py
unzip -p pjrt_plugin_tt-*.whl pjrt_plugin_tt/tt-metal/tt_metal/sfpi-version
```

Base image inspection (tt-metal version, Python, SFPI, environment):

```sh
docker run --rm --entrypoint /bin/bash tt-metalium-release:local -c '
  cat /etc/os-release | head -2; python3 --version; clang --version | head -1
  cat /opt/venv/lib/python3.10/site-packages/ttnn-*.dist-info/METADATA | head -3
  grep sfpi_version= /opt/venv/lib/python3.10/site-packages/ttnn/tt_metal/sfpi-version'
docker inspect tt-metalium-release:local --format '{{range .Config.Env}}{{println .}}{{end}}'
```

Results are summarized in the Status section of the plan.

## Step 1: Runtime image without the TT stack (parallel track)

```sh
docker build --platform linux/arm64 -f tt-media-server/Dockerfile.forge.arm64 \
  -t tt-media-server-forge:arm64 . > /tmp/forge-arm64-build.log 2>&1
```

The build passed on the first try; the `setup_env.sh` step took about 3 minutes and the image is about 11 GB.
Smoke test:

```sh
docker run --rm tt-media-server-forge:arm64 bash -c '
  tt-smi --help | head -1
  cd server && source venv-worker/bin/activate
  python -c "import torch, diffusers, transformers, timm, yolox, faster_fifo, av, imageio_ffmpeg, pycocotools, fastapi, blacksmith, flax; print(torch.__version__)"
  python -c "import main"
  $ADAPTER_MERGE_PYTHON -c "import transformers, peft"'
```

## Step 2: Builder image and tt-mlir toolchain (Phase 1.1 and 1.2)

File: `tt-media-server/Dockerfile.forge-builder.arm64`. The build context is
`tt-media-server/scripts/forge_arm64` (it only contains `patch_tt_xla.py`).

```sh
# builder: build deps, Python 3.12, bazelisk, ld.lld symlink (about 1 minute)
docker build -f tt-media-server/Dockerfile.forge-builder.arm64 --target builder \
  -t tt-forge-builder:builder-arm64 tt-media-server/scripts/forge_arm64 > /tmp/forge-builder.log 2>&1

# toolchain: tt-mlir env/ build into /opt/ttmlir-toolchain (about 19.5 minutes: 1166 s for the RUN step)
docker build -f tt-media-server/Dockerfile.forge-builder.arm64 --target toolchain \
  -t tt-forge-builder:toolchain-arm64 tt-media-server/scripts/forge_arm64 > /tmp/forge-toolchain.log 2>&1

# check
docker run --rm tt-forge-builder:toolchain-arm64 bash -c '
  T=/opt/ttmlir-toolchain; $T/bin/mlir-opt --version | head -3; $T/bin/flatc --version
  file $T/bin/mlir-opt; $T/venv/bin/python --version; du -sh $T'
```

Result: `mlir-opt` (LLVM 22.0.0git, aarch64), `flatc` 24.3.25, Python 3.12.14, 5.0 GB.

> The first toolchain run used `tt-media-server` as the build context. Stages before
> `pjrt-wheel` don't `COPY` anything, so switching the context later didn't invalidate them.

## Step 3: pjrt-plugin-tt and vllm_tt wheels (Phase 1.3)

### 3a. Patch script, tried locally first

`tt-media-server/scripts/forge_arm64/patch_tt_xla.py` applies 7 checked edits to the tt-xla
checkout (see its docstring). Every edit has to match exactly, so running it twice, or against a
different tt-xla commit, fails instead of silently doing nothing. To see the resulting diff:

```sh
git clone https://github.com/tenstorrent/tt-xla.git /tmp/tt-xla
git -C /tmp/tt-xla checkout 3cfcfcfdefac5e310f9864c82548021636e18771
python3 tt-media-server/scripts/forge_arm64/patch_tt_xla.py /tmp/tt-xla
git -C /tmp/tt-xla --no-pager diff
```

### 3b. Build

```sh
docker build -f tt-media-server/Dockerfile.forge-builder.arm64 --target pjrt-wheel \
  -t tt-forge-builder:pjrt-wheel-arm64 tt-media-server/scripts/forge_arm64 > /tmp/forge-pjrt.log 2>&1

# export just the wheels to tt-media-server/wheels/ (git-ignored via *.whl)
docker build -f tt-media-server/Dockerfile.forge-builder.arm64 --target wheels \
  --output type=local,dest=tt-media-server/wheels tt-media-server/scripts/forge_arm64
```

It took four attempts. Each failure was fixed in the Dockerfile or the patch script, so a fresh
run of the commands above should pass on the first try:

| # | Failed after | Error | Fix |
|---|---|---|---|
| 1 | ~8 min, in tt-mlir | `'get_temporary_buffer' is deprecated [-Werror]` (GCC 12 libstdc++ `stable_sort`) | patch 6: add `-Wno-error=deprecated-declarations` to tt-mlir's `CMAKE_CXX_FLAGS` in tt-xla `third_party/CMakeLists.txt` (an env `CXXFLAGS` doesn't work because that file sets the flags explicitly) |
| 2 | ~18 min, in tt-xla after tt-mlir and tt-metal finished | `'format' file not found` (`<format>` needs GCC 13's libstdc++) | patch 7: `pjrt_implementation/inc/utils/assert.h` uses header-only `{fmt}` |
| 3 | ~13 min (ccache warm) | `consteval ... not a constant expression` in `/usr/include/fmt` (Ubuntu's fmt 8.1.1 with clang 20) | install {fmt} 11.1.4 headers into `/usr/local/include` (same version tt-metal pins) instead of `libfmt-dev` |
| 4 | passed | | wheel `RUN` step 971 s (tt-mlir and tt-metal mostly ccache hits from earlier runs) |

A cold build (empty ccache) got through tt-mlir and tt-metal in about 18 minutes (attempt 2),
so a full cold run should take roughly 20 to 25 minutes on this machine. The ccache lives in a
BuildKit cache mount (`/ccache`) and survives between builds.

Image sizes: `builder-arm64` 3.7 GB, `toolchain-arm64` 9.1 GB, `pjrt-wheel-arm64` 26 GB (it
keeps the build tree for debugging).

Outputs:

```
tt-media-server/wheels/pjrt_plugin_tt-1.5.0.dev20260812001218-cp312-cp312-linux_aarch64.whl  (146 MB; x86 is 156 MB)
tt-media-server/wheels/vllm_tt-0.1-cp312-cp312-linux_aarch64.whl
```

### 3c. Wheel checks

```sh
docker run --rm -v "$PWD/tt-media-server/wheels:/w:ro" --entrypoint bash tt-forge-builder:builder-arm64 -c '
  cd /tmp && python3.12 -m zipfile -e /w/pjrt_plugin_tt-*.whl x >/dev/null
  grep -E "^(Version|Summary|Requires-Dist)" x/pjrt_plugin_tt-*.dist-info/METADATA
  grep Tag x/pjrt_plugin_tt-*.dist-info/WHEEL
  find x -name "*.so*" -type f | xargs file | grep -o -E "ARM aarch64|x86-64" | sort | uniq -c
  export LD_LIBRARY_PATH=/tmp/x/pjrt_plugin_tt/lib
  for f in x/pjrt_plugin_tt/pjrt_plugin_tt.so x/pjrt_plugin_tt/lib/*.so*; do ldd $f | grep "not found"; done
  find x -name "*.so*" -type f | xargs objdump -T | grep -o "GLIBC_[0-9.]*" | sort -V -u | tail -1
  find x -name "*.so*" -type f | xargs objdump -T | grep -o "GLIBCXX_[0-9.]*" | sort -V -u | tail -1'
```

Results:
- Version is `1.5.0.dev20260812001218` and the tag is `cp312-cp312-linux_aarch64`.
- `Summary` records the tt-xla, tt-mlir and tt-metal commits.
- `Requires-Dist` uses version pins (no x86 URLs).
- All 15 shared objects are `ARM aarch64`, and none has unresolved libraries.
- The highest versions required are `GLIBC_2.34` and `GLIBCXX_3.4.30`, both available on
  Ubuntu 22.04 (glibc 2.35, GCC 12).
- The entry points (`tt-forge-install`, the `jax_plugins` / `torch_xla.plugins` registrations)
  match the x86 wheel.

### 3d. On-device JAX test (no torch-xla needed)

This uses the `pjrt-wheel` image because it already has SFPI 7.67.0 at `/opt/tenstorrent/sfpi`.
The plugin is installed with `--no-deps` because torch-xla and tt_lang aren't built yet.

```sh
docker run --rm --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "$PWD/tt-media-server/wheels:/w:ro" --entrypoint bash tt-forge-builder:pjrt-wheel-arm64 -c '
  python3.12 -m venv /tmp/v && . /tmp/v/bin/activate && pip install -q --upgrade pip
  pip install -q jax==0.7.1 jaxlib==0.7.1 loguru requests pyyaml networkx pandas==3.0.0 seaborn graphviz click
  pip install -q --no-deps /w/pjrt_plugin_tt-*.whl
  cd /tmp && python -c "
import jax, jax.numpy as jnp, numpy as np
devs = jax.devices(\"tt\"); print(devs)
x = jnp.arange(32*32, dtype=jnp.float32).reshape(32, 32)
out = jax.jit(lambda a, b: a @ b + 1.0)(jax.device_put(x, devs[0]), jax.device_put(x, devs[0]))
ref = np.asarray(x) @ np.asarray(x) + 1.0
print(out.devices(), float(np.max(np.abs(np.asarray(out) - ref) / (np.abs(ref) + 1))))"'
```

Result:

```
devices: [TTDevice(id=0, arch=Blackhole), TTDevice(id=1, arch=Blackhole)]
result device: {TTDevice(id=0, arch=Blackhole)} max rel err: 6.713386028422974e-06
```

Kernels were JIT-compiled with the aarch64 SFPI, and fabric initialized on both devices. The only
warnings were about the unknown motherboard `MP32-AR0-JG` (tt-metal falls back to bus IDs), which
don't matter.

## Step 4: torch-xla wheel (Phase 1.5)

Stage `torch-xla-wheel` (`FROM builder`, so it doesn't depend on the tt-mlir stages). It builds
Tenstorrent's pytorch-xla fork at `ed8a4459` with Bazel 7.4.1 (arm64, through bazelisk and the
repo's `.bazelversion`) and GCC 12.

**Differences from upstream** (tt-xla `scripts/build_torch_xla.sh`, and pytorch-xla's
`_build_torch_xla_release.yml`):

- **No PyTorch source build.** The fork's Bazel `@torch` repo (`bazel/torch.BUILD`, path `../`)
  normally points at a built PyTorch tree. Instead, `/src/pytorch` is put together from:
  - a shallow `v2.11.0` source checkout, for `WORKSPACE` and the ATen yaml/template files that
    codegen reads (`torchgen_deps`)
  - the official aarch64 `torch-2.11.0+cpu` wheel's `torch/include`, copied to `torch/include`
  - that wheel's `torch/lib`, copied to `build/lib`
  - the wheel itself in `dist/`. OpenXLA's hermetic pip picks up `<workspace>/dist/*.whl` as
    `@pypi_torch`, which runs torchgen.

  So torch-xla is built against exactly the torch that pjrt-plugin-tt pins at runtime, and
  skips the roughly 1 to 2 hour PyTorch build. `setup.py` reads the C++ ABI flag from the
  installed torch.
- **Python 3.12 for `_XLAC`.** The first build linked `libpython3.10.so.1.0`.
  pybind11_bazel takes `python3-config` from next to the interpreter, falling back to next to
  its symlink target; in a venv that resolves to `/usr/bin/python3-config`, the system 3.10.
  The fix:
  - a `python3-config -> /usr/bin/python3.12-config` link in the venv
  - `PYTHON_CONFIG_BIN_PATH` (pybind11_bazel doesn't list it in the rule's `environ`, so
    changing it doesn't invalidate the cached repo)
  - a versioned Bazel cache-mount id (`torch-xla-bazel-py312`), so the stale config can't
    come back

  The stage now fails unless `_XLAC` links `libpython3.12`.

```sh
docker build -f tt-media-server/Dockerfile.forge-builder.arm64 --target torch-xla-wheel \
  -t tt-forge-builder:torch-xla-wheel-arm64 tt-media-server/scripts/forge_arm64 > /tmp/forge-torch-xla.log 2>&1

# re-export all wheels
docker build -f tt-media-server/Dockerfile.forge-builder.arm64 --target wheels \
  --output type=local,dest=tt-media-server/wheels tt-media-server/scripts/forge_arm64
```

Two attempts:

| # | Result | Bazel step |
|---|---|---|
| 1 | built, but `_XLAC` linked `libpython3.10.so.1.0` | 913 s |
| 2 | OK, `_XLAC` links `libpython3.12.so.1.0` | 892 s (cold cache, new mount id) |

Checks:

```sh
# Compare with the x86 release wheel
cd /tmp/forge-meta && python3 -m pip download --no-deps --only-binary=:all: --python-version 3.12 \
  --platform linux_x86_64 --extra-index-url https://pypi.eng.aws.tenstorrent.com "torch-xla==2.9.0+gited8a445"
unzip -o -q torch_xla-*x86_64.whl '_XLAC.cpython-312-x86_64-linux-gnu.so' -d x86
objdump -p x86/_XLAC.cpython-312-x86_64-linux-gnu.so | grep NEEDED

# The aarch64 one
docker run --rm -v "$PWD/tt-media-server/wheels:/w:ro" --entrypoint bash tt-forge-builder:builder-arm64 -c '
  cd /tmp && python3.12 -m zipfile -e /w/torch_xla-*.whl x >/dev/null
  grep -E "^(Version|Requires-Dist)" x/torch_xla-*.dist-info/METADATA; grep Tag x/torch_xla-*.dist-info/WHEEL
  objdump -p x/_XLAC.cpython-312-*.so | grep NEEDED'
```

Results:
- `torch_xla-2.9.0+gited8a445-cp312-cp312-linux_aarch64.whl` (88 MB; x86 is 107 MB).
- `Requires-Dist` is identical to x86.
- The `NEEDED` list is identical to x86 apart from the arch: `libc10`, `libtorch`, `libtorch_cpu`,
  `libtorch_python`, `libpython3.12.so.1.0`.
- The highest versions required are `GLIBC_2.35` and `GLIBCXX_3.4.30`, both on Ubuntu 22.04.

PyTorch/XLA on device (tt_lang left out with `--no-deps`):

```sh
docker run --rm --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "$PWD/tt-media-server/wheels:/w:ro" --entrypoint bash tt-forge-builder:pjrt-wheel-arm64 -c '
  python3.12 -m venv /tmp/v && . /tmp/v/bin/activate && pip install -q --upgrade pip
  pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu" jax==0.7.1 jaxlib==0.7.1 \
    loguru requests pyyaml networkx pandas==3.0.0 seaborn graphviz click absl-py nanobind==2.10.2
  pip install -q --no-deps /w/torch_xla-*.whl /w/pjrt_plugin_tt-*.whl
  cd /tmp && python -c "
import torch, torch_xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
xr.set_device_type(\"TT\"); dev = xm.xla_device(); print(dev, xr.global_runtime_device_count())
a, b = torch.randn(64, 128), torch.randn(128, 32)
m = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 16)).eval()
pcc = lambda x, y: torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()
print(pcc(torch.relu(a.to(dev) @ b.to(dev) + 1).cpu(), torch.relu(a @ b + 1)))
with torch.no_grad(): print(pcc(m.to(dev)(a.to(dev)).cpu(), m.cpu()(a)))"'
```

Result: `xla:0`, 2 devices. Matmul+ReLU PCC 1.0 and MLP PCC 1.0 (rounded to 5 digits).

## Step 5: Runtime image with the TT stack, end to end

`setup_env.sh`'s aarch64 path installs `$FORGE_WHEELS_DIR/*.whl` with `--no-deps`, then
resolves the wheels' `Requires-Dist` together with the forge requirements in a single pip run,
the same way the `tt-forge` install does on x86. It drops `tt_lang` with a warning when no
tt_lang wheel is present. It also removes the base image's SFPI (7.72.0) before installing
7.67.0.

```sh
docker build --platform linux/arm64 -f tt-media-server/Dockerfile.forge.arm64 \
  --build-arg FORGE_WHEELS=wheels -t tt-media-server-forge:arm64-tt . > /tmp/forge-arm64-tt-build.log 2>&1
```

The build takes about 8 minutes; the image is 16.4 GB (the released x86 0.18.0 image is 17.2 GB).

Resolved versions match the x86 pins: torch `2.11.0+cpu`, jax/jaxlib `0.7.1`, transformers
`5.14.1`, vllm `0.26.0`. pip's only complaint is `pjrt-plugin-tt requires tt_lang ..., which is
not installed`. vllm also brings in CUDA Python packages (`cuda-bindings`, `triton`, `nvidia-*`);
the released x86 0.18.0 image has the same ones.

### 5a. In-image device check

```sh
docker run --rm --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G tt-media-server-forge:arm64-tt bash -c '
  cd server && source venv-worker/bin/activate
  echo "$(id -un) $TT_METAL_HOME"; /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version | head -1
  python -c "
import torch, torch_xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
xr.set_device_type(\"TT\"); dev = xm.xla_device(); x, w = torch.randn(32, 64), torch.randn(64, 32)
print(dev, torch.corrcoef(torch.stack([(x.to(dev) @ w.to(dev)).cpu().flatten(), (x @ w).flatten()]))[0, 1].item())"'

# vllm platform plugin (let vllm discover it; importing vllm_tt directly first causes a
# harmless circular-import error)
docker run --rm tt-media-server-forge:arm64-tt bash -c 'cd server && source venv-worker/bin/activate &&
  VLLM_TARGET_DEVICE=empty python -c "from vllm.platforms import current_platform; print(type(current_platform).__name__)"'
```

Results:
- It runs as `container_app_user`, with `TT_METAL_HOME` pointing at the bundled tt-metal and SFPI
  `7.67.0[761]`.
- `xla:0`, PCC 1.0.
- vllm's platform resolves to `TTPlatform`.

### 5b. tt-media-server ResNet on device 0

```sh
docker run -d --name forge-arm64-resnet --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G -p 127.0.0.1:8123:8000 \
  -e MODEL_RUNNER=tt-xla-resnet -e DEVICE_IDS="(0)" tt-media-server-forge:arm64-tt

# request body: PyTorch Hub's sample image (a Samoyed), base64 in "prompt"
# (the README's curl example sends text, which this endpoint rejects)
curl -fsSL -o /tmp/dog.jpg https://github.com/pytorch/hub/raw/master/images/dog.jpg
python3 -c "import base64, json; json.dump({'prompt': base64.b64encode(open('/tmp/dog.jpg','rb').read()).decode(), 'top_k': 3, 'min_confidence': 0}, open('/tmp/dog-req.json','w'))"

# returns 405 "Model is not ready" until warmup finishes (~104 s), then:
curl -s -X POST http://127.0.0.1:8123/v1/cnn/search-image -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' -d @/tmp/dog-req.json

docker logs forge-arm64-resnet > /tmp/forge-arm64-resnet-server.log 2>&1; docker rm -f forge-arm64-resnet
```

Result:

```json
{"image_data":[{"top1_class_label":"Samoyed, Samoyede","top1_class_probability":"94.0000%",
  "output":{"labels":["Samoyed, Samoyede","Pomeranian","keeshond"],
  "probabilities":["94.0000%","1.7422%","0.7383%"],"indices":[258,259,261]}}],"status":"success"}
```

- The model loaders come from `fetch_models.sh` at startup, so the container needs network access.
- `[warmup] ... Forge model warmup` took 104 s (it was 104.4 s on a second run).
- Warm requests take about 0.40 s end to end (HTTP, base64 decode, preprocessing, device run).
- The only `ERROR` lines are the 405s from polling during warmup. There are no tracebacks.
  A case-insensitive grep for "exception" does match, but only on the SFPI compiler's
  `-fno-exceptions` flag in the kernel build commands.

## Not done yet

- **tt_lang** `1.1.5.dev20260704+light`. It's a compiled package with its own LLVM/MLIR and
  tt-metal build, and its version has to match tt-metal. It's only needed for custom tt-lang
  kernels (`@tt_torch.tt_lang_operation`, run by the `TTNNResolveTTLangKernels` pass). The forge
  runners don't use `tt_torch`, so the image works without it.
- The other runners (sdxl, vovnet, LoRA, ...) haven't been tested on device yet.
- The wheels end up in the runtime image twice: in the `COPY` of `tt-media-server/` and
  installed in the venv (about 230 MB).
