# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Native aarch64 rebuild of the amd64 tt-model package changh95/qwen3.8-27b-p150x2
# (image tt-model/qwen3.8-27b-p150x2:85f009466e2e): Qwen/Qwen3.8-27B on 2x Blackhole P150 with vLLM,
# profiles tp2 (default), p1d1, tp2-dflash2. Same pins, same image layout (/opt/tt-metal, /opt/tt-venv,
# user tt, entrypoint, serve scripts) and the same Python package versions as the validated x86 venv.
# See docs/arm64-qwen3.8-27b-p150x2-plan.md. Deviations from the x86 image are marked "arm64:".
#
#   docker build -f vllm-tt-metal/qwen3.8-27b-p150x2.arm64/Dockerfile \
#     -t tt-model/qwen3.8-27b-p150x2:arm64 vllm-tt-metal/qwen3.8-27b-p150x2.arm64

ARG BASE_IMAGE=tt-metalium-release:local

# ==============================================================================
# BUILDER: tt-metal (C++ + Python bindings), the Python 3.12 venv, vLLM, the TT plugin
# ==============================================================================
FROM ${BASE_IMAGE} AS builder

ARG TT_METAL_SHA=aa20afaf172ac35370b8d112edc0f771ee6d2571
# = the x86 image's ttnn version (setuptools_scm on the qwen36-p150x2 branch); pinned so the dist
# metadata matches regardless of which tags the clone sees.
ARG TT_METAL_SCM_VERSION=0.65.2.dev9815
ARG PLUGIN_SHA=ab5f7f3d795ee519b5d96f41f31cc1e0f635d6a7
ARG VLLM_VERSION=0.26.0
ARG BUILD_JOBS=48
ARG DEBIAN_FRONTEND=noninteractive

# arm64: the base activates its own ttnn venv (/opt/venv, Python 3.10) through PATH/VIRTUAL_ENV/
# PYTHON_ENV_DIR. Point all three at the 3.12 venv instead: build_metal.sh binds the Python extension
# to whatever `python3` is first on PATH, so the venv must exist and be active before the C++ build.
ENV PATH=/opt/tt-venv/bin:/opt/openmpi-v5.0.7-ulfm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/tt-venv \
    PYTHON_ENV_DIR=/opt/tt-venv \
    UV_PYTHON_INSTALL_DIR=/usr/local/share/uv \
    UV_HTTP_RETRIES=10 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    TT_METAL_HOME=/opt/tt-metal \
    TT_METAL_RUNTIME_ROOT=/opt/tt-metal \
    PYTHONPATH=/opt/tt-metal \
    LD_LIBRARY_PATH=/opt/tt-metal/build/lib:/opt/openmpi-v5.0.7-ulfm/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# The venv the way tt-metal's create_venv.sh makes it (uv-managed CPython, relocatable), but empty:
# its packages come from the x86 lock below, not from requirements-dev.txt.
RUN uv python install 3.12 \
    && uv venv --link-mode copy --relocatable --managed-python --python 3.12 /opt/tt-venv \
    && /opt/tt-venv/bin/python --version \
    && cat /opt/tt-venv/pyvenv.cfg

# Full history (blobless) so setuptools_scm / git describe behave as in a normal checkout; the editable
# ttnn install also asserts that .git exists.
RUN git clone --filter=blob:none https://github.com/tenstorrent/tt-metal.git /opt/tt-metal \
    && cd /opt/tt-metal \
    && git checkout ${TT_METAL_SHA} \
    && git submodule update --init --recursive

# arm64: build_metal.sh defaults to the x86_64 toolchain (-march=x86-64-v3); use the aarch64 one.
# Only build_Release/lib is kept (= the x86 image); objects and the CPM cache are dropped in this layer.
RUN cd /opt/tt-metal \
    && python3 -c 'import sys; print(sys.executable, sys.version)' \
    && CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} ./build_metal.sh --build-type Release \
        --toolchain-path cmake/aarch64-linux-clang-20-libstdcpp-toolchain.cmake \
    && (ls -la build build_Release/lib ttnn/ttnn/_ttnn.so runtime || true) \
    && find build_Release -mindepth 1 -maxdepth 1 ! -name lib -exec rm -rf {} + \
    && rm -rf .cpmcache \
    && (du -sh /opt/tt-metal build_Release/lib runtime || true)

# The validated x86 venv, pinned exactly (--no-deps: no resolver, so vllm-overrides.txt's
# transformers<5.15 cap can't pull the version away from the 5.17.0 that was validated).
# torch/torchvision/torchcodec +cpu come from the PyTorch CPU index (the PyPI aarch64 torch is a CUDA build).
COPY requirements-x86-lock.txt /tmp/requirements-x86-lock.txt
RUN uv pip install --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements-x86-lock.txt

# ttnn, editable at /opt/tt-metal (as in the x86 image). Needs .git; the trim step removes it afterwards.
RUN cd /opt/tt-metal \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN=${TT_METAL_SCM_VERSION} uv pip install --no-deps -e . \
    && python -c 'import ttnn; print("ttnn", ttnn.__file__)'

# vLLM from the sdist with the empty target, as the plugin's docs/install-vllm-tt.sh does. Its deps
# (requirements/common.txt) are already in the lock. Its Rust extensions are optional and are skipped
# (no cargo here); the x86 build shipped without them too.
RUN VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm \
        --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
        vllm==${VLLM_VERSION} \
    && python -c 'import vllm; print("vllm", vllm.__version__, vllm.__file__)'

# The TT plugin, non-editable (the x86 image installed it from a temporary clone too).
RUN git clone https://github.com/changh95/vllm-tt-plugin /tmp/vllm-tt-plugin \
    && cd /tmp/vllm-tt-plugin \
    && git checkout ${PLUGIN_SHA} \
    && uv pip install --no-deps /tmp/vllm-tt-plugin \
    && cd / \
    && rm -rf /tmp/vllm-tt-plugin \
    && uv pip freeze > /opt/tt-venv/arm64-venv-freeze.txt \
    && (uv pip check || true) \
    && (python -m compileall -j 0 -q /opt/tt-venv/lib > /dev/null 2>&1 || true)

# Trim /opt/tt-metal to what the x86 image shipped: build_Release/lib (+ the build symlink), runtime/,
# tt_metal/, ttnn/, tools/, setup.py, pyproject.toml, and the tt-model.yaml `source.code` allowlist of models/.
RUN set -eux; cd /opt/tt-metal; \
    keep=/tmp/models-keep; \
    for p in models/common models/tt_transformers models/demos/blackhole/qwen36 \
             models/demos/qwen3_vl/tt/common.py models/demos/utils \
             models/experimental/gated_attention_gated_deltanet models/perf \
             models/tt_dit/utils/tensor.py models/model_trace_region_sizes.yaml models/model_targets.yaml; do \
        mkdir -p "$keep/$(dirname "$p")"; cp -a "$p" "$keep/$p"; \
    done; \
    rm -rf models; mv "$keep/models" models; \
    find . -mindepth 1 -maxdepth 1 ! -name build ! -name build_Release ! -name models ! -name runtime \
        ! -name tt_metal ! -name ttnn ! -name tools ! -name setup.py ! -name pyproject.toml ! -name README.md \
        -exec rm -rf {} +; \
    find . -name .git -prune -exec rm -rf {} +; \
    du -sh /opt/tt-metal/* /opt/tt-venv /usr/local/share/uv

# ==============================================================================
# RUNTIME
# ==============================================================================
# arm64: the x86 image sits on plain ubuntu:22.04 plus a copied OpenMPI; the arm64 base already has the
# same OS, the OpenMPI ULFM build at the same path, and the runtime libs the x86 image apt-installs
# (libhwloc15, libnuma1, libmpc3, libevent, ...), except libgl1/libsndfile1.
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# The x86 image's environment. PYTHON_ENV_DIR is blanked because the base points it at /opt/venv.
ENV VENV=/opt/tt-venv \
    VIRTUAL_ENV=/opt/tt-venv \
    PYTHON_ENV_DIR= \
    PATH=/opt/tt-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TT_METAL_RUNTIME_ROOT=/opt/tt-metal \
    TT_METAL_HOME=/opt/tt-metal \
    PYTHONPATH=/opt/tt-metal \
    LD_LIBRARY_PATH=/opt/tt-metal/build/lib:/opt/openmpi-v5.0.7-ulfm/lib \
    EXTRA_MODELS_DIR=/opt/tt-metal/models/demos/blackhole/qwen36/vllm_bundle \
    TT_VLLM_BUILTIN_MODELS=0 \
    TT_MODEL_KIND=vllm-plugin \
    HF_HOME=/hf \
    TT_METAL_CACHE=/cache

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libsndfile1 ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Same user setup as the x86 image: uid 1000 (= the usual host user, so bind mounts work with
# --user 1000:1000), world-writable home/work/cache for other --user values.
# arm64: the base's SHELL is bash -euo pipefail and it has no UID 1000, so getent exits 2 and would
# abort the RUN; "no such user" is an expected outcome here, hence || true.
RUN existing="$(getent passwd 1000 | cut -d: -f1 || true)" \
    && if [ -n "$existing" ]; then userdel -r "$existing" 2>/dev/null || userdel "$existing"; fi \
    && useradd --uid 1000 --create-home --home-dir /home/tt --shell /bin/bash tt \
    && mkdir -p /home/tt/work/logs /cache /opt/tt-metal \
    && chown -R tt:tt /home/tt /cache /opt/tt-metal \
    && chmod 1777 /home/tt /home/tt/work /home/tt/work/logs /cache

# The uv-managed CPython the relocatable venv links to (pyvenv.cfg home = /usr/local/share/uv/...).
COPY --from=builder /usr/local/share/uv /usr/local/share/uv
COPY --from=builder /opt/tt-venv /opt/tt-venv
COPY --from=builder --chown=tt:tt /opt/tt-metal /opt/tt-metal

COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chmod=0755 serve-tp2.sh serve-tp2-dflash2.sh serve-p1d1.sh /usr/local/bin/
# The x86 image's default CMD name, kept so commands written for it keep working.
RUN ln -s serve-tp2.sh /usr/local/bin/serve-default.sh

ENV HOME=/home/tt \
    USER=tt \
    LOGNAME=tt
USER tt
WORKDIR /home/tt/work

COPY verify.sh /ctx/verify.sh
RUN bash /ctx/verify.sh

USER root
RUN chmod -R a+rwX /home/tt
USER tt

LABEL org.tenstorrent.tt-model=qwen3.8-27b-p150x2 \
      org.tenstorrent.tt-model.repo=changh95/qwen3.8-27b-p150x2 \
      org.tenstorrent.tt-model.weights=Qwen/Qwen3.8-27B \
      org.tenstorrent.tt-model.arch=blackhole \
      org.tenstorrent.tt-model.kind=vllm-plugin \
      org.tenstorrent.tt-model.profiles=tp2,p1d1,tp2-dflash2 \
      org.opencontainers.image.revision=aa20afaf172ac35370b8d112edc0f771ee6d2571 \
      org.tenstorrent.tt-model.tt-metal=v0.79.0-dev20260903-120-gaa20afaf17 \
      org.tenstorrent.tt-model.plugin=ab5f7f3d795ee519b5d96f41f31cc1e0f635d6a7 \
      org.opencontainers.image.architecture=arm64 \
      org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/local/bin/serve-tp2.sh"]
