# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#!/bin/bash
set -eo pipefail
# filepath: /localdev/idjuric/tt-inference-server/tt-media-server/scripts/simple_setup.sh

echo "Reseting environment..."
unset TT_METAL_HOME
unset PYTHONPATH
unset WH_ARCH_YAML
unset ARCH_NAME

virtual_env_name="venv-worker"
# Create virtual environment with available Python version
if command -v python3.12 >/dev/null 2>&1; then
    echo "Using python3.12 for virtual environment..."
    python3.12 -m venv ${virtual_env_name}
else
    echo "Error: No suitable Python version found!"
    exit 1
fi

# Activate virtual environment
source ${virtual_env_name}/bin/activate

# Set environment variables for vllm build
export VLLM_TARGET_DEVICE="empty"

# Install requirements
pip install --upgrade pip

# Install root requirements if exists
if [ -f "requirements.txt" ]; then
    # Dissable pip's package cache, reducing disk usage during installation and Docker size
    pip install --no-cache-dir -r requirements.txt
fi

# Tenstorrent publishes tt-forge / pjrt-plugin-tt / torch-xla / tt-lang for x86_64 only.
# On aarch64, install locally built wheels from FORGE_WHEELS_DIR instead of the tt-forge pin
# (or skip the TT stack entirely when FORGE_WHEELS_DIR is unset/empty).
# See docs/arm64-forge-image-plan.md.
install_sfpi_aarch64() {
    # tt-forge-install hardcodes sfpi_arch=x86_64, so fetch the aarch64 package ourselves.
    local sfpi_file
    sfpi_file="$(python -c 'import pjrt_plugin_tt, os; print(os.path.join(os.path.dirname(pjrt_plugin_tt.__file__), "tt-metal/tt_metal/sfpi-version"))')"
    (
        source "${sfpi_file}"
        local deb="sfpi_${sfpi_version}_aarch64_debian.deb"
        curl -fsSL "${sfpi_repo}/releases/download/${sfpi_version}/${deb}" -o "/tmp/${deb}"
        echo "${sfpi_aarch64_debian_deb_hash}  /tmp/${deb}" | sha256sum -c -
        # Don't mix files with a different pre-installed SFPI (tt-metalium base images ship one).
        rm -rf /opt/tenstorrent/sfpi
        apt-get install -y --allow-downgrades "/tmp/${deb}"
        rm -f "/tmp/${deb}"
    )
}

# Print the Requires-Dist entries of the given wheels, minus packages the wheels themselves
# provide. tt_lang (a pjrt-plugin-tt dependency only needed for custom tt-lang kernels, which the
# forge runners don't use) is dropped with a warning when no tt_lang wheel is present.
tt_wheel_requirements() {
    python - "$@" <<'EOF'
import email, os, re, sys, zipfile

def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()

provided = {norm(os.path.basename(w).split("-")[0]) for w in sys.argv[1:]}
for whl in sys.argv[1:]:
    with zipfile.ZipFile(whl) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        reqs = email.message_from_bytes(z.read(meta)).get_all("Requires-Dist") or []
    for req in reqs:
        name = norm(re.split(r"[\s;<>=!~\[@(]", req, maxsplit=1)[0])
        if name in provided:
            continue
        if name == "tt-lang":
            print(f"WARNING: no tt_lang wheel; skipping '{req}' (needed by {os.path.basename(whl)} "
                  "only for custom tt-lang kernels)", file=sys.stderr)
            continue
        print(req)
EOF
}

if [ "$(uname -m)" = "aarch64" ] && [ -f "tt_model_runners/forge_runners/requirements.txt" ]; then
    grep -v '^tt-forge' tt_model_runners/forge_runners/requirements.txt > /tmp/forge-requirements-arm64.txt
    if compgen -G "${FORGE_WHEELS_DIR:-/nonexistent}/*.whl" >/dev/null; then
        echo "aarch64: installing TT wheels from ${FORGE_WHEELS_DIR}"
        pip install --no-cache-dir --no-deps "${FORGE_WHEELS_DIR}"/*.whl
        # Resolve the wheels' dependencies together with the forge requirements, like the
        # single tt-forge install does on x86.
        tt_wheel_requirements "${FORGE_WHEELS_DIR}"/*.whl >> /tmp/forge-requirements-arm64.txt
    else
        echo "aarch64: FORGE_WHEELS_DIR has no wheels; skipping tt-forge (no TT device support in this venv)"
    fi
    pip install --no-cache-dir -r /tmp/forge-requirements-arm64.txt
    pip install --no-cache-dir --no-deps --no-build-isolation -r tt_model_runners/forge_runners/requirements-no-build-isolation.txt
    if python -c 'import pjrt_plugin_tt' 2>/dev/null; then
        install_sfpi_aarch64
    fi
# Install forge requirements if exists
elif [ -f "tt_model_runners/forge_runners/requirements.txt" ]; then
    # Dissable pip's package cache, reducing disk usage during installation and Docker size
    pip install --no-cache-dir -r tt_model_runners/forge_runners/requirements.txt
    if [ -f "tt_model_runners/forge_runners/requirements-no-build-isolation.txt" ]; then
        pip install --no-cache-dir --no-deps --no-build-isolation -r tt_model_runners/forge_runners/requirements-no-build-isolation.txt
    fi
    tt-forge-install
fi

echo "Setup complete in virtual environment ${virtual_env_name}."
echo "To activate, run: source ${virtual_env_name}/bin/activate"
