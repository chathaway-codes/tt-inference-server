#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Patch a tt-xla checkout (3cfcfcfd) so it can be built and packaged on aarch64.

Used by Dockerfile.forge-builder.arm64 (stage `pjrt-wheel`). Every edit must apply exactly;
the script fails loudly if upstream changed, so drift is noticed instead of silently ignored.

Edits:
  1. python_package/requirements.txt: replace x86_64-only direct wheel URLs
     (`torch@https://...x86_64.whl`) with plain version pins (`torch==2.11.0+cpu`), so the
     wheel's metadata resolves against aarch64 builds (PyTorch CPU index / --find-links).
  2. python_package/setup.py: allow overriding the wheel version via TTXLA_WHEEL_VERSION
     (upstream derives `0.1.<date>+dev.<sha>` from git; releases use e.g.
     1.5.0.dev20260812001218).
  3. python_package/setup.py: platform tag fallback also handles aarch64.
  4. python_package/setup.py: run CMake shell commands with bash. In CI mode
     (IN_CIBW_ENV=ON) it runs `source venv/activate && cmake ...`; /bin/sh is dash on Ubuntu.
  5. python_package/ttxla_tools/install_sfpi.py (the `tt-forge-install` script): use the host
     architecture instead of hardcoded x86_64.
  6. third_party/CMakeLists.txt: also pass -Wno-error=deprecated-declarations to tt-mlir.
     We build on Ubuntu 22.04, where clang uses GCC 12's libstdc++; its std::stable_sort calls
     the deprecated std::get_temporary_buffer, which trips tt-mlir's -Werror (GCC 13's
     libstdc++, used upstream on 24.04, suppresses that warning internally).
  7. pjrt_implementation/inc/utils/assert.h: use header-only {fmt} instead of C++20 <format>,
     which GCC 12's libstdc++ doesn't have (added in GCC 13). The header was adapted from
     tt-metal, which uses fmt; usage is plain `{}` format strings with no custom formatters.
     FMT_HEADER_ONLY avoids a runtime dependency on libfmt. Needs libfmt-dev at build time.

Usage: patch_tt_xla.py <tt-xla checkout>
"""

import re
import sys
from pathlib import Path
from urllib.parse import unquote


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        sys.exit(f"{path}: expected exactly 1 occurrence of {old!r}, found {count}")
    path.write_text(text.replace(old, new))
    print(f"patched {path}: {old.strip()[:60]!r}")


def pin_direct_urls(path: Path) -> None:
    # e.g. torch-xla@https://.../torch_xla-2.9.0%2Bgited8a445-cp312-cp312-linux_x86_64.whl
    pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_.-]+)@https://\S+/[A-Za-z0-9_.]+?-(?P<ver>[^-/]+)-cp312-cp312-\S*x86_64\.whl[ \t]*$",
        re.MULTILINE,
    )
    text = path.read_text()
    new_text, n = pattern.subn(
        lambda m: f"{m['name']}=={unquote(m['ver'])}", text
    )
    if n == 0:
        sys.exit(f"{path}: no x86_64 direct URL pins found")
    if "x86_64" in new_text:
        sys.exit(f"{path}: x86_64 references remain after patching")
    path.write_text(new_text)
    for line in new_text.splitlines():
        if "==" in line and not line.startswith("#"):
            print(f"  {path.name}: {line}")


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    pkg = root / "python_package"

    pin_direct_urls(pkg / "requirements.txt")

    setup_py = pkg / "setup.py"
    replace_once(
        setup_py,
        '        return "0.1." + date + "+dev." + short_hash\n',
        '        if os.environ.get("TTXLA_WHEEL_VERSION"):\n'
        '            return os.environ["TTXLA_WHEEL_VERSION"]\n'
        '        return "0.1." + date + "+dev." + short_hash\n',
    )
    replace_once(
        setup_py,
        '                plat = "linux_x86_64"\n',
        '                plat = "linux_x86_64"\n'
        '            elif machine in ("aarch64", "arm64"):\n'
        '                plat = "linux_aarch64"\n',
    )
    text = setup_py.read_text()
    n = text.count("shell=True,")
    if n < 3:
        sys.exit(f"{setup_py}: expected >= 3 'shell=True,' calls, found {n}")
    setup_py.write_text(text.replace("shell=True,", 'shell=True, executable="/bin/bash",'))
    print(f"patched {setup_py}: {n} shell calls now use /bin/bash")

    replace_once(
        root / "third_party" / "CMakeLists.txt",
        "-DCMAKE_CXX_FLAGS=-Wno-error=pass-failed ",
        '"-DCMAKE_CXX_FLAGS=-Wno-error=pass-failed -Wno-error=deprecated-declarations" ',
    )

    assert_h = root / "pjrt_implementation" / "inc" / "utils" / "assert.h"
    replace_once(
        assert_h,
        "#include <format>\n",
        "#ifndef FMT_HEADER_ONLY\n#define FMT_HEADER_ONLY\n#endif\n#include <fmt/format.h>\n",
    )
    text = assert_h.read_text()
    n = text.count("std::format")
    # 3x std::format(), 2x std::format_string, 1x in the header comment
    if n != 6:
        sys.exit(f"{assert_h}: expected 6 std::format occurrences, found {n}")
    assert_h.write_text(text.replace("std::format", "fmt::format"))
    print(f"patched {assert_h}: {n} std::format -> fmt::format")

    install_sfpi = pkg / "ttxla_tools" / "install_sfpi.py"
    replace_once(install_sfpi, "import re\n", "import platform\nimport re\n")
    replace_once(
        install_sfpi, '    sfpi_arch = "x86_64"\n', "    sfpi_arch = platform.machine()\n"
    )


if __name__ == "__main__":
    main()
