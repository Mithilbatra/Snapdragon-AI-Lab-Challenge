"""Build dist/SnapAssist.exe with PyInstaller. Run ON the Windows ARM64 device.

    pip install pyinstaller
    python scripts/build_exe.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sep = ";" if sys.platform.startswith("win") else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "SnapAssist",
        "--onefile",
        "--noconsole",
        "--add-data", f"{ROOT / 'app' / 'ui'}{sep}app/ui",
        "--collect-all", "onnxruntime",
        str(ROOT / "app" / "launcher.py"),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
