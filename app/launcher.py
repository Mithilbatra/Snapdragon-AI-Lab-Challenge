"""Desktop launcher: start the local server and open the browser. Entry point for the packaged app.

Used by PyInstaller (see docs/PACKAGING.md). Resolves the UI directory whether running from
source or from a PyInstaller bundle (sys._MEIPASS).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser


def _bundle_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    # Ensure the app package can find its UI when frozen.
    os.chdir(_bundle_dir())
    port = int(os.getenv("SNAP_PORT", "8000"))
    url = f"http://127.0.0.1:{port}"

    def _open():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    from app.server import app
    print(f"SnapAssist running at {url} — close this window to quit.")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
