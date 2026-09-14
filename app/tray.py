"""SnapAssist tray app + global hotkey (Phase 2).

Minimal shell: a system-tray icon, a global hotkey to start/stop recording, and a window that
shows the live transcript + Summarize / Draft reply / Ask buttons. Backed by engine.pipeline.

Stack decision (see docs/BUILD_PLAN.md): try PyQt6 first. If PyQt6 wheels are unavailable on
Windows ARM64, keep this file thin and move the UI to a local FastAPI + webview fallback — the
engine layer does not change either way.
"""

from __future__ import annotations


def main() -> None:
    # TODO Phase 2:
    #   - system tray icon (PyQt6 QSystemTrayIcon) with Start/Stop/Quit
    #   - global hotkey (keyboard lib) -> toggle recording
    #   - window: transcript pane, [Summarize] [Draft reply] [Ask], NPU badge, airplane indicator
    #   - wire buttons to engine.pipeline.MeetingPipeline
    raise NotImplementedError("Phase 2: tray app. Engine is ready to call from here.")


if __name__ == "__main__":
    main()
