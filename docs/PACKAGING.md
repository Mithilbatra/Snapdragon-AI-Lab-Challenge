# Packaging SnapAssist for Windows ARM64

Goal: a double-clickable app for the HP Snapdragon PC that starts the local server and opens the
browser — no terminal needed. **Build on the ARM64 device** (PyInstaller does not cross-compile).

## Option A — PyInstaller one-file (recommended for the demo)
```powershell
.venv\Scripts\activate
pip install pyinstaller
python scripts\build_exe.py           # wraps the PyInstaller call below
```
This produces `dist\SnapAssist.exe`. It bundles `app/ui/*` as data and launches the server, then
opens `http://127.0.0.1:8000`. Model files in `models/` are loaded at runtime (keep them beside
the exe or set `SNAP_MODELS_DIR`).

Manual equivalent:
```powershell
pyinstaller --name SnapAssist --onefile app\launcher.py ^
  --add-data "app\ui;app\ui" ^
  --collect-all onnxruntime ^
  --noconsole
```

## Option B — Tauri shell (nicer, more work)
Wrap the same local server in a Tauri (Rust + WebView2) window for a native title bar and icon.
Ship the Python engine as a sidecar. Only pursue this after the PyInstaller build works.

## Notes
- Large model artifacts (ONNX / QNN context binaries, Genie bundle) are **not** embedded in the
  exe — distribute them alongside, or download on first run via `scripts/fetch_models.py`.
- Test the packaged app in **airplane mode** — that's the whole point.
- Code-signing is optional for the challenge but removes SmartScreen warnings.
