# PyInstaller spec — build on the Windows ARM64 device: pyinstaller SnapAssist.spec
from PyInstaller.utils.hooks import collect_all

datas = [("app/ui", "app/ui")]
binaries, hiddenimports = [], []
for pkg in ("onnxruntime", "transformers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["app/launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas,
    name="SnapAssist",
    console=False,
    onefile=True,
)
