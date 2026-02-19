# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

block_cipher = None

# 1. Define the Source Path
ffmpeg_root = r"C:\Users\natem\PycharmProjects\audioProject\ffmpeg-8.0-essentials_build\bin"

# 2. Define the binaries explicitly (Source Path, Destination Folder)
# We send them to "." (root) so your script finds them easily.
ffmpeg_binaries = [
    (os.path.join(ffmpeg_root, "ffmpeg.exe"), "."),
    (os.path.join(ffmpeg_root, "ffprobe.exe"), ".")
]

# SciPy: pull signal + stats
scipy_submodules = collect_submodules("scipy.signal") + collect_submodules("scipy.stats")
scipy_datas = collect_data_files("scipy", include_py_files=True)

# 3. Add your python files to datas (removed the old 'ffmpeg_bin' tuple to avoid confusion)
added_files = [
    ("filters.py", "."),
    ("warps.py", "."),
    ("clarity.py", "."),
    ("helpers.py", "."),
    ("stream.py", "."),
] + scipy_datas

# sounddevice libs
sd_binaries = collect_dynamic_libs("sounddevice")

a = Analysis(
    ["main.py"],
    pathex=[],
    # 4. Combine sounddevice libs with our new FFmpeg binaries
    binaries=sd_binaries + ffmpeg_binaries,
    datas=added_files,
    hiddenimports=[
        "numpy",
        "sounddevice",
        "pydub",
        "scipy.signal",
        "scipy.stats",
        "scipy.special._cdflib",
        "scipy.stats._stats",
        "scipy.stats._distributions",
    ] + scipy_submodules,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter", "notebook", "IPython"],
    cipher=block_cipher,
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NatesAudioPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="NatesAudioPro",
)