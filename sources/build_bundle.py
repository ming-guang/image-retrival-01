#!/usr/bin/env python3
"""Pack the GUI and the C++ engine into ONE self-contained binary.

Uses PyInstaller. Build on the OS you want to target -- PyInstaller cannot
cross-compile, so run this on Linux to make ./releases/image-retrieval-gui and
on Windows to make releases\\image-retrieval-gui.exe. The engine binary and the
Tcl/Tk runtime data are embedded, so an end user launches a single file.

Prerequisites: the engine is already built (`make`), and pyinstaller + tkinter +
pillow are importable (they are inside the Nix dev shell: `nix develop`).

Note: the embedded engine still links against OpenCV at run time. On the build
machine / Nix dev shell those libraries are present; ship them alongside (or use
a static OpenCV build on Windows) if you run the binary on a bare machine.
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RELEASES = os.path.join(ROOT, "releases")
GUI = os.path.join(HERE, "gui", "image_retrieval_gui.py")
EXE = ".exe" if os.name == "nt" else ""
SEP = os.pathsep  # PyInstaller's SRC<sep>DEST separator (';' win, ':' posix)


def die(msg):
    sys.stderr.write("build_bundle: error: %s\n" % msg)
    sys.exit(1)


def tcl_data_dir():
    """Path to the Tcl script library (holds init.tcl); '' if not found.

    `Tcl().eval('info library')` works headless -- no Tk / display needed."""
    import tkinter
    d = tkinter.Tcl().eval("info library")
    return d if os.path.isfile(os.path.join(d, "init.tcl")) else ""


def tk_data_dir():
    """Path to the Tk script library (holds tk.tcl); '' if not found."""
    import tkinter
    ver = "tk%s" % tkinter.TkVersion  # e.g. "tk8.6"
    # Windows (and most distros): Tk data sits beside the Tcl data.
    tcl = tcl_data_dir()
    if tcl:
        cand = os.path.join(os.path.dirname(tcl), ver)
        if os.path.isfile(os.path.join(cand, "tk.tcl")):
            return cand
    # Nix/POSIX: Tk is a separate store path -- resolve the libtk the
    # interpreter is linked against; its data dir is the sibling <libdir>/tk<v>.
    try:
        import _tkinter
        out = subprocess.check_output(["ldd", _tkinter.__file__], text=True,
                                      stderr=subprocess.DEVNULL)
        m = re.search(r"=>\s*(\S*libtk\S*\.so\S*)", out)
        if m:
            cand = os.path.join(os.path.dirname(m.group(1)), ver)
            if os.path.isfile(os.path.join(cand, "tk.tcl")):
                return cand
    except Exception:
        pass
    return ""


def main():
    if not os.path.isfile(GUI):
        die("GUI script not found: %s" % GUI)
    engine = os.path.join(RELEASES, "image-retrieval" + EXE)
    if not os.path.isfile(engine):
        die("engine not built (%s) -- run `make` first." % engine)
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        die("PyInstaller not importable -- enter the dev shell (`nix develop`).")

    cmd = [sys.executable, "-m", "PyInstaller", "--onefile", "--clean",
           "--noconfirm", "--name", "image-retrieval-gui",
           "--distpath", RELEASES,
           "--workpath", os.path.join(tempfile.gettempdir(), "pyi-build"),
           "--specpath", os.path.join(tempfile.gettempdir(), "pyi-spec"),
           "--collect-all", "tkinter",
           # PIL's image codecs + the ImageTk bridge are loaded dynamically;
           # without collecting them the bundled GUI shows "(unreadable)".
           "--collect-all", "PIL",
           # PIL can *optionally* use numpy, so PyInstaller otherwise drags in
           # numpy + OpenBLAS/LAPACK (~78 MB). The GUI never touches numpy, so
           # excluding it (and the rest of the scientific stack) is safe and is
           # the single biggest size win.
           "--exclude-module", "numpy",
           "--exclude-module", "scipy",
           "--exclude-module", "matplotlib",
           "--exclude-module", "pandas",
           "--exclude-module", "PIL.ImageQt",  # pulls Qt if present
           "--add-binary", engine + SEP + "."]

    # Embed any DLLs the CI dropped beside the engine (Windows OpenCV +
    # runtime). The GUI prepends _MEIPASS to PATH so the engine finds them, so
    # the resulting .exe runs without OpenCV installed. No-op on Linux.
    for dll in sorted(glob.glob(os.path.join(RELEASES, "*.dll"))):
        cmd += ["--add-binary", dll + SEP + "."]

    tcl, tk = tcl_data_dir(), tk_data_dir()
    if tcl:
        cmd += ["--add-data", tcl + SEP + "_tcl_data"]
    if tk:
        cmd += ["--add-data", tk + SEP + "_tk_data"]
    if not (tcl and tk):
        sys.stderr.write("build_bundle: warning: could not locate the Tcl/Tk "
                         "data dir; the bundle may fail to start.\n")
    cmd.append(GUI)

    print("build_bundle: packing engine + Tcl/Tk into one binary via PyInstaller")
    subprocess.check_call(cmd)
    print("build_bundle: wrote %s" %
          os.path.join(RELEASES, "image-retrieval-gui" + EXE))


if __name__ == "__main__":
    main()
