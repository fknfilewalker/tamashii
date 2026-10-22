"""
The repo root is the package. Every install builds ./slangpy into _slangpy, a private copy that
coexists with an installed slangpy; the rest of the configuration is in pyproject.toml.

The submodule is left untouched: its working tree is copied to build/vendor_slangpy/src, renamed
there (C++ module names, nanobind domain), built without GLFW, and the Python package of the
wheel is renamed on the way into _slangpy. Extra CMake flags pass through CMAKE_ARGS,
e.g. a local Slang build.
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "slangpy"
WORK = ROOT / "build" / "vendor_slangpy"
TARGET = ROOT / "_slangpy"
PACKAGE = "tamashii._slangpy"

# rewrites of slangpy's own Python files: (pattern, replacement); the Slang module `slangpy`
# (`import slangpy;` in shader strings, load_module("slangpy")) keeps its name
PY_RULES = [
    (re.compile(r"^([ \t]*)from slangpy(\.[\w.]+)? import", re.M), rf"\1from {PACKAGE}\2 import"),
    (re.compile(r"^([ \t]*)import slangpy(\.[\w.]+)? as ", re.M), rf"\1import {PACKAGE}\2 as "),
    (re.compile(r"^([ \t]*)import slangpy([ \t]*#.*)?$", re.M), rf"\1import {PACKAGE} as slangpy\2"),
    # `import slangpy.a.b` binds `slangpy` and loads a.b
    (re.compile(r"^([ \t]*)import slangpy\.([\w.]+)([ \t]*#.*)?$", re.M), rf"\1import {PACKAGE}.\2\n\1import {PACKAGE} as slangpy\3"),
    (re.compile(r"([\"'])slangpy\.slangpy_ext\1"), rf"\1{PACKAGE}.slangpy_ext\1"),
]
# the extension imports its Python side by name; `slangpy_torch` is a separate package
CPP_RULE = (re.compile(r'import_\("slangpy(?=[".])'), f'import_("{PACKAGE}')
STALE_IMPORT = re.compile(r"^\s*(from slangpy[. ]|import slangpy(\.|\s|$))", re.M)


def substitute(path: Path, pattern: re.Pattern, replacement: str) -> int:
    text = path.read_text(encoding="utf-8")
    text, count = pattern.subn(replacement, text)
    if count:
        path.write_text(text, encoding="utf-8")
    return count


def copy_source(src: Path):
    """The tracked files as in the working tree, with local changes. vcpkg stays in the submodule:
    it needs its git history."""
    if src.exists():
        shutil.rmtree(src)
    files = subprocess.run(["git", "ls-files", "-z", "--recurse-submodules"], cwd=SOURCE,
                           capture_output=True, check=True).stdout.decode().split("\0")
    for name in files:
        if not name or name.startswith("external/vcpkg/") or not (SOURCE / name).is_file():
            continue
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, src / name)


def patch_source(src: Path):
    cpp = sum(substitute(f, *CPP_RULE) for f in (src / "src" / "slangpy_ext").rglob("*.[ch]*"))
    # own nanobind type registry, or both slangpys would register the same C++ types
    cmake = substitute(src / "src" / "slangpy_ext" / "CMakeLists.txt",
                       re.compile(r"nanobind_add_module\(slangpy_ext NB_STATIC"), "nanobind_add_module(slangpy_ext NB_STATIC NB_DOMAIN tamashii")
    # stub generation imports the extension as `slangpy`, which it no longer is; done after install
    stubs = substitute(src / "src" / "slangpy_ext" / "CMakeLists.txt",
                       re.compile(r"if\(NOT SGL_ENABLE_ASAN AND NOT SGL_ENABLE_UBSAN\)"), "if(FALSE)")
    assert cpp > 0 and cmake == 1 and stubs == 1, f"slangpy changed: {cpp} C++ module imports, {cmake} nanobind_add_module, {stubs} stub blocks"
    print(f"patched {cpp} C++ module imports, the nanobind domain and stub generation")


def find_uv() -> str:
    try:  # in an isolated build environment, from the `uv` build requirement
        from uv import find_uv_bin
        return find_uv_bin()
    except ImportError:
        return "uv"


def build_wheel(src: Path, dist: Path) -> Path:
    if dist.exists():
        shutil.rmtree(dist)
    toolchain = SOURCE / "external" / "vcpkg" / "scripts" / "buildsystems" / "vcpkg.cmake"
    cmake_args = f"{os.environ.get('CMAKE_ARGS', '')} -DSGL_ENABLE_GLFW=OFF -DCMAKE_TOOLCHAIN_FILE={toolchain}"
    subprocess.run([find_uv(), "build", "--wheel", "--python", sys.executable, "--out-dir", str(dist), str(src)],
                   env={**os.environ, "CMAKE_ARGS": cmake_args.strip()}, check=True)
    return next(dist.glob("slangpy-*.whl"))


def install_package(wheel: Path):
    if TARGET.exists():
        shutil.rmtree(TARGET)
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if parts[0] != "slangpy" or len(parts) > 1 and parts[1] in ("tests", "benchmarks", "include"):
                continue
            if Path(info.filename).suffix in (".a", ".lib", ".dwarf", ".pdb"):  # for linking and debugging only
                continue
            out = TARGET.joinpath(*parts[1:])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(info))
            if mode := info.external_attr >> 16:  # keep libraries executable
                out.chmod(mode & 0o777)

    rewrites = sum(substitute(f, pattern, replacement) for f in [*TARGET.rglob("*.py"), *TARGET.rglob("*.pyi")]
                   for pattern, replacement in PY_RULES)
    stale = [f"{f.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}" for f in TARGET.rglob("*.py")
             for text in [f.read_text(encoding="utf-8")] for m in STALE_IMPORT.finditer(text)]
    assert not stale, f"imports of slangpy left: {stale}"
    print(f"rewrote {rewrites} imports into {TARGET.relative_to(ROOT)}")


def generate_stubs():
    """Typing stubs of the extension, as slangpy's CMake generates them, for the renamed modules."""
    cmake = (SOURCE / "src" / "slangpy_ext" / "CMakeLists.txt").read_text(encoding="utf-8")
    submodules = re.search(r"foreach\(submodule IN ITEMS ([^)]*)\)", cmake).group(1).split()
    stubgen = SOURCE / "external" / "nanobind" / "src" / "stubgen.py"
    postprocess = SOURCE / "tools" / "postprocess_stub.py"
    # an empty tamashii around the bundle: the real one patches slangpy types, which stubgen would record
    host = WORK / "stubgen"
    if host.exists():
        shutil.rmtree(host)
    (host / "tamashii").mkdir(parents=True)
    (host / "tamashii" / "__init__.py").touch()
    (host / "tamashii" / "_slangpy").symlink_to(TARGET, target_is_directory=True)
    for submodule in ["", *submodules]:
        stub = TARGET / submodule / "__init__.pyi"
        args = ["-p", str(SOURCE / "src" / "slangpy_ext" / "slangpy_stub_patterns.nb")] if not submodule else []
        subprocess.run([sys.executable, str(stubgen), "-q", "-P", "-i", str(host), "-m", ".".join(filter(None, [PACKAGE, submodule])),
                        "-o", str(stub), *args], check=True)
        subprocess.run([sys.executable, str(postprocess), f"--file={stub}", "--quiet", *(["--submodule"] if submodule else [])], check=True)
    print(f"generated {1 + len(submodules)} stubs")


def build_slangpy():
    if not (SOURCE / "CMakeLists.txt").exists():
        print("slangpy submodule missing (git submodule update --init --recursive): not bundling slangpy")
        return
    print("building the bundled slangpy, this takes a while (-v shows progress)")
    src = WORK / "src"
    copy_source(src)
    patch_source(src)
    install_package(build_wheel(src, WORK / "dist"))
    generate_stubs()


def packages() -> list[str]:
    """tamashii and its subpackages; examples/ and slangpy/ next to them are not part of it."""
    return ["tamashii", *("tamashii." + p for p in find_packages(where=str(ROOT), include=["renderpasses*", "_slangpy*"]))]


class BuildPy(build_py):
    def run(self):
        build_slangpy()
        self.packages = self.distribution.packages = packages()  # with the bundle that was just built
        super().run()

    def find_package_modules(self, package, package_dir):  # this file shares the package's folder
        return [m for m in super().find_package_modules(package, package_dir) if (m[0], m[1]) != ("tamashii", "setup")]


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # the bundle is native code: platform and Python specific wheels
        return True


setup(package_dir={"tamashii": "."}, packages=packages(), cmdclass={"build_py": BuildPy}, distclass=BinaryDistribution)
