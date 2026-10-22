# The slangpy tamashii runs on: the bundled build (setup.py) if there is one,
# else the installed slangpy. Everything takes it from here, as `tm.spy`.
import importlib.util
import sys
from typing import TYPE_CHECKING

# type checkers see the bundle: they cannot evaluate find_spec and would take the fallback
if TYPE_CHECKING or importlib.util.find_spec("tamashii._slangpy") is not None:
    if "slangpy" in sys.modules:  # Linux and Windows load libraries by name, both would share them
        import warnings
        warnings.warn("slangpy is already imported, tamashii's bundled slangpy may clash with it")
    from tamashii import _slangpy as spy
else:
    import slangpy as spy
