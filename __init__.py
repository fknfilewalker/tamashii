from ._spy import spy

from .scene import *
from .interaction import *
from .optimizer import *
from . import renderpasses
from . import tev
from .torch import create_torch_device, torch_tensor, to_torch

from typing import Sequence

# make sure spy types are pickelable 🥒
spy.math.float4x4.__deepcopy__ = lambda self, memo: spy.math.float4x4(self) # type: ignore
spy.math.float3x3.__deepcopy__ = lambda self, memo: spy.math.float3x3(self) # type: ignore
spy.math.float4.__deepcopy__ = lambda self, memo: spy.math.float4(self) # type: ignore
spy.math.float3.__deepcopy__ = lambda self, memo: spy.math.float3(self) # type: ignore
spy.math.float2.__deepcopy__ = lambda self, memo: spy.math.float2(self) # type: ignore

# numpy-style item assignment for spy.Tensor: t[1] = row, t[:, 2] = 0.0, t[1:3, ::2] = block.
# __getitem__ already returns strided views natively; this adds the write side.
import numpy as _np

_SLANG_TO_NUMPY_DTYPE = {
    "float": _np.float32, "double": _np.float64, "half": _np.float16,
    "int": _np.int32, "uint": _np.uint32, "int64_t": _np.int64, "uint64_t": _np.uint64,
    "int16_t": _np.int16, "uint16_t": _np.uint16, "int8_t": _np.int8, "uint8_t": _np.uint8,
}

def _tensor_setitem(self: spy.Tensor, idx, value):
    view = self[idx]
    dtype = _SLANG_TO_NUMPY_DTYPE.get(view.dtype.full_name)
    if dtype is None and view.dtype.element_type is not None:  # vector elements, e.g. float3
        dtype = _SLANG_TO_NUMPY_DTYPE.get(view.dtype.element_type.full_name)
    if view.is_contiguous():
        # numpy-side shape includes the element dims: a (n,) float3 tensor holds (n, 3) floats
        shape = tuple(view.shape.as_list()) + tuple(view.dtype.shape.as_list())
        data = _np.ascontiguousarray(_np.broadcast_to(_np.asarray(value, dtype=dtype), shape))
        if not data.flags.writeable:  # broadcast views are read-only, which nanobind rejects
            data = data.copy()
        view.copy_from_numpy(data)
    else:
        # strided destination: read-modify-write through the contiguous base
        assert self.is_contiguous(), "cannot assign through a strided view of a non-contiguous tensor"
        data = self.to_numpy()
        data[idx] = value
        self.copy_from_numpy(data)

spy.Tensor.__setitem__ = _tensor_setitem # type: ignore

def include_paths() -> list:
    # shader search paths of the installed package
    import pathlib
    root = pathlib.Path(__file__).parent.absolute()
    return [root, root / "plugins", root / "renderpasses"]

def get_modules(device: spy.Device) -> Sequence[spy.SlangModule]:
    return [ device.load_module("tamashii.slang"), *core.get_plugin_modules(device) ]

def get_type_conformances() -> Sequence[spy.TypeConformance]:
    # Built-ins and externals are registered into each interface's TYPES via
    # register_plugin(); all_type_conformances() flattens them (index == id).
    return core.all_type_conformances()

def vk_or_mtl_device():
    import sys
    if sys.platform == "darwin":
        return spy.DeviceType.metal
    else:
        return spy.DeviceType.vulkan

