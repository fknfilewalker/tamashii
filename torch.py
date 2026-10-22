"""PyTorch interop: device creation and zero-copy tensor binding via DLPack."""
import ctypes
from typing import Any, Sequence

from tamashii._spy import spy


def create_torch_device(
    type: spy.DeviceType = spy.DeviceType.automatic,
    torch_device: Any = None,
    **kwargs: Any,
) -> spy.Device:
    """spy.create_torch_device() extended to Metal: it always initializes CUDA, while torch's
    MPS backend shares the system default MTLDevice with SlangPy instead."""
    import sys
    import torch

    if type == spy.DeviceType.automatic and sys.platform == "darwin":
        type = spy.DeviceType.metal
    if type != spy.DeviceType.metal:
        return spy.create_torch_device(type=type, torch_device=torch_device, **kwargs)

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS backend is not available, so it cannot share a Metal device")
    if torch_device is not None and torch.device(torch_device).type != "mps":
        raise ValueError(f"Metal devices require an MPS torch device, got '{torch_device}'")

    # Torch and SlangPy allocate from the same MTLDevice, so there is no CUDA context to
    # import and no interop layer to enable.
    return spy.create_device(type=type, **kwargs)


def _tensor_binding_layout(
    shape: Sequence[int],
    strides: Sequence[int],
    element_shape: Sequence[int],
    source: str,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Collapse trailing scalar dimensions into one Tamashii element."""
    shape = tuple(int(size) for size in shape)
    strides = tuple(int(stride) for stride in strides)
    element_shape = tuple(int(size) for size in element_shape)

    if len(shape) != len(strides):
        raise ValueError(f"{source} tensor shape and strides have different ranks")
    if any(size < 0 for size in shape):
        raise ValueError(f"{source} tensor dimensions cannot be negative")
    if any(stride < 0 for stride in strides):
        raise ValueError(f"{source} tensor strides cannot be negative")
    if any(size <= 0 for size in element_shape):
        raise ValueError("element_shape dimensions must be positive")
    if len(element_shape) > len(shape) or (
        element_shape and shape[-len(element_shape):] != element_shape
    ):
        raise ValueError(f"{source} tensor's trailing shape does not match element_shape")

    element_width = 1
    if element_shape:
        for size, stride in zip(reversed(element_shape), reversed(strides)):
            if stride != element_width:
                raise ValueError(f"{source} tensor elements must be contiguous")
            element_width *= size
        outer_shape = shape[:-len(element_shape)]
        outer_strides = strides[:-len(element_shape)]
    else:
        outer_shape = shape
        outer_strides = strides

    if any(stride % element_width for stride in outer_strides):
        raise ValueError(f"{source} tensor strides do not align to whole elements")
    return outer_shape, tuple(stride // element_width for stride in outer_strides), element_width


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLPACK_DELETER),
]

# Live exports, keyed by DLManagedTensor address. Holds the producer alive until the
# consumer calls the deleter, which it does when the imported tensor is destroyed.
_dlpack_exports: dict[int, tuple] = {}


def _dlpack_tensor(capsule: object) -> _DLTensor:
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, b"dltensor")
    if not pointer:
        raise ValueError("Tensor returned an invalid DLPack capsule")
    return ctypes.cast(pointer, ctypes.POINTER(_DLTensor)).contents


# DLPack device type -> the native buffer handle SlangPy imports it as. CUDA exports the
# device pointer itself, Metal the MTLBuffer holding the allocation.
_DLPACK_BUFFER_HANDLES = {
    2: spy.NativeHandleType.CUdeviceptr,  # kDLCUDA
    8: spy.NativeHandleType.MTLBuffer,  # kDLMetal
}


class DLPackTensorBinding:
    """Zero-copy binding from a DLPack tensor to Tamashii ``Tensor<T, D>``."""

    def __init__(self, device: spy.Device, tensor: object, element_shape: Sequence[int] = ()):
        if not hasattr(tensor, "__dlpack__") or not hasattr(tensor, "__dlpack_device__"):
            raise TypeError("Expected a tensor supporting DLPack")

        # Both owners must stay alive for the binding lifetime; the capsule frees the
        # export on GC.
        self.tensor: Any = tensor
        self._capsule = tensor.__dlpack__()
        dl_tensor = _dlpack_tensor(self._capsule)
        handle_type = _DLPACK_BUFFER_HANDLES.get(dl_tensor.device.device_type)
        if handle_type is None:
            raise ValueError(
                f"Tamashii cannot bind DLPack device type {dl_tensor.device.device_type}"
            )
        if not dl_tensor.data:
            raise ValueError("Tensor did not export a valid buffer")
        if dl_tensor.ndim < 0:
            raise ValueError("Tensor exported an invalid rank")

        shape = tuple(dl_tensor.shape[i] for i in range(dl_tensor.ndim))
        if dl_tensor.strides:
            scalar_strides = tuple(dl_tensor.strides[i] for i in range(dl_tensor.ndim))
        else:
            stride = 1
            reversed_strides = []
            for size in reversed(shape):
                reversed_strides.append(stride)
                stride *= size
            scalar_strides = tuple(reversed(reversed_strides))

        self.element_shape = tuple(element_shape)
        self.shape, self.strides, element_width = _tensor_binding_layout(
            shape, scalar_strides, self.element_shape, "DLPack"
        )

        dtype_bits = int(dl_tensor.dtype.bits) * int(dl_tensor.dtype.lanes)
        if dtype_bits <= 0 or dtype_bits % 8:
            raise ValueError("Tensor dtype is not byte-addressable")
        scalar_size = dtype_bits // 8
        if dl_tensor.byte_offset % (scalar_size * element_width):
            raise ValueError("Tensor offset does not align to a whole Tamashii element")
        self.offset = dl_tensor.byte_offset // (scalar_size * element_width)

        if any(size == 0 for size in shape):
            required_size = max(scalar_size, int(dl_tensor.byte_offset) + scalar_size)
        else:
            last_scalar = sum((size - 1) * stride for size, stride in zip(shape, scalar_strides))
            required_size = int(dl_tensor.byte_offset) + (last_scalar + 1) * scalar_size

        native_handle = spy.NativeHandle(handle_type, int(dl_tensor.data))
        self.buffer = device.create_buffer_from_native_handle(
            {
                "size": required_size,
                "usage": spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                "label": "DLPack tensor",
            },
            native_handle,
        )

    def get_this(self) -> dict[str, object]:
        """Return Tamashii's raw-pointer tensor fields for SlangPy cursor binding."""
        return {
            "_data": self.buffer.device_address,
            "_shape": list(self.shape),
            "_strides": list(self.strides),
            "_offset": self.offset,
        }


# Slang scalar type -> DLPack (type code, bits). Codes: 0 = int, 1 = uint, 2 = float.
_DLPACK_SCALAR_TYPES = {
    "float": (2, 32), "half": (2, 16), "double": (2, 64),
    "int": (0, 32), "int64_t": (0, 64), "int16_t": (0, 16), "int8_t": (0, 8),
    "uint": (1, 32), "uint64_t": (1, 64), "uint16_t": (1, 16), "uint8_t": (1, 8),
}


def _export_dlpack(owner: object, data: int, shape: Sequence[int], strides: Sequence[int],
                   byte_offset: int, code: int, bits: int) -> Any:
    """Wrap a native buffer in a DLPack capsule and import it as a torch tensor."""
    import torch

    managed = _DLManagedTensor()
    shape_array = (ctypes.c_int64 * len(shape))(*shape)
    stride_array = (ctypes.c_int64 * len(strides))(*strides)
    managed.dl_tensor.data = ctypes.c_void_p(data)
    managed.dl_tensor.device = _DLDevice(8, 0)  # kDLMetal
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
    managed.dl_tensor.shape = shape_array
    managed.dl_tensor.strides = stride_array
    managed.dl_tensor.byte_offset = byte_offset

    key = ctypes.addressof(managed)
    deleter = _DLPACK_DELETER(lambda _: _dlpack_exports.pop(key, None))
    managed.deleter = deleter
    _dlpack_exports[key] = (managed, shape_array, stride_array, deleter, owner)

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule_new.restype = ctypes.py_object
    try:
        return torch.from_dlpack(capsule_new(ctypes.byref(managed), b"dltensor", None))
    except BaseException:
        _dlpack_exports.pop(key, None)
        raise


def to_torch(tensor: spy.Tensor) -> Any:
    """View a SlangPy tensor as a ``torch.Tensor`` without copying.

    SlangPy's own ``Tensor.to_torch()`` only maps CUDA memory, so on Metal the storage
    ``MTLBuffer`` is handed to torch as a DLPack tensor instead.
    """
    if tensor.device.info.type != spy.DeviceType.metal:
        return tensor.to_torch()

    scalar_type = tensor.dtype.element_type or tensor.dtype
    scalar = _DLPACK_SCALAR_TYPES.get(scalar_type.full_name)
    if scalar is None:
        raise ValueError(f"Cannot view slang type '{tensor.dtype.full_name}' as a torch tensor")
    code, bits = scalar

    # A Tamashii element (e.g. float3) becomes trailing scalar dimensions on the torch side,
    # matching the shape Tensor.to_numpy() reports.
    element_shape = tuple(tensor.dtype.shape.as_list())
    element_width = 1
    element_strides = []
    for size in reversed(element_shape):
        element_strides.append(element_width)
        element_width *= size
    element_strides = tuple(reversed(element_strides))

    shape = tuple(tensor.shape.as_list()) + element_shape
    strides = tuple(s * element_width for s in tensor.strides.as_list()) + element_strides
    handle = tensor.storage.native_handle
    if handle.type != spy.NativeHandleType.MTLBuffer:
        raise ValueError(f"Expected an MTLBuffer storage handle, got {handle.type}")
    return _export_dlpack(
        tensor,
        handle.value,
        shape,
        strides,
        tensor.offset * element_width * (bits // 8),
        code,
        bits,
    )


def torch_tensor(
    tensor: object,
    element_shape: Sequence[int] = (),
    *,
    device: spy.Device,
) -> DLPackTensorBinding:
    """Bind a CUDA or MPS ``torch.Tensor`` to Tamashii without copying.

    The Tamashii ``device`` imports the tensor's allocation, which is what makes its
    address usable from a shader: registered for residency on Metal, aliased on CUDA.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if tensor.layout != torch.strided:
        raise ValueError("Tamashii direct tensor binding requires a strided tensor")
    if tensor.device.type == "mps":
        # PyTorch and SlangPy use separate Metal queues.
        torch.mps.synchronize()
    elif tensor.device.type != "cuda":
        raise ValueError("Tamashii direct tensor binding requires a CUDA or MPS tensor")
    return DLPackTensorBinding(device, tensor, element_shape)
