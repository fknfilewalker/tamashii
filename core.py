from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, List, Sequence

from tamashii._spy import spy
import numpy.typing as npt
import numpy as np
_spy_PYTHON_TYPES = spy.bindings.typeregistry.PYTHON_TYPES
_spy_create_vr_type_for_value = spy.builtin.valueref.create_vr_type_for_value

class Ref(spy.ValueRef):
    def __init__(self, value: object):
        super().__init__(value)

    def __getattr__(self, attr):
        v = object.__getattribute__(self, "value")
        if self.slangpy_signature == '[dict]' and attr in v:
            return v[attr]
        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        if hasattr(self, "value"):
            v = object.__getattribute__(self, "value")
            if self.slangpy_signature == '[dict]' and attr in v:
                object.__getattribute__(self, "value")[attr] = value
            else:
                object.__setattr__(self, "value", value)
        else:
            object.__setattr__(self, attr, value)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "value"))

_spy_PYTHON_TYPES[Ref] = _spy_create_vr_type_for_value

class SlangStruct:
    def __init__(self, module: spy.Module, name: str):
        self._module = module
        self._name = name
        _type = module.find_struct(name)
        assert _type is not None, f"Struct {name} not found in module"
        self._type: spy.Struct = _type

    def __call__(self, *args, **kwargs):
        return Ref(self._type(*args, **kwargs))

    def __getattr__(self, attr):
        if attr.startswith("static_"):
            fn = self._module.find_function(f"{self._name}::{attr[7:]}")
        else:
            fn = self._module.find_function_in_struct(self._name, attr)
        if fn is None:
            raise AttributeError(f"'{self._name}' has no member '{attr}'")
        return lambda *args, **kwargs: Ref(fn(*args, **kwargs))

class Context:
    def __init__(self, device: spy.Device):
        self.device = device
        self.module = spy.Module(device.load_module("tamashii.slang"))
        self.structs = {}
        # core
        self.Ray = self.add_struct("Ray<float>")
        self.Texture3f = self.add_struct("Texture<float3>")
        self.Transform = self.add_struct("Transform<float>")
        # sensors
        self.Perspective = self.add_struct("sensors::Perspective<float>")
        # shapes
        self.AABB = self.add_struct("AABB<float>")
        self.Mesh = self.add_struct("shapes::Mesh<float>")
        self.Sphere = self.add_struct("shapes::Sphere<float>")
        self.Rectangle = self.add_struct("shapes::Rectangle<float>")
        # bsdfs
        self.Diffuse = self.add_struct("bsdfs::Diffuse<float>")
        self.Plastic = self.add_struct("bsdfs::Plastic<float>")
        self.Dielectric = self.add_struct("bsdfs::Dielectric<float>")
        # emitters
        self.Point = self.add_struct("emitters::Point<float>")
        self.Area = self.add_struct("emitters::Area<float>")
        self.Constant = self.add_struct("emitters::Constant<float>")
        self.HosekWilkie = self.add_struct("emitters::HosekWilkie<float>")

    def add_struct(self, name: str):
        s = SlangStruct(self.module, name)
        self.structs[name] = s
        return s

    def call(self, name: str, *args):
        fn = self.module.find_function(name)
        assert fn is not None, f"Function {name} not found in module"
        return Ref(fn(*args))

# Standalone Slang module paths contributed by external plugins, queued at registration
# time and loaded lazily once a device is available (built-ins live in tamashii.slang
# and contribute nothing here).
_plugin_module_paths: list[str] = []
_plugin_modules: list[spy.SlangModule] = []
_loaded_module_paths: set[str] = set()

def _conformance_base(cls: type) -> type:
    """The interface base in cls's MRO that owns the TYPES list."""
    for c in cls.__mro__:
        if "TYPES" in c.__dict__:
            return c
    raise TypeError(f"{cls.__name__} has no interface with a TYPES list")

def register_plugin(cls: type, type_name: str | list[str], module: str | None = None, interfaces: list[str] | None = None) -> int:
    """Register an implementation with its interface. Call once per implementation,
    after the class is defined. Idempotent. Returns the conformance id of `type_name`
    (the first, if a list).

    `cls` is either an implementation class (its TYPE is set, and the type is registered
    under its interface base) or an interface base itself, for Slang-only types that have
    no Python class (e.g. orphan integrators or template variants).

    `type_name` is the concrete Slang type, or a list of types for one class (template
    variants — the class then overrides implementation_type_name to pick one). Its index
    in the interface's TYPES list, ordered by registration, is the conformance id.

    `module` queues a standalone Slang module to link (external plugins only; built-ins
    are already compiled into tamashii.slang). `interfaces` lists extra Slang interfaces
    the type also conforms to with the same id (e.g. a sensor that is also a camera).

        register_plugin(Diffuse, "bsdfs::Diffuse<float>")                        # built-in
        register_plugin(Velvet, "bsdfs::Velvet<float>", module="velvet")         # external
        register_plugin(SHFilm, ["films::SH<...,false>", "films::SH<...,true>"]) # variants
        register_plugin(Perspective, "sensors::Perspective<float>",
                        interfaces=["sensors::ICamera<float>"])                  # dual
        register_plugin(Integrator, "integrators::Ptracer")                      # Slang-only
    """
    types = [type_name] if isinstance(type_name, str) else list(type_name)
    base = cls if "TYPES" in cls.__dict__ else _conformance_base(cls)
    if base is not cls:  # cls is an implementation class, not the interface base itself
        cls.TYPE = types[0] if len(types) == 1 else types
    for t in types:
        if t not in base.TYPES:
            base.TYPES.append(t)
    if interfaces:  # extra interfaces every type of this interface also satisfies, same id
        base.EXTRA_INTERFACES = tuple(dict.fromkeys((*base.EXTRA_INTERFACES, *interfaces)))
    if module is not None:
        _plugin_module_paths.append(module)
    return base.TYPES.index(types[0])

def get_plugin_modules(device: spy.Device) -> list[spy.SlangModule]:
    """Load any not-yet-loaded external plugin modules and return all of them."""
    for path in _plugin_module_paths:
        if path not in _loaded_module_paths:
            _loaded_module_paths.add(path)
            _plugin_modules.append(device.load_module(path))
    return _plugin_modules

def all_type_conformances() -> list["spy.TypeConformance"]:
    """Flatten every interface's TYPES list into TypeConformances.

    Each interface base contributes `(INTERFACE, type, index)` for every entry in its
    TYPES list, plus the same under each EXTRA_INTERFACES name (sharing the id — e.g. a
    sensor is also a camera).
    """
    result: list[spy.TypeConformance] = []
    seen: set[tuple[str, str]] = set()
    stack = list(Interface.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "TYPES" not in cls.__dict__ or not cls.INTERFACE:
            continue
        for idx, type_name in enumerate(cls.TYPES):
            for iface in (cls.INTERFACE, *cls.EXTRA_INTERFACES):
                if (iface, type_name) not in seen:
                    seen.add((iface, type_name))
                    result.append(spy.TypeConformance(iface, type_name, idx))
    return result

def get_composed_module(
    device: spy.Device,
    name: str = "tamashii_with_type_conformances",
) -> spy.SlangModule:
    """Compose Tamashii's modules and registered runtime conformances into one module."""
    return device.compose_modules(
        name=name,
        modules=[device.load_module("tamashii.slang"), *get_plugin_modules(device)],
        type_conformances=all_type_conformances(),
    )

class Transform(spy.float4x4):
    def __init__(self, m: 'spy.float4x4param' = spy.float4x4().identity()):
        super().__init__(m)

    def translate(self, v: 'spy.float3param'):
        return Transform(spy.math.translate(self, spy.float3(*v)))
    
    def scale(self, s: 'spy.float3param'):
        return Transform(spy.math.scale(self, spy.float3(*s)))
    
    def rotate(self, q: 'spy.quatfparam'):
        return Transform(spy.math.mul(self, spy.math.float4x4(spy.math.matrix_from_quat(spy.quatf(*q)))))
    
    def multiply(self, other: 'spy.float4x4param'):
        return Transform(spy.math.mul(self, other))
    
    def left_multiply(self, other: 'spy.float4x4param'):
        return Transform(spy.math.mul(other, self))

    def transpose(self):
        return Transform(spy.math.transpose(self))

def get_type_and_layout(program: spy.ShaderProgram, type_name: str):
    type = program.layout.find_type_by_name(type_name)
    layout = program.layout.get_type_layout(type).element_type_layout
    return type, layout

def get_layout(program: spy.ShaderProgram, type_name: str):
    return get_type_and_layout(program, type_name)[1]

def create_or_resize_buffer(device: spy.Device, buffer: spy.Buffer | None, data: npt.NDArray, label: str) -> spy.Buffer:
    if buffer is None or buffer.size < data.nbytes:
        buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label=label,
            data=data,
        )
    else:
        buffer.copy_from_numpy(data)
    return buffer

def create_or_resize_tensor(device: spy.Device, tensor: spy.Tensor | None, data: npt.NDArray, label: str) -> spy.Tensor:
    if tensor is None or tensor.storage.size < data.nbytes:
        tensor = spy.Tensor.from_numpy(device, data, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
    else:
        tensor.copy_from_numpy(data)
    return tensor

# This version of create_or_resize_tensor checks for exact size match instead of just size >= data.nbytes.
def create_or_resize_tensor_exact(device: spy.Device, tensor: spy.Tensor | None, data: npt.NDArray, label: str = "", usage: spy.BufferUsage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access) -> spy.Tensor:
    if tensor is None or tensor.storage.size != data.nbytes:
        tensor = spy.Tensor.from_numpy(device, data, usage=usage)
    else:
        tensor.copy_from_numpy(data)
    return tensor

# Typed variant: data's trailing element_ndim axes are the element, e.g. (n, 3) with
# dtype="float3" gives Tensor<float3, 1> of shape (n,), and (2, 4, 4) with dtype="float4x4"
# and element_ndim=2 gives Tensor<float4x4, 1> of shape (2,). Empty data returns None.
def create_or_resize_tensor_typed(device: spy.Device, tensor: spy.Tensor | None, data: npt.NDArray, dtype: str, usage: spy.BufferUsage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access, element_ndim: int = 1) -> spy.Tensor | None:
    if data.size == 0:
        return None
    shape = tuple(data.shape[:data.ndim - element_ndim])
    if tensor is None or tuple(tensor.shape.as_list()) != shape:
        tensor = spy.Tensor.empty(device, shape, dtype=dtype, usage=usage)
    tensor.copy_from_numpy(np.ascontiguousarray(data))
    return tensor

# Binds a Tensor<T, D> / DiffTensor<T, D> field. Returns the tensor: keep it on the caller,
# grads attached with .with_grads() survive across uploads because the object is reused.
def upload_tensor(program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor, name: str, tensor: spy.Tensor | None, data: npt.NDArray, dtype: str, element_ndim: int = 1) -> spy.Tensor | None:
    tensor = create_or_resize_tensor_typed(program.device, tensor, data, dtype, element_ndim=element_ndim)
    setattr(cursor, name, set_tensor(getattr(cursor, name), tensor))
    return tensor

def download_tensor(tensor: spy.Tensor | None) -> npt.NDArray | None:
    return None if tensor is None else tensor.to_numpy()

# A Transform is the (2,) float4x4 case: [0] = forward, [1] = inverse, read on the GPU with
# .to_transform(). An unbound tensor downloads as identity, as it reads there.
def upload_transform(program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor, name: str, tensor: spy.Tensor | None, matrix: 'spy.float4x4') -> spy.Tensor | None:
    matrices = np.stack([matrix.to_numpy(), spy.math.inverse(matrix).to_numpy()])
    return upload_tensor(program, cursor, name, tensor, matrices, "float4x4", element_ndim=2)

def download_transform(tensor: spy.Tensor | None, index: int = 0) -> 'spy.float4x4':
    data = download_tensor(tensor)
    if data is None:
        return spy.float4x4().identity()
    return spy.float4x4(data[index].flatten().tolist())

class Interface(ABC):
    # Slang interface name, set on each interface base (e.g. "IBsdf<float>").
    INTERFACE: str = ""

    # Per-interface conformance registry: an ordered list of implementation type names,
    # each entry's index is its conformance id. Populated by register_plugin(); each
    # interface base gets its own fresh list (see __init_subclass__), and implementations
    # look up their own id in it.
    TYPES: list[str] = []

    # The implementation's concrete Slang type name(s), set by register_plugin();
    # a list when the class registers several types (then override implementation_type_name).
    TYPE: str | list[str] = ""

    # Extra Slang interfaces every type of this interface also conforms to (same id),
    # set on the interface base via register_plugin(interfaces=...) — e.g. a sensor is
    # also a camera.
    EXTRA_INTERFACES: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Give each interface base (a direct subclass of Interface) its own TYPES list,
        # so implementations inherit and register into their interface's single registry.
        if Interface in cls.__bases__:
            cls.TYPES = []

    @property
    def interface_type_name(self) -> str:
        """The Slang interface name, the value of the interface base's INTERFACE."""
        return self.INTERFACE

    @property
    def implementation_type_name(self) -> str:
        """The concrete Slang type name, the value set by register_plugin. Override only when
        one class maps to several types (e.g. the SH film's hsh template variants);
        the default returns the first when register_plugin was given a list."""
        return self.TYPE if isinstance(self.TYPE, str) else self.TYPE[0]

    @property
    def conformance_id(self) -> int:
        """The id for this instance's concrete type: its index in the interface's TYPES.
        upload() writes this to `cursor.id` so the value follows the static registry."""
        return self.TYPES.index(self.implementation_type_name)

    name: str = ""

    @staticmethod
    def is_dynamic(cursor: spy.BufferElementCursor | spy.ShaderCursor) -> bool:
        """Whether the cursor points at a dyn slot, a ConformanceWrapper (types.slang). Cursors don't
        expose their type, so match its fields in declaration order: _ first, then __, id, ___, data."""
        def offset(c):  # BufferElementCursor offsets are ints, ShaderCursor ones ShaderOffsets
            return c._offset if isinstance(c._offset, int) else c._offset.uniform_offset
        fields = ("_", "__", "id", "___", "data")
        if not all(cursor.has_field(f) for f in fields):
            return False
        offsets = [offset(cursor[f]) for f in fields]
        return offsets[0] == offset(cursor) and all(a < b for a, b in zip(offsets, offsets[1:]))

    def upload(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        """Upload into a dyn slot (ConformanceWrapper: id + data) or a concrete one."""
        if self.is_dynamic(cursor):
            cursor.id = self.conformance_id
            cursor = cursor.data
        self.write(program, cursor)

    @abstractmethod
    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        """Write the concrete type's fields."""
        pass

    def download(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        """Download from a dyn slot (ConformanceWrapper: id + data) or a concrete one."""
        self.read(program, cursor.data if self.is_dynamic(cursor) else cursor)

    @abstractmethod
    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        """Read the concrete type's fields."""
        pass

T = TypeVar('T', bound=Interface)
class ResourceBuffer(Generic[T]):
    def __init__(self, program: spy.ShaderProgram, entries: List[T]):
        self.program: spy.ShaderProgram = program
        self.device: spy.Device = program.device
        self.entries: List[T] = entries
        self.stride: int =  program.layout.get_type_layout(program.layout.find_type_by_name(f"StructuredBuffer<{entries[0].interface_type_name}>")).element_type_layout.stride if len(entries) > 0 else 0
        self.buffer: spy.Buffer = self.device.create_buffer(
            size=max(self.stride * len(entries), 1),
            usage=spy.BufferUsage.shader_resource,
            label="ResourceBuffer",
        )

    def upload_entry(self, index: int):
        entry = self.entries[index]
        _, layout = get_type_and_layout(self.program, f"StructuredBuffer<ConformanceWrapper<{entry.implementation_type_name}>>")
        cursor = spy.BufferCursor(layout, self.buffer, size=self.stride, offset=index * self.stride, load_before_write=False)
        entry.upload(self.program, cursor[0])
        cursor.apply()
        
    def upload(self):
        size = max(self.stride * len(self.entries), 1)
        if self.buffer.size < size:
            self.buffer = self.device.create_buffer(
                size=size,
                usage=spy.BufferUsage.shader_resource,
                label="ResourceBuffer",
            )
        for i in range(len(self.entries)):
            self.upload_entry(i)

    def download(self):
        for i, entry in enumerate(self.entries):
            _, layout = get_type_and_layout(self.program, f"StructuredBuffer<ConformanceWrapper<{entry.implementation_type_name}>>")
            cursor = spy.BufferCursor(layout, self.buffer, size=self.stride, offset=i * self.stride, load_before_write=False)
            cursor.load()
            entry.download(self.program, cursor[0])

    def data_at(self, index: int) -> "Ref":
        """Return a Ref wrapping the concrete struct data of the entry at `index`.
        Reads cursor[0]["data"] (past the ConformanceWrapper id) via .read() so slangpy gets
        a dict/struct value it can marshal, not a raw BufferElementCursor.
        Can be passed as `this` to SlangStruct function calls, e.g. ctx.Mesh.some_fn(buf.data_at(i), ...)."""
        entry = self.entries[index]
        _, layout = get_type_and_layout(self.program, f"StructuredBuffer<ConformanceWrapper<{entry.implementation_type_name}>>")
        cursor = spy.BufferCursor(layout, self.buffer, size=self.stride, offset=index * self.stride, load_before_write=False)
        cursor.load()
        return Ref(cursor[0]["data"].read())

    def address_of(self, index: int) -> int:
        return self.buffer.device_address + index * self.stride
    
    def __getitem__(self, index: int | str) -> T:
        if isinstance(index, str):
            for entry in self.entries:
                if entry.name == index:
                    return entry
            raise KeyError(f"No entry with name '{index}' found")
        else:
            return self.entries[index]
    
    def __len__(self) -> int:
        return len(self.entries)
    
class ImageResource:
    def __init__(self, tensor: spy.Tensor):
        self.tensor = tensor
        self.height, self.width = tensor.shape[0], tensor.shape[1]

    # (h, w) of float or float3, matching the Slang Tensor<T, 2> in Texture<T>
    @staticmethod
    def from_numpy(device: spy.Device, data: np.ndarray) -> 'ImageResource':
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data = data / 255.0
        if data.ndim == 2:
            data = data.reshape(*data.shape, 1)  # not [..., np.newaxis]: that axis has stride 0
        if data.shape[2] == 4:
            data = data[:, :, :3]
        dtype = "float" if data.shape[2] == 1 else f"float{data.shape[2]}"
        tensor = create_or_resize_tensor_typed(device, None, np.ascontiguousarray(data), dtype)
        assert tensor is not None, "empty image"
        return ImageResource(tensor)
    
    @staticmethod
    def from_image(device: spy.Device, path: str, keep_alpha: bool = False) -> 'ImageResource':
        bitmap = spy.Bitmap(path)
        if not keep_alpha and bitmap.channel_count == 4:
            bitmap = bitmap.convert(spy.Bitmap.PixelFormat.rgb, spy.DataStruct.Type.float32, False)
        return ImageResource.from_numpy(device, np.array(bitmap, copy=False))
    
V = TypeVar('V')
class Texture(Generic[V]):
    def __init__(self, factor, image: ImageResource | None = None):
        self.factor = factor
        self.image = image

    def set(self, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.factor = self.factor
        cursor.texture = set_tensor(cursor.texture, self.image.tensor if self.image is not None else None)

def _null_tensor_fields(rank: int) -> dict:
    return {"_data": 0, "_shape": np.zeros(rank, dtype=np.uint32), "_strides": np.ones(rank, dtype=np.int32), "_offset": 0}

def set_tensor(cursor: spy.BufferElementCursor | spy.ShaderCursor, tensor: spy.Tensor | None):
    """Value for assigning a spy.Tensor to a cursor field of Slang type Tensor<T, D>
    or DiffTensor<T, D>. None (and missing gradients) binds null tensors — zero shape,
    null data pointer — so is_valid()/has_grad() are false on the GPU side."""
    is_diff = cursor.has_field('_primal')
    if tensor is None:
        shape_cursor = cursor["_primal"]["_shape"] if is_diff else cursor["_shape"]
        null = _null_tensor_fields(len(shape_cursor.read()))
        return {"_primal": null, "_grad_in": null, "_grad_out": null} if is_diff else null
    if not is_diff:
        return tensor
    null = _null_tensor_fields(len(tensor.shape.as_list()))
    return {
        "_primal": tensor,
        "_grad_in": tensor.grad_in if tensor.grad_in is not None else null,
        "_grad_out": tensor.grad_out if tensor.grad_out is not None else null,
    }
