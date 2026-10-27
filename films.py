from tamashii._spy import spy
import tamashii.core as core
import numpy as np

class Film(core.Interface):
    INTERFACE = "IFilm<float, 4>"

class HDRFilm:
    def __init__(self, device: spy.Device, width: int=600, height: int=600):
        self.device = device
        self.resize(width, height)

    def aspect(self):
        return self.width / self.height

    def inv_aspect(self):
        return self.height / self.width

    @property
    def extent(self) -> spy.int2:
        return spy.int2(self.width, self.height)

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        data = np.zeros((height, width, 4), dtype=np.float32)
        # gradients are opt-in: film.texture = film.texture.with_grads(zero=True) and re-upload
        self.texture = core.create_or_resize_tensor_typed(self.device, getattr(self, 'texture', None), data, "float4", spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.shared)

    def clear(self, command_encoder=None):
        self.texture.clear(command_encoder)

    @property
    def dtexture(self) -> spy.Tensor:
        return self.texture.grad

    def dclear(self, command_encoder=None):
        self.dtexture.clear(command_encoder)

    def __deepcopy__(self, memo):
        return HDRFilm(self.device, self.width, self.height)

class SphericalHarmonicsFilm(Film):
    @property
    def implementation_type_name(self) -> str:
        return f"films::SphericalHarmonics<float, {self.order}, {self.channels}, {'true' if self.hsh else 'false'}>"

    def __init__(self, device: spy.Device, count: int, order: int = 3, channels: int = 4, hsh: bool = False):
        self.device = device
        self.order = order
        self.channels = channels
        self.hsh = hsh
        self.sh_coefs = (order + 1) ** 2
        self.resize(count)

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.sh.coefficients = core.set_tensor(cursor.sh.coefficients, self.entries)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

    def clear(self, command_encoder=None):
        if self.entries is not None:
            self.entries.clear(command_encoder)

    def resize(self, count: int):
        self.count = count
        data = np.zeros((count, self.sh_coefs, self.channels), dtype=np.float32)
        self.entries = core.create_or_resize_tensor_typed(self.device, getattr(self, 'entries', None), data, f"float{self.channels}")

core.register_plugin(Film, "films::HDRTexture2D<float, 4>")
