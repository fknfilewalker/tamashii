from enum import IntEnum
from tamashii._spy import spy
import tamashii.core as core

class MIS(IntEnum):
    Equal = 0
    Balanced = 1
    Power = 2


class Integrator(core.Interface):
    INTERFACE = "IIntegrator"


class Path(Integrator):

    def __init__(self,
                 max_depth: int = 0, rr_depth: int = 3, emitter_samples: bool = True, mis: MIS = MIS.Equal,
                 name: str=""):
        self.name = name
        self.max_depth = max_depth
        self.rr_depth = rr_depth
        self.emitter_samples = emitter_samples
        self.mis = mis

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.max_depth = self.max_depth
        cursor.rr_depth = self.rr_depth
        cursor.emitter_samples = self.emitter_samples
        cursor.mis = int(self.mis)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass


class Ptracer(Integrator):

    def __init__(self, max_depth: int = 0, rr_depth: int = 3, name: str=""):
        self.name = name
        self.max_depth = max_depth
        self.rr_depth = rr_depth

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.max_depth = self.max_depth
        cursor.rr_depth = self.rr_depth

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass


class AOV(Integrator):
    class Mode(IntEnum):
        Albedo = 0
        Depth = 1
        TextureCoordinates = 2
        GeometricNormal = 3
        ShadingNormal = 4
        Emittance = 5
        Film = 6
        Primitives = 7

    def __init__(self, mode: Mode = Mode.Albedo, name: str=""):
        self.name = name
        self.mode = mode

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.mode = int(self.mode)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass


core.register_plugin(Path, "integrators::Path")
core.register_plugin(Ptracer, "integrators::Ptracer")
core.register_plugin(AOV, "integrators::AOV")
