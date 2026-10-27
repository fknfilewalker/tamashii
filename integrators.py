from tamashii._spy import spy
import tamashii.core as core

class Integrator(core.Interface):
    INTERFACE = "IIntegrator"


class Path(Integrator):

    def __init__(self,
                 max_depth: int = 0, rr_depth: int = 3, emitter_samples: bool = False,
                 name: str=""):
        self.name = name
        self.max_depth = max_depth
        self.rr_depth = rr_depth
        self.emitter_samples = emitter_samples

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.max_depth = self.max_depth
        cursor.rr_depth = self.rr_depth
        cursor.emitter_samples = self.emitter_samples

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

# Slang-only integrators with no Python class (ids 1 and 2, after Path).

core.register_plugin(Path, "integrators::Path")
core.register_plugin(Integrator, "integrators::Ptracer")
core.register_plugin(Integrator, "integrators::AOV")
