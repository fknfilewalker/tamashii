from tamashii._spy import spy
from .. import core, integrators
from abc import ABC

class Renderpass(ABC):
    def __init__(self):
        super().__init__()

class PathTracerRayQuery(Renderpass):
    def __init__(self, device: spy.Device, entry_point: str = "primal", integrator: integrators.Integrator | None = None):
        super().__init__()
        self.device = device
        self.module = device.compose_modules(
            name="pt_rq_with_tamashii",
            modules=[device.load_module("pt_rq.slang"), core.get_composed_module(device)],
        )
        self.entry_point_name = entry_point
        self.programs = {}  # list of integrator types -> (program, pipeline)

        self.settings = {
            "seed" : 0,
            "spp" : 1,
        }
        self.integrator = integrator if integrator is not None else integrators.Path()

    def _specialized(self) -> tuple[spy.ShaderProgram, spy.ComputePipeline]:
        name = self.integrator.implementation_type_name
        if name not in self.programs:
            ep = self.module.entry_point(self.entry_point_name).specialize(
                [spy.SpecializationArg(spy.SpecializationArgKind.type, name)])
            program = self.device.link_program(modules=[self.module], entry_points=[ep])
            self.programs[name] = (program, self.device.create_compute_pipeline(program))
        return self.programs[name]

    @property
    def program(self) -> spy.ShaderProgram:
        return self._specialized()[0]

    def render(self, compute_pass: spy.ComputePassEncoder, scene, width, height):
        program, pipeline = self._specialized()
        entry_point = spy.ShaderCursor(compute_pass.bind_pipeline(pipeline)).find_entry_point(0)
        entry_point["scene"] = scene
        entry_point["settings"] = self.settings
        self.integrator.upload(program, entry_point["settings"]["integrator"])
        compute_pass.dispatch(thread_count=[width, height, 1])

    def __getitem__(self, key: str):
        return self.settings[key]

    def __setitem__(self, key: str, value):
        if key in self.settings:
            self.settings[key] = value
