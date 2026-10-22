from tamashii._spy import spy
from .. import core
from abc import ABC

class Renderpass(ABC):
    def __init__(self):
        super().__init__()

class PathTracerRayQuery(Renderpass):
    def __init__(self, device: spy.Device, entry_point: str = "primal"):
        super().__init__()
        self.device = device
        module = device.compose_modules(
            name="pt_rq_with_tamashii",
            modules=[device.load_module("pt_rq.slang"), core.get_composed_module(device)],
        )
        self.entry_point_name = entry_point
        ep = module.entry_point(entry_point)
        self.program = device.link_program(modules=[module], entry_points=[ep])
        self.pipeline = device.create_compute_pipeline(self.program)

        self.settings = {
            "seed" : 0,
            "spp" : 1,
            "mode" : 0,
            "aov_mode" : 0,
            "max_depth" : 0,
            "rr_depth" : 3,
            "nee" : True,
            "mis": 0,
        }

    def render(self, compute_pass: spy.ComputePassEncoder, scene, width, height):
        shader_object = compute_pass.bind_pipeline(self.pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor["_scene"] = scene
        cursor["_settings"] = self.settings
        compute_pass.dispatch(thread_count=[width, height, 1])

    def __getitem__(self, key: str):
        return self.settings[key]

    def __setitem__(self, key: str, value):
        if key in self.settings:
            self.settings[key] = value
