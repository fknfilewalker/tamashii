from abc import abstractmethod
from tamashii._spy import spy
import tamashii.core as core
import tamashii.rfilter as rfilter
import tamashii.films as films
from enum import Enum

class Sensor(core.Interface):
    INTERFACE = "ISensor<float>"

class FovAxis(Enum):
    X = 1
    Y = 2

class Camera(Sensor):
    to_sensor_tensor: spy.Tensor | None = None
    to_sample_tensor: spy.Tensor | None = None

    def __init__(self,
                 name: str,
                 to_sensor: 'spy.float4x4param',
                 distance: float, 
                 film: films.HDRFilm):
        self.name = name
        self.film: films.HDRFilm = film

        self.to_sensor = spy.math.transpose(to_sensor)
        to_world = spy.math.inverse(self.to_sensor)
        self.eye: spy.float3 = to_world[3].xyz
        self.up: spy.float3 = to_world[1].xyz
        self.target: spy.float3 = self.eye - to_world[2].xyz * distance
        self.rfilter = rfilter.Box()

    def look_at(self, eye: 'spy.float3param', target: 'spy.float3param', up: 'spy.float3param'):
        self.eye = spy.float3(*eye)
        self.target = spy.float3(*target)
        self.up = spy.float3(*up)
        self.to_sensor = spy.math.transpose(spy.math.matrix_from_look_at(self.eye, self.target, self.up, spy.math.Handedness.right_handed))

    @abstractmethod
    def update_projection(self):
        pass

    def resize(self, width: int, height: int):
        self.film.resize(width, height)
        self.update_projection()

class Perspective(Camera):

    def __init__(self,
                 to_sensor: 'spy.float4x4param',
                 film: films.HDRFilm, 
                 fov: float=60, 
                 fov_axis: FovAxis=FovAxis.X, 
                 near: float=0.01, 
                 far: float=10000,
                 name: str="",
                 distance: float=1.0):
        super().__init__(name, to_sensor, distance, film)
        self.near, self.far, self.fov_x = near, far, fov
        if fov_axis == FovAxis.Y:
            self.fov_x = spy.math.degrees(2.0 * spy.math.atan(spy.math.tan(spy.math.radians(fov) * 0.5) * self.film.aspect()))
        self.update_projection()

    @staticmethod
    def from_look_at(eye: 'spy.float3param', 
                 target: 'spy.float3param', 
                 up: 'spy.float3param', 
                 film: films.HDRFilm, 
                 fov: float=60, 
                 fov_axis: FovAxis=FovAxis.X, 
                 near: float=0.01, 
                 far: float=10000,
                 name: str = ""):
        return Perspective(spy.math.matrix_from_look_at(spy.float3(*eye), spy.float3(*target), spy.float3(*up), spy.math.Handedness.right_handed), film, fov, fov_axis, near, far, name, spy.math.length(spy.float3(*eye) - spy.float3(*target)))

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        # cursor.film.samples = 0
        cursor.sfilm.texture = core.set_tensor(cursor.sfilm.texture, self.film.texture)
        self.to_sensor_tensor = core.upload_transform(program, cursor, "to_sensor", self.to_sensor_tensor, self.to_sensor)
        self.to_sample_tensor = core.upload_transform(program, cursor, "to_sample", self.to_sample_tensor, self.to_sample)
        cursor.near = self.near
        cursor.far = self.far
        cursor.rfilter.radius = self.rfilter.radius

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

    def update_projection(self):
        self.to_sample = spy.math.perspective(spy.math.radians(self.fov_x), self.film.aspect(), self.near, self.far)
        # flip y axis
        self.to_sample = spy.math.transpose(spy.math.mul(spy.float4x4([1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1]), self.to_sample))

core.register_plugin(Perspective, "sensors::Perspective<float>", interfaces=["sensors::ICamera<float>"])
