from tamashii._spy import spy
import tamashii.core as core

class RFilter(core.Interface):
    INTERFACE = "IRFilter<float>"

class Box(RFilter):

    def __init__(self,
                 radius: float=0.5, 
                 name: str=""):
        self.name = name
        self.radius = radius

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.radius = self.radius

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        self.radius = cursor.radius

class Tent(RFilter):

    def __init__(self,
                 radius: float=1.0,
                 name: str=""):
        self.name = name
        self.radius = radius

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.radius = self.radius

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        self.radius = cursor.radius

core.register_plugin(Box, "rfilters::Box<float>")
core.register_plugin(Tent, "rfilters::Tent<float>")
