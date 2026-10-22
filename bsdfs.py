from tamashii._spy import spy
import tamashii.core as core

class Bsdf(core.Interface):
    INTERFACE = "IBsdf<float>"

# https://en.wikipedia.org/wiki/Refractive_index#Typical_values
class Fresnel:
    vacuum: float = 1.0
    air: float = 1.000293
    water: float = 1.333
    ice: float = 1.31
    glass: float = 1.52
    flint_glass: float = 1.69
    sapphire: float = 1.77
    diamond: float = 2.42

class Diffuse(Bsdf):

    def __init__(self,
                 reflectance: core.Texture[spy.float3]=core.Texture(spy.float3(0.5)),
                 name: str=""):
        self.name = name
        self.reflectance = reflectance

    def upload(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        cursor.id = self.conformance_id
        self.reflectance.set(cursor.data.reflectance)

    def download(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class Plastic(Bsdf):

    def __init__(self,
                 reflectance: core.Texture[spy.float3]=core.Texture(spy.float3(0.5)),
                 roughness: core.Texture[float]=core.Texture(0.5),
                 eta_i: core.Texture[float]=core.Texture(Fresnel.air),
                 eta_t: core.Texture[float]=core.Texture(Fresnel.glass),
                 name: str=""):
        self.name = name
        self.reflectance = reflectance
        self.roughness = roughness
        self.eta_i = eta_i
        self.eta_t = eta_t

    def upload(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        cursor.id = self.conformance_id
        self.reflectance.set(cursor.data.reflectance)
        self.roughness.set(cursor.data.roughness)
        self.eta_i.set(cursor.data.eta_i)
        self.eta_t.set(cursor.data.eta_t)

    def download(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class Dielectric(Bsdf):

    def __init__(self,
                 reflectance: core.Texture[spy.float3]=core.Texture(spy.float3(0.5)),
                 roughness: core.Texture[float]=core.Texture(0.5),
                 eta_i: core.Texture[float]=core.Texture(Fresnel.air),
                 eta_t: core.Texture[float]=core.Texture(Fresnel.glass),
                 name: str=""):
        self.name = name
        self.reflectance = reflectance
        self.roughness = roughness
        self.eta_i = eta_i
        self.eta_t = eta_t

    def upload(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        cursor.id = self.conformance_id
        self.reflectance.set(cursor.data.reflectance)
        self.roughness.set(cursor.data.roughness)
        self.eta_i.set(cursor.data.eta_i)
        self.eta_t.set(cursor.data.eta_t)

    def download(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class Conductor(Bsdf):

    def __init__(self,
                 reflectance: core.Texture[spy.float3]=core.Texture(spy.float3(0.5)),
                 roughness: core.Texture[float]=core.Texture(0.5),
                 eta: core.Texture[spy.float3]=core.Texture(spy.float3(1.657460, 0.880369, 0.521229)),
                 k: core.Texture[spy.float3]=core.Texture(spy.float3(9.223869, 6.269523, 4.837001)),
                 name: str=""):
        self.name = name
        self.reflectance = reflectance
        self.roughness = roughness
        self.eta = eta
        self.k = k

    def upload(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        cursor.id = self.conformance_id
        self.reflectance.set(cursor.data.reflectance)
        self.roughness.set(cursor.data.roughness)
        self.eta.set(cursor.data.eta)
        self.k.set(cursor.data.k)

    def download(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

core.register_plugin(Diffuse, "bsdfs::Diffuse<float>")
core.register_plugin(Plastic, "bsdfs::Plastic<float>")
core.register_plugin(Dielectric, "bsdfs::Dielectric<float>")
core.register_plugin(Conductor, "bsdfs::Conductor<float>")
