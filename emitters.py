from tamashii._spy import spy
import tamashii.core as core
import tamashii.hosek_wilkie as hosek_wilkie
import numpy as np

class Emitter(core.Interface):
    INTERFACE = "IEmitter<float>"
    to_world: 'spy.float4x4'
    to_world_tensor: spy.Tensor | None = None

class Point(Emitter):

    @classmethod
    def from_ies(cls, device: spy.Device, filepath: str, color: 'spy.float3param'=spy.float3(1.0), intensity: float=1.0, to_world: 'spy.float4x4param'=spy.float4x4().identity(), name: str="", width: int=512, height: int=256):
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        tilt_line_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('TILT='):
                tilt_line_idx = i
                break
                
        if tilt_line_idx == -1:
            raise ValueError("Invalid IES file: No TILT= line found")
            
        tilt_value = lines[tilt_line_idx].split('=')[1].strip()
        
        data_str = " ".join(lines[tilt_line_idx+1:])
        tokens = data_str.split()
        
        if tilt_value != 'NONE':
            num_tilt_angles = int(tokens[1])
            tokens = tokens[2 + 2 * num_tilt_angles:]
            
        num_lamps = int(tokens[0])
        lumens_per_lamp = float(tokens[1])
        multiplier = float(tokens[2])
        num_vert_angles = int(tokens[3])
        num_horiz_angles = int(tokens[4])
        photometric_type = int(tokens[5])
        
        idx = 13
        vert_angles = np.array([float(tokens[i]) for i in range(idx, idx + num_vert_angles)])
        idx += num_vert_angles
        
        horiz_angles = np.array([float(tokens[i]) for i in range(idx, idx + num_horiz_angles)])
        idx += num_horiz_angles
        
        candela_values = np.zeros((num_horiz_angles, num_vert_angles), dtype=np.float32)
        for i in range(num_horiz_angles):
            for j in range(num_vert_angles):
                candela_values[i, j] = float(tokens[idx])
                idx += 1
                
        ballast_factor = float(tokens[10])
        ballast_lamp_factor = float(tokens[11])
        candela_values *= multiplier * ballast_factor * ballast_lamp_factor
        
        if vert_angles[-1] < 180.0:
            last_val = vert_angles[-1]
            vert_angles = np.append(vert_angles, [last_val + 1e-3, 180.0])
            zeros = np.zeros((candela_values.shape[0], 2), dtype=np.float32)
            candela_values = np.hstack((candela_values, zeros))
        
        if horiz_angles[-1] == 0:
            horiz_angles = np.array([0.0, 360.0])
            candela_values = np.vstack((candela_values[0], candela_values[0]))
        elif horiz_angles[-1] == 90:
            h_90_to_180 = 180.0 - horiz_angles[-2::-1]
            c_90_to_180 = candela_values[-2::-1, :]
            horiz_angles = np.concatenate((horiz_angles, h_90_to_180))
            candela_values = np.vstack((candela_values, c_90_to_180))
            h_180_to_360 = 360.0 - horiz_angles[-2::-1]
            c_180_to_360 = candela_values[-2::-1, :]
            horiz_angles = np.concatenate((horiz_angles, h_180_to_360))
            candela_values = np.vstack((candela_values, c_180_to_360))
        elif horiz_angles[-1] == 180:
            h_180_to_360 = 360.0 - horiz_angles[-2::-1]
            c_180_to_360 = candela_values[-2::-1, :]
            horiz_angles = np.concatenate((horiz_angles, h_180_to_360))
            candela_values = np.vstack((candela_values, c_180_to_360))

        u = np.linspace(0, 1, width, endpoint=False)
        phi = (u - 0.5) * 360.0
        phi = np.where(phi < 0, phi + 360.0, phi)
        
        v = np.linspace(0, 1, height)
        query_vert = (1.0 - v) * 180.0
        
        h_idx = np.searchsorted(horiz_angles, phi) - 1
        h_idx = np.clip(h_idx, 0, len(horiz_angles)-2)
        h_t = (phi - horiz_angles[h_idx]) / (horiz_angles[h_idx+1] - horiz_angles[h_idx] + 1e-8)
        
        v_idx = np.searchsorted(vert_angles, query_vert) - 1
        v_idx = np.clip(v_idx, 0, len(vert_angles)-2)
        v_t = (query_vert - vert_angles[v_idx]) / (vert_angles[v_idx+1] - vert_angles[v_idx] + 1e-8)
        
        H_idx, V_idx = np.meshgrid(h_idx, v_idx)
        H_t, V_t = np.meshgrid(h_t, v_t)
        
        c00 = candela_values[H_idx, V_idx]
        c10 = candela_values[H_idx+1, V_idx]
        c01 = candela_values[H_idx, V_idx+1]
        c11 = candela_values[H_idx+1, V_idx+1]
        
        c0 = c00 * (1 - H_t) + c10 * H_t
        c1 = c01 * (1 - H_t) + c11 * H_t
        img = c0 * (1 - V_t) + c1 * V_t
        
        max_candela = np.max(img)
        if max_candela > 0:
            img /= max_candela
            
        profile_tex = core.Texture[float](max_candela * intensity, core.ImageResource.from_numpy(device, img))
        return cls(color=color, to_world=to_world, profile=profile_tex, name=name)

    def __init__(self,
                 color: 'spy.float3param'=spy.float3(1.0),
                 profile: 'core.Texture[float]'=core.Texture(1.0),
                 to_world: 'spy.float4x4param'=spy.float4x4().identity(), 
                 name: str=""):
        self.name = name
        self.to_world = spy.math.transpose(to_world)
        self.color = color
        self.profile = profile

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.color = self.color
        self.profile.set(cursor.profile)
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        self.color = cursor.color
        self.to_world = core.download_transform(self.to_world_tensor)

class Area(Emitter):

    def __init__(self,
                 shape: int,
                 to_world: 'spy.float4x4param'=spy.float4x4().identity(),
                 name: str=""):
        self.name = name
        self.to_world = spy.math.transpose(to_world)
        self.shape = shape

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)
        cursor.shape = self.shape

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass
    
class Constant(Emitter):

    def __init__(self,
                 radiance: core.Texture[spy.float3]=core.Texture(spy.float3(1.0)),
                 name: str=""):
        self.name = name
        self.to_world = spy.math.transpose(spy.float4x4().identity())
        self.radiance = radiance

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        self.radiance.set(cursor.radiance)
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class HosekWilkie(Emitter):
    """Analytic sky from the Hosek-Wilkie model; the fit ships in the tables module.

    Zero below the horizon, no atmospheric transmittance. Once the sun sets the sky
    is held at sunset and the disc switched off. Radiance is in the model's units
    (spectral, reduced to linear sRGB); `scale` converts to the scene's.
    """

    def __init__(self,
                 turbidity: float = 3.0,
                 elevation: float = 20.0,
                 azimuth: float = 0.0,
                 albedo: float = 0.3,
                 scale: float = 1.0,
                 sun_scale: float = 1.0,
                 sun_select_pdf: float = 0.5,
                 to_world: 'spy.float4x4param' = spy.float4x4().identity(),
                 name: str = ""):
        self.name = name
        self.to_world = spy.math.transpose(to_world)
        self.scale = scale
        self.sun_scale = sun_scale
        # one would leave the sky unsampled and zero the density away from the disc
        self.sun_select_pdf = min(max(sun_select_pdf, 0.0), 0.95)
        self.data_tensor: spy.Tensor | None = None
        # remembered so `configure_from_time` takes partial updates
        self.latitude, self.longitude, self.timezone = 0.0, 0.0, 0.0
        self.year, self.month, self.day = 2025, 6, 21
        self.hour, self.minute, self.second = 12.0, 0.0, 0.0
        # set by configure()
        self.turbidity: float
        self.elevation: float
        self.azimuth: float
        self.albedo: float
        self.configure(turbidity, elevation, azimuth, albedo)

    def configure(self,
                  turbidity: float | None = None,
                  elevation: float | None = None,
                  azimuth: float | None = None,
                  albedo: float | None = None):
        """Set the sky; angles in degrees, turbidity in [1, 10].

        Only the given values change. Re-upload the emitter afterwards.
        """
        for key, value in (("turbidity", turbidity), ("elevation", elevation),
                           ("azimuth", azimuth), ("albedo", albedo)):
            if value is not None:
                setattr(self, key, value)

        elevation_rad, azimuth_rad = np.radians(self.elevation), np.radians(self.azimuth)
        self.packed, self.layout = hosek_wilkie.pack(
            self.turbidity, self.albedo, elevation_rad, azimuth_rad)

        # model is Z-up, tamashii Y-up
        self.sun_direction = spy.float3(
            float(np.cos(azimuth_rad) * np.cos(elevation_rad)),
            float(np.sin(elevation_rad)),
            float(-np.sin(azimuth_rad) * np.cos(elevation_rad)))
        self.sun_cos_radius = float(np.cos(hosek_wilkie.SOLAR_RADIUS))

    def configure_from_time(self,
                            latitude: float | None = None, longitude: float | None = None,
                            timezone: float | None = None,
                            year: int | None = None, month: int | None = None, day: int | None = None,
                            hour: float | None = None, minute: float | None = None,
                            second: float | None = None):
        """Place the sun from a location and a moment instead of by hand.

        Degrees north/east, `timezone` offset from UTC, `hour` local clock time.
        Only the given values change. Night is not modelled. Last call wins,
        between this and `configure`.
        """
        for key, value in (("latitude", latitude), ("longitude", longitude), ("timezone", timezone),
                           ("year", year), ("month", month), ("day", day),
                           ("hour", hour), ("minute", minute), ("second", second)):
            if value is not None:
                setattr(self, key, value)

        elevation, azimuth = hosek_wilkie.sun_position(
            self.latitude, self.longitude, self.timezone,
            self.year, self.month, self.day, self.hour, self.minute, self.second)
        self.configure(elevation=float(np.degrees(elevation)), azimuth=float(np.degrees(azimuth)))
        return self

    def sky_rgb(self, direction: 'spy.float3param') -> np.ndarray:
        """CPU evaluation of sky radiance (no sun) along a world-space direction."""
        d = np.array([direction[0], direction[1], direction[2]], dtype=np.float64)
        model = np.array([d[0], -d[2], d[1]])  # Y-up world -> Z-up model
        model /= np.linalg.norm(model)
        elevation_rad, azimuth_rad = np.radians(self.elevation), np.radians(self.azimuth)
        sun = np.array([np.cos(azimuth_rad) * np.cos(elevation_rad),
                        np.sin(azimuth_rad) * np.cos(elevation_rad),
                        np.sin(elevation_rad)])
        params, radiance = hosek_wilkie.cook(self.turbidity, self.albedo, elevation_rad)
        spectrum = hosek_wilkie.sky_radiance(
            params, radiance,
            float(np.arccos(np.clip(model[2], -1, 1))),
            float(np.arccos(np.clip(np.dot(model, sun), -1, 1))))
        return self.scale * (spectrum @ hosek_wilkie.rgb_weights())

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)

        self.data_tensor = core.create_or_resize_tensor_typed(
            program.device, self.data_tensor, self.packed, "float", element_ndim=0)
        dataset = cursor.dataset
        dataset.channels = self.layout["channels"]
        dataset.data = core.set_tensor(dataset.data, self.data_tensor)

        cursor.sun_direction = self.sun_direction
        cursor.sun_cos_radius = self.sun_cos_radius
        # the fit is held at sunset, but the disc has genuinely set
        up = self.elevation > 0.0
        cursor.sun_scale = self.sun_scale if up else 0.0
        cursor.sun_select_pdf = self.sun_select_pdf if up and self.sun_scale > 0.0 else 0.0
        cursor.scale = self.scale

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        self.scale = cursor.scale

core.register_plugin(Point, "emitters::Point<float>")
core.register_plugin(Area, "emitters::Area<float>")
core.register_plugin(Constant, "emitters::Constant<float>")
core.register_plugin(HosekWilkie, "emitters::HosekWilkie<float>")
