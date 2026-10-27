import math
import pathlib
import copy
from tamashii._spy import spy
import numpy as np
import tamashii.shapes as shapes
import tamashii.bsdfs as bsdfs
import tamashii.emitters as emitters
import tamashii.core as core
import tamashii.sensors as sensors
import tamashii.films as films
import tamashii.integrators as integrators

class Utils:
    @staticmethod
    def trimesh_transforms_to_slang_matrix(transform):
        return spy.float4x4(transform.astype(np.float32))

    @staticmethod
    def radiance_to_color_intensity(radiance: spy.float3):
        intensity = np.max([radiance.x, radiance.y, radiance.z])
        return radiance / intensity, intensity

class Scene():
    def __init__(self, program: spy.ShaderProgram):
        self.program = program
        self.sensors: core.ResourceBuffer[sensors.Sensor]
        self.bsdfs: core.ResourceBuffer[bsdfs.Bsdf]
        self.shapes: core.ResourceBuffer[shapes.Shape]
        self.emitters: core.ResourceBuffer[emitters.Emitter]
        self.acceleration_structure: shapes.AccelerationStructure

    def update_shapes(self):
        self.shapes.upload()
        self.acceleration_structure.update_blas(self.shapes.entries)
        self.acceleration_structure.update_tlas(self.shapes.entries)

    def update_as(self):
        self.acceleration_structure.update_blas(self.shapes.entries)
        self.acceleration_structure.update_tlas(self.shapes.entries)

    def get(self):
        return {
            "sensors": self.sensors.address_of(0),
            "as": self.acceleration_structure.tlas,
            "shapes": self.shapes.address_of(0),
            "emitters": self.emitters.address_of(0),
            "environment_emitter": next((self.emitters.address_of(i) for i, e in enumerate(self.emitters.entries) if isinstance(e, (emitters.Constant, emitters.HosekWilkie))), 0),
            "sensor_count": len(self.sensors),
            "shape_count": len(self.shapes),
            "emitter_count": len(self.emitters),
        }

    @staticmethod
    def from_file(program: spy.ShaderProgram, file: str | pathlib.Path) -> "Scene":
        file = pathlib.Path(file)
        if file.suffix == ".xml":
            import xml.etree.ElementTree as ET
            directory = file.parent.absolute()
            root = ET.parse(file).getroot()
    
            def parse_transform(element):
                # Mitsuba applies transforms in XML listing order (first listed = innermost, applied first to the object).
                # To build the matrix, each new transform pre-multiplies (left-multiplies) the accumulator.
                # E.g. <scale/><rotate/> → Rotate * Scale (scale object first, then rotate).
                transform = core.Transform()
                if (e := element.find('transform[@name="to_world"]')) is not None:
                    for child in reversed(list(e)):
                        match child.tag:
                            case 'matrix':
                                transform = transform.multiply(core.Transform(spy.math.float4x4(np.fromstring(child.get('value').replace(',', ' '), sep=' ', dtype=np.float32))))
                            case 'translate':
                                if (v := child.get('value')) is not None:
                                    vals = [float(x) for x in v.replace(',', ' ').split()]
                                    transform = transform.translate(spy.float3(*vals))
                                else:
                                    transform = transform.translate(spy.float3(float(child.get('x', '0')), float(child.get('y', '0')), float(child.get('z', '0'))))
                            case 'scale':
                                if (v := child.get('value')) is not None:
                                    vals = [float(x) for x in v.replace(',', ' ').split()]
                                    s = spy.float3(*vals) if len(vals) == 3 else spy.float3(vals[0])
                                    transform = transform.scale(s)
                                else:
                                    transform = transform.scale(spy.float3(float(child.get('x', '1')), float(child.get('y', '1')), float(child.get('z', '1'))))
                            case 'rotate':
                                transform = transform.rotate(spy.math.quat_from_angle_axis(spy.math.radians(float(child.get('angle', '0'))), spy.float3(float(child.get('x', '0')), float(child.get('y', '0')), float(child.get('z', '0')))))
                            case 'lookat':
                                target = spy.float3(*[float(x) for x in child.get('target').replace(',', ' ').split()])
                                origin = spy.float3(*[float(x) for x in child.get('origin').replace(',', ' ').split()])
                                up = spy.float3(*[float(x) for x in child.get('up').replace(',', ' ').split()])
                                transform = transform.multiply(core.Transform(spy.math.inverse(spy.math.matrix_from_look_at(origin, target, up, spy.math.Handedness.right_handed))))
                    return transform
                if (e := element.find("*[@name='position']")) is not None:
                    transform = transform.translate(spy.float3(float(e.get('x', '0')), float(e.get('y', '0')), float(e.get('z', '0'))))
                return transform
            
            def parse_texture(element):
                match element.tag:
                    case 'rgb': 
                        val = element.get('value', '0.5 0.5 0.5')
                        return core.Texture(spy.float3(*[float(x) for x in val.replace(',', ' ').split()]))
                    case 'texture':
                        filename = element.find(".//string[@name='filename']").get('value')
                        raw = element.find(".//boolean[@name='raw']")
                        raw = raw is not None and raw.get('value', 'false').lower() == 'true'
                        return core.Texture(spy.float3(1), core.ImageResource.from_image(program.device, spy.Bitmap(str(directory / filename)), raw=raw))
                return core.Texture(spy.float3(0.5))

            def parse_bsdf(element):
                if element.get('type') == 'bumpmap':
                    element = element.find("bsdf")
                id = element.get('id', '')
                if element.get('type') == 'twosided':
                    element = element.find("bsdf")
                match element.get('type'):
                    case 'diffuse':
                        e = element.find("*[@name='reflectance']")
                        return bsdfs.Diffuse(name=id, reflectance=parse_texture(e))
                    case 'roughplastic':
                        e = element.find("*[@name='diffuse_reflectance']")
                        return bsdfs.Diffuse(name=id, reflectance=parse_texture(e))
                    case 'roughconductor':
                        return bsdfs.Diffuse(name=id, reflectance=core.Texture(spy.float3(0.5)))
                    case 'dielectric':
                        return bsdfs.Diffuse(name=id, reflectance=core.Texture(spy.float3(0.5)))
                    case 'conductor':
                        return bsdfs.Diffuse(name=id, reflectance=core.Texture(spy.float3(0.5)))
                return None
            
            def parse_color(element):
                match element.tag:
                    case 'rgb': 
                        val = element.get('value', '0.5 0.5 0.5')
                        return spy.float3(*[float(x) for x in val.replace(',', ' ').split()])
                
                return spy.float3(0.5)  

            bsdf_list = []
            shape_list = []
            emitter_list = []
            sensor_list = []

            for child in root:
                if child.tag == 'bsdf':
                    bsdf = parse_bsdf(child)
                    bsdf_list.append(bsdf)
                    
                if child.tag == 'shape':
                    stype = child.get('type')
                    id = child.get('id', '')
                    bsdf_idx = len(bsdf_list)
                    if (ref := child.find('ref')) is not None:
                        bsdf_id = ref.get('id')
                        bsdf_idx = next((i for i, b in enumerate(bsdf_list) if b.name == bsdf_id), None)
                    else:
                        bsdf = parse_bsdf(child.find('bsdf'))
                        bsdf_list.append(bsdf)

                    emission = spy.float3(0)
                    if (emitter := child.find('emitter')) is not None:
                        val = emitter.find(".//rgb[@name='radiance']").get('value')
                        emission = spy.float3(*[float(x) for x in val.replace(',', ' ').split()])
                    
                    if stype in ['obj', 'ply']:
                        filename = child.find(".//string[@name='filename']").get('value')
                        if id == '':
                            id = filename
                        mesh = shapes.Mesh.from_file(name=id, file=directory / filename, bsdf=bsdf_idx, to_world=parse_transform(child))
                        mesh.emission = emission
                        if len(mesh.texcoords) > 0:
                            mesh.texcoords[1::2] = 1.0 - mesh.texcoords[1::2]
                        shape_list.append(mesh)
                    if stype == 'rectangle':
                        # Mitsuba rectangle is in XY plane ([-1,1]x[-1,1]), renderer rectangle is in XZ plane ([-0.5,0.5]x[-0.5,0.5])
                        # Rotate 90° around X to convert XY→XZ, scale by 2 to match Mitsuba's [-1,1] range
                        shape_list.append(shapes.Rectangle(name=id, emission=emission, scale=spy.float2(2), bsdf=bsdf_idx, to_world=core.Transform().multiply(parse_transform(child)).rotate(spy.math.quat_from_angle_axis(spy.math.radians(90), spy.float3(1,0,0)))))
                    if spy.math.any(spy.math.gt(emission, spy.float3(0))):
                        emitter_list.append(emitters.Area(name=id, shape=len(shape_list)-1))

                if child.tag == 'emitter':
                    etype = child.get('type')
                    id = child.get('id', '')
                    if etype == 'point':
                        color, intensity = Utils.radiance_to_color_intensity(parse_color(child.find("*[@name='intensity']")))
                        emitter_list.append(emitters.Point(name=id, color=color, profile=core.Texture[float](intensity), to_world=parse_transform(child)))
                
                if child.tag == 'sensor':
                    stype = child.get('type')
                    id = child.get('id', '')
                    if stype == 'perspective':
                        film = films.HDRFilm(program.device)
                        fov = 45.0
                        fov_element = child.find(".//float[@name='fov']")
                        if fov_element is not None:
                            fov = float(fov_element.get('value', '45.0'))
                        if (to_world_element := child.find("*[@name='to_world']")):
                            if (e := to_world_element.find('lookat')) is not None:
                                target = spy.float3(*[float(x) for x in e.get('target').replace(',', ' ').split()])
                                eye = spy.float3(*[float(x) for x in e.get('origin').replace(',', ' ').split()])
                                up = spy.float3(*[float(x) for x in e.get('up').replace(',', ' ').split()])
                                sensor_list.append(sensors.Perspective.from_look_at(name=id, eye=eye, target=target, up=up, film=film, fov=fov))
                            else:
                                # Mitsuba camera looks along +Z, renderer along -Z; flip X and Z (180° around Y)
                                mitsuba_flip = spy.float4x4([-1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1])
                                to_world = spy.math.mul(parse_transform(child), mitsuba_flip)
                                sensor_list.append(sensors.Perspective(name=id, to_sensor=spy.math.inverse(to_world), film=film, fov=fov))
                        
            _bsdfs = core.ResourceBuffer[bsdfs.Bsdf](program, bsdf_list)
            _bsdfs.upload()

            # Update shape bsdf addresses
            for shape in shape_list:
                shape.bsdf = _bsdfs.address_of(shape.bsdf)
            _shapes = core.ResourceBuffer[shapes.Shape](program, shape_list)
            _shapes.upload()

            _emitters = core.ResourceBuffer[emitters.Emitter](program, emitter_list)
            for emitter in emitter_list:
                if isinstance(emitter, emitters.Area):
                    emitter.shape = _shapes.address_of(emitter.shape)
            _emitters.upload()

            _sensors = core.ResourceBuffer[sensors.Sensor](program, sensor_list)
            _sensors.upload()

            _acceleration_structure = shapes.AccelerationStructure(program.device)
            _acceleration_structure.update_blas(_shapes.entries)
            _acceleration_structure.update_tlas(_shapes.entries)

            _scene = Scene(program)
            _scene.sensors = _sensors
            _scene.bsdfs = _bsdfs
            _scene.shapes = _shapes
            _scene.emitters = _emitters
            _scene.acceleration_structure = _acceleration_structure
            return _scene
        else:
            import pygltflib
            scene = pygltflib.GLTF2.load(file)
            if scene is None:
                raise Exception(f"Failed to load scene from {file}")

            image_list: list[core.ImageResource] = []
            for image in scene.images:
                if image.uri is not None:
                    image_list.append(core.ImageResource.from_image(program.device, spy.Bitmap(str(file.parent / image.uri))))
                elif image.bufferView is not None:
                    bufferView = scene.bufferViews[image.bufferView]
                    buffer = scene.buffers[bufferView.buffer]
                    data = scene.get_data_from_buffer_uri(buffer.uri)[bufferView.byteOffset:bufferView.byteOffset+bufferView.byteLength]
                    bitmap = spy.Bitmap.load_from_numpy(np.array(np.frombuffer(data, np.uint8)))
                    image_list.append(core.ImageResource.from_image(program.device, bitmap))

            texture_list = []
            for texture in scene.textures:
                texture_list.append(image_list[texture.source])
        
            bsdf_list = [] # aka materials
            for material in scene.materials:
                diffuse_color = material.pbrMetallicRoughness.baseColorFactor[0:3]
                diffuse_texture = None
                if material.pbrMetallicRoughness.baseColorTexture is not None:
                    diffuse_texture = texture_list[material.pbrMetallicRoughness.baseColorTexture.index]
                bsdf_list.append(bsdfs.Diffuse(name=material.name, reflectance=core.Texture(diffuse_color, diffuse_texture)))
            _bsdfs = core.ResourceBuffer(program, bsdf_list)
            _bsdfs.upload()

            def load_accessor(index):
                accessor = scene.accessors[index]
                bufferView = scene.bufferViews[accessor.bufferView]
                buffer = scene.buffers[bufferView.buffer]
                
                dtype = np.dtype(np.float32)
                match accessor.componentType:
                    case pygltflib.BYTE: 
                        dtype = np.dtype(np.int8)
                    case pygltflib.UNSIGNED_BYTE: 
                        dtype = np.dtype(np.uint8)
                    case pygltflib.SHORT: 
                        dtype = np.dtype(np.int16)
                    case pygltflib.UNSIGNED_SHORT: 
                        dtype = np.dtype(np.uint16)
                    case pygltflib.UNSIGNED_INT: 
                        dtype = np.dtype(np.uint32)

                element_count = 1
                match accessor.type:
                    case pygltflib.VEC2: 
                        element_count = 2
                    case pygltflib.VEC3: 
                        element_count = 3
                    case pygltflib.VEC4: 
                        element_count = 4
                    case pygltflib.MAT2: 
                        element_count = 4
                    case pygltflib.MAT3: 
                        element_count = 9
                    case pygltflib.MAT4: 
                        element_count = 16

                stride = bufferView.byteStride if (bufferView.byteStride and bufferView.byteStride > 0) else (dtype.itemsize * element_count)
                return np.ndarray(
                    shape=(accessor.count, element_count),
                    dtype=dtype,
                    buffer=scene.get_data_from_buffer_uri(buffer.uri),
                    offset=(bufferView.byteOffset or 0) + (accessor.byteOffset or 0),
                    strides=(stride, dtype.itemsize)
                )

            meshes = []
            for mesh in scene.meshes:
                meshes.append([])
                for i, primitive in enumerate(mesh.primitives):
                    assert primitive.mode == pygltflib.TRIANGLES, "Only triangle primitives are supported"
                    indices = load_accessor(primitive.indices).flatten().astype(np.uint32)
                    vertices = load_accessor(primitive.attributes.POSITION).flatten()
                    texcoords = load_accessor(primitive.attributes.TEXCOORD_0).flatten() if primitive.attributes.TEXCOORD_0 is not None else {}
                    meshes[-1].append(shapes.Mesh(name=f"{mesh.name}_{i}", bsdf=primitive.material, vertices=vertices, indices=indices, texcoords=texcoords))

            lights = []
            if scene.extensions:
                if 'KHR_lights_punctual' in scene.extensions:
                    for light in scene.extensions['KHR_lights_punctual']['lights']:
                        match light['type']:
                            case 'point':
                                intensity = light['intensity']
                                if True: # as_candela
                                    # candela (lm/sr) to watt
                                    candela = intensity
                                    lumen = candela * math.pi * 4.0
                                    watt = lumen / 683.0
                                    intensity = watt
                                color = spy.float3(light['color'][0], light['color'][1], light['color'][2])
                                lights.append(emitters.Point(name=light['name'], color=color, profile=core.Texture[float](intensity)))
                if 'EXT_lights_ies' in scene.extensions:
                    for light in scene.extensions['EXT_lights_ies']['lights']:
                        #TODO
                        if 'uri' in light:
                            pass
                        if 'bufferView' in light:
                            pass

            cameras = []
            for camera in scene.cameras:
                if camera.type == 'perspective':
                    cameras.append(sensors.Perspective(name=camera.name, to_sensor=core.Transform(), film=films.HDRFilm(program.device), fov=spy.math.degrees(camera.perspective.yfov), fov_axis=sensors.FovAxis.Y))

            shape_list, emitter_list, sensor_list = [], [], []
            nodes = scene.nodes

            def traverse_graph(node: pygltflib.Node, transform: core.Transform):
                if node.translation is not None:
                    transform = transform.translate(node.translation)
                if node.rotation is not None:
                    transform = transform.rotate(node.rotation)
                if node.scale is not None:
                    transform = transform.scale(node.scale)
                if node.matrix is not None:
                    transform = transform.multiply(core.Transform(node.matrix))
                
                if node.mesh is not None:
                    for mesh in meshes[node.mesh]:
                        mesh.to_world = spy.math.transpose(transform)
                        shape_list.append(copy.deepcopy(mesh))
                if node.camera is not None:
                    sensor = cameras[node.camera]
                    sensor.to_sensor = spy.math.transpose(spy.math.inverse(transform))
                    to_world = spy.math.transpose(transform)
                    sensor.eye = to_world[3].xyz
                    sensor.up = to_world[1].xyz
                    sensor.target = sensor.eye - to_world[2].xyz
                    sensor_list.append(copy.deepcopy(sensor))
                if node.extensions:
                    light = None
                    if 'KHR_lights_punctual' in node.extensions:
                        light = lights[node.extensions['KHR_lights_punctual']['light']]
                    if 'EXT_lights_ies' in node.extensions:
                        light = lights[node.extensions['EXT_lights_ies']['light']]
                    if light is not None:
                        light.to_world = spy.math.transpose(transform)
                        emitter_list.append(copy.deepcopy(light))
                for child in node.children:
                    traverse_graph(nodes[child], core.Transform(transform))
            
            for node in scene.scenes[0].nodes:
                traverse_graph(nodes[node], core.Transform())
                
            if len(sensor_list) == 0:
                sensor_list.append(sensors.Perspective.from_look_at(name="default_camera", eye=[0, 1.0, 1.0], target=[0, 0, 0], up=[0, 1, 0], film=films.HDRFilm(program.device), fov=45))

            # Update shape bsdf addresses
            for shape in shape_list:
                shape.bsdf = _bsdfs.address_of(shape.bsdf)
            _shapes = core.ResourceBuffer[shapes.Shape](program, shape_list)
            _shapes.upload()

            _emitters = core.ResourceBuffer[emitters.Emitter](program, emitter_list)
            _emitters.upload()

            _sensors = core.ResourceBuffer[sensors.Sensor](program, sensor_list)
            _sensors.upload()

            _acceleration_structure = shapes.AccelerationStructure(program.device)
            _acceleration_structure.update_blas(_shapes.entries)
            _acceleration_structure.update_tlas(_shapes.entries)

            _scene = Scene(program)
            _scene.sensors = _sensors
            _scene.bsdfs = _bsdfs
            _scene.shapes = _shapes
            _scene.emitters = _emitters
            _scene.acceleration_structure = _acceleration_structure
            return _scene
    
    @staticmethod
    def test_scene(program: spy.ShaderProgram):
        _device = program.device
        _sensors = core.ResourceBuffer[sensors.Sensor](program, [
            sensors.Perspective.from_look_at(name="cam", eye=[0, 0.0, -4], target=[0, 0, 0], up=[0, 1, 0], film=films.HDRFilm(_device), fov=45)
        ])
        _sensors.upload()

        _bsdfs = core.ResourceBuffer[bsdfs.Bsdf](program, [
            bsdfs.Diffuse(name="", reflectance=core.Texture(spy.float3(0.8, 0.2, 0.2))),
            bsdfs.Diffuse(name="", reflectance=core.Texture(spy.float3(0.2, 0.8, 0.2))),
        ])
        _bsdfs.upload()

        _shapes = core.ResourceBuffer[shapes.Shape](program, [
            shapes.Sphere(bsdf=_bsdfs.address_of(0), to_world=core.Transform().translate([1.0, 1.0, 1.0]), radius=0.5),
            shapes.Sphere(bsdf=_bsdfs.address_of(1), to_world=core.Transform().translate([-1.0, -1.0, 1.0]), radius=0.5),
            shapes.Mesh(bsdf=_bsdfs.address_of(0), vertices=np.array([0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0], dtype=np.float32), indices=np.array([0, 1, 2], dtype=np.uint32)),
            shapes.Mesh.from_file(bsdf=_bsdfs.address_of(1), file="_cache/mitsuba_scenes/scenes/meshes/bunny.ply", to_world=core.Transform().translate([0.0, -0.3, -2.0])),
            shapes.Mesh.from_file(bsdf=_bsdfs.address_of(0), file="_cache/mitsuba_scenes/scenes/meshes/sphere.obj", to_world=core.Transform().scale([0.1, 0.1, 0.1])),
            shapes.Sphere(bsdf=_bsdfs.address_of(0), radius=0.2, emission=spy.float3(40)),
            shapes.Rectangle(name="rect", bsdf=_bsdfs.address_of(0), to_world=core.Transform().translate([0.0, -1.0, 0.0])),
        ])
        _shapes.upload()

        _emitters = core.ResourceBuffer[emitters.Emitter](program, [
            # emitters.Point(name="", color=spy.float3(1.0, 1.0, 1.0), profile=core.Texture[float](6.0), to_world=core.Transform().translate([0.0, 0.0, -4.0])),
            # emitters.Constant(name="", radiance==core.Texture(spy.float3(0.5))),
            emitters.Area(name="", shape=_shapes.address_of(5)),
        ])
        _emitters.upload()

        _acceleration_structure = shapes.AccelerationStructure(_device)
        _acceleration_structure.update_blas(_shapes.entries)
        _acceleration_structure.update_tlas(_shapes.entries)
        
        _scene = Scene(program)
        _scene.sensors = _sensors
        _scene.bsdfs = _bsdfs
        _scene.shapes = _shapes
        _scene.emitters = _emitters
        _scene.acceleration_structure = _acceleration_structure
        return _scene

    @staticmethod
    def simple_shadow(program: spy.ShaderProgram):
        _device = program.device
        _sensors = core.ResourceBuffer[sensors.Sensor](program, [
            sensors.Perspective.from_look_at(name="cam", eye=[0, 1.1, -4], target=[0, 0, 0], up=[0, 1, 0], film=films.HDRFilm(_device), fov=60)
        ])
        _sensors.upload()

        _bsdfs = core.ResourceBuffer[bsdfs.Bsdf](program, [
            bsdfs.Diffuse(name="", reflectance=core.Texture(spy.float3(1.0, 1.0, 1.0))),
        ])
        _bsdfs.upload()

        _shapes = core.ResourceBuffer[shapes.Shape](program, [
            shapes.Mesh(name="floor", bsdf=_bsdfs.address_of(0), vertices=np.array([10.0, 0.0, -10.0, -10.0, 0.0, -10.0, -10.0, 0.0, 10.0, 10.0, 0.0, 10.0], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Mesh(name="blocker", bsdf=_bsdfs.address_of(0), vertices=np.array([1.5, 1.0, -1.5, -1.5, 1.0, -1.5, -1.5, 1.0, 1.5], dtype=np.float32), indices=np.array([0, 1, 2], dtype=np.uint32)),
            shapes.Mesh(name="light", bsdf=_bsdfs.address_of(0), emission=spy.float3(1.0), vertices=np.array([1.0, 2.0, -1.0, 1.0, 2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0, -1.0], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
        ])
        _shapes.upload()

        _emitters = core.ResourceBuffer[emitters.Emitter](program, [
            emitters.Area(name="", shape=_shapes.address_of(2)),
            # emitters.Constant(name="", radiance=spy.float3(0.5)),
        ])
        _emitters.upload()

        _acceleration_structure = shapes.AccelerationStructure(_device)
        _acceleration_structure.update_blas(_shapes.entries)
        _acceleration_structure.update_tlas(_shapes.entries)

        _scene = Scene(program)
        _scene.sensors = _sensors
        _scene.bsdfs = _bsdfs
        _scene.shapes = _shapes
        _scene.emitters = _emitters
        _scene.acceleration_structure = _acceleration_structure
        return _scene

    @staticmethod
    def hosek_wilkie(program: spy.ShaderProgram, turbidity: float = 3.0, scale: float = 0.02,
                     latitude: float = 48.21, longitude: float = 16.37, timezone: float = 1.0,
                     date: tuple[int, int, int] = (2025, 6, 21), hour: float = 12.0):
        """A sphere on a ground plane, lit only by the Hosek-Wilkie sky.

        Midsummer noon in Vienna by default. Move the sun later via `emitters["sky"]`.
        """
        _device = program.device
        _sensors = core.ResourceBuffer[sensors.Sensor](program, [
            sensors.Perspective.from_look_at(name="cam", eye=[0, 1.0, -5], target=[0, 0.8, 0], up=[0, 1, 0], film=films.HDRFilm(_device), fov=55)
        ])
        _sensors.upload()

        _bsdfs = core.ResourceBuffer[bsdfs.Bsdf](program, [
            bsdfs.Diffuse(name="ground", reflectance=core.Texture(spy.float3(0.5))),
            bsdfs.Diffuse(name="sphere", reflectance=core.Texture(spy.float3(0.8, 0.3, 0.2))),
        ])
        _bsdfs.upload()

        _shapes = core.ResourceBuffer[shapes.Shape](program, [
            # huge: a plane's edge sits atan(eye height / extent) below the true
            # horizon, and the sky is black there, so a small one leaves a dark band
            shapes.Mesh(name="ground", bsdf=_bsdfs.address_of(0), vertices=np.array([1e4, 0.0, -1e4, -1e4, 0.0, -1e4, -1e4, 0.0, 1e4, 1e4, 0.0, 1e4], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Sphere(name="sphere", bsdf=_bsdfs.address_of(1), radius=0.8, to_world=core.Transform().translate([0.0, 0.8, 0.0])),
        ])
        _shapes.upload()

        _sky = emitters.HosekWilkie(turbidity=turbidity, scale=scale, name="sky")
        _sky.configure_from_time(latitude, longitude, timezone, *date, hour)
        _emitters = core.ResourceBuffer[emitters.Emitter](program, [_sky])
        _emitters.upload()

        _acceleration_structure = shapes.AccelerationStructure(_device)
        _acceleration_structure.update_blas(_shapes.entries)
        _acceleration_structure.update_tlas(_shapes.entries)

        _scene = Scene(program)
        _scene.sensors = _sensors
        _scene.bsdfs = _bsdfs
        _scene.shapes = _shapes
        _scene.emitters = _emitters
        _scene.acceleration_structure = _acceleration_structure
        return _scene

    @staticmethod
    def cornell_box(program: spy.ShaderProgram):
        _device = program.device
        _sensors = core.ResourceBuffer[sensors.Sensor](program, [
            sensors.Perspective.from_look_at(name="cam", eye=[278, 273, -800], target=[278, 273, 0], up=[0,1,0], film=films.HDRFilm(_device), fov=60)
        ])
        _sensors.upload()

        _bsdfs = core.ResourceBuffer[bsdfs.Bsdf](program, [
            bsdfs.Diffuse(name="white", reflectance=core.Texture(spy.float3(1.0, 1.0, 1.0))),
            bsdfs.Diffuse(name="green", reflectance=core.Texture(spy.float3(0.0, 1.0, 0.0))),
            bsdfs.Diffuse(name="red", reflectance=core.Texture(spy.float3(1.0, 0.0, 0.0))),
        ])
        _bsdfs.upload()

        _shapes = core.ResourceBuffer[shapes.Shape](program, [
            shapes.Mesh(name="Floor", bsdf=_bsdfs.address_of(0), vertices=np.array([552.8, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 559.2, 549.6, 0.0, 559.2], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Mesh(name="Ceiling", bsdf=_bsdfs.address_of(0), vertices=np.array([556.0, 548.8, 0.0, 556.0, 548.8, 559.2, 0.0, 548.8, 559.2, 0.0, 548.8, 0.0], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Mesh(name="Back", bsdf=_bsdfs.address_of(0), vertices=np.array([549.6, 0.0, 559.2, 0.0, 0.0, 559.2, 0.0, 548.8, 559.2, 556.0, 548.8, 559.2], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Mesh(name="Right", bsdf=_bsdfs.address_of(1), vertices=np.array([0.0, 0.0, 559.2, 0.0, 0.0, 0.0, 0.0, 548.8, 0.0, 0.0, 548.8, 559.2], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Mesh(name="Left", bsdf=_bsdfs.address_of(2), vertices=np.array([552.8, 0.0, 0.0, 549.6, 0.0, 559.2, 556.0, 548.8, 559.2, 556.0, 548.8, 0.0], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
            shapes.Mesh(name="Tall Block", bsdf=_bsdfs.address_of(0), vertices=np.array([
                423.0, 330.0, 247.0, 265.0, 330.0, 296.0, 314.0, 330.0, 456.0, 472.0, 330.0, 406.0, 
                423.0, 0.0, 247.0, 423.0, 330.0, 247.0, 472.0, 330.0, 406.0, 472.0, 0.0, 406.0, 
                472.0, 0.0, 406.0, 472.0, 330.0, 406.0, 314.0, 330.0, 456.0, 314.0, 0.0, 456.0,
                314.0, 0.0, 456.0, 314.0, 330.0, 456.0, 265.0, 330.0, 296.0, 265.0, 0.0, 296.0,
                265.0, 0.0, 296.0, 265.0, 330.0, 296.0, 423.0, 330.0, 247.0, 423.0, 0.0, 247.0,
                ], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19], dtype=np.uint32)),
            shapes.Mesh(name="Short Block", bsdf=_bsdfs.address_of(0), vertices=np.array([ 
                130.0, 165.0, 65.0, 82.0, 165.0, 225.0, 240.0, 165.0, 272.0, 290.0, 165.0, 114.0,
                290.0, 0.0, 114.0, 290.0, 165.0, 114.0, 240.0, 165.0, 272.0, 240.0, 0.0, 272.0,
                130.0, 0.0, 65.0, 130.0, 165.0, 65.0, 290.0, 165.0, 114.0, 290.0, 0.0, 114.0,
                82.0, 0.0, 225.0, 82.0, 165.0, 225.0, 130.0, 165.0, 65.0, 130.0, 0.0, 65.0,
                240.0, 0.0, 272.0, 240.0, 165.0, 272.0, 82.0, 165.0, 225.0, 82.0, 0.0, 225.0,
                ], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19], dtype=np.uint32)),
            shapes.Mesh(name="Light", bsdf=_bsdfs.address_of(0), emission=spy.float3(5.0), vertices=np.array([343.0, 548.7, 227.0, 343.0, 548.7, 332.0, 213.0, 548.7, 332.0, 213.0, 548.7, 227.0], dtype=np.float32), indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)),
        ])
        _shapes.upload()

        _emitters = core.ResourceBuffer[emitters.Emitter](program, [
            emitters.Area(name="", shape=_shapes.address_of(7)),
        ])
        _emitters.upload()

        _acceleration_structure = shapes.AccelerationStructure(_device)
        _acceleration_structure.update_blas(_shapes.entries)
        _acceleration_structure.update_tlas(_shapes.entries)

        _scene = Scene(program)
        _scene.sensors = _sensors
        _scene.bsdfs = _bsdfs
        _scene.shapes = _shapes
        _scene.emitters = _emitters
        _scene.acceleration_structure = _acceleration_structure
        return _scene
