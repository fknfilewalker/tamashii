from tamashii._spy import spy
import tamashii.core as core
import numpy as np
import numpy.typing as npt
import tamashii.emitters as emitters

class Shape(core.Interface):
    INTERFACE = "IShape<float>"
    blas: spy.AccelerationStructure | None = None
    mask: int = 0xFF
    bsdf: int
    emission: 'spy.float3param'
    to_world: 'spy.float4x4'
    to_world_tensor: spy.Tensor | None = None

class Rectangle(Shape):

    def __init__(self, 
                 bsdf: int, 
                 scale: 'spy.float2param'=spy.float2(1.0),
                 emission: 'spy.float3param'=spy.float3(0.0), 
                 to_world: core.Transform=core.Transform(), 
                 mask: int=0xFF,
                 name: str=""):
        self.name = name
        self.mask = mask
        self.to_world = spy.math.transpose(to_world.scale(spy.float3(scale[0], 1.0, scale[1])))
        self.bsdf = bsdf
        self.film = 0
        self.emission = emission

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.bsdf = self.bsdf
        cursor.film = self.film
        cursor.emission = self.emission
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)
        cursor.aabb.min = spy.float3(-0.5, 0.0, -0.5)
        cursor.aabb.max = spy.float3(0.5, 0.0, 0.5)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class Sphere(Shape):

    def __init__(self, 
                 bsdf: int, 
                 radius: float=0.5, 
                 emission: 'spy.float3param'=spy.float3(0.0), 
                 to_world: core.Transform=core.Transform(), 
                 mask: int=0xFF,
                 name: str=""):
        self.name = name
        self.mask = mask
        self.to_world = spy.math.transpose(to_world.scale(spy.float3(2 * radius)))
        self.bsdf = bsdf
        self.film = 0
        self.emission = emission

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.bsdf = self.bsdf
        cursor.emission = self.emission
        cursor.film = self.film
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)
        cursor.aabb.min = spy.float3(-0.5)
        cursor.aabb.max = spy.float3(0.5)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class Mesh(Shape):

    def __init__(self, 
                 bsdf: int, 
                 vertices: npt.NDArray[np.float32], *, 
                 indices: npt.NDArray[np.uint32]=np.array([], dtype=np.uint32),
                 normals: npt.NDArray[np.uint32]=np.array([], dtype=np.uint32),
                 texcoords: npt.NDArray[np.uint32]=np.array([], dtype=np.uint32),
                 colors: npt.NDArray[np.float32]=np.array([], dtype=np.float32),
                 emission: 'spy.float3param'=spy.float3(0.0), 
                 to_world: core.Transform=core.Transform(), 
                 mask: int=0xFF,
                 name: str=""):
        assert len(vertices), "Mesh needs at least one vertex"
        self.name = name
        self.mask = mask
        self.to_world = spy.math.transpose(to_world)
        self.bsdf = bsdf
        self.film = 0
        self.vertices = vertices
        self.indices = indices
        self.normals = normals
        self.texcoords = texcoords
        self.colors = colors
        self.vertex_count = len(vertices)//3
        self.emission = emission
        self.index_count = len(indices)

        verts = vertices.reshape(-1, 3)
        self.aabb_min = spy.float3(*verts.min(axis=0))
        self.aabb_max = spy.float3(*verts.max(axis=0))

        self.barycentric_areas = np.zeros(self.vertex_count, dtype=np.float32)
        primitive_count = self.index_count // 3 if self.index_count > 0 else self.vertex_count // 3
        for i in range(primitive_count):
            if self.index_count > 0:
                idx0 = self.indices[i*3+0] * 3
                idx1 = self.indices[i*3+1] * 3
                idx2 = self.indices[i*3+2] * 3
            else:
                idx0 = i*9 + 0
                idx1 = i*9 + 3
                idx2 = i*9 + 6
            v0 = self.vertices[idx0:idx0+3]
            v1 = self.vertices[idx1:idx1+3]
            v2 = self.vertices[idx2:idx2+3]
            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
            self.barycentric_areas[idx0//3] += area / 3.0
            self.barycentric_areas[idx1//3] += area / 3.0
            self.barycentric_areas[idx2//3] += area / 3.0

        # Build neighbor adjacency: for each triangle, find the neighboring triangle
        # sharing each edge. neighbors[i] = (n01, n12, n20) where:
        #   n01 = triangle sharing edge (v0, v1), or -1
        #   n12 = triangle sharing edge (v1, v2), or -1
        #   n20 = triangle sharing edge (v2, v0), or -1
        self.neighbors = np.full((primitive_count, 3), -1, dtype=np.int32)
        edge_map: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(primitive_count):
            if self.index_count > 0:
                tri = (int(self.indices[i*3]), int(self.indices[i*3+1]), int(self.indices[i*3+2]))
            else:
                tri = (i*3, i*3+1, i*3+2)
            for e in range(3):
                a, b = tri[e], tri[(e + 1) % 3]
                key = (min(a, b), max(a, b))
                if key in edge_map:
                    other_tri, other_edge = edge_map[key]
                    self.neighbors[i, e] = other_tri
                    self.neighbors[other_tri, other_edge] = i
                else:
                    edge_map[key] = (i, e)
        self.neighbors = self.neighbors.flatten()

        # GPU vertex positions; attach gradients with spy's own API when needed:
        #   mesh.positions = mesh.positions.with_grads(zero=True)
        self.positions: spy.Tensor | None = None
        self.ibuffer: spy.Tensor | None = None
        self.nbuffer: spy.Tensor | None = None
        self.tcbuffer: spy.Tensor | None = None
        self.babuffer: spy.Tensor | None = None
        self.neighbuffer: spy.Tensor | None = None

    def subdivide(self, levels: int = 1, catmull_clark: bool = True):
        """
        Subdivides the mesh. 
        If catmull_clark=True, implements Loop Subdivision (Catmull-Clark equivalent for triangles).
        If catmull_clark=False, implements simple linear subdivision (adds geometry without smoothing).
        """

        if levels <= 0:
            return

        for level in range(levels):
            # 1. Reshape current vertices and ensure mesh is indexed
            V = self.vertices.reshape(-1, 3)
            if self.index_count > 0:
                T = self.indices.reshape(-1, 3)
            else:
                T = np.arange(self.vertex_count, dtype=np.uint32).reshape(-1, 3)
                
            num_v = V.shape[0]
            num_t = T.shape[0]
            
            edge_to_new_v = {}
            new_vertices = []
            
            # Adjacency tracking for original vertex updates
            adj = [set() for _ in range(num_v)]
            boundary_edges = [[] for _ in range(num_v)]
            
            # 2. Compute new edge vertices and build adjacency lists
            for i in range(num_t):
                for e in range(3):
                    vA = int(T[i, e])
                    vB = int(T[i, (e + 1) % 3])
                    vC = int(T[i, (e + 2) % 3])
                    
                    adj[vA].add(vB)
                    adj[vB].add(vA)
                    
                    key = (min(vA, vB), max(vA, vB))
                    if key not in edge_to_new_v:
                        neighbor_tri = self.neighbors[i * 3 + e]
                        
                        if catmull_clark and neighbor_tri != -1:
                            # Smoothed interior edge rule
                            nT = T[neighbor_tri]
                            vD = int([v for v in nT if v != vA and v != vB][0])
                            new_v = 0.375 * (V[vA] + V[vB]) + 0.125 * (V[vC] + V[vD])
                        else:
                            # Linear midpoint rule (used for boundaries, or if catmull_clark=False)
                            new_v = 0.5 * (V[vA] + V[vB])
                            
                        if neighbor_tri == -1:
                            boundary_edges[vA].append(vB)
                            boundary_edges[vB].append(vA)
                            
                        edge_to_new_v[key] = num_v + len(new_vertices)
                        new_vertices.append(new_v)
                        
            # 3. Compute smoothed positions for original vertices
            updated_V = np.zeros_like(V)
            for v in range(num_v):
                if not catmull_clark:
                    # Keep original positions for simple linear subdivision
                    updated_V[v] = V[v]
                elif len(boundary_edges[v]) >= 2:
                    # Boundary vertex rule
                    b1, b2 = boundary_edges[v][:2]
                    updated_V[v] = 0.75 * V[v] + 0.125 * (V[b1] + V[b2])
                else:
                    # Interior vertex rule
                    n = len(adj[v])
                    if n > 0:
                        beta = (1.0 / n) * (0.625 - (0.375 + 0.25 * np.cos(2.0 * np.pi / n))**2)
                        neighbor_sum = np.sum([V[nb] for nb in adj[v]], axis=0)
                        updated_V[v] = (1.0 - n * beta) * V[v] + beta * neighbor_sum
                    else:
                        updated_V[v] = V[v]
                        
            # Combine original updated vertices with newly generated edge vertices
            if new_vertices:
                all_V = np.vstack([updated_V, np.array(new_vertices, dtype=np.float32)])
            else:
                all_V = updated_V

            # 4. Form new triangles (1 original -> 4 new)
            new_T = []
            for i in range(num_t):
                v0, v1, v2 = int(T[i, 0]), int(T[i, 1]), int(T[i, 2])
                e01 = edge_to_new_v[(min(v0, v1), max(v0, v1))]
                e12 = edge_to_new_v[(min(v1, v2), max(v1, v2))]
                e20 = edge_to_new_v[(min(v2, v0), max(v2, v0))]
                
                new_T.extend([
                    [v0, e01, e20],
                    [v1, e12, e01],
                    [v2, e20, e12],
                    [e01, e12, e20]
                ])
                
            self.vertices = all_V.flatten()
            self.indices = np.array(new_T, dtype=np.uint32).flatten()
            self.vertex_count = len(all_V)
            self.index_count = len(self.indices)
            
            # 5. Rebuild neighbor adjacency for the new geometry
            primitive_count = self.index_count // 3
            self.neighbors = np.full((primitive_count, 3), -1, dtype=np.int32)
            edge_map = {}
            for i in range(primitive_count):
                tri_indices = (int(self.indices[i*3]), int(self.indices[i*3+1]), int(self.indices[i*3+2]))
                for e in range(3):
                    a, b = tri_indices[e], tri_indices[(e + 1) % 3]
                    key = (min(a, b), max(a, b))
                    if key in edge_map:
                        other_tri, other_edge = edge_map[key]
                        self.neighbors[i, e] = other_tri
                        self.neighbors[other_tri, other_edge] = i
                    else:
                        edge_map[key] = (i, e)
            self.neighbors = self.neighbors.flatten()

        # 6. Finalize bounds and recompute barycentric areas
        verts = self.vertices.reshape(-1, 3)
        self.aabb_min = spy.float3(*verts.min(axis=0))
        self.aabb_max = spy.float3(*verts.max(axis=0))
        
        self.barycentric_areas = np.zeros(self.vertex_count, dtype=np.float32)
        primitive_count = self.index_count // 3
        for i in range(primitive_count):
            idx0, idx1, idx2 = self.indices[i*3], self.indices[i*3+1], self.indices[i*3+2]
            v0, v1, v2 = verts[idx0], verts[idx1], verts[idx2]
            
            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
            area_third = area / 3.0
            
            self.barycentric_areas[idx0] += area_third
            self.barycentric_areas[idx1] += area_third
            self.barycentric_areas[idx2] += area_third

    @staticmethod
    def from_file(bsdf: int, 
                  file: str, *, 
                  to_world: 'spy.float4x4param'=spy.float4x4().identity(),
                  mask: int=0xFF,
                  name: str=""):
        import trimesh
        mesh = trimesh.load(file)
        vertices = mesh.vertices.astype(np.float32).flatten()
        indices = mesh.faces.astype(np.uint32).flatten()
        normals = mesh.vertex_normals.astype(np.float32).flatten()
        if hasattr(mesh.visual, 'uv'):
            texcoords = mesh.visual.uv.astype(np.float32).flatten()
        else:
            texcoords = np.zeros(0, dtype=np.float32)
        if hasattr(mesh.visual, 'vertex_colors'):
            # Convert 0-255 uint8 to 0.0-1.0 float32
            # We take [:3] if you only want RGB, or leave as is for RGBA
            colors = (mesh.visual.vertex_colors / 255.0).astype(np.float32).flatten()
        else:
            colors = np.ones(len(mesh.vertices) * 3, dtype=np.float32)
        return Mesh(name=name, bsdf=bsdf, vertices=vertices, indices=indices, normals=normals, texcoords=texcoords, to_world=to_world, mask=mask)

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        cursor.bsdf = self.bsdf
        cursor.film = self.film
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)
        cursor.emission = self.emission
        # cursor.colors = 0
        self.positions = core.upload_tensor(program, cursor, "positions", self.positions, self.vertices.reshape(-1, 3), "float3")
        self.ibuffer = core.upload_tensor(program, cursor, "indices", self.ibuffer, self.indices.reshape(-1, 3), "uint3")
        self.nbuffer = core.upload_tensor(program, cursor, "normals", self.nbuffer, self.normals.reshape(-1, 3), "float3")
        self.tcbuffer = core.upload_tensor(program, cursor, "texcoords_0", self.tcbuffer, self.texcoords.reshape(-1, 2), "float2")
        self.babuffer = core.upload_tensor(program, cursor, "barycentric_areas", self.babuffer, self.barycentric_areas.reshape(-1, 1), "float")
        self.neighbuffer = core.upload_tensor(program, cursor, "neighbors", self.neighbuffer, self.neighbors.reshape(-1, 3), "int3")

        if len(self.colors):
            self.cbuffer = core.create_or_resize_buffer(program.device, self.cbuffer, self.colors, "Mesh.color_buffer")
            cursor.colors = self.cbuffer.device_address

        cursor.aabb.min = self.aabb_min
        cursor.aabb.max = self.aabb_max

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class Instance(Shape):
    """Another placement of `shape`: shares its geometry buffers and BLAS, with its own
    bsdf, emission, transform and mask (defaulting to the shape's)."""

    def __init__(self,
                 shape: Shape,
                 bsdf: int | None=None,
                 emission: 'spy.float3param | None'=None,
                 to_world: core.Transform | None=None,
                 mask: int | None=None,
                 name: str=""):
        self.shape = shape
        self.bsdf = shape.bsdf if bsdf is None else bsdf
        self.emission = shape.emission if emission is None else emission
        self.to_world = shape.to_world if to_world is None else spy.math.transpose(to_world)
        self.mask = shape.mask if mask is None else mask
        self.name = name

    blas = property(lambda self: self.shape.blas)  # type: ignore[assignment]
    implementation_type_name = property(lambda self: self.shape.implementation_type_name)  # type: ignore[override]

    def write(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor | spy.ShaderCursor):
        self.shape.write(program, cursor)
        cursor.bsdf = self.bsdf
        cursor.emission = self.emission
        self.to_world_tensor = core.upload_transform(program, cursor, "to_world", self.to_world_tensor, self.to_world)

    def read(self, program: spy.ShaderProgram, cursor: spy.BufferElementCursor):
        pass

class AccelerationStructure:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self._blas_in_use: dict[int, spy.AccelerationStructure] = {}  # BLASes of the last TLAS build, by id
        self._retired_blas: list[spy.AccelerationStructure] = []  # of removed shapes, see update_tlas

        # build general blas 
        aux_data = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5,
                            -0.5, 0.0, -0.5, 0.5, 0.0, 0.5,
                            -0.5, 0.0, -0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5,
                            0.5, 0.0, 0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5], dtype=np.float32)

        self.aux_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="aux_buffer",
            data=aux_data,
        )

        blas_input = spy.AccelerationStructureBuildInputProceduralPrimitives({
            "aabb_buffers": [self.aux_buffer],
            "aabb_stride": 24,
            "primitive_count": 1,
            "flags": spy.AccelerationStructureGeometryFlags.opaque | spy.AccelerationStructureGeometryFlags.no_duplicate_any_hit_invocation,
        })

        blas_build_desc = spy.AccelerationStructureBuildDesc({
            "inputs": [blas_input],
        })

        blas_sizes = device.get_acceleration_structure_sizes(blas_build_desc)

        blas_scratch_buffer = device.create_buffer(
            size=blas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="blas_scratch_buffer",
        )

        self.unit_aabb_blas = device.create_acceleration_structure(
            kind=spy.AccelerationStructureKind.bottom_level,
            size=blas_sizes.acceleration_structure_size,
            label="blas_aabb",
        )
        command_encoder = device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=blas_build_desc, dst=self.unit_aabb_blas, src=None, scratch_buffer=blas_scratch_buffer
        )
        device.wait_for_submit(device.submit_command_buffer(command_encoder.finish()))

        # rect
        blas_input =  spy.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [{
                        "buffer": self.aux_buffer,
                        "offset": 2 * 24,
                    }],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": 6,
                "vertex_stride": 12,
                "flags": spy.AccelerationStructureGeometryFlags.opaque | spy.AccelerationStructureGeometryFlags.no_duplicate_any_hit_invocation,
            })

        blas_build_desc = spy.AccelerationStructureBuildDesc({
            "inputs": [blas_input],
        })

        blas_sizes = device.get_acceleration_structure_sizes(blas_build_desc)

        blas_scratch_buffer = device.create_buffer(
            size=blas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="blas_scratch_buffer",
        )

        self.unit_rect_blas = device.create_acceleration_structure(
            kind=spy.AccelerationStructureKind.bottom_level,
            size=blas_sizes.acceleration_structure_size,
            label="blas_rect",
        )

        command_encoder = device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=blas_build_desc, dst=self.unit_rect_blas, src=None, scratch_buffer=blas_scratch_buffer
        )
        device.wait_for_submit(device.submit_command_buffer(command_encoder.finish()))

    def update_blas(self, shapes: list[Shape]):
        for i in range(len(shapes)):
            shape = shapes[i]
            if isinstance(shape, emitters.Area):
                shape = shape.shape
            if isinstance(shape, Instance):
                shape = shape.shape
            if not isinstance(shape, Mesh):
                continue
            
            blas_input_triangles = spy.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [shape.positions.storage],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": shape.vertex_count,
                "vertex_stride": shape.vertices.itemsize * 3,
                "index_buffer": shape.ibuffer.storage if shape.ibuffer is not None else {},
                "index_format": spy.IndexFormat.uint32,
                "index_count": shape.index_count,
                "flags": spy.AccelerationStructureGeometryFlags.opaque | spy.AccelerationStructureGeometryFlags.no_duplicate_any_hit_invocation,
            })
            blas_build_desc = spy.AccelerationStructureBuildDesc({
                "inputs": [blas_input_triangles],
            })

            blas_sizes = self.device.get_acceleration_structure_sizes(blas_build_desc)

            blas_scratch_buffer = self.device.create_buffer(
                size=blas_sizes.scratch_size,
                usage=spy.BufferUsage.unordered_access,
                label="blas_scratch_buffer",
            )

            shape.blas = self.device.create_acceleration_structure(
                kind=spy.AccelerationStructureKind.bottom_level,
                size=blas_sizes.acceleration_structure_size,
                label="blas",
            )

            command_encoder = self.device.create_command_encoder()
            command_encoder.build_acceleration_structure(
                desc=blas_build_desc, dst=shape.blas, src=None, scratch_buffer=blas_scratch_buffer
            )
            self.device.wait_for_submit(self.device.submit_command_buffer(command_encoder.finish()))
            self.device.wait_for_idle()

    def update_tlas(self, shapes: list[Shape]):
        # slang-rhi's Metal backend nulls a freed acceleration structure in the list every TLAS build
        # hands to Metal, and the next build aborts. Keep the BLASes of removed or rebuilt shapes alive.
        in_use = {id(s.blas): s.blas for s in shapes if s.blas is not None}
        self._retired_blas += [b for key, b in self._blas_in_use.items() if key not in in_use]
        self._blas_in_use = in_use

        instance_list = self.device.create_acceleration_structure_instance_list(len(shapes))
        for i, shape in enumerate(shapes):
            geometry = shape
            if isinstance(geometry, Instance):
                geometry = geometry.shape
            if isinstance(geometry, Sphere):
                blas = self.unit_aabb_blas
                hit_index = 2
            elif isinstance(geometry, Rectangle):
                blas = self.unit_rect_blas
                hit_index = 1
            else:
                blas = geometry.blas
                hit_index = 0
            assert blas is not None, f"shape '{geometry.name}' has no BLAS, run update_blas first"

            instance_list.write(i, {
                "transform": spy.float3x4(spy.math.transpose(shape.to_world)),
                "instance_id": i,
                "instance_mask": shape.mask,
                "instance_contribution_to_hit_group_index": hit_index,
                "flags": spy.AccelerationStructureInstanceFlags.none,
                "acceleration_structure": blas.handle,
            })

        tlas_build_desc = spy.AccelerationStructureBuildDesc({
            "inputs": [instance_list.build_input_instances()],
        })

        tlas_sizes = self.device.get_acceleration_structure_sizes(tlas_build_desc)

        tlas_scratch_buffer = self.device.create_buffer(
            size=tlas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch_buffer",
        )

        tlas = self.device.create_acceleration_structure(
            kind=spy.AccelerationStructureKind.top_level,
            size=tlas_sizes.acceleration_structure_size,
            label="tlas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=tlas_build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
        )
        self.device.wait_for_submit(self.device.submit_command_buffer(command_encoder.finish()))
        self.device.wait_for_idle()
        self.tlas = tlas

core.register_plugin(Rectangle, "shapes::Rectangle<float>")
core.register_plugin(Sphere, "shapes::Sphere<float>")
core.register_plugin(Mesh, "shapes::Mesh<float>")
