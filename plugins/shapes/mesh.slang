implementing tamashii;

namespace shapes {
    public struct Edge<Real : floating_point> : IDifferentiable {
        public v3<Real> pos[2];
        public v3<Real> interpolate(Real t) { return fma(pos[0], Real(1.0) - t, pos[1] * t); }
        public v3<Real> direction() { return pos[1] - pos[0]; }
    };

    public struct Triangle<Real : floating_point> : IDifferentiable {
        public v3<Real> pos[3];

        public Edge<Real> edge(uint i) {
            return { { pos[i], pos[(i + 1) % 3] } };
        }

        // https://people.cs.clemson.edu/~dhouse/courses/404/notes/barycentric.pdf
        // updates tmax on hit
        [Differentiable]
        public bool intersect(inout Ray<Real> ray, const CullMode cullmode, inout v2<Real> barycentric)
        {
            // clockwise
            const v3<Real> e1 = pos[1].xyz - pos[0].xyz;
            const v3<Real> e2 = pos[2].xyz - pos[0].xyz;
            // counter clockwise
            // const v3<Real> e1 = pos[2].xyz - pos[0].xyz;
            // const v3<Real> e2 = pos[1].xyz - pos[0].xyz;

            const v3<Real> h = cross(ray.direction, e2);
            const Real a = dot(e1, h);

            // if (a > -0.00001f && a < 0.00001f) return false;
            bool front_facing = a > Real(0);
            if ((CullMode::Front == cullmode && front_facing) ||
                (CullMode::Back == cullmode && !front_facing) ||
                CullMode::Both == cullmode) return false;

            const Real f = Real(1) / a;
            const v3<Real> s = ray.origin - pos[0].xyz;
            const Real u = f * (dot(s, h));

            if (u < Real(0) || u > Real(1))
                return false;

            const v3<Real> q = cross(s, e1);
            const Real v = f * dot(ray.direction, q);

            if (v < Real(0) || u + v > Real(1))
                return false;

            // at this stage we can compute t to find out where
            // the intersection point is on the line
            Real t = f * dot(e2, q);

            // this means that there is a line intersection
            // but not a ray intersection
            if (t <= Real(0.00001)) return false;
            if (t < ray.tmin || t > ray.tmax) return false;

            // compute the intersection point using equation 1
            const v3<Real> hitPos = ray.origin + t * ray.direction;
            const v3<Real> normal = cross(e1, e2); // N
            const Real denom = dot(normal, normal);

            // Step 2: barycentric coordinates
            // vector perpendicular to triangle's plane
            v3<Real> C = cross(pos[1].xyz - pos[0].xyz, hitPos - pos[0].xyz);
            if (dot(normal, C) < Real(0)) return false; // P is on the right side

            v2<Real> bary;
            // u
            C = cross(pos[0].xyz - pos[2].xyz, hitPos - pos[2].xyz);
            if ((bary.x = dot(normal, C)) < Real(0)) return false; // P is on the right side
            // v
            C = cross(pos[1].xyz - pos[0].xyz, hitPos - pos[0].xyz);
            if ((bary.y = dot(normal, C)) < Real(0)) return false; // P is on the right side;

            barycentric = bary / denom;
            ray.tmax = t;
            return true;
        }
    };

    public struct Mesh<Real : atomicable_floating_point> : IShape<Real>, IDifferentiable {
        typealias Real2 = vector<Real, 2>;
        typealias Real3 = vector<Real, 3>;
        typealias Real4 = vector<Real, 4>;

        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public AABB<Real> aabb;
        public Ptr<IBsdf<Real>> bsdf;
        public IFilm<Real, 4>* film;
        public v3<Real> emission;
        public no_diff DiffTensor<v3<Real>, 1> positions;
        public no_diff Tensor<v3<Real>, 1> normals;
        public no_diff Tensor<v2<Real>, 1> texcoords_0;
        public no_diff Tensor<Real, 1> barycentric_areas;
        public no_diff Tensor<uint3, 1> indices;   // one uint3 element per triangle
        public no_diff Tensor<int3, 1> neighbors;  // one int3 element per triangle, -1 = no neighbor

        public __init() {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.bsdf = nullptr;
            this.film = nullptr;
            this.emission = v3<Real>(Real(0), Real(0), Real(0));
            this.positions = DiffTensor<v3<Real>, 1>();
            this.normals = Tensor<v3<Real>, 1>();
            this.texcoords_0 = Tensor<v2<Real>, 1>();
            this.barycentric_areas = Tensor<Real, 1>();
            this.indices = Tensor<uint3, 1>();
            this.neighbors = Tensor<int3, 1>();
            this.aabb = AABB<Real>(v3<Real>(Real(0)), v3<Real>(Real(0)));
        }

        public uint primitive_count() { return triangle_count(); }

        public bool has_indices() { return indices.is_valid(); }
        public uint vertex_count() { return positions.shape[0]; }
        public uint index_count() { return indices.shape[0] * 3u; }
        public uint triangle_count() {
            if (has_indices()) return indices.shape[0];
            else return vertex_count() / 3u;
        }
        public bool has_normals() { return normals.is_valid(); }
        public bool has_texcoords() { return texcoords_0.is_valid(); }
        public bool has_neighbors() { return neighbors.is_valid(); }
        public bool has_dpositions() { return positions.has_grad(); }

        // returns for each edge whether there is a discontinuity (neighboring triangle is facing different direction)
        public bool3 has_discontinuity(uint primitive, v3<Real> ray_dir) {
            if (!has_dpositions()) return bool3(false);
            bool3 result = bool3(true);
            if (!has_neighbors()) return result;
            int3 n = neighbors[{primitive}];
            [ForceUnroll]
            for (int i = 0; i < 3; i++) {
                if (n[i] >= 0) {
                    var tri = get_triangle(get_indices(uint(n[i])), to_world.to_transform());
                    v3<Real> geo_normal = normalize(cross(tri.pos[1] - tri.pos[0], tri.pos[2] - tri.pos[0]));
                    result[i] = dot(geo_normal, ray_dir) < Real(0);
                }
            }
            return result;
        }

        public uint3 get_indices(uint primitive) {
            if (!has_indices()) {
                const uint trivert_offset = primitive * 3;
                return { trivert_offset, trivert_offset + 1, trivert_offset + 2 };
            }
            return indices[{primitive}];
        }

        public uint2 get_edge_indices(uint3 indices, uint i) {
            return { indices[i], indices[(i + 1) % 3] };
        }

        [Differentiable]
        public Triangle<Real> get_triangle(uint3 indices, Transform<Real> transform = Transform<Real>::identity()) {
            Triangle<Real> tri;
            tri.pos[0] = mul(v4<Real>(positions[{indices.x}], Real(1)), transform.forward).xyz;
            tri.pos[1] = mul(v4<Real>(positions[{indices.y}], Real(1)), transform.forward).xyz;
            tri.pos[2] = mul(v4<Real>(positions[{indices.z}], Real(1)), transform.forward).xyz;
            return tri;
        }

        [Differentiable, PreferRecompute]
        __generic<T : floating_point, int N>
        public static vector<T, N> interpolate(vector<T, N> v0, vector<T, N> v1, vector<T, N> v2, vector<T, 2> b) {
            return fma(v0, T(1.0) - b.x - b.y, fma(v1, b.x, v2 * b.y));
        }

        [Differentiable, PreferRecompute]
        __generic<T : floating_point, int N>
        public static vector<T, N> interpolate(vector<T, N> v0, vector<T, N> v1, vector<T, N> v2, vector<T, 3> b) {
            return fma(v0, b.x, fma(v1, b.y, v2 * b.z));
        }

        [Differentiable]
        public SurfacePoint<Real> surface_point(uint32_t primitive, v2<Real> attribute, Transform<Real> transform) {
            uint3 indices = get_indices(primitive);
            Transform<Real> t = to_world.to_transform() * transform;
            var tri = get_triangle(indices, t);
            SurfacePoint<Real> sp;
            sp.position = interpolate(tri.pos[0], tri.pos[1], tri.pos[2], attribute);
            sp.geometric_normal = cross(tri.pos[1] - tri.pos[0], tri.pos[2] - tri.pos[0]);
            sp.area = length(sp.geometric_normal) * Real(0.5);
            let geo_normal = normalize(sp.geometric_normal);
            // if (has_normals()) {
            //     sp.frame = Frame<Real>(normalize(mul(interpolate(normals[{indices.x}], normals[{indices.y}], normals[{indices.z}], attribute), transpose(m3x3<Real>(t.inverse)))));
            // } else
            sp.frame = Frame<Real>(geo_normal);
            sp.geometric_normal = v3<Real>(Real(0), Real(0), Real(1));
            sp.uv = v2<Real>(Real(0), Real(0));
            if (has_texcoords())
                sp.uv = interpolate(texcoords_0[{indices.x}], texcoords_0[{indices.y}], texcoords_0[{indices.z}], attribute);
            sp.emission = emission;
            sp.primitive = primitive;
            sp.attribute = attribute;
            return sp;
        }

        [Differentiable] public SurfaceInteraction<Real> _surface_interaction(Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
            var sp = surface_point(primitive, attribute, transform);
            let geo_normal_world = sp.frame.to_world(sp.geometric_normal);
            if (dot(ray.direction, geo_normal_world) > Real(0)) {
                sp.frame = Frame<Real>(-sp.frame.normal());
                sp.geometric_normal = sp.frame.to_local(geo_normal_world);
            }
            SurfaceInteraction<Real> si;
            si.sp = sp;
            si.wi = sp.frame.to_local(-ray.direction);
            return si;
        }

        public SurfaceInteraction<Real> surface_interaction(Ray<Real> ray, uint32_t primitive, no_diff v2<Real> attribute, bool attach_to_geometry = false, Transform<Real> transform = Transform<Real>::identity()) {
            return _surface_interaction(ray, primitive, attribute, transform);
        }

        [Differentiable, PrimalSubstituteOf(surface_interaction)] public SurfaceInteraction<Real> sub_surface_interaction(Ray<Real> ray, uint32_t primitive, no_diff v2<Real> attribute, bool attach_to_geometry = false, Transform<Real> transform = Transform<Real>::identity()) {
            return surface_intersection(ray, primitive, attach_to_geometry, transform);
        }

        [Differentiable]
        public SurfaceInteraction<Real> surface_intersection(Ray<Real> ray, uint32_t primitive, bool attach_to_geometry, Transform<Real> transform = Transform<Real>::identity()) {
            uint3 indices = get_indices(primitive);
            var tri = get_triangle(indices, to_world.to_transform());
            v3<Real> v[3] = { tri.pos[0], tri.pos[1], tri.pos[2] };
            let t = compute_ray_distance_to_triangle(v, ray.origin, ray.direction);
            let bary = compute_barycentrics(v, ray.origin + ray.direction * t);
            var attribute = bary_to_attr(bary);
            if (attach_to_geometry) attribute = detach(attribute);
            return _surface_interaction(ray, primitive, attribute, transform);
        }

        [Differentiable] public SurfaceSample<Real> sample_surface(v3<Real> uv, Transform<Real> transform) {
            let tri_idx = DiscreteDistribution::sample_evenly_spaced_range(0u, triangle_count() - 1u, uv.x);
            SurfaceSample<Real> ss;
            ss.sp = surface_point(tri_idx, warp::square_to_uniform_triangle(uv.yz), transform);
            ss.pdf = Real(1) / (Real(triangle_count()) * ss.sp.area);
            ss.delta = false;
            return ss;
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, v3<Real> uv, Transform<Real> transform) {
            let tri_idx = DiscreteDistribution::sample_evenly_spaced_range(0u, triangle_count() - 1u, uv.x);
            DirectionSample<Real> ds;
            ds.sp = surface_point(tri_idx, warp::square_to_uniform_triangle(uv.yz), transform);
            ds.direction = ds.sp.position - si.sp.position;
            ds.distance = length(ds.direction);
            ds.direction /= ds.distance;
            ds.delta = false;
            ds.pdf = warp::area_to_solid_angle(Real(primitive_count()) * ds.sp.area, ds.distance, dot(-ds.direction, ds.sp.frame.normal()));
            return ds;
        }

        [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds) {
            return warp::area_to_solid_angle(Real(primitive_count()) * ds.sp.area, ds.distance, dot(-ds.direction, ds.sp.frame.normal()));
        }

        public EdgeSamplePixel<Real> sample_confined_edge(uint32_t primitive, v2<Real> attribute, ISensor<Real> *_sensor, experiment::AABB2D<Real> pixel_aabb_uv, v2<Real> sample, Transform<Real> transform) {
            uint3 indices = get_indices(primitive);
            var tri = get_triangle(indices, to_world.to_transform());
            v3<Real> hp = interpolate(tri.pos[0], tri.pos[1], tri.pos[2], attribute);
            var geo_normal = normalize(cross(tri.pos[1] - tri.pos[0], tri.pos[2] - tri.pos[0]));

            bool flip = dot(geo_normal, _sensor->sample_lens({{hp}, {}}, {}).direction) < Real(0);
            // bool flip = dot(geo_normal, normalize(hp - _sensor->position().xyz)) > Real(0);

            // transform pixel to ndc space
            experiment::AABB2D<Real> pixel_aabb_ndc;
            pixel_aabb_ndc.min = pixel_aabb_uv.min * Real(2) - Real(1);
            pixel_aabb_ndc.max = pixel_aabb_uv.max * Real(2) - Real(1);

            // transform triangle to ndc space
            v4<Real> v0_cs = _sensor->splat(v4<Real>(tri.pos[0], Real(1)));
            v4<Real> v1_cs = _sensor->splat(v4<Real>(tri.pos[1], Real(1)));
            v4<Real> v2_cs = _sensor->splat(v4<Real>(tri.pos[2], Real(1)));
            v3<Real> v0_ndc = v0_cs.xyz / v0_cs.w; // v0_ndc.y = -v0_ndc.y;
            v3<Real> v1_ndc = v1_cs.xyz / v1_cs.w; // v1_ndc.y = -v1_ndc.y;
            v3<Real> v2_ndc = v2_cs.xyz / v2_cs.w; // v2_ndc.y = -v2_ndc.y;
            // clip against pixel
            experiment::Triangle2D<Real> t2d = experiment::Triangle2D<Real>( { v0_ndc.xy, v1_ndc.xy, v2_ndc.xy });
            var clipped = t2d.clip(pixel_aabb_ndc);

            // area of triangle in pixel // not 100% accurate?
            // https://web.archive.org/web/20100405070507/http://valis.cs.uiuc.edu/~sariel/research/CG/compgeom/msg00831.html
            // https://stackoverflow.com/a/451482
            // https://en.wikipedia.org/wiki/Shoelace_formula
            Real area = Real(0);
            for (int i = 0; i < clipped._1; i++) {
                v2<Real> a = clipped._0[i % clipped._1];
                v2<Real> b = clipped._0[(i + 1) % clipped._1];
                area += (a.y + b.y) * (a.x - b.x); // Trapezoid formula
                // area += (a.x * b.y - a.y * b.x); // Triangle formula
            }
            area = abs(Real(0.5) * area) / pixel_aabb_ndc.area(); // scale to unit pixel size

            // edge sampler preparation
            var intersected = t2d.intersect(pixel_aabb_ndc);
            int count = 0;
            Real pmf[3], cdf[4];
            {
                cdf[0] = Real(0);
                for (int i = 0; i < 3; i++) {
                    if (intersected._0[i]) {
                        pmf[count] = intersected._1[i].length();
                        cdf[++count] = cdf[count - 1] + pmf[count - 1];
                    } else {
                        pmf[count] = Real(0);
                        cdf[++count] = cdf[count - 1];
                    }
                }
                Real length_sum = cdf[count];
                for (int i = 0; i < count;) {
                    pmf[i] /= length_sum;
                    cdf[++i] /= length_sum;
                }
            }
            if ((intersected._0[0] + intersected._0[1] + intersected._0[2]) == 0) return {};
            // select one point on edges contained in pixel
            Real rand = sample.x;
            Real t = Real(0);
            int edge_idx = 0;
            for (int i = 0; i < count; i++) {
                if (rand < cdf[i + 1]) {
                    edge_idx = i;
                    t = (sample.x - cdf[i]) / (cdf[i + 1] - cdf[i]);
                    break;
                }
            }
            // t = sample.y;

            var orig_edge = tri.edge(edge_idx);

            var ndc_edge = intersected._1[edge_idx];
            var ndc_edge_dir = ndc_edge.direction();
            // var ndc_edge_length = length(ndc_edge_dir);
            // var ndc_edge_dir_n = ndc_edge_dir / ndc_edge_length;
            var ndc_edge_normal_n = normalize(ndc_edge.normal());
            if (flip) ndc_edge_normal_n = -ndc_edge_normal_n;

            // sample point on edge, find in/out points
            v2<Real> point = intersected._1[edge_idx].interpolate(t);
            var epsilon = Real(1e-5);
            var point_in = point + ndc_edge_normal_n * epsilon;
            var point_out = point - ndc_edge_normal_n * epsilon;
            
            // find t in ws
            var ray_a = _sensor->sample_ray(intersected._1[edge_idx].A * Real(0.5) + Real(0.5)).ray;
            var ray_b = _sensor->sample_ray(intersected._1[edge_idx].B * Real(0.5) + Real(0.5)).ray;
            var ray_p = _sensor->sample_ray(point * Real(0.5) + Real(0.5)).ray;
            experiment::Plane<Real> plane = experiment::Plane<Real>(hp, geo_normal);
            v3<Real> hit_a = plane.intersect(ray_a);
            v3<Real> hit_b = plane.intersect(ray_b);
            v3<Real> hit_p = plane.intersect(ray_p);

            Real orig_a_to_p = length(hit_p - orig_edge.pos[0]);
            Real orig_a_to_b = length(orig_edge.direction());
            Real orig_t_ws = orig_a_to_p / orig_a_to_b;
            v2<Real> ndc_pixel_dims = abs(pixel_aabb_ndc.max - pixel_aabb_ndc.min);
            v2<Real> normalized_edge_length = length(ndc_edge_dir / ndc_pixel_dims);
            EdgeSamplePixel<Real> es;
            es.pos_in_uv = point_in * Real(0.5) + Real(0.5);   // for sensor.sample_ray(..., pos_in_uv)
            es.pos_out_uv = point_out * Real(0.5) + Real(0.5); // for sensor.sample_ray(..., pos_out_uv)
            es.t = orig_t_ws;
            es.barycentric = v3<Real>(Real(0));
            es.barycentric[edge_idx] = Real(1) - es.t;
            es.barycentric[(edge_idx + 1) % 3] = es.t;
            es.attribute = bary_to_attr(es.barycentric);
            es.area = area;
            es.length = normalized_edge_length;
            es.pdf = pmf[edge_idx];
            es.position = hit_p;
            es.normal = ndc_edge_normal_n;
            es.ndc_pixel_dims = ndc_pixel_dims;
            return es;
        }
   
        [Differentiable]
        static v3<Real> slerp_sample(v3<Real> a, v3<Real> b, vector<Real, 1> t) {
            Real cos_theta = dot(a, b);
            if (cos_theta > Real(0.9995)) {
                return normalize(a + t[0] * (b - a));
            }
            Real theta = acos(cos_theta);
            Real sin_theta = sin(theta);
            Real w1 = sin((Real(1) - t[0]) * theta) / sin_theta;
            Real w2 = sin(t[0] * theta) / sin_theta;
            return normalize(a * w1 + b * w2);
        }

        [Differentiable]
        static Real spherical_arc(v3<Real> e0, v3<Real> e1, v3<Real> origin, Real radius = Real(1)) {
            v3<Real> da = e0 - origin;
            v3<Real> db = e1 - origin;
            Real la = length(da);
            Real lb = length(db);
            if (la < Real(1e-8) || lb < Real(1e-8)) return Real(0);
            return radius * acos(clamp(dot(da / la, db / lb), Real(-1), Real(1)));
        }

        public EdgeSampleHemisphere<Real> sample_confined_edge(uint32_t primitive, experiment::Hemisphere<Real> hemisphere, v2<Real> sample, Transform<Real> transform) {
            const uint3 indices = get_indices(primitive);
            // Unit sphere. Arc lengths and the solid angle returned by clip_solid_angle_area() must
            // live in the same measure, otherwise es.length / es.area disagree by a constant factor.
            const Real _r = Real(1);

            let tri = get_triangle(indices, to_world.to_transform());
            var geo_normal = normalize(cross(tri.pos[1] - tri.pos[0], tri.pos[2] - tri.pos[0]));
            bool flip = dot(geo_normal, normalize(tri.pos[0] - hemisphere.center)) < Real(0);
            if (flip) {
                // geo_normal = -geo_normal;
            }
            var clipped_edge = hemisphere.clip_edges(tri.pos);
            v3<Real> arcs;
            for (uint i = 0; i < 3; i++) arcs[i] = spherical_arc(clipped_edge._1[i].A, clipped_edge._1[i].B, hemisphere.center, _r);
            const Real clipped_spherical_area = hemisphere.clip_solid_angle_area(tri.pos);
            // if (clipped_spherical_area <= 0.0f) {
            //     printf("clipped_spherical_area <= 0.0f\n");
            // }
            // if (isnan(clipped_spherical_area)) {
            //     printf("clipped_spherical_area is nan\n");
            // }
            if ((clipped_edge._0[0] + clipped_edge._0[1] + clipped_edge._0[2]) == 0) return {}; // should never happen
            
            // noclip
            // const float3 arcs = spherical_arcs(tri, hemisphere.center, _r);
            // const float clipped_spherical_area = experiment::spherical_triangle_area(
            //     normalize(tri.pos[0] - hemisphere.center),
            //     normalize(tri.pos[1] - hemisphere.center),
            //     normalize(tri.pos[2] - hemisphere.center)
            // );

            v3<Real> lengths_ws;
            for (int i = 0; i < 3; i++) {
                lengths_ws[i] = length(tri.edge(i).direction());
            }

            // edge sampler preparation
            int count = 0;
            Real pmf[3], cdf[4];
            {
                cdf[0] = Real(0);
                for (int i = 0; i < 3; i++) {
                    if (clipped_edge._0[i]) {
                        pmf[count] = arcs[i];
                        cdf[++count] = cdf[count - 1] + pmf[count - 1];
                    } else {
                        pmf[count] = Real(0);
                        cdf[++count] = cdf[count - 1];
                    }
                }
                Real length_sum = cdf[count];
                if (length_sum < Real(1e-8)) return {};
                for (int i = 0; i < count;) {
                    pmf[i] /= length_sum;
                    cdf[++i] /= length_sum;
                }
            }
            // printf("pmf: %f, %f, %f cdf: %f, %f, %f, %f\n", pmf[0], pmf[1], pmf[2], cdf[0], cdf[1], cdf[2], cdf[3]);

            // select one point on edges contained in hemisphere
            int edge_idx = 0;
            Real t = Real(0);
            for (int i = 0; i < count; i++) {
                if (sample.x < cdf[i + 1]) {
                    edge_idx = i;
                    t = (sample.x - cdf[i]) / (cdf[i + 1] - cdf[i]);
                    break;
                }
            }
            if (isnan(pmf[edge_idx])) return {};
            // edge_idx = DiscreteDistribution::sample_evenly_spaced_range(0, count-1, sample.x);
            
            var c_edge_ws = clipped_edge._1[edge_idx];
            var edge_ws = tri.edge(edge_idx);
            
            EdgeSampleHemisphere<Real> es = EdgeSampleHemisphere<Real>();
            {
                var ldir = c_edge_ws.B - c_edge_ws.A;
                var llength = length(ldir);
                v3<Real> da = c_edge_ws.A - hemisphere.center;
                v3<Real> db = c_edge_ws.B - hemisphere.center;
                Real la = length(da);
                Real lb = length(db);
                if (la < Real(1e-8) || lb < Real(1e-8)) return {};
                v3<Real> a = da / la;
                v3<Real> b = db / lb;
                Real theta = acos(clamp(dot(a, b), Real(-1), Real(1)));

                bool use_psa = llength > Real(1e-8) && theta > Real(1e-8);
                if (use_psa) {
                    ldir = ldir / llength;
                    var llights = linear_lights::prepare_projected_solid_angle_line_sampling_without_clipping(
                        hemisphere.normal, c_edge_ws.A - hemisphere.center, ldir, llength);
                    use_psa = !linear_lights::is_projected_solid_angle_line_empty(llights);
                    if (use_psa) {
                        var sdir = linear_lights::sample_projected_solid_angle_line(llights, sample.y);

                        t = clamp(acos(clamp(dot(sdir, a), Real(-1), Real(1))) / theta, Real(0), Real(1));

                        // Jacobian from Peters' omega_s to slerp parameter t:
                        //   p_{omega_s} = 2*<n,omega>/E  (paper Eq. 3 derivative)
                        //   d(omega_s)/dt = omega_t * theta
                        //   p_t = p_{omega_s} * omega_t * theta = 2*<n,omega>*omega_t*theta / E
                        Real omega_s = clamp(dot(sdir, ldir), Real(-1), Real(1));
                        Real omega_t = sqrt(max(Real(0), Real(1) - omega_s * omega_s));
                        Real cos_n = max(dot(hemisphere.normal, sdir), Real(0));
                        if (cos_n < Real(1e-8)) return {};
                        Real E = max(abs(llights.extent_cdf), Real(1e-8));
                        es.pdf = pmf[edge_idx] * Real(2) * cos_n * omega_t * theta / E;
                    }
                }
                if (!use_psa) {
                    es.pdf = pmf[edge_idx]; // t is uniform in [0, 1], so its density is 1
                }
                if (!(es.pdf > Real(0))) return {}; // degenerate or NaN
            }

            let pair_cart = __fwd_diff(slerp_sample)(
                diffPair<v3<Real>>(normalize(c_edge_ws.A - hemisphere.center), v3<Real>(Real(0))),
                diffPair<v3<Real>>(normalize(c_edge_ws.B - hemisphere.center), v3<Real>(Real(0))),
                diffPair<vector<Real, 1>>(t, vector<Real, 1>(1))
            );
            es.position = experiment::Plane<Real>(tri.pos[0], geo_normal).intersect(Ray<Real>(hemisphere.center, Real(0.01), normalize(pair_cart.p), Real(10000.0)));
            es.t = length(es.position - edge_ws.pos[0]) / length(edge_ws.pos[1] - edge_ws.pos[0]);
            es.normal = normalize(cross(normalize(pair_cart.p), normalize(pair_cart.d) /*tangent*/));
            if (flip) es.normal = -es.normal;

            es.barycentric = v3<Real>(Real(0));
            es.barycentric[edge_idx] = Real(1) - es.t;
            es.barycentric[(edge_idx + 1) % 3] = es.t;
            es.attribute = bary_to_attr(es.barycentric);

            // Offset the direction, not the position: es.normal is a unit-sphere tangent, so offsetting
            // the world position gives an angular offset of epsilon/distance, which collapses to zero
            // for distant edges and silently kills the gradient.
            var epsilon = Real(1e-4);
            let w = normalize(es.position - hemisphere.center);
            es.dir_in = normalize(w + es.normal * epsilon);
            es.dir_out = normalize(w - es.normal * epsilon);
            es.area = clipped_spherical_area;
            es.length = arcs[edge_idx];
            return es;     
        }

        [Differentiable] public void store(vector<Real, 4> value, Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
            if (film) {
                let indices = get_indices(primitive);
                let bary = attr_to_bary(attribute);
                let areas = v3<Real>(barycentric_areas[{indices.x}], barycentric_areas[{indices.y}], barycentric_areas[{indices.z}]);
                film.interpolate_add<3>(value,
                    { { indices.x, 0, 0, 0 }, { indices.y, 0, 0, 0 } , { indices.z, 0, 0, 0 } },
                    { { bary.x / areas.x }, { bary.y / areas.y } , { bary.z / areas.z } },
                    _surface_interaction(ray, primitive, attribute, transform)
                );
            }
        }

        [Differentiable] public vector<Real, 4> load(Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
            if (film) {
                let indices = get_indices(primitive);
                let bary = attr_to_bary(attribute);
                return film.interpolate_read<3>(
                    { { indices.x, 0, 0, 0 }, { indices.y, 0, 0, 0 } , { indices.z, 0, 0, 0 } },
                    { { bary.x }, { bary.y }, { bary.z } },
                    // attached: the direction the cache is evaluated in rides with the geometry
                    _surface_interaction(ray, primitive, detach(attribute), transform)
                );
            }
            return vector<Real, 4>(Real(0), Real(0), Real(0), Real(0));
        }

        public uint intersect(inout Ray<Real> ray, const uint cullmode,
                           inout v2<Real> barycentric, inout uint primitive_idx, inout uint geometry_idx) {
            bool hit = false;
            var _ray_os = ray.transform(to_world.to_transform().inverse);
            for (uint i = 0; i < triangle_count(); i++) {
                if (get_triangle(get_indices(i)).intersect(_ray_os, CullMode(cullmode), barycentric)) {
                    primitive_idx = i;
                    hit = true;
                };
            }
            ray.tmax = _ray_os.tmax;
            return hit;
        }
    };
}
