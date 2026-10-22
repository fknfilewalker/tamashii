implementing tamashii;

namespace shapes {
    public struct Rectangle<Real : atomicable_floating_point> : IShape<Real>, IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public AABB<Real> aabb;
        public IBsdf<Real>* bsdf;
        public IFilm<Real, 4>* film;
        public v3<Real> emission;

        __init(v2<Real> scale = v2<Real>(Real(1))) {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.aabb = AABB<Real>(v3<Real>(Real(-0.5), Real(0), Real(-0.5)), v3<Real>(Real(0.5), Real(0), Real(0.5)));
            this.bsdf = nullptr;
            this.film = nullptr;
            this.emission = v3<Real>(Real(0));
        }

        public uint primitive_count() { return 2; }

        [Differentiable]
        public Real area(matrix<Real, 4, 4> transform = diagonal(v4<Real>(Real(1)))) {
            let scale = extract_scale(mul(to_world.to_transform().forward, transform));
            return scale.x * scale.z * Real(0.5);
        }

        [Differentiable]
        public SurfacePoint<Real> surface_point(uint32_t primitive, v2<Real> attribute, Transform<Real> transform) {
            SurfacePoint<Real> sp;
            v2<Real> offset = attribute - Real(0.5);
            if (primitive == 1) offset = v2<Real>(-offset.x, -offset.y);
            sp.position = mul(v4<Real>(offset.y, Real(0), offset.x, Real(1)), mul(to_world.to_transform().forward, transform.forward)).xyz;
            let normal = normalize(mul(v3<Real>(Real(0), Real(1), Real(0)), transpose(m3x3<Real>(mul(to_world.to_transform().inverse, transform.inverse)))));
            sp.frame = Frame<Real>(normal);
            sp.geometric_normal = v3<Real>(Real(0), Real(0), Real(1));
            sp.area = area(transform.forward);
            sp.uv = attribute;
            sp.attribute = attribute;
            sp.emission = emission;
            sp.primitive = 0;
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

        [Differentiable] public SurfaceInteraction<Real> surface_intersection(Ray<Real> ray, uint32_t primitive, bool attach_to_geometry, Transform<Real> transform = Transform<Real>::identity()) {
            let _ray_os = ray.transform(to_world.to_transform().inverse);

            v3<Real> pos = v3<Real>(Real(0), Real(0), Real(0));
            v3<Real> tangent = v3<Real>(Real(1), Real(0), Real(0));
            v3<Real> bitangent = v3<Real>(Real(0), Real(0), Real(1));
            v3<Real> normal = v3<Real>(Real(0), Real(-1), Real(0));

            Real t = dot(pos - _ray_os.origin, normal) / dot(_ray_os.direction, normal);
            v3<Real> hit_pos = _ray_os.origin + _ray_os.direction * t;

            Real x = dot(hit_pos - pos, tangent);
            Real y = dot(hit_pos - pos, bitangent);
            v2<Real> attribute = v2<Real>(x + Real(0.5), y + Real(0.5));
            if (primitive == 1) attribute = v2<Real>(Real(1) - attribute.x, Real(1) - attribute.y);
            if (attach_to_geometry) attribute = detach(attribute);
            return _surface_interaction(ray, primitive, attribute, transform);
        }

        [Differentiable] public SurfaceSample<Real> sample_surface(v3<Real> uv, Transform<Real> transform) {
            SurfaceSample<Real> ss;
            ss.sp = surface_point(0, uv.xy, transform);
            ss.delta = false;
            ss.pdf = Real(1) / ss.sp.area;
            return ss;
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, v3<Real> uv, Transform<Real> transform) {
            DirectionSample<Real> ds;
            ds.sp = surface_point(0, uv.xy, transform);
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

        [Differentiable] public void store(vector<Real, 4> value, Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
            if (film) {
            }
        }

        [Differentiable] public vector<Real, 4> load(Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
            return vector<Real, 4>(Real(0), Real(0), Real(0), Real(0));
        }

        public uint intersect(inout Ray<Real> ray, const uint cullmode,
            inout v2<Real> barycentric, inout uint primitive_idx, inout uint geometry_idx) {
            let _ray_os = ray.transform(to_world.to_transform().inverse);

            // float scale_x = to_world.to_transform().forward[0][0];
            // float scale_y = to_world.to_transform().forward[1][1];
            // float scale_z = to_world.to_transform().forward[2][2];

            v3<Real> pos = v3<Real>(Real(0), Real(0), Real(0));
            v3<Real> tangent = v3<Real>(Real(1), Real(0), Real(0));//column(to_world.to_transform().forward, 0).xyz;
            v3<Real> bitangent = v3<Real>(Real(0), Real(0), Real(1));//column(to_world.to_transform().forward, 2).xyz;
            v3<Real> normal = v3<Real>(Real(0), Real(-1), Real(0)); //-column(to_world.to_transform().forward, 1).xyz;//cross(tangent, bitangent);

            Real t = dot(pos - _ray_os.origin, normal) / dot(_ray_os.direction, normal);
            if (t < ray.tmin || t > ray.tmax) return false;
            v3<Real> hit_pos = _ray_os.origin + _ray_os.direction * t;
            if (t > Real(0)) {
                Real x = dot(hit_pos - pos, tangent);
                Real y = dot(hit_pos - pos, bitangent);
                if (x >= Real(-0.5) && x <= Real(0.5) && y >= Real(-0.5) && y <= Real(0.5)) {
                    barycentric = v2<Real>(x + Real(0.5), y + Real(0.5));
                    primitive_idx = geometry_idx = 0;
                    ray.tmax = t;
                    return true;
                }
            }

            return false;
        }
    };
}
