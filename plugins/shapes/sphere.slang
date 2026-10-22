implementing tamashii;

namespace shapes {
    public struct Sphere<Real : atomicable_floating_point> : IShape<Real>, IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public AABB<Real> aabb;
        public IBsdf<Real> *bsdf;
        public IFilm<Real, 4>* film;
        public v3<Real> emission;

        __init(Real radius = Real(0.5)) {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.aabb = AABB<Real>(v3<Real>(Real(-0.5)), v3<Real>(Real(0.5)));
            this.bsdf = nullptr;
            this.film = nullptr;
            this.emission = v3<Real>(Real(0));
        }

        public uint primitive_count() { return 1; }

        [Differentiable] public SurfacePoint<Real> surface_point(uint32_t primitive, v2<Real> attribute, Transform<Real> transform) {
            var _transform = mul(to_world.to_transform().forward, transform.forward);
            let _radius = _transform[0][0] * Real(0.5);
            SurfacePoint<Real> sp;
            let ssample = warp::square_to_uniform_sphere(attribute);
            sp.position = _transform[3].xyz + ssample * _radius;
            let normal = normalize(ssample);
            sp.frame = Frame<Real>(normal);
            sp.geometric_normal = v3<Real>(Real(0), Real(0), Real(1));
            sp.uv = attribute;
            sp.attribute = attribute;
            sp.emission = emission;
            sp.area = constants<Real>::four_pi * _radius * _radius;
            sp.primitive = 0;
            return sp;
        }

        [Differentiable] public SurfaceInteraction<Real> _surface_interaction(Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
            SurfaceInteraction<Real> si;
            si.sp = surface_point(primitive, attribute, transform);
            let geo_normal_world = si.sp.frame.to_world(si.sp.geometric_normal);
            if (dot(ray.direction, geo_normal_world) > Real(0)) {
                si.sp.frame = Frame<Real>(-si.sp.frame.normal());
                si.sp.geometric_normal = si.sp.frame.to_local(geo_normal_world);
            }
            si.wi = si.sp.frame.to_local(-ray.direction);
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

            v3<Real> center = v3<Real>(Real(0), Real(0), Real(0));
            Real radius = Real(0.5);
            const v3<Real> oc = _ray_os.origin - center;
            const Real a = dot(_ray_os.direction, _ray_os.direction);
            const Real b = dot(oc, _ray_os.direction);
            const Real c = dot(oc, oc) - radius * radius;
            const Real discriminant = b * b - a * c;

            const Real t1 = (-b - sqrt(discriminant)) / a;
            const Real t2 = (-b + sqrt(discriminant)) / a;
            Real t = select(ray.tmin <= t1, t1, t2);

            v2<Real> attribute = warp::uniform_sphere_to_square(normalize(_ray_os.origin + _ray_os.direction * t));
            if (attach_to_geometry) attribute = detach(attribute);
            return _surface_interaction(ray, primitive, attribute, transform);
        }

        [Differentiable] public SurfaceSample<Real> sample_surface(v3<Real> uv, Transform<Real> transform) {
            SurfaceSample<Real> ss;
            ss.sp = surface_point(0, uv.xy, transform);
            ss.delta = false;
            ss.pdf = Real(1) / ss.sp.area; // TODO wrong pdf for non-uniform scale
            return ss;
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, v3<Real> uv, Transform<Real> transform) {
            var _transform = mul(to_world.to_transform().forward, transform.forward);
            let _radius = _transform[0][0] * Real(0.5);
            // let _center = _transform[3].xyz; // don't work idk matmul is buggy
            let _center = to_world.to_transform().forward[3].xyz + transform.forward[3].xyz;

            var L = _center - si.sp.position;
            Real d = length(L);
            Real C = d * d - _radius * _radius;

            L = normalize(L);
            Real cosTheta = sqrt(C) / d;
            DirectionSample<Real> ds;
            ds.direction = warp::square_to_uniform_cone(uv.xy, cosTheta);

            Real B = d * ds.direction.z;
            Real det = safe_sqrt(B * B - C);
            ds.distance = B - det;

            var frame = Frame<Real>(L);
            ds.direction = frame.to_world(ds.direction);
            ds.sp.position = si.sp.position + ds.direction * ds.distance;
            let sp_normal = normalize(ds.sp.position - _center);
            ds.sp.frame = Frame<Real>(sp_normal);
            ds.sp.geometric_normal = v3<Real>(Real(0), Real(0), Real(1));
            ds.sp.area = constants<Real>::four_pi * _radius * _radius;
            ds.pdf = warp::square_to_uniform_cone_pdf<false>(ds.direction, cosTheta);
            ds.delta = false;
            ds.sp.emission = emission;
            return ds;
        }

        [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds) {
            var _transform = mul(to_world.to_transform().forward, Transform<Real>::identity().forward);
            let _radius = _transform[0][0] * Real(0.5);
            let _center = to_world.to_transform().forward[3].xyz;
            Real d = length(_center - si.sp.position);
            Real cosTheta = safe_sqrt(d * d - _radius * _radius) / d;
            return warp::square_to_uniform_cone_pdf<false>(ds.direction, cosTheta);
        }

        public Real pdf_surface(Transform<Real> transform) { 
            return Real(1) / (constants<Real>::four_pi * pow(to_world.to_transform().forward[0][0] * transform.forward[0][0], Real(2))); 
        }

        public static Tuple<bool /*hit*/, Real /*t*/, v2<Real>/*attr*/> intersect_os(
            v3<Real> origin, v3<Real> direction, no_diff Real t_min
        ) {
            // https://en.wikipedia.org/wiki/Quadratic_formula
            v3<Real> center = v3<Real>(Real(0), Real(0), Real(0));
            Real radius = Real(0.5);
            const v3<Real> oc = origin - center;
            const Real a = dot(direction, direction);
            const Real b = dot(oc, direction);
            const Real c = dot(oc, oc) - radius * radius;
            const Real discriminant = b * b - a * c;

            if (discriminant >= Real(0))
            {
                const Real t1 = (-b - sqrt(discriminant)) / a;
                const Real t2 = (-b + sqrt(discriminant)) / a;

                if (t_min <= t1 || t_min <= t2)
                {
                    Real t = (t_min <= t1) ? t1 : t2;
                    v3<Real> hp = origin + direction * t;
                    return Tuple<bool, Real, v2<Real>>( true, t, warp::uniform_sphere_to_square(normalize(hp)) );
                }
            }
            return Tuple<bool, Real, v2<Real>>( false, Real(0), v2<Real>(Real(0), Real(0)) );
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
            Tuple<bool, Real, v2<Real>> result = intersect_os(_ray_os.origin, _ray_os.direction, ray.tmin);
            if (result._0 && result._1 < ray.tmax) {
                ray.tmax = result._1;
                barycentric = result._2;
                return true;
            }
            return false;
        }
    };
}
