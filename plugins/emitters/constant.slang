implementing tamashii;

namespace emitters {
    public struct Constant<Real : atomicable_floating_point> : IEmitter<Real>, IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public Texture<v3<Real>> radiance;

        public __init() {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.radiance = Texture<v3<Real>>(v3<Real>(Real(1.0)));
        }

        [Differentiable] public v2<Real> uv_from_direction(v3<Real> dir) {
            v3<Real> local_dir = normalize(mul(v4<Real>(dir, Real(0.0)), to_world.to_transform().inverse).xyz);
            Real u = atan2(local_dir.z, local_dir.x) * constants<Real>::inv_two_pi + Real(0.5);
            Real v = -asin(clamp(local_dir.y, Real(-1.0), Real(1.0))) * constants<Real>::inv_pi + Real(0.5);
            return v2<Real>(u, v);
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, v3<Real> uv) {
            DirectionSample<Real> ds;
            ds.sp.position = v3<Real>(0);
            ds.direction = si.sp.frame.to_world(warp::square_to_uniform_hemisphere<Real>(uv.xy));
            ds.distance = Real(10000.0);
            ds.sp.frame = Frame<Real>(-ds.direction);
            ds.sp.geometric_normal = v3<Real>(Real(0.0), Real(0.0), Real(1.0));
            ds.sp.area = Real(0);
            ds.pdf = warp::square_to_uniform_hemisphere_pdf<Real>();
            ds.delta = false;
            ds.sp.emission = radiance.sample(uv_from_direction(ds.direction));
            return ds;
        }

        [Differentiable] public RaySample<Real> sample_ray(v3<Real> uv_pos, v2<Real> uv_dir) {
            v3<Real> direction = warp::square_to_uniform_sphere<Real>(uv_dir);
            let weight = radiance.sample(uv_from_direction(direction)) * constants<Real>::four_pi;
            return { Ray<Real>(warp::square_to_uniform_hemisphere<Real>(uv_pos.xy) * Real(10000.0), direction), weight };
        }

        [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds) {
            return warp::square_to_uniform_hemisphere_pdf<Real>();
        }

        [Differentiable] public v3<Real> evaluate_direction(SurfaceInteraction<Real> si, v3<Real> uv) {
            let ds = sample_direction(si, uv);
            return ds.sp.emission * abs(dot(ds.direction, si.sp.frame.normal())) / ds.pdf;
        }

        [Differentiable] public v3<Real> evaluate_hit(Ray<Real> ray, HitInfo hi) {
            return radiance.sample(uv_from_direction(ray.direction));
        }

        public uint intersect(inout Ray<Real> ray, const uint cullmode,
                        inout v2<Real> barycentric, inout uint primitive_idx, inout uint geometry_idx) {
            return 0;
        }
    };
}
