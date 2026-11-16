implementing tamashii;

namespace emitters {
    public struct Point<Real : atomicable_floating_point> : IEmitter<Real>, IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public Texture<Real> profile; // equal to watt/sr or watt/4pi
        public Real3 color;

        public __init() {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.color = Real3(Real(1.0));
            this.profile = Texture<Real>(Real(1.0));
        }

        [Differentiable] public Real2 uv_from_direction(Real3 dir) {
            Real3 local_dir = normalize(mul(Real4(dir, Real(0.0)), to_world.to_transform().inverse).xyz);
            Real u = atan2(local_dir.z, local_dir.x) * constants<Real>::inv_two_pi + Real(0.5);
            Real v = -asin(clamp(local_dir.y, Real(-1.0), Real(1.0))) * constants<Real>::inv_pi + Real(0.5);
            return Real2(u, v);
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, Real3 uv) {
            DirectionSample<Real> ds;
            ds.sp.position = to_world.to_transform().forward[3].xyz;
            ds.direction = ds.sp.position - si.sp.position.xyz;
            Real distance2 = dot(ds.direction, ds.direction);
            ds.distance = sqrt(distance2);
            ds.direction /= ds.distance;
            ds.sp.frame = Frame<Real>(-ds.direction);
            ds.sp.geometric_normal = Real3(Real(0), Real(0), Real(1));
            ds.sp.area = Real(0);
            ds.pdf = Real(1);
            ds.delta = true;
            ds.sp.emission = color * profile.sample(uv_from_direction(-ds.direction)) / distance2;
            return ds;
        }
        
        [Differentiable] public Real3 sample_position() {
            return to_world.to_transform().forward[3].xyz;
        }

        [Differentiable] public RaySample<Real> sample_ray(Real3 uv_pos, Real2 uv_dir) {
            Real3 direction = warp::square_to_uniform_sphere<Real>(uv_dir);
            let weight = color * profile.sample(uv_from_direction(direction)) * constants<Real>::four_pi;
            return { Ray<Real>(sample_position(), direction), weight };
        }

        [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds) {
            return Real(0.0);
        }

        [Differentiable] public Real3 evaluate_direction(SurfaceInteraction<Real> si, Real3 uv) {
            let ds = sample_direction(si, uv);
            return ds.sp.emission * abs(dot(ds.direction, si.sp.frame.normal())) / ds.pdf;
        }

        [Differentiable] public Real3 evaluate_hit(Ray<Real> ray, HitInfo hi) {
            return Real3(Real(0.0));
        }

        public uint intersect(inout Ray<Real> ray, const uint cullmode,
                        inout Real2 barycentric, inout uint primitive_idx, inout uint geometry_idx) {
            return 0;
        }
    };
}