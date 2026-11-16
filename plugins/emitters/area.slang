implementing tamashii;

namespace emitters {
    public struct Area<Real : atomicable_floating_point> : IEmitter<Real>, IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public IShape<Real>* shape;

        public __init() {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.shape = nullptr;
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, Real3 uv) {
            return shape->sample_direction(si, uv);
        }

        [Differentiable] public RaySample<Real> sample_ray(Real3 uv_pos, Real2 uv_dir) {
            SurfaceSample<Real> ss = shape->sample_surface(uv_pos);
            Real3 wo = warp::square_to_cosine_hemisphere<Real>(uv_dir);
            let weight = shape->emission * constants<Real>::pi / ss.pdf;
            return { Ray<Real>(ss.sp.position, ss.sp.frame.to_world(wo)), weight };
        }

        [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds) {
            return shape->pdf_direction(si, ds);
        }

        [Differentiable] public Real3 evaluate_direction(SurfaceInteraction<Real> si, Real3 uv) {
            let ds = shape->sample_direction(si, uv);
            return ds.sp.emission * abs(dot(ds.direction, si.sp.frame.normal())) / ds.pdf;
        }

        [Differentiable] public Real3 evaluate_hit(Ray<Real> ray, HitInfo hi) {
            return shape->surface_interaction(ray, hi.primitive, Real2(hi.attribute)).sp.emission;
        }

        public uint intersect(inout Ray<Real> ray, const uint cullmode,
                        inout Real2 barycentric, inout uint primitive_idx, inout uint geometry_idx) {
            return shape->intersect(ray, cullmode, barycentric, primitive_idx, geometry_idx);
        }
    };
}