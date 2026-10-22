implementing tamashii;

namespace sensors {
    public struct Area<Real : atomicable_floating_point> : ISensor<Real> , IDifferentiable {
        public IShape<Real>* shape; // must have film

        typealias Real2 = vector<Real, 2>;
        typealias Real3 = vector<Real, 3>;
        typealias Real4 = vector<Real, 4>;

        public __init() {
            shape = nullptr;
        }

        [Differentiable]
        public Real4 position() {
            return shape.to_world.to_transform().forward[3];
        };


        [Differentiable]
        public Real4 splat(vector<Real, 4> point) {
            return {};
        }

        [Differentiable]
        public RaySample<Real> sample_ray(Real2 uv)
        {
            return {};
        }

        [Differentiable]
        public RaySample<Real> sample_ray(Real2 uv, vector<int, 4> coords)
        {
            return {};
        }

        [Differentiable]
        public LensSample<Real> sample_lens(SurfaceInteraction<Real> si, v3<Real> uv) {
            let ssample = shape.sample_surface(uv); 
            LensSample<Real> ls;
            ls.direction = ssample.sp.position - si.sp.position;
            if (dot(ls.direction, si.sp.frame.normal()) >= Real(0)) return {}; // backfacing
            ls.distance = length(ls.direction);
            ls.direction /= ls.distance;
            ls.weight = Real(1) / ssample.pdf;
            ls.valid = true;
            ls.coords = Coord(ssample.sp.primitive);
            ls.uv = ssample.sp.uv;
            return ls;
        }

        public IFilm<Real, 4> film() {
            return *shape.film;
        }
    };
}
