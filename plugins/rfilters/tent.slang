implementing tamashii;

namespace rfilters {
    public struct Tent<Real : floating_point> : IRFilter<Real> {
        public Real radius;

        public __init(Real radius = Real(1)) {
            this.radius = radius;
        }

        // Inverse cdf of the 1D tent on [-radius, radius].
        Real sample_offset(Real u) {
            u = clamp(u, Real(1e-7), Real(1) - Real(1e-7));
            if (u < Real(0.5)) return radius * (sqrt(Real(2) * u) - Real(1));
            return radius * (Real(1) - sqrt(Real(2) * (Real(1) - u)));
        }

        [Differentiable]
        public FilterSample<Real> sample(v2<Real> uv) {
            let offset = v2<Real>(
                no_diff sample_offset(uv.x), no_diff sample_offset(uv.y));
            return { offset, Real(1) };
        }

        [Differentiable]
        public Real eval(Real x) {
            return max(Real(0), Real(1) - abs(x) / radius) / radius;
        }
    };
};
