implementing tamashii;

namespace rfilters {
    public struct Box<Real : floating_point> : IRFilter<Real> {
        public Real radius;

        public __init(Real radius = Real(0.5)) {
            this.radius = radius;
        }

        [Differentiable]
        public FilterSample<Real> sample(v2<Real> uv) {
            return { (uv * Real(2) - Real(1)) * radius, Real(1) };
        }

        [Differentiable]
        public Real eval(Real x) {
            return select(abs(x) <= radius, Real(0.5) / radius, Real(0));
        }
    };
};