implementing tamashii;

namespace films {
    // Get world-space incoming direction from surface interaction
    [Differentiable]
    v3<Real> sh_direction<Real : atomicable_floating_point>(Optional<SurfaceInteraction<Real>> si) {
        if (let s = si) {
            return s.sp.frame.to_world(s.wi);
        }
        return v3<Real>(Real(0), Real(0), Real(1));
    }

    public struct SphericalHarmonics<Real : atomicable_floating_point, let Order : int, let N : int, bool HSH> : IFilm<Real, N> {
        public no_diff SphericalHarmonicsArray<Real, Order, N, HSH> sh;

        public int dimensions() {
            return 1;
        }

        public vector<int, 4> shape() {
            return vector<int, 4>(int(sh.coefficients.shape[0]), 0, 0, 0);
        }

        // Project value onto SH basis and accumulate
        // coords.x = cache point index, si provides the direction
        [Differentiable]
        public void add(vector<Real, N> value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {}) {
            sh.project(coords.x, sh_direction(si), value);
        }

        [Differentiable] public void atomic_add(vector<Real, N> value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {}) {
            add(value, coords, si);
        }

        // Evaluate SH at direction from si to reconstruct radiance
        [Differentiable]
        public vector<Real, N> read(vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si) {
            return sh.evaluate(coords.x, sh_direction(si));
        }

        [Differentiable, mutating]
        public void interpolate_add<let M : int>(vector<Real, N> value, Array<vector<int, 4>, M> coords, Array<vector<Real, N>, M> weights, Optional<SurfaceInteraction<Real>> si = {}) {
            [ForceUnroll]
            for (int i = 0; i < M; i++) {
                add(value * weights[i], coords[i], si);
            }
        }

        [Differentiable]
        public vector<Real, N> interpolate_read<let M : int>(Array<vector<int, 4>, M> coords, Array<vector<Real, N>, M> weights, Optional<SurfaceInteraction<Real>> si = {}) {
            vector<Real, N> result = vector<Real, N>(0);
            [ForceUnroll]
            for (int i = 0; i < M; i++) {
                result += read(coords[i], si) * weights[i];
            }
            return result;
        }
    };
}
