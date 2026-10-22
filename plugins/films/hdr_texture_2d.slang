implementing tamashii;

namespace films {
    public struct HDRTexture2D<Real : atomicable_floating_point, let N : int> : IFilm<Real, N> {
        public no_diff DiffTensor<vector<Real, N>, 2> texture;

        public Real aspect() {
            return Real(texture.shape[1]) / Real(texture.shape[0]);
        }
        public Real inv_aspect() {
            return Real(texture.shape[0]) / Real(texture.shape[1]);
        }

        public int dimensions() {
            return 2;
        }

        public vector<int, 4> shape() {
            return vector<int, 4>(int(texture.shape[1]), int(texture.shape[0]), 0, 0);
        }

        [Differentiable]
        public void add(vector<Real, N> value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {}) {
            texture.add(coords.xy, value);
        }

        [Differentiable]
        public void atomic_add(vector<Real, N> value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {}) {
            texture.atomic_add(coords.xy, value);
        }

        [Differentiable]
        public vector<Real, N> read(vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si) {
            return no_diff texture[coords.xy];
        }

        [Differentiable, mutating] public void interpolate_add<let M : int>(vector<Real, N> value, Array<vector<int, 4>, M> coords, Array<vector<Real, N>, M> weights, Optional<SurfaceInteraction<Real>> si = {}) {

        }

        [Differentiable] public vector<Real, N> interpolate_read<let M : int>(Array<vector<int, 4>, M> coords, Array<vector<Real, N>, M> weights, Optional<SurfaceInteraction<Real>> si = {}) {
            return {};
        }

    };
}