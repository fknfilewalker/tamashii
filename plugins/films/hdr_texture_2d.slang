implementing tamashii;

namespace films {
    public struct HDRTexture2D<Real : atomicable_floating_point, let N : int> : IFilm<Real, N> {
        public no_diff DiffTensor<RealN, 2> texture;

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
        public void add(RealN value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {}) {
            texture.add(coords.xy, value);
        }

        [Differentiable]
        public void atomic_add(RealN value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {}) {
            texture.atomic_add(coords.xy, value);
        }

        [Differentiable]
        public RealN read(vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si) {
            return no_diff texture[coords.xy];
        }

        [Differentiable, mutating] public void interpolate_add<let M : int>(RealN value, Array<vector<int, 4>, M> coords, Array<RealN, M> weights, Optional<SurfaceInteraction<Real>> si = {}) {

        }

        [Differentiable] public RealN interpolate_read<let M : int>(Array<vector<int, 4>, M> coords, Array<RealN, M> weights, Optional<SurfaceInteraction<Real>> si = {}) {
            return {};
        }

    };
}