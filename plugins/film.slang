implementing tamashii;

[anyValueSize(settings::film_interface_size)]
public dyn interface IFilm<Real : atomicable_floating_point, let N : int> {
    // number of dimensions of the film, rest of indices in coords are ignored
    public int dimensions();
    // number of entries in each dimension of the film
    public vector<int, 4> shape();
    // add/read value at specific coordinates, where coords is used to determine where the value is added on the film
    [Differentiable] public void add(RealN value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {});
    [Differentiable] public RealN read(vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si);
    [Differentiable] public void atomic_add(RealN value, vector<int, 4> coords, Optional<SurfaceInteraction<Real>> si = {});

    // add/read value via distributed interpolation on M entries, where coords and weights are used to determine how the value is distributed/collected across the film
    [Differentiable, mutating] public void interpolate_add<let M : int>(RealN value, Array<vector<int, 4>, M> coords, Array<RealN, M> weights, Optional<SurfaceInteraction<Real>> si = {});
    [Differentiable] public RealN interpolate_read<let M : int>(Array<vector<int, 4>, M> coords, Array<RealN, M> weights, Optional<SurfaceInteraction<Real>> si = {});
};

extension<Real, let N : int, T> T
where Real : __BuiltinFloatingPointType
where Real : IArithmeticAtomicable
where T : IFilm<Real, N>
{
    typealias Real2 = vector<Real, 2>;
    typealias Real3 = vector<Real, 3>;
    typealias Real4 = vector<Real, 4>;
    typealias RealN = vector<Real, N>;
}

namespace films {
    [Differentiable]
    public void add<Real : atomicable_floating_point, let N : int>(IFilm<Real, N> film, vector<Real, N> value, int4 location)
    {
        film.add(value, location);
    }
    [Differentiable]
    public void atomic_add<Real : atomicable_floating_point, let N : int>(IFilm<Real, N> film, vector<Real, N> value, int4 location)
    {
        film.atomic_add(value, location);
    }
}