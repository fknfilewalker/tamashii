implementing tamashii;

[anyValueSize(settings::emitter_interface_size)]
public dyn interface IEmitter<Real : atomicable_floating_point> {
    public property DiffTensor<m4x4<Real>, 1> to_world;

    /* Returns a direction sample to the emitter from the given surface point */
    [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, Real3 uv);
    /* Returns a ray sample from the emitter */
    [Differentiable] public RaySample<Real> sample_ray(Real3 uv_pos, Real2 uv_dir);
    /* Returns the solid angle pdf for sampling this direction (for MIS) */
    [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds);

    /* Returns the throughput from si to the point `uv` on the emitter. */
    [Differentiable] public Real3 evaluate_direction(SurfaceInteraction<Real> si, Real3 uv);

    /* Returns the radiance in case this emitter was hit by a ray */
    [Differentiable] public Real3 evaluate_hit(Ray<Real> ray, HitInfo hi);

    public uint intersect(inout Ray<Real> ray, const uint cullmode,
                              inout Real2 barycentric, inout uint primitive_idx, inout uint geometry_idx);
};

extension<Real, T> T
where Real : __BuiltinFloatingPointType
where Real : IArithmeticAtomicable
where T : IEmitter<Real>
{
    typealias Real2 = vector<Real, 2>;
    typealias Real3 = vector<Real, 3>;
    typealias Real4 = vector<Real, 4>;
}

namespace emitters {
    [Differentiable]
    public v3<Real> evaluate_direction<Real : atomicable_floating_point>(IEmitter<Real> emitter, no_diff SurfaceInteraction<Real> si, no_diff v3<Real> uv) {
        return emitter.evaluate_direction(si, uv);
    }

    // solution to a slang bug
    /* Backward casts to the concrete type first: transposing a dyn call whose implementation makes
       another dyn call -- Area goes through shape-> -- fails to pack. Forward has no such problem,
       so it uses the wrapper above directly. An emitter missing here gets no gradient. */
    [Differentiable]
    public v3<Real> evaluate_direction<Real : atomicable_floating_point>(no_diff Area<Real> emitter, no_diff SurfaceInteraction<Real> si, no_diff v3<Real> uv) {
        return emitter.evaluate_direction(si, uv);
    }
    [Differentiable]
    public v3<Real> evaluate_direction<Real : atomicable_floating_point>(no_diff Point<Real> emitter, no_diff SurfaceInteraction<Real> si, no_diff v3<Real> uv) {
        return emitter.evaluate_direction(si, uv);
    }
    [Differentiable]
    public v3<Real> evaluate_direction<Real : atomicable_floating_point>(no_diff Constant<Real> emitter, no_diff SurfaceInteraction<Real> si, no_diff v3<Real> uv) {
        return emitter.evaluate_direction(si, uv);
    }
    [Differentiable]
    public v3<Real> evaluate_direction<Real : atomicable_floating_point>(no_diff HosekWilkie<Real> emitter, no_diff SurfaceInteraction<Real> si, no_diff v3<Real> uv) {
        return emitter.evaluate_direction(si, uv);
    }

    /* `d` lands on to_world's gradient tensor -- for an area light, on the shape's. */
    public void evaluate_direction_bwd<Real : atomicable_floating_point>(IEmitter<Real> *emitter, SurfaceInteraction<Real> si, v3<Real> uv, v3<Real>.Differential d) {
        IEmitter<Real> e = *emitter;
        if (let area = e as Area<Real>) __bwd_diff(evaluate_direction)(area, si, uv, d);
        else if (let point = e as Point<Real>) __bwd_diff(evaluate_direction)(point, si, uv, d);
        else if (let constant = e as Constant<Real>) __bwd_diff(evaluate_direction)(constant, si, uv, d);
        else if (let hosek = e as HosekWilkie<Real>) __bwd_diff(evaluate_direction)(hosek, si, uv, d);
    }
}
