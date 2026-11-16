implementing tamashii;

public struct BsdfSample<Real : floating_point> : IDifferentiable {
    public vector<Real, 3> wo;
    public Real wo_pdf; // only for current outgoing direction, does not consider mixed/layered bsdf
    public vector<Real, 3> weight;
    public Real pdf;
    public bool delta;
};

// in z frame
[anyValueSize(settings::bsdf_interface_size)]
public dyn interface IBsdf<Real : floating_point> {
    [Differentiable] public Real3 get_reflectance(SurfaceInteraction<Real> si);

    [Differentiable] public BsdfSample<Real> sample(SurfaceInteraction<Real> si, no_diff Real3 uvw);
    [Differentiable] public Real3 eval(SurfaceInteraction<Real> si, Real3 wo);
    [Differentiable] public Real pdf(SurfaceInteraction<Real> si, Real3 wo);
};

extension<Real, T> T
where Real : __BuiltinFloatingPointType
where T : IBsdf<Real>
{
    typealias Real2 = vector<Real, 2>;
    typealias Real3 = vector<Real, 3>;
    typealias Real4 = vector<Real, 4>;
}

// Shading normals make the BSDF non-symmetric. Multiply bsdf eval/sample weights by this when
// transporting importance (light tracing); 1 for flat shading (Veach 1997, Eq. 5.19)
[Differentiable]
public Real shading_normal_correction<Real : floating_point>(SurfaceInteraction<Real> si, vector<Real, 3> wo) {
    let ng = si.sp.geometric_normal;
    return safe_div(abs(si.wi.z) * abs(dot(wo, ng)), abs(dot(si.wi, ng)) * abs(wo.z));
}
