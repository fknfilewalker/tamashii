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
    [Differentiable] public vector<Real, 3> get_reflectance(SurfaceInteraction<Real> si);

    [Differentiable] public BsdfSample<Real> sample(SurfaceInteraction<Real> si, no_diff vector<Real, 3> uvw);
    [Differentiable] public vector<Real, 3> eval(SurfaceInteraction<Real> si, vector<Real, 3> wo);
    [Differentiable] public Real pdf(SurfaceInteraction<Real> si, vector<Real, 3> wo);
};
