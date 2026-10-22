implementing tamashii;

public struct FilterSample<Real : floating_point> : IDifferentiable {
    public v2<Real> offset;
    public Real weight;
};

[anyValueSize(settings::rfilter_interface_size)]
__generic<Real : floating_point>
public dyn interface IRFilter {
    public property Real radius;

    [Differentiable]
    public FilterSample<Real> sample(v2<Real> uv);
    // x is the distance from the center of the filter kernel
    [Differentiable]
    public Real eval(Real x);
};
