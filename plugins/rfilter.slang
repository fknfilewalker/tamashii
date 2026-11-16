implementing tamashii;

public struct FilterSample<Real : floating_point> : IDifferentiable {
    public v2<Real> offset;
    public Real weight;
};

[anyValueSize(settings::rfilter_interface_size)]
public dyn interface IRFilter<Real : floating_point> {
    public property Real radius;

    [Differentiable]
    public FilterSample<Real> sample(Real2 uv);
    // x is the distance from the center of the filter kernel
    [Differentiable]
    public Real eval(Real x);
};

extension<Real, T> T
where Real : __BuiltinFloatingPointType
where T : IRFilter<Real>
{
    typealias Real2 = vector<Real, 2>;
    typealias Real3 = vector<Real, 3>;
    typealias Real4 = vector<Real, 4>;
}
