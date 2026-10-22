implementing tamashii;

public struct LensSample<Real : floating_point> : IDifferentiable {
    public v3<Real> direction;
    public Real distance;
    public int4 coords;
    public v2<Real> uv;
    public Real weight;
    public bool valid;
};

[anyValueSize(settings::sensor_interface_size)]
public interface ISensor<Real : atomicable_floating_point> {
    [Differentiable] public v4<Real> splat(v4<Real> point);
    [Differentiable] public RaySample<Real> sample_ray(v2<Real> uv);
    [Differentiable] public RaySample<Real> sample_ray(v2<Real> uv, vector<int, 4> coords); // includes rfilter
    [Differentiable] public LensSample<Real> sample_lens(SurfaceInteraction<Real> si, v3<Real> uv);
    public IFilm<Real, 4> film();
};

namespace sensors {
    public interface ICamera<Real : atomicable_floating_point> : ISensor<Real> {
        public property DiffTensor<m4x4<Real>, 1> to_sensor; // view matrix
        public property DiffTensor<m4x4<Real>, 1> to_sample; // projection matrix
        public property rfilters::Box<Real> rfilter;
        public property films::HDRTexture2D<Real, 4> sfilm;
        [mutating] public void update();
    };

    [Differentiable]
    public v2<Real> project<Real : atomicable_floating_point>(ISensor<Real>* sensor, v3<Real> v0, v3<Real> v1, no_diff Real t)
    {
        v3<Real> dir = v1 - v0;
        v4<Real> p = v4<Real>(v0 + t * dir, Real(1));
        v4<Real> cs = sensor->splat(p);
        v3<Real> ndc = cs.xyz / cs.w;
        return ndc.xy;
    }

    [Differentiable]
    public v2<Real> project<Real : atomicable_floating_point>(ISensor<Real> *sensor, v3<Real> p)
    {
        v4<Real> p = v4<Real>(p, Real(1));
        v4<Real> cs = sensor->splat(p);
        v3<Real> ndc = cs.xyz / cs.w;
        return ndc.xy;
    }
    [Differentiable]
    public v2<Real> project<Real : atomicable_floating_point>(ISensor<Real> sensor, v3<Real> p)
    {
        v4<Real> p = v4<Real>(p, Real(1));
        v4<Real> cs = sensor.splat(p);
        v3<Real> ndc = cs.xyz / cs.w;
        return ndc.xy;
    }
    [Differentiable]
    public LensSample<Real> sample_lens<Real : atomicable_floating_point>(ICamera<Real> camera, SurfaceInteraction<Real> si, v3<Real> uv)
    {
        return camera.sample_lens(si, uv);
    }
    
}