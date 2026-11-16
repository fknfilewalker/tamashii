implementing tamashii;

namespace bsdfs {
    public struct Diffuse<Real : floating_point> : IBsdf<Real>, IDifferentiable {
        public Texture<Real3> reflectance;

        public __init(Texture<Real3> reflectance) {
            this.reflectance = reflectance;
        }

        [Differentiable]
        public Real3 get_reflectance(SurfaceInteraction<Real> si) {
            return reflectance.sample(si.sp.uv);
        }

        [Differentiable]
        public BsdfSample<Real> sample(SurfaceInteraction<Real> si, no_diff Real3 uvw) {
            BsdfSample<Real> sample;
            sample.wo = warp::square_to_cosine_hemisphere(uvw.xy);
            sample.pdf = warp::square_to_cosine_hemisphere_pdf(sample.wo);
            sample.wo_pdf = sample.pdf;
            sample.delta = false;
            sample.weight = reflectance.sample(si.sp.uv);
            return sample;
        }

        [Differentiable]
        public Real3 eval(SurfaceInteraction<Real> si, Real3 wo) {
            return reflectance.sample(si.sp.uv) * constants<Real>::inv_pi * max(Real(0), wo.z);
        }

        [Differentiable]
        public Real pdf(SurfaceInteraction<Real> si, Real3 wo) {
            return warp::square_to_cosine_hemisphere_pdf(wo);
        }
    };
}