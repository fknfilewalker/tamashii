implementing tamashii;

namespace bsdfs {
    public struct Plastic<Real : floating_point> : IBsdf<Real>, IDifferentiable {
        public Texture<Real3> reflectance;
        public Texture<Real> roughness;
        public Texture<Real> eta_i;
        public Texture<Real> eta_t;

        public __init(Texture<Real3> reflectance) {
            this.reflectance = reflectance;
            eta_i = Texture<Real>(1.0f);
            eta_t = Texture<Real>(1.5f);
            roughness = Texture<Real>(0.5f);
        }

        [Differentiable]
        public Real3 get_reflectance(SurfaceInteraction<Real> si) {
            return reflectance.sample(si.sp.uv);
        }

        [Differentiable]
        public BsdfSample<Real> sample(SurfaceInteraction<Real> si, no_diff Real3 uvw) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            Real _eta_i = eta_i.sample(si.sp.uv);
            Real _eta_t = eta_t.sample(si.sp.uv);
            Real3 m = rough ? mf.sample_vndf(si.wi, uvw.xy) : si.sp.geometric_normal;
            // make sure m points in same hemisphere direction
            m *= Real(sign(dot(m, si.sp.geometric_normal)));

            Real cosThetaT;
            const Real fi = fresnel::dielectric(_eta_i, _eta_t, abs(Frame<Real>::cos_theta(si.wi)), cosThetaT);
            const bool reflection = uvw.z < fi;

            BsdfSample<Real> sample;
            if (reflection) sample.wo = reflect(-si.wi, m);
            else sample.wo = warp::square_to_cosine_hemisphere(uvw.xy);

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(sample.wo, m);
            Real mdotn = Frame<Real>::cos_theta(m);
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(sample.wo);

            // weight
            sample.pdf = Real(1);
            sample.weight = reflectance.sample(si.sp.uv);
            if (reflection) /* specular */ {
                if (rough) {
                    sample.weight *= safe_div(Real(1) + mf.g1(widotn), Real(1) + mf.g1(widotn) * mf.g1(wodotn));
                    sample.wo_pdf = max(mf.pdf(mdotn), Real(0)) * microfacet::d_pdf_h_to_wo(wodotm);
                    sample.pdf = fi * sample.wo_pdf;
                }
                sample.delta = !rough;
            }
            else {
                sample.wo_pdf = warp::square_to_cosine_hemisphere_pdf(sample.wo);
                sample.pdf = (Real(1) - fi) * sample.wo_pdf;
                sample.delta = false;
            }
            
            return sample;
        }

        [Differentiable]
        public Real3 eval(SurfaceInteraction<Real> si, Real3 wo) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            Real _eta_i = eta_i.sample(si.sp.uv);
            Real _eta_t = eta_t.sample(si.sp.uv);
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(wo);
            const bool reflection = (widotn * wodotn) >= Real(0.0f);
            Real3 m;
            if (reflection) m = halfvector(si.wi, wo);
            else m = halfvector(si.wi, wo, _eta_i, _eta_t);
            m *= Real(sign(dot(m, si.sp.geometric_normal)));

            Real cosThetaT;
            const Real fi = fresnel::dielectric(_eta_i, _eta_t, abs(Frame<Real>::cos_theta(si.wi)), cosThetaT);
            Real ggx = mf.d(Frame<Real>::cos_theta(m)) * mf.g(widotn, wodotn);

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(wo, m);
            Real3 diffuse = constants<Real>::inv_pi;
            Real specular = rough ? ggx * microfacet::torrance_sparrow(widotn, wodotn) : Real(0);
            return reflectance.sample(si.sp.uv) * lerp(diffuse, specular, fi) * max(wodotn, Real(0));
        }

        [Differentiable]
        public Real pdf(SurfaceInteraction<Real> si, Real3 wo) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            Real _eta_i = eta_i.sample(si.sp.uv);
            Real _eta_t = eta_t.sample(si.sp.uv);
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(wo);
            const bool reflection = (widotn * wodotn) >= Real(0.0f);
            Real3 m;
            if (reflection) m = halfvector(si.wi, wo);
            else m = halfvector(si.wi, wo, _eta_i, _eta_t);
            m *= Real(sign(dot(m, si.sp.geometric_normal)));

            Real cosThetaT;
            const Real fi = fresnel::dielectric(_eta_i, _eta_t, abs(Frame<Real>::cos_theta(si.wi)), cosThetaT);

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(wo, m);
            Real diffuse = warp::square_to_cosine_hemisphere_pdf(wo);
            Real specular = rough ? mf.pdf_vndf(widotn, widotm, Frame<Real>::cos_theta(m)) * microfacet::d_pdf_h_to_wo(wodotm) : Real(0);
            return lerp(diffuse, specular, fi);
        }
    };
}