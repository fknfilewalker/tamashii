implementing tamashii;

namespace bsdfs {
    public struct Conductor<Real : floating_point> : IBsdf<Real>, IDifferentiable {
        public Texture<vector<Real, 3>> reflectance;
        public Texture<Real> roughness;
        public Texture<vector<Real, 3>> eta;
        public Texture<vector<Real, 3>> k;

        [Differentiable]
        public vector<Real, 3> get_reflectance(SurfaceInteraction<Real> si) {
            return reflectance.sample(si.sp.uv);
        }

        [Differentiable]
        public BsdfSample<Real> sample(SurfaceInteraction<Real> si, no_diff v3<Real> uvw) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            v3<Real> m = rough ? mf.sample_vndf(si.wi, uvw.xy) : si.sp.geometric_normal;
            // v3<Real> m = rough ? mf.sample(uvw.xy) : si.sp.geometric_normal;
            // make sure m points in same hemisphere direction
            m *= Real(sign(dot(m, si.sp.geometric_normal)));

            const v3<Real> fi = fresnel::schlick(abs(Frame<Real>::cos_theta(si.wi)), reflectance.sample(si.sp.uv));

            BsdfSample<Real> sample;
            sample.wo = reflect(-si.wi, m);
            if (sample.wo.z * si.wi.z < Real(0)) return {};

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(sample.wo, m);
            Real mdotn = Frame<Real>::cos_theta(m);
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(sample.wo);

            // weight
            sample.pdf = sample.wo_pdf = Real(1);
            sample.weight = fi;
            if (rough) {
                // sample.weight = safe_div(abs(widotm) * mf.g(widotn, wodotn), abs(widotn) * abs(mdotn));
                // sample.wo_pdf = max(mf.pdf(mdotn), Real(0)) * microfacet::d_pdf_h_to_wo(wodotm);

                sample.weight *= safe_div(Real(1) + mf.g1(widotn), Real(1) + mf.g1(widotn) * mf.g1(wodotn));            // vndf height correlated
                // sample.weight *= mf.g1(wodotn);                                                                      // vndf uncorrelated
                sample.wo_pdf = max(mf.pdf_vndf(widotn, widotm, mdotn), Real(0)) * microfacet::d_pdf_h_to_wo(wodotm);   // vndf
                sample.pdf = sample.wo_pdf;
            }
            sample.delta = !rough;

            return sample;
        }

        [Differentiable]
        public vector<Real, 3> eval(SurfaceInteraction<Real> si, v3<Real> wo) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            v3<Real> m = halfvector(si.wi, wo);
            m *= Real(sign(dot(m, si.sp.geometric_normal)));
            
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(wo);
            const v3<Real> fi = fresnel::schlick(abs(widotn), reflectance.sample(si.sp.uv));
            Real ggx = mf.d(Frame<Real>::cos_theta(m)) * mf.g(widotn, wodotn);
            Real specular = rough ? ggx * microfacet::torrance_sparrow(widotn, wodotn) : Real(0);
            return specular * fi * max(wodotn, Real(0));
        }

        [Differentiable]
        public Real pdf(SurfaceInteraction<Real> si, v3<Real> wo) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(wo);
            const bool reflection = (widotn * wodotn) >= Real(0.0f);
            v3<Real> m = halfvector(si.wi, wo);
            m *= Real(sign(dot(m, si.sp.geometric_normal)));

            const v3<Real> fi = fresnel::schlick(abs(widotn), reflectance.sample(si.sp.uv));

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(wo, m);
            // Real specular = rough ? mf.pdf(Frame<Real>::cos_theta(m)) * microfacet::d_pdf_h_to_wo(wodotm) : Real(0);
            Real specular = rough ? max(mf.pdf_vndf(widotn, widotm, Frame<Real>::cos_theta(m)), Real(0)) * microfacet::d_pdf_h_to_wo(wodotm) : Real(0); // vndf
            return specular;
        }
    };
}