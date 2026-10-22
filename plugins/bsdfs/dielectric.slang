implementing tamashii;

namespace bsdfs {
__generic<Real : floating_point>
    public struct Dielectric : IBsdf<Real>, IDifferentiable {
        public Texture<vector<Real, 3>> reflectance;
        public Texture<Real> roughness;
        public Texture<Real> eta_i;
        public Texture<Real> eta_t;

        public __init(Texture<vector<Real, 3>> reflectance) {
            this.reflectance = reflectance;
            eta_i = Texture<Real>(1.0f);
            eta_t = Texture<Real>(1.5f);
            roughness = Texture<Real>(0.5f);
        }

        [Differentiable]
        public vector<Real, 3> get_reflectance(SurfaceInteraction<Real> si) {
            return reflectance.sample(si.sp.uv);
        }

        [Differentiable]
        public BsdfSample<Real> sample(SurfaceInteraction<Real> si, no_diff v3<Real> uvw) {
            Real widotn = Frame<Real>::cos_theta(si.wi);
            const Real sample_roughness = (Real(1.2f) - Real(0.2f) * sqrt(abs(widotn))) * roughness.sample(si.sp.uv);
            let mf = microfacet::GGX<Real>(sample_roughness);

            bool rough = mf.roughness > Real(0.01);

            v3<Real> m = rough ? mf.sample_vndf(si.wi, uvw.xy) : v3<Real>(Real(0), Real(0), Real(1));
            // make sure m points in same hemisphere direction
            // m *= Real(sign(dot(m, si.sp.geometric_normal)));
            Real cosThetaT;
            Real _eta_i = eta_i.sample(si.sp.uv);
            Real _eta_t = eta_t.sample(si.sp.uv);
            const Real fi = fresnel::dielectric(_eta_i, _eta_t, (Frame<Real>::cos_theta(si.wi)), cosThetaT);
            // const Real fi = fresnel::schlick(abs(Frame<Real>::cos_theta(si.wi)), no_diff fresnel::f0(_eta_i, _eta_t));
            const bool reflection = uvw.z < fi;

            BsdfSample<Real> sample;
            if (reflection) {
                sample.wo = reflect(-si.wi, m);
                if (sample.wo.z * si.wi.z < Real(0)) return {};
            }
            else {
                sample.wo = refract(-si.wi, m, _eta_i / _eta_t);
                if (_eta_i == _eta_t) sample.wo = -si.wi;
                if (sample.wo.z * si.wi.z > Real(0) || all(sample.wo.x == Real(0) && sample.wo.y == Real(0) && sample.wo.z == Real(0))) {
                    return {};
                }
            }
            if (sample.wo.z * si.wi.z == Real(0)) return {};

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(sample.wo, m);
            Real mdotn = Frame<Real>::cos_theta(m);
            Real wodotn = Frame<Real>::cos_theta(sample.wo);

            // pdf
            sample.wo_pdf = rough ? mf.pdf(mdotn) : Real(1);

            // weight
            Real g = mf.g(widotn, wodotn);
            Real denom = abs(widotn) * abs(mdotn);
            sample.weight = reflectance.sample(si.sp.uv) * (rough ? safe_div(Real(1) + mf.g1(widotn), Real(1) + mf.g1(widotn) * mf.g1(wodotn)) : Real(1));
            if (reflection) /* specular */ {
                if (rough) sample.wo_pdf *= microfacet::d_pdf_h_to_wo(wodotm);
                // equivalent (non shortened version):
                // sample.value = vec3((max(woDotN, 0.0f) * ggx_eval(e.wi,wo,e.n,m,e.roughness) * BSDF_GGX(wiDotN, woDotN)) / (pdf));
                sample.pdf = fi * sample.wo_pdf;
            }
            else /* refract */ {
                if (rough) sample.wo_pdf *= microfacet::d_pdf_h_to_wo(widotm, wodotm, _eta_i, _eta_t);
                // equivalent (non shortened version)
                // sample.value = vec3((abs(woDotN) * ggx_eval(e.wi,wo,e.n,m,e.roughness) * BSDF_GGX(wiDotM, woDotM, wiDotN, woDotN, eta_i, eta_t)) / (pdf));
                sample.pdf = (Real(1) - fi) * sample.wo_pdf;
            }

            sample.delta = !rough;
            return sample;
        }

        [Differentiable]
        public vector<Real, 3> eval(SurfaceInteraction<Real> si, v3<Real> wo) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);

            Real _eta_i = eta_i.sample(si.sp.uv);
            Real _eta_t = eta_t.sample(si.sp.uv);
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(wo);
            const bool reflection = (widotn * wodotn) >= Real(0.0f);
            v3<Real> m;
            if (reflection) m = halfvector(si.wi, wo);
            else m = halfvector(si.wi, wo, _eta_i, _eta_t);
            // m *= Real(sign(dot(m, si.sp.geometric_normal)));

            Real cosThetaT;
            const Real fi = fresnel::dielectric(_eta_i, _eta_t, (Frame<Real>::cos_theta(si.wi)), cosThetaT);
            Real ggx = mf.d(Frame<Real>::cos_theta(m)) * mf.g(widotn, wodotn);

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(wo, m);
            if (!rough) return v3<Real>(0);
            v3<Real> value = reflectance.sample(si.sp.uv);
            if (reflection) value *= fi * ggx * microfacet::torrance_sparrow(widotn, wodotn);
            else {
                value *= (Real(1) - fi) * ggx * microfacet::torrance_sparrow(widotm, wodotm, widotn, wodotn, _eta_i, _eta_t);
            }
            value *= abs(wodotn);
            return value;
        }

        [Differentiable]
        public Real pdf(SurfaceInteraction<Real> si, v3<Real> wo) {
            let mf = microfacet::GGX<Real>(roughness.sample(si.sp.uv));
            bool rough = mf.roughness > Real(0.01);
            
            Real _eta_i = eta_i.sample(si.sp.uv);
            Real _eta_t = eta_t.sample(si.sp.uv);
            Real widotn = Frame<Real>::cos_theta(si.wi);
            Real wodotn = Frame<Real>::cos_theta(wo);
            const bool reflection = (widotn * wodotn) >= Real(0.0f);
            v3<Real> m;
            if (reflection) m = halfvector(si.wi, wo);
            else m = halfvector(si.wi, wo, _eta_i, _eta_t);
            m *= Real(sign(dot(m, si.sp.geometric_normal)));

            Real cosThetaT;
            const Real fi = fresnel::dielectric(_eta_i, _eta_t, abs(Frame<Real>::cos_theta(si.wi)), cosThetaT);

            Real widotm = dot(si.wi, m);
            Real wodotm = dot(wo, m);
            Real pdf = mf.pdf_vndf(widotn, widotm, Frame<Real>::cos_theta(m));
            if (reflection) pdf *= fi * microfacet::d_pdf_h_to_wo(wodotm);
            else pdf *= (Real(1) - fi) * microfacet::d_pdf_h_to_wo(widotm, wodotm, _eta_i, _eta_t);
            return pdf;
        }
    };
}