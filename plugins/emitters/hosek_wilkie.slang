implementing tamashii;

namespace emitters {
    /* Analytic sky from the Hosek-Wilkie model (utils/hosek_wilkie.slang).

       Zero below the horizon, no atmospheric transmittance. Once the sun sets the fit is
       held at sunset and `sun_scale` switches the disc off. Sampling mixes a cone on the
       disc with a uniform hemisphere, chosen with probability `sun_select_pdf`: the disc
       is too small to find by chance, the sky too broad to reach through the cone. */
    public struct HosekWilkie<Real : atomicable_floating_point> : IEmitter<Real>, IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_world;
        public no_diff hosek_wilkie::Dataset dataset;
        public no_diff v3<Real> sun_direction;  // world space, normalized
        public no_diff Real sun_cos_radius;     // cosine of the disc's angular radius
        public no_diff Real sun_select_pdf;     // probability of sampling the disc
        public no_diff Real sun_scale;          // disc brightness; zero switches it off
        public Real scale;

        public __init() {
            this.to_world = DiffTensor<m4x4<Real>, 1>();
            this.dataset = hosek_wilkie::Dataset();
            this.sun_direction = v3<Real>(Real(0), Real(1), Real(0));
            this.sun_cos_radius = Real(0.99999010);  // cos(0.51 degrees / 2)
            this.sun_select_pdf = Real(0.5);
            this.sun_scale = Real(1);
            this.scale = Real(1);
        }

        public Real sun_solid_angle() {
            return constants<Real>::two_pi * (Real(1) - sun_cos_radius);
        }

        /* World direction into the sky's frame, still Y-up. */
        public v3<Real> to_local_direction(v3<Real> direction) {
            return normalize(mul(v4<Real>(direction, Real(0.0)), to_world.to_transform().inverse).xyz);
        }

        /* Sky radiance, without the solar disc. */
        [Differentiable]
        public v3<Real> sky(no_diff v3<Real> direction) {
            let model = no_diff v3<float>(hosek_wilkie::to_model<Real>(no_diff to_local_direction(direction)));
            return scale * v3<Real>(no_diff hosek_wilkie::sky_radiance(dataset, model));
        }

        /* Sky plus the solar disc if `direction` falls inside it. */
        [Differentiable]
        public v3<Real> radiance(no_diff v3<Real> direction) {
            var result = sky(direction);
            if (sun_scale > Real(0) && dot(direction, sun_direction) >= sun_cos_radius) {
                let model = no_diff v3<float>(hosek_wilkie::to_model<Real>(no_diff to_local_direction(direction)));
                result += scale * sun_scale * v3<Real>(no_diff hosek_wilkie::sun_radiance(dataset, model));
            }
            return result;
        }

        /* Uniform direction in the cone around the sun. */
        public v3<Real> sample_sun_cone(v2<Real> uv) {
            let cos_theta = Real(1) - uv.y * (Real(1) - sun_cos_radius);
            let sin_theta = safe_sqrt(Real(1) - cos_theta * cos_theta);
            let phi = constants<Real>::two_pi * uv.x;
            return Frame<Real>(sun_direction).to_world(v3<Real>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta));
        }

        /* Solid-angle density of the two lobes, for MIS. The hemisphere term is counted
           for every direction so the density never hits zero, which would make an
           integrator's MIS weight divide zero by zero. */
        public Real pdf_of(SurfaceInteraction<Real> si, v3<Real> direction) {
            var pdf = (Real(1) - sun_select_pdf) * warp::square_to_uniform_hemisphere_pdf<Real>();
            if (dot(direction, sun_direction) >= sun_cos_radius)
                pdf += sun_select_pdf / sun_solid_angle();
            return pdf;
        }

        [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, v3<Real> uv) {
            DirectionSample<Real> ds;
            ds.sp.position = v3<Real>(0);
            // uv.z picks the lobe, uv.xy warps within it
            if (uv.z < sun_select_pdf)
                ds.direction = no_diff sample_sun_cone(uv.xy);
            else
                ds.direction = si.sp.frame.to_world(warp::square_to_uniform_hemisphere<Real>(uv.xy));
            ds.distance = Real(10000.0);
            ds.sp.frame = Frame<Real>(-ds.direction);
            ds.sp.geometric_normal = v3<Real>(Real(0.0), Real(0.0), Real(1.0));
            ds.sp.area = Real(0);
            ds.pdf = no_diff pdf_of(si, ds.direction);
            ds.delta = false;
            ds.sp.emission = radiance(ds.direction);
            return ds;
        }

        [Differentiable] public RaySample<Real> sample_ray(v3<Real> uv_pos, v2<Real> uv_dir) {
            let direction = warp::square_to_uniform_sphere<Real>(uv_dir);
            let weight = radiance(direction) * constants<Real>::four_pi;
            return { Ray<Real>(warp::square_to_uniform_hemisphere<Real>(uv_pos.xy) * Real(10000.0), direction), weight };
        }

        [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds) {
            return no_diff pdf_of(si, ds.direction);
        }

        [Differentiable] public v3<Real> evaluate_direction(SurfaceInteraction<Real> si, v3<Real> uv) {
            let ds = sample_direction(si, uv);
            if (ds.pdf <= Real(0)) return v3<Real>(Real(0));
            return ds.sp.emission * abs(dot(ds.direction, si.sp.frame.normal())) / ds.pdf;
        }

        [Differentiable] public v3<Real> evaluate_hit(Ray<Real> ray, HitInfo hi) {
            return radiance(normalize(ray.direction));
        }

        public uint intersect(inout Ray<Real> ray, const uint cullmode,
                        inout v2<Real> barycentric, inout uint primitive_idx, inout uint geometry_idx) {
            return 0;
        }
    };
}
