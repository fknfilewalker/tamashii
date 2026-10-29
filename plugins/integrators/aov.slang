implementing tamashii;

namespace integrators {
    public struct AOV : IIntegrator {
        public enum class Mode {
            Albedo = 0,
            Depth = 1,
            TextureCoordinates = 2,
            GeometricNormal = 3,
            ShadingNormal = 4,
            Emittance = 5,
            Film = 6,
            Primitives = 7
        } mode;

        public __init(Mode mode = Mode::Albedo) {
            this.mode = mode;
        }

        [TreatAsDifferentiable]
        public void sample(uint2 id, IScene<float> scene, inout IRandom rng) {
            let camera = (scene.sensors[0] as sensors::Perspective<float>).value;
            if (any(id >= camera.film().shape().xy)) return;
            var rs = camera.sample_ray(rng.next_float2(), Coord(id));

            HitInfo hit = scene.intersect(rs.ray, RAY_FLAG_NONE, HitMask::all);
            if (hit.instance.is_none()) {
                if (scene.environment_emitter) {
                    rs.weight = scene.environment_emitter.evaluate_hit(rs.ray, hit);
                } else rs.weight = float3(0);
            } else {
                IShape<float> shape = scene.shapes[hit.instance.index()];
                SurfaceInteraction<float> si = shape.surface_interaction(rs.ray, hit.primitive, hit.attribute);
                switch (mode) {
                    case Mode::Albedo:
                        rs.weight *= shape.bsdf.get_reflectance(si);
                        break;
                    case Mode::Depth:
                        rs.weight *= float3(pow((hit.t - rs.ray.tmin) / (rs.ray.tmax - rs.ray.tmin), 0.4));
                        break;
                    case Mode::TextureCoordinates:
                        rs.weight *= float3(si.sp.uv, 0);
                        break;
                    case Mode::GeometricNormal:
                        rs.weight *= si.sp.frame.to_world(si.sp.geometric_normal) * 0.5f + 0.5f;
                        break;
                    case Mode::ShadingNormal:
                        rs.weight *= si.sp.frame.normal() * 0.5f + 0.5f;
                        break;
                    case Mode::Emittance:
                        rs.weight *= shape.emission;
                        break;
                    case Mode::Film:
                        if (shape.film) rs.weight *= shape.load(rs.ray, hit.primitive, si.sp.attribute).xyz;
                        break;
                    case Mode::Primitives:
                        let h = random::TEA::hash(hit.instance.index(), hit.primitive);
                        let hue = float(uint(h & 0xFFFFu)) / 65535.0f;
                        let rawSat = float(uint((h >> 16) & 0xFFu)) / 255.0f;
                        let rawVal = float(uint((h >> 24) & 0xFFu)) / 255.0f;
                        rs.weight *= post::hsv_to_rgb(float3(hue, lerp(0.5f, 1.0f, rawSat), lerp(0.5f, 1.0f, rawVal)));
                        break;
                }
            }   
            camera.film().add(float4(post::dither<float>(rs.weight, 0.3, rng.next_float()), 1), int4(id, 0, 0));
        }
    };
};