implementing tamashii;

namespace integrators {
    public struct Path : IIntegrator {
        i32 max_depth;
        i32 rr_depth;
        bool emitter_samples;
        MIS mis;

        public __init(i32 max_depth = 0, i32 rr_depth = 3, bool emitter_samples = false, MIS mis = MIS::Power) {
            this.max_depth = max_depth;
            this.rr_depth = rr_depth;
            this.emitter_samples = emitter_samples;
            this.mis = mis;
        }

        [Differentiable]
        public void sample(uint2 id, IScene<float> scene, inout IRandom rng) {
            var spectrum = v3<float>(0);

            let sensor = scene.sensors[DiscreteDistribution::sample_evenly_spaced_range(0, scene.sensor_count - 1, rng.next_float())];
            if (any(id >= sensor.film().shape().xy)) return;

            var rs = sensor.sample_ray(rng.next_float2(), Coord(id));
            rs.weight *= scene.sensor_count;

            let nee = emitter_samples && scene.emitter_count > 0;
            let emitter_choice_pdf = DiscreteDistribution::pdf_evenly_spaced_range<float>(scene.emitter_count);
            
            float prev_bsdf_pdf = 1.0f; // Store the previous direction pdf (MIS)
            bool prev_bsdf_delta = true; // Suppress MIS on the primary ray
            SurfaceInteraction<float> prev_si = {{}};
            
            uint32_t depth = 0;
            [MaxIters(10)]
            while (true) {
                HitInfo hit = scene.intersect(rs.ray, RAY_FLAG_NONE, HitMask::all); depth++;
                war::handle_intersection_discontinuity(scene, rs, rng, depth, prev_bsdf_delta); // Only AD

                // Miss/Env Map
                if (hit.instance.is_none()) {
                    if (scene.environment_emitter) {
                        let radiance = scene.environment_emitter.evaluate_hit(rs.ray, hit);
                        float weight = 1.0f;
                        if (nee && !prev_bsdf_delta) {
                            DirectionSample<float> ds = { {}, rs.ray.direction };
                            let emitter_pdf = scene.environment_emitter->pdf_direction(prev_si, ds) * emitter_choice_pdf;
                            weight = mis(prev_bsdf_pdf, emitter_pdf);
                        }
                        spectrum += rs.weight * radiance * weight;
                    }
                    break;
                }

                // Surface Interaction
                IShape<float> shape = scene.shapes[hit.instance.index()];
                SurfaceInteraction<float> si = shape.surface_interaction(rs.ray, hit.primitive, hit.attribute);
                if (any(si.sp.emission > 0.0f)) {
                    float weight = 1.0f;
                    if (nee && !prev_bsdf_delta) {
                        DirectionSample<float> ds = { si.sp, rs.ray.direction, hit.t };
                        let emitter_pdf = shape.pdf_direction(prev_si, ds) * emitter_choice_pdf;
                        weight = mis(prev_bsdf_pdf, emitter_pdf);
                    }
                    spectrum += rs.weight * si.sp.emission * weight;
                    break;
                }

                // Russian Roulette Path Termination
                if (rr_depth > 0 && depth > rr_depth) {
                    const float p = min(max(rs.weight), 0.95f);
                    if (rng.next_float() >= p) break;
                    rs.weight /= p;
                }

                // Next Event Estimation
                if (nee) {
                    let emitter = scene.emitters[DiscreteDistribution::sample_evenly_spaced_range(0, scene.emitter_count - 1, rng.next_float())];
                    let sample = emitter.sample_direction(si, rng.next_float3());
                    let emitter_pdf = emitter_choice_pdf * sample.pdf;
                    if (emitter_pdf > 0.0f) {
                        Ray<float> shadowRay = { si.sp.position, 0.001f, sample.direction, sample.distance - 0.001f };
                        if (scene.unoccluded(shadowRay)) {
                            let wo_local = si.sp.frame.to_local(sample.direction);
                            let bsdf_value = shape.bsdf->eval(si, wo_local);
                            let weight = select(sample.delta, 1.0f, mis(emitter_pdf, shape.bsdf->pdf(si, wo_local)));
                            spectrum += rs.weight * sample.sp.emission * bsdf_value * weight / emitter_pdf;
                        }
                    }
                }

                // Next
                let sample = shape.bsdf->sample(si, rng.next_float3());
                rs.weight *= sample.weight;

                prev_bsdf_pdf = sample.pdf;
                prev_bsdf_delta = sample.delta;
                prev_si = si;

                rs.ray.origin = si.sp.position;
                rs.ray.direction = si.sp.frame.to_world(sample.wo);

                if (max_depth > 0 && depth >= max_depth) break;
            }
            sensor.film().add(float4(spectrum, 1), int4(id, 0, 0));
        }
    };
};
