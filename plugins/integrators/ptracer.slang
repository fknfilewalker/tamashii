implementing tamashii;

namespace integrators {
    public struct Ptracer : IIntegrator {
        i32 max_depth;
        i32 rr_depth;

        public __init(i32 max_depth = 0, i32 rr_depth = 3) {
            this.max_depth = max_depth;
            this.rr_depth = rr_depth;
        }

        [Differentiable]
        public void sample(uint2 id, IScene<float> scene, inout IRandom rng) {
            if (scene.emitter_count <= 0) return;

            let emitter = scene.emitters[DiscreteDistribution::sample_evenly_spaced_range(0, scene.emitter_count - 1, rng.next_float())];
            var rs = emitter.sample_ray(rng.next_float3(), rng.next_float2());
            rs.weight *= scene.emitter_count;

            uint32_t depth = 0;
            [MaxIters(10)]
            while (true) {
                HitInfo hit = scene.intersect(rs.ray, RAY_FLAG_NONE, HitMask::all); depth++;
                if (hit.instance.is_none()) break;
                IShape<float> shape = scene.shapes[hit.instance.index()];
                SurfaceInteraction<float> si = shape.surface_interaction(rs.ray, hit.primitive, hit.attribute);

                // Connect to sensor
                [MaxIters(10)]
                for (int i = 0; i < scene.sensor_count; i++) {
                    let sensor = scene.sensors[i];
                    let ls = sensor.sample_lens(si, rng.next_float3());
                    if (ls.valid) {
                        // Ray<float> visibilityRay = { si.sp.position, ray.tmin, ls.direction, ls.distance - 0.001f };
                        // if (scene.unoccluded(visibilityRay)) {
                            // let brdf = shape.bsdf->eval(si, si.sp.frame.to_local(ls.direction));
                            // sensor.film.atomic_add(v4<float>(spectrum * brdf * ls.weight, 1.0f), ls.coords);
                        // }
                        
                        // In order to get the sensor related part of the gradient resulting from connection ray movements within the pixel, 
                        // we would need to linearly distribute it on the sensor which requires 4 writes and is very noisy
                        // Instead we reparametize this part so that the connection ray stays fixed.
                        // Todo: Improve code structure (find cleaner solution)
                        let fixed_sensor_direction = detach(ls.direction);
                        let sensor_origin = detach(si.sp.position + ls.direction * ls.distance);
                        Ray<float> sensor_ray = Ray<float>(sensor_origin, rs.ray.tmin, -fixed_sensor_direction, ls.distance + 0.001f);
                        let sensor_target_si = shape.surface_intersection(sensor_ray, hit.primitive, false);
                        let light_to_sensor = sensor_target_si.sp.position - rs.ray.origin;
                        Ray<float> connection_ray = Ray<float>(rs.ray.origin, rs.ray.tmin, normalize(light_to_sensor), length(light_to_sensor) + 0.001f);
                        let sensor_si = shape.surface_intersection(connection_ray, hit.primitive, false);

                        let sensor_distance = length(sensor_origin - sensor_si.sp.position);
                        Ray<float> visibility_ray = Ray<float>(sensor_si.sp.position, rs.ray.tmin, fixed_sensor_direction, sensor_distance - 0.001f);
                        if (!isnan(sensor_distance) && scene.unoccluded(visibility_ray)) {
                            let sensor_direction_local = sensor_si.sp.frame.to_local(fixed_sensor_direction);
                            let light_jacobian = abs(sensor_si.wi.z) / length2(rs.ray.origin - sensor_si.sp.position);
                            let sensor_jacobian = abs(sensor_direction_local.z) * ls.weight;
                            let measure_jacobian = safe_div(light_jacobian, sensor_jacobian);
                            let reparam_weight = safe_div(measure_jacobian, detach(measure_jacobian));
                            let brdf = shape.bsdf->eval(sensor_si, sensor_direction_local);
                            sensor.film().atomic_add(v4<float>(rs.weight * brdf * ls.weight * reparam_weight, 1.0f), ls.coords);
                        }
                    }
                }

                // Russian Roulette Path Termination
                if (rr_depth > 0 && depth > rr_depth) {
                    const float p = min(max(rs.weight), 0.95f);
                    if (rng.next_float() >= p || p == 0.0f) break;
                    rs.weight /= p; // pdf
                }

                let sample = shape.bsdf->sample(si, rng.next_float3());
                rs.weight *= sample.weight;

                rs.ray.origin = si.sp.position;
                rs.ray.direction = si.sp.frame.to_world(sample.wo);

                if (shape.film) {
                    Ray<float> out_ray = Ray<float>(rs.ray.origin, rs.ray.tmin, -rs.ray.direction, rs.ray.tmax);
                    shape.store(float4(rs.weight, 1), out_ray, hit.primitive, si.sp.attribute);
                }

                if (max_depth > 0 && depth >= max_depth) break;
            };
        }
    };
};
