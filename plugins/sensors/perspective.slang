implementing tamashii;

namespace sensors {
    public struct Perspective<Real : atomicable_floating_point> : ICamera<Real> , IDifferentiable {
        public no_diff DiffTensor<m4x4<Real>, 1> to_sensor; // view matrix
        public no_diff DiffTensor<m4x4<Real>, 1> to_sample; // projection matrix
        public rfilters::Box<Real> rfilter;
        Real fov_x, near, far;
        public films::HDRTexture2D<Real, 4> sfilm;

        typealias Real2 = vector<Real, 2>;
        typealias Real3 = vector<Real, 3>;
        typealias Real4 = vector<Real, 4>;

        public enum class FovAxis { X, Y };
        public __init(Real fov, FovAxis fov_axis = FovAxis::X, Real near = Real(0.01), Real far = Real(10000)) {
            if (fov_axis == FovAxis::Y) {
                fov = degrees(Real(2) * atan(tan(radians(fov) * Real(0.5)) * sfilm.aspect()));
            }
            this.fov_x = fov; this.near = near; this.far = far;
            update();
        }

        [Differentiable]
        public Real4 position() {
            return to_sensor.to_transform().inverse[3];
        };

        [mutating]
        public void update() {
            if (!to_sample.is_valid()) return;
            var projection = Transform<Real>::perspective(fov_x, near, far);
            projection.forward[0][0] *= sfilm.inv_aspect();
            projection.inverse[0][0] *= sfilm.aspect();
            to_sample[{0}] = projection.forward;
            to_sample[{1}] = projection.inverse;
        }

        // world space to sensor space
        [Differentiable]
        public Real4 splat(vector<Real, 4> point) {
            return mul(mul(point, to_sensor.to_transform().forward), to_sample.to_transform().forward);
        }

        [Differentiable]
        Ray<Real> get_camera_ray(Real2 clip_space_xy) {
            let m = transpose(mul(to_sensor.to_transform().forward, to_sample.to_transform().forward));
            // https://momentsingraphics.de/CameraRays.html
            let u = m[0].xyz - clip_space_xy.x * m[3].xyz;
            let v = m[1].xyz - clip_space_xy.y * m[3].xyz;
            var direction = normalize(cross(u, v));
            // Orient the direction into the scene (the camera looks down -z in view space).
            let forward = -to_sensor.to_transform().inverse[2].xyz;
            direction = dot(direction, forward) < Real(0) ? -direction : direction;
            return Ray<Real>(position().xyz, detach(near), direction, detach(far));
        }

        // rand = (random() * 2.0f) - 1.0f // between -1 and 1
        // center_offset = vector<Real, 2>(0.5) // center of pixel
        // rand_offset = center_offset + rand * sample_radius // offset by radius around pixel center
        // sample_pos = (index.xy + rand_offset) / size.xy
        [Differentiable]
        public RaySample<Real> sample_ray(Real2 uv)
        {
            const Real2 screen_space_clipped = uv * Real(2) - Real(1);
            return { get_camera_ray(screen_space_clipped), Real(1) };
        }

        [Differentiable]
        public RaySample<Real> sample_ray(Real2 uv, vector<int, 4> coords)
        {
            let filter_sample = rfilter.sample(uv);
            uv = (Real2(coords.xy) + Real2(0.5) + filter_sample.offset) / Real2(sfilm.shape().xy);
            const Real2 screen_space_clipped = uv * Real(2) - Real(1);
            return { get_camera_ray(screen_space_clipped), filter_sample.weight };
        }

        [Differentiable]
        public Real importance(Real3 d) {
            let inv_ct = Real(1) / abs(d.z);
            let p = d.xy * inv_ct;

            Real3 pmin = mul(Real4(Real(-1), Real(-1), Real(0), Real(1)), to_sample.to_transform().inverse).xyz;
            Real3 pmax = mul(Real4(Real(1), Real(1), Real(0), Real(1)), to_sample.to_transform().inverse).xyz;
            Real2 a = pmin.xy / abs(pmin.z);
            Real2 b = pmax.xy / abs(pmax.z);
            Real2 bb_min = min(a, b);
            Real2 bb_max = max(a, b);
            return inv_ct * inv_ct * inv_ct / prod(bb_max - bb_min);
        }

        [Differentiable]
        public LensSample<Real> sample_lens(SurfaceInteraction<Real> si, Real3 uv) {
            Real3 ref_p = mul(Pos(si.sp.position), to_sensor.to_transform().forward).xyz;

            LensSample<Real> ls;
            let cs = mul(Pos(ref_p), to_sample.to_transform().forward);
            var ndc = cs.xyz / cs.w;
            if (any(abs(ndc) >= Real(1))) return {};
            ls.direction = position().xyz - si.sp.position;
            ls.distance = length(ls.direction);
            let inv_dist = Real(1) / ls.distance;
            ls.direction *= inv_dist;
            
            let local_d = ref_p * inv_dist;
            ls.weight = inv_dist * inv_dist * importance(local_d);
            ls.valid = true;
            ls.uv = (ndc.xy + Real(1)) * Real(0.5);
            ls.coords = Coord(int2(ls.uv * Real2(sfilm.shape().xy)));
            if (any(ls.coords.xy < 0) || any(ls.coords.xy >= sfilm.shape().xy)) return {};
            return ls;
        }

        public IFilm<Real, 4> film() {
            return sfilm;
        }
    };
}
