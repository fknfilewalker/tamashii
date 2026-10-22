implementing tamashii;

// Hosek-Wilkie sky [2012] and solar radiance [2013], ported from the reference release
// (https://cgg.mff.cuni.cz/projects/SkylightModelling/). Per wavelength,
//
//     L = (1 + A exp(B / (cos(theta) + 0.01)))
//       * (C + D exp(E gamma) + F cos^2(gamma) + G mie(I, gamma) + H sqrt(cos(theta)))
//
// `tamashii.hosek_wilkie.pack` cooks the parameters and writes the buffer this reads.
// Zero below the horizon, where the fit does not apply. Angles are Z-up (`to_model`).

namespace hosek_wilkie {

    /* tamashii is Y-up, the model is Z-up. Both frames stay right-handed. */
    public v3<Real> to_model<Real : floating_point>(v3<Real> world) {
        return v3<Real>(world.x, -world.z, world.y);
    }

    public static const uint SKY_PARAM_COUNT = 9;
    public static const uint LIMB_DARKENING_ORDER = 6;

    public struct Dataset {
        public uint channels;
        public Tensor<float, 1> data;

        public __init() {
            this.channels = 0;
            this.data = Tensor<float, 1>();
        }

        public bool is_valid() { return data.is_valid() && channels > 0; }

        /* Section offsets; the first eight floats are the header. */
        public property sky_params : uint { get { return 8; } }
        public property sky_radiance : uint { get { return sky_params + channels * SKY_PARAM_COUNT; } }
        public property rgb_weights : uint { get { return sky_radiance + channels; } }
        public property limb_darkening : uint { get { return rgb_weights + channels * 3; } }

        public v3<float> sun_direction() { return v3<float>(data[{0}], data[{1}], data[{2}]); }
        /* 1 / sin^2(solar radius) */
        public float disc_scale() { return data[{3}]; }

        [ForceInline]
        public v3<float> rgb_at(uint offset) {
            let i = int(offset);
            return v3<float>(data[{i}], data[{i + 1}], data[{i + 2}]);
        }
    };

    /* Sky radiance along a normalized model-space direction, reduced to linear sRGB. */
    public v3<float> sky_radiance(Dataset ds, v3<float> direction) {
        if (!ds.is_valid()) return v3<float>(0.0);

        let cos_theta = direction.z;
        // sqrt(cos(theta)) is not real below the horizon
        if (cos_theta <= 0.0) return v3<float>(0.0);

        let cos_gamma = clamp(dot(direction, ds.sun_direction()), -1.0, 1.0);
        let gamma = acos(cos_gamma);
        let rayleigh = cos_gamma * cos_gamma;
        let zenith = sqrt(cos_theta);
        let horizon = 1.0 / (cos_theta + 0.01);

        v3<float> rgb = v3<float>(0.0);
        for (uint ch = 0; ch < ds.channels; ++ch) {
            // the nine fitted parameters, named as in the paper
            let base = int(ds.sky_params + ch * SKY_PARAM_COUNT);
            let a = ds.data[{base + 0}]; // horizon gradient amplitude
            let b = ds.data[{base + 1}]; // horizon gradient falloff
            let c = ds.data[{base + 2}]; // base radiance
            let d = ds.data[{base + 3}]; // solar aureole amplitude
            let e = ds.data[{base + 4}]; // solar aureole falloff
            let f = ds.data[{base + 5}]; // Rayleigh term
            let g = ds.data[{base + 6}]; // Mie term
            let h = ds.data[{base + 7}]; // zenith gradient
            let i = ds.data[{base + 8}]; // Mie eccentricity

            let mie = (1.0 + rayleigh) / pow(1.0 + i * i - 2.0 * i * cos_gamma, 1.5);
            let radiance = (1.0 + a * exp(b * horizon))
                         * (c + d * exp(e * gamma) + f * rayleigh + g * mie + h * zenith)
                         * ds.data[{int(ds.sky_radiance + ch)}];

            rgb += radiance * ds.rgb_at(ds.rgb_weights + ch * 3);
        }
        // a saturated sky can fall outside the sRGB gamut and go slightly negative
        return max(rgb, v3<float>(0.0));
    }

    /* Radiance of the solar disc along a direction known to fall inside it. */
    public v3<float> sun_radiance(Dataset ds, v3<float> direction) {
        if (!ds.is_valid()) return v3<float>(0.0);

        let cos_gamma = clamp(dot(direction, ds.sun_direction()), -1.0, 1.0);
        let sin2_gamma = max(1.0 - cos_gamma * cos_gamma, 0.0);
        // position across the disc, as its cosine: one at the centre, zero at the limb
        let position = sqrt(max(1.0 - ds.disc_scale() * sin2_gamma, 0.0));

        v3<float> rgb = v3<float>(0.0);
        [ForceUnroll]
        for (int k = int(LIMB_DARKENING_ORDER) - 1; k >= 0; --k)
            rgb = rgb * position + ds.rgb_at(ds.limb_darkening + uint(k) * 3);
        return rgb;
    }
}
