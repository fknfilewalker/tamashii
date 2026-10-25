implementing tamashii;

namespace colormap::diverging {
    public struct Fast : IColormap {
        public __init() {
        }

        // "Fast" from "A New Default Colormap for ParaView" by Francesca Samsel, W. Alan Scott and
        // Kenneth Moreland, IEEE CG&A 44(4), 2024. Interpolated in CIE Lab like the ParaView preset.
        // scalar, red, green, blue
        static const int colormap_size = 9;
        static const float4 colormap[colormap_size] = {
            { 0, 0.0564, 0.0564, 0.47 },
            { 0.171592239, 0.243, 0.46035, 0.81 },
            { 0.298491482, 0.356814383, 0.745024649, 0.954367703 },
            { 0.432128737, 0.6882, 0.93, 0.91791 },
            { 0.5, 0.899495955, 0.944646395, 0.768656714 },
            { 0.588226035, 0.957107977, 0.833818511, 0.50891563 },
            { 0.706141261, 0.92752076, 0.621438909, 0.315357058 },
            { 0.847639531, 0.8, 0.352, 0.16 },
            { 1, 0.59, 0.0767, 0.119475 }
        };

        public float3 eval(float t) {
            // [-1,1] -> [0,1]
            t = t * 0.5f + 0.5f;
            if (t <= colormap[0].x) return colormap[0].yzw;
            for (int i = 1; i < colormap_size; i++) {
                const float4 b = colormap[i];
                if (t <= b.x) {
                    const float4 a = colormap[i - 1];
                    const float u = (t - a.x) / (b.x - a.x);
                    let lab_a = post::xyz_to_lab(post::linear_srgb_to_xyz(post::srgb_to_linear(a.yzw)));
                    let lab_b = post::xyz_to_lab(post::linear_srgb_to_xyz(post::srgb_to_linear(b.yzw)));
                    let lab = lerp(lab_a, lab_b, u);
                    return post::linear_to_srgb(saturate(post::xyz_to_linear_srgb(post::lab_to_xyz(lab))));
                }
            }
            return colormap[colormap_size - 1].yzw;
        }
    };
};
