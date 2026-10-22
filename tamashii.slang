module tamashii;

public struct settings {
    public static const uint32_t interface_header_size = 16;
    public static const uint32_t shape_interface_size = 400;
    public static const uint32_t emitter_interface_size = 256;
    public static const uint32_t sensor_interface_size = 400;
    public static const uint32_t rfilter_interface_size = 32;
    public static const uint32_t bsdf_interface_size = 192;
    public static const uint32_t integrator_interface_size = 64;
    public static const uint32_t film_interface_size = 112;
};

__include "types.slang";
__include "tensor.slang";
__include "random.slang";

// Lightweight module variant for basic types
#ifndef TAMASHII_CORE_ONLY
__include "utils/linear_lights.slang";
__include "utils/ltc.slang";
__include "utils/war.slang";
__include "utils/hosek_wilkie.slang";

__include "bsdf.slang";
__include "bsdfs/diffuse.slang";
__include "bsdfs/dielectric.slang";
__include "bsdfs/plastic.slang";
__include "bsdfs/conductor.slang";

__include "rfilter.slang";
__include "rfilters/box.slang";
__include "rfilters/tent.slang";

__include "shape.slang";
__include "shapes/mesh.slang";
__include "shapes/sphere.slang";
__include "shapes/rectangle.slang";

__include "emitter.slang";
__include "emitters/point.slang";
__include "emitters/area.slang";
__include "emitters/constant.slang";
__include "emitters/hosek_wilkie.slang";

__include "sensor.slang";
__include "sensors/perspective.slang";
__include "sensors/area.slang";

__include "integrator.slang";
__include "integrators/path.slang";
__include "integrators/ptracer.slang";
__include "integrators/aov.slang";

__include "colormap.slang";
__include "colormaps/coolwarm.slang";

__include "film.slang";
__include "films/hdr_texture_2d.slang";
__include "films/sh.slang";
#endif
