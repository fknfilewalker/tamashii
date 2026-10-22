// implementing tamashii;

// namespace shapes {
//     public struct Instance : IShape, IDifferentiable {
//         public Transform<float> to_world;
//         public IBsdf<float>* bsdf;
//         public IShape* instance_shape;

//         public uint primitive_count() { return 1; }

//         [Differentiable] public SurfacePoint<float> surface_point(uint32_t primitive, no_diff float2 attribute, Transform<float> transform) {
            
//         }

//         [Differentiable] public SurfaceInteraction<float> surface_interaction(Ray<float> ray, uint32_t primitive, no_diff float2 attribute, Transform<float> transform = Transform<float>::identity()) {
            
//         }

//         [Differentiable] public SurfaceInteraction<float> surface_intersection(Ray<float> ray, uint32_t primitive, Transform<float> transform = Transform<float>::identity()) {
//             SurfaceInteraction<float> si;
//             return si;
//         }

//         public SurfaceSample<float> sample_surface(v3<float> uv, Transform<float> transform) {
            
//         }

//         [Differentiable] public DirectionSample<float> sample_direction(SurfaceInteraction<float> si, v3<float> uv, Transform<float> transform) {
            
//         }

//         // public SurfaceSample sample_surface(v2<float> uv, Transform<float> transform) {
//         //     SurfaceSample ss;
//         //     ss.sp = surface_point(0, uv, transform);
//         //     ss.pdf = warp::square_to_uniform_sphere_pdf();
//         //     return ss;
//         // }

//         public float pdf_surface(Transform<float> transform) { 
            
//         }

//         public static Tuple<bool /*hit*/, float /*t*/, v2<float>/*attr*/> intersect_os(
//             v3<float> origin, v3<float> direction, float t_min
//         ) {
            
//         }

//         // public bool intersect(REF(RayDesc, ray), const CullMode cullmode,
//         //         REF(v2<float>, barycentric), REF(uint64_t, primitive_idx), REF(uint64_t, geometry_idx)) {

//         //     v3<float> ray_origin_os = mul(v4<float>(ray.Origin, 1), to_world.inverse).xyz;
//         //     v3<float> ray_direction_os = mul(v4<float>(ray.Direction, 0), to_world.inverse).xyz;
//         //     Tuple<bool, float, v2<float>> result = intersect_os(ray_origin_os, ray_direction_os, ray.TMin);
//         //     if (result._0 && result._1 < ray.TMax) {
//         //         ray.TMax = result._1;
//         //         barycentric = result._2;
//         //         return true;
//         //     }
//         //     return false;
//         // }
//     };
// }