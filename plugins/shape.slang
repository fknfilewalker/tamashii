implementing tamashii;

[anyValueSize(settings::shape_interface_size)]
public dyn interface IShape<Real : atomicable_floating_point> {

    public property DiffTensor<m4x4<Real>, 1> to_world;
    public property AABB<Real> aabb;
    public property IBsdf<Real>* bsdf;
    public property IFilm<Real, 4>* film; // optional
    public property v3<Real> emission;

    /* Number of primitives in this shape */
    public uint primitive_count();
    /* Returns the surface point for the given primitive and attribute */
    [Differentiable] public SurfacePoint<Real> surface_point(uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity());
    /* Returns the surface interaction for the given ray, primitive and attribute */
    /* When differentiating, a software ray intersection is used instead for the differential computation (intersection must be guaranteed). */
    /* If attach_to_geometry is true, the differential hit uses fixed attributes. */
    [Differentiable] public SurfaceInteraction<Real> surface_interaction(Ray<Real> ray, uint32_t primitive, no_diff v2<Real> attribute, bool attach_to_geometry = false, Transform<Real> transform = Transform<Real>::identity());
    /* Returns the surface point for the given primitive and ray (intersection must be guaranteed). */
    [Differentiable] public SurfaceInteraction<Real> surface_intersection(Ray<Real> ray, uint32_t primitive, bool attach_to_geometry, Transform<Real> transform = Transform<Real>::identity());
    /* Returns a sample on the surface */
    [Differentiable] public SurfaceSample<Real> sample_surface(v3<Real> uv, Transform<Real> transform = Transform<Real>::identity());
    /* Returns a direction sample to the shape from the given surface point */
    [Differentiable] public DirectionSample<Real> sample_direction(SurfaceInteraction<Real> si, v3<Real> uv, Transform<Real> transform = Transform<Real>::identity());
    /* Returns the pdf for the given direction sample */
    [Differentiable] public Real pdf_direction(SurfaceInteraction<Real> si, DirectionSample<Real> ds);

    [Differentiable] public void store(vector<Real, 4> value, Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity());
    [Differentiable] public vector<Real, 4> load(Ray<Real> ray, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity());

    public uint intersect(inout Ray<Real> ray, const uint cullmode /*none, front, back*/,
                              inout v2<Real> barycentric, inout uint primitive_idx, inout uint geometry_idx);
};

namespace shapes {
    [Differentiable]
    public SurfacePoint<Real> surface_point<Real : atomicable_floating_point>(IShape<Real> shape, uint32_t primitive, v2<Real> attribute, Transform<Real> transform = Transform<Real>::identity()) {
        return shape.surface_point(primitive, attribute, transform);
    }
    [Differentiable]
    public SurfaceInteraction<Real> surface_intersection<Real : atomicable_floating_point>(IShape<Real> shape, Ray<Real> ray, uint32_t primitive, bool attach_to_geometry, Transform<Real> transform = Transform<Real>::identity()) {
        return shape.surface_intersection(ray, primitive, attach_to_geometry, transform);
    }
}
