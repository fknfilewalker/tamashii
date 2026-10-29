import tamashii;

struct PipelineScene<Real : atomicable_floating_point> : IScene<Real> {
    IShape<Real> *shapes;
    IEmitter<Real> *emitters;
    ISensor<Real> *sensors;
    RaytracingAccelerationStructure as;
    IEmitter<Real> *environment_emitter;
    int sensor_count;
    int shape_count;
    int emitter_count;

    HitInfo intersect(Ray<Real> ray, uint32_t flags, uint32_t mask) {
        HitInfo payload = { {}, InstanceIndex(InstanceIndex.miss), {}, {} };
        TraceRay(as, flags, mask, 0, 1, 0, ray.desc(), payload);
        return payload;
    }

    bool unoccluded(Ray<Real> ray) {
        HitInfo payload = { {}, InstanceIndex(0), {}, {} };
        TraceRay(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, HitMask::all, 0, 1, 0, ray.desc(), payload);
        return payload.instance.is_none();
    }
};

struct Settings<I : IIntegrator> {
    uint seed;
    uint spp;
    I integrator;
};

[shader("miss")]
void miss(inout HitInfo payload) {
    payload.instance = InstanceIndex(InstanceIndex.miss);
}

[shader("closesthit")]
void triangle_closest_hit(inout HitInfo payload, BuiltInTriangleIntersectionAttributes attribs) {
    payload.attribute = attribs.barycentrics;
    payload.instance = InstanceIndex(InstanceID());
    payload.primitive = PrimitiveIndex();
    payload.t = RayTCurrent();
}

[shader("closesthit")]
void sphere_closest_hit(inout HitInfo payload, float2 attribs) {
    payload.attribute = attribs;
    payload.instance = InstanceIndex(InstanceID());
    payload.primitive = PrimitiveIndex();
    payload.t = RayTCurrent();
}

[shader("intersection")]
void sphere_intersection() {
    float3 origin = ObjectRayOrigin();
    float3 dir = ObjectRayDirection();
    float t_min = RayTMin();

    Tuple<bool, float, v2<float>> hit = shapes::Sphere<float>::intersect_os(origin, dir, t_min);
    if (hit._0) {
        ReportHitOptix(hit._1, 0, bit_cast<uint64_t>(hit._2));
    }
}

[shader("raygeneration")]
void primal<I : IIntegrator>(uniform PipelineScene<float> scene, uniform Settings<I> settings) {
    uint2 tid = DispatchRaysIndex().xy;
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(settings.seed));
    for (int i = 0; i < settings.spp; i++)
        settings.integrator.sample(tid, scene, rng);
}

[shader("raygeneration")]
void forward<I : IIntegrator>(uniform PipelineScene<float> scene, uniform Settings<I> settings) {
    uint2 tid = DispatchRaysIndex().xy;
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(settings.seed));
    for (int i = 0; i < settings.spp; i++)
        __fwd_diff(integrators::sample)(settings.integrator, tid, scene, rng);
}

[shader("raygeneration")]
void backward<I : IIntegrator>(uniform PipelineScene<float> scene, uniform Settings<I> settings) {
    uint2 tid = DispatchRaysIndex().xy;
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(settings.seed));
    for (int i = 0; i < settings.spp; i++)
        __bwd_diff(integrators::sample)(settings.integrator, tid, scene, rng);
}
