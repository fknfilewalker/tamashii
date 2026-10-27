implementing tamashii;

public interface IScene {
    public property IShape<float> *shapes;
    public property IEmitter<float> *emitters;
    public property ISensor<float> *sensors;
    public property RaytracingAccelerationStructure as;
    public property IEmitter<float> *environment_emitter; // pointer to environment emitter, or null
    public property int sensor_count;
    public property int shape_count;
    public property int emitter_count;

    [PreferCheckpoint] public HitInfo intersect(Ray<float> ray, uint32_t flags, uint32_t mask);
    [PreferCheckpoint] public bool unoccluded(Ray<float> ray);
};

public struct RayQueryScene : IScene {
    public IShape<float> *shapes;
    public IEmitter<float> *emitters;
    public ISensor<float> *sensors;
    public RaytracingAccelerationStructure as;
    public IEmitter<float> *environment_emitter;
    public int sensor_count;
    public int shape_count;
    public int emitter_count;

    [require(glsl_hlsl_metal_spirv, rayquery)]
    public HitInfo intersect(Ray<float> ray, uint32_t flags, uint32_t mask) {
        HitInfo hit_info = { {}, InstanceIndex(InstanceIndex.miss), {}, {} };
        RayQuery<0> rayQuery;
        rayQuery.TraceRayInline(as, flags, mask, ray.desc());
        v2<float> attr = process_hit_candidates(rayQuery);

        uint status = rayQuery.CommittedStatus();
        if (status == COMMITTED_NOTHING) return hit_info;  
        hit_info.instance = InstanceIndex(rayQuery.CommittedInstanceID());
        hit_info.t = rayQuery.CommittedRayT();
        if (status == COMMITTED_TRIANGLE_HIT) {
            hit_info.primitive = rayQuery.CommittedPrimitiveIndex();
            hit_info.attribute = rayQuery.CommittedTriangleBarycentrics();
        } else if (status == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
            hit_info.attribute = attr;
        }
        return hit_info;
    }

    [require(glsl_metal_spirv, rayquery)]
    public bool unoccluded(Ray<float> ray) {
        RayQuery<0> rayQuery;
        rayQuery.TraceRayInline(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, HitMask::all, ray.desc());
        process_hit_candidates(rayQuery);
        return rayQuery.CommittedStatus() == COMMITTED_NOTHING;
    }

    [require(glsl_metal_spirv, rayquery)]
    public v2<float> process_hit_candidates<let T : uint>(inout RayQuery<T> rayQuery) {
        v2<float> attr = v2<float>(.0f);
        while (rayQuery.Proceed()) {
            if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                if (rayQuery.CandidateTriangleRayT() < rayQuery.CommittedRayT()) {
                    attr = rayQuery.CandidateTriangleBarycentrics();
                    rayQuery.CommitNonOpaqueTriangleHit();
                }
            } else if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                if (this.shapes[rayQuery.CandidateInstanceID()] is shapes::Sphere<float>) {
                    Tuple<bool, float, v2<float>> hit = shapes::Sphere<float>::intersect_os(rayQuery.CandidateRayObjectRayOrigin(), rayQuery.CandidateRayObjectRayDirection(), rayQuery.RayTMin());
                    if (hit._0 && hit._1 < rayQuery.CommittedRayT()) {
                        attr = hit._2;
                        rayQuery.CommitProceduralPrimitiveHit(hit._1);
                    }
                }
                else rayQuery.CommitProceduralPrimitiveHit(rayQuery.RayTMin());
            }
        }
        return attr;
    }
};

[anyValueSize(settings::integrator_interface_size)]
public dyn interface IIntegrator {
    [Differentiable] public void sample(uint2 id, IScene scene, inout IRandom rng);
};

namespace integrators {
    [Differentiable]
    public void sample(IIntegrator integrator, uint2 id, IScene scene, inout IRandom rng) {
        integrator.sample(id, scene, rng);
    }
}
