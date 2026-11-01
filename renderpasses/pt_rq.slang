import tamashii;

struct Settings<I : IIntegrator> {
    uint seed;
    uint spp;
    I integrator;
};

[shader("compute"), numthreads(8, 8, 1)]
void primal<I : IIntegrator>(uint3 tid: SV_DispatchThreadID, uniform RayQueryScene<float> scene, uniform Settings<I> settings)
{
    for (int i = 0; i < settings.spp; i++) {
        IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), random::TEA::hash(settings.seed, uint(i)));
        settings.integrator.sample(tid.xy, scene, rng);
    }
}

[shader("compute"), numthreads(8, 8, 1)]
void forward<I : IIntegrator>(uint3 tid: SV_DispatchThreadID, uniform RayQueryScene<float> scene, uniform Settings<I> settings)
{
    for (int i = 0; i < settings.spp; i++) {
        IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), random::TEA::hash(settings.seed, uint(i)));
        __fwd_diff(integrators::sample)(settings.integrator, tid.xy, scene, rng);
    }
}

[shader("compute"), numthreads(8, 8, 1)]
void backward<I : IIntegrator>(uint3 tid: SV_DispatchThreadID, uniform RayQueryScene<float> scene, uniform Settings<I> settings)
{
    for (int i = 0; i < settings.spp; i++) {
        // A fresh stream per sample: __bwd_diff passes a no_diff inout rng as in, so it
        // would never advance. Seeding all entry points the same way keeps their samples equal.
        IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), random::TEA::hash(settings.seed, uint(i)));
        __bwd_diff(integrators::sample)(settings.integrator, tid.xy, scene, rng);
    }
}
