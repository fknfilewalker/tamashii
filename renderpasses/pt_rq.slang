import tamashii;

struct Settings<I : IIntegrator> {
    uint seed;
    uint spp;
    I integrator;
};

[shader("compute"), numthreads(8, 8, 1)]
void primal<I : IIntegrator>(uint3 tid: SV_DispatchThreadID, uniform RayQueryScene<float> scene, uniform Settings<I> settings)
{
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(settings.seed));
    for (int i = 0; i < settings.spp; i++)
        settings.integrator.sample(tid.xy, scene, rng);
}

[shader("compute"), numthreads(8, 8, 1)]
void forward<I : IIntegrator>(uint3 tid: SV_DispatchThreadID, uniform RayQueryScene<float> scene, uniform Settings<I> settings)
{
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(settings.seed));
    for (int i = 0; i < settings.spp; i++)
        __fwd_diff(integrators::sample)(settings.integrator, tid.xy, scene, rng);
}

[shader("compute"), numthreads(8, 8, 1)]
void backward<I : IIntegrator>(uint3 tid: SV_DispatchThreadID, uniform RayQueryScene<float> scene, uniform Settings<I> settings)
{
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(settings.seed));
    for (int i = 0; i < settings.spp; i++)
        __bwd_diff(integrators::sample)(settings.integrator, tid.xy, scene, rng);
}
