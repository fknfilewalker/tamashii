import tamashii;

RayQueryScene _scene;
struct Settings {
    uint seed;
    uint spp;
    uint mode;
    uint aov_mode;
    uint max_depth;
    uint rr_depth;
    bool nee;
    MIS mis;
} _settings;

[shader("compute"), numthreads(8, 8, 1)]
void primal(uint3 tid: SV_DispatchThreadID)
{
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(_settings.seed));
    for (int i = 0; i < _settings.spp; i++) {
        if (_settings.mode == 0)
            integrators::Path(_settings.max_depth, _settings.rr_depth, _settings.nee, _settings.mis).sample(tid.xy, _scene, rng);
        if (_settings.mode == 1)
            integrators::Ptracer(_settings.max_depth, _settings.rr_depth).sample(tid.xy, _scene, rng);
        if (_settings.mode == 2)
            integrators::AOV(integrators::AOV::Mode(_settings.aov_mode)).sample(tid.xy, _scene, rng);
    }
}

[shader("compute"), numthreads(8, 8, 1)]
void forward(uint3 tid: SV_DispatchThreadID)
{
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(_settings.seed));
    for (int i = 0; i < _settings.spp; i++) {
        if (_settings.mode == 0)
            __fwd_diff(integrators::sample)(integrators::Path(_settings.max_depth, _settings.rr_depth, _settings.nee, _settings.mis), tid.xy, _scene, rng);
        if (_settings.mode == 1)
            __fwd_diff(integrators::sample)(integrators::Ptracer(_settings.max_depth, _settings.rr_depth), tid.xy, _scene, rng);
    }
}

[shader("compute"), numthreads(8, 8, 1)]
void backward(uint3 tid: SV_DispatchThreadID)
{
    IRandom rng = random::PCG32(random::TEA::hash(tid.x, tid.y), uint64_t(_settings.seed));
    for (int i = 0; i < _settings.spp; i++) {
        if (_settings.mode == 0)
            __bwd_diff(integrators::sample)(integrators::Path(_settings.max_depth, _settings.rr_depth, _settings.nee, _settings.mis), tid.xy, _scene, rng);
        if (_settings.mode == 1)
            __bwd_diff(integrators::sample)(integrators::Ptracer(_settings.max_depth, _settings.rr_depth), tid.xy, _scene, rng);
    }
}
