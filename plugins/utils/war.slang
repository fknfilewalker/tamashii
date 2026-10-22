implementing tamashii;

namespace war {
    [PreferRecompute]
    Real compute_boundary_term<Real : floating_point, int N>(
        vector<Real, N> normal,
        vector<Real, N> direction)
    {
        Real dotND = dot(normal, direction);
        return dotND * dotND;
    }

    [__NoSideEffect, PreferRecompute, ForwardDerivative(fwd_compute_harmonic_weight)]
    float compute_harmonic_weight(
        no_diff SurfaceInteraction<float> si,
        no_diff float3 origin,
        no_diff float3 auxDirection,
        no_diff float auxSampleY,
        no_diff float kappa,
        float3 direction)
    {
        float boundaryTerm = no_diff compute_boundary_term(
            si.sp.frame.normal(), auxDirection);
        float sy = max(1.0f - auxSampleY, 1e-6f);
        float invVMFDensity =
            1.0f / ((1.0f - sy) * exp(-2.0f * kappa) + sy);
        float weightDenominator = invVMFDensity - 1.0f + boundaryTerm;
        float invWeightDenominator = select(
            weightDenominator > 1e-4f, 1.0f / weightDenominator, 0.0f);
        return invWeightDenominator * invWeightDenominator *
            invWeightDenominator * invVMFDensity;
    }

    [Differentiable, __NoSideEffect, PreferRecompute]
    DifferentialPair<float> fwd_compute_harmonic_weight(
        SurfaceInteraction<float> si,
        float3 origin,
        float3 auxDirection,
        float auxSampleY,
        float kappa,
        DifferentialPair<float3> direction)
    {
        float boundaryTerm = no_diff compute_boundary_term(
            si.sp.frame.normal(), auxDirection);
        float sy = max(1.0f - auxSampleY, 1e-6f);
        float invVMFDensity =
            1.0f / ((1.0f - sy) * exp(-2.0f * kappa) + sy);
        float weightDenominator = invVMFDensity - 1.0f + boundaryTerm;
        float invWeightDenominator = select(
            weightDenominator > 1e-4f, 1.0f / weightDenominator, 0.0f);
        float weight = invWeightDenominator * invWeightDenominator *
            invWeightDenominator * invVMFDensity;

        float derivativeScale =
            invVMFDensity * weight * invWeightDenominator * kappa * 3.0f;
        float3 tangentDerivative =
            -cross(direction.p, cross(direction.p, direction.d));
        float weightDerivative =
            derivativeScale * dot(tangentDerivative, auxDirection);
        return DifferentialPair<float>(
            weight, clamp(weightDerivative, -1e10f, 1e10f));
    }

    [Differentiable]
    float compute_harmonic_weight_directional_derivative(
        no_diff SurfaceInteraction<float> si,
        no_diff float3 auxDirection,
        no_diff float auxSampleY,
        no_diff float kappa,
        float3 direction,
        no_diff float3 directionDerivative)
    {
        float boundaryTerm = no_diff compute_boundary_term(
            si.sp.frame.normal(), auxDirection);
        float sy = max(1.0f - auxSampleY, 1e-6f);
        float invVMFDensity =
            1.0f / ((1.0f - sy) * exp(-2.0f * kappa) + sy);
        float weightDenominator = invVMFDensity - 1.0f + boundaryTerm;
        float invWeightDenominator = select(
            weightDenominator > 1e-4f, 1.0f / weightDenominator, 0.0f);
        float weight = invWeightDenominator * invWeightDenominator *
            invWeightDenominator * invVMFDensity;

        float derivativeScale = invVMFDensity * weight *
            invWeightDenominator * kappa * 3.0f;
        float3 tangentDerivative =
            -cross(direction, cross(direction, directionDerivative));
        return clamp(
            derivativeScale * dot(tangentDerivative, auxDirection),
            -1e10f,
            1e10f);
    }

    void sample_auxiliary_stream<uint WarpAuxSampleCount>(
        no_diff IRandom rng,
        no_diff uint stream,
        out float2 auxSamples[WarpAuxSampleCount])
    {
        // Fork a deterministic substream without advancing the path sampler.
        IRandom seedRng = rng;
        uint64_t auxSeed = random::TEA::hash(
            seedRng.next_uint() ^ (0xa511e9b3u + stream * 0x9e3779b9u),
            seedRng.next_uint() ^ (0x63d83595u + stream * 0x85ebca6bu));
        IRandom auxRng = random::PCG32(
            auxSeed,
            auxSeed ^ 0x9e3779b97f4a7c15ULL ^ uint64_t(stream));

        auxSamples = {};
        [ForceUnroll]
        for (uint i = 0; i < WarpAuxSampleCount; i++) {
            if ((i & 1u) != 0u)
                auxSamples[i] = auxSamples[i - 1u];
            else
                auxSamples[i] = auxRng.next_float2();
        }
    }

    [Differentiable]
    void compute_warped_intersection_direction<uint WarpAuxSampleCount>(
        no_diff IScene scene,
        float3 origin,
        no_diff float tmin,
        no_diff float tmax,
        no_diff float3 baseDirection,
        float3 direction,
        no_diff float kappa,
        no_diff float2 auxSamples[WarpAuxSampleCount],
        out float3 warpedDirection,
        out float3 warpedDirectionDerivative[3])
    {
        float totalWeight = 0.0f;
        float3 totalDirection = float3(0.0f);
        float totalWeightDerivative[3] = { 0.0f, 0.0f, 0.0f };
        float3 totalDirectionDerivative[3] = {
            float3(0.0f), float3(0.0f), float3(0.0f)
        };
        const float3 basis[3] = {
            float3(1.0f, 0.0f, 0.0f),
            float3(0.0f, 1.0f, 0.0f),
            float3(0.0f, 0.0f, 1.0f)
        };

        // Keep the sampling construction independent of the pathwise ray
        // derivative. The attached auxiliary hits carry the geometry derivative.
        Frame<float> baseFrame = detach(Frame<float>(baseDirection));

        [MaxIters(WarpAuxSampleCount)]
        for (uint i = 0; i < WarpAuxSampleCount; i++) {
            float2 sample = auxSamples[i];
            float3 localDirection = no_diff warp::square_to_von_mises_fisher(
                sample, kappa);
            float3 auxDirection = detach(
                normalize(baseFrame.to_world(localDirection)));

            if ((i & 1u) != 0u)
                auxDirection = detach(reflect(-auxDirection, baseDirection));

            Ray<float> auxRay = { origin, tmin, auxDirection, tmax };
            HitInfo auxHit = scene.intersect(
                auxRay, RAY_FLAG_NONE, HitMask::all);

            SurfaceInteraction<float> auxSi = {{}};
            float3 attachedDirection = auxDirection;
            if (!auxHit.instance.is_none()) {
                IShape<float> auxShape = scene.shapes[auxHit.instance.index()];
                auxSi = auxShape.surface_interaction(
                    auxRay, auxHit.primitive, auxHit.attribute, true);

                float3 offset = auxSi.sp.position - origin;
                float offsetLength2 = length2(offset);
                if (offsetLength2 > 1e-12f)
                    attachedDirection = offset * rsqrt(offsetLength2);
            } else {
                // A missed ray has the asymptotic boundary term B = 1.
                auxSi.sp.frame = Frame<float>(auxDirection);
            }

            float weight = no_diff compute_harmonic_weight(
                auxSi,
                origin,
                auxDirection,
                sample.y,
                kappa,
                baseDirection);
            if (!(no_diff is_finite(weight)) ||
                !(no_diff is_finite(attachedDirection)))
                continue;

            totalWeight += weight;
            totalDirection += weight * attachedDirection;

            // Compute the inner directional Jacobian explicitly so the outer
            // geometry AD remains independent of the ray-query derivative.
            [ForceUnroll]
            for (int j = 0; j < 3; j++) {
                float weightDerivative =
                    compute_harmonic_weight_directional_derivative(
                        auxSi,
                        auxDirection,
                        sample.y,
                        kappa,
                        direction,
                        basis[j]);
                if (!(no_diff is_finite(weightDerivative)))
                    weightDerivative = 0.0f;
                totalWeightDerivative[j] += weightDerivative;
                totalDirectionDerivative[j] +=
                    weightDerivative * attachedDirection;
            }
        }

        if (!(no_diff is_finite(totalWeight)) ||
            !(no_diff is_finite(totalDirection)) ||
            totalWeight <= 1e-8f || length2(totalDirection) <= 1e-16f) {
            warpedDirection = baseDirection;
            [ForceUnroll]
            for (int j = 0; j < 3; j++)
                warpedDirectionDerivative[j] = basis[j];
            return;
        }

        float invTotalWeight = 1.0f / totalWeight;
        float3 meanDirection = totalDirection * invTotalWeight;
        float invMeanLength = rsqrt(length2(meanDirection));
        warpedDirection = meanDirection * invMeanLength;

        [ForceUnroll]
        for (int j = 0; j < 3; j++) {
            float3 meanDerivative =
                (totalDirectionDerivative[j] -
                    meanDirection * totalWeightDerivative[j]) *
                invTotalWeight;
            float3 directionDerivative =
                (meanDerivative -
                    warpedDirection * dot(warpedDirection, meanDerivative)) *
                invMeanLength;
            warpedDirectionDerivative[j] = no_diff is_finite(directionDerivative)
                ? directionDerivative
                : basis[j];
        }
    }

    [Differentiable]
    float reparameterize_intersection_ray<uint WarpAuxSampleCount>(
        no_diff IScene scene,
        inout Ray<float> ray,
        no_diff float kappa,
        no_diff float2 auxSamples[WarpAuxSampleCount])
    {
        float3 direction = normalize(ray.direction);
        float3 baseDirection = detach(direction);
        float3 warpedDirection;
        float3 warpedDirectionDerivative[3];
        compute_warped_intersection_direction(
            scene,
            ray.origin,
            detach(ray.tmin),
            detach(ray.tmax),
            baseDirection,
            direction,
            kappa,
            auxSamples,
            warpedDirection,
            warpedDirectionDerivative);

        float3 warp = infinitesimal(warpedDirection);
        float3 warpDerivativeX = infinitesimal(warpedDirectionDerivative[0]);
        float3 warpDerivativeY = infinitesimal(warpedDirectionDerivative[1]);
        float3 warpDerivativeZ = infinitesimal(warpedDirectionDerivative[2]);
        m3x3<float> jacobian = m3x3<float>(
            float3(1.0f, 0.0f, 0.0f) + warpDerivativeX,
            float3(0.0f, 1.0f, 0.0f) + warpDerivativeY,
            float3(0.0f, 0.0f, 1.0f) + warpDerivativeZ);

        ray.direction += warp;
        return determinant(jacobian);
    }

    // The ordinary renderer must remain bit-for-bit unchanged. Slang replaces
    // this empty hook with the differentiable implementation only for AD calls.
    void handle_intersection_discontinuity(
        no_diff IScene scene,
        inout RaySample<float> rs,
        no_diff IRandom rng,
        no_diff uint depth,
        no_diff bool rayDelta)
    {}

    [Differentiable, PrimalSubstituteOf(handle_intersection_discontinuity)]
    void sub_handle_intersection_discontinuity(
        no_diff IScene scene,
        inout RaySample<float> rs,
        no_diff IRandom rng,
        no_diff uint depth,
        no_diff bool rayDelta)
    {
        // Primary rays use the 2D film measure, while delta rays have no
        // continuous solid-angle sampling measure to reparameterize.
        if (depth == 1u || rayDelta) return;

        const uint WarpAuxSampleCount = 4u;
        const float kappa = 1e4f;
        float2 auxSamples[WarpAuxSampleCount];
        war::sample_auxiliary_stream(rng, depth, auxSamples);
        float detJ = war::reparameterize_intersection_ray(
            scene, rs.ray, kappa, auxSamples);
        rs.weight *= detJ;
    }
}
