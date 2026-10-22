implementing tamashii;

namespace ltc {
    __generic<int SIZE = 64>
    float3x3 get_matrix(float3x3 *data, const float theta, const float alpha)
    {
        const int t = max<int>(0, min<int>(SIZE - 1, (int)floor(theta / constants<float>::half_pi * SIZE)));
        const int a = max<int>(0, min<int>(SIZE - 1, (int)floor(sqrt(alpha) * SIZE)));
        return data[a + t * SIZE];
    }
    __generic<int SIZE = 64>
    float get_magnitude(float *data, const float theta, const float alpha)
    {
        const int t = max<int>(0, min<int>(SIZE - 1, (int)floor(theta / constants<float>::half_pi * SIZE)));
        const int a = max<int>(0, min<int>(SIZE - 1, (int)floor(sqrt(alpha) * SIZE)));
        return data[a + t * SIZE];
    }
}
