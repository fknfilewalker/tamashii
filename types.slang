implementing tamashii;

public struct ConformanceWrapper<T> {
    uint _ = 1;
    uint __;
    uint id;
    uint ___;
    T data;
};

enum class TextureWrap {
    Repeat = 0,
    Clamp = 1
}

public struct Texture<T : IArithmetic> {

    public T factor;
    public no_diff Tensor<T, 2> texture; // (rows, cols)

    public __init(T factor) {
        this.factor = factor;
        this.texture = Tensor<T, 2>();
    }

    public property width : uint { get { return texture.shape[1]; } }
    public property height : uint { get { return texture.shape[0]; } }

    public bool has_texture() {
        return texture.is_valid() && width > 0 && height > 0;
    }

    public vector<U, 2> wrap<U : floating_point>(vector<U, 2> uv) {
        TextureWrap wrap_mode = TextureWrap::Repeat;
        if (wrap_mode == TextureWrap::Repeat) {
            return uv - floor(uv);
        }
        return clamp(uv, U(0), U(1));
    }

    public T fetch(uint x, uint y) {
        if (!has_texture()) return T(0);
        return texture[{int(y % height), int(x % width)}];
    }
};

public extension<T : __BuiltinFloatingPointType> Texture<T>
{
    [TreatAsDifferentiable]
    public T sample(vector<T, 2> uv) {
        if (!has_texture()) return factor;
        uv = wrap(uv);
        let fxy = uv * vector<T, 2>(T(width), T(height)) - vector<T, 2>(T(0.5));
        let flxy = floor(fxy);
        uint2 xy0 = uint2(max(flxy, T(0.0f)));
        uint2 xy1 = xy0 + 1;
        let sxy = fxy - flxy;
        let v01 = lerp(fetch(xy0.x, xy0.y), fetch(xy1.x, xy0.y), T(sxy.x));
        let v23 = lerp(fetch(xy0.x, xy1.y), fetch(xy1.x, xy1.y), T(sxy.x));
        return factor * lerp(v01, v23, T(sxy.y));
    }
};

__generic<T : __BuiltinFloatingPointType, let N : int>
public extension Texture<vector<T, N>>
{
    [TreatAsDifferentiable]
    public vector<T, N> sample(vector<T, 2> uv) {
        if (!has_texture()) return factor;
        uv = wrap(uv);
        let fxy = uv * vector<T, 2>(T(width), T(height)) - vector<T, 2>(T(0.5));
        let flxy = floor(fxy);
        uint2 xy0 = uint2(max(flxy, T(0.0f)));
        uint2 xy1 = xy0 + 1;
        let sxy = fxy - flxy;
        let v01 = lerp(fetch(xy0.x, xy0.y), fetch(xy1.x, xy0.y), T(sxy.x));
        let v23 = lerp(fetch(xy0.x, xy1.y), fetch(xy1.x, xy1.y), T(sxy.x));
        return factor * lerp(v01, v23, T(sxy.y));
    }
};

public typealias integral = __BuiltinIntegerType;
public typealias floating_point = __BuiltinFloatingPointType;
public typealias atomicable_floating_point = __BuiltinFloatingPointType & IArithmeticAtomicable;
public typealias arithmetic = __BuiltinArithmeticType;

public typealias i32 = int;
public typealias u32 = uint32_t;
public typealias i64 = int64_t;
public typealias u64 = uint64_t;
public typealias f32 = float;
public typealias f64 = double;

public typealias v2<T : arithmetic> = vector<T, 2>;
public typealias v3<T : arithmetic> = vector<T, 3>;
public typealias v4<T : arithmetic> = vector<T, 4>;

public typealias i32v2 = v2<int32_t>;
public typealias i32v3 = v3<int32_t>;
public typealias i32v4 = v4<int32_t>;

public typealias u32v2 = v2<uint32_t>;
public typealias u32v3 = v3<uint32_t>;
public typealias u32v4 = v4<uint32_t>;

public typealias i64v2 = v2<int64_t>;
public typealias i64v3 = v3<int64_t>;
public typealias i64v4 = v4<int64_t>;

public typealias u64v2 = v2<uint64_t>;
public typealias u64v3 = v3<uint64_t>;
public typealias u64v4 = v4<uint64_t>;

public typealias f32v2 = v2<float>;
public typealias f32v3 = v3<float>;
public typealias f32v4 = v4<float>;

public typealias f64v2 = v2<double>;
public typealias f64v3 = v3<double>;
public typealias f64v4 = v4<double>;

public typealias m2x2<T : arithmetic> = matrix<T, 2, 2>;
public typealias m3x3<T : arithmetic> = matrix<T, 3, 3>;
public typealias m4x4<T : arithmetic> = matrix<T, 4, 4>;

public typealias f32m2x2 = m2x2<float>;
public typealias f32m3x3 = m3x3<float>;
public typealias f32m4x4 = m4x4<float>;

public typealias f64m2x2 = m2x2<double>;
public typealias f64m3x3 = m3x3<double>;
public typealias f64m4x4 = m4x4<double>;

public typealias Spectrum<T : floating_point> = v3<T>;
[ForceInline, PreferRecompute, Differentiable] public v4<T> Pos<T : floating_point>(T p = T(0)) { return v4<T>(p, p, p, T(1)); }
[ForceInline, PreferRecompute, Differentiable] public v4<T> Pos<T : floating_point>(T x, T y, T z) { return v4<T>(x, y, z, T(1)); }
[ForceInline, PreferRecompute, Differentiable] public v4<T> Pos<T : floating_point>(v3<T> p) { return v4<T>(p, T(1)); }
[ForceInline, PreferRecompute, Differentiable] public v4<T> Dir<T : floating_point>(T x, T y, T z) { return v4<T>(x, y, z, T(0)); }
[ForceInline, PreferRecompute, Differentiable] public v4<T> Dir<T : floating_point>(v3<T> d) { return v4<T>(d, T(0)); }

[ForceInline] public int4 Coord(int x) { return int4(x, 0, 0, 0); }
[ForceInline] public int4 Coord(int x, int y) { return int4(x, y, 0, 0); }
[ForceInline] public int4 Coord(int x, int y, int z) { return int4(x, y, z, 0); }
[ForceInline] public int4 Coord(int x, int y, int z, int w) { return int4(x, y, z, w); }
[ForceInline] public int4 Coord(int2 xy) { return int4(xy.x, xy.y, 0, 0); }

public struct Ray<Real : floating_point> : IDifferentiable {
    public vector<Real, 3> origin;
    public Real tmin;
    public vector<Real, 3> direction;
    public Real tmax;

    [Differentiable]
    public __init() {
        this.origin = vector<Real, 3>(0);
        this.tmin = Real(0.01);
        this.direction = vector<Real, 3>(Real(0), Real(0), Real(1));
        this.tmax = Real(1000);
    }

    [Differentiable]
    public __init(vector<Real, 3> origin, Real tmin, vector<Real, 3> direction, Real tmax) {
        this.origin = origin;
        this.tmin = tmin;
        this.direction = direction;
        this.tmax = tmax;
    }

    [Differentiable]
    public __init(vector<Real, 3> origin, vector<Real, 3> direction) {
        this.origin = origin;
        this.tmin = Real(0.01);
        this.direction = direction;
        this.tmax = Real(1000);
    }

    public RayDesc desc() {
        return { vector<float, 3>(origin), floatCast<float>(tmin), vector<float, 3>(direction), floatCast<float>(tmax) };
    }

    [Differentiable] public Ray<Real> transform(const m4x4<Real> m) {
        Ray<Real> r = this;
        r.origin = mul(v4<Real>(this.origin, Real(1)), m).xyz;
        r.direction = mul(v4<Real>(this.direction, Real(0)), m).xyz;
        return r;
    }
};

public struct DiffPtr<T> : IDifferentiablePtrType {
    typealias Differential = DiffPtr<T>;
    public T* ptr;

    public __subscript(int x) -> T
    {
        get { return ptr[x]; }
        set { ptr[x] = newValue; }
    }
};

__generic<T : floating_point, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public vector<T, C> operator *(vector<T, R> left, matrix<T, R, C> right) {
    return mul(left, right);
}
__generic<T : floating_point, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public vector<T, R> operator *(matrix<T, R, C> left, vector<T, C> right) {
    return mul(left, right);
}
__generic<T : floating_point, let R : int, let N : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public matrix<T, R, C> operator *(matrix<T, R, N> left, matrix<T, N, C> right) {
    return mul(left, right);
}

__generic<T : floating_point, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public vector<T, C> row(matrix<T, R, C> m, int i) {
    return m[i];
}
__generic<T : floating_point, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public void row(inout matrix<T, R, C> m, int i, vector<T, C> v) {
    m[i] = v;
}
__generic<T : floating_point, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public vector<T, R> column(matrix<T, R, C> m, int i) {
    vector<T, R> result;
    [ForceUnroll] for (int j = 0; j < R; j++) result[j] = m[j][i];
    return result;
}
__generic<T : floating_point, let R : int, let C : int>
[ForceInline, PreferRecompute, Differentiable]
public void column(inout matrix<T, R, C> m, int i, vector<T, R> v) {
    [ForceUnroll] for (int j = 0; j < R; j++) m[j][i] = v[j];
}

[ForceInline, PreferRecompute, Differentiable]
public Real safe_div<Real : floating_point>(Real x, Real y) {
    return y != Real(0) ? x / y : Real(0);
}

[ForceInline, PreferRecompute, Differentiable]
public vector<Real, N> safe_div<Real : floating_point, int N>(vector<Real, N> x, vector<Real, N> y) {
    vector<Real, N> result;
    [ForceUnroll] for (int i = 0; i < N; i++) result[i] = safe_div(x[i], y[i]);
    return result;
}

// acos outside of [-1, 1] is NaN, so we clamp to avoid that
// e.g, dot(a, b) could mathematically be 1.0, but computationally it results in 1.0000002
[ForceInline, PreferRecompute, Differentiable]
public Real clamped_acos<Real : floating_point>(Real x) {
    return acos(clamp(x, Real(-1), Real(1)));
}

[ForceInline, PreferRecompute, Differentiable]
public T sum<T : arithmetic, int N>(vector<T, N> v) {
    var result = v[0];
    [ForceUnroll]
    for (int i = 1; i < N; i++) result += v[i];
    return result;
}

[ForceInline, PreferRecompute, Differentiable]
public T prod<T : arithmetic, int N>(vector<T, N> v) {
    var result = v[0];
    [ForceUnroll]
    for (int i = 1; i < N; i++) result *= v[i];
    return result;
}

[ForceInline, PreferRecompute, Differentiable]
public T max<T : arithmetic, int N>(vector<T, N> v) {
    var result = v[0];
    [ForceUnroll]
    for (int i = 1; i < N; i++) result = max(result, v[i]);
    return result;
}

[ForceInline, PreferRecompute, Differentiable]
public T min<T : arithmetic, int N>(vector<T, N> v) {
    var result = v[0];
    [ForceUnroll]
    for (int i = 1; i < N; i++) result = min(result, v[i]);
    return result;
}

// https://erkaman.github.io/posts/model_matrix_recover.html
[ForceInline, PreferRecompute, Differentiable]
public v4<T> extract_translation<T : floating_point>(matrix<T, 4, 4> m) {
    return m[3];
}

[ForceInline, PreferRecompute, Differentiable]
public v3<T> extract_scale<T : floating_point>(matrix<T, 4, 4> m) {
    return v3<T>(length(m[0]), length(m[1]), length(m[2]));
}

[ForceInline, PreferRecompute, Differentiable]
public m3x3<T> extract_rotation<T : floating_point>(matrix<T, 4, 4> m) {
    v3<T> scale = extract_scale(m);
    return m3x3<T>(m[0].xyz / scale.x, m[1].xyz / scale.y, m[2].xyz / scale.z);
}

namespace topology {
    // Utility function to get a vector perpendicular to an input vector
    // (from "Efficient Construction of Perpendicular Vectors Without Branching")
    // https://blog.selfshadow.com/2011/10/17/perp-vectors/
    // v should be normalized
    [ForceInline, PreferRecompute, Differentiable]
    public v3<float> calc_stark_tangent(const v3<float> normal) {
        const v3<float> a = abs(normal);
        const uint32_t xm = ((a.x - a.y) < 0 && (a.x - a.z) < 0) ? 1 : 0;
        const uint32_t ym = (a.y - a.z) < 0 ? (1 ^ xm) : 0;
        const uint32_t zm = 1 ^ (xm | ym);
        return cross(normal, v3<float>(xm, ym, zm));
    }

    // [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    [ForceInline, PreferRecompute, Differentiable]
    public m3x3<Real> calc_orthonormal_basis<Real : floating_point>(v3<Real> normal) {
        const Real s = normal.z >= Real(0.0) ? Real(1.0) : Real(-1.0);
        const Real a = Real(-1.0) / (s + normal.z);
        const Real b = normal.x * normal.y * a;
        v3<Real> tangent = v3<Real>(Real(1.0) + s * normal.x * normal.x * a, s * b, -s * normal.x);
        v3<Real> bitangent = v3<Real>(b, s + normal.y * normal.y * a, -normal.y);
        return m3x3<Real>(tangent, bitangent, normal);
    }
}

[ForceInline, PreferRecompute, Differentiable]
public T fmadd<T : floating_point>(T a, T b, T c) {
    return fma(a, b, c);
}

[ForceInline, PreferRecompute, Differentiable]
public T fmsub<T : floating_point>(T a, T b, T c) {
    return fma(a, b, -c);
}

[ForceInline, PreferRecompute, Differentiable]
public T fnmadd<T : floating_point>(T a, T b, T c) {
    return fma(-a, b, c);
}

[ForceInline, PreferRecompute, Differentiable]
public T fnmsub<T : floating_point>(T a, T b, T c) {
    return fma(-a, b, -c);
}

[ForceInline, PreferRecompute, Differentiable]
public Real safe_sqrt<Real : floating_point>(Real x) {
    return sqrt(max(x, Real(0)));
}

[ForceInline, PreferRecompute, Differentiable]
public Real length2<Real : floating_point, int N>(vector<Real, N> v) {
    return dot(v, v);
}

// Abramowitz & Stegun 7.1.26 approximation of the error function (max error ~1.5e-7).
[ForceInline, PreferRecompute, Differentiable]
public Real erf<Real : floating_point>(Real x) {
    const Real a1 = Real( 0.254829592);
    const Real a2 = Real(-0.284496736);
    const Real a3 = Real( 1.421413741);
    const Real a4 = Real(-1.453152027);
    const Real a5 = Real( 1.061405429);
    const Real p  = Real( 0.3275911);
    Real sign = x < Real(0) ? Real(-1) : Real(1);
    Real ax = abs(x);
    Real t = Real(1) / (Real(1) + p * ax);
    Real y = Real(1) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign * y;
}

[ForceInline, PreferRecompute, Differentiable]
T circ<T : floating_point>(T x) { 
    return safe_sqrt(fnmadd(x, x, T(1.f)));
}

[Differentiable, ForceInline, PreferRecompute]
public v2<Real> sincos<Real : floating_point>(Real angle) {
    v2<Real> sc;
    sincos(angle, sc.x, sc.y);
    return sc;
}

[Differentiable, ForceInline, PreferRecompute]
public v2<Real> cossin<Real : floating_point>(Real angle) {
    v2<Real> sc;
    sincos(angle, sc.x, sc.y);
    return sc.yx;
}

[ForceInline, PreferRecompute, Differentiable]
public matrix<T, N, N> diagonal<T : arithmetic, int N>(vector<T, N> v) {
    matrix<T, N, N> result = matrix<T, N, N>(T(0));
    [ForceUnroll]
    for (int i = 0; i < N; i++) result[i][i] = v[i];
    return result;
}

[ForceInline, PreferRecompute, Differentiable]
vector<T, N> diagonal<T : arithmetic, int N>(matrix<T, N, N> n) {
    vector<T, N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++) result[i] = n[i][i];
    return result;
}

public struct Transform<Real : floating_point> : IDifferentiable {
    public m4x4<Real> forward;
    public m4x4<Real> inverse;

    [Differentiable]
    public __init() {
        forward = inverse = diagonal(v4<Real>(Real(1)));
    }
    [Differentiable]
    public __init(m4x4<Real> f, m4x4<Real> i) {
        forward = f; inverse = i;
    }

    [Differentiable]
    public Transform<Real> _inverse() {
        return Transform<Real>(inverse, forward);
    }

    [Differentiable]
    public Transform<Real> _transpose() {
        return Transform<Real>(transpose(forward), transpose(inverse));
    }

    [Differentiable]
    public static Transform<Real> identity() {
        return Transform<Real>();
    }

    public static Transform<Real> translation(v3<Real> v) {
        m4x4<Real> m = diagonal(v4<Real>(Real(1)));
        column(m, 3, v4<Real>(v, Real(1)));
        m4x4<Real> inv_m = diagonal(v4<Real>(Real(1)));
        column(inv_m, 3, v4<Real>(-v, Real(1)));
        return Transform<Real>(m, transpose(inv_m));
    }

    public static Transform<Real> rotation(v3<Real> axis, Real angle) {
        axis = normalize(axis);
        let sc = sincos(angle);
        let s = sc.y;
        let c = sc.x;
        let oc = Real(1) - c;
        m4x4<Real> m = diagonal(v4<Real>(Real(1)));
        m[0][0] = oc * axis.x * axis.x + c;
        m[0][1] = oc * axis.x * axis.y - axis.z * s;
        m[0][2] = oc * axis.x * axis.z + axis.y * s;
        m[1][0] = oc * axis.y * axis.x + axis.z * s;
        m[1][1] = oc * axis.y * axis.y + c;
        m[1][2] = oc * axis.y * axis.z - axis.x * s;
        m[2][0] = oc * axis.z * axis.x - axis.y * s;
        m[2][1] = oc * axis.z * axis.y + axis.x * s;
        m[2][2] = oc * axis.z * axis.z + c;
        m4x4<Real> inv_m = transpose(m);
        return Transform<Real>(m, inv_m);
    }

    public static Transform<Real> scaling(v3<Real> v) {
        return Transform<Real>(diagonal(v4<Real>(v, Real(1))), diagonal(v4<Real>(Real(1) / v, Real(1))));
    }

    // fov in degrees (!!for now works only on cpu side with column access!!)
    // we use a left/top -1/-1 to right/bottom 1/1 projection
    public static Transform<Real> perspective(Real fov, Real near, Real far) {
        Real recip = Real(1) / (near - far);
        Real tan_half_fov = tan(radians(fov * Real(0.5)));
        Real cot = Real(1) / tan_half_fov;

        m4x4<Real> m = diagonal(v4<Real>(cot /* missing 1/aspect */, -cot, far * recip, Real(0)));

        m[2][3] = -Real(1);
        m[3][2] = near * far * recip;

        m4x4<Real> inv_m = diagonal(v4<Real>(tan_half_fov, -tan_half_fov, Real(0), Real(1) / near));
        inv_m[3][2] = -Real(1);
        inv_m[2][3] = (near - far) / (near * far);

        return Transform<Real>(m, inv_m);
    }

    // formula https://gsteph.blogspot.com/2012/05/world-view-and-projection-matrix.html
    // this is the inverse camera matrix, also called view matrix
    static Transform<Real> view(v3<Real> position, v3<Real> right, v3<Real> up, v3<Real> forward) {
        m4x4<Real> view;
        view[0] = v4<Real>(right[0], up[0], forward[0], Real(0.f));
        view[1] = v4<Real>(right[1], up[1], forward[1], Real(0.f));
        view[2] = v4<Real>(right[2], up[2], forward[2], Real(0.f));
        view[3] = v4<Real>(dot(-position, right), dot(-position, up), dot(-position, forward), Real(1.f));
        m4x4<Real> model_transpose = view;
        model_transpose[0] = v4<Real>(right, Real(0.f));
        model_transpose[1] = v4<Real>(up, Real(0.f));
        model_transpose[2] = v4<Real>(forward, Real(0.f));
        model_transpose[3] = v4<Real>(position, Real(1.f));
        return Transform<Real>(view, model_transpose);
    }
    static Transform<Real> look_at(v3<Real> eye, v3<Real> target, v3<Real> up) {
        v3<Real> forward = normalize(eye - target);
        v3<Real> right = normalize(cross(up, forward));
        v3<Real> new_up = cross(forward, right);
        return view(eye, right, new_up, forward);
    }

    static Transform<Real> model(v3<Real> position, v3<Real> right, v3<Real> up, v3<Real> forward) {
       return view(position, right, up, forward)._inverse();
    }
};

[ForceInline, PreferRecompute, Differentiable]
public Transform<T> operator*<T : floating_point>(Transform<T> left, Transform<T> right) {
    return Transform<T>(mul(left.forward, right.forward), mul(right.inverse, left.inverse));
}

public extension<Real : atomicable_floating_point> DiffTensor<m4x4<Real>, 1>
    where m4x4<Real> : IAtomicAccumulable
{
    [Differentiable]
    public Transform<Real> to_transform() {
        if (!is_valid()) return Transform<Real>::identity();
        return Transform<Real>(load({0}), load({1}));
    }
}

// __generic<Real : floating_point>
// public struct ModelMatrix {
//     public matrix<float, 4, 4> data;
//     public __init CPP_ONLY(ModelMatrix)() {
//         data = { 1.0, 0.0, 0.0, 0.0,
//                  0.0, 1.0, 0.0, 0.0,
//                  0.0, 0.0, 1.0, 0.0,
//                  0.0, 0.0, 0.0, 1.0 };
//     }
//     vector<float, 3> position() { return {data[3].x, data[3].y, data[3].z}; }
//     vector<float, 3> scale() { return { data[0].x, data[1].y, data[2].z }; }
//     vector<float, 3> direction() { return { -data[2].x, -data[2].y, -data[2].z }; }
//     MUTATING void position(vector<float, 3> p) { 
//         data[3].x = p.x;
//         data[3].y = p.y;
//         data[3].z = p.z;
//     }
//     MUTATING void scale(vector<float, 3> s) {
//         data[0].x = s.x;
//         data[1].y = s.y;
//         data[2].z = s.z;
//     }
// };

public enum class CullMode {
    None = 0, Front = 1, Back = 2, Both = 3
};

public enum class RenderMode {
    Primal = 0, Forward = 1, Backward = 2
};

public struct HitMask {
    public static const uint32_t none = 0x00;
    public static const uint32_t all = 0xFF;
};

public struct InstanceIndex {
    public static const uint32_t miss = 0xFFFFFFFF;

    public __init(uint32_t index = miss) {
        _index = index;
    }

    public uint32_t index() { return _index; }
    public bool is_none() { return _index == miss; }
    public uint32_t _index;
};

public struct Frame<Real : floating_point> : IDifferentiable {
    public m3x3<Real> tbn;

    public __init() {
        tbn = {};
    }

    [ForceInline, PreferRecompute, Differentiable]
    public __init(v3<Real> n) {
        tbn = topology::calc_orthonormal_basis<Real>(n);
    }

    [ForceInline, PreferRecompute, Differentiable]
    public __init(v3<Real> t, v3<Real> b, v3<Real> n) {
        tbn = m3x3<Real>(t, b, n);
    }

    [ForceInline, PreferRecompute, Differentiable]
    public v3<Real> tangent() {
        return tbn[0];
    }

    [ForceInline, PreferRecompute, Differentiable]
    public v3<Real> bitangent() {
        return tbn[1];
    }
    
    [ForceInline, PreferRecompute, Differentiable]
    public v3<Real> normal() {
        return tbn[2];
    }

    [ForceInline, PreferRecompute, Differentiable]
    public v3<Real> to_local(v3<Real> v) {
        return mul(tbn, v);
        // return v3<Real>(dot(v, tbn[0]), dot(v, tbn[1]), dot(v, tbn[2]));
    }

    [ForceInline, PreferRecompute, Differentiable]
    public v3<Real> to_world(v3<Real> v) {
        return mul(v, tbn);
        // return fma(v.z, tbn[2], fma(v.y, tbn[1], v.x * tbn[0]));
    }

    [ForceInline, PreferRecompute, Differentiable]
    public static Real cos_theta(v3<Real> v) {
        return v.z;
    }
};

public struct HitInfo {
    public float2 attribute;
    public InstanceIndex instance;
    public uint32_t primitive;
    public float t;
};

public struct SurfacePoint<Real : floating_point> : IDifferentiable {
    public v3<Real> position;
    public Real area;
    public Frame<Real> frame; // smooth normal -> .normal()
    public v3<Real> geometric_normal; // in local frame
    public v3<Real> emission;
    public uint primitive;
    public v2<Real> uv;
    public v2<Real> attribute;
};

public struct SurfaceSample<Real : floating_point> : IDifferentiable {
    public SurfacePoint<Real> sp;
    public Real pdf;
    public bool delta;
};

public struct DirectionSample<Real : floating_point> : IDifferentiable {
    public SurfacePoint<Real> sp;
    public v3<Real> direction;
    public Real distance;
    public Real pdf;
    public bool delta;
};

public struct RaySample<Real : floating_point> : IDifferentiable {
    public Ray<Real> ray;
    public v3<Real> weight; // importance/radiance aka throughput
};

public struct SurfaceInteraction<Real : floating_point> : IDifferentiable {
    public SurfacePoint<Real> sp;
    public v3<Real> wi; // in local frame
};

public struct EdgeSamplePixel<Real : floating_point> : IDifferentiable {
    public v2<Real> pos_in_uv;
    public v2<Real> pos_out_uv;
    public v3<Real> position;

    public Real t;
    public v2<Real> normal;
    public Real pdf; // pdf of selecting this sample
    public v2<Real> length; // arclength
    public Real area; // pixel area
    public v3<Real> barycentric;
    public v2<Real> attribute;
    public v2<Real> ndc_pixel_dims;
};

public struct EdgeSampleHemisphere<Real : floating_point> : IDifferentiable {
    public v3<Real> dir_in;
    public v3<Real> dir_out;
    public v3<Real> normal;
    public Real t;

    public v3<Real> position;  // position of sample in world space
    public Real pdf;           // density in the slerp parameter t, including the edge-selection pmf
    public Real length;        // arclength on the unit sphere
    public Real area;          // solid angle of the (clipped) triangle, in steradians
    public v3<Real> barycentric;
    public v2<Real> attribute;
};

public struct constants<Real : floating_point> {
    public static const Real e = Real(2.718281828459045);
    public static const Real log2e = Real(1.4426950408889634);
    public static const Real log10e = Real(0.4342944819032518);
    public static const Real pi = Real(3.141592653589793);
    public static const Real half_pi = Real(1.5707963267948966);
    public static const Real quarter_pi = Real(0.7853981633974483);
    public static const Real two_pi = Real(6.283185307179586);
    public static const Real four_pi = Real(12.566370614359172);
    public static const Real inv_pi = Real(0.3183098861837907);
    public static const Real inv_two_pi = Real(0.15915494309189535);
    public static const Real inv_four_pi = Real(0.07957747154594767);
    public static const Real ln2 = Real(0.6931471805599453);
    public static const Real ln10 = Real(2.302585092994046);
    public static const Real inv_ln2 = Real(1.442695040888963387004650940071);
    public static const Real sqrt2 = Real(1.4142135623730951);
    public static const Real sqrt3 = Real(1.7320508075688772);
    public static const Real egamma = Real(0.5772156649015329);
    public static const Real phi = Real(1.618033988749895);
    
    public static const Real epsilon = __type_equals<Real, double>() ? Real(0x1p-53) : Real(0x1p-24);
    public static const Real ray_epsilon = epsilon * Real(1500);
    public static const Real shadow_epsilon = ray_epsilon * Real(10);
};


public struct AABB<Real : floating_point> {
    public v3<Real> min;
    public v3<Real> max;

    public __init() {
        this.min = v3<Real>(double::maxValue);
        this.max = v3<Real>(double::minValue);
    }

    public __init(v3<Real> min, v3<Real> max) {
        this.min = min;
        this.max = max;
    }

    public AABB<Real> merge(const AABB<Real> other)
    {
        return AABB<Real>( { ::min(this.min.x, other.min.x), ::min(this.min.y, other.min.y), ::min(this.min.z, other.min.z) },
                           { ::max(this.max.x, other.max.x), ::max(this.max.y, other.max.y), ::max(this.max.z, other.max.z) });
    }

    public bool inside(const v3<Real> point)
    {
        return (this.min.x <= point.x && point.x <= this.max.x
		    && this.min.y <= point.y && point.y <= this.max.y
		    && this.min.z <= point.z && point.z <= this.max.z);
    }

    public uint intersect(inout Ray<Real> ray) {
        const v3<Real> inv_dir = Real(1) / ray.direction;
        const v3<Real> t_lo = (this.min - ray.origin) * inv_dir;
        const v3<Real> t_hi = (this.max - ray.origin) * inv_dir;
        const v3<Real> t_near = ::min(t_lo, t_hi);
        const v3<Real> t_far  = ::max(t_lo, t_hi);

        const Real tmin = ::max(::max(t_near.x, t_near.y), t_near.z);
        const Real tmax = ::min(::min(t_far.x, t_far.y), t_far.z);

        if (tmin > tmax) return 0;
        if (tmax < ray.tmin || tmin > ray.tmax) return 0;
        return 1;
    }
};

[ForceInline]
bool is_finite<Real : floating_point>(Real value)
{
    return !isnan(value) && !isinf(value);
}

[ForceInline]
bool is_finite<Real : floating_point, int N>(vector<Real, N> value)
{
    return !any(isnan(value)) && !any(isinf(value));
}

[Differentiable, PreferRecompute, ForceInline]
T infinitesimal<T : IFloat>(T x)
{
    return x - detach(x);
}

[Differentiable, PreferRecompute, ForceInline]
DifferentialPair<float> infinitesimal(DifferentialPair<float> x)
{
    return diffPair(x.p - detach(x.p), x.d - detach(x.d));
}

[Differentiable, PreferRecompute, ForceInline]
DifferentialPair<vector<Real, N>> infinitesimal<Real : floating_point, int N>(DifferentialPair<vector<Real, N>> x)
{
    return diffPair(x.p - detach(x.p), x.d - detach(x.d));
}

T detach<let b : bool, T : IDifferentiable>(T x) {
    return select(b, detach(x), x);
}

Real factorial<Real : floating_point>(const int x) {
    switch (x) {
        case 0: return Real(1.0);
        case 1: return Real(1.0);
        case 2: return Real(2.0);
        case 3: return Real(6.0);
        case 4: return Real(24.0);
        case 5: return Real(120.0);
        case 6: return Real(720.0);
        case 7: return Real(5040.0);
        case 8: return Real(40320.0);
        case 9: return Real(362880.0);
        case 10: return Real(3628800.0);
        case 11: return Real(39916800.0);
        case 12: return Real(479001600.0);
        case 13: return Real(6227020800.0);
        case 14: return Real(87178291200.0);
        case 15: return Real(1307674368000.0);
        case 16: return Real(20922789888000.0);
        case 17: return Real(355687428096000.0);
        default: break;
    }
    Real s = Real(1);
    for (int n = 1; n <= x; n++) {
        s *= Real(n);
    }
    return s;
}

Real double_factorial<Real : floating_point>(const int x) {
    switch (x) {
        case 0: return Real(1.0);
        case 1: return Real(1.0);
        case 2: return Real(2.0);
        case 3: return Real(3.0);
        case 4: return Real(8.0);
        case 5: return Real(15.0);
        case 6: return Real(48.0);
        case 7: return Real(105.0);
        case 8: return Real(384.0);
        case 9: return Real(945.0);
        case 10: return Real(3840.0);
        case 11: return Real(10395.0);
        case 12: return Real(46080.0);
        case 13: return Real(135135.0);
        case 14: return Real(645120.0);
        case 15: return Real(2027025.0);
        case 16: return Real(10321920.0);
        case 17: return Real(34459425.0);
        default: break;
    }
    Real s = Real(1);
    Real n = Real(x);
    while (n >= Real(2)) {
        s *= n;
        n -= Real(2);
    }
    return s;
}

// physics convention: radius, theta [0, pi], phi [0, 2pi]
[Differentiable]
public v3<Real> to_spherical<Real : floating_point>(const v3<Real> xyz) {
    Real r = length(xyz);
    Real theta = acos(xyz.z / r);
    Real phi = atan2(xyz.y, xyz.x);
    return v3<Real>(r, theta, phi);
}

// physics convention: radius, theta [0, pi], phi [0, 2pi]
[Differentiable]
public v3<Real> to_cartesian<Real : floating_point>(const v3<Real> radius_theta_phi) {
    Real r = radius_theta_phi.x;
    Real theta = radius_theta_phi.y;
    Real phi = radius_theta_phi.z;
    Real x = r * sin(theta) * cos(phi);
    Real y = r * sin(theta) * sin(phi);
    Real z = r * cos(theta);
    return v3<Real>(x, y, z);
}

namespace fresnel {
    // https://en.wikipedia.org/wiki/Refractive_index#Typical_values
    public struct refractive_index<Real : floating_point> {
        public static const Real vacuum = Real(1.0f);
        public static const Real air = Real(1.000293f);
        public static const Real water = Real(1.333f);
        public static const Real ice = Real(1.31f);
        public static const Real glass = Real(1.52f);
        public static const Real flint_glass = Real(1.69f);
        public static const Real sapphire = Real(1.77f);
        public static const Real diamond = Real(2.42f);
    };

    public Real f0<Real : floating_point>(Real ri1, Real ri2) {
        Real r0 = (ri1 - ri2) / (ri1 + ri2);
        return r0 * r0;
    }

    // Schlick's approximation
    // F0: reflectance at 0 degree angle (parallel to normal) (achromatic for dielectrics, and chromatic for metals)
    // F90: reflectance at 90 degree angle (orthogonal to normal) (both dielectrics and conductors exhibit achromatic specular reflectance)
    // same as lerp(F0, F90, (1 - cos(theta))^5) see lerp(x, y, s) -> x + s(y-x)
    [Differentiable]
    public vector<Real, N> schlick<Real : floating_point, int N>(vector<Real, N> ndot, vector<Real, N> f0, vector<Real, N> f90 = Real(1))
    {
        vector<Real, N> m = clamp(Real(1) - ndot, Real(0), Real(1));
        vector<Real, N> m2 = m * m;
        // return f0 + (f90 - f0) * m2 * m2 * m;
        return lerp(f0, f90, m2 * m2 * m);
    }
    [Differentiable]
    public Real schlick<Real : floating_point>(Real ndot, Real f0, Real f90 = Real(1))
    {
        Real m = clamp(Real(1) - ndot, Real(0), Real(1));
        Real m2 = m * m;
        // return f0 + (f90 - f0) * m2 * m2 * m;
        return lerp(f0, f90, m2 * m2 * m);
    }

    /*
     *   https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
     *   https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#FresnelReflectance
     *   https://en.wikipedia.org/wiki/Fresnel_equations
     *   eta_i: index of refraction for medium outside object
     *   eta_t: index of refraction for medium inside object
     *   cosThetaI: angle between incoming ray and normal
     *   cosThetaT: angle between refracted ray and negative normal
     *   return: reflection amount
     */
    [Differentiable]
    Real dielectric<Real : floating_point>(inout Real eta_i, inout Real eta_t, Real cosThetaI, inout Real cosThetaT) {
        // swap etas when inside of object -> IncomingDotN < 0
        // since we handle eta outside -> not used
        if (cosThetaI <= Real(0)) {
            const Real buff = eta_i;
            eta_i = eta_t;
            eta_t = buff;
            cosThetaI = -cosThetaI;
        }
        const Real eta = eta_i / eta_t;
        // no reflection -> straight transmission
        if (eta == Real(1)) {
            cosThetaT = -cosThetaI;
            return Real(0);
        }
        /* Using Snell's law, calculate the squared sine of the
           angle between the normal and the transmitted ray */
        const Real sinThetaTSq = eta * eta * (Real(1) - cosThetaI * cosThetaI);

        // Total internal reflection
        if (sinThetaTSq >= Real(1)) {
            cosThetaT = Real(0);
            return Real(1);
        }

        cosThetaT = sqrt(max(Real(1) - sinThetaTSq, Real(0)));

        const Real rs = (eta_i * cosThetaI - eta_t * cosThetaT)
                    / (eta_i * cosThetaI + eta_t * cosThetaT); /*⊥*/
        const Real rp = (eta_t * cosThetaI - eta_i * cosThetaT)
                    / (eta_t * cosThetaI + eta_i * cosThetaT); /*∥*/

        /* No polarization -- return the unpolarized reflectance */
        return Real(0.5) * (rs * rs + rp * rp);
    }

    /*
     *   https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#FresnelReflectance
     *   eta:
     *   k: extinction coefficient
     *   cosThetaI: angle between incoming ray and normal
     *   return: reflection amount
     */
    [Differentiable]
    v3<Real> F_Conductor<Real : floating_point>(const v3<Real> eta, const v3<Real> k, const Real cosThetaI)
    {
        const Real cosThetaI2 = cosThetaI * cosThetaI;
        const Real sinThetaI2 = max(Real(1) - cosThetaI2, Real(0));
        const Real sinThetaI4 = sinThetaI2 * sinThetaI2;

        const v3<Real> eta2 = eta * eta;
        const v3<Real> k2 = k * k;
        const v3<Real> innerTerm = eta2 - k2 - sinThetaI2;
        const v3<Real> a2PlusB2 = sqrt(max(innerTerm * innerTerm + Real(4) * eta2 * k2, Real(0)));
        const v3<Real> a = sqrt(max((a2PlusB2 + innerTerm) * Real(0.5), Real(0)));

        const v3<Real> A = a2PlusB2 + cosThetaI2;
        const v3<Real> B = Real(2) * cosThetaI * a;
        const v3<Real> C = A + sinThetaI4;
        const v3<Real> D = B * sinThetaI2;

        const v3<Real> rs = (A - B) / (A + B);      /*⊥*/
        const v3<Real> rp = rs * (C - D) / (C + D); /*∥*/

        return Real(0.5) * (rs + rp);
    }
}

// BRDF Explorer
// https://github.com/wdas/brdf/tree/main/src/brdfs
namespace microfacet {
    // https://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
    // but there are different versions
    // -> https://computergraphics.stackexchange.com/questions/1515/what-is-the-accepted-method-of-converting-shininess-to-roughness-and-vice-versa
    // e.g for blinn phong
    [Differentiable]
    public Real shininess_to_roughness<Real : floating_point>(Real shininess) {
        return sqrt(Real(2) / (shininess + Real(2)));
    }

    [Differentiable]
    public Real roughness_to_shininess<Real : floating_point>(Real roughness) {
        return (Real(2) / (roughness * roughness)) - Real(2);
    }

    // https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions#MicrofacetBxDFs
    // has to be applied to the ndf pdf in order to sample incident direction (wi) instead of micro normals (m)
    [Differentiable]
    public Real d_pdf_h_to_wo<Real : floating_point>(Real woDotM /*VdotM*/) {
        return safe_div(Real(0.25), abs(woDotM));
    }
    // in this case wiDotM != woDotM
    [Differentiable]
    public Real d_pdf_h_to_wo<Real : floating_point>(const Real wiDotM, const Real woDotM, const Real eta_i, const Real eta_o) {
        const Real a = eta_i * wiDotM + eta_o * woDotM;
        return safe_div(eta_o * eta_o * abs(woDotM), a * a);
    }

    // Torrance–Sparrow BSDF (BRDF + BTDF)
    [Differentiable]
    Real torrance_sparrow<Real : floating_point>(const Real wiDotN, const Real woDotN) {
        return safe_div(Real(1), Real(4) * abs(wiDotN) * abs(woDotN));
    }

    [Differentiable]
    Real torrance_sparrow<Real : floating_point>(const Real wiDotHt, const Real woDotHt, const Real wiDotN, const Real woDotN, const Real eta_i, const Real eta_t) {
        const Real a = safe_div(abs(wiDotHt) * abs(woDotHt), abs(wiDotN) * abs(woDotN));
        const Real b = (eta_i * wiDotHt + eta_t * woDotHt);
        return a * safe_div(eta_t * eta_t, b * b);
    }

    public struct GGX<Real : floating_point> : IDifferentiable {
        public Real roughness, alpha, alpha2;

        [Differentiable]
        public __init(Real roughness = Real(0.5)) {
            this.roughness = roughness;
            this.alpha = roughness * roughness;
            this.alpha2 = this.alpha * this.alpha;
        }

        [Differentiable]
        public Real d(Real ndoth) {
            Real d = ((ndoth * alpha2 - ndoth) * ndoth + Real(1));
            return alpha2 / (d * d * constants<Real>::pi);
        }

        [Differentiable]
        public v3<Real> sample(v2<Real> uv) {
            Real cos_theta = sqrt((Real(1) - uv.x) / (Real(1) + (alpha2 - Real(1)) * uv.x));
            Real sin_theta = safe_sqrt(Real(1) - cos_theta * cos_theta);
            Real phi = constants<Real>::two_pi * uv.y;
            return { sin_theta * cos(phi), sin_theta * sin(phi), cos_theta };
        }

        [Differentiable]
        v3<Real> sample_vndf(v3<Real> Ve, v2<Real> uv)
        {
            // Section 3.2: transforming the view direction to the hemisphere configuration
            v3<Real> Vh = normalize(v3<Real>(alpha * Ve.x, alpha * Ve.y, Ve.z));
            // Section 4.1: orthonormal basis (with special case if cross product is zero)
            Real lensq = Vh.x * Vh.x + Vh.y * Vh.y;
            v3<Real> T1 = lensq > Real(0) ? v3<Real>(-Vh.y, Vh.x, Real(0)) * (Real(1) / sqrt(lensq)) : v3<Real>(Real(1), Real(0), Real(0));
            v3<Real> T2 = cross(Vh, T1);
            // Section 4.2: parameterization of the projected area
            Real r = sqrt(uv.x);
            Real phi = Real(2) * constants<Real>::pi * uv.y;
            Real t1 = r * cos(phi);
            Real t2 = r * sin(phi);
            Real s = Real(0.5) * (Real(1) + Vh.z);
            t2 = (Real(1) - s) * sqrt(Real(1) - t1 * t1) + s * t2;
            // Section 4.3: reprojection onto hemisphere
            v3<Real> Nh = t1 * T1 + t2 * T2 + sqrt(max(Real(0), Real(1) - t1 * t1 - t2 * t2)) * Vh;
            // Section 3.4: transforming the normal back to the ellipsoid configuration
            v3<Real> Ne = normalize(v3<Real>(alpha * Nh.x, alpha * Nh.y, max(Real(0), Nh.z)));
            return Ne;
        }

        [Differentiable]
        public Real pdf(Real ndoth) {
            return d(ndoth) * ndoth;
        }

        [Differentiable]
        public Real pdf_vndf(Real ndotv, Real mdotv, Real ndoth) {
            return g1(ndotv) * abs(mdotv) * d(ndoth) / ndotv;
        }

        [Differentiable]
        public Real g_lambda(Real ndot) {
            Real cos_sqr = ndot * ndot;
            Real tan_sqr = safe_div(max(Real(1) - cos_sqr, Real(0)), cos_sqr);
            return Real(0.5) * (sqrt(Real(1) + alpha2 * tan_sqr) - Real(1));
        }

        [Differentiable]
        public Real g1(Real ndot) {
            return Real(1) / (Real(1) + g_lambda(ndot));
        }

        [Differentiable]
        public Real g(Real ndotv, Real ndotl) {
            return safe_div(Real(1), (Real(1) + g_lambda(ndotv) + g_lambda(ndotl)));
            // return g1(ndotv) * g1(ndotl);
        }
    };
}

public struct DiscreteDistribution {
    // Sample a integer value in the range [low, high] with uniform probability
    public static T sample_evenly_spaced_range<Real : floating_point, T : integral>(T low, T high, no_diff Real rand) {
        return __float_to_int_cast<T>(__int_to_float_cast<Real>((high + T(1)) - low) * rand) + low;
    }

    // stopped working...
    // public static T sample_evenly_spaced_range<Real : floating_point, T>(T* start, int count, no_diff Real rand) {
    //     return start[__float_to_int_cast<int>(__int_to_float_cast<Real>(count) * rand)];
    // }

    public static no_diff Real pdf_evenly_spaced_range<Real : floating_point, T : integral>(T length) {
        return select(length > T(0), Real(1) / __int_to_float_cast<Real>(length), Real(0));
    }
};

namespace warp {
    [Differentiable]
    public v2<Real> square_to_uniform_disk_concentric<Real : floating_point>(v2<Real> uv) {
        Real x = fmsub(Real(2), uv.x, Real(1));
        Real y = fmsub(Real(2), uv.y, Real(1));

        // Real phi, r;
        // if (x == Real(0) && y == Real(0)) {
        //     r = phi = Real(0);
        // } else if (x * x > y * y) {
        //     r = x;
        //     phi = constants<Real>::quarter_pi * y / x;
        // } else {
        //     r = y;
        //     phi = constants<Real>::half_pi - (x / y) * (constants<Real>::quarter_pi);
        // }
        // return { r * cos(phi), r * sin(phi) };

        bool quadrant_1_or_3 = abs(x) < abs(y);
        bool is_zero = x == Real(0) && y == Real(0);
        Real r = select(quadrant_1_or_3, y, x);
        Real rp = select(quadrant_1_or_3, x, y);
        Real phi = constants<Real>::quarter_pi * safe_div(rp, r);
        phi = select(quadrant_1_or_3, constants<Real>::half_pi - phi, phi);
        phi = select(is_zero, Real(0), phi);
        let sc = sincos(phi);
        return { r * sc.y, r * sc.x };
    }

    [Differentiable]
    public Real square_to_uniform_disk_concentric_pdf<Real : floating_point>(v2<Real> uv) {
        return select(length(uv) > Real(1), Real(0), constants<Real>::inv_pi);
    }

    [Differentiable]
    public v2<Real> square_to_uniform_triangle<Real : floating_point>(v2<Real> uv) {
        Real t = safe_sqrt(Real(1) - uv.x);
        return { Real(1) - t, t * uv.y };
    }

    [Differentiable]
    __generic<bool test_domain>
    public float square_to_uniform_triangle_pdf(v2<float> p) {
        if (test_domain) return select(p.x < 0.0f || p.y < 0.0f || (p.x + p.y) > 1.0f, 0.0f, 2.0f);
        else return 2.0f;
    }

    // physics convention: radius, theta [0, pi], phi [0, 2pi]
    [Differentiable]
    public v3<Real> square_to_uniform_hemisphere<Real : floating_point>( v2<Real> uv) {
        const Real z = uv.y;                               // z = cos(theta)
        const Real r = safe_sqrt(Real(1) - z * z);            // r = sin(theta)
        const Real phi = constants<Real>::two_pi * uv.x;  // phi in [0, 2PI]
        return { r * cos(phi), r * sin(phi), z };
    }

    [Differentiable]
    public Real square_to_uniform_hemisphere_pdf<Real : floating_point>() {
        return constants<Real>::inv_two_pi;
    }

    // physics convention: radius, theta [0, pi], phi [0, 2pi]
    [Differentiable]
    public v3<Real> square_to_uniform_sphere<Real : floating_point>(v2<Real> uv) {
        const Real z = Real(1) - Real(2) * uv.y;            // z = cos(theta)
        const Real r = safe_sqrt(Real(1) - z * z);          // r = sin(theta)
        const Real phi = constants<Real>::two_pi * uv.x;    // phi in [0, 2PI]
        return { r * cos(phi), r * sin(phi), z };
    }

    [Differentiable]
    public Real square_to_uniform_sphere_pdf<Real : floating_point>() {
        return constants<Real>::inv_four_pi;
    }

    [Differentiable]
    public v2<Real> uniform_sphere_to_square<Real : floating_point>(v3<Real> p) {
        const Real phi = atan2(p.y, p.x) * constants<Real>::inv_two_pi;
        return { select(phi < Real(0), phi + Real(1), phi), Real(0.5) * (Real(1) - p.z) };
    }

    [Differentiable]
    public v3<Real> square_to_von_mises_fisher<Real : floating_point>(v2<Real> uv, Real kappa) {
        if (kappa == Real(0)) return square_to_uniform_sphere(uv);

        Real sy = max(Real(1) - uv.y, Real(1e-6f));
        Real cos_theta = Real(1) + log(fmadd(Real(1) - sy, exp(Real(-2) * kappa), sy)) / kappa;

        v2<Real> sc = sincos(constants<Real>::two_pi * uv.x);
        Real sin_theta = safe_sqrt(Real(1) - cos_theta * cos_theta);
        return { sc.y * sin_theta, sc.x * sin_theta, cos_theta };
    }

    [Differentiable]
    public v2<Real> von_mises_fisher_to_square<Real : floating_point>(v3<Real> v, Real kappa) {
        if (kappa == Real(0)) return uniform_sphere_to_square(v);

        Real expm2k = exp(Real(-2) * kappa);
        Real t = exp((v.z - Real(1)) * kappa);
        Real x = atan2(v.y, v.x) * constants<Real>::inv_two_pi;
        x = select(x < Real(0), x + Real(1), x);

        return { x, (Real(1) - t) / (Real(1) - expm2k) };
    }

    [Differentiable]
    public Real square_to_von_mises_fisher_pdf<Real : floating_point>(v3<Real> v, Real kappa) {
        if (kappa <= Real(0)) return constants<Real>::inv_four_pi;
        return exp(kappa * (v.z - Real(1))) * (kappa * constants<Real>::inv_two_pi) /
               (Real(1) - exp(Real(-2) * kappa));
    }

    [Differentiable]
    public v3<Real> square_to_cosine_hemisphere<Real : floating_point>(v2<Real> uv) {
        v2<Real> p = square_to_uniform_disk_concentric(uv);
        Real z = safe_sqrt(Real(1) - dot(p, p));
        return { p.x, p.y, z };
    }

    [Differentiable]
    public Real square_to_cosine_hemisphere_pdf<Real : floating_point>(v3<Real> v) {
        return constants<Real>::inv_pi * v.z;
    }

    [Differentiable]
    public v3<Real> square_to_uniform_cone<Real : floating_point>(v2<Real> uv, Real const_theta_max) {
        if (false) { // warping method based on standard disk mapping
            const Real phi = constants<Real>::two_pi * uv.x;
            const Real cos_theta = uv.y * (Real(1) - const_theta_max) + const_theta_max;
            const Real sin_theta = safe_sqrt(Real(1) - cos_theta * cos_theta);
            return { sin_theta * cos(phi), sin_theta * sin(phi), cos_theta };
        } else { // low-distortion warping technique based on concentric disk mapping
            Real one_minus_const_theta_max = Real(1) - const_theta_max;
            v2<Real> p = square_to_uniform_disk_concentric(uv);
            Real pn = dot(p, p);
            Real z = const_theta_max + one_minus_const_theta_max * (Real(1) - pn);
            p *= safe_sqrt(one_minus_const_theta_max * (Real(2) - one_minus_const_theta_max * pn));
            return { p.x, p.y, z };
        }
    }

    [Differentiable]
    __generic<let test_domain : bool, Real : floating_point>
    public Real square_to_uniform_cone_pdf(v3<Real> v, Real const_theta_max) {
        if (test_domain) return select(abs(dot(v, v) - Real(1)) > constants<Real>::ray_epsilon || v.z < const_theta_max, Real(0), constants<Real>::inv_two_pi / (Real(1) - const_theta_max));
        else return constants<Real>::inv_two_pi / (Real(1) - const_theta_max);
    }

    // https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#fragment-Convertlightsampleweighttosolidanglemeasure-0
    [Differentiable]
    public Real solid_angle_to_area<Real : floating_point>(const Real area, const Real dist, const Real cosTheta) {
        return (area * max(abs(cosTheta), Real(1e-8f))) / (dist * dist);
    }

    [Differentiable]
    public Real area_to_solid_angle<Real : floating_point>(const Real area, const Real dist, const Real cosTheta) {
        return (dist * dist) / (area * max(abs(cosTheta), Real(1e-8f)));
    }

    [Differentiable]
    public v2<Real> box_muller<Real : floating_point>(v2<Real> rand) {
        Real r = sqrt(-Real(2) * log(rand.x));
        Real phi = constants<Real>::two_pi * rand.y;
        return { r * cos(phi), r * sin(phi) };
    }
}

namespace mis {
    [Differentiable]
    public Real balanced<Real : floating_point>(Real pdf_a, Real pdf_b) {
        return pdf_a / (pdf_a + pdf_b);
    }

    [Differentiable]
    public Real power<Real : floating_point>(Real pdf_a, Real pdf_b) {
        let a2 = pdf_a * pdf_a;
        return a2 / (a2 + pdf_b * pdf_b);
    }

    [Differentiable]
    public Real equal<Real : floating_point>(Real pdf_a, Real pdf_b) {
        return Real(0.5);
    }
}

public enum MIS {
    Equal = 0,
    Balanced = 1,
    Power = 2,
}

public extension MIS {
    [Differentiable]
    __generic<Real : floating_point>
    public Real operator()(Real pdf_a, Real pdf_b) {
        if (this == MIS::Balanced) return mis::balanced(pdf_a, pdf_b);
        else if (this == MIS::Power) return mis::power(pdf_a, pdf_b);
        else if (this == MIS::Equal) return mis::equal(pdf_a, pdf_b);
        return Real(0);
    }
}

namespace spherical_harmonics {
    // Evaluate the associated Legendre polynomial of degree @l and order @m at
    // coordinate @x. The inputs must satisfy:
    // 1. l >= 0
    // 2. 0 <= m <= l
    // 3. -1 <= x <= 1
    // See http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    //
    // This implementation is based off the approach described in [1],
    // instead of computing Pml(x) directly, Pmm(x) is computed. Pmm can be
    // lifted to Pmm+1 recursively until Pml is found
    public static Real eval_legendre_polynomial<Real : floating_point>(const int l, const int m, const Real x) {
        // Compute Pmm(x) = (-1)^m(2m - 1)!!(1 - x^2)^(m/2), where !! is the float
        // factorial.
        Real pmm = Real(1);
        // P00 is defined as 1.0, do don't evaluate Pmm unless we know m > 0
        if (m > 0) {
            Real sign = Real(m % 2 == 0 ? 1 : -1);
            pmm = sign * double_factorial<Real>(2 * m - 1) * pow(Real(1) - x * x, Real(m) / Real(2));
        }
    
        if (l == m) {
            // Pml is the same as Pmm so there's no lifting to higher bands needed
            return pmm;
        }

        // Compute Pmm+1(x) = x(2m + 1)Pmm(x)
        Real pmm1 = x * Real(2 * m + 1) * pmm;
        if (l == m + 1) {
            // Pml is the same as Pmm+1 so we are done as well
            return pmm1;
        }
    
        // Use the last two computed bands to lift up to the next band until l is
        // reached, using the recurrence relationship:
        // Pml(x) = (x(2l - 1)Pml-1 - (l + m - 1)Pml-2) / (l - m)
        for (int n = m + 2; n <= l; n++) {
            const Real pmn = (x * Real(2 * n - 1) * pmm1 - Real(n + m - 1) * pmm) / Real(n - m);
            pmm = pmm1;
            pmm1 = pmn;
        }
        // Pmm1 at the end of the above loop is equal to Pml
        return pmm1;
    }

    [Differentiable, PreferRecompute]
    public static Real basis<Real : floating_point, bool HSH, bool NORMALIZED>(const int l, const int m, Real theta, const Real phi) {
        // CHECK(l >= 0, "l must be at least 0.");
        // CHECK(-l <= m && m <= l, "m must be between -l and l.");
        let normalization = select(NORMALIZED, select(HSH, constants<Real>::inv_two_pi, constants<Real>::inv_four_pi), Real(1));

        Real cos_theta = cos(theta);
        if (HSH) cos_theta = Real(2) * cos_theta - Real(1);
        const int abs_m = abs(m);
        const Real kml = sqrt(normalization * Real(2 * l + 1) * factorial<Real>(l - abs_m) / factorial<Real>(l + abs_m));
        const Real kml_times_lpoly = kml * no_diff eval_legendre_polynomial(l, abs_m, cos_theta);
        if (m == 0) return kml_times_lpoly;

        const Real m_times_phi = Real(abs_m) * phi;
        const Real sh = constants<Real>::sqrt2 * kml_times_lpoly;
        if (m > 0) return sh * cos(m_times_phi);
        else return sh * sin(m_times_phi);
    }

    [Differentiable, PreferRecompute]
    public static v2<Real> spherical_coords<Real : floating_point, bool HSH>(const v3<Real> dir) {
        v2<Real> theta_phi = to_spherical(dir).yz;
        if (HSH) theta_phi.x = clamp(theta_phi.x, Real(0), constants<Real>::half_pi);
        return theta_phi;
    }

    [Differentiable, PreferRecompute]
    public static Real basis<Real : floating_point, bool HSH, bool NORMALIZED>(const int l, const int m, const v3<Real> dir) {
        let theta_phi = spherical_coords<Real, HSH>(dir);
        return basis<Real, HSH, NORMALIZED>(l, m, theta_phi.x, theta_phi.y);
    }

    public int coefs<int Channels>(int order) {
        return Channels * (order + 1) * (order + 1);
    }

    public static int basis_index(const int l, const int m) {
        return l * (l + 1) + m;
    }

    public static int basis_band(const int index) {
        int l = 0;
        while ((l + 1) * (l + 1) <= index) l++;
        return l;
    }

    static int coef_index<int Channels>(const int l, const int m, int channel = 0) {
        return basis_index(l, m) * Channels + channel;
    }

    // Laplacian smoothing in the frequency domain -- see also page 19 in StupidSH36.pdf
    public static Real band_smoothing<Real : floating_point>(const int l, const Real smoothing) {
        return Real(1) / (Real(1) + smoothing * Real((l * (l + 1)) * (l * (l + 1))));
    }

    // Cosine lobe transfer coefficients A_l = 2π ∫₀¹ t·Pₗ(t) dt (Funk-Hecke)
    // Multiply SH band l by A_l to convert radiance SH → irradiance SH
    public static Real cosine_transfer<Real : floating_point, bool HSH>(int l) {
        if (HSH) {
            if (l == 0) return constants<Real>::pi;
            if (l == 1) return constants<Real>::pi / Real(3);
            return Real(0);
        }

        if (l == 0) return constants<Real>::pi;
        if (l == 1) return constants<Real>::two_pi / Real(3);
        if (l % 2 != 0) return Real(0); // All odd l > 1 are 0

        // For even l >= 2
        int k = l / 2;
        Real sign = (k % 2 == 0) ? Real(-1) : Real(1);
        
        // Compute the factorial ratio safely using a running product
        // Equivalent to (2k)! / (2^(2k) * (k!)^2)
        Real double_fact_ratio = Real(1);
        for (int i = 1; i <= k; i++) {
            double_fact_ratio *= (Real(2 * i) - Real(1)) / Real(2 * i);
        }

        Real term = sign / (Real(l + 2) * Real(l - 1));
        return constants<Real>::two_pi * term * double_fact_ratio;
    }
}

public struct SphericalHarmonics<Real : floating_point, int Order, int Channels, bool HSH> {
    public vector<Real, Channels * (Order + 1) * (Order + 1)> coefficients;

    public vector<Real, Channels> evaluate(Real theta, Real phi, vector<Real, Channels> cweights = vector<Real, Channels>(1)) {
        vector<Real, Channels> result = vector<Real, Channels>(0);
        [ForceUnroll]
        for (int l = 0; l <= Order; l++) {
            [ForceUnroll]
            for (int m = -l; m <= l; m++) {
                const Real yml = spherical_harmonics::basis<Real, HSH, true>(l, m, theta, phi);
                [ForceUnroll]
                for (int c = 0; c < Channels; c++) {
                    result[c] += cweights[c] * coefficients[spherical_harmonics::coef_index<Channels>(l, m, c)] * yml;
                }
            }
        }
        return result;
    }

    public vector<Real, Channels> evaluate(v3<Real> dir, vector<Real, Channels> cweights = vector<Real, Channels>(1)) {
        vector<Real, Channels> result = vector<Real, Channels>(0);
        [ForceUnroll]
        for (int l = 0; l <= Order; l++) {
            [ForceUnroll]
            for (int m = -l; m <= l; m++) {
                const Real yml = spherical_harmonics::basis<Real, HSH, true>(l, m, dir);
                [ForceUnroll]
                for (int c = 0; c < Channels; c++) {
                    result[c] += cweights[c] * coefficients[spherical_harmonics::coef_index<Channels>(l, m, c)] * yml;
                }
            }
        }
        return result;
    }
};

public struct SphericalHarmonicsArray<Real : atomicable_floating_point, int Order, int Channels, bool HSH>
{
    public DiffTensor<vector<Real, Channels>, 2> coefficients;

    public static const int basis_count = (Order + 1) * (Order + 1);

    static int[2] coef_index(int index, const int i) {
        int idx[2] = { index, i };
        return idx;
    }

    [Differentiable]
    public vector<Real, Channels> evaluate(int index, Real theta, Real phi, vector<Real, Channels> cweights = vector<Real, Channels>(1)) {
        vector<Real, Channels> result = vector<Real, Channels>(0);
        [ForceUnroll]
        for (int i = 0; i < basis_count; i++) {
            let l = spherical_harmonics::basis_band(i);
            let yml = spherical_harmonics::basis<Real, HSH, true>(l, i - l * (l + 1), theta, phi);
            result += cweights * coefficients.load(coef_index(index, i)) * yml;
        }
        return result;
    }

    [Differentiable]
    public vector<Real, Channels> evaluate(int index, v3<Real> dir, vector<Real, Channels> cweights = vector<Real, Channels>(1)) {
        let theta_phi = spherical_harmonics::spherical_coords<Real, HSH>(dir);
        return evaluate(index, theta_phi.x, theta_phi.y, cweights);
    }

    [Differentiable]
    public void project(int index, Real theta, Real phi, vector<Real, Channels> value, no_diff Real smoothing = Real(0.3)) {
        [ForceUnroll]
        for (int i = 0; i < basis_count; i++) {
            let l = spherical_harmonics::basis_band(i);
            let _smoothing = no_diff spherical_harmonics::band_smoothing(l, smoothing);
            let yml = _smoothing * spherical_harmonics::basis<Real, HSH, true>(l, i - l * (l + 1), theta, phi);
            coefficients.atomic_add(coef_index(index, i), value * yml);
        }
    }

    [Differentiable]
    public void project(int index, v3<Real> dir, vector<Real, Channels> value, no_diff Real smoothing = Real(0.3)) {
        let theta_phi = spherical_harmonics::spherical_coords<Real, HSH>(dir);
        project(index, theta_phi.x, theta_phi.y, value, smoothing);
    }

};

// half vector for reflection
// wi: normalized incoming dir pointing away
// wo: normalized outgoing dir pointing away
// result: normalized half dir pointing away
[Differentiable, ForceInline, PreferRecompute]
v3<Real> halfvector<Real : floating_point>(const v3<Real> wi, const v3<Real> wo) {
    return normalize(wi + wo);
}

// half vector for refraction
// wi: normalized incoming dir pointing away
// wo: normalized outgoing dir pointing away
// result: normalized half dir pointing away
[Differentiable, ForceInline, PreferRecompute]
v3<Real> halfvector<Real : floating_point>(const v3<Real> wi, const v3<Real> wo, const Real eta_i, const Real eta_o) {
    return normalize(-eta_i * wi - eta_o * wo);
}

[Differentiable, PreferRecompute]
public Real compute_ray_distance_to_triangle<Real : floating_point>(const v3<Real> v[3], const v3<Real> origin, const v3<Real> direction)
{
    v3<Real> edge1 = v[1] - v[0];
    v3<Real> edge2 = v[2] - v[0];
    v3<Real> pvec = cross(direction, edge2);
    Real det = dot(edge1, pvec);
    v3<Real> tvec = origin - v[0];
    // Real u = dot(tvec, pvec);
    v3<Real> qvec = cross(tvec, edge1);
    //Real v = dot(direction, qvec);
    Real t = dot(edge2, qvec) / det;
    return t;
}

/**
 * Computes the barycentric coordinates of a point on a triangle.
 * @param[in] v Positions of the three vertices.
 * @param[in] p Intersection point.
 * @param[out] barycentrics Barycentric coordinates of the intersection.
 */
[Differentiable, PreferRecompute]
public v3<Real> compute_barycentrics<Real : floating_point>(const v3<Real> v[3], const v3<Real> p)
{
    let edgep0 = v[0] - p;
    let edgep1 = v[1] - p;
    let edgep2 = v[2] - p;

    let area0 = length(cross(edgep1, edgep2));
    let area1 = length(cross(edgep2, edgep0));
    let area2 = length(cross(edgep0, edgep1));
    let area = area0 + area1 + area2;

    let bx = area0 / area;
    let by = area1 / area;
    let bz = area2 / area;
    return v3<Real>(bx, by, bz);
}

[Differentiable, PreferRecompute]
public v3<Real> attr_to_bary<Real : floating_point>(v2<Real> attr) {
    return v3<Real>(Real(1) - attr.x - attr.y, attr.x, attr.y);
}

[Differentiable, PreferRecompute]
public v2<Real> bary_to_attr<Real : floating_point>(v3<Real> bary) {
    return v2<Real>(bary.y, bary.z);
}

[Differentiable, PreferRecompute]
__generic<Real : floating_point, int N>
public vector<Real, N> bilerp(vector<Real, N> v0, vector<Real, N> v1, vector<Real, N> v2, vector<Real, N> v3, v2<Real> uv) {
    const vector<Real, N> v01 = lerp(v0, v1, uv.x);
    const vector<Real, N> v23 = lerp(v2, v3, uv.x);
    return lerp(v01, v23, uv.y);
}

__generic<Real : floating_point, int N>
public vector<Real, N> catmull_rom_spline(vector<Real, N> p0, vector<Real, N> p1, vector<Real, N> t0, vector<Real, N> t1, Real t) {
    const Real t2 = t * t;
    const Real t3 = t * t2;
    return (Real(2) * t3 - Real(3) * t2 + Real(1)) * p0 + (t3 - Real(2) * t2 + t) * t0 + (Real(-2) * t3 + Real(3) * t2) * p1 + (t3 - t2) * t1;
}

__generic<Real : floating_point, int N>
public vector<Real, N> catmull_rom_spline_tangent(vector<Real, N> p0, vector<Real, N> p1, vector<Real, N> t0, vector<Real, N> t1, Real t) {
    const Real t2 = t * t;
    return (Real(6) * t2 - Real(6) * t) * p0 + (Real(3) * t2 - Real(4) * t + Real(1)) * t0 + (Real(-6) * t2 + Real(6) * t) * p1 + (Real(3) * t2 - Real(2) * t) * t1;
}

namespace post {
    __generic<Real : floating_point>
    v2<Real> pixel_to_uv(int2 pixel, int2 resolution)
    {
        return (v2<Real>(pixel) + Real(0.5)) / v2<Real>(resolution);
    }

    [Differentiable, ForceInline, PreferRecompute]
    __generic<Real : floating_point, int N>
    public vector<Real, N> linear_to_srgb(const vector<Real, N> linear) {
        return select(linear <= Real(0.0031308), linear * Real(12.92), Real(1.055) * pow(linear, vector<Real, N>(1.0 / 2.4)) - Real(0.055));
    }

    [Differentiable, ForceInline, PreferRecompute]
    __generic<Real : floating_point, int N>
    public vector<Real, N> srgb_to_linear(const vector<Real, N> srgb) {
        return select(srgb <= Real(0.04045), srgb / Real(12.92), pow((srgb + Real(0.055)) / Real(1.055), vector<Real, N>(2.4)));
    }

    [Differentiable, ForceInline, PreferRecompute]
    __generic<Real : floating_point>
    public vector<Real, 3> hsv_to_rgb(const vector<Real, 3> c) {
        vector<Real, 3> p = abs(frac(c.xxx + vector<Real, 3>(Real(1), Real(2.0 / 3.0), Real(1.0 / 3.0))) * vector<Real, 3>(Real(6.0)) - vector<Real, 3>(Real(3.0)));
        return c.z * lerp(vector<Real, 3>(Real(1)), saturate(p - vector<Real, 3>(Real(1))), c.y);
    }

    // from https://developer.blender.org/diffusion/B/browse/master/intern/opencolorio/gpu_shader_display_transform.glsl$142
    __generic<Real : floating_point, int N>
    public vector<Real, N> dither(const vector<Real, N> color, const Real strength, const Real random) {
        return color + (random * Real(0.0033) * strength);
    }

    public vector<Real, N> tonemap_aces_filmic<Real : floating_point, int N>(vector<Real, N> x)
    {
        Real a = Real(2.51);
        Real b = Real(0.03);
        Real c = Real(2.43);
        Real d = Real(0.59);
        Real e = Real(0.14);
        return (x * (a * x + b)) / (x * (c * x + d) + e);
    }

    // AgX tonemapping based on Filament's implementation using Rec.2020 primaries
    // Ref: https://github.com/google/filament/pull/7236
    // Ref: https://iolite-engine.com/blog_posts/minimal_agx_implementation
    // All matrices transposed from GLSL column-major to Slang row-major
    public vector<Real, 3> tonemap_agx<Real : floating_point>(vector<Real, 3> color)
    {
        // Linear sRGB → Linear Rec.2020
        const matrix<Real, 3, 3> srgb_to_rec2020 = matrix<Real, 3, 3>(
            Real(0.6274), Real(0.3293), Real(0.0433),
            Real(0.0691), Real(0.9195), Real(0.0113),
            Real(0.0164), Real(0.0880), Real(0.8956)
        );

        // AgX Inset Matrix (Rec.2020 → AgX log space)
        const matrix<Real, 3, 3> agx_inset = matrix<Real, 3, 3>(
            Real(0.856627153315983),  Real(0.0951212405381588), Real(0.0482516061458583),
            Real(0.137318972929847),  Real(0.761241990602591),  Real(0.101439036467562),
            Real(0.11189821299995),   Real(0.0767994186031903), Real(0.811302368396859)
        );

        const Real min_ev = Real(-12.47393);
        const Real max_ev = Real(4.026069);

        color = mul(srgb_to_rec2020, color);
        color = mul(agx_inset, color);

        // Log2 encoding
        color = max(color, Real(1e-10));
        color = log2(color);
        color = (color - min_ev) / (max_ev - min_ev);
        color = clamp(color, Real(0), Real(1));

        // AgX default contrast approximation (6th-degree polynomial)
        vector<Real, 3> x2 = color * color;
        vector<Real, 3> x4 = x2 * x2;
        color = Real(15.5) * x4 * x2
              - Real(40.14) * x4 * color
              + Real(31.96) * x4
              - Real(6.868) * x2 * color
              + Real(0.4298) * x2
              + Real(0.1191) * color
              - Real(0.00232);

        // AgX Outset Matrix
        const matrix<Real, 3, 3> agx_outset = matrix<Real, 3, 3>(
            Real(1.1271005818144368),   Real(-0.11060664309660323), Real(-0.016493938717834573),
            Real(-0.1413297634984383),  Real(1.157823702216272),    Real(-0.016493938717834257),
            Real(-0.14132976349843826), Real(-0.11060664309660294), Real(1.2519364065950405)
        );
        color = mul(agx_outset, color);

        // Linearize
        color = pow(max(color, Real(0)), vector<Real, 3>(Real(2.2)));

        // Linear Rec.2020 → Linear sRGB
        const matrix<Real, 3, 3> rec2020_to_srgb = matrix<Real, 3, 3>(
            Real(1.6605),  Real(-0.5876), Real(-0.0728),
            Real(-0.1246), Real(1.1329),  Real(-0.0083),
            Real(-0.0182), Real(-0.1006), Real(1.1187)
        );
        color = mul(rec2020_to_srgb, color);

        return saturate(color);
    }
}

namespace optimization {
    public struct Adam<Real : __BuiltinFloatingPointType> {
        public Real beta1, beta2, epsilon;
        public Real m, v;
        public uint t;

        public __init(Real beta1 = Real(0.9), Real beta2 = Real(0.999), Real epsilon = Real(1e-8)) {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.m = Real(0);
            this.v = Real(0);
            this.t = 0;
        }

        [mutating]
        public void update(Real lr, Real grad, inout Real param) {
            t += 1;
            m = beta1 * m + (Real(1) - beta1) * grad;
            v = beta2 * v + (Real(1) - beta2) * grad * grad;
            Real m_hat = m / (Real(1) - pow(beta1, Real(t)));
            Real v_hat = v / (Real(1) - pow(beta2, Real(t)));
            param -= lr * m_hat / (sqrt(v_hat) + epsilon);
        }
    };
}

namespace experiment {
    Real cross2<Real : floating_point>(vector<Real, 2> a, vector<Real, 2> b) { return a.x * b.y - b.x * a.y; }

    public struct Edge2D<Real : floating_point> {
        public v2<Real> A, B;
        v2<Real> intersection(Edge2D<Real> other) {
            return ((other.A - other.B) * cross2(A, B) - (A - B) * cross2(other.A, other.B)) *
                (Real(1) / cross2(A - B, other.A - other.B));
        }

        bool is_inside(v2<Real> point) // check if a point is on the LEFT side of an edge
        {
            return (cross2(A - B, point) + cross2(B, A)) < Real(0);
        }

        public v2<Real> direction() { return B - A; }
        public v2<Real> normal() {
            v2<Real> dir = direction();
            return normalize(v2<Real>(dir.y, -dir.x));
        }

        public Real length2() {
            v2<Real> d = B - A;
            return d.x * d.x + d.y * d.y;
        }
        public Real length() { return sqrt(length2()); }
        public v2<Real> interpolate(Real t) { return A + t * (B - A); }
    };

    public struct AABB2D<Real : floating_point> {
        public v2<Real> min, max;

        // left-bottom, right-bottom, right-top, left-top
        v2<Real> get_vertex(int i) {
            if (i == 0) return min;
            if (i == 1) return v2<Real>(max.x, min.y);
            if (i == 2) return max;
            if (i == 3) return v2<Real>(min.x, max.y);
            return v2<Real>(Real(0), Real(0));
        }
        public Real area() { return (max.x - min.x) * (max.y - min.y); }

        // Liang-Barsky: https://www.skytopia.com/project/articles/compsci/clipping.html
        public Tuple<bool, Edge2D<Real>> intersect_segment(v2<Real> A, v2<Real> B) {
            Tuple<bool, Edge2D<Real>> output;
            output._0 = false; output._1 = Edge2D<Real>(v2<Real>(Real(0), Real(0)), v2<Real>(Real(0), Real(0)));
            Real t0 = Real(0), t1 = Real(1);
            Real dx = B.x - A.x, dy = B.y - A.y;
            Real p, q, r;

            for (int edge = 0; edge < 4; edge++) { // Traverse through left, right, bottom, top edges.
                if (edge == 0) { p = -dx; q = -(min.x - A.x); }
                if (edge == 1) { p = dx; q = (max.x - A.x); }
                if (edge == 2) { p = -dy; q = -(min.y - A.y); }
                if (edge == 3) { p = dy; q = (max.y - A.y); }

                r = q / p;

                if (p == Real(0) && q <= Real(0)) return output; // Don't draw line at all. (parallel line outside and parallel line)

                Real epsilon = Real(0.0001);
                if (p < Real(0)) {
                    if (r >= t1 - epsilon) return output; // Don't draw line at all. equals to make sure lines exactly through corners are clipped!
                    else if (r > t0) t0 = r; // Line is clipped!
                } else if (p > Real(0)) {
                    if (r <= t0 + epsilon) return output; // Don't draw line at all. equals to make sure lines exactly through corners are clipped!
                    else if (r < t1) t1 = r; // Line is clipped!
                }
            }

            output._0 = true;
            output._1.A = v2<Real>(A.x + t0 * dx, A.y + t0 * dy);
            output._1.B = v2<Real>(A.x + t1 * dx, A.y + t1 * dy);
            return output;
        }
    };

    public struct Plane<Real : floating_point> : IDifferentiable {
        public vector<Real, 3> center;
        public vector<Real, 3> normal;
        [Differentiable]
        public vector<Real, 3> intersect(Ray<Real> ray) {
            Real t = dot(center - ray.origin, normal) / dot(normal, ray.direction);
            return ray.origin + t * ray.direction;
        }
    };

    public struct Arc<Real : floating_point> {
        public vector<Real, 3> dir_0, dir_1;

        // https://math.stackexchange.com/a/231225
        public Real length() {
            return acos(dot(dir_0, dir_1));
        }
        // https://www.reddit.com/r/gamedev/comments/5j4efg/equation_of_a_arc_between_two_points_on_a_spheres/
        public vector<Real, 3> interpolate(Real t) {
            var omega = length();
            var d = sin(omega);
            var s0 = sin((Real(1) - t) * omega);
            var s1 = sin(t * omega);
            return (dir_0 * s0 + dir_1 * s1) / d;
            //return normalize(dir_0 * sin((1.0f - t) * length()) + dir_1 * sin(t * length()));
        }
    };

    public struct Edge<Real : floating_point> {
        public __init() {
            A = B = vector<Real, 3>(Real(0));
        }
        public __init(vector<Real, 3> a, vector<Real, 3> b) {
            A = a; B = b;
        }
        public vector<Real, 3> A, B;
    };

    public Real spherical_triangle_area<Real : floating_point>(vector<Real, 3> a, vector<Real, 3> b, vector<Real, 3> c) {
        // 1. Calculate plane normals (unit normal to the great circle arcs)
        vector<Real, 3> n_c = normalize(cross(a, b)); // Normal for side a-b
        vector<Real, 3> n_a = normalize(cross(b, c)); // Normal for side b-c
        vector<Real, 3> n_b = normalize(cross(c, a)); // Normal for side c-a
        // 2. Calculate interior angles using the angle between plane normals
        //    The angle between the outward-facing normals is the *exterior* angle.
        //    Interior angle = PI - acos(dot(n1, n2))
        // Angle at vertex a (between n_c and n_b)
        Real alpha = constants<Real>::pi - clamped_acos(dot(n_c, n_b));
        // Angle at vertex b (between n_a and n_c)
        Real beta = constants<Real>::pi - clamped_acos(dot(n_a, n_c));
        // Angle at vertex c (between n_b and n_a)
        Real gamma = constants<Real>::pi - clamped_acos(dot(n_b, n_a));
        // 3. Apply Girard's Theorem for unit sphere (r=1): Area = alpha + beta + gamma - pi
        Real area = alpha + beta + gamma - constants<Real>::pi;
        // Ensure the area is positive (can be negative due to winding/orientation)
        return abs(area);
    }

    public struct Hemisphere<Real : floating_point> {
        public vector<Real, 3> center;
        public vector<Real, 3> normal;
        // return clipped vertices (3 or 4)
        // not edge ordered
        public Tuple<vector<Real, 3>[4], int> clip(const vector<Real, 3> triangle[3]) {
            Tuple<vector<Real, 3>[4], int> out; out._1 = 0;

            vector<Real, 3> prev = triangle[2];
            Real prev_side = dot(center - prev, normal); // if <=0 front, else back
            bool prev_front = prev_side <= Real(0);

            for (int i = 0; i < 3; i++) {
                vector<Real, 3> next = triangle[i];
                Real next_side = dot(center - next, normal);
                bool next_front = next_side <= Real(0);
                vector<Real, 3> dir = normalize(next - prev);

                if (prev_front ^ next_front) { // different sides -> intersection
                    Real denom = dot(normal, dir);
                    if (abs(denom) > Real(1e-6)) {
                        Real t = prev_side / denom;
                        if (t >= Real(0)) out._0[out._1++] = prev + t * dir;
                    }
                }
                if (next_front) out._0[out._1++] = next;

                prev = next;
                prev_side = next_side;
                prev_front = next_front;
            }
            return out;
        }

        // return clipped edges without parallel edge
        // edge ordered
        public Tuple<bool[3], Edge<Real>[3]> clip_edges(const vector<Real, 3> triangle[3]) {
            Tuple<bool[3], Edge<Real>[3]> out;
            out._0[0] = out._0[1] = out._0[2] = false;

            vector<Real, 3> prev = triangle[0];
            Real prev_side = dot(center - prev, normal); // if <=0 front, else back
            bool prev_front = prev_side <= Real(0);

            for (int i = 1; i < 4; i++) {
                vector<Real, 3> next = triangle[i % 3];
                Real next_side = dot(center - next, normal);
                bool next_front = next_side <= Real(0);
                vector<Real, 3> dir = normalize(next - prev);

                if (prev_front ^ next_front) { // different sides -> intersection
                    Real denom = dot(normal, dir);
                    if (abs(denom) > Real(1e-6)) {
                        Real t = prev_side / denom;
                        if (t >= Real(0)) {
                            vector<Real, 3> intersection = prev + t * dir;
                            if (prev_front) out._1[i-1] = Edge<Real>(prev, intersection);
                            else out._1[i-1] = Edge<Real>(intersection, next);
                            out._0[i-1] = true;
                        }
                    }
                }
                else if (prev_front) {
                    out._1[i-1] = Edge<Real>(prev, next);
                    out._0[i-1] = true;
                }

                prev = next;
                prev_side = next_side;
                prev_front = next_front;
            }

            return out;
        }

        public Real clip_area(vector<Real, 3> triangle[3]) {
            let p = clip(triangle);

            Real area = Real(0);
            if (p._1 >= 3) {
                var dir0 = p._0[1] - p._0[0];
                var dir1 = p._0[2] - p._0[0];
                area = length(cross(dir0, dir1)) * Real(0.5);
            }
            if (p._1 == 4) {
                var dir0 = p._0[2] - p._0[0];
                var dir1 = p._0[3] - p._0[0];
                area += length(cross(dir0, dir1)) * Real(0.5);
            }
            return area;
        }

        // girard's theorem
        // https://vanderbei.princeton.edu/WebGL/GirardThmProof.html
        public Real clip_solid_angle_area(const vector<Real, 3> triangle[3]) {
            let p = clip(triangle);
            if (p._1 < 3) return Real(0); // degenerate clip, p._0 is uninitialized

            var A = normalize(p._0[0] - center);
            var B = normalize(p._0[1] - center);
            var C = normalize(p._0[2] - center);
            Real area = spherical_triangle_area(A, B, C);
            if (p._1 == 4) {
                var D = normalize(p._0[3] - center);
                area += spherical_triangle_area(A, C, D);
            }
            return area;
        }
    };

    public struct Triangle2D<Real : floating_point> {
        public v2<Real> v[3];

        // Sutherland–Hodgman poly algo
        // https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#C++
        // not edge ordered
        public Tuple<v2<Real>[6], int> clip(AABB2D<Real> aabb) {

            Tuple<v2<Real>[6], int> output; output._0[0] = v[0]; output._0[1] = v[1]; output._0[2] = v[2]; output._1 = 3;

            v2<Real> p1 = aabb.get_vertex(3);
            for (int i = 0; i < 4; i++) { // over aabb
                v2<Real> p2 = aabb.get_vertex(i);
                int input_size = output._1;
                v2<Real> input[6]; for (int j = 0; j < input_size; j++) input[j] = output._0[j];

                v2<Real> start = input[input_size - 1];
                output._1 = 0;
                for (int j = 0; j < input_size; j++) {
                    v2<Real> end = input[j];
                    Edge2D<Real> e = Edge2D<Real>(p1, p2);
                    if (e.is_inside(end)) {
                        if (!e.is_inside(start)) {
                            output._0[output._1++] = e.intersection(Edge2D<Real>(start, end));
                        }
                        output._0[output._1++] = end;
                    } else if (e.is_inside(start)) {
                        output._0[output._1++] = e.intersection(Edge2D<Real>(start, end));
                    }
                    start = end;
                }
                p1 = p2;
            }
            return output;
        }

        // Liang-Barsky: https://www.skytopia.com/project/articles/compsci/clipping.html
        // edge ordered
        public Tuple<bool[3], Edge2D<Real>[3]> intersect(AABB2D<Real> aabb) {
            Tuple<bool[3], Edge2D<Real>[3]> output;
            output._0[0] = output._0[1] = output._0[2] = false;
            for (int i = 0; i < 3; i++) {
                Tuple<bool, Edge2D<Real>> inter = aabb.intersect_segment(v[i], v[(i + 1) % 3]);
                if (inter._0) {
                    output._0[i] = inter._0;
                    output._1[i] = inter._1;
                } else {
                    output._0[i] = false;
                }
            }
            return output;
        }
    };
}

namespace clipping {
    bool _liang_barsky(float p, float q, inout float t0, inout float t1) {
        if (p == 0) {
            if (q < 0) return false; // parallel line outside
        } else {
            float r = q / p;
            if (p < 0) {
                if (r > t1) return false;
                if (r > t0) t0 = r;
            } else {
                if (r < t0) return false;
                if (r < t1) t1 = r;
            }
        }
        return true;
    }

    // in NDC
    // https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm
    public bool liang_barsky(inout float3 p0, inout float3 p1, inout float t0, inout float t1) {
        const float3 min = float3(-1, -1, 0);
        const float3 max = float3(1, 1, 1);
        const float3 d = p1 - p0;
        if (!_liang_barsky(-d.x, p0.x - min.x, t0, t1)) return false;
        if (!_liang_barsky(d.x, max.x - p0.x, t0, t1)) return false;
        if (!_liang_barsky(-d.y, p0.y - min.y, t0, t1)) return false;
        if (!_liang_barsky(d.y, max.y - p0.y, t0, t1)) return false;
        if (!_liang_barsky(-d.z, p0.z - min.z, t0, t1)) return false;
        if (!_liang_barsky(d.z, max.z - p0.z, t0, t1)) return false;

        if (t1 < t0) return false;
        p1 = p0 + t1 * d;
        p0 = p0 + t0 * d;
        return true;
    }
}
