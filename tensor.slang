implementing tamashii;

// N-dimensional GPU tensor over a raw device pointer (buffer device address),
// modeled after slangpy's Tensor but bindable from plain renderpass shaders via
// ShaderCursor on any backend with pointer support (Metal, Vulkan, CUDA).
// Bind a spy.Tensor directly from Python: cursor["_name"] = spy_tensor.
//
// Conventions (matching slangpy):
// - Array coordinates are in shape order: idx[0] indexes the outermost
//   dimension, so an image of shape (H, W) is t[{y, x}].
// - Vector coordinates are the transpose: int2(x, y) for shape (H, W).
// - Strides are in elements, not bytes. Prefer scalar element types with an
//   extra component dimension over vector elements: on Metal sizeof(float3) is
//   16 in MSL, so a float3* over tightly packed Python data would misindex.
// Raw-pointer tensors have no hardware robustness (no robustBufferAccess /
// StructuredBuffer clamping), so out-of-bounds coordinates are wild memory
// accesses. Enabling this clamps every coordinate into [0, shape) — incorrect
// results instead of corruption/crashes. Off by default to keep the hot paths
// branch-free; flip it here or per-device via
// create_device(compiler_options={"defines": {"TAMASHII_TENSOR_BOUNDS_CHECK": "1"}}).
#ifndef TAMASHII_TENSOR_BOUNDS_CHECK
#define TAMASHII_TENSOR_BOUNDS_CHECK 0
#endif

public struct Tensor<T, let D : int>
{
    public T* _data;
    public uint[D] _shape;
    public int[D] _strides; // element strides
    public uint _offset;    // element offset of the first element
    uint _pad; // explicit tail padding

    public __init()
    {
        _data = nullptr;
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            _shape[i] = 0;
            _strides[i] = 0;
        }
        _offset = 0;
        _pad = 0;
    }

    public bool is_valid() { return _data != nullptr; }

    public property shape : uint[D] { get { return _shape; } }

    public property element_count : uint
    {
        get
        {
            uint n = 1;
            [ForceUnroll]
            for (int i = 0; i < D; ++i)
                n *= _shape[i];
            return n;
        }
    }

    // linear element index of an array coordinate ([0] = outermost dimension)
    public int index(int idx[D])
    {
        int lin = int(_offset);
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            int c = idx[i];
#if TAMASHII_TENSOR_BOUNDS_CHECK
            c = clamp(c, 0, int(_shape[i]) - 1);
#endif
            lin += c * _strides[i];
        }
        return lin;
    }

    // the vector form of a coordinate is the transpose of the array form
    public int index<I : integral>(vector<I, D> idx)
    {
        int lin = int(_offset);
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            int c = idx[D - 1 - i].toInt();
#if TAMASHII_TENSOR_BOUNDS_CHECK
            c = clamp(c, 0, int(_shape[i]) - 1);
#endif
            lin += c * _strides[i];
        }
        return lin;
    }

    // explicit in-bounds test against the shape, for callers that want to
    // guard instead of clamp (e.g. `if (!t.contains(p)) return;`)
    public bool contains(int idx[D])
    {
        bool inside = true;
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
            inside = inside && uint(idx[i]) < _shape[i];
        return inside;
    }
    public bool contains<I : integral>(vector<I, D> idx)
    {
        bool inside = true;
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
            inside = inside && uint(idx[D - 1 - i].toInt()) < _shape[i];
        return inside;
    }

    [ForceInline] public T load(int idx[D]) { return _data[index(idx)]; }
    [ForceInline] public T load<I : integral>(vector<I, D> idx) { return _data[index(idx)]; }
    [ForceInline] public void store(int idx[D], T value) { _data[index(idx)] = value; }
    [ForceInline] public void store<I : integral>(vector<I, D> idx, T value) { _data[index(idx)] = value; }

    public __subscript(int idx[D]) -> T
    {
        get { return load(idx); }
        [nonmutating] set { store(idx, newValue); }
    }
    public __subscript<I : integral>(vector<I, D> idx) -> T
    {
        get { return load(idx); }
        [nonmutating] set { store(idx, newValue); }
    }

    // view of the trailing SliceD dimensions at a leading coordinate,
    // e.g. Tensor<float, 3> image; image.slice<1>({y, x}) is the pixel's channels
    public Tensor<T, SliceD> slice<let SliceD : int>(int idx[D - SliceD])
    {
        Tensor<T, SliceD> result;
        result._data = _data;
        result._offset = _offset;
        result._pad = 0;
        [ForceUnroll]
        for (int i = 0; i < D - SliceD; ++i)
            result._offset += uint(idx[i] * _strides[i]);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
        {
            result._shape[i] = _shape[D - SliceD + i];
            result._strides[i] = _strides[D - SliceD + i];
        }
        return result;
    }
}

namespace tensor_impl
{
    void get_sample_indices<let N : int>(vector<float, N> uv, vector<int, N> shape, out vector<int, N> idx0, out vector<int, N> idx1, out vector<float, N> interp)
    {
        vector<int, N> maxidx = shape - 1;
        vector<float, N> scaled = uv * vector<float, N>(shape);
        vector<int, N> texel = vector<int, N>(scaled);
        interp = frac(scaled);
        idx0 = clamp(texel, vector<int, N>(0), maxidx);
        idx1 = min(texel + vector<int, N>(1), maxidx);
    }
}

// Element types that support atomic accumulation through a device pointer.
// Vectors accumulate per component ("atomic" in slangpy's loose sense: memory
// is consistent after the kernel, intermediate states may mix components).
public interface IAtomicAccumulable
{
    static void atomic_accumulate(This* buf, int idx, This value);
}
public extension<T : atomicable_floating_point> T : IAtomicAccumulable
{
    public static void atomic_accumulate(T* buf, int idx, T value) { InterlockedAdd(buf[idx], value); }
}
public extension uint : IAtomicAccumulable
{
    public static void atomic_accumulate(uint* buf, int idx, uint value) { InterlockedAdd(buf[idx], value); }
}
public extension int : IAtomicAccumulable
{
    public static void atomic_accumulate(int* buf, int idx, int value) { InterlockedAdd(buf[idx], value); }
}
public extension<T : IAtomicAccumulable, let N : int> vector<T, N> : IAtomicAccumulable
{
    public static void atomic_accumulate(vector<T, N>* buf, int idx, vector<T, N> value)
    {
        T* p = (T*)(buf + idx);
        [ForceUnroll]
        for (int c = 0; c < N; ++c)
            T::atomic_accumulate(p, c, value[c]);
    }
}
// Per cell, flattened row-major (SlangMatrixLayout::row_major, the slangpy default).
public extension<T : IAtomicAccumulable, let R : int, let C : int> matrix<T, R, C> : IAtomicAccumulable
{
    public static void atomic_accumulate(matrix<T, R, C>* buf, int idx, matrix<T, R, C> value)
    {
        T* p = (T*)(buf + idx);
        [ForceUnroll]
        for (int r = 0; r < R; ++r)
            [ForceUnroll]
            for (int c = 0; c < C; ++c)
                T::atomic_accumulate(p, r * C + c, value[r][c]);
    }
}

// atomic accumulation (gradient-style adds)
public extension<T : IAtomicAccumulable, let D : int> Tensor<T, D>
{
    [ForceInline] public void add(int idx[D], T value) { T::atomic_accumulate(_data, index(idx), value); }
    [ForceInline] public void add<I : integral>(vector<I, D> idx, T value) { T::atomic_accumulate(_data, index(idx), value); }
}

// Multilinear sampling over uv in [0,1]^D — bilinear for an image, trilinear for a volume —
// as the weighted sum of the 2^D surrounding corners. uv is a vector coordinate, so uv.x
// indexes the innermost dimension. T covers scalar and vector elements alike via IFloat.
public extension<T : IFloat, let D : int> Tensor<T, D>
{
    public T sample(vector<float, D> uv)
    {
        vector<int, D> shape_v;
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
            shape_v[i] = int(_shape[D - 1 - i]);

        vector<int, D> idx0, idx1;
        vector<float, D> interp;
        tensor_impl::get_sample_indices(uv, shape_v, idx0, idx1, interp);

        T result = T(0.0f);
        [ForceUnroll]
        for (int corner = 0; corner < (1 << D); ++corner)
        {
            vector<int, D> idx;
            float weight = 1.0f;
            [ForceUnroll]
            for (int i = 0; i < D; ++i)
            {
                bool high = ((corner >> i) & 1) != 0;
                idx[i] = high ? idx1[i] : idx0[i];
                weight *= high ? interp[i] : (1.0f - interp[i]);
            }
            result = result + load(idx).scale(weight);
        }
        return result;
    }
}

// Differentiable tensor: a primal tensor paired with gradient tensors of the
// element's differential type. Field names match slangpy's DiffTensor
// (_primal/_grad_in/_grad_out), so a spy.Tensor with grads attached binds
// directly from Python: cursor["_x"] = spy_tensor.with_grads().
// _grad_in feeds forward-mode tangents and store() backwards; _grad_out
// receives atomically accumulated load() backwards. They usually alias the
// same buffer (spy .with_grads() default).
public struct DiffTensor<T : IDifferentiable & IArithmetic & IAtomicAccumulable, let D : int>
    where T.Differential : IAtomicAccumulable
{
    public Tensor<T, D> _primal;
    public Tensor<T.Differential, D> _grad_in;
    public Tensor<T.Differential, D> _grad_out;

    public __init()
    {
        _primal = Tensor<T, D>();
        _grad_in = Tensor<T.Differential, D>();
        _grad_out = Tensor<T.Differential, D>();
    }

    public bool is_valid() { return _primal.is_valid(); }
    public bool has_grad() { return _grad_out.is_valid(); }
    public bool has_grad_in() { return _grad_in.is_valid(); }
    public bool has_grad_out() { return _grad_out.is_valid(); }

    public property shape : uint[D] { get { return _primal.shape; } }
    public bool contains(int idx[D]) { return _primal.contains(idx); }
    public bool contains<I : integral>(vector<I, D> idx) { return _primal.contains(idx); }

    [ForceInline, Differentiable, ForwardDerivative(_load_fwd), BackwardDerivative(_load_bwd)]
    public T load(int idx[D]) { return _primal.load(idx); }
    [ForceInline, Differentiable, ForwardDerivative(_load_fwd_vector), BackwardDerivative(_load_bwd_vector)]
    public T load<I : integral>(vector<I, D> idx) { return _primal.load(idx); }
    [ForceInline] DifferentialPair<T> _load_fwd(int idx[D])
    {
        T primal = _primal.load(idx);
        if (has_grad_in()) return diffPair(primal, _grad_in.load(idx));
        return diffPair<T>(primal);
    }
    [ForceInline] DifferentialPair<T> _load_fwd_vector<I : integral>(vector<I, D> idx)
    {
        T primal = _primal.load(idx);
        if (has_grad_in()) return diffPair(primal, _grad_in.load(idx));
        return diffPair<T>(primal);
    }
    [ForceInline] void _load_bwd(int idx[D], T.Differential grad)
    {
        if (has_grad_out()) _grad_out.add(idx, grad);
    }
    [ForceInline] void _load_bwd_vector<I : integral>(vector<I, D> idx, T.Differential grad)
    {
        if (has_grad_out()) _grad_out.add(idx, grad);
    }

    // load whose backward stores instead of accumulating — cheaper when each
    // element is read by exactly one thread
    [ForceInline, Differentiable, ForwardDerivative(_load_fwd), BackwardDerivative(_load_once_bwd)]
    public T load_once(int idx[D]) { return _primal.load(idx); }
    [ForceInline, Differentiable, ForwardDerivative(_load_fwd_vector), BackwardDerivative(_load_once_bwd_vector)]
    public T load_once<I : integral>(vector<I, D> idx) { return _primal.load(idx); }
    [ForceInline] void _load_once_bwd(int idx[D], T.Differential grad)
    {
        if (has_grad_out()) _grad_out.store(idx, grad);
    }
    [ForceInline] void _load_once_bwd_vector<I : integral>(vector<I, D> idx, T.Differential grad)
    {
        if (has_grad_out()) _grad_out.store(idx, grad);
    }

    // note: bwd_diff of a function containing a store REPLAYS the primal store
    // (Slang re-executes the forward part to reconstruct intermediates) before
    // reading the gradient — same-value re-stores in backward passes are by design
    [ForceInline, Differentiable, ForwardDerivative(_store_fwd), BackwardDerivative(_store_bwd)]
    public void store(int idx[D], T value) { _primal.store(idx, value); }
    [ForceInline, Differentiable, ForwardDerivative(_store_fwd_vector), BackwardDerivative(_store_bwd_vector)]
    public void store<I : integral>(vector<I, D> idx, T value) { _primal.store(idx, value); }
    [ForceInline] void _store_fwd(int idx[D], DifferentialPair<T> value)
    {
        _primal.store(idx, value.p);
        if (has_grad_out()) _grad_out.store(idx, value.d);
    }
    [ForceInline] void _store_fwd_vector<I : integral>(vector<I, D> idx, DifferentialPair<T> value)
    {
        _primal.store(idx, value.p);
        if (has_grad_out()) _grad_out.store(idx, value.d);
    }
    [ForceInline] void _store_bwd(int idx[D], inout DifferentialPair<T> value)
    {
        if (has_grad_in()) value = diffPair(value.p, _grad_in.load(idx));
        else value = diffPair<T>(value.p);
    }
    [ForceInline] void _store_bwd_vector<I : integral>(vector<I, D> idx, inout DifferentialPair<T> value)
    {
        if (has_grad_in()) value = diffPair(value.p, _grad_in.load(idx));
        else value = diffPair<T>(value.p);
    }

    // non-differentiable gradient access
    [ForceInline] public T.Differential grad(int idx[D])
    {
        if (has_grad_in()) return _grad_in.load(idx);
        return diffPair<T>(_primal.load(idx)).d;
    }
    [ForceInline] public T.Differential grad<I : integral>(vector<I, D> idx)
    {
        if (has_grad_in()) return _grad_in.load(idx);
        return diffPair<T>(_primal.load(idx)).d;
    }
    [ForceInline] public void accumulate(int idx[D], T.Differential value)
    {
        if (has_grad_out()) _grad_out.add(idx, value);
    }
    [ForceInline] public void accumulate<I : integral>(vector<I, D> idx, T.Differential value)
    {
        if (has_grad_out()) _grad_out.add(idx, value);
    }

    [ForceInline, Differentiable, ForwardDerivative(_add_fwd), BackwardDerivative(_add_bwd)]
    public void add(int idx[D], T value) { _primal.store(idx, _primal.load(idx) + detach(value)); }
    [ForceInline, Differentiable, ForwardDerivative(_add_fwd_vector), BackwardDerivative(_add_bwd_vector)]
    public void add<I : integral>(vector<I, D> idx, T value) { _primal.store(idx, _primal.load(idx) + detach(value)); }
    [ForceInline] void _add_fwd(int idx[D], DifferentialPair<T> value)
    {
        _primal.store(idx, _primal.load(idx) + value.p);
        if (has_grad_out()) _grad_out.store(idx, T.dadd(_grad_out.load(idx), value.d));
    }
    [ForceInline] void _add_fwd_vector<I : integral>(vector<I, D> idx, DifferentialPair<T> value)
    {
        _primal.store(idx, _primal.load(idx) + value.p);
        if (has_grad_out()) _grad_out.store(idx, T.dadd(_grad_out.load(idx), value.d));
    }
    [ForceInline] void _add_bwd(int idx[D], inout DifferentialPair<T> value)
    {
        value = diffPair(value.p, grad(idx));
    }
    [ForceInline] void _add_bwd_vector<I : integral>(vector<I, D> idx, inout DifferentialPair<T> value)
    {
        value = diffPair(value.p, grad(idx));
    }

    [ForceInline, Differentiable, ForwardDerivative(_atomic_add_fwd), BackwardDerivative(_atomic_add_bwd)]
    public void atomic_add(int idx[D], T value) { _primal.add(idx, detach(value)); }
    [ForceInline, Differentiable, ForwardDerivative(_atomic_add_fwd_vector), BackwardDerivative(_atomic_add_bwd_vector)]
    public void atomic_add<I : integral>(vector<I, D> idx, T value) { _primal.add(idx, detach(value)); }
    [ForceInline] void _atomic_add_fwd(int idx[D], DifferentialPair<T> value)
    {
        _primal.add(idx, value.p);
        if (has_grad_out()) _grad_out.add(idx, value.d);
    }
    [ForceInline] void _atomic_add_fwd_vector<I : integral>(vector<I, D> idx, DifferentialPair<T> value)
    {
        _primal.add(idx, value.p);
        if (has_grad_out()) _grad_out.add(idx, value.d);
    }
    [ForceInline] void _atomic_add_bwd(int idx[D], inout DifferentialPair<T> value)
    {
        if (has_grad_in()) value = diffPair(value.p, _grad_in.load(idx));
        else value = diffPair<T>(value.p);
    }
    [ForceInline] void _atomic_add_bwd_vector<I : integral>(vector<I, D> idx, inout DifferentialPair<T> value)
    {
        if (has_grad_in()) value = diffPair(value.p, _grad_in.load(idx));
        else value = diffPair<T>(value.p);
    }

    public __subscript(int idx[D]) -> T
    {
        [Differentiable] get { return load(idx); }
        [nonmutating] set { store(idx, newValue); }
    }
    public __subscript<I : integral>(vector<I, D> idx) -> T
    {
        [Differentiable] get { return load(idx); }
        [nonmutating] set { store(idx, newValue); }
    }
}


