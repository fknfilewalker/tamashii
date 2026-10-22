implementing tamashii;

public dyn interface IColormap {
    // t in [0, 1] for sequential colormaps, [-1, 1] for diverging colormaps
    public float3 eval(float t);
};