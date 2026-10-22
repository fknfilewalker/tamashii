import sys
from enum import Enum

from tamashii import spy
import tamashii as tm
import torch


class Interop(Enum):
    """Which side owns the allocation the other one views."""

    EXPORT_SPY_TENSOR = "spy -> torch"
    IMPORT_TORCH_TENSOR = "torch -> spy"


usage = (
    spy.BufferUsage.shared
    | spy.BufferUsage.shader_resource
    | spy.BufferUsage.unordered_access
)

shader = r"""
import tamashii;

Tensor<float3, 1> x;
Tensor<float3, 1> y;
DiffTensor<float3, 1> z;

[Differentiable]
float3 weighted(no_diff DiffTensor<float3, 1> t, no_diff int idx[1])
{
    return 3.0f * t.load(idx);
}

[shader("compute"), numthreads(8, 1, 1)]
void interop(uint3 tid : SV_DispatchThreadID)
{
    int index[1] = { int(tid.x) };
    y[index] = y[index] + 2.0f * x[index];
    // Reads z's primal and accumulates d(3z)/dz = 3 into its gradient.
    bwd_diff(weighted)(z, index, float3(1.0f));
}
"""


def run(backend, interop):
    device = tm.create_torch_device(type=backend, include_paths=tm.include_paths())
    module = device.load_module_from_source("torch_interop", shader)
    program = device.link_program(
        modules=[module],
        entry_points=[module.entry_point("interop")],
    )
    kernel = device.create_compute_kernel(program)

    if interop == Interop.EXPORT_SPY_TENSOR:
        # SlangPy owns the allocation and torch views it. The only direction Vulkan can do,
        # since torch memory cannot be imported there.
        x = spy.Tensor.empty(device, shape=(8,), dtype="float3", usage=usage)
        y = spy.Tensor.empty(device, shape=(8,), dtype="float3", usage=usage)
        # with_grads() attaches a gradient tensor; grad_in and grad_out alias it by default.
        z = spy.Tensor.empty(device, shape=(8,), dtype="float3", usage=usage).with_grads(zero=True)
        torch_x = tm.to_torch(x)
        torch_y = tm.to_torch(y)
        torch_z = tm.to_torch(z)
        torch_dz = tm.to_torch(z.grad_out)
        z = {"_primal": z, "_grad_in": z.grad_in, "_grad_out": z.grad_out}
        torch_x.copy_(torch.arange(24, device=torch_x.device, dtype=torch.float32).reshape(8, 3))
        torch_y.fill_(10.0)
        torch_z.fill_(5.0)
        torch_dz.fill_(0.0)
    else:
        # torch owns the allocation and Tamashii imports it through DLPack -- CUDA hands out
        # its device pointer, MPS the MTLBuffer -- so neither side copies.
        torch_device = "cuda" if backend == spy.DeviceType.cuda else "mps"
        torch_x = torch.arange(24, device=torch_device, dtype=torch.float32).reshape(8, 3)
        torch_y = torch.full((8, 3), 10.0, device=torch_device)
        torch_z = torch.full((8, 3), 5.0, device=torch_device)
        torch_dz = torch.zeros((8, 3), device=torch_device)
        x = tm.torch_tensor(torch_x, element_shape=(3,), device=device)
        y = tm.torch_tensor(torch_y, element_shape=(3,), device=device)
        # The gradient is a second torch tensor, bound as DiffTensor's grad fields.
        grad = tm.torch_tensor(torch_dz, element_shape=(3,), device=device)
        z = {
            "_primal": tm.torch_tensor(torch_z, element_shape=(3,), device=device),
            "_grad_in": grad,
            "_grad_out": grad,
        }

    # Only the CUDA interop backends have semaphores to signal: a native CUDA device shares
    # torch's default stream, so its dispatch is already ordered against torch's work.
    if backend == spy.DeviceType.vulkan:
        device.sync_to_cuda()
    elif backend == spy.DeviceType.metal:
        torch.mps.synchronize()
    kernel.dispatch([8, 1, 1], vars={"x": x, "y": y, "z": z})
    if backend == spy.DeviceType.vulkan:
        device.sync_to_device()
    elif backend == spy.DeviceType.metal:
        # SlangPy and torch submit to separate Metal queues.
        device.wait()

    expected = 10.0 + 2.0 * torch.arange(24, device=torch_y.device).reshape(8, 3)
    assert torch.equal(torch_y, expected)
    # The primal is only read, the gradient receives d(3z)/dz from every thread.
    assert torch.equal(torch_z, torch.full_like(torch_z, 5.0))
    assert torch.equal(torch_dz, torch.full_like(torch_dz, 3.0))
    print(backend, interop.value, torch_y)
    print("  z", torch_z[0].tolist(), "-> grad", torch_dz[0].tolist())


if sys.platform == "darwin":
    run(spy.DeviceType.metal, Interop.EXPORT_SPY_TENSOR)
    run(spy.DeviceType.metal, Interop.IMPORT_TORCH_TENSOR)
else:
    run(spy.DeviceType.cuda, Interop.EXPORT_SPY_TENSOR)
    run(spy.DeviceType.cuda, Interop.IMPORT_TORCH_TENSOR)
    # Vulkan can only export: torch's allocator has no handle Vulkan could import.
    run(spy.DeviceType.vulkan, Interop.EXPORT_SPY_TENSOR)
