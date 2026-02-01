from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from memory import AddressSpace
from mojo_mnist.activation import relu_scalar, identity_scalar


comptime TILE_SIZE = 32

fn conv2d_gpu_kernel_impl[
    dtype: DType,
    layout_input: Layout,  # Input: (batch_size, in_channels, height, width)
    layout_weights: Layout,  # Weights: (out_channels, in_channels, kernel_height, kernel_width)
    layout_bias: Layout,  # Bias: (out_channels,)
    layout_kernel: Layout,  # Output: (batch_size, out_channels, kernel_height, kernel_width)
    layout_output: Layout,  # Output: (batch_size, out_channels, height, width)
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
](
    input: LayoutTensor[dtype, layout_input, MutAnyOrigin],
    weights: LayoutTensor[dtype, layout_weights, MutAnyOrigin],
    bias: LayoutTensor[dtype, layout_bias, MutAnyOrigin],
    kernel: LayoutTensor[dtype, layout_kernel, MutAnyOrigin],
    output: LayoutTensor[dtype, layout_output, MutAnyOrigin],
):
    """Convolution layer kernel with configurable activation.

    Computes Y = activation(convolution(input, kernel) + bias).
    Each thread computes a single output element for a single output channel.
    """
    comptime batch_size = Int(layout_x.shape[0])
    comptime in_channels = Int(layout_x.shape[1])
    comptime out_channels = Int(layout_y.shape[1])

    comptime height = Int(layout_x.shape[2])
    comptime width = Int(layout_x.shape[3])

    comptime kernel_height = Int(layout_kernel.shape[2])
    comptime kernel_width = Int(layout_kernel.shape[3])
    debug_assert(kernel_height == kernel_width, "Kernel height != width is not implemented")

    comptime stride = Int(layout_kernel.shape[4])
    debug_assert(stride == 1, "Stride != 1 is not implemented")
    comptime padding = Int(layout_kernel.shape[5]) # = halo

    comptime effective_tile_size = TILE_SIZE + kernel_width - 1

    # global_idx = Int(
    #     thread_idx.x
    #     + blockIdx.x * blockDim.x
    #     + blockIdx.y * gridDim.x * blockDim.x
    #     + blockIdx.z * gridDim.y * gridDim.x * blockDim.x
    # )
    # local_idx = Int(
    #     thread_idx.x
    #     + threadIdx.y * blockDim.x
    #     + threadIdx.z * blockDim.y * blockDim.x
    # )
    batch_idx = Int(blockIdx.z)

    # Load in shared memory a tile containing the input data for the whole block
    input_shared = LayoutTensor[
        dtype,
        Layout.row_major(effective_tile_size, effective_tile_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    # Load in shared memory a tile containing the kernel data for the whole block
    kernel_shared = LayoutTensor[
        dtype,
        Layout.row_major(kernel_height, kernel_width),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
