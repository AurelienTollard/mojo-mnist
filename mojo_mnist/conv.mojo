from gpu import thread_idx, block_idx, block_dim, barrier, grid_dim
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from memory import AddressSpace
from mojo_mnist.activation import relu_scalar, identity_scalar


fn conv2d_gpu_kernel_impl[
    dtype: DType,
    layout_input: Layout,  # Input: (batch_size, in_channels, height, width)
    layout_weights: Layout,  # Weights: (out_channels, in_channels, kernel_height, kernel_width)
    layout_bias: Layout,  # Bias: (out_channels,)
    layout_kernel: Layout,  # Output: (batch_size, out_channels, kernel_height, kernel_width)
    layout_output: Layout,  # Output: (batch_size, out_channels, height, width)
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
    tile_size: Int,
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
    block = (TILE_SIZE, TILE_SIZE)
    grid = (ceil(height / TILE_SIZE), ceil(width / TILE_SIZE), batch * out_channels)
    """
    comptime batch_size = Int(layout_input.shape[0])
    comptime in_channels = Int(layout_input.shape[1])
    comptime out_channels = Int(layout_output.shape[1])

    comptime height = Int(layout_input.shape[2])
    comptime width = Int(layout_input.shape[3])

    comptime kernel_height = Int(layout_kernel.shape[2])
    comptime kernel_width = Int(layout_kernel.shape[3])
    debug_assert(kernel_height == kernel_width, "Kernel height != width is not implemented")

    comptime stride = Int(layout_kernel.shape[4])
    debug_assert(stride == 1, "Stride != 1 is not implemented")
    comptime padding = Int(layout_kernel.shape[5]) # = halo

    comptime effective_tile_size = tile_size + kernel_width - 1

    depth_idx = Int(block_idx.z)
    out_channel_idx = depth_idx % out_channels
    batch_idx = Int(depth_idx // out_channels)

    # # Load in shared memory a tile containing the input data for the whole block
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

    # Copy kernel and input data to shared memory
    @parameter
    if local_idx == 0:
        @parameter
        for i in range(kernel_height):
            @parameter
            for j in range(kernel_width):
                kernel_shared[i, j] = kernel[i, j]

        @parameter
        for i in range(effective_tile_size):
            @parameter
            for j in range(effective_tile_size):
                input_shared[i, j] = input[i, j]


    barrier()
