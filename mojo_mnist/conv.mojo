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
    layout_output: Layout,  # Output: (batch_size, out_channels, height, width)
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
    tile_size: Int,
    stride: Int,
    padding: Int,
](
    input: LayoutTensor[dtype, layout_input, MutAnyOrigin],
    weights: LayoutTensor[dtype, layout_weights, MutAnyOrigin],
    bias: LayoutTensor[dtype, layout_bias, MutAnyOrigin],
    output: LayoutTensor[dtype, layout_output, MutAnyOrigin],
):
    """Convolution layer kernel with configurable activation.

    Computes Y = activation(convolution(input, weights) + bias).
    Each thread computes a single output element for a single output channel.
    block = (TILE_SIZE, TILE_SIZE)
    grid = (ceil(height / TILE_SIZE), ceil(width / TILE_SIZE), batch * out_channels)
    """
    comptime batch_size = Int(layout_input.shape[0])
    comptime in_channels = Int(layout_input.shape[1])
    comptime out_channels = Int(layout_output.shape[1])

    comptime height = Int(layout_input.shape[2])
    comptime width = Int(layout_input.shape[3])

    comptime kernel_height = Int(layout_weights.shape[2])
    comptime kernel_width = Int(layout_weights.shape[3])
    debug_assert(
        kernel_height == kernel_width,
        "Kernel height != width is not implemented",
    )
    debug_assert(stride == 1, "Stride != 1 is not implemented")

    comptime effective_tile_size = tile_size + kernel_width - 1

    depth_idx = Int(block_idx.z)
    out_channel_idx = depth_idx % out_channels
    batch_idx = Int(depth_idx // out_channels)

    local_x = Int(thread_idx.x)
    local_y = Int(thread_idx.y)
    local_idx = Int(thread_idx.y * block_dim.x + thread_idx.x)

    tile_origin_x = Int(block_idx.x) * tile_size - padding
    tile_origin_y = Int(block_idx.y) * tile_size - padding

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
        Layout.row_major(in_channels, kernel_height, kernel_width),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Copy kernel and input data to shared memory. Assume kernel is small so its fine that thread 0 load it
    @parameter
    if local_idx == 0:
        for channel in range(in_channels):

            @parameter
            for i in range(kernel_height):

                @parameter
                for j in range(kernel_width):
                    kernel_shared[channel, i, j] = kernel[channel, i, j]

    # each thread loads a tile containing the input data for the whole block
    for channel in range(in_channels):
        in_x = tile_origin_x + local_x
        in_y = tile_origin_y + local_y
        if 0 <= in_x < width and 0 <= in_y < height:
            input_shared[channel, local_y, local_x] = input[
                batch_idx, channel, in_y, in_x
            ]
        else:
            input_shared[channel, local_y, local_x] = 0

    barrier()
