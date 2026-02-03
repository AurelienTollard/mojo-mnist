from gpu import thread_idx, block_idx, block_dim, barrier, grid_dim
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from memory import AddressSpace
from mojo_mnist.activation import relu_scalar, identity_scalar
from utils.numerics import min_finite


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

    comptime out_height = Int(layout_output.shape[2])
    comptime out_width = Int(layout_output.shape[3])

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

    global_idx_x = Int(block_idx.x) * tile_size + local_x
    global_idx_y = Int(block_idx.y) * tile_size + local_y

    # Load in shared memory a tile containing the input data for the whole block
    input_shared = LayoutTensor[
        dtype,
        Layout.row_major(in_channels, effective_tile_size, effective_tile_size),
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
    if local_idx == 0:
        for channel in range(in_channels):

            @parameter
            for i in range(kernel_height):

                @parameter
                for j in range(kernel_width):
                    kernel_shared[channel, i, j] = weights[
                        out_channel_idx, channel, i, j
                    ]

    # each thread loads a portion of the input tile (including halo)
    for channel in range(in_channels):
        for tile_y in range(local_y, effective_tile_size, Int(block_dim.y)):
            in_y = tile_origin_y + tile_y
            for tile_x in range(local_x, effective_tile_size, Int(block_dim.x)):
                in_x = tile_origin_x + tile_x
                if 0 <= in_x < width and 0 <= in_y < height:
                    input_shared[channel, tile_y, tile_x] = input[
                        batch_idx, channel, in_y, in_x
                    ]
                else:
                    input_shared[channel, tile_y, tile_x] = 0

    barrier()

    sum: output.element_type = 0
    if local_x < tile_size and local_y < tile_size:
        for channel in range(in_channels):
            for i in range(kernel_height):
                for j in range(kernel_width):
                    sum += (
                        input_shared[channel, local_y + i, local_x + j]
                        * kernel_shared[channel, i, j]
                    )

    if (
        batch_idx < batch_size
        and out_channel_idx < out_channels
        and global_idx_y < out_height
        and global_idx_x < out_width
    ):
        result = sum + bias[out_channel_idx]
        output[batch_idx, out_channel_idx, global_idx_y, global_idx_x] = activation[
            dtype
        ](rebind[Scalar[dtype]](result))


fn conv2d_maxpool_gpu_kernel_impl[
    dtype: DType,
    layout_input: Layout,  # Input: (batch_size, in_channels, height, width)
    layout_weights: Layout,  # Weights: (out_channels, in_channels, kernel_height, kernel_width)
    layout_bias: Layout,  # Bias: (out_channels,)
    layout_output: Layout,  # Output: (batch_size, out_channels, out_height, out_width)
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
    tile_size: Int,
    stride: Int,
    padding: Int,
    pool_size: Int,
    pool_stride: Int,
](
    input: LayoutTensor[dtype, layout_input, MutAnyOrigin],
    weights: LayoutTensor[dtype, layout_weights, MutAnyOrigin],
    bias: LayoutTensor[dtype, layout_bias, MutAnyOrigin],
    output: LayoutTensor[dtype, layout_output, MutAnyOrigin],
):
    """Convolution + MaxPool layer kernel with configurable activation.

    Each thread computes one pooled output element for one output channel.
    block = (tile_size, tile_size)
    grid = (ceil(out_width / tile_size), ceil(out_height / tile_size), batch * out_channels)
    """
    comptime batch_size = Int(layout_input.shape[0])
    comptime in_channels = Int(layout_input.shape[1])
    comptime out_channels = Int(layout_output.shape[1])

    comptime height = Int(layout_input.shape[2])
    comptime width = Int(layout_input.shape[3])

    comptime out_height = Int(layout_output.shape[2])
    comptime out_width = Int(layout_output.shape[3])

    comptime kernel_height = Int(layout_weights.shape[2])
    comptime kernel_width = Int(layout_weights.shape[3])
    debug_assert(
        kernel_height == kernel_width,
        "Kernel height != width is not implemented",
    )
    debug_assert(stride == 1, "Stride != 1 is not implemented")

    comptime conv_out_height = (height + 2 * padding - kernel_height) // stride + 1
    comptime conv_out_width = (width + 2 * padding - kernel_width) // stride + 1

    comptime effective_tile_size = tile_size * pool_stride + kernel_width - 1

    depth_idx = Int(block_idx.z)
    out_channel_idx = depth_idx % out_channels
    batch_idx = Int(depth_idx // out_channels)

    local_x = Int(thread_idx.x)
    local_y = Int(thread_idx.y)
    local_idx = Int(thread_idx.y * block_dim.x + thread_idx.x)

    tile_origin_x = Int(block_idx.x) * tile_size * pool_stride - padding
    tile_origin_y = Int(block_idx.y) * tile_size * pool_stride - padding

    global_pool_x = Int(block_idx.x) * tile_size + local_x
    global_pool_y = Int(block_idx.y) * tile_size + local_y

    # Shared memory tiles
    input_shared = LayoutTensor[
        dtype,
        Layout.row_major(in_channels, effective_tile_size, effective_tile_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    kernel_shared = LayoutTensor[
        dtype,
        Layout.row_major(in_channels, kernel_height, kernel_width),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Load kernel for this out_channel
    if local_idx == 0:
        for channel in range(in_channels):
            @parameter
            for i in range(kernel_height):
                @parameter
                for j in range(kernel_width):
                    kernel_shared[channel, i, j] = weights[
                        out_channel_idx, channel, i, j
                    ]

    # Load input tile (with halo) cooperatively
    for channel in range(in_channels):
        for tile_y in range(local_y, effective_tile_size, Int(block_dim.y)):
            in_y = tile_origin_y + tile_y
            for tile_x in range(local_x, effective_tile_size, Int(block_dim.x)):
                in_x = tile_origin_x + tile_x
                if 0 <= in_x < width and 0 <= in_y < height:
                    input_shared[channel, tile_y, tile_x] = input[
                        batch_idx, channel, in_y, in_x
                    ]
                else:
                    input_shared[channel, tile_y, tile_x] = 0

    barrier()

    if (
        local_x < tile_size
        and local_y < tile_size
        and batch_idx < batch_size
        and out_channel_idx < out_channels
        and global_pool_y < out_height
        and global_pool_x < out_width
    ):
        var max_val: output.element_type = min_finite[dtype]()

        for py in range(pool_size):
            for px in range(pool_size):
                out_y = global_pool_y * pool_stride + py
                out_x = global_pool_x * pool_stride + px

                if out_y < conv_out_height and out_x < conv_out_width:
                    var sum: output.element_type = 0
                    for channel in range(in_channels):
                        for ky in range(kernel_height):
                            for kx in range(kernel_width):
                                sum += (
                                    input_shared[
                                        channel,
                                        local_y * pool_stride + py + ky,
                                        local_x * pool_stride + px + kx,
                                    ]
                                    * kernel_shared[channel, ky, kx]
                                )

                    sum += bias[out_channel_idx]
                    if sum > max_val:
                        max_val = sum


        # We assume the activation function is monotonic and non-decreasing so we only need to check the maximum value
        output[
            batch_idx,
            out_channel_idx,
            global_pool_y,
            global_pool_x,
        ] = activation[dtype](rebind[Scalar[dtype]](max_val))
