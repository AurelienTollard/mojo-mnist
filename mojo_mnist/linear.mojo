from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from memory import AddressSpace
from mojo_mnist.activation import relu_scalar, identity_scalar

comptime TILE_SIZE = 32
comptime BLOCK_DIM_COUNT = 2


fn linear_relu_gpu_kernel[
    dtype: DType,
    layout_x: Layout,
    layout_w: Layout,
    layout_b: Layout,
    layout_y: Layout,
](
    x: LayoutTensor[dtype, layout_x, ImmutAnyOrigin],
    w: LayoutTensor[dtype, layout_w, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout_b, ImmutAnyOrigin],
    y: LayoutTensor[dtype, layout_y, MutAnyOrigin],
):
    """Fused Linear + ReLU kernel.

    Convenience wrapper that hardcodes relu_scalar as the activation.
    Dimensions are extracted at compile-time from layout parameters.
    """
    linear_gpu_kernel_impl[
        dtype,
        layout_x,
        layout_w,
        layout_b,
        layout_y,
        relu_scalar,
    ](x, w, b, y)


fn linear_identity_gpu_kernel[
    dtype: DType,
    layout_x: Layout,
    layout_w: Layout,
    layout_b: Layout,
    layout_y: Layout,
](
    x: LayoutTensor[dtype, layout_x, ImmutAnyOrigin],
    w: LayoutTensor[dtype, layout_w, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout_b, ImmutAnyOrigin],
    y: LayoutTensor[dtype, layout_y, MutAnyOrigin],
):
    """Fused Linear kernel with identity activation.

    Convenience wrapper that hardcodes identity_scalar as the activation.
    Dimensions are extracted at compile-time from layout parameters.
    """
    linear_gpu_kernel_impl[
        dtype,
        layout_x,
        layout_w,
        layout_b,
        layout_y,
        identity_scalar,
    ](x, w, b, y)


fn linear_gpu_kernel_impl[
    dtype: DType,
    layout_x: Layout,  # Input: (batch_size, in_features)
    layout_w: Layout,  # Weights: (in_features, out_features)
    layout_b: Layout,  # Bias: (out_features,)
    layout_y: Layout,  # Output: (batch_size, out_features)
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
](
    x: LayoutTensor[dtype, layout_x, ImmutAnyOrigin],  # Input tensor
    w: LayoutTensor[dtype, layout_w, ImmutAnyOrigin],  # Weight matrix
    b: LayoutTensor[dtype, layout_b, ImmutAnyOrigin],  # Bias vector
    y: LayoutTensor[dtype, layout_y, MutAnyOrigin],  # Output tensor
):
    """Linear layer kernel with configurable activation.

    Computes Y = activation(X @ W + b) using tiled matrix multiplication
    with async memory copies for optimal performance.

    Dimensions are extracted at compile-time from layout parameters.
    Uses 2D grid: (batch_size, out_features).
    """
    # Extract dimensions at compile-time from layouts
    comptime batch_size = Int(layout_x.shape[0])
    comptime in_features = Int(layout_x.shape[1])
    comptime out_features = Int(layout_y.shape[1])

    tile_size_x = block_dim.x
    tile_size_y = block_dim.y

    local_row = thread_idx.y
    local_col = thread_idx.x

    global_row = Int(block_idx.y * tile_size_y + local_row)
    global_col = Int(block_idx.x * tile_size_x + local_col)

    out_tile = y.tile[TILE_SIZE, TILE_SIZE](Int(block_idx.y), Int(block_idx.x))

    x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    w_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE_SIZE, TILE_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    acc: x.element_type = 0

    comptime load_x_layout = Layout.row_major(1, TILE_SIZE)
    comptime load_w_layout = Layout.row_major(1, TILE_SIZE)

    num_tiles = (in_features + TILE_SIZE - 1) // TILE_SIZE

    for tile_idx in range(num_tiles):
        x_tile = x.tile[TILE_SIZE, TILE_SIZE](Int(block_idx.y), tile_idx)
        w_tile = w.tile[TILE_SIZE, TILE_SIZE](tile_idx, Int(block_idx.x))

        # Asynchronously copy tiles to shared memory
        copy_dram_to_sram_async[
            thread_layout=load_x_layout,
            num_threads = TILE_SIZE * TILE_SIZE,
            block_dim_count=BLOCK_DIM_COUNT,
        ](x_shared, x_tile)

        copy_dram_to_sram_async[
            thread_layout=load_w_layout,
            num_threads = TILE_SIZE * TILE_SIZE,
            block_dim_count=BLOCK_DIM_COUNT,
        ](w_shared, w_tile)

        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        for k in range(TILE_SIZE):
            k_global = tile_idx * TILE_SIZE + k
            if (
                local_row < TILE_SIZE
                and local_col < TILE_SIZE
                and k < TILE_SIZE
                and k_global < in_features
                and global_row < batch_size
                and global_col < out_features
            ):
                acc += x_shared[local_row, k] * w_shared[k, local_col]

        barrier()

    if global_row < batch_size and global_col < out_features:
        result = acc + b[global_col]
        out_tile[local_row, local_col] = activation[dtype](
            rebind[Scalar[dtype]](result)
        )
