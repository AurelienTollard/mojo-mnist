# Fused Linear + ReLU GPU kernel for MLP
# Computes: Y = ReLU(X @ W + b) where X is batched input
#
# This kernel fuses the linear transformation and ReLU activation
# to avoid extra memory round-trips between GPU global memory.

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from memory import AddressSpace
from mojo_mnist.activation import relu_scalar


fn linear_relu_gpu_kernel[
    dtype: DType,
    layout_x: Layout,
    layout_w: Layout,
    layout_b: Layout,
    layout_y: Layout,
    tile_size: Int,
](
    x: LayoutTensor[dtype, layout_x, ImmutAnyOrigin],
    w: LayoutTensor[dtype, layout_w, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout_b, ImmutAnyOrigin],
    y: LayoutTensor[dtype, layout_y, MutAnyOrigin],
    batch_size: Int,
    in_features: Int,
    out_features: Int,
):
    """Fused Linear + ReLU kernel.

    Convenience wrapper around linear_gpu_kernel using relu_scalar.
    """
    linear_gpu_kernel[
        dtype,
        layout_x,
        layout_w,
        layout_b,
        layout_y,
        tile_size,
    ](
        x,
        w,
        b,
        y,
        batch_size,
        in_features,
        out_features,
        relu_scalar,
    )


fn linear_gpu_kernel[
    dtype: DType,
    layout_x: Layout,      # Input: (batch_size, in_features)
    layout_w: Layout,      # Weights: (in_features, out_features)
    layout_b: Layout,      # Bias: (out_features,)
    layout_y: Layout,      # Output: (batch_size, out_features)
    tile_size: Int,        # Tile size for shared memory blocking
](
    x: LayoutTensor[dtype, layout_x, ImmutAnyOrigin],        # Input tensor
    w: LayoutTensor[dtype, layout_w, ImmutAnyOrigin],        # Weight matrix
    b: LayoutTensor[dtype, layout_b, ImmutAnyOrigin],        # Bias vector
    y: LayoutTensor[dtype, layout_y, MutAnyOrigin],          # Output tensor
    batch_size: Int,
    in_features: Int,
    out_features: Int,
    activation: fn[dtype: DType](x: Scalar[dtype]) unified -> Scalar[dtype],
):
    """Linear layer kernel with configurable activation.

    Computes Y = activation(X @ W + b) using tiled matrix multiplication.
    """
    local_row = thread_idx.y
    local_col = thread_idx.x

    global_row = Int(block_idx.y * tile_size + local_row)
    global_col = Int(block_idx.x * tile_size + local_col)

    comptime load_layout = Layout.row_major(1, tile_size)

    x_shared = LayoutTensor[
        dtype,
        Layout.row_major(tile_size, tile_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    w_shared = LayoutTensor[
        dtype,
        Layout.row_major(tile_size, tile_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    acc: x.element_type = 0

    num_tiles = (in_features + tile_size - 1) // tile_size

    for tile_idx in range(num_tiles):
        x_row = global_row
        x_col = tile_idx * tile_size + local_col

        w_row = tile_idx * tile_size + local_row
        w_col = global_col

        if x_row < batch_size and x_col < in_features:
            x_shared[local_row, local_col] = x[x_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        if w_row < in_features and w_col < out_features:
            w_shared[local_row, local_col] = w[w_row, w_col]
        else:
            w_shared[local_row, local_col] = 0

        barrier()

        for k in range(tile_size):
            k_global = tile_idx * tile_size + k
            if k_global < in_features:
                acc += x_shared[local_row, k] * w_shared[k, local_col]

        barrier()

    if global_row < batch_size and global_col < out_features:
        result = acc + b[global_col]
        y[global_row, global_col] = activation[dtype](rebind[Scalar[dtype]](result))
