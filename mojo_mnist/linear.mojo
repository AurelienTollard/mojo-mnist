from compiler import register
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from max.tensor import InputTensor, OutputTensor
from memory import AddressSpace
from mojo_mnist.activation import relu_scalar, identity_scalar
from runtime.asyncrt import DeviceContextPtr

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


fn _execute_linear_gpu[
    dtype: DType,
    in_layout: Layout,
    w_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    kernel_fn: fn[
        dtype: DType,
        layout_x: Layout,
        layout_w: Layout,
        layout_b: Layout,
        layout_y: Layout,
    ] (
        LayoutTensor[dtype, layout_x, ImmutAnyOrigin],
        LayoutTensor[dtype, layout_w, ImmutAnyOrigin],
        LayoutTensor[dtype, layout_b, ImmutAnyOrigin],
        LayoutTensor[dtype, layout_y, MutAnyOrigin],
    ) -> None,
](
    output: OutputTensor[dtype=dtype, rank=2],
    input: InputTensor[dtype=dtype, rank=2],
    weight: InputTensor[dtype=dtype, rank=2],
    bias: InputTensor[dtype=dtype, rank=1],
    ctx: DeviceContextPtr,
) raises:
    """Helper to execute linear kernel on GPU with proper grid/block setup.

    Dimensions are extracted at compile-time from layout parameters.
    """
    out_tensor = output.to_layout_tensor()
    in_tensor = input.to_layout_tensor()
    w_tensor = weight.to_layout_tensor()
    b_tensor = bias.to_layout_tensor()

    var gpu_ctx = ctx.get_device_context()

    comptime batch_size = in_layout.shape[0]()
    comptime out_features = out_layout.shape[1]()

    comptime blocks_y = (batch_size + TILE_SIZE - 1) // TILE_SIZE
    comptime blocks_x = (out_features + TILE_SIZE - 1) // TILE_SIZE
    comptime threads = (TILE_SIZE, TILE_SIZE)

    # Zero output buffer first
    gpu_ctx.enqueue_memset(
        DeviceBuffer[dtype](
            gpu_ctx,
            out_tensor.ptr,
            batch_size * out_features,
            owning=False,
        ),
        0,
    )

    comptime kernel = kernel_fn[
        dtype, in_layout, w_layout, b_layout, out_layout
    ]
    gpu_ctx.enqueue_function[kernel, kernel](
        in_tensor,
        w_tensor,
        b_tensor,
        out_tensor,
        grid_dim=(blocks_x, blocks_y),
        block_dim=threads,
    )

    gpu_ctx.synchronize()


@register("linear_relu")
struct LinearReLUCustomOp:
    """Linear layer with ReLU activation as PyTorch custom operator."""

    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        in_features: Int,
        out_features: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        input: InputTensor[dtype=dtype, rank=2],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        """Execute linear layer: Y = ReLU(X @ W + b)."""
        comptime in_layout = Layout.row_major(batch_size, in_features)
        comptime w_layout = Layout.row_major(in_features, out_features)
        comptime b_layout = Layout.row_major(out_features)
        comptime out_layout = Layout.row_major(batch_size, out_features)

        @parameter
        if target == "gpu":
            _execute_linear_gpu[
                dtype,
                in_layout,
                w_layout,
                b_layout,
                out_layout,
                linear_relu_gpu_kernel,
            ](output, input, weight, bias, ctx)
        else:
            raise Error("Unsupported target: " + target)


@register("linear")
struct LinearCustomOp:
    """Linear layer (no activation) as PyTorch custom operator."""

    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        in_features: Int,
        out_features: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        input: InputTensor[dtype=dtype, rank=2],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        """Execute linear layer: Y = X @ W + b (no activation)."""
        comptime in_layout = Layout.row_major(batch_size, in_features)
        comptime w_layout = Layout.row_major(in_features, out_features)
        comptime b_layout = Layout.row_major(out_features)
        comptime out_layout = Layout.row_major(batch_size, out_features)

        @parameter
        if target == "gpu":
            _execute_linear_gpu[
                dtype,
                in_layout,
                w_layout,
                b_layout,
                out_layout,
                linear_identity_gpu_kernel,
            ](output, input, weight, bias, ctx)
        else:
            raise Error("Unsupported target: " + target)
