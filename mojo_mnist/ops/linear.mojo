from compiler import register
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from tensor import InputTensor, OutputTensor
from mojo_mnist.linear import linear_relu_gpu_kernel, linear_identity_gpu_kernel, TILE_SIZE
from runtime.asyncrt import DeviceContextPtr

comptime dtype = DType.float32


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
        LayoutTensor[dtype, layout_x, MutAnyOrigin],
        LayoutTensor[dtype, layout_w, MutAnyOrigin],
        LayoutTensor[dtype, layout_b, MutAnyOrigin],
        LayoutTensor[dtype, layout_y, MutAnyOrigin],
    ) -> None,
](
    out_tensor: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    in_tensor: LayoutTensor[dtype, in_layout, MutAnyOrigin],
    w_tensor: LayoutTensor[dtype, w_layout, MutAnyOrigin],
    b_tensor: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    ctx: DeviceContextPtr,
) raises:
    """Helper to execute linear kernel on GPU with proper grid/block setup.

    Dimensions are extracted at compile-time from layout parameters.
    """
    var gpu_ctx = ctx.get_device_context()

    comptime batch_size = Int(in_layout.shape[0])
    comptime out_features = Int(out_layout.shape[1])

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
        out_tensor = output.to_layout_tensor()
        in_tensor = input.to_layout_tensor()
        w_tensor = weight.to_layout_tensor()
        b_tensor = bias.to_layout_tensor()

        comptime in_layout = in_tensor.layout
        comptime w_layout = w_tensor.layout
        comptime b_layout = b_tensor.layout
        comptime out_layout = out_tensor.layout

        @parameter
        if target == "gpu":
            _execute_linear_gpu[
                dtype,
                in_layout,
                w_layout,
                b_layout,
                out_layout,
                linear_relu_gpu_kernel,
            ](out_tensor, in_tensor, w_tensor, b_tensor, ctx)
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
        out_tensor = output.to_layout_tensor()
        in_tensor = input.to_layout_tensor()
        w_tensor = weight.to_layout_tensor()
        b_tensor = bias.to_layout_tensor()

        comptime in_layout = in_tensor.layout
        comptime w_layout = w_tensor.layout
        comptime b_layout = b_tensor.layout
        comptime out_layout = out_tensor.layout

        @parameter
        if target == "gpu":
            _execute_linear_gpu[
                dtype,
                in_layout,
                w_layout,
                b_layout,
                out_layout,
                linear_identity_gpu_kernel,
            ](out_tensor, in_tensor, w_tensor, b_tensor, ctx)
        else:
            raise Error("Unsupported target: " + target)
