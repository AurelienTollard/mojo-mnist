from compiler import register
from gpu.host import DeviceBuffer
from layout import Layout, LayoutTensor
from tensor import InputTensor, OutputTensor
from mojo_mnist.activation import relu_scalar, identity_scalar
from mojo_mnist.conv import conv2d_gpu_kernel_impl, conv2d_maxpool_gpu_kernel_impl
from runtime.asyncrt import DeviceContextPtr

comptime dtype = DType.float32


fn _execute_conv_gpu[
    dtype: DType,
    in_layout: Layout,
    w_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
    tile_size: Int,
    stride: Int,
    padding: Int,
](
    out_tensor: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    in_tensor: LayoutTensor[dtype, in_layout, MutAnyOrigin],
    w_tensor: LayoutTensor[dtype, w_layout, MutAnyOrigin],
    b_tensor: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    ctx: DeviceContextPtr,
) raises:
    var gpu_ctx = ctx.get_device_context()

    comptime batch_size = Int(in_layout.shape[0])
    comptime out_channels = Int(out_layout.shape[1])
    comptime out_height = Int(out_layout.shape[2])
    comptime out_width = Int(out_layout.shape[3])

    comptime blocks_x = (out_width + tile_size - 1) // tile_size
    comptime blocks_y = (out_height + tile_size - 1) // tile_size

    gpu_ctx.enqueue_memset(
        DeviceBuffer[dtype](
            gpu_ctx,
            out_tensor.ptr,
            batch_size * out_channels * out_height * out_width,
            owning=False,
        ),
        0,
    )

    comptime kernel = conv2d_gpu_kernel_impl[
        dtype,
        in_layout,
        w_layout,
        b_layout,
        out_layout,
        activation,
        tile_size,
        stride,
        padding,
    ]
    gpu_ctx.enqueue_function[kernel, kernel](
        in_tensor,
        w_tensor,
        b_tensor,
        out_tensor,
        grid_dim=(blocks_x, blocks_y, batch_size * out_channels),
        block_dim=(tile_size, tile_size),
    )

    gpu_ctx.synchronize()


fn _execute_conv_maxpool_gpu[
    dtype: DType,
    in_layout: Layout,
    w_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    activation: fn[dtype: DType] (x: Scalar[dtype]) -> Scalar[dtype],
    tile_size: Int,
    stride: Int,
    padding: Int,
    pool_size: Int,
    pool_stride: Int,
](
    out_tensor: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    in_tensor: LayoutTensor[dtype, in_layout, MutAnyOrigin],
    w_tensor: LayoutTensor[dtype, w_layout, MutAnyOrigin],
    b_tensor: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    ctx: DeviceContextPtr,
) raises:
    var gpu_ctx = ctx.get_device_context()

    comptime batch_size = Int(in_layout.shape[0])
    comptime out_channels = Int(out_layout.shape[1])
    comptime out_height = Int(out_layout.shape[2])
    comptime out_width = Int(out_layout.shape[3])

    comptime blocks_x = (out_width + tile_size - 1) // tile_size
    comptime blocks_y = (out_height + tile_size - 1) // tile_size

    gpu_ctx.enqueue_memset(
        DeviceBuffer[dtype](
            gpu_ctx,
            out_tensor.ptr,
            batch_size * out_channels * out_height * out_width,
            owning=False,
        ),
        0,
    )

    comptime kernel = conv2d_maxpool_gpu_kernel_impl[
        dtype,
        in_layout,
        w_layout,
        b_layout,
        out_layout,
        activation,
        tile_size,
        stride,
        padding,
        pool_size,
        pool_stride,
    ]
    gpu_ctx.enqueue_function[kernel, kernel](
        in_tensor,
        w_tensor,
        b_tensor,
        out_tensor,
        grid_dim=(blocks_x, blocks_y, batch_size * out_channels),
        block_dim=(tile_size, tile_size),
    )

    gpu_ctx.synchronize()


@register("conv2d")
struct Conv2dCustomOp:
    """Conv2d without pooling (identity activation)."""

    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        in_channels: Int,
        height: Int,
        width: Int,
        out_channels: Int,
        kernel_height: Int,
        kernel_width: Int,
        tile_size: Int,
        stride: Int,
        padding: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        weight: InputTensor[dtype=dtype, rank=4],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
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
            _execute_conv_gpu[
                dtype,
                in_layout,
                w_layout,
                b_layout,
                out_layout,
                identity_scalar,
                tile_size,
                stride,
                padding,
            ](out_tensor, in_tensor, w_tensor, b_tensor, ctx)
        else:
            raise Error("Unsupported target: " + target)


@register("conv2d_maxpool_relu")
struct Conv2dMaxPoolReluCustomOp:
    """Conv2d + maxpool with ReLU activation."""

    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        in_channels: Int,
        height: Int,
        width: Int,
        out_channels: Int,
        kernel_height: Int,
        kernel_width: Int,
        tile_size: Int,
        stride: Int,
        padding: Int,
        pool_size: Int,
        pool_stride: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        weight: InputTensor[dtype=dtype, rank=4],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
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
            _execute_conv_maxpool_gpu[
                dtype,
                in_layout,
                w_layout,
                b_layout,
                out_layout,
                relu_scalar,
                tile_size,
                stride,
                padding,
                pool_size,
                pool_stride,
            ](out_tensor, in_tensor, w_tensor, b_tensor, ctx)
        else:
            raise Error("Unsupported target: " + target)
