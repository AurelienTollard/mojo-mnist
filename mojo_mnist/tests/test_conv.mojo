from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_mnist.activation import identity_scalar
from mojo_mnist.conv import conv2d_gpu_kernel_impl

comptime dtype = DType.float32

comptime BATCH_SIZE = 2
comptime IN_CHANNELS = 1
comptime OUT_CHANNELS = 4
comptime HEIGHT = 28
comptime WIDTH = 28
comptime KERNEL_SIZE = 3
comptime TILE_SIZE = 16


fn main() raises:
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * IN_CHANNELS * HEIGHT * WIDTH
        )
        input_buf.enqueue_fill(0)

        weights_buf = ctx.enqueue_create_buffer[dtype](
            OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
        )
        weights_buf.enqueue_fill(0)

        bias_buf = ctx.enqueue_create_buffer[dtype](OUT_CHANNELS)
        bias_buf.enqueue_fill(0)

        kernel_buf = ctx.enqueue_create_buffer[dtype](
            OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
        )
        kernel_buf.enqueue_fill(0)

        output_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * OUT_CHANNELS * HEIGHT * WIDTH
        )
        output_buf.enqueue_fill(0)

        comptime input_layout = Layout.row_major(
            BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH
        )
        comptime weights_layout = Layout.row_major(
            OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE
        )
        comptime bias_layout = Layout.row_major(OUT_CHANNELS)
        comptime kernel_layout = Layout.row_major(
            OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE
        )
        comptime output_layout = Layout.row_major(
            BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH
        )

        input_tensor = LayoutTensor[dtype, input_layout, MutAnyOrigin](
            input_buf.unsafe_ptr()
        )
        weights_tensor = LayoutTensor[dtype, weights_layout, MutAnyOrigin](
            weights_buf.unsafe_ptr()
        )
        bias_tensor = LayoutTensor[dtype, bias_layout, MutAnyOrigin](
            bias_buf.unsafe_ptr()
        )
        kernel_tensor = LayoutTensor[dtype, kernel_layout, MutAnyOrigin](
            kernel_buf.unsafe_ptr()
        )
        output_tensor = LayoutTensor[dtype, output_layout, MutAnyOrigin](
            output_buf.unsafe_ptr()
        )

        comptime kernel = conv2d_gpu_kernel_impl[
            dtype,
            input_layout,
            weights_layout,
            bias_layout,
            kernel_layout,
            output_layout,
            identity_scalar,
            TILE_SIZE,
        ]

        ctx.enqueue_function[kernel, kernel](
            input_tensor,
            weights_tensor,
            bias_tensor,
            kernel_tensor,
            output_tensor,
            grid_dim=(HEIGHT, WIDTH, OUT_CHANNELS * BATCH_SIZE),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )

        ctx.synchronize()
