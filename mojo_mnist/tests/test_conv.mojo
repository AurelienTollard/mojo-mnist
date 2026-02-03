from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_almost_equal
from mojo_mnist.activation import identity_scalar
from mojo_mnist.conv import conv2d_gpu_kernel_impl, conv2d_maxpool_gpu_kernel_impl

comptime dtype = DType.float32

comptime BATCH_SIZE = 2
comptime IN_CHANNELS = 3
comptime OUT_CHANNELS = 4
comptime HEIGHT = 28
comptime WIDTH = 28
comptime KERNEL_SIZE = 3
comptime TILE_SIZE = 16
comptime STRIDE = 1
comptime PADDING = 1
comptime OUT_HEIGHT = (HEIGHT + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
comptime OUT_WIDTH = (WIDTH + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
comptime POOL_SIZE = 2
comptime POOL_STRIDE = 2
comptime OUT_HEIGHT_POOL = (OUT_HEIGHT - POOL_SIZE) // POOL_STRIDE + 1
comptime OUT_WIDTH_POOL = (OUT_WIDTH - POOL_SIZE) // POOL_STRIDE + 1


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

        output_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH
        )
        output_buf.enqueue_fill(0)

        output_pool_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * OUT_CHANNELS * OUT_HEIGHT_POOL * OUT_WIDTH_POOL
        )
        output_pool_buf.enqueue_fill(0)

        expected_buf = ctx.enqueue_create_host_buffer[dtype](
            BATCH_SIZE * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH
        )
        expected_buf.enqueue_fill(0)

        expected_pool_buf = ctx.enqueue_create_host_buffer[dtype](
            BATCH_SIZE * OUT_CHANNELS * OUT_HEIGHT_POOL * OUT_WIDTH_POOL
        )
        expected_pool_buf.enqueue_fill(0)

        comptime input_layout = Layout.row_major(
            BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH
        )
        comptime weights_layout = Layout.row_major(
            OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE
        )
        comptime bias_layout = Layout.row_major(OUT_CHANNELS)
        comptime output_layout = Layout.row_major(
            BATCH_SIZE, OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH
        )
        comptime output_pool_layout = Layout.row_major(
            BATCH_SIZE, OUT_CHANNELS, OUT_HEIGHT_POOL, OUT_WIDTH_POOL
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
        output_tensor = LayoutTensor[dtype, output_layout, MutAnyOrigin](
            output_buf.unsafe_ptr()
        )
        output_pool_tensor = LayoutTensor[dtype, output_pool_layout, MutAnyOrigin](
            output_pool_buf.unsafe_ptr()
        )

        with input_buf.map_to_host() as input_host, weights_buf.map_to_host() as weights_host, bias_buf.map_to_host() as bias_host:
            for b in range(BATCH_SIZE):
                for c in range(IN_CHANNELS):
                    for y in range(HEIGHT):
                        for x in range(WIDTH):
                            idx = (
                                (b * IN_CHANNELS + c) * HEIGHT + y
                            ) * WIDTH + x
                            input_host[idx] = Float32(
                                (x + y + c + b) % 7
                            ) / Float32(7.0)

            for oc in range(OUT_CHANNELS):
                for ic in range(IN_CHANNELS):
                    for ky in range(KERNEL_SIZE):
                        for kx in range(KERNEL_SIZE):
                            idx = (
                                (oc * IN_CHANNELS + ic) * KERNEL_SIZE + ky
                            ) * KERNEL_SIZE + kx
                            weights_host[idx] = Float32(
                                (oc + ic + ky + kx) % 5
                            ) / Float32(5.0)

            for oc in range(OUT_CHANNELS):
                bias_host[oc] = Float32(oc) * Float32(0.01)

            for b in range(BATCH_SIZE):
                for oc in range(OUT_CHANNELS):
                    for oy in range(OUT_HEIGHT):
                        for ox in range(OUT_WIDTH):
                            var acc: Float32 = bias_host[oc]
                            for ic in range(IN_CHANNELS):
                                for ky in range(KERNEL_SIZE):
                                    for kx in range(KERNEL_SIZE):
                                        in_y = oy * STRIDE + ky - PADDING
                                        in_x = ox * STRIDE + kx - PADDING
                                        if (
                                            in_y >= 0
                                            and in_y < HEIGHT
                                            and in_x >= 0
                                            and in_x < WIDTH
                                        ):
                                            input_idx = (
                                                (b * IN_CHANNELS + ic) * HEIGHT
                                                + in_y
                                            ) * WIDTH + in_x
                                            weight_idx = (
                                                (oc * IN_CHANNELS + ic)
                                                * KERNEL_SIZE
                                                + ky
                                            ) * KERNEL_SIZE + kx
                                            acc += (
                                                input_host[input_idx]
                                                * weights_host[weight_idx]
                                            )

                            out_idx = (
                                (b * OUT_CHANNELS + oc) * OUT_HEIGHT + oy
                            ) * OUT_WIDTH + ox
                            expected_buf[out_idx] = acc

            for b in range(BATCH_SIZE):
                for oc in range(OUT_CHANNELS):
                    for py in range(OUT_HEIGHT_POOL):
                        for px in range(OUT_WIDTH_POOL):
                            var max_val: Float32 = -1.0e20
                            for ky in range(POOL_SIZE):
                                for kx in range(POOL_SIZE):
                                    oy = py * POOL_STRIDE + ky
                                    ox = px * POOL_STRIDE + kx
                                    if oy < OUT_HEIGHT and ox < OUT_WIDTH:
                                        idx = (
                                            (b * OUT_CHANNELS + oc) * OUT_HEIGHT
                                            + oy
                                        ) * OUT_WIDTH + ox
                                        val = expected_buf[idx]
                                        if val > max_val:
                                            max_val = val

                            pool_idx = (
                                (b * OUT_CHANNELS + oc) * OUT_HEIGHT_POOL + py
                            ) * OUT_WIDTH_POOL + px
                            expected_pool_buf[pool_idx] = max_val

        print("conv2d CPU reference computed")

        comptime kernel = conv2d_gpu_kernel_impl[
            dtype,
            input_layout,
            weights_layout,
            bias_layout,
            output_layout,
            identity_scalar,
            TILE_SIZE,
            STRIDE,
            PADDING,
        ]

        ctx.enqueue_function[kernel, kernel](
            input_tensor,
            weights_tensor,
            bias_tensor,
            output_tensor,
            grid_dim=(
                (OUT_WIDTH + TILE_SIZE - 1) // TILE_SIZE,
                (OUT_HEIGHT + TILE_SIZE - 1) // TILE_SIZE,
                OUT_CHANNELS * BATCH_SIZE,
            ),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )

        ctx.synchronize()

        with output_buf.map_to_host() as output_host:
            for b in range(BATCH_SIZE):
                for oc in range(OUT_CHANNELS):
                    for oy in range(OUT_HEIGHT):
                        for ox in range(OUT_WIDTH):
                            idx = (
                                (b * OUT_CHANNELS + oc) * OUT_HEIGHT + oy
                            ) * OUT_WIDTH + ox
                            assert_almost_equal(
                                output_host[idx],
                                expected_buf[idx],
                                atol=1e-3,
                                rtol=2e-2,
                            )

        print("✓ conv2d GPU matches CPU reference")

        comptime kernel_pool = conv2d_maxpool_gpu_kernel_impl[
            dtype,
            input_layout,
            weights_layout,
            bias_layout,
            output_pool_layout,
            identity_scalar,
            TILE_SIZE,
            STRIDE,
            PADDING,
            POOL_SIZE,
            POOL_STRIDE,
        ]

        ctx.enqueue_function[kernel_pool, kernel_pool](
            input_tensor,
            weights_tensor,
            bias_tensor,
            output_pool_tensor,
            grid_dim=(
                (OUT_WIDTH_POOL + TILE_SIZE - 1) // TILE_SIZE,
                (OUT_HEIGHT_POOL + TILE_SIZE - 1) // TILE_SIZE,
                OUT_CHANNELS * BATCH_SIZE,
            ),
            block_dim=(TILE_SIZE, TILE_SIZE),
        )

        ctx.synchronize()

        with output_pool_buf.map_to_host() as output_pool_host:
            for b in range(BATCH_SIZE):
                for oc in range(OUT_CHANNELS):
                    for py in range(OUT_HEIGHT_POOL):
                        for px in range(OUT_WIDTH_POOL):
                            idx = (
                                (b * OUT_CHANNELS + oc) * OUT_HEIGHT_POOL + py
                            ) * OUT_WIDTH_POOL + px
                            assert_almost_equal(
                                output_pool_host[idx],
                                expected_pool_buf[idx],
                                atol=1e-3,
                                rtol=2e-2,
                            )

        print("✓ conv2d + maxpool GPU matches CPU reference")
