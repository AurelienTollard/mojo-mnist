from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_equal, assert_almost_equal
from mojo_mnist.linear import linear_identity_gpu_kernel, TILE_SIZE

comptime dtype = DType.float32

comptime BATCH_SIZE = 4
comptime IN_FEATURES = 1024
comptime OUT_FEATURES = 512


fn main() raises:
    with DeviceContext() as ctx:
        x_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * IN_FEATURES)
        x_buf.enqueue_fill(0)

        w_buf = ctx.enqueue_create_buffer[dtype](IN_FEATURES * OUT_FEATURES)
        w_buf.enqueue_fill(0)

        b_buf = ctx.enqueue_create_buffer[dtype](OUT_FEATURES)
        b_buf.enqueue_fill(0)

        y_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUT_FEATURES)
        y_buf.enqueue_fill(0)

        expected = ctx.enqueue_create_host_buffer[dtype](
            BATCH_SIZE * OUT_FEATURES
        )
        expected.enqueue_fill(0)

        with x_buf.map_to_host() as x_host, w_buf.map_to_host() as w_host, b_buf.map_to_host() as b_host:
            for batch in range(BATCH_SIZE):
                for i in range(IN_FEATURES):
                    x_host[batch * IN_FEATURES + i] = Float32(i)

            for i in range(IN_FEATURES):
                for j in range(OUT_FEATURES):
                    w_host[i * OUT_FEATURES + j] = Float32(i * OUT_FEATURES + j)

            for j in range(OUT_FEATURES):
                b_host[j] = Float32(1.0)

            for batch in range(BATCH_SIZE):
                for j in range(OUT_FEATURES):
                    var sum: Float32 = 0.0
                    for i in range(IN_FEATURES):
                        sum += (
                            x_host[batch * IN_FEATURES + i]
                            * w_host[i * OUT_FEATURES + j]
                        )
                    expected[batch * OUT_FEATURES + j] = sum + Float32(
                        1.0
                    )  # add bias

        # x layout: 2D row-major (BATCH_SIZE, IN_FEATURES)
        comptime x_layout = Layout.row_major(BATCH_SIZE, IN_FEATURES)
        # w layout: 2D row-major (IN_FEATURES, OUT_FEATURES)
        comptime w_layout = Layout.row_major(IN_FEATURES, OUT_FEATURES)
        # b layout: 1D row-major
        comptime b_layout = Layout.row_major(OUT_FEATURES)
        # y layout: 2D row-major (BATCH_SIZE, OUT_FEATURES)
        comptime y_layout = Layout.row_major(BATCH_SIZE, OUT_FEATURES)

        x_tensor = LayoutTensor[dtype, x_layout, ImmutAnyOrigin](
            x_buf.unsafe_ptr()
        )
        w_tensor = LayoutTensor[dtype, w_layout, ImmutAnyOrigin](
            w_buf.unsafe_ptr()
        )
        b_tensor = LayoutTensor[dtype, b_layout, ImmutAnyOrigin](
            b_buf.unsafe_ptr()
        )
        y_tensor = LayoutTensor[dtype, y_layout, MutAnyOrigin](
            y_buf.unsafe_ptr()
        )

        comptime blocks_per_grid_y = (BATCH_SIZE + TILE_SIZE - 1) // TILE_SIZE
        comptime blocks_per_grid_x = (OUT_FEATURES + TILE_SIZE - 1) // TILE_SIZE
        comptime threads_per_block = (TILE_SIZE, TILE_SIZE)

        comptime kernel = linear_identity_gpu_kernel[
            dtype,
            x_layout,
            w_layout,
            b_layout,
            y_layout,
        ]
        ctx.enqueue_function[kernel, kernel](
            x_tensor,
            w_tensor,
            b_tensor,
            y_tensor,
            BATCH_SIZE,
            IN_FEATURES,
            OUT_FEATURES,
            grid_dim=(blocks_per_grid_x, blocks_per_grid_y),
            block_dim=threads_per_block,
        )

        ctx.synchronize()

        with y_buf.map_to_host() as y_host:
            print("Testing linear layer with identity activation")
            print("Batch size:", BATCH_SIZE)
            print("Input features:", IN_FEATURES)
            print("Output features:", OUT_FEATURES)
            print("")

            for batch in range(BATCH_SIZE):
                for j in range(OUT_FEATURES):
                    var idx = batch * OUT_FEATURES + j
                    assert_almost_equal(
                        abs(y_host[idx]),
                        abs(expected[idx]),
                        atol=1e-3,
                        rtol=2e-2,
                    )

            print("✓ All tests passed!")
